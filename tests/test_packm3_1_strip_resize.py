"""Pack M3.1, item A -- THE LABELS STRIP GROWS AND SHRINKS LIVE.

The strip's height ratio, `max(1.0, 0.75 * K)`, was read ONCE at figure
build (core/lanes.py, view_build/canvas.py), so a lane that appeared
after the window was up got no room and every lane on the strip simply
got thinner. Measured at base on this three-row pane: the Labels row
stayed 0.2817 of the figure from K=1 to K=4, against 0.5019 on a fresh
four-lane build -- a whole data panel's worth of height that never
arrived.

THE MECHANISM. The SAME rule the builder used, applied in place to the
SAME GridSpec the axes' SubplotSpecs already point at, and then Pack 5
R4a's constrained-layout freeze is asked for ONE more solve. No axes is
created, destroyed, re-parented or moved: the selectors, the pick
connections, the blit caches and the painter are wired to those objects.
The freeze IS the mechanism -- measured, changing the ratios on a frozen
figure and drawing moves every axes by 0.000e+00.

THE TWO RATIFIED LIMITS, pinned here:

  * J.E.'s R1. Every transition among two lanes or more lands where a
    FRESH build at that K lands, to 1e-5 of the figure. Coming back DOWN
    to ONE lane does NOT match a freshly launched one-lane figure -- the
    as-built one-lane figure is solved once and is the odd one out -- it
    matches a one-lane figure that has been solved once more, which is
    what a window resize produces anyway. A one-lane session whose lane
    count never changes stays BYTE-IDENTICAL: the retune returns False
    and asks for nothing.
  * J.E.'s R2. A pane whose `layout_spec` carries its own
    `height_ratios` does NOT resize. "An explicit height_ratios always
    wins" is a shipped rule with a pin on it and two of the user's own
    files carry one.

THE TRAP, pinned: the retune must NOT sit behind Pack M2.1's `_K > 1`
guard. A shrink back to one lane changes the ratio, and without a
re-solve the strip would keep its old height with the new ratio -- a
silently wrong picture rather than a crash.
"""

import hashlib
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from chronotagger.core.lanes import labels_row_height
from chronotagger.core.tracks import Track

DAY = "2015-01-03 "
NAMES = ["Region (human)", "Wake (umbra)", "Agent (C-MMAE)", "Fourth lane"]

LAYOUT = {
    "nrows": 3, "ncols": 1,
    "areas": [
        {"key": "panel1", "row": 0, "col": 0, "role": "time"},
        {"key": "panel2", "row": 1, "col": 0, "role": "time"},
        {"key": "labels", "row": 2, "col": 0, "role": "labels"},
    ],
}

TOL = 1.0e-5


@pytest.fixture(autouse=True)
def boxes(monkeypatch):
    """No dialog in this module, and none may appear if one is reached.

    Pack M3.0.1's lesson: GitHub CI runs plain `pytest tests/` with no
    global stub, so a pin that can reach a box answers for it itself.
    """
    import tkinter.messagebox as mb
    import tkinter.filedialog as fdlg
    import tkinter.simpledialog as sdlg
    seen = []
    for name in ("showerror", "showinfo", "showwarning"):
        monkeypatch.setattr(
            mb, name,
            lambda *a, _n=name, **k: seen.append((_n,) + tuple(a[:1])))
    for name in ("askyesno", "askyesnocancel", "askokcancel",
                 "askretrycancel"):
        monkeypatch.setattr(
            mb, name,
            lambda *a, _n=name, **k: (seen.append((_n,) + tuple(a[:1]))
                                      or False))
    for name in ("askopenfilename", "asksaveasfilename", "askdirectory"):
        monkeypatch.setattr(fdlg, name, lambda *a, **k: "")
    for name in ("askstring", "askinteger", "askfloat"):
        monkeypatch.setattr(sdlg, name, lambda *a, **k: None)
    return seen


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def frame():
    idx = pd.date_range(DAY + "00:00:00", periods=120, freq="30s")
    return pd.DataFrame({"a": np.linspace(0.0, 1.0, len(idx))}, index=idx)


def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["a"])
    axs["panel2"].plot(df.index, df["a"])


def track_dicts(k):
    return [{"id": "lane%d" % i, "name": NAMES[i],
             "classes": ["sw", "msh", "UNKNOWN"], "order": i}
            for i in range(k)]


def build(k, tmp_path, spec=None):
    """A labeler with exactly k lanes, drawn once, root withdrawn."""
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(
        df=frame(), plot_fn=plot_fn,
        layout_spec=dict(spec or LAYOUT),
        window=pd.Timedelta("60min"), autosave_folder=str(tmp_path),
        tracks=track_dicts(k))
    lbl._build_gui()
    lbl.root.withdraw()
    lbl._update_plot()
    for _ in range(3):
        lbl.root.update()
    return lbl


def set_k(lbl, k):
    """The LIVE lane count becomes k, the way a caller changes it."""
    while len(lbl.tracks) < k:
        i = len(lbl.tracks)
        lbl.tracks.append(Track(id="lane%d" % i, name=NAMES[i],
                                classes=["sw", "msh", "UNKNOWN"], order=i))
    while len(lbl.tracks) > k:
        del lbl.tracks[-1]
    lbl._reconcile_active_track()
    lbl._update_plot()
    for _ in range(2):
        lbl.root.update()


def settle(lbl):
    """ONE more constrained-layout solve, as a window resize asks."""
    for p in (lbl.panes or []):
        lbl._invalidate_layout_freeze(p)
    lbl._update_plot()
    for _ in range(2):
        lbl.root.update()


def positions(pane):
    out = {}
    known = set()
    for key, ax in pane.user_axes.items():
        out[key] = tuple(ax.get_position().bounds)
        known.add(id(ax))
    out["labels"] = tuple(pane.strip_ax.get_position().bounds)
    known.add(id(pane.strip_ax))
    extra = [ax for ax in pane.fig.axes if id(ax) not in known]
    extra.sort(key=lambda a: (round(a.get_position().x0, 5),
                              round(a.get_position().y0, 5)))
    for i, ax in enumerate(extra):
        out["extra%d" % i] = tuple(ax.get_position().bounds)
    return out


def ids_of(pane):
    out = {k: id(ax) for k, ax in pane.user_axes.items()}
    out["labels"] = id(pane.strip_ax)
    return out


def maxdiff(a, b):
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    worst, where = 0.0, ""
    for key in sorted(a):
        for i in range(4):
            d = abs(a[key][i] - b[key][i])
            if d > worst:
                worst, where = d, "%s[%s]" % (key, "xywh"[i])
    return worst, where


def hrs_of(pane):
    return list(pane.strip_ax.get_subplotspec()
                .get_gridspec().get_height_ratios())


# ======================================================= 1. the ratio


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_the_labels_row_takes_the_rule_the_builder_uses(k, tmp_path):
    """Live, the row ends on exactly `labels_row_height(K)`."""
    lbl = build(1, tmp_path)
    row = lbl.panes[0].strip_ax.get_subplotspec().rowspan.start
    set_k(lbl, k)
    assert hrs_of(lbl.panes[0])[row] == pytest.approx(
        labels_row_height(k)), \
        "the Labels row must carry the SAME rule the builder used"


def test_the_retune_says_whether_it_changed_anything(tmp_path):
    lbl = build(2, tmp_path)
    assert lbl._retune_strip_row(lbl.panes[0]) is False, \
        "nothing changed, so nothing was asked for"
    lbl.tracks.append(Track(id="lane2", name=NAMES[2], classes=["sw"],
                            order=2))
    assert lbl._retune_strip_row(lbl.panes[0]) is True
    assert lbl._retune_strip_row(lbl.panes[0]) is False


def test_every_pane_is_retuned_not_only_the_active_one(tmp_path):
    """`_retune_strip_rows` is the session-wide form."""
    lbl = build(1, tmp_path)
    assert lbl._retune_strip_rows() == 0
    lbl.tracks.append(Track(id="lane1", name=NAMES[1], classes=["sw"],
                            order=1))
    assert lbl._retune_strip_rows() == len(lbl.panes)


# ============================ 2. it lands where a fresh build lands


@pytest.mark.parametrize("a,b", [(1, 2), (2, 3), (3, 2), (1, 4), (3, 4),
                                 (4, 2)])
def test_a_live_change_among_two_lanes_or_more_matches_a_fresh_build(
        a, b, tmp_path):
    """R1: equal to an AS-BUILT fresh build, to 1e-5 of the figure."""
    lbl = build(a, tmp_path / ("live%d%d" % (a, b)))
    before = ids_of(lbl.panes[0])
    set_k(lbl, b)
    live = positions(lbl.panes[0])
    assert ids_of(lbl.panes[0]) == before, \
        "the axes OBJECTS must survive: the selectors are wired to them"
    fresh = build(b, tmp_path / ("fresh%d%d" % (a, b)))
    worst, where = maxdiff(live, positions(fresh.panes[0]))
    assert worst < TOL, "%.3e at %s" % (worst, where)


@pytest.mark.parametrize("a", [2, 3, 4])
def test_coming_back_down_to_one_lane_matches_an_EQUALLY_SOLVED_build(
        a, tmp_path):
    """R1, the accepted difference, measured both ways.

    Against a freshly launched one-lane figure this misses by about
    5e-02 of the figure on this layout. That is the as-built one-lane
    figure being solved once and sitting somewhere its own second solve
    would move it anyway -- not the live path drifting. The pin measures
    the comparison J.E. ratified AND records the other number, so a
    later pack cannot quietly change which one is true.
    """
    lbl = build(a, tmp_path / ("live%d1" % a))
    before = ids_of(lbl.panes[0])
    set_k(lbl, 1)
    live = positions(lbl.panes[0])
    assert ids_of(lbl.panes[0]) == before
    fresh = build(1, tmp_path / ("fresh%d1" % a))
    as_built, _ = maxdiff(live, positions(fresh.panes[0]))
    settle(fresh)
    settled, where = maxdiff(live, positions(fresh.panes[0]))
    assert settled < TOL, "%.3e at %s" % (settled, where)
    assert as_built > 1.0e-3, \
        ("the as-built one-lane figure is the odd one out and this pin "
         "records it; if this ever passes, the difference J.E. accepted "
         "has gone away and WHAT GETS WORSE should say so")


def test_hiding_and_un_hiding_a_lane_resizes_too(tmp_path):
    """Ctrl+H is a lane-count change: K is over VISIBLE lanes."""
    lbl = build(3, tmp_path / "hide")
    lbl.tracks[1].visible = False
    lbl._update_plot()
    for _ in range(2):
        lbl.root.update()
    two = build(2, tmp_path / "two")
    worst, where = maxdiff(positions(lbl.panes[0]),
                           positions(two.panes[0]))
    assert worst < TOL, "%.3e at %s" % (worst, where)
    lbl.tracks[1].visible = True
    lbl._update_plot()
    for _ in range(2):
        lbl.root.update()
    three = build(3, tmp_path / "three")
    worst, where = maxdiff(positions(lbl.panes[0]),
                           positions(three.panes[0]))
    assert worst < TOL, "%.3e at %s" % (worst, where)


def test_a_round_trip_comes_back(tmp_path):
    lbl = build(3, tmp_path / "rt")
    set_k(lbl, 1)
    set_k(lbl, 3)
    fresh = build(3, tmp_path / "rt3")
    worst, where = maxdiff(positions(lbl.panes[0]),
                           positions(fresh.panes[0]))
    assert worst < TOL, "%.3e at %s" % (worst, where)


# ============================ 3. the floors that must not move


def test_one_lane_with_nothing_changed_is_byte_identical(tmp_path):
    """The Pack M2 floor: the retune asks for NOTHING at K == 1."""
    lbl = build(1, tmp_path)
    buf = io.BytesIO()
    lbl.panes[0].fig.savefig(buf, format="png")
    before = hashlib.sha256(buf.getvalue()).hexdigest()
    assert lbl._retune_strip_rows() == 0
    lbl._update_plot()
    for _ in range(2):
        lbl.root.update()
    buf2 = io.BytesIO()
    lbl.panes[0].fig.savefig(buf2, format="png")
    assert hashlib.sha256(buf2.getvalue()).hexdigest() == before


def test_an_explicit_height_ratios_is_left_alone(tmp_path):
    """R2, KEPT: a driver's own ratios always win."""
    spec = dict(LAYOUT)
    spec["height_ratios"] = [3.2, 1.0, 0.9]
    lbl = build(1, tmp_path, spec=spec)
    assert hrs_of(lbl.panes[0]) == [3.2, 1.0, 0.9]
    set_k(lbl, 3)
    assert lbl._retune_strip_row(lbl.panes[0]) is False
    assert hrs_of(lbl.panes[0]) == [3.2, 1.0, 0.9], \
        "a layout_spec that writes its own height_ratios does not resize"


# ============================ 4. what the retune asks for


def test_the_retune_is_NOT_behind_the_K_greater_than_one_guard(tmp_path):
    """THE TRAP the spike found: a shrink to ONE lane must re-solve.

    THROUGH `_update_plot`, not by calling the retune by hand: the trap
    is in the CALL SITE. Pack M2.1's own invalidation is gated on
    `_K > 1`, and putting the retune behind that guard changes the ratio
    at K == 1 and then never re-solves -- the strip keeps its old height
    with the new ratio, a silently wrong picture rather than a crash. A
    pin that calls `_retune_strip_row` itself cannot see that.
    """
    lbl = build(3, tmp_path)
    pane = lbl.panes[0]
    row = pane.strip_ax.get_subplotspec().rowspan.start
    assert pane._layout_frozen is True
    assert hrs_of(pane)[row] == pytest.approx(labels_row_height(3))

    seen = []
    real = lbl._invalidate_layout_freeze

    def spy(p=None):
        seen.append(p)
        return real(p)
    lbl._invalidate_layout_freeze = spy

    del lbl.tracks[-1]
    del lbl.tracks[-1]
    lbl._reconcile_active_track()
    lbl._update_plot()
    for _ in range(2):
        lbl.root.update()

    assert hrs_of(pane)[row] == pytest.approx(labels_row_height(1)), \
        "the ratio must come back down"
    assert pane in seen, (
        "the shrink to one lane must ask for a solve, or the strip "
        "keeps its old height with the new ratio")


def test_the_blit_backgrounds_are_invalidated(tmp_path):
    """A blit would restore a background copied at the OLD geometry."""
    lbl = build(2, tmp_path)
    pane = lbl.panes[0]
    pane._blit.recache()
    assert pane._blit._bg, "the fixture must start with a cache"
    lbl.tracks.append(Track(id="lane2", name=NAMES[2], classes=["sw"],
                            order=2))
    assert lbl._retune_strip_row(pane) is True
    assert pane._blit._bg == {}, \
        "the cached backgrounds belong to the geometry that just changed"


def test_a_pane_with_no_strip_is_not_touched(tmp_path):
    """A pane built without a Labels area answers False and no more."""
    lbl = build(2, tmp_path)

    class _Bare(object):
        strip_ax = None
        layout_spec = None
    assert lbl._retune_strip_row(_Bare()) is False


def test_the_update_plot_path_is_what_calls_it(tmp_path):
    """A caller changes the table and calls _update_plot. Nothing else."""
    lbl = build(2, tmp_path)
    seen = []
    real = lbl._retune_strip_row

    def spy(pane=None):
        seen.append(pane)
        return real(pane)
    lbl._retune_strip_row = spy
    lbl.tracks.append(Track(id="lane2", name=NAMES[2], classes=["sw"],
                            order=2))
    lbl._update_plot()
    assert seen, "_update_strip's own signature check calls the retune"
