"""Pack M2 -- THE LANES ON SCREEN: the painter, the two hit tests, the
preview, the lane switch, hiding, the sidebar, and the strip's height.

Every test here builds a REAL labeler with a REAL Tk root and a REAL
figure (Agg), because every defect this pack fixes was a defect in what
an artist or a widget actually holds, not in what a function returns.

THE FLOOR THIS FILE DEFENDS IS BACKWARDS-LOOKING: with ONE lane the strip
is what it was before Pack M2 -- no lane name, no focus ring, no legend,
the band at 0.1..0.9 and the preview at 0.05..0.95. The pack's own
validation log carries the whole-figure PNG sha256 for six conditions;
a PNG hash is platform-fragile, so what is pinned HERE is the geometry and
the artist census that produce it.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from chronotagger.core.lanes import (lane_band, pad_for, preview_band,
                                     strip_band_gid, strip_focus_gid,
                                     strip_preview_gid)
from chronotagger.core.models import Interval

DAY = "2015-01-03 "
BANDS_GID = strip_band_gid()

ONE_LANE = [{"id": "human", "name": "Human", "classes": ["sw", "msh"],
             "class_colors": {"sw": "#4e79a7", "msh": "#f28e2b"}}]
TWO_LANES = ONE_LANE + [
    {"id": "rules", "name": "Rules", "classes": ["sw", "quiet"],
     "class_colors": {"sw": "#111111", "quiet": "#cccccc"}, "order": 1},
]
FOUR_LANES = [
    {"id": "l0", "name": "L0", "classes": ["a", "b", "c", "d"], "order": 0},
    {"id": "l1", "name": "L1", "classes": ["a", "b", "c", "d"], "order": 1},
    {"id": "l2", "name": "L2", "classes": ["a", "b", "c", "d"], "order": 2},
    {"id": "l3", "name": "L3", "classes": ["a", "b", "c", "d"], "order": 3},
]

LAYOUT = {
    "nrows": 3, "ncols": 1,
    "areas": [
        {"key": "panel1", "row": 0, "col": 0, "role": "time"},
        {"key": "panel2", "row": 1, "col": 0, "role": "time"},
        {"key": "labels", "row": 2, "col": 0, "role": "labels"},
    ],
}


@pytest.fixture(autouse=True)
def _stub_dialogs(monkeypatch):
    """STANDING RULE (Pack 3): stub every dialog on any reachable path."""
    import tkinter.messagebox as mb
    import tkinter.filedialog as fd
    calls = []
    for kind in ("showinfo", "showwarning", "showerror", "askyesno",
                 "askyesnocancel", "askokcancel"):
        monkeypatch.setattr(
            mb, kind, lambda *a, _k=kind, **kw: calls.append(_k) or True)
    for kind in ("asksaveasfilename", "askopenfilename", "askdirectory"):
        monkeypatch.setattr(fd, kind, lambda *a, **kw: "")
    return calls


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def ts(hhmmss):
    return pd.Timestamp(DAY + hhmmss)


def frame():
    idx = pd.date_range(DAY + "00:00:00", periods=120, freq="30s")
    return pd.DataFrame({"a": np.linspace(0, 1, len(idx)),
                         "b": np.linspace(1, 0, len(idx))}, index=idx)


def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["a"])
    axs["panel2"].plot(df.index, df["b"])


def build(tracks, tmp_path, layout=None, panes=None):
    from chronotagger.labeler import TimeIntervalLabeler
    kw = dict(df=frame(), window=pd.Timedelta("30min"),
              autosave_folder=str(tmp_path), tracks=[dict(t) for t in tracks])
    if panes is not None:
        kw["panes"] = panes
    else:
        kw["plot_fn"] = plot_fn
        kw["layout_spec"] = dict(layout or LAYOUT)
    lbl = TimeIntervalLabeler(**kw)
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    return lbl


@pytest.fixture
def one(tmp_path):
    lbl = build(ONE_LANE, tmp_path)
    yield lbl
    lbl.root.destroy()


@pytest.fixture
def two(tmp_path):
    lbl = build(TWO_LANES, tmp_path)
    yield lbl
    lbl.root.destroy()


@pytest.fixture
def four(tmp_path):
    lbl = build(FOUR_LANES, tmp_path)
    yield lbl
    lbl.root.destroy()


def rings(lbl, pane=None):
    """The focus ring artists, by GID. A Patch's get_transform() is a
    COMPOSITE, so comparing it to ax.transAxes never matches."""
    p = pane if pane is not None else lbl.active_pane
    return [a for a in p.strip_ax.patches
            if a.get_gid() == strip_focus_gid()]


def previews(lbl, pane=None):
    """The dashed preview rectangles, by GID. The strip also carries ONE
    pre-existing always-invisible Rectangle (the time overlay), on the
    pre-M2 tree too, which a bare `ax.patches` census would count."""
    p = pane if pane is not None else lbl.active_pane
    return [a for a in p.strip_ax.patches
            if a.get_gid() == strip_preview_gid()]


def collection(lbl, pane=None):
    p = pane if pane is not None else lbl.active_pane
    for c in p.strip_ax.collections:
        if c.get_gid() == BANDS_GID:
            return c
    return None


def band_ys(coll):
    """(lo, hi) per face, rounded, in face order."""
    out = []
    for p in coll.get_paths():
        v = p.vertices
        out.append((round(float(v[:, 1].min()), 6),
                    round(float(v[:, 1].max()), 6)))
    return out


def four_per_lane(lbl, ids):
    ivs = []
    for tid in ids:
        for j, lab in enumerate(["a", "b", "c", "d"]):
            s = ts("00:%02d:00" % (1 + j * 8))
            ivs.append(Interval(s, s + pd.Timedelta("4min"), lab, None,
                                track=tid))
    lbl.intervals[:] = ivs
    return ivs


def press(lbl, xfrac, yfrac, pane=None):
    from matplotlib.backend_bases import MouseEvent
    p = pane if pane is not None else lbl.active_pane
    ax = p.strip_ax
    bb = ax.get_window_extent()
    ev = MouseEvent("button_press_event", ax.figure.canvas,
                    int(round(bb.x0 + xfrac * bb.width)),
                    int(round(bb.y0 + yfrac * bb.height)), button=1)
    lbl._on_strip_press(ev, p)
    return ev


def pick_at(lbl, xfrac, yfrac, pane=None):
    """One real button_press_event; returns the PickEvents on the bands."""
    from matplotlib.backend_bases import MouseEvent
    p = pane if pane is not None else lbl.active_pane
    ax = p.strip_ax
    bb = ax.get_window_extent()
    ev = MouseEvent("button_press_event", ax.figure.canvas,
                    int(round(bb.x0 + xfrac * bb.width)),
                    int(round(bb.y0 + yfrac * bb.height)), button=1)
    got = []
    cid = ax.figure.canvas.mpl_connect(
        "pick_event",
        lambda pe: got.append(pe) if (
            getattr(pe.artist, "get_gid", None)
            and pe.artist.get_gid() == BANDS_GID) else None)
    ax.figure.canvas.callbacks.process("button_press_event", ev)
    ax.figure.canvas.mpl_disconnect(cid)
    return got


def xfrac_of(lbl, t):
    return float((t - lbl.t0) / (lbl.t1 - lbl.t0))


# =================================================== the painter, K == 1


def test_one_lane_paints_exactly_the_pre_m2_band(one):
    one.intervals[:] = [Interval(ts("00:02:00"), ts("00:05:00"), "sw", None,
                                 track="human")]
    one._update_plot()
    c = collection(one)
    assert c is not None
    assert band_ys(c) == [(0.1, 0.9)], "the four constants plotting.py had"
    assert one.active_pane.strip_ax.get_ylim() == (0.0, 1.0)


def test_one_lane_has_no_lane_name_no_ring_and_no_legend(one):
    one.intervals[:] = [Interval(ts("00:02:00"), ts("00:05:00"), "sw", None,
                                 track="human")]
    one._update_plot()
    ax = one.active_pane.strip_ax
    assert list(ax.get_yticks()) == []
    assert ax.get_legend() is None
    assert rings(one) == []
    assert ax.get_ylabel() == "Labels"


def test_one_lane_preview_is_exactly_the_pre_m2_rectangle(one):
    one.current_selection = (ts("00:03:00"), ts("00:07:00"))
    one._update_plot()
    rects = previews(one)
    assert len(rects) == 1
    assert round(float(rects[0].get_xy()[1]), 6) == 0.05
    assert round(float(rects[0].get_height()), 6) == 0.9


def test_one_lane_multi_span_preview_is_the_same_two_numbers(one):
    one.current_spans = [(ts("00:03:00"), ts("00:05:00")),
                         (ts("00:10:00"), ts("00:12:00"))]
    one._update_plot()
    rects = previews(one)
    assert len(rects) == 2
    for r in rects:
        assert round(float(r.get_xy()[1]), 6) == 0.05
        assert round(float(r.get_height()), 6) == 0.9


def test_the_preview_pool_is_built_on_the_active_band_and_is_0_05_at_one_lane(one):
    pool = one._ensure_strip_preview_pool(2)
    assert len(pool) == 2
    for r in pool:
        assert round(float(r.get_xy()[1]), 6) == 0.05
        assert round(float(r.get_height()), 6) == 0.9


def test_the_pool_is_CONSTRUCTED_on_the_active_lanes_band_not_the_strip(four):
    """The rectangle's geometry at BIRTH, before any frame sets it: with
    four lanes it must be the active lane's band, not 0.05..0.95."""
    four._set_active_track("l2", announce=False, repaint=False)
    four._update_plot()
    four.active_pane._strip_preview_pool = []
    pool = four._ensure_strip_preview_pool(1)
    y, h = preview_band(2, 4, pad_for(4))
    assert round(float(pool[0].get_xy()[1]), 6) == round(y, 6)
    assert round(float(pool[0].get_height()), 6) == round(h, 6)


# =================================================== the painter, K > 1


def test_one_collection_and_four_k_faces_at_every_k(four):
    for k in (1, 2, 3, 4):
        for i, t in enumerate(four.tracks):
            t.visible = (i < k)
        four._set_active_track("l0", announce=False, repaint=False)
        four_per_lane(four, [t["id"] for t in FOUR_LANES[:k]])
        four._update_plot()
        strip = four.active_pane.strip_ax
        colls = [c for c in strip.collections if c.get_gid() == BANDS_GID]
        assert len(colls) == 1, "ONE PolyCollection at every K"
        assert len(colls[0].get_paths()) == 4 * k
        assert colls[0].get_pickradius() == 0
        assert strip.get_ylim() == (0.0, 1.0)
        assert four.active_pane._strip_lane_count == k


def test_every_face_sits_in_its_own_lanes_band(four):
    four_per_lane(four, ["l0", "l1", "l2", "l3"])
    four._update_plot()
    c = collection(four)
    pad = pad_for(4)
    want = [lane_band(r, 4, pad) for r in range(4)]
    want = [(round(lo, 6), round(hi, 6)) for lo, hi in want]
    rows = four.active_pane._strip_band_rows
    for (lo, hi), r in zip(band_ys(c), rows):
        assert (lo, hi) == want[r]


def test_the_lane_names_are_the_y_ticks_and_the_active_one_is_bold(four):
    four_per_lane(four, ["l0", "l1"])
    four._set_active_track("l2", announce=False, repaint=False)
    four._update_plot()
    ax = four.active_pane.strip_ax
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["L0", "L1", "L2", "L3"]
    weights = [t.get_fontweight() for t in ax.get_yticklabels()]
    assert weights == ["normal", "normal", "bold", "normal"]


def test_the_lane_name_is_bold_even_when_the_active_lane_is_empty(four):
    """The case a per-face edge highlight cannot mark: nothing is painted
    on the active lane in this window."""
    four_per_lane(four, ["l0"])
    four._set_active_track("l3", announce=False, repaint=False)
    four._update_plot()
    ax = four.active_pane.strip_ax
    weights = [t.get_fontweight() for t in ax.get_yticklabels()]
    assert weights == ["normal", "normal", "normal", "bold"]
    assert len(rings(four)) == 1, "the ring marks the empty active lane"


def test_the_focus_ring_is_one_artist_constant_in_k_and_not_pickable(four):
    for k in (2, 3, 4):
        for i, t in enumerate(four.tracks):
            t.visible = (i < k)
        four._set_active_track("l0", announce=False, repaint=False)
        four_per_lane(four, [t["id"] for t in FOUR_LANES[:k]])
        four._update_plot()
        got = rings(four)
        assert len(got) == 1
        r = got[0]
        assert r.get_picker() is None and not r.pickable()
        lo, hi = lane_band(0, k, pad_for(k))
        assert round(float(r.get_xy()[1]), 6) == round(lo, 6)
        assert round(float(r.get_height()), 6) == round(hi - lo, 6)
        assert round(float(r.get_width()), 6) == 1.0


def test_one_legend_for_the_active_lane_titled_with_its_name(four):
    four_per_lane(four, ["l0", "l1"])
    four._set_active_track("l1", announce=False, repaint=False)
    four._update_plot()
    leg = four.active_pane.strip_ax.get_legend()
    assert leg is not None
    assert leg.get_title().get_text() == "L1"
    assert [t.get_text() for t in leg.get_texts()] == ["a", "b", "c", "d"]


def test_each_lane_paints_in_its_own_colours(two):
    """Two lanes, ONE shared class name, two different colours."""
    two.intervals[:] = [
        Interval(ts("00:02:00"), ts("00:05:00"), "sw", None, track="human"),
        Interval(ts("00:02:00"), ts("00:05:00"), "sw", None, track="rules"),
    ]
    two._update_plot()
    c = collection(two)
    faces = [tuple(round(float(x), 3) for x in fc)
             for fc in c.get_facecolor()]
    assert len(set(faces)) == 2, "a shared class name is not a shared colour"


def test_the_preview_follows_the_active_lane(four):
    four.current_selection = (ts("00:03:00"), ts("00:07:00"))
    for tid, row in (("l0", 0), ("l2", 2), ("l3", 3)):
        four._set_active_track(tid, announce=False, repaint=False)
        four._update_plot()
        rects = previews(four)
        assert len(rects) == 1
        y, h = preview_band(row, 4, pad_for(4))
        assert round(float(rects[0].get_xy()[1]), 6) == round(y, 6)
        assert round(float(rects[0].get_height()), 6) == round(h, 6)


def test_the_painter_records_the_active_band_on_the_pane(four):
    """The drag path reads this instead of recomputing per motion event:
    measured, the live call cost +0.98 ms on a 1.78 ms blit."""
    for tid, row in (("l0", 0), ("l2", 2)):
        four._set_active_track(tid, announce=False, repaint=False)
        four._update_plot()
        got = four.active_pane._strip_active_band
        want = preview_band(row, 4, pad_for(4))
        assert (round(got[0], 9), round(got[1], 9)) == (round(want[0], 9),
                                                        round(want[1], 9))


def test_the_pool_still_follows_when_the_pane_was_never_painted(one, tmp_path):
    """The fallback: a pool built before any paint asks core.lanes live."""
    lbl = build(ONE_LANE, tmp_path)
    try:
        if hasattr(lbl.active_pane, "_strip_active_band"):
            del lbl.active_pane._strip_active_band
        pool = lbl._ensure_strip_preview_pool(1)
        assert round(float(pool[0].get_xy()[1]), 6) == 0.05
        assert round(float(pool[0].get_height()), 6) == 0.9
    finally:
        lbl.root.destroy()


def test_the_pool_survives_a_repaint_so_the_blit_can_still_group_it(two):
    """A PRE-EXISTING DEFECT this pack had to fix to make the lane preview
    visible at all.

    `_update_strip` calls `ax.clear()`, which detaches every pooled
    rectangle; nothing re-added it, and `BlitHelper.draw` groups its
    artists BY `a.axes` -- so from the first repaint onwards the drag
    preview drew NOTHING. Measured on the pre-M2 tree: after a repaint
    `r.axes is None` and `r in ax.patches` is False.
    """
    import matplotlib.dates as mdates
    x0 = mdates.date2num(ts("00:03:00"))
    x1 = mdates.date2num(ts("00:06:00"))
    two._draw_strip_preview_spans([(x0, x1)])
    pane = two.active_pane
    assert pane._strip_preview_pool[0].axes is pane.strip_ax
    two._update_plot()
    two._draw_strip_preview_spans([(x0, x1)])
    r = pane._strip_preview_pool[0]
    assert r.axes is pane.strip_ax, "a detached rectangle blits nothing"
    assert r in pane.strip_ax.patches
    # and it is on the ACTIVE lane's band, not the old axes' geometry
    y, h = preview_band(0, 2, pad_for(2))
    assert round(float(r.get_xy()[1]), 6) == round(y, 6)
    assert round(float(r.get_height()), 6) == round(h, 6)


def test_the_drag_preview_pool_follows_the_active_lane_every_frame(four):
    four._set_active_track("l2", announce=False, repaint=False)
    four._update_plot()
    import matplotlib.dates as mdates
    x0 = mdates.date2num(ts("00:03:00"))
    x1 = mdates.date2num(ts("00:06:00"))
    four._draw_strip_preview_spans([(x0, x1)])
    pool = four.active_pane._strip_preview_pool
    y, h = preview_band(2, 4, pad_for(4))
    assert round(float(pool[0].get_xy()[1]), 6) == round(y, 6)
    assert round(float(pool[0].get_height()), 6) == round(h, 6)
    four._set_active_track("l0", announce=False, repaint=False)
    four._draw_strip_preview_spans([(x0, x1)])
    y, h = preview_band(0, 4, pad_for(4))
    assert round(float(pool[0].get_xy()[1]), 6) == round(y, 6)


# ====================================================== the two hit tests


def test_the_pick_path_names_the_right_interval_on_every_lane(four):
    for k in (1, 2, 3, 4):
        for i, t in enumerate(four.tracks):
            t.visible = (i < k)
        four._set_active_track("l0", announce=False, repaint=False)
        four_per_lane(four, [t["id"] for t in FOUR_LANES[:k]])
        four._update_plot()
        ivs = list(four.active_pane._strip_band_ivs)
        pad = pad_for(k)
        ok = 0
        for r in range(k):
            lo, hi = lane_band(r, k, pad)
            for j, lab in enumerate(["a", "b", "c", "d"]):
                s = ts("00:%02d:00" % (1 + j * 8))
                x = xfrac_of(four, s + pd.Timedelta("2min"))
                got = None
                for pe in pick_at(four, x, 0.5 * (lo + hi)):
                    for n in list(pe.ind):
                        if 0 <= n < len(ivs):
                            got = ivs[n]
                if got is not None and got.track == FOUR_LANES[r]["id"] \
                        and got.label == lab:
                    ok += 1
        assert ok == 4 * k, "%d of %d at K=%d" % (ok, 4 * k, k)


def test_the_press_path_names_the_right_interval_on_every_lane(four):
    for k in (1, 2, 3, 4):
        for i, t in enumerate(four.tracks):
            t.visible = (i < k)
        four._set_active_track("l0", announce=False, repaint=False)
        four_per_lane(four, [t["id"] for t in FOUR_LANES[:k]])
        four._update_plot()
        pad = pad_for(k)
        ok = 0
        for r in range(k):
            lo, hi = lane_band(r, k, pad)
            for j, lab in enumerate(["a", "b", "c", "d"]):
                s = ts("00:%02d:00" % (1 + j * 8))
                x = xfrac_of(four, s + pd.Timedelta("2min"))
                four.selected_interval = None
                press(four, x, 0.5 * (lo + hi))
                sel = four.selected_interval
                if sel is not None and sel.track == FOUR_LANES[r]["id"] \
                        and sel.label == lab:
                    ok += 1
        assert ok == 4 * k, "%d of %d at K=%d" % (ok, 4 * k, k)


def test_a_press_in_the_gutter_or_outside_every_band_selects_nothing(four):
    four_per_lane(four, ["l0", "l1", "l2", "l3"])
    four._update_plot()
    x = xfrac_of(four, ts("00:03:00"))
    for yf in (0.002, 0.25, 0.5, 0.75, 0.998):
        four.selected_interval = None
        press(four, x, yf)
        assert four.selected_interval is None, "y=%.3f" % yf


def test_a_pick_declines_in_the_gutter(four):
    four_per_lane(four, ["l0", "l1", "l2", "l3"])
    four._update_plot()
    x = xfrac_of(four, ts("00:03:00"))
    for yf in (0.25, 0.5, 0.75):
        assert pick_at(four, x, yf) == [], "y=%.3f" % yf


def test_the_resize_handles_answer_on_one_band_only(four):
    four_per_lane(four, ["l0", "l1", "l2", "l3"])
    four.selected_interval = [iv for iv in four.intervals
                              if iv.track == "l2"][0]
    four._update_plot()
    from matplotlib.backend_bases import MouseEvent
    ax = four.active_pane.strip_ax
    bb = ax.get_window_extent()
    iv = four.selected_interval
    xm = xfrac_of(four, iv.start + (iv.end - iv.start) / 2)
    x0 = xfrac_of(four, iv.start)
    pad = pad_for(4)
    for r in range(4):
        lo, hi = lane_band(r, 4, pad)
        for xf, want in ((xm, "move"), (x0, "resize_left")):
            ev = MouseEvent("motion_notify_event", ax.figure.canvas,
                            int(round(bb.x0 + xf * bb.width)),
                            int(round(bb.y0 + 0.5 * (lo + hi) * bb.height)))
            got = four._hit_test_selected(ev)
            assert got == (want if r == 2 else None), \
                "row %d gave %r" % (r, got)


def test_a_click_on_an_unlocked_lane_selects_and_activates_it(two):
    two.intervals[:] = [
        Interval(ts("00:02:00"), ts("00:05:00"), "sw", None, track="human"),
        Interval(ts("00:10:00"), ts("00:14:00"), "quiet", None,
                 track="rules"),
    ]
    two._update_plot()
    lo, hi = lane_band(1, 2, pad_for(2))
    pick_at(two, xfrac_of(two, ts("00:12:00")), 0.5 * (lo + hi))
    assert two.selected_interval is not None
    assert two.selected_interval.track == "rules"
    assert two.active_track_id == "rules"


def test_a_click_on_a_LOCKED_lane_selects_without_activating(two):
    two.track_by_id("rules").locked = True
    two.intervals[:] = [
        Interval(ts("00:02:00"), ts("00:05:00"), "sw", None, track="human"),
        Interval(ts("00:10:00"), ts("00:14:00"), "quiet", None,
                 track="rules"),
    ]
    two._update_plot()
    lo, hi = lane_band(1, 2, pad_for(2))
    pick_at(two, xfrac_of(two, ts("00:12:00")), 0.5 * (lo + hi))
    assert two.selected_interval is not None
    assert two.selected_interval.track == "rules"
    assert two.active_track_id == "human", "a locked lane is not a place " \
                                           "to draw, so do not go there"


# ========================================================= the lane switch


def test_the_lane_switch_re_points_the_class_dropdown(two):
    assert list(two.class_combo["values"]) == ["sw", "msh"]
    two._set_active_track("rules")
    assert list(two.class_combo["values"]) == ["sw", "quiet"]
    assert two.current_class_var.get() in ("sw", "quiet")


def test_the_selected_class_survives_a_switch_when_both_lanes_declare_it(two):
    two.current_class_var.set("sw")
    two._set_active_track("rules")
    assert two.current_class_var.get() == "sw"


def test_an_add_straight_after_a_switch_names_a_class_the_new_lane_declares(two):
    two.current_class_var.set("msh")
    two._set_active_track("rules")
    assert two.current_class_var.get() in two.track_by_id("rules").classes
    two.current_selection = (ts("00:02:00"), ts("00:05:00"))
    two._add_interval()
    assert len(two.intervals) == 1
    iv = two.intervals[0]
    assert iv.track == "rules"
    assert iv.label in two.track_by_id("rules").classes, \
        "an off-vocabulary interval is what makes a session unexportable"


def test_ctrl_down_and_ctrl_up_cycle_the_visible_lanes(four):
    class E(object):
        pass
    for keysym, want in (("Down", "l1"), ("Down", "l2"), ("Down", "l3"),
                         ("Down", "l0"), ("Up", "l3")):
        e = E()
        e.keysym = keysym
        e.state = 0x4
        four._on_key_press(e)
        assert four.active_track_id == want


def test_cycling_skips_a_HIDDEN_lane(four):
    """An invisible lane is not a place a gesture may land, so the cycle
    steps over it."""
    four.track_by_id("l1").visible = False
    four._set_active_track("l0", announce=False, repaint=False)

    class E(object):
        pass
    e = E()
    e.keysym = "Down"
    e.state = 0x4
    four._on_key_press(e)
    assert four.active_track_id == "l2", "l1 is hidden"
    four._on_key_press(e)
    assert four.active_track_id == "l3"
    four._on_key_press(e)
    assert four.active_track_id == "l0", "and it wraps over the hidden one"


def test_an_unmodified_arrow_key_is_not_a_lane_switch(four):
    """ONE key at a time, and asserted after each: Down-then-Up cancels
    out, and a pin that only checked the end state could not tell a live
    binding from a dead one (measured -- it let a mutant through)."""
    class E(object):
        pass
    before = four.active_track_id
    for keysym in ("Down", "Up"):
        e = E()
        e.keysym = keysym
        e.state = 0
        four._on_key_press(e)
        assert four.active_track_id == before, \
            "a bare %s must not move the active lane" % keysym


def test_the_sidebar_lane_list_holds_every_lane_and_marks_the_hidden_ones(two):
    assert two._lane_choices() == ["Human", "Rules"]
    two.track_by_id("rules").visible = False
    assert two._lane_choices() == ["Human", "Rules (hidden)"]
    two.lane_var.set("Rules (hidden)")
    two._on_lane_combo_change()
    assert two.active_track_id == "rules"
    assert two.track_by_id("rules").visible is True, \
        "picking a hidden lane is the only way back to it"


def test_ctrl_l_toggles_the_lock_and_says_so(two):
    assert two.track_by_id("human").locked is False
    two._toggle_active_lane_locked()
    assert two.track_by_id("human").locked is True
    assert "LOCKED" in two.status_var.get()
    two._toggle_active_lane_locked()
    assert two.track_by_id("human").locked is False


# ============================================================== hiding


def test_hiding_the_active_lane_moves_the_active_lane_and_says_which(two):
    two.intervals[:] = [
        Interval(ts("00:10:00"), ts("00:14:00"), "quiet", None,
                 track="rules"),
    ]
    two._toggle_active_lane_visible()
    assert two.track_by_id("human").visible is False
    assert two.active_track_id == "rules"
    msg = two.status_var.get()
    assert "hidden" in msg and "Rules" in msg
    # v2 fold F12 (V05): the toggle REPAINTS on its own -- nothing else is
    # going to. Without this the user presses Ctrl+H and keeps looking at
    # the old strip until something unrelated redraws.
    assert two.active_pane._strip_lane_count == 1
    assert len(collection(two).get_paths()) == 1


def test_the_last_visible_lane_refuses_to_hide(two):
    two.track_by_id("rules").visible = False
    two._set_active_track("human", announce=False, repaint=False)
    two._toggle_active_lane_visible()
    assert two.track_by_id("human").visible is True
    assert "refused" in two.status_var.get()


def test_a_hidden_lane_is_not_painted_but_is_still_HELD(two):
    two.intervals[:] = [
        Interval(ts("00:02:00"), ts("00:05:00"), "sw", None, track="human"),
        Interval(ts("00:10:00"), ts("00:14:00"), "quiet", None,
                 track="rules"),
    ]
    two._update_plot()
    assert len(collection(two).get_paths()) == 2
    two.track_by_id("rules").visible = False
    two._update_plot()
    assert len(collection(two).get_paths()) == 1
    assert len(two.intervals) == 2, "hiding is a VIEW change"
    assert two.active_pane._strip_lane_count == 1


def test_a_hidden_lane_is_unreachable_by_both_hit_tests(two):
    two.intervals[:] = [
        Interval(ts("00:02:00"), ts("00:05:00"), "sw", None, track="human"),
        Interval(ts("00:10:00"), ts("00:14:00"), "quiet", None,
                 track="rules"),
    ]
    two.track_by_id("rules").visible = False
    two._update_plot()
    x = xfrac_of(two, ts("00:12:00"))
    two.selected_interval = None
    press(two, x, 0.5)
    assert two.selected_interval is None
    got = []
    for pe in pick_at(two, x, 0.5):
        got.extend(list(pe.ind))
    assert got == []


def test_hiding_the_TOP_lane_does_not_shift_the_hit_test_by_one(four):
    """The lane ids both hit tests index MUST be the VISIBLE lanes.

    v2 fold F12 (V04). Built from the WHOLE table instead, this is silent
    when the hidden lane is the LAST one -- which is the case the `two`
    fixture exercises -- and off by one lane the moment the hidden lane is
    above a painted one. Measured on a tree carrying only that mutation,
    with the TOP lane hidden: a press on painted row 0 selected an interval
    on the HIDDEN lane, and no pin in v1 turned red.
    """
    four.track_by_id("l0").visible = False
    four._set_active_track("l1", announce=False, repaint=False)
    four_per_lane(four, ["l0", "l1", "l2", "l3"])
    four._update_plot()
    pane = four.active_pane
    assert list(pane._strip_lane_ids) == ["l1", "l2", "l3"]
    assert pane._strip_lane_count == 3
    x = xfrac_of(four, ts("00:03:00"))
    for row, want in ((0, "l1"), (1, "l2"), (2, "l3")):
        lo, hi = lane_band(row, 3, pad_for(3))
        four.selected_interval = None
        press(four, x, 0.5 * (lo + hi), pane)
        assert four.selected_interval is not None, \
            "painted row %d selected nothing" % row
        assert four.selected_interval.track == want, \
            "painted row %d must be lane %s" % (row, want)


def test_a_hidden_ACTIVE_lane_is_announced_on_every_repaint(two):
    """v2 fold F1. No control can leave a hidden lane active, but an UNDO
    can: the track table rides in the gesture snapshot, so undoing an edit
    made before the toggle restores the table as that edit saw it. The
    strip then has no ring, no bold name and no legend, and the next Add
    still lands there -- so the repaint says so."""
    two.track_by_id("human").visible = False       # the undo's shape
    assert two.active_track_id == "human"
    two._update_plot()
    assert two.active_pane._strip_active_row is None
    assert rings(two) == []
    msg = two.status_var.get()
    assert "is HIDDEN" in msg and "Human" in msg
    assert "Ctrl+H" in msg


def test_a_hidden_lane_still_exports(two, tmp_path):
    two.intervals[:] = [
        Interval(ts("00:02:00"), ts("00:05:00"), "sw", None, track="human"),
        Interval(ts("00:10:00"), ts("00:14:00"), "quiet", None,
                 track="rules"),
    ]
    two.track_by_id("rules").visible = False
    two._update_plot()
    p = tmp_path / "iv.csv"
    two.export_intervals(str(p), fmt="csv")
    text = p.read_text()
    assert "rules" in text and "human" in text


# ======================================================= strays, multipane


def test_an_interval_on_a_lane_the_table_does_not_hold_is_skipped_and_named(two):
    two.intervals[:] = [
        Interval(ts("00:02:00"), ts("00:05:00"), "sw", None, track="human"),
        Interval(ts("00:10:00"), ts("00:14:00"), "x", None, track="ghost"),
    ]
    two._update_plot()
    assert len(collection(two).get_paths()) == 1
    assert "1 interval(s) sit on tracks this table does not hold" \
        in two.status_var.get()


def test_painting_one_pane_does_not_corrupt_another_panes_lane_arrays(tmp_path):
    panes = [
        {"title": "A", "plot_fn": plot_fn, "layout_spec": dict(LAYOUT)},
        {"title": "B", "plot_fn": plot_fn, "layout_spec": dict(LAYOUT)},
    ]
    lbl = build(TWO_LANES, tmp_path, panes=panes)
    try:
        lbl.intervals[:] = [
            Interval(ts("00:02:00"), ts("00:05:00"), "sw", None,
                     track="human"),
            Interval(ts("00:10:00"), ts("00:14:00"), "quiet", None,
                     track="rules"),
        ]
        lbl._update_plot()
        for p in lbl.panes:
            assert p._strip_lane_count == 2
            assert len(p._strip_band_ivs) == 2
        # hide a lane and repaint ONE pane by hand: the other pane's arrays
        # must not be the ones this paint wrote
        lbl.track_by_id("rules").visible = False
        lbl._update_strip(lbl.panes[1])
        assert lbl.panes[1]._strip_lane_count == 1
        assert len(lbl.panes[1]._strip_band_ivs) == 1
        assert len(lbl.panes[0]._strip_band_ivs) == 2
    finally:
        lbl.root.destroy()


def test_every_panes_strip_repaints_on_one_update_plot(tmp_path):
    panes = [
        {"title": "A", "plot_fn": plot_fn, "layout_spec": dict(LAYOUT)},
        {"title": "B", "plot_fn": plot_fn, "layout_spec": dict(LAYOUT)},
        {"title": "C", "plot_fn": plot_fn, "layout_spec": dict(LAYOUT)},
    ]
    lbl = build(TWO_LANES, tmp_path, panes=panes)
    try:
        lbl.intervals[:] = [
            Interval(ts("00:02:00"), ts("00:05:00"), "sw", None,
                     track="human")]
        lbl._update_plot()
        for p in lbl.panes:
            c = collection(lbl, p)
            assert c is not None and len(c.get_paths()) == 1
        lbl.intervals[:] = []
        lbl._update_plot()
        for p in lbl.panes:
            assert collection(lbl, p) is None
            assert p._strip_band_ivs == []
    finally:
        lbl.root.destroy()


# ========================================================= the strip height


def hrs(lbl):
    gs = lbl.active_pane.strip_ax.get_subplotspec().get_gridspec()
    return [round(float(x), 6) for x in gs.get_height_ratios()]


def test_the_labels_row_grows_with_the_lane_count(tmp_path):
    """What the edit writes is the gridspec's height_ratios. The axes'
    PIXEL height also carries constrained_layout's padding, so the pixel
    ratio at K=4 is 1.80x rather than 3.0x -- pinned as a direction."""
    a = build(ONE_LANE, tmp_path)
    b = build(FOUR_LANES, tmp_path)
    try:
        assert hrs(a) == [1.0, 1.0, 1.0], "K == 1 is a no-op"
        assert hrs(b) == [1.0, 1.0, 3.0], "max(1.0, 0.75 * 4)"
        ha = a.active_pane.strip_ax.get_window_extent().height
        hb = b.active_pane.strip_ax.get_window_extent().height
        assert hb > ha * 1.5
    finally:
        a.root.destroy()
        b.root.destroy()


def test_two_and_three_lanes_get_their_own_share(tmp_path):
    for n, want in ((2, 1.5), (3, 2.25)):
        lbl = build(FOUR_LANES[:n], tmp_path)
        try:
            assert hrs(lbl) == [1.0, 1.0, want]
        finally:
            lbl.root.destroy()


def test_an_explicit_height_ratios_always_wins(tmp_path):
    layout = dict(LAYOUT)
    layout["height_ratios"] = [1.0, 1.0, 1.0]
    a = build(FOUR_LANES, tmp_path, layout=layout)
    try:
        assert hrs(a) == [1.0, 1.0, 1.0], "the driver's own numbers win"
    finally:
        a.root.destroy()


def test_the_labels_row_is_the_one_the_layout_names_not_the_last(tmp_path):
    """v2 fold F12 (V07). `labels_area["row"]` is read for a reason: every
    other layout in this file happens to put labels LAST, and with
    `nrows - 1` hard-coded instead the height goes onto whatever row is at
    the bottom. No v1 pin could tell the difference."""
    layout = {
        "nrows": 3, "ncols": 1,
        "areas": [
            {"key": "labels", "row": 0, "col": 0, "role": "labels"},
            {"key": "panel1", "row": 1, "col": 0, "role": "time"},
            {"key": "panel2", "row": 2, "col": 0, "role": "time"},
        ],
    }
    lbl = build(FOUR_LANES, tmp_path, layout=layout)
    try:
        assert hrs(lbl) == [3.0, 1.0, 1.0], "the LABELS row grew, not the last"
    finally:
        lbl.root.destroy()
