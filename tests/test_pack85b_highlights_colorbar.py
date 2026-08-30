"""
Pack 8.5-B (highlights + colorbar gutter) regression tests.

Everything here runs in the regime the suite never covered: DRAW
DECIMATION ACTIVE, a DUPLICATED timestamp in the frame, and a pane that
reserved a colorbar gutter column.

What each group owns:
  B1 highlights   -- the extractor reads the DRAWN frame, so the marks
                     survive decimation; the pre-redraw fallback survives
  B2 fastindex    -- positions_nearest on a duplicated index, and
                     value-identical to pandas on unique ones
  B3 error path   -- our own callbacks reach the forensic log and redden
                     the status line, without monkeypatching matplotlib
  B4 colorbar     -- a bar spans exactly its owner when the pane reserves
                     a gutter column, and keeps doing so across redraws,
                     a pan, a resize round trip and a tab switch
  B6 known limits -- an image panel has no vertices to mark, and says so
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from chronotagger.labeler.utils.fastindex import (
    _nearest_on_sorted_duplicates,
    positions_nearest,
)
from chronotagger.labeler.utils.spectrogram import (
    attach_colorbar,
    draw_spectrogram,
    gutter_column,
    registered_colorbar,
    take_layout_dirty,
)

PREVIEW_GID = "chronotagger:preview-highlight"
INTERVAL_GID = "chronotagger:interval-highlight"


@pytest.fixture(autouse=True)
def _stub_messagebox(monkeypatch):
    """STANDING RULE (Pack 3): any dialog-reachable path, real-Tk or not --
    tkinter.messagebox creates its own root on demand."""
    import tkinter.messagebox as mb
    calls = []
    for kind in ("showinfo", "showwarning", "showerror", "askyesno",
                 "askyesnocancel"):
        monkeypatch.setattr(
            mb, kind, lambda *a, _k=kind, **kw: calls.append(_k) or True)
    return calls


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------- helpers

def marks(lbl, gid):
    """(artist count, total marker count) for one highlight family."""
    arts = [c for ax in lbl.user_axes.values() for c in ax.collections
            if (c.get_gid() or "") == gid]
    return len(arts), sum(int(a.get_offsets().shape[0]) for a in arts)


class Ev:
    """The event stand-in tests/test_gui_events.py uses."""

    def __init__(self, xdata=None, ydata=None, inaxes=None, button=1):
        self.xdata = xdata
        self.ydata = ydata
        self.inaxes = inaxes
        self.button = button


class NoY:
    """No ydata attribute at all -> the full-height (time-only) branch."""

    def __init__(self, xdata, inaxes, button=1):
        self.xdata = xdata
        self.inaxes = inaxes
        self.button = button


# --------------------------------------------------- B1: under decimation

DEC_ROWS = 20000


@pytest.fixture
def df_dec():
    """Wide enough that decimation engages: 1400 px panel, 4 samples per
    pixel is the early exit, so ~5,600 rows is the floor."""
    idx = pd.date_range("2015-01-03", periods=DEC_ROWS, freq="1s")
    t = np.linspace(0, 200, DEC_ROWS)
    return pd.DataFrame({"BX": np.sin(t) * 10, "BY": np.cos(t) * 5},
                        index=idx)


@pytest.fixture
def dec_labeler(df_dec, tmp_path):
    """All-time pane, decimation ON -- the regime the suite never pinned."""
    from chronotagger.labeler import TimeIntervalLabeler
    layout = {
        "nrows": 3, "ncols": 1,
        "areas": [
            {"key": "panel1", "row": 0, "col": 0, "role": "time"},
            {"key": "panel2", "row": 1, "col": 0, "role": "time"},
            {"key": "labels", "row": 2, "col": 0, "role": "labels"},
        ],
    }

    def plot_fn(axs, df, t0, t1):
        axs["panel1"].plot(df.index, df["BX"])
        axs["panel2"].plot(df.index, df["BY"])

    lbl = TimeIntervalLabeler(
        df=df_dec, plot_fn=plot_fn, layout_spec=layout,
        window=df_dec.index[-1] - df_dec.index[0],
        autosave_folder=str(tmp_path))
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    yield lbl
    lbl.root.destroy()


def test_the_fixture_is_actually_decimating(dec_labeler):
    """Every B1 pin below is vacuous if this is False -- which is exactly
    how the defect survived 583 green tests."""
    lbl = dec_labeler
    assert lbl._decim_active is True
    drawn = len(lbl.user_axes["panel1"].lines[0].get_xdata(orig=False))
    full = len(lbl.df.loc[lbl.t0:lbl.t1])
    assert drawn < full, "%d drawn of %d" % (drawn, full)
    assert len(lbl._last_windowed_index) == drawn


def test_preview_highlights_appear_while_decimation_is_active(dec_labeler):
    """B1's headline. Measured on the investigator's 1,024,926-row frame
    at a 2-day window (44,572 in the window, 36,361 drawn): ZERO marks
    before this pack, 2,026 after."""
    lbl = dec_labeler
    idx = lbl.df.index
    lbl._commit_spans = [(idx[2000], idx[9000])]
    lbl._show_selected_point_highlights(redraw=False)

    n_art, n_marks = marks(lbl, PREVIEW_GID)
    assert n_art == 2, "one artist per time panel; got %d" % n_art
    assert n_marks > 0, "the marks vanished under decimation again"
    # 7,000 timestamps -> cap step 7 -> 1000 probes, one per panel line
    assert n_marks == 2000, n_marks


def test_the_marks_land_on_drawn_vertices(dec_labeler):
    """WYSIWYG: every highlighted x is a vertex of the drawn line, which
    is the property the length gate exists to guarantee and the one that
    was silently false under decimation."""
    lbl = dec_labeler
    idx = lbl.df.index
    lbl._commit_spans = [(idx[500], idx[3000])]
    lbl._show_selected_point_highlights(redraw=False)

    line = lbl.user_axes["panel1"].lines[0]
    drawn_x = set(np.asarray(line.get_xdata(orig=False), dtype=float))
    art = [c for c in lbl.user_axes["panel1"].collections
           if (c.get_gid() or "") == PREVIEW_GID]
    assert art, "no preview artist"
    got = np.asarray(art[0].get_offsets())[:, 0]
    assert got.size > 0
    assert set(np.asarray(got, dtype=float)) <= drawn_x, \
        "a highlight landed on a sample that was never drawn"


def test_interval_highlights_appear_while_decimation_is_active(dec_labeler):
    """The blue family shares the extractor, so it shared the defect."""
    lbl = dec_labeler
    from chronotagger.core.models import Interval
    idx = lbl.df.index
    lbl.selected_interval = Interval(start=idx[1000], end=idx[6000],
                                     label="UNKNOWN")
    lbl._show_selected_interval_highlights()
    n_art, n_marks = marks(lbl, INTERVAL_GID)
    assert n_art == 2 and n_marks > 0


def test_a_full_height_drag_highlights_under_decimation(dec_labeler):
    """The gesture the user reported: a full-height time drag takes the
    time-only branch, which does NOT suspend decimation (only the box
    branch does, selection.py:143) -- so it was the branch that broke."""
    lbl = dec_labeler
    ax = lbl.user_axes["panel1"]
    x0 = mdates.date2num(lbl.t0 + (lbl.t1 - lbl.t0) * 0.25)
    x1 = mdates.date2num(lbl.t0 + (lbl.t1 - lbl.t0) * 0.75)
    lbl._on_rectangle_select(NoY(x0, ax), NoY(x1, ax), lbl.active_pane)
    assert lbl._decim_active is True, "the branch must not have re-rendered"
    n_art, n_marks = marks(lbl, PREVIEW_GID)
    assert n_art == 2 and n_marks > 0


def test_the_cached_frame_and_index_always_agree(dec_labeler):
    """They are written together; a future edit that moves one and not the
    other puts the length gate back where it was."""
    lbl = dec_labeler
    for _ in range(3):
        lbl._update_plot()
        assert lbl._last_windowed_frame.index.equals(lbl._last_windowed_index)


def test_an_extraction_before_any_redraw_still_works(dup_labeler):
    """Pack 6 D11's slice survives as the FALLBACK: a host that never
    redrew -- the suite's hand-built cross-plot hosts, and any pane whose
    first extraction beats its first frame -- reads `df.loc[t0:t1]`
    exactly as it always did. This frame does not decimate, so the two
    sources agree and the fallback is measurable against the cache."""
    lbl = dup_labeler
    assert lbl._decim_active is False
    with_cache = lbl._extract_data_at_indices(lbl.user_axes["bmag"],
                                              [0, 1, 2, 3])
    del lbl._last_windowed_frame
    assert getattr(lbl, "_last_windowed_frame", None) is None
    without = lbl._extract_data_at_indices(lbl.user_axes["bmag"],
                                           [0, 1, 2, 3])
    assert len(without[0]) == 4, "the fallback stopped extracting"
    assert without == with_cache


def test_decimation_is_still_draw_only(dec_labeler):
    """The invariant B1 must not have moved: what is LABELLED still comes
    from the full frame, whatever the marks show."""
    lbl = dec_labeler
    idx = lbl.df.index
    lbl._commit_spans = [(idx[2000], idx[9000])]
    ts = lbl._get_preview_timestamps()
    assert len(ts) == 7000, len(ts)
    assert len(lbl._last_windowed_index) < len(lbl.df.loc[lbl.t0:lbl.t1])


# ------------------------------------------------------ B2: fastindex

def _dup_index(n=600):
    idx = pd.date_range("2011-08-14", periods=n, freq="3851ms")
    vals = list(idx)
    vals.insert(300, vals[299])          # one repeated stamp, as real data
    return pd.DatetimeIndex(vals)


def test_positions_nearest_survives_a_duplicated_index():
    """One repeated timestamp in 22,282 killed every rectangle select in
    the tool with InvalidIndexError (evidence pack85b_g1_regressions.md
    section 2). The twin at fastindex.py:105 had the guard all along."""
    idx = _dup_index()
    assert not idx.is_unique and idx.is_monotonic_increasing
    with pytest.raises(Exception):
        idx.get_indexer(idx[:5], method="nearest")   # what pandas does
    got = positions_nearest(idx, idx[:5])
    assert got == [0, 1, 2, 3, 4]


def test_a_duplicated_stamp_resolves_to_its_first_row():
    """Same convention as the sibling helper's `slice.start`."""
    idx = _dup_index()
    assert idx[299] == idx[300]
    assert positions_nearest(idx, [idx[299]]) == [299]


def test_positions_nearest_is_value_identical_to_pandas_on_unique_indexes():
    """The equivalence B2 asks to be pinned: the duplicate-safe path must
    agree with get_indexer(method="nearest") everywhere pandas will
    answer at all. Measured over 25,200 probes and both environments:
    zero mismatches."""
    rng = np.random.default_rng(0)
    for n, m in ((10, 200), (2000, 5000)):
        base = pd.date_range("2011-08-14", periods=n, freq="3s")
        span = int((base[-1] - base[0]).value)
        off = rng.integers(-5_000_000_000, span + 5_000_000_000, size=m)
        probes = pd.to_datetime(pd.DatetimeIndex(base[0].value + off))
        want = np.asarray(base.get_indexer(probes, method="nearest"))
        got = np.asarray(_nearest_on_sorted_duplicates(base, probes))
        assert np.array_equal(want, got), \
            "%d of %d probes differ at n=%d" % ((want != got).sum(), m, n)


def test_the_nearest_tie_goes_the_way_pandas_sends_it():
    """A probe exactly between two samples: pandas takes the LATER one
    (np.where(left_dist < right_dist, left, right)). A discriminating
    case -- flipping the comparison passes every random test above."""
    idx = pd.date_range("2015-01-01", periods=5, freq="10s")
    tie = idx[0] + pd.Timedelta("5s")
    probes = pd.DatetimeIndex([idx[0] - pd.Timedelta("1h"), idx[0], tie,
                               idx[-1], idx[-1] + pd.Timedelta("1h")])
    want = [int(v) for v in idx.get_indexer(probes, method="nearest")]
    got = [int(v) for v in _nearest_on_sorted_duplicates(idx, probes)]
    assert want == got == [0, 0, 1, 4, 4]


def test_a_non_monotonic_index_still_raises():
    """The module's whole reason for refusing searchsorted: on an
    unsorted frame it answered 500 probes confidently and wrongly. That
    frame must keep the pandas raise."""
    a = pd.date_range("2015-01-03 00:00:00", periods=300, freq="30s")
    b = pd.date_range("2015-01-03 00:00:15", periods=300, freq="30s")
    nm = a.append(b)
    assert not nm.is_monotonic_increasing
    with pytest.raises(ValueError):
        positions_nearest(nm, nm[:5])
    nm_dup = nm.append(pd.DatetimeIndex([nm[0]]))
    assert not nm_dup.is_unique
    with pytest.raises(Exception):
        positions_nearest(nm_dup, nm_dup[:5])


def test_mixed_datetime_resolutions_do_not_derail_the_fallback():
    """The unit trap the module docstring records, in its 2026 form:
    Index.searchsorted REFUSES ns probes against a us index rather than
    converting ('Cannot losslessly cast ... ns to us'), so the helper
    promotes both sides to a common datetime64 unit first."""
    import datetime as dt
    py = [dt.datetime(2015, 1, 1) + dt.timedelta(seconds=10 * i)
          for i in range(6)]
    idx = pd.DatetimeIndex(py)
    probes = pd.DatetimeIndex(np.array(
        [np.datetime64("2015-01-01T00:00:04.000000001"),
         np.datetime64("2015-01-01T00:00:26.000000001")]))
    got = [int(v) for v in _nearest_on_sorted_duplicates(idx, probes)]
    assert got == [0, 3]


@pytest.fixture
def dup_labeler(tmp_path):
    """A single-line time panel on a frame with one repeated stamp. One
    line, deliberately: a box across several components routes into the
    component dialog instead (selection.py:358)."""
    from chronotagger.labeler import TimeIntervalLabeler
    idx = _dup_index(1200)
    df = pd.DataFrame({"BX": np.sin(np.linspace(0, 30, len(idx))) * 10},
                      index=idx)
    layout = {
        "nrows": 2, "ncols": 1,
        "areas": [
            {"key": "bmag", "row": 0, "col": 0, "role": "time"},
            {"key": "labels", "row": 1, "col": 0, "role": "labels"},
        ],
    }
    lbl = TimeIntervalLabeler(
        df=df, plot_fn=lambda axs, d, t0, t1: axs["bmag"].plot(
            d.index, d["BX"]),
        layout_spec=layout, window=idx[-1] - idx[0],
        autosave_folder=str(tmp_path))
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    yield lbl
    lbl.root.destroy()


def test_a_box_select_completes_on_a_duplicated_index(dup_labeler):
    """End to end, the gesture that died: a y-banded box on a frame whose
    index repeats one stamp. Before this pack it raised InvalidIndexError
    out of the callback, the status bar kept saying 'Ready', and nothing
    reached the log."""
    lbl = dup_labeler
    assert not lbl.df.index.is_unique
    ax = lbl.user_axes["bmag"]
    ylo, yhi = ax.get_ylim()
    mid = 0.5 * (ylo + yhi)
    band = 0.30 * (yhi - ylo)
    x0 = mdates.date2num(lbl.t0 + (lbl.t1 - lbl.t0) * 0.20)
    x1 = mdates.date2num(lbl.t0 + (lbl.t1 - lbl.t0) * 0.80)

    lbl._on_rectangle_select(Ev(x0, mid - band, ax),
                             Ev(x1, mid + band, ax), lbl.active_pane)

    assert lbl._commit_spans, "the gesture selected nothing"
    n_art, n_marks = marks(lbl, PREVIEW_GID)
    assert n_marks > 0
    assert "No points" not in lbl.status_var.get()


# ------------------------------------------------- B3: error surfacing

@pytest.fixture
def guard_labeler(tmp_path):
    from chronotagger.labeler import TimeIntervalLabeler
    idx = pd.date_range("2015-01-03", periods=300, freq="30s")
    df = pd.DataFrame({"BX": np.linspace(0, 1, len(idx))}, index=idx)
    layout = {
        "nrows": 2, "ncols": 1,
        "areas": [
            {"key": "bmag", "row": 0, "col": 0, "role": "time"},
            {"key": "labels", "row": 1, "col": 0, "role": "labels"},
        ],
    }
    lbl = TimeIntervalLabeler(
        df=df, plot_fn=lambda axs, d, t0, t1: axs["bmag"].plot(
            d.index, d["BX"]),
        layout_spec=layout, window=pd.Timedelta("1h"),
        autosave_folder=str(tmp_path))
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    yield lbl
    lbl.root.destroy()


def _status_foregrounds(lbl):
    want = str(lbl.status_var)
    out, stack = [], [lbl.root]
    while stack:
        w = stack.pop()
        try:
            stack.extend(w.winfo_children())
        except Exception:
            pass
        try:
            if str(w.cget("textvariable")) == want:
                out.append(str(w.cget("foreground")))
        except Exception:
            continue
    return out


def test_our_callbacks_are_wrapped_at_all(guard_labeler):
    """A DISCRIMINATING check: the guard is what the selector holds, not
    something the labeler merely owns.

    It sees exactly ONE of G5's nineteen registrations. The six
    edge-clamping connects have no object to ask, so the pin below reads
    the source instead -- SECTION 0a G15."""
    key = sorted(guard_labeler.active_pane.rect_selectors)[0]
    onselect = guard_labeler.active_pane.rect_selectors[key].onselect
    assert onselect.__name__ == "guarded_rectangle_select"


#: G5's enumeration, name for name: every callback canvas.py registers
#: that dispatches into OUR code. Nineteen of them.
GUARDED_REGISTRATIONS = (
    "resize",
    "scroll_zoom",
    "time_click",
    "time_motion",
    "rectangle_select",
    "zoom_box",
    "strip_click",
    "strip_press",
    "strip_motion",
    "strip_release",
    "right_click_cancel",
    "gate_press",
    "gate_release",
    "rect_clamp_motion",
    "rect_clamp_press",
    "rect_clamp_release",
    "zoom_clamp_motion",
    "zoom_clamp_press",
    "zoom_clamp_release",
)


def test_every_registration_we_own_goes_through_the_guard():
    """B3's coverage floor, and the only pin that can see six of them.

    The edge-clamping connects are figure-level `mpl_connect`
    registrations with no accessor, so no behavioural pin in this file
    reaches them. Measured before this pin existed: with the `_guard_cb`
    wrap removed from `rect_clamp_motion`, the whole 613-test suite
    stayed GREEN in both environments.

    The set is exact on purpose. Dropping a wrap removes its name;
    renaming one changes it; adding a twentieth registration means G5's
    enumeration is stale and this pin is where that gets noticed.

    A source scan is deliberately weaker than driving the handler. It is
    what an unreachable registration admits of without reading
    matplotlib's private callback registry, which B3's fence forbids.
    SECTION 0a G15.
    """
    import re
    from chronotagger.labeler.mixins.view_build import canvas as mod

    assert len(GUARDED_REGISTRATIONS) == 19, "G5 enumerates NINETEEN"

    # utf-8 explicitly: canvas.py carries 279 non-ASCII bytes and the
    # Windows default (cp1252) cannot decode them.
    with open(mod.__file__, encoding="utf-8") as fh:
        src = fh.read()

    want = set(GUARDED_REGISTRATIONS)
    found = set(re.findall(r'_guard_cb\(\s*"([A-Za-z_]+)"', src))
    assert found == want, (
        "the guarded registrations moved -- UNWRAPPED or renamed: %s; "
        "new and unenumerated: %s"
        % (sorted(want - found), sorted(found - want)))

    # ... and the one registration that must stay UNwrapped. It runs from
    # INSIDE a draw, where setting a Tk variable is a re-entrancy hazard
    # rather than error surfacing (G5).
    assert 'mpl_connect("draw_event", pane._blit.recache)' in src, \
        "the blit draw_event connect must stay UNwrapped (G5)"


def test_a_failing_callback_reaches_the_forensic_log_and_reddens_status(
        guard_labeler, tmp_path):
    """Measured baseline on the shipped tree, same gesture: status 'Ready'
    before and 'Ready' after, forensic log +0 bytes on the second
    identical gesture."""
    lbl = guard_labeler
    logfile = tmp_path / "chronotagger.log"
    before = logfile.stat().st_size if logfile.exists() else 0
    assert lbl.status_var.get() == "Ready"
    assert _status_foregrounds(lbl) == [""]

    def boom(*_a, **_k):
        raise ValueError("pinned: deliberate callback failure")

    lbl._on_rectangle_select = boom
    key = sorted(lbl.active_pane.rect_selectors)[0]
    onselect = lbl.active_pane.rect_selectors[key].onselect
    ax = lbl.user_axes["bmag"]

    onselect(Ev(0.0, 0.0, ax), Ev(1.0, 1.0, ax))       # must NOT raise

    assert "ValueError" in lbl.status_var.get()
    assert "chronotagger.log" in lbl.status_var.get()
    assert _status_foregrounds(lbl) == ["#b00020"]
    after1 = logfile.stat().st_size
    assert after1 > before, "the forensic log did not grow"
    text = logfile.read_text(encoding="utf-8", errors="replace")
    assert "pinned: deliberate callback failure" in text
    assert "Traceback" in text

    # and the SECOND identical failure is not swallowed by a _warn_once
    onselect(Ev(0.0, 0.0, ax), Ev(1.0, 1.0, ax))
    assert logfile.stat().st_size > after1, \
        "the second identical failure went unrecorded"


def test_the_red_clears_when_anything_else_writes_the_status(guard_labeler):
    lbl = guard_labeler

    def boom(*_a, **_k):
        raise RuntimeError("pinned")

    lbl._on_rectangle_select = boom
    key = sorted(lbl.active_pane.rect_selectors)[0]
    ax = lbl.user_axes["bmag"]
    lbl.active_pane.rect_selectors[key].onselect(Ev(0.0, 0.0, ax),
                                                 Ev(1.0, 1.0, ax))
    assert _status_foregrounds(lbl) == ["#b00020"]
    lbl.status_var.set("Ready")
    assert _status_foregrounds(lbl) == [""]


def test_matplotlibs_exception_handler_is_left_alone(guard_labeler):
    """B3's fence: we wrap OUR handlers, we do not monkeypatch
    matplotlib. Replacing cbook._exception_printer would change the
    behaviour of every callback in the process, ours and matplotlib's
    own widgets alike."""
    from matplotlib import cbook
    assert (guard_labeler.canvas.callbacks.exception_handler
            is cbook._exception_printer)


# -------------------------------------------------- B4: colorbar gutter

N_CH = 8
CH = ["C%d" % i for i in range(N_CH)]


@pytest.fixture
def spec_df():
    idx = pd.date_range("2011-08-14", periods=1500, freq="3851ms")
    base = np.logspace(4.0, 6.0, N_CH)[:, None]
    ripple = 1.0 + 0.25 * np.sin(np.linspace(0, 12, len(idx)))[None, :]
    Z = base * ripple
    data = {"ion_n": np.linspace(0.1, 3.0, len(idx))}
    for i, c in enumerate(CH):
        data[c] = Z[i]
    return pd.DataFrame(data, index=idx)


def _spec_labeler(df, tmp_path, n_spec=1, gutter=True, extra_areas=None):
    from chronotagger.labeler import TimeIntervalLabeler
    keys = ["spec%d" % i for i in range(n_spec)]
    nrows = n_spec + 1
    layout = {"nrows": nrows, "ncols": 2 if gutter else 1, "hspace": 0.05,
              "areas": []}
    if gutter:
        layout["width_ratios"] = [1.0, 0.07]
    for i, k in enumerate(keys):
        layout["areas"].append({"key": k, "row": i, "col": 0, "role": "time"})
    layout["areas"].append({"key": "labels", "row": n_spec, "col": 0,
                            "role": "labels"})
    for a in (extra_areas or []):
        layout["areas"].append(a)

    def plot_fn(axs, d, t0, t1):
        Z = d[CH].to_numpy(dtype=float).T
        for k in keys:
            im = draw_spectrogram(axs[k], d.index, Z, n_cols=200,
                                  norm=LogNorm(1e3, 1e8), cmap="jet")
            attach_colorbar(axs[k], im, label="eflux")

    lbl = TimeIntervalLabeler(
        df=df, plot_fn=plot_fn, layout_spec=layout, classes=["UNKNOWN"],
        window=pd.Timedelta("1h"), step=pd.Timedelta("30min"),
        autosave_folder=str(tmp_path))
    lbl._build_gui()
    lbl.root.withdraw()
    lbl.fig.set_size_inches(14, 8, forward=False)
    return lbl, keys


def _ratio(lbl, key):
    cb = registered_colorbar(lbl.user_axes[key])
    op = lbl.user_axes[key].get_position()
    bp = cb.ax.get_position()
    return (round(float(bp.height / op.height), 4),
            abs(float(bp.y0 - op.y0)) < 0.01 and abs(float(bp.y1 - op.y1)) < 0.01)


def test_gutter_column_is_the_first_free_column_right_of_the_owner():
    """Not "the last column": the flagship puts cross-plots in column 2
    and the gutter in column 1, so the bar lands beside the panel it
    belongs to."""
    fig = plt.figure()
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.07, 1.0])
    owner = fig.add_subplot(gs[0, 0])
    fig.add_subplot(gs[1, 0])
    fig.add_subplot(gs[0:2, 2])
    assert gutter_column(owner) == 1
    plt.close(fig)

    fig2 = plt.figure()
    gs2 = fig2.add_gridspec(2, 1)
    only = fig2.add_subplot(gs2[0, 0])
    fig2.add_subplot(gs2[1, 0])
    assert gutter_column(only) is None
    plt.close(fig2)


def test_a_gutter_bar_spans_exactly_its_owner(spec_df, tmp_path):
    """B4's headline. The shipped whole-group design measured 2.118 times
    its owner's height with one spectrogram and 4.666 with three."""
    lbl, keys = _spec_labeler(spec_df, tmp_path, n_spec=1)
    try:
        lbl._update_plot()
        ratio, spans = _ratio(lbl, keys[0])
        assert spans, "the bar does not sit on its owner's rows"
        assert ratio == 1.0, ratio
    finally:
        lbl.root.destroy()


def test_two_spectrograms_get_one_bar_each_on_its_own_panel(spec_df,
                                                            tmp_path):
    lbl, keys = _spec_labeler(spec_df, tmp_path, n_spec=2)
    try:
        lbl._update_plot()
        assert len(keys) == 2
        for k in keys:
            ratio, spans = _ratio(lbl, k)
            assert spans and ratio == 1.0, (k, ratio)
        y0s = sorted(registered_colorbar(lbl.user_axes[k]).ax
                     .get_position().y0 for k in keys)
        assert y0s[1] - y0s[0] > 0.1, "the two bars are stacked on top"
    finally:
        lbl.root.destroy()


def test_the_gutter_geometry_survives_redraws_a_pan_and_a_resize(spec_df,
                                                                 tmp_path):
    """The stress the reanchor prototype failed: it returned 0.0153 to
    0.0734 of drift over a resize round trip. The gutter returns
    0.000000."""
    lbl, keys = _spec_labeler(spec_df, tmp_path, n_spec=2)
    try:
        lbl._update_plot()

        def snap():
            return [tuple(np.round(registered_colorbar(lbl.user_axes[k])
                                   .ax.get_position().extents, 6))
                    for k in keys]

        birth = snap()
        for _ in range(3):
            lbl._update_plot()
        assert snap() == birth, "a plain redraw moved the bars"
        lbl._next_window()
        assert snap() == birth, "a pan moved the bars"

        lbl.fig.set_size_inches(10.0, 6.0)
        lbl._invalidate_layout_freeze()
        lbl._update_plot()
        assert snap() != birth, "the resize did not re-solve at all"
        lbl.fig.set_size_inches(14.0, 8.0)
        lbl._invalidate_layout_freeze()
        lbl._update_plot()
        assert snap() == birth, "the resize round trip did not come back"
        for k in keys:
            ratio, spans = _ratio(lbl, k)
            assert spans and ratio == 1.0
    finally:
        lbl.root.destroy()


def test_the_bar_is_created_once_and_the_figure_does_not_grow(spec_df,
                                                              tmp_path):
    """Pack 8.5's idempotence pin, re-run on the gutter path."""
    lbl, keys = _spec_labeler(spec_df, tmp_path, n_spec=1)
    try:
        counts, bars = [], []
        for _ in range(4):
            lbl._update_plot()
            counts.append(len(lbl.fig.axes))
            bars.append(registered_colorbar(lbl.user_axes[keys[0]]))
        assert len(set(counts)) == 1, counts
        assert all(b is bars[0] for b in bars)
    finally:
        lbl.root.destroy()


def test_the_layout_is_re_solved_once_per_bar_and_not_once_per_frame(
        spec_df, tmp_path, monkeypatch):
    """ONE extra constrained-layout solve per BAR. The Pack 5 freeze is
    worth ~120 ms a frame and a bar that asked for a re-solve every frame
    would quietly hand all of it back."""
    lbl, keys = _spec_labeler(spec_df, tmp_path, n_spec=2)
    try:
        calls = []
        real = lbl._invalidate_layout_freeze
        monkeypatch.setattr(
            lbl, "_invalidate_layout_freeze",
            lambda pane=None: calls.append(1) or real(pane))
        lbl._update_plot()
        assert len(calls) == 1, (
            "the birth frame must ask for exactly one re-solve: %d"
            % len(calls))
        for _ in range(3):
            lbl._update_plot()
        assert len(calls) == 1, "a later frame asked for another solve"
        assert take_layout_dirty(lbl.user_axes.values()) is False
        assert lbl.active_pane._layout_frozen is True
    finally:
        lbl.root.destroy()


def test_a_pane_with_no_free_column_keeps_the_shared_group_bar(spec_df,
                                                               tmp_path):
    """The fallback is the Pack 8.5 behaviour, unchanged: the bar is laid
    out against the whole shared-x group, which is why it is 2.1x its
    owner's height here. Every x axis still matches, which is what that
    design buys."""
    lbl, keys = _spec_labeler(spec_df, tmp_path, n_spec=1, gutter=False)
    try:
        lbl._update_plot()
        assert gutter_column(lbl.user_axes[keys[0]]) is None
        ratio, spans = _ratio(lbl, keys[0])
        assert not spans and ratio > 1.5, ratio
        assert (round(lbl.user_axes[keys[0]].get_position().x1, 6)
                == round(lbl.active_pane.strip_ax.get_position().x1, 6))
    finally:
        lbl.root.destroy()


def test_gutter_false_forces_the_fallback_on_a_pane_that_has_one(spec_df,
                                                                 tmp_path):
    """The escape hatch, and a discriminating check that the gutter branch
    is what produces the 1.000 above."""
    from chronotagger.labeler import TimeIntervalLabeler
    layout = {"nrows": 2, "ncols": 2, "width_ratios": [1.0, 0.07],
              "areas": [
                  {"key": "spec", "row": 0, "col": 0, "role": "time"},
                  {"key": "labels", "row": 1, "col": 0, "role": "labels"}]}

    def plot_fn(axs, d, t0, t1):
        im = draw_spectrogram(axs["spec"], d.index,
                              d[CH].to_numpy(dtype=float).T, n_cols=200,
                              norm=LogNorm(1e3, 1e8), cmap="jet")
        attach_colorbar(axs["spec"], im, gutter=False)

    lbl = TimeIntervalLabeler(
        df=spec_df, plot_fn=plot_fn, layout_spec=layout,
        classes=["UNKNOWN"], window=pd.Timedelta("1h"),
        autosave_folder=str(tmp_path))
    lbl._build_gui()
    lbl.root.withdraw()
    try:
        lbl._update_plot()
        assert gutter_column(lbl.user_axes["spec"]) == 1
        cb = registered_colorbar(lbl.user_axes["spec"])
        op = lbl.user_axes["spec"].get_position()
        assert cb.ax.get_position().height / op.height > 1.5
    finally:
        lbl.root.destroy()


def test_a_gutter_bar_survives_a_tab_switch(spec_df, tmp_path):
    """_on_tab_changed ends in _update_plot, and the pane that is not
    drawn must not lose its bar either."""
    from chronotagger.labeler import TimeIntervalLabeler

    def make_layout():
        return {"nrows": 2, "ncols": 2, "width_ratios": [1.0, 0.07],
                "areas": [
                    {"key": "spec", "row": 0, "col": 0, "role": "time"},
                    {"key": "labels", "row": 1, "col": 0, "role": "labels"}]}

    def plot_fn(axs, d, t0, t1):
        im = draw_spectrogram(axs["spec"], d.index,
                              d[CH].to_numpy(dtype=float).T, n_cols=200,
                              norm=LogNorm(1e3, 1e8), cmap="jet")
        attach_colorbar(axs["spec"], im, label="eflux")

    panes = [{"title": "A", "plot_fn": plot_fn, "layout_spec": make_layout()},
             {"title": "B", "plot_fn": plot_fn, "layout_spec": make_layout()}]
    lbl = TimeIntervalLabeler(
        df=spec_df, panes=panes, classes=["UNKNOWN"],
        window=pd.Timedelta("1h"), autosave_folder=str(tmp_path))
    lbl._build_gui()
    lbl.root.withdraw()
    try:
        lbl._update_plot()
        first = tuple(np.round(registered_colorbar(
            lbl.user_axes["spec"]).ax.get_position().extents, 6))
        lbl.notebook.select(1)
        lbl._on_tab_changed(None)           # ends in _update_plot()
        assert lbl.active_pane_idx == 1
        ratio, spans = _ratio(lbl, "spec")
        assert spans and ratio == 1.0, ratio
        lbl.notebook.select(0)
        lbl._on_tab_changed(None)
        assert lbl.active_pane_idx == 0
        assert tuple(np.round(registered_colorbar(
            lbl.user_axes["spec"]).ax.get_position().extents, 6)) == first
    finally:
        lbl.root.destroy()


# ------------------------------------------------- B6: documented limits

def test_a_spectrogram_only_pane_shows_no_point_highlights(spec_df,
                                                           tmp_path):
    """DOCUMENTED LIMIT (B6), pinned so nobody files it as a bug twice: an
    image has no vertices, so there is nothing for the extractor to mark.
    A pane made only of spectrograms shows zero red dots under every
    gesture, and that is complete behaviour, not a failure."""
    lbl, keys = _spec_labeler(spec_df, tmp_path, n_spec=1)
    try:
        lbl._update_plot()
        ax = lbl.user_axes[keys[0]]
        assert len(ax.images) == 1 and len(ax.lines) == 0
        idx = lbl.df.index
        lbl._commit_spans = [(idx[10], idx[400])]
        lbl._show_selected_point_highlights(redraw=False)
        assert marks(lbl, PREVIEW_GID) == (0, 0)
    finally:
        lbl.root.destroy()


def test_a_box_on_an_image_panel_reports_no_points_in_selection(spec_df,
                                                                tmp_path):
    """DOCUMENTED LIMIT (B6): the artist scan reads ax.lines and
    ax.collections; an AxesImage is neither. The refusal is quiet and
    correct, and SP-R7 (what a y band on an image should even mean) stays
    parked."""
    lbl, keys = _spec_labeler(spec_df, tmp_path, n_spec=1)
    try:
        lbl._update_plot()
        ax = lbl.user_axes[keys[0]]
        ylo, yhi = ax.get_ylim()
        mid = 0.5 * (ylo + yhi)
        band = 0.20 * (yhi - ylo)
        x0 = mdates.date2num(lbl.t0 + (lbl.t1 - lbl.t0) * 0.25)
        x1 = mdates.date2num(lbl.t0 + (lbl.t1 - lbl.t0) * 0.75)
        lbl._on_rectangle_select(Ev(x0, mid - band, ax),
                                 Ev(x1, mid + band, ax), lbl.active_pane)
        assert lbl.status_var.get() == "No points in selection"
    finally:
        lbl.root.destroy()
