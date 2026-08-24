"""
Pack 3 (selection correctness) regression tests.

GUI-free MockSelectionHost on the Pack 1/2 __get__-binding pattern over bare
Agg figures. No tk.Tk() is constructed, so these tests are immune to the
machine-specific Tk-init flake -- but note that tkinter.messagebox CREATES
ITS OWN ROOT on demand, so the _stub_messagebox fixture below is what keeps
the warning branches from popping a real modal. Both are required. One
end-to-end test at the bottom uses the real-Tk `labeler` fixture.

Targets (deep-review ledger 5.7 + 7.x; evidence pack3_g1/g2):
  T1  artist scan reads the tool's own overlays -> phantom intervals
  T2  preview closed vs commit half-open -> one-sample lie
  T3  snap inert for box commits
  T4  last sample unlabelable (tail clamps + half-open exclusion)
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from chronotagger.core.models import Interval
from chronotagger.labeler.mixins.events.selection import SelectionMixin
from chronotagger.labeler.mixins.events.base import EventsBaseMixin, TOOL_GID_PREFIX
from chronotagger.labeler.mixins.events.mouse import MouseEventsMixin
from chronotagger.labeler.mixins.events.overlays import OverlaysMixin
from chronotagger.labeler.mixins.intervals.crud import IntervalCRUDMixin
from chronotagger.labeler.mixins.intervals.commands import IntervalCommandsMixin
from chronotagger.labeler.mixins.intervals.validation import IntervalValidationMixin
from chronotagger.labeler.mixins.intervals.merge import IntervalMergeMixin
from chronotagger.labeler.mixins.io_export import IOExportMixin


@pytest.fixture(autouse=True)
def _stub_messagebox(monkeypatch):
    """Any dialog-reachable path must never block headless (Pack 1 lesson)."""
    import tkinter.messagebox as mb
    calls = []
    for kind in ("showinfo", "showwarning", "showerror", "askyesno"):
        monkeypatch.setattr(
            mb, kind, lambda *a, _k=kind, **kw: calls.append(_k) or True)
    yield calls


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class _Var:
    def __init__(self, value=None):
        self._v = value

    def get(self):
        return self._v

    def set(self, v):
        self._v = v


class _Sync:
    def sync_intervals_changed(self):
        pass


class _Canvas:
    def __init__(self):
        self.draws = 0

    def draw_idle(self):
        self.draws += 1

    def draw(self):
        self.draws += 1


_BOUND_MIXINS = (
    SelectionMixin,
    MouseEventsMixin,
    EventsBaseMixin,
    IntervalCRUDMixin,
    IntervalCommandsMixin,
    IntervalValidationMixin,
    IntervalMergeMixin,
    IOExportMixin,
)


class MockSelectionHost:
    """Real mixin methods bound onto a plain object (the execution-proven
    pack3_g1 harness). The drawing/persistence surface is stubbed by the
    class methods below; stubs win because the binding loop skips names
    already present on the instance/class."""

    def __init__(self, df, layout_spec, axes_meta, user_axes,
                 snap=False, highlight=True):
        for mixin in _BOUND_MIXINS:
            for name, fn in vars(mixin).items():
                if name.startswith("__"):
                    continue
                if callable(fn) and not hasattr(self, name):
                    setattr(self, name, fn.__get__(self))

        self.df = df
        self.data_start = df.index[0]
        self.data_end = df.index[-1]
        self.t0 = df.index[0]
        self.t1 = df.index[-1]

        self.layout_spec = layout_spec
        self.axes_meta = axes_meta
        self.user_axes = user_axes
        self.strip_ax = None

        self._last_windowed_index = df.index.copy()

        self.current_spans = []
        self._commit_spans = []
        self.current_selection = None
        self._selected_component_labels = None
        self.selected_interval = None
        self.intervals = []

        self.undo_stack = []
        self.redo_stack = []
        self.max_undo = 50
        self.modified = False

        self.classes = ["UNKNOWN", "PS", "LOBE"]
        self.class_colors = {c: "#cccccc" for c in self.classes}
        self.current_class_var = _Var("PS")
        self.snap_var = _Var(bool(snap))
        self.status_var = _Var("")
        self.enable_point_highlighting = bool(highlight)

        self.canvas = _Canvas()
        self.sync_manager = _Sync()
        self.root = None

        self._preview_highlights = []
        self._interval_highlights = []

        self.CLICK_DRAG_SLOP_PX = 6
        self._press_xy_px = None
        self._dragging_box = False

        self.active_pane = self
        self.rect_selectors = {k: object() for k in user_axes}

    # ---- stubbed drawing / persistence surface ----
    def _update_strip(self):
        pass

    def _update_plot(self):
        pass

    def _save_autosave(self):
        pass

    def _update_time_overlays_for_multi_spans(self, spans):
        pass

    def _update_time_overlays(self, x0, x1, color=None):
        pass

    def _hide_time_overlays(self):
        pass

    def _draw_strip_preview_spans(self, spans_float):
        pass

    def _update_intervals_list(self):
        pass

    def _clear_two_click_state(self, keep_selection=False):
        pass


def make_df(n=120, freq="30s", xy_offset=0.0):
    idx = pd.date_range("2015-01-03 00:00:00", periods=n, freq=freq)
    return pd.DataFrame(
        {
            "log10n": np.linspace(0.5, 2.0, n),
            "X": np.arange(n, dtype=float) + xy_offset,
            "Y": np.arange(n, dtype=float) + xy_offset,
        },
        index=idx,
    )


def make_crossplot_host(df, snap=False, highlight=True):
    """One 'not-time' axis drawn with ax.scatter, NO x_col/y_col: the
    artist-scan fallback runs (the production front door, per pack3_g1)."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(df["X"].values, df["Y"].values, s=5)
    area = {"key": "xy", "role": "not-time", "row": 0, "col": 1}
    host = MockSelectionHost(
        df=df,
        layout_spec={"nrows": 1, "ncols": 2, "areas": [area]},
        axes_meta={"xy": dict(area)},
        user_axes={"xy": ax},
        snap=snap,
        highlight=highlight,
    )
    return host, fig, ax


def half_open_samples(df, spans):
    got = set()
    for s, e in spans:
        got.update(df.index[(df.index >= s) & (df.index < e)])
    return got


BOX = dict(xlo=59.5, xhi=89.5, ylo=59.5, yhi=89.5)   # rows 60..89


# ------------------------------------------------------------------ T1

def test_artist_scan_ignores_red_preview_overlay():
    """Two identical boxes must select the same single block. Today the
    second box also returns a phantom pinned to the window start
    (pack3_g1 T1-A: shot #2 = 2 spans / 60 samples)."""
    df = make_df()
    host, fig, ax = make_crossplot_host(df)

    first = host._box_select_via_artists(ax, **BOX)
    assert first == [(df.index[60], df.index[89])]

    # leave the red preview overlay behind, exactly as the app does
    host.current_spans = list(first)
    host._commit_spans = list(first)
    host._show_selected_point_highlights(redraw=False)
    assert host._preview_highlights, "harness must create the overlay"

    second = host._box_select_via_artists(ax, **BOX)
    assert second == first


def test_artist_scan_ignores_blue_interval_overlay():
    """A selected interval's blue overlay poisons the FIRST box today
    (pack3_g1 T1-C). It must not contribute rows."""
    df = make_df()
    host, fig, ax = make_crossplot_host(df)

    clean = host._box_select_via_artists(ax, **BOX)

    host.selected_interval = Interval(df.index[70], df.index[86], "PS")
    host._show_selected_interval_highlights()
    assert host._interval_highlights, "harness must create the blue overlay"

    poisoned = host._box_select_via_artists(ax, **BOX)
    assert poisoned == clean


def test_artist_scan_ignores_multispan_polycollection():
    """PolyCollection.get_offsets() reports [[0, 0]]; a box straddling the
    origin must not emit a phantom at windowed_idx[0] (pack3_g2 finding).
    Probed both ways: tagged (gid skip) and untagged (length belt)."""
    from matplotlib.collections import PolyCollection
    df = make_df(xy_offset=5.0)          # no data point at the origin
    host, fig, ax = make_crossplot_host(df)

    verts = [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]
    tagged = PolyCollection(verts, facecolors="yellow", alpha=0.25)
    tagged.set_gid(TOOL_GID_PREFIX + "multispan-overlay")
    ax.add_collection(tagged)
    untagged = PolyCollection(verts, facecolors="yellow", alpha=0.25)
    ax.add_collection(untagged)          # the length belt must catch this one

    spans = host._box_select_via_artists(ax, xlo=-1.0, xhi=1.0,
                                         ylo=-1.0, yhi=1.0)
    assert spans == []


def test_artist_scan_skips_partial_length_decorations():
    """A plot_fn decoration with fewer points than the windowed frame must
    not contribute ordinal-mapped rows: the ordinal assumption is only
    valid for one-point-per-row artists."""
    df = make_df()
    host, fig, ax = make_crossplot_host(df)
    ax.plot([60.0, 89.0], [60.0, 89.0], marker="o")   # 2-point marker line

    spans = host._box_select_via_artists(ax, **BOX)
    assert spans == [(df.index[60], df.index[89])]


def test_full_length_data_artists_still_scanned():
    """Control: the legitimate data cloud keeps working through the scan."""
    df = make_df()
    host, fig, ax = make_crossplot_host(df)
    spans = host._box_select_via_artists(ax, **BOX)
    assert spans == [(df.index[60], df.index[89])]


def test_tool_highlight_artists_carry_gid():
    """The gid invariant (grill Q6): every overlay the tool creates is
    name-tagged, so the length belt can never mask a missing tag."""
    df = make_df()
    host, fig, ax = make_crossplot_host(df)

    host._commit_spans = [(df.index[10], df.index[20])]
    host._show_selected_point_highlights(redraw=False)
    host.selected_interval = Interval(df.index[40], df.index[50], "PS")
    host._show_selected_interval_highlights()

    assert host._preview_highlights and host._interval_highlights
    for artist in host._preview_highlights + host._interval_highlights:
        assert str(artist.get_gid() or "").startswith(TOOL_GID_PREFIX)


def test_multispan_overlay_carries_gid():
    """The REAL PolyCollection creator must tag its artist (EDIT 059)."""
    df = make_df()
    fig, ax = plt.subplots()
    ax.plot(mdates.date2num(df.index.to_pydatetime()), df["log10n"].values)
    area = {"key": "t1", "role": "time", "row": 0, "col": 0}
    host = MockSelectionHost(
        df=df,
        layout_spec={"nrows": 1, "ncols": 1, "areas": [area]},
        axes_meta={"t1": dict(area)},
        user_axes={"t1": ax},
    )
    host._time_axis_keys = ["t1"]
    # bind the REAL overlay creator (the class-level stub must not win here)
    host._update_time_overlays_for_multi_spans = (
        OverlaysMixin._update_time_overlays_for_multi_spans.__get__(host))
    host._build_span_vertices = OverlaysMixin._build_span_vertices.__get__(host)
    host._time_overlays = {"seed": object()}      # bypass _init_time_overlays
    host._multi_span_overlay_collections = {}

    host._update_time_overlays_for_multi_spans([(df.index[5], df.index[10])])

    polys = list(host._multi_span_overlay_collections.values())
    assert polys, "the real creator must have made a PolyCollection"
    for poly in polys:
        assert str(poly.get_gid() or "").startswith(TOOL_GID_PREFIX)


# ------------------------------------------------- T2 / T3 / T4 (box lanes)

def _finalize(host, df, i0, i1):
    host._finalize_box_selection([df.index[i] for i in range(i0, i1 + 1)])


def test_snap_on_box_commit_is_sample_aligned_half_open():
    """T3 + R3: snap ON stores [t_first, t_after_last); the display spans
    stay closed on the samples (R11). Today commit is byte-identical with
    the checkbox on and off (pack3_g1 T3)."""
    df = make_df()
    host, fig, ax = make_crossplot_host(df, snap=True)
    _finalize(host, df, 60, 89)
    assert host._commit_spans == [(df.index[60], df.index[90])]
    assert host.current_spans == [(df.index[60], df.index[89])]


def test_snap_off_box_commit_is_padded_and_labels_same_samples():
    """R3: snap OFF keeps padded midpoints; the labeled sample set is
    identical to snap ON."""
    df = make_df()
    on, fig1, _ax1 = make_crossplot_host(df, snap=True)
    off, fig2, _ax2 = make_crossplot_host(df, snap=False)
    _finalize(on, df, 60, 89)
    _finalize(off, df, 60, 89)

    (s, e), = off._commit_spans
    assert s not in df.index and e not in df.index   # midpoints, not samples
    assert (half_open_samples(df, off._commit_spans)
            == half_open_samples(df, on._commit_spans)
            == set(df.index[60:90]))


def test_box_commit_labels_the_last_sample_snap_on():
    """T4: a tail box labels the final sample; the end is the one-index-unit
    cap. make_df's index is the pandas-3.0 default, MICROSECOND, so the
    literal +1ns this asserted before Pack 6.5 was an end the index cannot
    represent (R65-1)."""
    df = make_df()
    host, fig, ax = make_crossplot_host(df, snap=True)
    _finalize(host, df, 115, 119)
    (s, e), = host._commit_spans
    assert e == df.index[119] + pd.Timedelta(1, unit=df.index.unit)
    assert e.unit == df.index.unit
    assert Interval(s, e, "PS").contains(df.index[-1])


def test_box_commit_labels_the_last_sample_snap_off():
    """T4 on the padded lane (EDIT 065): today the clamp pins the end AT
    data_end and the final sample exports as -1 (pack3_g1 T4-A)."""
    df = make_df()
    host, fig, ax = make_crossplot_host(df, snap=False)
    _finalize(host, df, 115, 119)
    assert df.index[-1] in half_open_samples(df, host._commit_spans)


def test_single_last_sample_box_labels_one_sample():
    """T4-B was a silent false success: 'Added 1 interval(s)' labeling ZERO
    samples. Now it labels exactly the final sample."""
    df = make_df()
    host, fig, ax = make_crossplot_host(df, snap=True)
    _finalize(host, df, 119, 119)
    assert half_open_samples(df, host._commit_spans) == {df.index[-1]}


def test_preview_equals_commit_for_box_lanes():
    """T2/R1: the WYSIWYG invariant, both lanes. Today snapped previews
    show one more sample than commits label (21 vs 20, pack3_g1 T2)."""
    df = make_df()
    for snap in (True, False):
        host, fig, ax = make_crossplot_host(df, snap=snap)
        _finalize(host, df, 60, 89)
        assert (set(host._get_preview_timestamps())
                == half_open_samples(df, host._commit_spans)), snap
        plt.close(fig)


def test_preview_equals_commit_for_single_span_flow():
    """T2/R5: current_selection flows through the same converter the commit
    door uses, so the red dots equal the labeled samples."""
    df = make_df()
    host, fig, ax = make_crossplot_host(df)
    host.current_selection = (df.index[10], df.index[30])
    shown = set(host._get_preview_timestamps())
    committed = half_open_samples(
        df, host._exact_spans_to_half_open([host.current_selection]))
    assert shown == committed == set(df.index[10:31])


def test_preview_is_derived_from_commit_spans_not_display_spans():
    """R1 single source of truth: the highlighter reads _commit_spans, never
    the closed display spans. Pre-pack it read current_spans with a closed
    mask, so a stale display value leaked into the highlight."""
    df = make_df()
    host, fig, ax = make_crossplot_host(df)
    host._commit_spans = [(df.index[10], df.index[20])]
    host.current_spans = [(df.index[60], df.index[89])]   # stale display value
    assert set(host._get_preview_timestamps()) == set(df.index[10:20])


def test_exact_spans_to_half_open_converter():
    """Unit contract of the new converter (EDIT 060)."""
    df = make_df()
    host, fig, ax = make_crossplot_host(df)
    idx = df.index

    # end ON a sample -> advances to the next sample
    assert (host._exact_spans_to_half_open([(idx[5], idx[10])])
            == [(idx[5], idx[11])])
    # end BETWEEN samples -> unchanged (already exclusive-correct)
    mid = idx[10] + pd.Timedelta(seconds=7)
    assert host._exact_spans_to_half_open([(idx[5], mid)]) == [(idx[5], mid)]
    # end ON the final sample -> one unit of the index's resolution past
    # data_end (R65-1; make_df's index is microsecond)
    assert (host._exact_spans_to_half_open([(idx[115], idx[119])])
            == [(idx[115], idx[119] + pd.Timedelta(1, unit=idx.unit))])


def test_add_interval_door_converts_current_selection():
    """R5 at the commit door: a two-click/full-height selection ending ON
    sample 20 stores [idx[5], idx[21]) and labels sample 20. Today it
    stores [idx[5], idx[20]) and silently drops it."""
    df = make_df()
    host, fig, ax = make_crossplot_host(df)
    host.current_selection = (df.index[5], df.index[20])
    host._add_interval()

    assert len(host.intervals) == 1
    iv = host.intervals[0]
    assert iv.start == df.index[5]
    assert iv.end == df.index[21]
    assert iv.contains(df.index[20])


def test_selected_interval_highlight_matches_contains():
    """R12: the blue overlay marks exactly the samples the interval
    contains (today: one extra, closed mask at selection.py:1255)."""
    df = make_df()
    host, fig, ax = make_crossplot_host(df)
    iv = Interval(df.index[10], df.index[20], "PS")
    host.selected_interval = iv
    host._show_selected_interval_highlights()

    assert len(host._interval_highlights) == 1
    n_marks = host._interval_highlights[0].get_offsets().shape[0]
    n_contained = int(sum(iv.contains(t) for t in df.index))
    assert n_marks == n_contained == 10


def test_export_preview_count_matches_csv():
    """R13: the export dialog's row count equals the CSV's labeled-row
    count (today: one extra per interval)."""
    df = make_df()
    host, fig, ax = make_crossplot_host(df)
    host.intervals = [Interval(df.index[5], df.index[15], "PS"),
                      Interval(df.index[40], df.index[60], "LOBE")]
    _preview_df, total = host._get_first_labeled_rows(5)
    ids = host._compute_label_id_series()
    assert total == int((ids.values != -1).sum()) == 30


# ---------------------------------------------------- real-Tk end-to-end

def test_box_select_then_add_is_one_gesture_and_labels_tail(labeler):
    """Pack 1 interaction pin + T4 on the real app: a snap-ON box over the
    DATA TAIL commits through the real door as ONE undo entry and labels the
    final sample of the dataset (the mid-window box this test used to draw
    exercised neither the tail clamp nor the end cap)."""
    df = labeler.df
    labeler.t0 = df.index[90]
    labeler.t1 = labeler.data_end
    labeler._update_plot()
    labeler.snap_var.set(True)
    labeler._finalize_box_selection([t for t in df.index[109:120]])

    depth_before = len(labeler.undo_stack)
    labeler._add_interval()

    assert len(labeler.undo_stack) == depth_before + 1
    assert len(labeler.intervals) == 1
    iv = labeler.intervals[0]
    assert iv.end == df.index[-1] + pd.Timedelta(1, unit=df.index.unit)
    assert iv.contains(df.index[-1])
