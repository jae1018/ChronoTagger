"""Pack 6 PART B -- pins for the ported clamp, the fixes, and the deletions.

Three kinds of test live here.

1. BEHAVIOUR pins for the FIVE edits that change what the code does:
   the inverted-y clamp port (R1), the load-time index sort (R9), the
   directory-fsync warning (F7), the loader's numeric-'time' refusal (F8)
   and the export sidecar (F14).
2. One REGRESSION pin for a deletion that is only behaviour-preserving if
   it is done a particular way (D11).
3. ABSENCE checks: the deleted names must no longer resolve.  Deletions
   are otherwise pinned by the suite staying green.

No timing assertion appears anywhere in this file.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import logging
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import chronotagger
from chronotagger.labeler import TimeIntervalLabeler
from chronotagger.labeler.tab_pane import TabPane
from chronotagger.labeler.mixins.events.selection import SelectionMixin
from chronotagger.labeler.mixins.events.strip import StripInteractionMixin
from chronotagger.labeler.utils import atomic_io

SRC_ROOT = Path(chronotagger.__file__).resolve().parent


def _src_text(*parts: str) -> str:
    return (SRC_ROOT.joinpath(*parts)).read_text(encoding="utf-8")


def _all_src_files():
    return [p for p in SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts]


# =====================================================================
# R1 -- the ported clamp: an inverted y-axis snaps to the right edge
# =====================================================================

class _ClampHost:
    """Bare host carrying only the method under test (the MockPersistHost
    pattern).  It has no active_pane, no _blit and no canvas, so the
    clamp's redraw tail is a no-op and the assertion is purely about the
    geometry it computed."""

    _clamp_rectangle_to_axes = SelectionMixin._clamp_rectangle_to_axes


class _Press:
    def __init__(self, xdata, ydata):
        self.xdata = xdata
        self.ydata = ydata


class _Selector:
    def __init__(self, x_start, y_start, artist):
        self._eventpress = _Press(x_start, y_start)
        self._selection_artist = artist


class _Event:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def _axes_for_clamp(ylim, xlim=(0.0, 10.0)):
    fig = Figure(figsize=(4, 3), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    fig.canvas.draw()
    return fig, ax


def _drag(ax, x_start, y_start, event_x, event_y):
    """Run one clamped motion and return the rectangle's (bottom, top)."""
    artist = Rectangle((0, 0), 0, 0, visible=False)
    ax.add_patch(artist)
    sel = _Selector(x_start, y_start, artist)
    _ClampHost()._clamp_rectangle_to_axes(_Event(event_x, event_y), ax, sel)
    return artist.get_y(), artist.get_y() + artist.get_height()


def test_clamp_snaps_to_the_top_edge_of_an_INVERTED_y_axis():
    """ax.invert_yaxis() is routine for altitude, pressure and energy
    panels.  On such an axis get_ylim() returns (hi, lo), and the affine
    clamp the live code used -- max(ymin, min(ymax, y)) -- collapses to
    the constant ymin no matter where the mouse is.  The ported block
    decides the edge in SCREEN space, so a mouse above the axes gets the
    top edge, which on an inverted axis is ymax."""
    fig, ax = _axes_for_clamp(ylim=(4.2, -0.2))
    bbox = ax.bbox

    bottom, top = _drag(ax, x_start=5.0, y_start=2.0,
                        event_x=(bbox.x0 + bbox.x1) / 2.0,
                        event_y=bbox.y1 + 60.0)

    # Mouse above the axes -> the rectangle reaches the TOP of the data
    # range, which is get_ylim()[1] == -0.2 here.
    assert bottom == pytest.approx(-0.2)
    assert top == pytest.approx(2.0)
    # And NOT the pre-port answer, which was the constant ymin.
    assert top != pytest.approx(4.2)


def test_clamp_snaps_to_the_bottom_edge_of_an_INVERTED_y_axis():
    fig, ax = _axes_for_clamp(ylim=(4.2, -0.2))
    bbox = ax.bbox

    bottom, top = _drag(ax, x_start=5.0, y_start=2.0,
                        event_x=(bbox.x0 + bbox.x1) / 2.0,
                        event_y=bbox.y0 - 60.0)

    assert bottom == pytest.approx(2.0)
    assert top == pytest.approx(4.2)


def test_clamp_is_unchanged_on_an_ordinary_axis():
    """The port must be inert where the old arithmetic was already right:
    the census measured 0 differences in 100 positions on a normal linear
    y-axis and 0 in 100 on a datetime x-axis."""
    fig, ax = _axes_for_clamp(ylim=(-1.0, 3.0))
    bbox = ax.bbox

    bottom, top = _drag(ax, x_start=5.0, y_start=1.0,
                        event_x=(bbox.x0 + bbox.x1) / 2.0,
                        event_y=bbox.y1 + 60.0)

    assert bottom == pytest.approx(1.0)
    assert top == pytest.approx(3.0)


def test_clamp_snaps_x_to_the_left_edge_when_the_mouse_leaves_sideways():
    fig, ax = _axes_for_clamp(ylim=(-1.0, 3.0), xlim=(0.0, 10.0))
    bbox = ax.bbox
    artist = Rectangle((0, 0), 0, 0, visible=False)
    ax.add_patch(artist)
    sel = _Selector(6.0, 1.0, artist)

    _ClampHost()._clamp_rectangle_to_axes(
        _Event(bbox.x0 - 80.0, (bbox.y0 + bbox.y1) / 2.0), ax, sel)

    assert artist.get_x() == pytest.approx(0.0)
    assert artist.get_x() + artist.get_width() == pytest.approx(6.0)


def test_clamp_snaps_x_to_the_LEFT_edge_on_a_LOG_y_axis():
    """DRAFT FLAG F5, pinned.  The log-y half of P6-R1 IS discriminable.

    On a log y-axis transData is non-affine: far above the axes the inverse
    evaluates 10**exponent, which overflows to +inf, and the affine matrix
    multiply that finishes the inverse then computes 0.0 * inf for the X
    row -- so x_data comes back NaN.  The pre-port arithmetic evaluated
    max(xmin, min(xmax, nan)); Python's min/max return the FIRST operand on
    a NaN comparison, so that resolved to xmax.  The rubber band snapped to
    the RIGHT edge while the mouse was to the LEFT of the axes.  The ported
    block decides the edge in screen space and is immune.

    These are the census's own 3-of-100 differing log-y positions
    (pack6_repro_clamp_trap.py's grid uses bbox.y1 + 100000 with three
    x-values left of the axes).
    """
    fig, ax = _axes_for_clamp(ylim=(1.0, 1000.0))
    ax.set_yscale("log")
    ax.set_ylim(1.0, 1000.0)
    fig.canvas.draw()
    bbox = ax.bbox

    event_x = bbox.x0 - 40.0            # LEFT of the axes
    event_y = bbox.y1 + 100000.0        # far above it

    # Assert the mechanism, so a matplotlib change that removes the
    # overflow turns this into a failure rather than a silent tautology.
    x_raw, y_raw = ax.transData.inverted().transform((event_x, event_y))
    assert math.isnan(x_raw), "expected the non-affine inverse to go NaN in x"
    assert math.isinf(y_raw)

    artist = Rectangle((0, 0), 0, 0, visible=False)
    ax.add_patch(artist)
    sel = _Selector(6.0, 10.0, artist)

    _ClampHost()._clamp_rectangle_to_axes(_Event(event_x, event_y), ax, sel)

    # Mouse LEFT of the axes -> the LEFT edge.  Pre-port this was xmax.
    assert artist.get_x() == pytest.approx(0.0)
    assert artist.get_x() + artist.get_width() == pytest.approx(6.0)
    # The y half is unaffected on a normally-oriented log axis.
    assert artist.get_y() == pytest.approx(10.0)
    assert artist.get_y() + artist.get_height() == pytest.approx(1000.0)

# =====================================================================
# R9 -- a non-monotonic index is sorted, loudly
# =====================================================================

def _frame(n=120):
    idx = pd.date_range("2015-01-03 00:00:00", periods=n, freq="30s")
    return pd.DataFrame({"log10n": np.linspace(0.5, 2.0, n),
                         "BX": np.linspace(-5.0, 5.0, n)}, index=idx)


def _plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["log10n"])


LAYOUT = {
    "nrows": 2, "ncols": 1,
    "areas": [
        {"key": "panel1", "row": 0, "col": 0, "role": "time"},
        {"key": "labels", "row": 1, "col": 0, "role": "labels"},
    ],
}


def test_an_unsorted_frame_is_sorted_at_construction_with_a_warning(
        tmp_path, caplog):
    """Before this, an unsorted index produced silent nonsense rather than
    an error: data_end landed BEFORE data_start, t1 < t0, and
    df.loc[t0:t1] fell back to a boolean mask returning the WHOLE dataset,
    so every 'window' contained everything and the app drew and navigated
    without complaint."""
    ordered = _frame()
    shuffled = ordered.sample(frac=1.0, random_state=0)
    assert not shuffled.index.is_monotonic_increasing

    with caplog.at_level(logging.WARNING, logger="chronotagger.labeler.app"):
        lbl = TimeIntervalLabeler(
            df=shuffled, plot_fn=_plot_fn, layout_spec=LAYOUT,
            window=pd.Timedelta("10min"), autosave_folder=str(tmp_path))

    assert lbl.df.index.is_monotonic_increasing
    assert lbl.data_start == ordered.index[0]
    assert lbl.data_end == ordered.index[-1]
    assert lbl.t0 < lbl.t1

    # The window is a window again, not the whole dataset.
    assert 0 < len(lbl.df.loc[lbl.t0:lbl.t1]) < len(ordered)

    records = [r for r in caplog.records
               if r.name == "chronotagger.labeler.app"
               and r.levelno == logging.WARNING]
    assert len(records) == 1
    msg = records[0].getMessage()
    assert "monotonic" in msg

    # The sort also moves dataset_fingerprint(), because that hashes
    # index[0] and index[-1] -- so the Pack 2 autosave filename changes and
    # an autosave written from the unsorted frame is no longer offered for
    # recovery. The user is told in the same breath as the sort.
    assert "fingerprint" in msg
    assert "autosave" in msg


def test_the_callers_frame_is_not_mutated(tmp_path):
    """sort_index() returns a copy; the frame the caller handed in keeps
    whatever order it had."""
    shuffled = _frame().sample(frac=1.0, random_state=0)
    before = list(shuffled.index)

    TimeIntervalLabeler(df=shuffled, plot_fn=_plot_fn, layout_spec=LAYOUT,
                        window=pd.Timedelta("10min"),
                        autosave_folder=str(tmp_path))

    assert list(shuffled.index) == before


def test_a_sorted_frame_is_silent(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="chronotagger.labeler.app"):
        TimeIntervalLabeler(df=_frame(), plot_fn=_plot_fn, layout_spec=LAYOUT,
                            window=pd.Timedelta("10min"),
                            autosave_folder=str(tmp_path))

    assert [r for r in caplog.records
            if r.name == "chronotagger.labeler.app"
            and r.levelno == logging.WARNING] == []


# =====================================================================
# F7 -- an EIO from the directory fsync is no longer invisible
# =====================================================================

def test_a_failing_directory_fsync_logs_exactly_once_and_never_raises(
        tmp_path, monkeypatch, caplog):
    """sync_dir=True exists to promise rename-durability for a
    user-initiated save.  When the fsync fails that promise is broken, and
    before this the failure produced no exception and no log line -- the
    user was told the save had succeeded."""
    calls = {"open": 0, "close": 0}

    def fake_open(path, flags):
        calls["open"] += 1
        return 4242

    def boom(fd):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(atomic_io, "_dirsync_warn_logged", False)
    monkeypatch.setattr(atomic_io.os, "open", fake_open)
    monkeypatch.setattr(atomic_io.os, "fsync", boom)
    monkeypatch.setattr(atomic_io.os, "close",
                        lambda fd: calls.__setitem__("close",
                                                     calls["close"] + 1))

    target = tmp_path / "session.json"
    logger_name = "chronotagger.labeler.utils.atomic_io"
    with caplog.at_level(logging.WARNING, logger=logger_name):
        atomic_io._sync_dir(target)      # must not raise
        atomic_io._sync_dir(target)      # warn-ONCE per session

    records = [r for r in caplog.records
               if r.name == logger_name and r.levelno == logging.WARNING]
    assert len(records) == 1
    assert "durable" in records[0].getMessage()
    # The fd is still closed on the failing path, both times.
    assert calls["open"] == 2 and calls["close"] == 2


def test_a_windows_style_open_failure_stays_quiet(tmp_path, monkeypatch,
                                                  caplog):
    """os.open(dir, O_RDONLY) raises PermissionError on EVERY Windows
    call by design.  Warning there would fire once per session on every
    Windows box and mean nothing, so the guard sits on the fsync."""
    def refuse(path, flags):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(atomic_io, "_dirsync_warn_logged", False)
    monkeypatch.setattr(atomic_io.os, "open", refuse)

    logger_name = "chronotagger.labeler.utils.atomic_io"
    with caplog.at_level(logging.WARNING, logger=logger_name):
        atomic_io._sync_dir(tmp_path / "session.json")

    assert [r for r in caplog.records if r.name == logger_name] == []


# =====================================================================
# F8 -- a numeric column named 'time' is not a time axis
# =====================================================================

def _auto_detect(df):
    from chronotagger.quickstart.file_loader import FileLoaderDialog

    class _Bare:
        pass

    return FileLoaderDialog._auto_detect_time_column(_Bare(), df)


def test_a_numeric_column_named_time_is_refused_and_the_real_one_wins():
    """The frame a space physicist actually has: a float 'time' column of
    seconds, and the genuine timestamps under another name.  Detection
    used to return 'time', pandas read it as nanoseconds since 1970, the
    whole dataset collapsed into a single 1970 instant, and validation
    passed."""
    n = 60
    df = pd.DataFrame({
        "time": np.arange(n, dtype="float64"),
        "Bx_nT": np.linspace(-5.0, 5.0, n),
        "epoch": pd.date_range("2015-01-03", periods=n, freq="30s"),
    })

    assert _auto_detect(df) == "epoch"


def test_a_numeric_time_column_alone_detects_nothing():
    n = 60
    df = pd.DataFrame({
        "time": np.arange(n, dtype="float64"),
        "Bx_nT": np.linspace(-5.0, 5.0, n),
    })

    assert _auto_detect(df) is None


def test_a_genuine_time_column_is_still_detected_by_name():
    """The gate must not break the case it exists to serve."""
    n = 60
    df = pd.DataFrame({
        "time": pd.date_range("2015-01-03", periods=n, freq="30s"),
        "Bx_nT": np.linspace(-5.0, 5.0, n),
    })

    assert _auto_detect(df) == "time"


def test_a_string_time_column_is_still_detected_by_name():
    n = 12
    stamps = pd.date_range("2015-01-03", periods=n, freq="1min")
    df = pd.DataFrame({
        "time": [s.isoformat() for s in stamps],
        "Bx_nT": np.linspace(-5.0, 5.0, n),
    })

    assert _auto_detect(df) == "time"


# =====================================================================
# F14 -- export_per_sample writes a label map beside its output
# =====================================================================

@pytest.fixture
def exporter(tmp_path):
    from chronotagger.core.models import Interval

    lbl = TimeIntervalLabeler(
        df=_frame(), plot_fn=_plot_fn, layout_spec=LAYOUT,
        classes=["UNKNOWN", "plasmasheet", "lobe"],
        window=pd.Timedelta("10min"), autosave_folder=str(tmp_path))
    idx = lbl.df.index
    lbl.intervals = [
        Interval(start=idx[10], end=idx[20], label="plasmasheet"),
        Interval(start=idx[40], end=idx[50], label="lobe"),
    ]
    return lbl


def test_export_per_sample_writes_a_label_map_beside_the_csv(exporter,
                                                             tmp_path):
    out = tmp_path / "per_sample.csv"
    exporter.export_per_sample(str(out), fmt="csv")

    sidecar = tmp_path / "per_sample_label_map.json"
    assert out.exists()
    assert sidecar.exists()

    mapping = json.loads(sidecar.read_text(encoding="utf-8"))
    assert mapping == {"UNKNOWN": 0, "plasmasheet": 1, "lobe": 2}

    # Every id the data file carries is either -1 or a value the map
    # explains -- which is the whole point of shipping the map.
    written = pd.read_csv(out, index_col=0)
    ids = set(int(v) for v in written["label_id"].unique())
    assert ids - {-1} <= set(mapping.values())


def test_export_per_sample_writes_a_label_map_beside_the_parquet(exporter,
                                                                 tmp_path):
    out = tmp_path / "per_sample.parquet"
    exporter.export_per_sample(str(out), fmt="parquet")

    assert out.exists()
    assert (tmp_path / "per_sample_label_map.json").exists()


def test_export_intervals_gets_no_sidecar(exporter, tmp_path):
    """export_intervals writes label STRINGS; there is nothing to map."""
    out = tmp_path / "intervals.csv"
    exporter.export_intervals(str(out), fmt="csv")

    assert out.exists()
    assert not (tmp_path / "intervals_label_map.json").exists()


def test_a_failing_sidecar_write_is_reported_not_swallowed(exporter,
                                                           tmp_path,
                                                           monkeypatch):
    """The data file is written first, so the only reachable partial state
    is 'complete data, missing map' -- and the caller is told."""
    from chronotagger.labeler.mixins import io_export as ioe

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(ioe, "atomic_write_json", boom)
    out = tmp_path / "per_sample.csv"

    with pytest.raises(RuntimeError) as excinfo:
        exporter.export_per_sample(str(out), fmt="csv")

    assert "label" in str(excinfo.value)
    assert out.exists()          # the data file did land


# =====================================================================
# D11 -- removing the dead read must not silently kill cross-plot
#        highlights
# =====================================================================

@pytest.fixture
def labeler_with_cross_plot(tmp_path):
    def fn(axs, df, t0, t1):
        axs["time1"].plot(df.index, df["log10n"])
        axs["xy_plot"].plot(df["BX"], df["log10n"])

    layout = {
        "nrows": 2, "ncols": 2,
        "areas": [
            {"key": "time1", "role": "time", "row": 0, "col": 0},
            {"key": "xy_plot", "role": "not-time", "row": 0, "col": 1,
             "x_col": "BX", "y_col": "log10n"},
            {"key": "labels", "role": "labels", "row": 1, "col": 0},
        ],
    }
    lbl = TimeIntervalLabeler(df=_frame(), plot_fn=fn, layout_spec=layout,
                             window=pd.Timedelta("30min"),
                             autosave_folder=str(tmp_path))
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    yield lbl
    lbl.root.destroy()


def test_cross_plot_highlights_still_find_their_points(
        labeler_with_cross_plot):
    """The regression this pins is subtle and the rest of the suite is
    blind to it.  _last_windowed_index IS written (plotting.py:342), so
    deleting the dead _last_windowed_df read while KEEPING a guard on
    _last_windowed_index would skip the block that binds windowed_df; the
    resulting NameError is swallowed by the not-time branch's own
    `except Exception` and the highlights silently return nothing.
    Measured on this exact shape: 5 points before, 0 after."""
    lbl = labeler_with_cross_plot
    assert len(getattr(lbl, "_last_windowed_index", [])) > 0

    ax = lbl.user_axes["xy_plot"]
    xs, ys = lbl._extract_data_at_indices(ax, [0, 1, 2, 3, 4])

    assert len(xs) == 5
    assert len(ys) == 5

    ax_time = lbl.user_axes["time1"]
    xs_t, ys_t = lbl._extract_data_at_indices(ax_time, [0, 1, 2, 3, 4])
    assert len(xs_t) == 5
    assert len(ys_t) == 5


# =====================================================================
# R6 -- the colorbar module is gone; the layout-freeze input is not
# =====================================================================

def test_every_pane_records_that_it_has_a_layout_solver(tmp_path):
    """Pack 5's freeze/re-solve pair reads pane._layout_constrained.  The
    colorbar deletion removed the only thing that could make it False, and
    the attribute itself must survive."""
    lbl = TimeIntervalLabeler(df=_frame(), plot_fn=_plot_fn,
                              layout_spec=LAYOUT,
                              window=pd.Timedelta("30min"),
                              autosave_folder=str(tmp_path))
    lbl._build_gui()
    lbl.root.withdraw()
    try:
        pane = lbl.active_pane
        assert pane._layout_constrained is True
        assert pane._layout_frozen in (True, False)
        assert pane.fig.get_layout_engine() is not None
    finally:
        lbl.root.destroy()


def test_the_dead_layout_key_no_longer_switches_the_solver_off(tmp_path):
    """A layout_spec carrying the old time_lane_cbar_gutter key used to
    lose constrained_layout in exchange for a gutter nothing ever drew."""
    spec = dict(LAYOUT)
    spec["time_lane_cbar_gutter"] = {"width": 0.05}

    lbl = TimeIntervalLabeler(df=_frame(), plot_fn=_plot_fn,
                              layout_spec=spec,
                              window=pd.Timedelta("30min"),
                              autosave_folder=str(tmp_path))
    lbl._build_gui()
    lbl.root.withdraw()
    try:
        assert lbl.active_pane._layout_constrained is True
        assert lbl.active_pane.fig.get_layout_engine() is not None
    finally:
        lbl.root.destroy()


# =====================================================================
# ABSENCE -- the deleted names no longer resolve
# =====================================================================

@pytest.mark.parametrize("module", [
    "chronotagger.labeler.utils.colorbar",
    "chronotagger.quickstart.config",
])
def test_the_deleted_modules_are_gone(module):
    assert importlib.util.find_spec(module) is None


def test_quickstart_star_import_still_works():
    """__all__ had to lose 'config' with config.py: a star-import resolves
    every name in __all__ as a submodule attribute."""
    import chronotagger.quickstart as qs

    assert "config" not in qs.__all__
    ns = {}
    exec("from chronotagger.quickstart import *", ns)   # must not raise


@pytest.mark.parametrize("name", [
    "dirty", "last_window", "manual_zooms",
])
def test_the_deleted_tabpane_fields_are_gone(name):
    assert name not in TabPane.__dataclass_fields__


@pytest.mark.parametrize("name", [
    "needs_update", "mark_clean", "mark_dirty",
])
def test_the_deleted_tabpane_methods_are_gone(name):
    assert not hasattr(TabPane, name)


@pytest.mark.parametrize("name", [
    "_squelch_xlim_events", "_apply_time_axis_format",
    "_ts_from_event", "_update_strip",
])
def test_the_shadowed_strip_copies_are_gone(name):
    """The live definitions still resolve on the app class -- what is gone
    is the copy StripInteractionMixin carried, which only the MRO order
    kept from winning."""
    assert name not in StripInteractionMixin.__dict__
    assert hasattr(TimeIntervalLabeler, name)


def test_the_live_strip_preview_machinery_is_untouched():
    """Measured at 79-81x on every preview frame: it stays."""
    for name in ("_ensure_strip_preview_pool", "_draw_strip_preview_spans"):
        assert name in StripInteractionMixin.__dict__


def test_the_shadowed_clamp_copy_is_gone_and_the_live_one_has_the_port():
    from chronotagger.labeler.mixins.events import overlays as ov

    assert "_clamp_rectangle_to_axes" not in ov.OverlaysMixin.__dict__
    live = inspect.getsourcefile(
        TimeIntervalLabeler._clamp_rectangle_to_axes)
    assert live.endswith("selection.py")
    assert "bbox.y1" in inspect.getsource(
        TimeIntervalLabeler._clamp_rectangle_to_axes)


def _code_only(path: Path) -> str:
    """The file's CODE tokens, with comments AND string literals dropped.

    Used only where the assertion is about token SHAPE -- the
    `tk . Toplevel ( )` needle below depends on this joining convention.
    For NAME-absence use `_code_and_literals` instead: this helper is
    blind to any name that lives inside a string, which is exactly how two
    absence parameters shipped vacuous at v2 (recheck FINDING 3).
    """
    import tokenize

    out = []
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    return " ".join(out)


def _strip_docstrings(node):
    body = getattr(node, "body", None)
    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                         ast.AsyncFunctionDef)) and body:
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    for child in ast.iter_child_nodes(node):
        _strip_docstrings(child)


def _code_and_literals(path: Path) -> str:
    """The file's code INCLUDING string literals, with comments and
    docstrings removed.

    This is the detector every NAME-absence check must use, and the reason
    is a measured defect, not a preference.

    Three of the names this pack deletes exist at base ONLY inside a
    string literal -- `getattr(self, "_two_click_last_x", ...)`,
    `getattr(self, '_last_windowed_df', None)` and
    `spec.get("time_lane_cbar_gutter", None)`. A CODE-TOKENS-ONLY scan
    reads 0 hits for those at base, so the corresponding checks could
    never fail: a full EDIT 181 revert left the entire 345-test suite
    green. One instance of this blindness was found and fixed at v2 (EDIT
    189's pin); the independent recheck then found the two siblings that
    were not swept.

    Comments and docstrings still have to go, because this pack
    deliberately leaves comments NAMING what it deleted and why -- a plain
    substring scan would match the rationale and pass or fail for the
    wrong reason. `ast.unparse` drops comments for free (they are not in
    the tree) and `_strip_docstrings` removes the rest.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    _strip_docstrings(tree)
    return ast.unparse(tree)


def test_the_absence_detector_sees_strings_but_not_prose(tmp_path):
    """Guard against the v2 blindness returning.

    Every NAME-absence check below is only as good as this: the detector
    must SEE a name that lives in a string literal, and must NOT see one
    that lives in a comment or a docstring. If a future refactor swaps the
    detector back to code-tokens-only, this test fails instead of a dozen
    absence checks silently going vacuous.
    """
    sample = tmp_path / "sample.py"
    sample.write_text(
        '"""A docstring mentioning DOCSTRING_NAME."""\n'
        "# a comment mentioning COMMENT_NAME\n"
        "def f(self):\n"
        '    """Inner docstring mentioning INNER_DOC_NAME."""\n'
        "    return getattr(self, 'STRING_NAME', None)\n"
        "CODE_NAME = 1\n",
        encoding="utf-8")

    seen = _code_and_literals(sample)
    assert "STRING_NAME" in seen        # the failure mode being guarded
    assert "CODE_NAME" in seen
    assert "COMMENT_NAME" not in seen
    assert "DOCSTRING_NAME" not in seen
    assert "INNER_DOC_NAME" not in seen

    # ...and the token-shape helper really is the blind one, which is why
    # it is reserved for the Toplevel needle.
    assert "STRING_NAME" not in _code_only(sample)


@pytest.mark.parametrize("name", ["_last_windowed_df", "_two_click_last_x",
                                  "_manual_zooms"])
def test_the_deleted_attribute_names_appear_nowhere_in_src(name):
    """`_code_and_literals`, NOT `_code_only`. Two of these three live at
    base only inside a string literal, so a code-tokens-only scan reads 0
    at base and can never go red (recheck FINDING 3)."""
    hits = [str(p.relative_to(SRC_ROOT))
            for p in _all_src_files()
            if name in _code_and_literals(p)]
    assert hits == []


@pytest.mark.parametrize("name", ["needs_update", "mark_clean", "mark_dirty",
                                  "ensure_lane_colorbar",
                                  "time_lane_cbar_gutter"])
def test_the_deleted_call_sites_are_gone_from_src(name):
    """Same detector, same reason: `time_lane_cbar_gutter` exists at base
    only as the string argument of `spec.get(...)`."""
    hits = [str(p.relative_to(SRC_ROOT))
            for p in _all_src_files()
            if name in _code_and_literals(p)]
    assert hits == []


def test_the_dead_two_click_read_is_gone_and_the_live_rebuild_stayed():
    """EDIT 189's pin, and DRAFT FLAG F1's.

    The parametrized `_two_click_last_x` absence check above cannot see
    EDIT 189: there the name lives inside a STRING literal
    (`getattr(self, "_two_click_last_x", ...)`) and `_code_only()` drops
    STRING tokens, so a 189-revert left that test green.  This one reads
    the raw source.

    It also pins the NARROWING (DRAFT FLAG F1).  P6-R3 named
    plotting.py:427-435, but line 430 is
    `self._rebuild_time_overlays_if_needed()` -- this block's only live
    statement and that method's only caller in the tree.  Taking the
    literal range detaches every two-click preview band from its axes on
    the first redraw (measured 3/3 attached with the call, 0/3 without)
    and the whole suite stays green.  So the call must still be here.
    """
    raw = _src_text("labeler", "mixins", "plotting.py")

    # The dead read, and the call it fed, are gone.
    assert 'getattr(self, "_two_click_last_x"' not in raw
    assert "self._update_time_overlays(self._two_click_t0, last)" not in raw

    # The live statement the ruling's line range would also have taken is
    # still there, exactly once, and still inside the two-click guard.
    assert raw.count("self._rebuild_time_overlays_if_needed()") == 1
    assert raw.count("def _rebuild_time_overlays_if_needed") == 1


def test_no_unparented_toplevel_is_left_in_src():
    """EDIT 209's pin.

    The needle has to match what _code_only() actually emits.  It joins
    TOKENS with single spaces, so `tk.Toplevel()` becomes the six tokens
    `tk . Toplevel ( )` -- with a space between `tk` and the dot.  v1 of
    this pack searched for "tk.Toplevel ( )", which the tokenizer can
    never produce, so the test passed on a tree with the bare masterless
    Toplevel restored.  Mutation-executed both ways at v2.
    """
    needle = "tk . Toplevel ( )"
    # Guard against the v1 failure mode returning: prove the needle is one
    # the tokenizer really emits, using a parented call we know is there.
    widgets = SRC_ROOT / "labeler" / "mixins" / "view_build" / "widgets.py"
    assert "tk . Toplevel ( widget )" in _code_only(widgets)

    hits = [str(p.relative_to(SRC_ROOT))
            for p in _all_src_files()
            if needle in _code_only(p)]
    assert hits == []


def test_the_rect_selectors_dict_the_parked_guards_read_still_exists(
        tmp_path):
    """Pack 6 R5 deleted the doubly-dead wrapper loop but PARKED the three
    guards that read self.rect_selectors.  They are only harmless while
    the name keeps resolving to an empty dict."""
    lbl = TimeIntervalLabeler(df=_frame(), plot_fn=_plot_fn,
                              layout_spec=LAYOUT,
                              window=pd.Timedelta("30min"),
                              autosave_folder=str(tmp_path))
    assert lbl.rect_selectors == {}

    canvas_code = _code_only(
        SRC_ROOT / "labeler" / "mixins" / "view_build" / "canvas.py")
    assert "make_press_wrapper" not in canvas_code
    assert "make_release_wrapper" not in canvas_code

    selection_src = _src_text("labeler", "mixins", "events", "selection.py")
    assert "for key, selector in self.rect_selectors.items():" in selection_src


def test_the_examples_that_were_fixed_are_the_ones_that_exist():
    examples = SRC_ROOT.parent.parent / "examples"
    if not examples.is_dir():          # installed without the repo layout
        pytest.skip("examples/ not present beside the package")
    assert not (examples / "simple_layout_test.py").exists()
    for name in ("timeseries_only.py", "mixed_layout.py",
                 "multi_pane_magnetosphere.py", "spectrogram_multipane.py"):
        assert (examples / name).exists()


def test_no_example_carries_a_cp1252_unencodable_print():
    """The 19 characters that crash a default Windows console were all
    print() emoji and checkmarks in examples/."""
    examples = SRC_ROOT.parent.parent / "examples"
    if not examples.is_dir():
        pytest.skip("examples/ not present beside the package")
    bad = []
    for p in sorted(examples.glob("*.py")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if "print(" not in line:
                continue
            try:
                line.encode("cp1252")
            except UnicodeEncodeError:
                bad.append("%s:%d" % (p.name, i))
    assert bad == []
