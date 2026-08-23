"""
Pack 4 (error visibility) regression tests.

Recipe (evidence pack4_g2 section 5.4): every failure test asserts THREE
legs -- the call survives, the failure is logged WITH exc_info (a message-
only assertion reproduces the old bug at a new severity), and the very
next operation still works. File-log tests add the fourth leg: utf-8
encoding round-trips a non-ASCII character.

Capture uses a local handler fixture, not caplog, so the tests keep
working regardless of propagation settings (measured M16-M18).
"""

import logging

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


class _Capture(logging.Handler):
    """Propagation-independent record capture (pack4_g2 M17)."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def logcap():
    lg = logging.getLogger("chronotagger")
    h = _Capture()
    prev = lg.level
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    try:
        yield h
    finally:
        lg.removeHandler(h)
        lg.setLevel(prev)


def _errors(logcap):
    return [r for r in logcap.records if r.levelno >= logging.ERROR]


def _warnings(logcap):
    return [r for r in logcap.records if r.levelno == logging.WARNING]


# ------------------------------------------------------------ plot_fn path

def test_plot_fn_failure_logged_and_survives(labeler, logcap):
    def boom(axs, df, t0, t1):
        raise KeyError("Bz")
    labeler.active_pane.plot_fn = boom

    # LEG 1 -- survives
    labeler._update_plot()

    # LEG 2 -- logged with a real traceback
    errs = _errors(logcap)
    assert errs, "plot_fn failure produced no log record"
    assert errs[0].exc_info is not None, "logged without exc_info"

    # LEG 3 -- the next redraw works
    labeler.active_pane.plot_fn = lambda axs, df, t0, t1: None
    labeler._update_plot()
    assert labeler.canvas is not None


def test_plot_fn_failure_keeps_window_cache_fresh(labeler, logcap):
    """R7: a failed render must NOT leave the previous window cached --
    the stale cache committed intervals displaced by 1 hour (T1)."""
    step = pd.Timedelta("15min")
    labeler.t0 = labeler.t0 + step
    labeler.t1 = labeler.t1 + step

    def boom(axs, df, t0, t1):
        raise KeyError("Bz")
    labeler.active_pane.plot_fn = boom
    labeler._update_plot()

    expected = labeler.df.loc[labeler.t0:labeler.t1].index
    assert labeler._last_windowed_index.equals(expected)


def test_plot_error_text_drawn_once_with_type_and_pointer(labeler, logcap):
    """R6: one panel carries the three-line summary (type + user frame +
    log pointer); the panels that rendered fine stay clean."""
    def boom(axs, df, t0, t1):
        raise KeyError("Bz")
    labeler.active_pane.plot_fn = boom
    labeler._update_plot()

    carrying = []
    for ax in labeler.user_axes.values():
        texts = [t.get_text() for t in ax.texts
                 if "Plot error" in t.get_text()]
        if texts:
            carrying.append(texts)
    assert len(carrying) == 1, "error text must be drawn on exactly one axis"
    msg = carrying[0][0]
    assert "KeyError" in msg
    assert "chronotagger.log" in msg


def test_empty_window_notice_instead_of_plot_error(labeler, logcap):
    """Zoom finer than the cadence: plot_fn is NOT called and the panel
    says why (was: user code explodes on an empty frame -> 'Plot error')."""
    idx = labeler.df.index
    labeler.t0 = idx[10] + pd.Timedelta(seconds=1)
    labeler.t1 = idx[10] + pd.Timedelta(seconds=5)

    called = []
    labeler.active_pane.plot_fn = (
        lambda axs, df, t0, t1: called.append(len(df)))
    labeler._update_plot()

    assert called == [], "plot_fn must not be called for an empty window"
    all_text = " ".join(t.get_text() for ax in labeler.user_axes.values()
                        for t in ax.texts)
    assert "No samples" in all_text
    assert len(labeler._last_windowed_index) == 0


# ------------------------------------------------------------ backbone

def test_file_logging_utf8_idempotent_and_roundtrips(tmp_path):
    from chronotagger._logging import configure_file_logging, _SENTINEL

    p1 = configure_file_logging(tmp_path)
    p2 = configure_file_logging(tmp_path)   # second labeler, same folder
    assert p1 == p2 == tmp_path / "chronotagger.log"

    lg = logging.getLogger("chronotagger")
    fh = [h for h in lg.handlers if getattr(h, _SENTINEL, False)]
    assert len(fh) == 1, "idempotence: exactly ONE file handler"
    assert fh[0].stream is None or fh[0].stream.encoding == "utf-8"

    lg.info("Window: 00:00:00 \u2192 00:30:00")
    for h in fh:
        h.flush()
    content = (tmp_path / "chronotagger.log").read_text(encoding="utf-8")
    assert "\u2192" in content, "non-ASCII record must round-trip"


def test_log_file_lives_beside_the_autosave(labeler, tmp_path):
    """R3: the labeler fixture passes autosave_folder=tmp_path; the log
    must be created there (session banner writes on attach)."""
    assert labeler._log_path is not None
    assert labeler._log_path.parent == labeler.autosave_folder
    assert labeler._log_path.exists()


def test_debug_env_raises_level(tmp_path, monkeypatch):
    from chronotagger._logging import configure_file_logging
    lg = logging.getLogger("chronotagger")

    monkeypatch.delenv("CHRONOTAGGER_DEBUG", raising=False)
    configure_file_logging(tmp_path / "a")
    assert lg.level == logging.INFO

    monkeypatch.setenv("CHRONOTAGGER_DEBUG", "1")
    configure_file_logging(tmp_path / "b")
    assert lg.level == logging.DEBUG


# ------------------------------------------------------------ autosave

def test_autosave_failure_logged_status_and_dialog_once(
        labeler, logcap, _stub_messagebox, monkeypatch):
    """R13: ERROR + statusbar every time; dialog on the FIRST failure
    only. Was: statusbar only, and only if the statusbar existed."""
    import chronotagger.labeler.mixins.io_export as io_mod

    def broken_write(*a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr(io_mod, "atomic_write_json", broken_write)

    labeler._save_autosave()
    labeler._save_autosave()

    errs = _errors(logcap)
    assert len(errs) == 2
    assert all(r.exc_info is not None for r in errs)
    assert "Autosave failed" in labeler.status_var.get()
    assert _stub_messagebox.count("showerror") == 1, \
        "dialog exactly once per session"


# ------------------------------------------------------------ file loader

def _detect(df):
    from chronotagger.quickstart.file_loader import FileLoaderDialog
    return FileLoaderDialog._auto_detect_time_column(None, df)


def test_autodetect_never_coerces_numeric_columns(logcap):
    """R8 (amended after verification): numeric columns are NEVER read
    as timestamps -- monotonicity is no gate (the gather's own Bx_nT was
    a LINEAR RAMP; ramps, counters and L-shells are monotonic)."""
    n = 50
    ramp = np.linspace(-5.0, 5.0, n)          # monotonic physics column
    epoch = np.arange(n, dtype=float) * 1e9   # monotonic numeric column

    ordered = pd.DataFrame({"epoch_ns": epoch, "Bx_nT": ramp})
    reordered = pd.DataFrame({"Bx_nT": ramp, "epoch_ns": epoch})

    assert _detect(ordered) is None
    assert _detect(reordered) is None, "column order chose the time axis"
    assert any("never coerced" in r.getMessage()
               for r in _warnings(logcap))


def test_autodetect_accepts_string_datetime_column():
    """A real (string) datetime column is still found, even UNSORTED:
    the downstream validator owns the 'must be sorted' message. The v1
    monotonic gate refused it and handed the win to a numeric column
    (verifier B4)."""
    df = pd.DataFrame({
        "Bx_nT": np.linspace(-5.0, 5.0, 4),
        "when": ["2015-01-02", "2015-01-01", "2015-01-04", "2015-01-03"],
    })
    assert _detect(df) == "when"


def test_autodetect_keyboardinterrupt_propagates(monkeypatch):
    """R10: the bare excepts swallowed Ctrl+C (three per pass, T4)."""
    import chronotagger.quickstart.file_loader as fl_mod

    def interrupted(*a, **kw):
        raise KeyboardInterrupt
    monkeypatch.setattr(fl_mod.pd, "to_datetime", interrupted)

    df = pd.DataFrame({"time": ["2015-01-01", "2015-01-02"]})
    with pytest.raises(KeyboardInterrupt):
        _detect(df)


# ------------------------------------------------------------ downgrades

def test_box_filter_downgrade_logged(labeler, logcap, monkeypatch):
    """A7: the good algorithm failing must RECORD its silent switch to
    the phantom-prone artist scan."""
    monkeypatch.setattr(
        labeler, "_find_contiguous_runs",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=True)
    labeler.axes_meta.setdefault("panel1", {})
    labeler.axes_meta["panel1"]["x_col"] = "BX"
    labeler.axes_meta["panel1"]["y_col"] = "BY"

    result = labeler._try_dataframe_box_filter("panel1", -100, 100,
                                               -100, 100)
    assert result is None
    assert any("artist scan" in r.getMessage() for r in _warnings(logcap))


def test_blit_degrade_logged_once(logcap, monkeypatch):
    from chronotagger.labeler.utils import fastdraw

    # The once-flag is MODULE state (per session, across panes); reset it
    # so this test is order-independent within the suite.
    monkeypatch.setattr(fastdraw, "_degrade_logged", False)

    class _Canvas:
        def __init__(self):
            self.idle = 0

        def draw_idle(self):
            self.idle += 1

    helper = object.__new__(fastdraw.BlitHelper)
    helper.canvas = _Canvas()
    helper.axes = []
    helper._bg = {}

    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1], [0, 1])

    helper.draw([line])   # no cached background -> fallback
    helper.draw([line])   # second degrade: no second record

    degrade = [r for r in _warnings(logcap)
               if "blit" in r.getMessage()]
    assert len(degrade) == 1
    assert helper.canvas.idle == 2


# ------------------------------------------------------------ exports

def test_export_prints_migrated_to_log(labeler, logcap, capsys, tmp_path):
    """R11: public-API success chatter moves off stdout."""
    from chronotagger.core.models import Interval
    idx = labeler.df.index
    labeler.intervals.append(Interval(idx[5], idx[15], "PS"))

    out_path = str(tmp_path / "iv.csv")
    labeler.export_intervals(out_path, fmt="csv")

    assert capsys.readouterr().out == ""
    assert any("Exported intervals" in r.getMessage()
               for r in logcap.records if r.levelno == logging.INFO)


# ------------------------------------------------------------ launcher

def test_launcher_honest_on_runtime_import_error(monkeypatch, capsys):
    """R10: an ImportError from the RUNNING app must not be reported as
    'wizard not available' plus an advert."""
    import matplotlib as mpl
    monkeypatch.setattr(mpl, "use", lambda *a, **kw: None)
    import importlib
    launcher = importlib.import_module("chronotagger.launcher")
    import chronotagger.quickstart.wizard as wizard

    def run_raises():
        raise ImportError("No module named 'pyarrow'")
    monkeypatch.setattr(wizard, "run", run_raises)

    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "Error launching ChronoTagger" in out
    assert "wizard not available" not in out


# --------------------------------------------------- fold pins (recheck)

def test_plot_error_text_names_the_users_frame_not_pandas(labeler, logcap):
    """B3 pin: the middle line must name the frame the USER wrote. A
    reversed traceback walk returns the DEEPEST foreign frame instead --
    `base.py, line 3648, in get_loc` (verifier B3, reproduced)."""
    def boom(axs, df, t0, t1):
        axs["panel1"].plot(df.index, df["Bz"])   # KeyError, deep in pandas
    labeler.active_pane.plot_fn = boom
    labeler._update_plot()

    msg = [t.get_text() for ax in labeler.user_axes.values()
           for t in ax.texts if "Plot error" in t.get_text()][0]
    middle = msg.split("\n")[1]
    assert "in boom" in middle, middle
    assert "get_loc" not in middle and "base.py" not in middle, middle


def test_autosave_failure_dialog_needs_a_real_tk_root(
        labeler, logcap, _stub_messagebox, monkeypatch):
    """B1 pin: the GUI-free hosts in this suite set root=object(); an
    unstubbed messagebox from one of them deadlocked the entire run
    (test_persistence_safety::test_crash_mid_autosave_preserves_previous_
    and_surfaces). The gate is isinstance(root, tk.Misc) -- a
    `root is not None` gate passes every existing test and still hangs."""
    import chronotagger.labeler.mixins.io_export as io_mod

    def broken_write(*a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr(io_mod, "atomic_write_json", broken_write)

    # Explicit save/restore, NOT monkeypatch: the monkeypatch fixture is
    # created before `labeler` (conftest's autouse _isolate_cwd needs it),
    # so its undo runs AFTER the labeler teardown -- root.destroy() would
    # be called on the stand-in.
    real_root = labeler.root
    labeler.root = object()          # MockPersistHost's exact shape
    try:
        labeler._save_autosave()
    finally:
        labeler.root = real_root

    assert _stub_messagebox == [], \
        "a non-Tk root must never reach messagebox (it deadlocks the suite)"
    assert _errors(logcap), "headless: the log line is the record"


def test_file_logging_guard_is_live_on_an_unopenable_path(tmp_path):
    """B2 pin: the handler must open EAGERLY, inside the guard. With
    delay=True the open happens at the session banner -- outside the
    guard -- and labeler construction died with PermissionError."""
    from chronotagger._logging import configure_file_logging, _SENTINEL

    folder = tmp_path / "writable"
    folder.mkdir()
    (folder / "chronotagger.log").mkdir()   # the log PATH cannot be opened

    assert configure_file_logging(folder) is None    # returns, never raises
    lg = logging.getLogger("chronotagger")
    assert not [h for h in lg.handlers if getattr(h, _SENTINEL, False)], \
        "a failed attach must leave NO sentinel handler behind"
