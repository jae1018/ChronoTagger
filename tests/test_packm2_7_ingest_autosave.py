"""Pack M2.7 -- THE LAUNCH-TIME INGEST MUST NOT OVERWRITE THE RECOVERABLE
AUTOSAVE.

THE DEFECT, measured end to end (probe_s4_recovery_clobber Q1-Q3).
`add_track_from_column` ends in an unconditional `_save_autosave()`, and
the feel-test driver calls it BEFORE `app.run()` -- and run() is where
the recovery question is asked. So every launch wrote a model-only
autosave over the previous session's file before the user was offered it:

    session one   259 intervals on disk (256 ingested + 3 hand-labelled)
    relaunch one  the ingest rewrites the main file to 256 agent-only and
                  the 259-interval file is demoted to .bak -- which
                  _check_autosave reads ONLY when main is unreadable, so
                  the intact backup is invisible to the dialog
    relaunch two  main and .bak are both 256 agent-only: the hand labels
                  are on NO file in the folder

THE FIX IS ONE FLAG. `_recovery_resolved` is False at construction, and
`run()` sets it True the moment the recovery question is settled -- when
`_check_autosave` returned None (nothing to offer, or a future-version
file already refused) and after every dialog branch that keeps the
window. The ingest's DISK WRITE waits for it. Nothing else changes: the
in-memory state, the gesture and the undo are untouched, because the
write sits OUTSIDE the ingest's `with self._gesture(...)` block.

AND WHAT WAS DELIBERATELY NOT DONE, pinned here as a control: the guard
is NOT inside `_save_autosave`. A process-wide version of it silently
disables autosave for every head-less consumer that never calls run() --
measured, three gestures, three writes skipped, no word to anyone.

This module uses a plain two-column labeler over a small synthetic frame
and a tmp_path case directory. It does NOT need the C05 case frame.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from chronotagger.core.models import Interval

DAY = "2015-01-03 "

HUMAN = {"id": "human", "name": "Human", "classes": ["sw", "msh", "UNKNOWN"],
         "class_colors": {"sw": "#4e79a7", "msh": "#f28e2b",
                          "UNKNOWN": "#7f7f7f"}, "order": 0}

LAYOUT = {
    "nrows": 2, "ncols": 1,
    "areas": [
        {"key": "panel1", "row": 0, "col": 0, "role": "time"},
        {"key": "labels", "row": 1, "col": 0, "role": "labels"},
    ],
}


def ts(hhmmss):
    return pd.Timestamp(DAY + hhmmss)


def frame():
    idx = pd.date_range(DAY + "00:00:00", periods=120, freq="30s")
    return pd.DataFrame({"a": np.linspace(0.0, 1.0, len(idx)),
                         "rule": ["0"] * 60 + ["1"] * 60}, index=idx)


def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["a"])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def launch(tmp_path):
    """Build a labeler over ONE case directory, the way a relaunch does."""
    built = []

    def _launch():
        from chronotagger.labeler import TimeIntervalLabeler
        lbl = TimeIntervalLabeler(
            df=frame(), plot_fn=plot_fn, layout_spec=dict(LAYOUT),
            window=pd.Timedelta("60min"), autosave_folder=str(tmp_path),
            tracks=[dict(HUMAN)])
        lbl._build_gui()
        lbl._update_plot()
        lbl.root.withdraw()
        built.append(lbl)
        return lbl

    yield _launch
    for lbl in built:
        try:
            lbl.root.destroy()
        except Exception:
            pass


@pytest.fixture
def app(launch):
    return launch()


def three_hand_labels(lbl):
    lbl.intervals[:] = [
        Interval(ts("00:05:00"), ts("00:08:00"), "sw", None, track="human"),
        Interval(ts("00:20:00"), ts("00:25:00"), "msh", None, track="human"),
        Interval(ts("00:40:00"), ts("00:45:00"), "sw", None, track="human")]


def payload_on_disk(lbl):
    return json.loads(lbl.autosave_file.read_text(encoding="utf-8"))


# ============================================ 1. a pre-run ingest is quiet


def test_a_pre_run_ingest_writes_no_autosave(app):
    assert app._recovery_resolved is False, "nothing has been asked yet"
    app.add_track_from_column("rule", "agent", name="Agent")
    assert not app.autosave_file.exists()


def test_the_ingest_itself_is_completely_unchanged(app):
    """Only the DISK WRITE waits. The lane, the intervals, the status line
    and the single undo entry are exactly what they always were."""
    app.add_track_from_column("rule", "agent", name="Agent")
    assert app.track_by_id("agent") is not None
    assert app.track_by_id("agent").locked is True
    assert len(app.intervals) == 2
    assert {iv.track for iv in app.intervals} == {"agent"}
    assert app.status_var.get().startswith("Imported 2 interval(s)")
    assert len(app.undo_stack) == 1
    app._undo()
    assert app.track_by_id("agent") is None
    assert app.intervals == []


def test_a_pre_run_ingest_leaves_an_existing_autosave_byte_identical(app):
    three_hand_labels(app)
    app._save_autosave()
    before = app.autosave_file.read_bytes()
    app.add_track_from_column("rule", "agent", name="Agent")
    assert app.autosave_file.read_bytes() == before, \
        "the file the recovery dialog is about to offer is untouched"


# ==================================== 2. the three-launch reproduction


def test_the_three_launch_reproduction_keeps_offering_session_one(launch):
    # SESSION ONE: a window that was up, three hand-labelled intervals,
    # closed with "No" (which leaves the autosave exactly where it is).
    one = launch()
    one._recovery_resolved = True
    three_hand_labels(one)
    one._save_autosave()
    assert len(payload_on_disk(one)["intervals"]) == 3
    session_one = one.autosave_file.read_bytes()
    one.root.destroy()

    # RELAUNCH ONE: the driver ingests BEFORE run() asks about recovery.
    two = launch()
    two.add_track_from_column("rule", "agent", name="Agent")
    assert two.autosave_file.read_bytes() == session_one
    offered = two._check_autosave()
    assert offered is not None
    assert len(offered["intervals"]) == 3
    assert {iv.track for iv in offered["intervals"]} == {"human"}
    two.root.destroy()

    # RELAUNCH TWO: the file used to be gone by now -- main AND .bak both
    # rewritten to the ingest's own lane.
    three = launch()
    three.add_track_from_column("rule", "agent", name="Agent")
    offered = three._check_autosave()
    assert offered is not None
    assert len(offered["intervals"]) == 3
    assert three.autosave_file.read_bytes() == session_one
    three.root.destroy()


# ============================== 3. run() is what settles the question


@pytest.mark.parametrize("choice", ["recover", "start_fresh", "save_backup"])
def test_every_branch_that_keeps_the_window_settles_it(launch, monkeypatch,
                                                       choice):
    one = launch()
    one._recovery_resolved = True
    three_hand_labels(one)
    one._save_autosave()
    one.root.destroy()

    app = launch()
    app.add_track_from_column("rule", "agent", name="Agent")
    assert app._recovery_resolved is False
    monkeypatch.setattr(app.root, "mainloop", lambda *a, **k: None)
    monkeypatch.setattr(app, "_show_recovery_dialog", lambda d: choice)
    app.run()

    assert app._recovery_resolved is True
    app.autosave_file.unlink()
    app.add_track_from_column("rule", "agent2", name="Agent 2")
    assert app.autosave_file.exists(), "the ingest writes again"


def test_nothing_to_offer_is_also_an_answer(launch, monkeypatch):
    """No autosave, another dataset's autosave, or a refused
    future-version file: _check_autosave answers None and the user has
    been told everything he is going to be told."""
    app = launch()
    app.add_track_from_column("rule", "agent", name="Agent")
    assert not app.autosave_file.exists()
    asked = []
    monkeypatch.setattr(app.root, "mainloop", lambda *a, **k: None)
    monkeypatch.setattr(app, "_show_recovery_dialog",
                        lambda d: asked.append(d) or "start_fresh")
    app.run()
    assert asked == [], "there was nothing to offer"
    assert app._recovery_resolved is True
    app.add_track_from_column("rule", "agent2", name="Agent 2")
    assert app.autosave_file.exists()


def test_a_cancelled_launch_never_settles_it(launch, monkeypatch):
    """'Exit ChronoTagger' destroys the root and returns before the flag
    is set -- correct, because there is no session left to autosave."""
    one = launch()
    one._recovery_resolved = True
    three_hand_labels(one)
    one._save_autosave()
    kept = one.autosave_file.read_bytes()
    one.root.destroy()

    app = launch()
    app.add_track_from_column("rule", "agent", name="Agent")
    monkeypatch.setattr(app, "_show_recovery_dialog", lambda d: "cancel")
    app.run()
    assert app._recovery_resolved is False
    assert app.autosave_file.read_bytes() == kept


def test_an_ingest_then_a_recover_duplicates_nothing(launch, monkeypatch):
    one = launch()
    one._recovery_resolved = True
    three_hand_labels(one)
    one._save_autosave()
    one.root.destroy()

    app = launch()
    app.add_track_from_column("rule", "agent", name="Agent")
    assert len(app.intervals) == 2
    monkeypatch.setattr(app.root, "mainloop", lambda *a, **k: None)
    monkeypatch.setattr(app, "_show_recovery_dialog", lambda d: "recover")
    app.run()

    assert app.status_var.get() == "Recovered 3 intervals from autosave"
    assert len(app.intervals) == 3
    keys = {(iv.start, iv.end, iv.label, iv.track) for iv in app.intervals}
    assert len(keys) == 3, "no interval arrived twice"
    assert {iv.track for iv in app.intervals} == {"human"}


# ================================== 4. the controls: what is NOT gated


def test_the_gesture_handlers_still_write_with_the_flag_False(app):
    """The ten OTHER _save_autosave call sites are GUI gesture handlers,
    reachable only once the window is up. None of them is gated, so a
    head-less consumer that drives them keeps its autosave."""
    assert app._recovery_resolved is False
    app.current_class_var.set("sw")
    app.current_selection = (ts("00:10:00"), ts("00:15:00"))
    app._add_interval()
    assert app.status_var.get() == "Added 1 sw interval(s)"
    assert app.autosave_file.exists()
    assert len(payload_on_disk(app)["intervals"]) == 1


def test_save_autosave_itself_is_NOT_guarded():
    """THE RULED NO. A process-wide guard inside _save_autosave silently
    disables autosave for every head-less consumer that never calls
    run() -- three gestures, three writes skipped, no dialog, no status
    line and no log line. The guard is on the INGEST and nowhere else."""
    import inspect
    from chronotagger.labeler.mixins.io_export import IOExportMixin
    src = inspect.getsource(IOExportMixin._save_autosave)
    assert "_recovery_resolved" not in src


def test_check_autosave_still_reads_the_bak_only_when_main_is_unreadable():
    """The other RULED NO: the .bak policy does not change here. It
    guards a torn write, not a clobber, and that is recorded in the pack's
    scope fence rather than fixed."""
    import inspect
    from chronotagger.labeler.mixins.io_export import IOExportMixin
    src = inspect.getsource(IOExportMixin._check_autosave)
    assert "candidates = [main]" in src
    assert "if main_existed:" in src
