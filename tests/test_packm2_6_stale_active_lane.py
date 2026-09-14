"""Pack M2.6 -- A STALE ACTIVE LANE MUST NOT FAIL OPEN.

THE DEFECT, in three gestures on the advertised ingest path.
`add_track_from_column` with a NEW id creates a LOCKED lane inside ONE
gesture (app.py). Click that lane. Press Ctrl+Z once. The row is gone --
and `_active_track_id` still names it, because the id is VIEW state and
does not ride in the gesture snapshot.

From there the guard and the writers disagreed about what "the active
lane" means. `is_locked()` looked the id up, found no row, and answered
False. `active_track_of()` fell back to the first row. Every writer
stamped `active_id_of(self)`, which was the GHOST STRING. Measured on the
feel-test driver (edit_pack/evidence/scratch/m2_orch/logs/
probe_s3_refute_reach.txt):

    Fill Gaps OK      not refused; 256 -> 257 intervals, the new one on a
                      track no table holds, painted nowhere, counted
                      nowhere, and persisted by the next autosave
    Manage Labels OK  not refused; rewrote the FIRST row's vocabulary,
                      said "Labels updated", wiped the undo stack 2 -> 0
    Clear Range       not refused; deleted
    the locked lane   when it is the first row, its vocabulary is
                      rewritten with no refusal at all
    recovery          the autosave that holds the ghost interval RAISES
                      ValueError in _apply_recovered_autosave, uncaught
                      by run() -- a traceback instead of a window

THE FIX IS TWO HALVES AND BOTH ARE PINNED HERE.
`core.tracks.resolve_active_row` is the ONE rule every reader asks: the
row the id names, or -- when it names no row -- the FIRST VISIBLE lane,
which is exactly the lane `lane_layout` already paints.
`_reconcile_active_track` then repairs the MODEL after every undo and
redo, refreshes the lane controls and says so on the status bar.

This module uses a plain two-lane labeler with a class column to ingest
from. It does NOT need the C05 case frame and runs everywhere.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from chronotagger.core.lanes import is_locked, lane_layout, refuse_if_locked
from chronotagger.core.models import Interval
from chronotagger.core.tracks import (active_id_of, active_track_of,
                                      intervals_on, resolve_active_row,
                                      stray_tracks, visible_rows)

DAY = "2015-01-03 "

HUMAN = {"id": "human", "name": "Human", "classes": ["sw", "msh", "UNKNOWN"],
         "class_colors": {"sw": "#4e79a7", "msh": "#f28e2b",
                          "UNKNOWN": "#7f7f7f"}, "order": 0}
MODEL = {"id": "model", "name": "Model", "classes": ["0", "1"],
         "class_colors": {"0": "#111111", "1": "#222222"}, "order": 1}

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
def app(tmp_path):
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(
        df=frame(), plot_fn=plot_fn, layout_spec=dict(LAYOUT),
        window=pd.Timedelta("60min"), autosave_folder=str(tmp_path),
        tracks=[dict(HUMAN), dict(MODEL)])
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    yield lbl
    lbl.root.destroy()


def three_gestures(app):
    """Ingest a NEW locked lane, activate it, undo once. The reachable path."""
    row = app.add_track_from_column("rule", "agent2", name="Agent2 (ingested)",
                                    locked=True)
    assert row.id == "agent2" and row.locked is True
    assert app.track_by_id("agent2") is not None
    app._set_active_track("agent2", announce=False, repaint=False)
    assert active_id_of(app) == "agent2"
    app._undo()
    return row


# ==================================================== 1. the three gestures


def test_one_undo_after_the_ingest_leaves_a_REAL_active_lane(app):
    three_gestures(app)
    assert app.track_by_id("agent2") is None, "the undo removed the row"
    assert app._active_track_id == "human", \
        "the MODEL was repaired, not just the readers"
    assert active_id_of(app) == "human"
    assert active_track_of(app).id == "human"
    assert app.lane_var.get() == "Human", "the Lane box names the real lane"


def test_the_status_bar_says_which_lane_went_and_where_you_are_now(app):
    three_gestures(app)
    said = app.status_var.get()
    assert said == ("lane 'agent2' no longer exists -- active lane is now "
                    "'Human'"), said


def test_fill_gaps_after_the_undo_lands_on_the_lane_you_can_see(app):
    three_gestures(app)
    gaps = app._find_gaps_in_current_range()
    assert gaps, "the human lane holds nothing, so the window is one gap"
    app._assign_gaps_to_label(gaps[:1], "sw")
    assert stray_tracks(app.intervals, app.tracks) == [], \
        "no interval may name a track the table does not hold"
    assert [iv.track for iv in app.intervals] == ["human"]
    assert len(intervals_on(app.intervals, "human")) == 1


def test_manage_labels_after_the_undo_edits_the_lane_you_can_see(app):
    from chronotagger.labeler.dialogs.label_manager import LabelManagerResult
    three_gestures(app)
    app._assign_gaps_to_label(
        [(ts("00:10:00"), ts("00:20:00"))], "sw")
    res = LabelManagerResult(
        classes=["solar", "msh", "UNKNOWN"],
        class_colors={"solar": "#4e79a7", "msh": "#f28e2b",
                      "UNKNOWN": "#7f7f7f"},
        rename_map={"sw": "solar"}, reassign_map={})
    app._apply_label_manager_result(res)
    assert app.track_by_id("human").classes == ["solar", "msh", "UNKNOWN"]
    assert [iv.label for iv in intervals_on(app.intervals, "human")] == \
        ["solar"], "its OWN intervals followed the rename"
    assert app.track_by_id("model").classes == ["0", "1"], \
        "the other lane's vocabulary is untouched"


# ================================================= 2. a stale id, by hand


def test_a_forced_stale_id_resolves_to_the_first_visible_lane(app):
    app._active_track_id = "ghost"
    assert app.track_by_id("ghost") is None
    assert active_id_of(app) == "human"
    assert active_track_of(app).id == "human"
    assert resolve_active_row(app).id == "human"
    assert lane_layout(app)["active_id"] == "human"


def test_the_lock_judges_the_resolved_row_not_the_ghost(app):
    app._active_track_id = "ghost"
    assert is_locked(app) is False
    assert refuse_if_locked(app, what="add") is False
    app.track_by_id("human").locked = True
    assert is_locked(app) is True, \
        "the lane the user can see is LOCKED, so the guard must refuse"
    assert refuse_if_locked(app, what="add") is True
    assert "Human" in app.status_var.get()
    assert "add refused" in app.status_var.get()


def test_a_stale_id_resolves_past_a_HIDDEN_first_lane(app):
    app.track_by_id("human").visible = False
    app._active_track_id = "ghost"
    assert [t.id for t in visible_rows(app.tracks)] == ["model"]
    assert active_id_of(app) == "model", \
        "the fallback is the first VISIBLE lane -- what lane_layout paints"
    assert lane_layout(app)["active_id"] == "model"


def test_a_HIDDEN_active_lane_is_not_stale_and_is_not_resolved_away(app):
    app._set_active_track("model", announce=False, repaint=False)
    app.track_by_id("model").visible = False
    assert active_id_of(app) == "model", "the row exists; it is not painted"
    lay = lane_layout(app)
    assert lay["active_id"] == "model" and lay["active_row"] is None


def test_every_writer_stamps_a_real_row_under_a_stale_id(app):
    app._active_track_id = "ghost"
    app.current_selection = (ts("00:05:00"), ts("00:08:00"))
    app.current_class_var.set("sw")
    app._add_interval()
    assert stray_tracks(app.intervals, app.tracks) == []
    assert [iv.track for iv in app.intervals] == ["human"]


def test_the_reconcile_is_a_no_op_when_the_active_id_is_fine(app):
    app._set_active_track("model", announce=False, repaint=False)
    app.status_var.set("untouched")
    assert app._reconcile_active_track() is False
    assert app._active_track_id == "model"
    assert app.status_var.get() == "untouched"


# ================================================= 3. run()'s recover branch


def _payload_with_a_stray_track():
    return {
        "version": 2,
        "tracks": [dict(HUMAN), dict(MODEL)],
        "active_track": "human",
        "intervals": [
            Interval(ts("00:05:00"), ts("00:08:00"), "sw", None,
                     track="agent2").to_dict(),
        ],
    }


def test_a_recovery_that_refuses_gives_ONE_message_and_a_fresh_session(
        app, monkeypatch):
    """The traceback at launch, closed.

    The payload is exactly the shape the stale-id fail-open used to
    write: an interval on a track the saved table does not hold.
    `_apply_recovered_autosave` raises ValueError on it -- pinned here --
    and run() must turn that into one showerror and a fresh session.
    """
    import tkinter.messagebox as mb

    payload = _payload_with_a_stray_track()
    with pytest.raises(ValueError):
        app._apply_recovered_autosave(dict(payload))

    seen = []
    monkeypatch.setattr(mb, "showerror",
                        lambda *a, **k: seen.append(a[:2]) or None)
    monkeypatch.setattr(mb, "showinfo", lambda *a, **k: seen.append("info"))
    monkeypatch.setattr(mb, "showwarning", lambda *a, **k: seen.append("warn"))
    monkeypatch.setattr(app, "_check_autosave", lambda: dict(payload))
    monkeypatch.setattr(app, "_show_recovery_dialog", lambda d: "recover")
    monkeypatch.setattr(app.root, "mainloop", lambda *a, **k: None)

    app.run()

    assert len(seen) == 1, seen
    assert seen[0][0] == "Recovery Failed"
    assert "agent2" in seen[0][1]
    assert app.status_var.get() == ("Recovery failed -- starting fresh "
                                   "session (autosave not loaded)")
    assert app.intervals == [], "a fresh session, exactly as start-fresh"
    assert bool(app.root.winfo_exists()), "the window survived"


def test_a_recovery_that_succeeds_is_unchanged(app, monkeypatch):
    """The control: the same branch, with a payload that validates."""
    import tkinter.messagebox as mb
    payload = _payload_with_a_stray_track()
    payload["intervals"] = [
        Interval(ts("00:05:00"), ts("00:08:00"), "sw", None,
                 track="human").to_dict()]
    seen = []
    monkeypatch.setattr(mb, "showerror",
                        lambda *a, **k: seen.append(a[:2]) or None)
    monkeypatch.setattr(app, "_check_autosave", lambda: dict(payload))
    monkeypatch.setattr(app, "_show_recovery_dialog", lambda d: "recover")
    monkeypatch.setattr(app.root, "mainloop", lambda *a, **k: None)
    app.run()
    assert seen == []
    assert len(app.intervals) == 1
    assert app.status_var.get() == "Recovered 1 intervals from autosave"
