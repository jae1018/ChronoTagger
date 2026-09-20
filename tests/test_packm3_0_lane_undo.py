"""Pack M3.0, item C -- A LANE-LIST WRITE IS ONE UNDO STEP.

Ctrl+L (lock), Ctrl+H (hide) and picking a hidden lane in the sidebar's
Lane list wrote the lane table BARE, outside any gesture, while EVERY
undo entry restores the WHOLE table (core/commands.py). So:

    add an interval, lock a lane, press Ctrl+Z once

took the interval away AND silently flipped the lock back, because the
undo restored the table as it stood before the interval was added. One
press, two things, and nothing said the second had happened. Measured at
base, `edit_pack/evidence/scratch/m30_draft/probe_m30.py` C.2-C.4:
"undo entries pushed 0", then "ONE undo -> region.locked False".

Each write is now one undo entry of its own, named for the act and the
lane, so the bar can say `Undo: lock lane Region (human)`.

WHAT IS DELIBERATELY UNCHANGED, and pinned here as such:

  * the REFUSAL paths write nothing and push nothing;
  * `modified` is what it was before the toggle -- a bare lane write did
    not mark the session modified and did not autosave, and this pack
    does not change that;
  * hiding still MOVES the active lane, and undoing a hide brings the
    lane back without moving the active lane back: the active lane is
    view state and is in no snapshot.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from chronotagger.core.models import Interval

DAY = "2015-01-03 "

REGION = {"id": "region", "name": "Region (human)",
          "classes": ["sw", "msh", "UNKNOWN"], "order": 0}
WAKE = {"id": "wake", "name": "Wake (umbra)", "classes": ["umbra"],
        "order": 1}
AGENT = {"id": "agent", "name": "Agent (C-MMAE)", "classes": ["0", "1"],
         "locked": True, "order": 2}

LAYOUT = {
    "nrows": 3, "ncols": 1,
    "areas": [
        {"key": "panel1", "row": 0, "col": 0, "role": "time"},
        {"key": "panel2", "row": 1, "col": 0, "role": "time"},
        {"key": "labels", "row": 2, "col": 0, "role": "labels"},
    ],
}


def ts(hhmmss):
    return pd.Timestamp(DAY + hhmmss)


def frame():
    idx = pd.date_range(DAY + "00:00:00", periods=120, freq="30s")
    return pd.DataFrame({"a": np.linspace(0.0, 1.0, len(idx))}, index=idx)


def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["a"])
    axs["panel2"].plot(df.index, df["a"])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def build(tracks, tmp_path):
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(
        df=frame(), plot_fn=plot_fn, layout_spec=dict(LAYOUT),
        window=pd.Timedelta("60min"), autosave_folder=str(tmp_path),
        tracks=[dict(t) for t in tracks])
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    return lbl


@pytest.fixture
def app(tmp_path):
    lbl = build([REGION, WAKE, AGENT], tmp_path)
    yield lbl
    lbl.root.destroy()


def row_of(app, tid):
    return [t for t in app.tracks if t.id == tid][0]


def add_G(app):
    """One ordinary edit, through the ordinary gesture machinery."""
    with app._gesture("add interval G"):
        app.intervals.append(Interval(start=ts("00:10:00"),
                                      end=ts("00:15:00"),
                                      label="sw", track="region"))


# ======================================================= 1. the lock


def test_a_lock_is_one_undo_entry_and_undo_puts_it_back(app):
    n = len(app.undo_stack)
    app._toggle_active_lane_locked()
    assert row_of(app, "region").locked is True
    assert len(app.undo_stack) == n + 1, "a lock must push exactly one entry"
    assert app.undo_stack[-1].name == "lock lane Region (human)"
    assert app.status_var.get() == "lane 'Region (human)' is now LOCKED"

    app._undo()
    assert row_of(app, "region").locked is False
    assert app.status_var.get() == "Undo: lock lane Region (human)"

    app._redo()
    assert row_of(app, "region").locked is True
    assert app.status_var.get() == "Redo: lock lane Region (human)"


def test_unlocking_names_itself_unlock(app):
    app._set_active_track("agent", announce=False, repaint=False)
    assert row_of(app, "agent").locked is True
    app._toggle_active_lane_locked()
    assert row_of(app, "agent").locked is False
    assert app.undo_stack[-1].name == "unlock lane Agent (C-MMAE)"
    app._undo()
    assert row_of(app, "agent").locked is True
    assert app.status_var.get() == "Undo: unlock lane Agent (C-MMAE)"


def test_the_sidebar_checkbox_follows_an_undo(app):
    app._toggle_active_lane_locked()
    assert app.lane_locked_var.get() is True
    app._undo()
    assert app.lane_locked_var.get() is False, \
        "the sidebar must agree with the table after an undo"


def test_a_lock_still_does_not_mark_the_session_modified(app):
    """Today's behaviour, kept on purpose -- the pack changes no save."""
    app.modified = False
    app._toggle_active_lane_locked()
    assert app.modified is False
    app.modified = True
    app._toggle_active_lane_locked()
    assert app.modified is True


def test_a_new_lock_clears_the_redo_stack_like_any_gesture(app):
    add_G(app)
    app._undo()
    assert len(app.redo_stack) == 1
    app._toggle_active_lane_locked()
    assert app.redo_stack == []


# ================================== 2. the scenario that was silent


def test_undoing_an_earlier_edit_no_longer_flips_the_lock(app):
    """THE BUG, end to end. Add G, lock, Ctrl+Z ONCE."""
    add_G(app)
    app._toggle_active_lane_locked()
    assert row_of(app, "region").locked is True
    assert len(app.intervals) == 1

    app._undo()
    assert row_of(app, "region").locked is False, "the lock was undone"
    assert len(app.intervals) == 1, "and G is STILL THERE"
    assert app.status_var.get() == "Undo: lock lane Region (human)"

    app._undo()
    assert len(app.intervals) == 0, "the second press takes G"
    assert row_of(app, "region").locked is False, \
        "and the lock does not come back as a side effect"
    assert app.status_var.get() == "Undo: add interval G"

    app._redo()
    assert len(app.intervals) == 1
    assert row_of(app, "region").locked is False
    app._redo()
    assert row_of(app, "region").locked is True
    assert len(app.intervals) == 1


# ======================================================= 3. hide / show


def test_a_hide_is_one_undo_entry_and_undo_restores_visibility(app):
    n = len(app.undo_stack)
    app._toggle_active_lane_visible()
    assert row_of(app, "region").visible is False
    assert app.active_track_id == "wake", "hiding moves the active lane"
    assert len(app.undo_stack) == n + 1
    assert app.undo_stack[-1].name == "hide lane Region (human)"

    app._undo()
    assert row_of(app, "region").visible is True
    assert app.status_var.get() == "Undo: hide lane Region (human)"
    assert app.active_track_id == "wake", \
        "the active lane is view state and does NOT come back"

    app._redo()
    assert row_of(app, "region").visible is False


def test_showing_the_active_lane_again_is_one_undo_entry(app):
    app._toggle_active_lane_visible()          # region hidden, wake active
    app._set_active_track("region", announce=False, repaint=False)
    n = len(app.undo_stack)
    app._toggle_active_lane_visible()
    assert row_of(app, "region").visible is True
    assert len(app.undo_stack) == n + 1
    assert app.undo_stack[-1].name == "show lane Region (human)"
    app._undo()
    assert row_of(app, "region").visible is False


def test_the_refusal_to_hide_the_last_visible_lane_pushes_nothing(tmp_path):
    lbl = build([REGION], tmp_path)
    try:
        n = len(lbl.undo_stack)
        lbl._toggle_active_lane_visible()
        assert lbl.tracks[0].visible is True
        assert len(lbl.undo_stack) == n, "a refusal is not an edit"
        assert "is refused" in lbl.status_var.get()
    finally:
        lbl.root.destroy()


# ============================== 4. un-hiding from the sidebar's list


def test_picking_a_hidden_lane_unhides_it_in_one_undo_step(app):
    app._toggle_active_lane_visible()          # region hidden, wake active
    n = len(app.undo_stack)
    choices = app._lane_choices()
    assert choices[0].endswith(" (hidden)")
    app.lane_combo.current(0)
    app.root.update()
    app._on_lane_combo_change()
    assert row_of(app, "region").visible is True
    assert app.active_track_id == "region"
    assert len(app.undo_stack) == n + 1
    assert app.undo_stack[-1].name == "show lane Region (human)"
    assert "is visible again" in app.status_var.get()

    app._undo()
    assert row_of(app, "region").visible is False
    # `region` is still the ACTIVE lane and it is hidden again, so the
    # repaint's standing warning about that outranks the `Undo: ...`
    # line -- which is right, and is behaviour this pack does not touch.
    assert "is HIDDEN" in app.status_var.get()


def test_picking_a_visible_lane_pushes_nothing(app):
    n = len(app.undo_stack)
    app.lane_combo.current(1)
    app.root.update()
    app._on_lane_combo_change()
    assert app.active_track_id == "wake"
    assert len(app.undo_stack) == n, "a plain lane switch is not an edit"
