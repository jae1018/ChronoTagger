"""Pack M2.7 -- A GESTURE'S OWN STATUS SENTENCE SURVIVES THE REPAINT IT
TRIGGERS.

THE DEFECT is an ORDERING one. Four gestures write their own sentence and
then immediately ask for the repaint that refills the interval list, and
the refill announced "the selected interval sits on lane 'X', which this
list is not showing" straight over the top of it. Measured, all four
(probe_s4_undo_status Q1-Q2):

    Undo / Redo    the snapshot restore swaps the selected interval for a
                   value-equal COPY, and the announce key started with
                   id(), so a re-point looked like a brand new selection
    drag-resize    ResizeIntervalCommand builds a NEW object at NEW
                   bounds, and the redraw is coalesced onto idle, so the
                   refill lands a moment after "Resized: ..."
    lane switch    the active lane is part of the fact, so the switch
                   really does make it true -- and then overwrites
                   "Active lane: ..." with it

Add, Delete, Re-label, Clear range, Fill Gaps and the rule commit were
measured and keep their own line; they are not touched.

THE FIX IS TWO HALVES.
  * `stats.hidden_selection_key` is a VALUE key, so a re-point is not a
    change. That alone closes Undo and Redo.
  * A gesture that owns the bar CLAIMS the sentence before it repaints
    (`_claim_hidden_selection_line`). For the lane switch the claim hands
    the clause back and the switch says BOTH things in one sentence:
        Active lane: Model -- the selected interval stays on 'Human',
        which this list is not showing

EVERY PACK M2.6 PIN STAYS GREEN. This module adds what those pins never
had: a fixture that really does hold a selection on a lane the list is
not showing while the gesture runs.
"""

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
MODEL = {"id": "model", "name": "Model", "classes": ["0", "1"],
         "class_colors": {"0": "#111111", "1": "#222222"}, "order": 1}

LAYOUT = {
    "nrows": 2, "ncols": 1,
    "areas": [
        {"key": "panel1", "row": 0, "col": 0, "role": "time"},
        {"key": "labels", "row": 1, "col": 0, "role": "labels"},
    ],
}

HIDDEN_LINE = ("the selected interval sits on lane 'Model', which this "
               "list is not showing")

STAYS_LINE = ("Active lane: Model -- the selected interval stays on "
              "'Human', which this list is not showing")


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


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    import tkinter.messagebox as mb
    calls = []
    for kind in ("showinfo", "showwarning", "showerror", "askyesno",
                 "askyesnocancel", "askokcancel"):
        monkeypatch.setattr(
            mb, kind, lambda *a, _k=kind, **kw: calls.append(_k) or True)
    return calls


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


@pytest.fixture
def hidden(app):
    """THE FIXTURE THE M2.6 PINS NEVER HAD: the active lane is Human, the
    list is filtered to the active lane, and the SELECTION sits on Model
    -- so the sentence is live before the gesture even starts."""
    app.intervals[:] = [
        Interval(ts("00:05:00"), ts("00:08:00"), "0", None, track="model")]
    assert app.active_track_id == "human"
    assert app._interval_track_scope() == "active"
    app.selected_interval = app.intervals[0]
    app._update_intervals_list()
    assert app.status_var.get() == HIDDEN_LINE
    return app


# ======================================== 1. Undo and Redo keep the name


def test_undo_keeps_its_own_sentence(hidden):
    app = hidden
    app.current_class_var.set("sw")
    app.current_selection = (ts("00:20:00"), ts("00:25:00"))
    app._add_interval()
    assert app.status_var.get() == "Added 1 sw interval(s)"
    app.selected_interval = [iv for iv in app.intervals
                             if iv.track == "model"][0]
    app._update_intervals_list()

    app._undo()

    assert app.status_var.get() == "Undo: add sw interval(s)"
    assert app.selected_interval is not None, \
        "the hidden selection is still selected -- Delete still works"
    assert app.selected_interval.track == "model"


def test_redo_keeps_its_own_sentence(hidden):
    app = hidden
    app.current_class_var.set("sw")
    app.current_selection = (ts("00:20:00"), ts("00:25:00"))
    app._add_interval()
    app.selected_interval = [iv for iv in app.intervals
                             if iv.track == "model"][0]
    app._update_intervals_list()
    app._undo()
    app._redo()
    assert app.status_var.get() == "Redo: add sw interval(s)"


def test_an_undo_that_LOSES_THE_ACTIVE_LANE_keeps_the_reconcile_sentence(
        app):
    """The case that makes _undo's claim load-bearing. Undoing an ingest
    removes the lane `_active_track_id` names, so Pack M2.6's reconcile
    re-points it and writes the sentence that outranks everything else.
    The ACTIVE LANE is part of the fact this module is about, and it just
    changed -- so without the claim the repaint would write the
    hidden-lane sentence straight over the reconcile's."""
    app.intervals[:] = [
        Interval(ts("00:05:00"), ts("00:08:00"), "0", None, track="model")]
    app.add_track_from_column("rule", "agent2", name="Agent2", locked=True)
    app._set_active_track("agent2", announce=False, repaint=False)
    app.selected_interval = [iv for iv in app.intervals
                             if iv.track == "model"][0]
    app._update_intervals_list()
    assert app.status_var.get() == HIDDEN_LINE

    app._undo()

    assert app.track_by_id("agent2") is None
    assert app.active_track_id == "human"
    assert app.status_var.get() == (
        "lane 'agent2' no longer exists -- active lane is now 'Human'")
    assert app.selected_interval is not None, \
        "and the hidden selection is still selected"
    assert app.selected_interval.track == "model"


def test_undoing_a_RESIZE_keeps_its_own_sentence(hidden):
    """The case a value key CANNOT reach. Undoing a resize really does
    move the selected interval's bounds back, so the fact's own terms
    change and any key says "this is new". Only the claim in _undo closes
    it -- and the same for the redo."""
    app = hidden
    iv = app.selected_interval
    app._lock_refusal_owed = None
    app._drag_mode = "resize_right"
    app._drag_iv = iv
    app._drag_preview = (iv.start, iv.end + pd.Timedelta("2min"))
    app._on_strip_release(None, app.active_pane)
    app._run_pending_redraw()

    app._undo()
    assert app.status_var.get() == "Undo", \
        "a drag-resize carries no gesture name, so plain 'Undo' is right"
    app._redo()
    assert app.status_var.get() == "Redo"


def test_the_announce_key_is_a_VALUE_key(app):
    """Half the fix, on its own terms: a snapshot restore hands back a
    value-equal COPY, and that must not read as a new selection."""
    from chronotagger.labeler.mixins.stats import hidden_selection_key
    one = Interval(ts("00:05:00"), ts("00:08:00"), "0", None, track="model")
    twin = Interval(ts("00:05:00"), ts("00:08:00"), "0", None, track="model")
    assert one is not twin
    assert hidden_selection_key(one, "active", "human") == \
        hidden_selection_key(twin, "active", "human")
    moved = Interval(ts("00:05:00"), ts("00:09:00"), "0", None, track="model")
    assert hidden_selection_key(moved, "active", "human") != \
        hidden_selection_key(one, "active", "human")


# ====================================== 2. the drag-resize keeps its line


def test_a_drag_resize_keeps_its_own_sentence(hidden):
    """The Model lane is VISIBLE on the strip -- only the sidebar list is
    filtered -- so dragging one of its bands while Human is active is an
    ordinary gesture. The redraw it asks for is COALESCED, so the refill
    lands after the line is written."""
    app = hidden
    iv = app.selected_interval
    app._lock_refusal_owed = None
    app._drag_mode = "resize_right"
    app._drag_iv = iv
    app._drag_preview = (iv.start, iv.end + pd.Timedelta("2min"))

    app._on_strip_release(None, app.active_pane)

    said = app.status_var.get()
    assert said.startswith("Resized: 0 ["), said
    assert app.selected_interval is not None
    assert app.selected_interval.track == "model"
    assert app.selected_interval.end == iv.end + pd.Timedelta("2min")

    # and it still says it once the coalesced redraw has been flushed
    app._run_pending_redraw()
    assert app.status_var.get() == said


# ========================================== 3. the lane switch says both


def test_a_lane_switch_that_pushes_the_selection_out_says_both(app):
    app.intervals[:] = [
        Interval(ts("00:20:00"), ts("00:25:00"), "sw", None, track="human")]
    app.selected_interval = app.intervals[0]
    app._update_intervals_list()
    assert app.status_var.get() != HIDDEN_LINE, \
        "the list SHOWS it while Human is active"
    app.status_var.set("quiet")

    app._set_active_track("model")

    assert app.status_var.get() == STAYS_LINE


def test_a_lane_switch_with_no_selection_says_only_the_lane(app):
    """The control, and the M2.6 wording, unchanged."""
    app.selected_interval = None
    app._set_active_track("model")
    assert app.status_var.get() == "Active lane: Model"


def test_a_lane_switch_that_BRINGS_the_selection_back_says_only_the_lane(app):
    app.intervals[:] = [
        Interval(ts("00:05:00"), ts("00:08:00"), "0", None, track="model")]
    app.selected_interval = app.intervals[0]
    app._update_intervals_list()
    app._set_active_track("model")
    assert app.status_var.get() == "Active lane: Model"


# ============================== 4. the sentence is still said when it is due


def test_selecting_a_DIFFERENT_hidden_lane_interval_still_announces(hidden):
    app = hidden
    app.status_var.set("quiet")
    app.intervals.append(
        Interval(ts("00:50:00"), ts("00:55:00"), "1", None, track="model"))
    app.selected_interval = app.intervals[-1]
    app._update_intervals_list()
    assert app.status_var.get() == HIDDEN_LINE


def test_a_filter_round_trip_still_announces(hidden):
    app = hidden
    app.interval_track_scope_var.set("all")
    app._update_intervals_list()
    app.status_var.set("quiet")
    app.interval_track_scope_var.set("active")
    app._update_intervals_list()
    assert app.status_var.get() == HIDDEN_LINE


def test_a_plain_refill_still_leaves_the_bar_alone(hidden):
    app = hidden
    app.status_var.set("Rule preview: 134 points -> 2 spans")
    for _ in range(3):
        app._update_intervals_list()
    assert app.status_var.get() == "Rule preview: 134 points -> 2 spans"


def test_the_claim_forgets_the_fact_when_the_fact_is_not_true(hidden):
    app = hidden
    assert app._claim_hidden_selection_line() != ""
    app.selected_interval = None
    assert app._claim_hidden_selection_line() == ""
    assert app._hidden_selection_announced is None
