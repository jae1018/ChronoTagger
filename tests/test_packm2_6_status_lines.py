"""Pack M2.6 -- THE STATUS BAR SAYS ONE TRUE THING AT A TIME.

Four sentences, four defects, all of them visible in computer-use
session 3 and all of them small.

  (a) THE CLOBBER. `stats.py` wrote "the selected interval sits on lane
      'X', which this list is not showing" on EVERY refill of the
      interval list, and a refill happens on every repaint. `rules.py`
      writes "Rule preview: 134 points -> 2 spans" and then calls
      _update_plot(), which refills the list, which wrote over it -- so
      the preview figure the user asked for never reached the bar.
      Measured: s3_4_skip_preview_0.png and s3_4_skip_committed.png both
      carry the lane sentence and neither carries the preview.
  (b) TWO SENTENCES FOR ONE GESTURE. Picking a hidden lane in the Lane
      list said "lane 'Wake (umbra)' is visible again" and then the
      active-lane setter said "Active lane: Wake (umbra)" over the top.
  (c) UNDO SAID "Undo". The gesture's own label has existed since Pack 1
      and was never printed, so the bar never said what came back.
  (d) A CANCELLED ADD LOOKED LIKE A SUCCESSFUL ONE. Cancel in the
      Overlap Detected box clears the whole selection (ratified, and
      unchanged here) and said nothing, so the previous "Added 1 umbra
      interval(s)" stayed on the bar.
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


def ts(hhmmss):
    return pd.Timestamp(DAY + hhmmss)


def frame():
    idx = pd.date_range(DAY + "00:00:00", periods=120, freq="30s")
    return pd.DataFrame({"a": np.linspace(0.0, 1.0, len(idx))}, index=idx)


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


def select_a_hidden_one(app):
    """Select an interval that the ACTIVE-lane filter leaves out."""
    app.intervals[:] = [
        Interval(ts("00:05:00"), ts("00:08:00"), "0", None, track="model"),
        Interval(ts("00:20:00"), ts("00:25:00"), "sw", None, track="human")]
    assert app.active_track_id == "human"
    app.selected_interval = app.intervals[0]


# ============================================ (a) the clobber


def test_the_line_is_written_when_the_selection_becomes_a_hidden_one(app):
    select_a_hidden_one(app)
    app.status_var.set("something else")
    app._update_intervals_list()
    assert app.status_var.get() == HIDDEN_LINE
    assert app.selected_interval is not None, \
        "it is still SELECTED -- Delete still works on it"


def test_a_PLAIN_REFILL_does_not_touch_the_status_bar(app):
    """THE DEFECT. Every repaint refills this list, and every refill
    wrote that sentence again, over whatever else was there."""
    select_a_hidden_one(app)
    app._update_intervals_list()
    assert app.status_var.get() == HIDDEN_LINE
    app.status_var.set("Rule preview: 134 points -> 2 spans")
    for _ in range(3):
        app._update_intervals_list()
    assert app.status_var.get() == "Rule preview: 134 points -> 2 spans"


def test_a_repaint_does_not_eat_the_rule_preview_line(app):
    """The whole path, end to end: rules.py writes the preview line and
    then repaints, which is where the sentence used to be lost."""
    select_a_hidden_one(app)
    app._update_intervals_list()
    app.status_var.set("Rule preview: 134 points -> 2 spans")
    app._update_plot()
    assert app.status_var.get() == "Rule preview: 134 points -> 2 spans"


def test_changing_the_selection_says_it_again(app):
    select_a_hidden_one(app)
    app._update_intervals_list()
    app.status_var.set("quiet")
    app.intervals.append(
        Interval(ts("00:40:00"), ts("00:45:00"), "1", None, track="model"))
    app.selected_interval = app.intervals[-1]
    app._update_intervals_list()
    assert app.status_var.get() == HIDDEN_LINE


def test_changing_the_filter_and_coming_back_says_it_again(app):
    select_a_hidden_one(app)
    app._update_intervals_list()
    assert app.status_var.get() == HIDDEN_LINE
    app.interval_track_scope_var.set("all")
    app._update_intervals_list()
    app.status_var.set("quiet")
    app.interval_track_scope_var.set("active")
    app._update_intervals_list()
    assert app.status_var.get() == HIDDEN_LINE, \
        "the filter changed, so the question is live again"


# ============================================ (b) one gesture, one line


def test_re_showing_a_lane_from_the_list_says_ONE_thing(app):
    app._set_active_track("model", announce=False, repaint=False)
    app._toggle_active_lane_visible()
    assert app.track_by_id("model").visible is False
    assert app.active_track_id == "human"

    app.lane_var.set("Model (hidden)")
    app._on_lane_combo_change()

    assert app.track_by_id("model").visible is True
    assert app.active_track_id == "model"
    assert app.status_var.get() == "lane 'Model' is visible again -- active lane"


def test_picking_a_visible_lane_still_announces_it_the_old_way(app):
    app._on_lane_combo_change_value = None
    app.lane_var.set("Model")
    app._on_lane_combo_change()
    assert app.status_var.get() == "Active lane: Model"


# ============================================ (c) Undo / Redo name it


def test_undo_and_redo_name_the_gesture(app):
    app.current_class_var.set("sw")
    app.current_selection = (ts("00:10:00"), ts("00:15:00"))
    app._add_interval()
    assert app.status_var.get() == "Added 1 sw interval(s)"
    app._undo()
    assert app.status_var.get() == "Undo: add sw interval(s)"
    app._redo()
    assert app.status_var.get() == "Redo: add sw interval(s)"


def test_an_empty_stack_still_says_the_old_thing(app):
    app._undo()
    assert app.status_var.get() == "Nothing to undo"
    app._redo()
    assert app.status_var.get() == "Nothing to redo"


# ============================================ (d) a cancelled Add says so


def test_cancelling_the_overlap_box_says_the_add_was_cancelled(app,
                                                               monkeypatch):
    import chronotagger.labeler.dialogs.overlap_resolution as OR

    class Cancelled(object):
        def __init__(self, parent=None, overlap_count=0,
                     on_policy_selected=None, **kw):
            self.policy = None

    monkeypatch.setattr(OR, "OverlapResolutionDialog", Cancelled)
    monkeypatch.setattr(app.root, "wait_window", lambda *a, **k: None)

    app.current_class_var.set("sw")
    app.current_selection = (ts("00:10:00"), ts("00:15:00"))
    app._add_interval()
    assert app.status_var.get() == "Added 1 sw interval(s)"

    # now an Add that overlaps it, cancelled in the box
    app.current_selection = (ts("00:12:00"), ts("00:18:00"))
    app._add_interval()

    assert len(app.intervals) == 1, "nothing was added"
    assert app.current_selection is None, \
        "Cancel still clears the selection -- ratified, unchanged"
    assert app.status_var.get() == "Add cancelled -- selection cleared"


# ====================================== (c2) and an un-gestured command


def test_a_bare_command_does_not_put_its_class_name_on_the_bar(app):
    """_execute_command names an un-gestured command after its CLASS, and
    a class name is not a sentence a user reads. Delete, Re-label and a
    drag-resize are exactly those three; without the filter the bar said
    `Undo: DeleteIntervalCommand`."""
    app.intervals[:] = [
        Interval(ts("00:10:00"), ts("00:15:00"), "sw", None, track="human")]
    app.selected_interval = app.intervals[0]
    app._delete_interval()
    app._undo()
    assert app.status_var.get() == "Undo"
    app._redo()
    assert app.status_var.get() == "Redo"


# ================================== (b2) unhiding with a rule staged


def test_unhiding_a_lane_with_a_rule_staged_still_says_the_rule_went(app):
    """The two-writes-one-line defect, one door further on. The setter
    drops the staged rule and -- because the unhide silenced its
    announcement -- its own note about the drop is what the combined
    sentence would write over. Both halves have to be in ONE line."""
    from chronotagger.labeler.dialogs.label_by_rule import (
        LabelByRuleResult, RuleCondition)
    app._set_active_track("model", announce=False, repaint=False)
    app._toggle_active_lane_visible()
    app._rule_preview_apply(LabelByRuleResult(
        conditions=[RuleCondition(column="a", op=">=", value=0.0)],
        combine_mode="AND", nan_as_true=False, overlap_policy="skip",
        scope="window"))
    assert app._commit_spans
    app.lane_var.set("Model (hidden)")
    app._on_lane_combo_change()
    assert list(app._commit_spans) == []
    assert app.status_var.get() == (
        "lane 'Model' is visible again -- active lane -- rule preview "
        "cleared (it belonged to the lane you left)")


# ============================ (a2) the memory is dropped with the selection


def test_clearing_the_selection_lets_the_line_be_said_again(app):
    """EDIT 570's `else` branch. With the selection gone the sentence is
    no longer true, so its memory has to go too -- otherwise re-selecting
    the SAME interval says nothing at all."""
    select_a_hidden_one(app)
    app._update_intervals_list()
    assert app.status_var.get() == HIDDEN_LINE
    app.selected_interval = None
    app._update_intervals_list()
    app.status_var.set("quiet")
    app.selected_interval = app.intervals[0]
    app._update_intervals_list()
    assert app.status_var.get() == HIDDEN_LINE
