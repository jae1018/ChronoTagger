"""Pack M2.6 -- THE By-Rule BOX'S OK COMMITS WHAT IT PREVIEWED.

THE DEFECT, in three parts, all measured on the feel-test driver
(edit_pack/evidence/scratch/m2_orch/logs/probe_s3_refute.txt PARTs A-C
and probe_s3_rule_flow.txt).

  1. OK ONLY STAGED. It wrote `self._overlap_policy` and returned; the
     spans sat in `_commit_spans` until the user pressed Add.
  2. THE POLICY WAS DEAD. `_add_interval` never read `_overlap_policy`.
     It re-detected the same overlaps and PROMPTED AGAIN. Choose Replace
     in the rule box, answer Skip in that second prompt, and the commit
     is 2 carved spans over 125 points where the preview showed 1
     replacing span over 134. Cancel in the second prompt wiped the
     whole staging -- at scope=dataset, a 30-day result -- while the
     status bar still advertised the preview.
  3. THE STAGING WAS BOUND TO NO LANE. Stage on Region, switch to Wake,
     press Add: "Added 2 umbra interval(s)", with REGION-CARVED geometry
     on Wake and the class auto-substituted, no prompt.

AFTER THIS PACK. OK commits, at once, as ONE gesture, on the ACTIVE
lane, with the class in the dropdown and the policy the user picked in
that box. One undo takes the whole commit back. No overlap dialog is
constructed for a rule commit at all -- the question it asks was
answered by the radio button. A locked lane refuses on the status bar
with nothing built and nothing staged. And a lane switch drops a staged
rule, because those spans were carved against the lane you left.

The entry point is `_commit_rule_result(res)`, which is a method rather
than the tail of `_open_label_by_rule_dialog` precisely so these pins can
drive the whole commit without a Toplevel.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tkinter as tk

from chronotagger.core.models import Interval
from chronotagger.core.tracks import intervals_on
from chronotagger.labeler.dialogs.label_by_rule import (LabelByRuleResult,
                                                        RuleCondition)

DAY = "2015-01-03 "

HUMAN = {"id": "human", "name": "Human", "classes": ["sw", "msh", "UNKNOWN"],
         "class_colors": {"sw": "#4e79a7", "msh": "#f28e2b",
                          "UNKNOWN": "#7f7f7f"}, "order": 0}
MODEL = {"id": "model", "name": "Model", "classes": ["0", "1"],
         "class_colors": {"0": "#111111", "1": "#222222"}, "order": 1}
LOCKED = dict(MODEL)
LOCKED["locked"] = True

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
    return pd.DataFrame({"a": np.linspace(0.0, 1.0, len(idx))}, index=idx)


def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["a"])


def rule(policy):
    return LabelByRuleResult(
        conditions=[RuleCondition(column="a", op=">=", value=0.0)],
        combine_mode="AND", nan_as_true=False, overlap_policy=policy,
        scope="window")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    """Every messagebox entry point, COUNTED. Zero is an assertion here."""
    import tkinter.messagebox as mb
    calls = []
    for kind in ("showinfo", "showwarning", "showerror", "askyesno",
                 "askyesnocancel", "askokcancel"):
        monkeypatch.setattr(
            mb, kind, lambda *a, _k=kind, **kw: calls.append(_k) or True)
    return calls


@pytest.fixture(autouse=True)
def _overlap_spy(monkeypatch):
    """The overlap box, COUNTED AT ITS CLASS.

    `OverlapResolutionDialog` SUBCLASSES `tk.Toplevel` and binds that
    base class at class-definition time, so patching `tkinter.Toplevel`
    afterwards never sees it -- measured in probe_s3_locked_dialogs'
    own spy control. Counting at the class the caller imports is the
    measurement that cannot miss, and the caller imports it lazily from
    this module, so this patch reaches it.
    """
    import chronotagger.labeler.dialogs.overlap_resolution as OR
    built = []

    class Fake(object):
        def __init__(self, parent=None, overlap_count=0,
                     on_policy_selected=None, **kw):
            built.append(overlap_count)
            self.policy = None

    monkeypatch.setattr(OR, "OverlapResolutionDialog", Fake)
    return built


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
    lbl = build([HUMAN, MODEL], tmp_path)
    yield lbl
    lbl.root.destroy()


@pytest.fixture
def locked(tmp_path):
    lbl = build([HUMAN, LOCKED], tmp_path)
    yield lbl
    lbl.root.destroy()


def tops(app):
    return [w for w in app.root.winfo_children() if isinstance(w, tk.Toplevel)]


def seed(app):
    """One msh interval in the middle of the window, on the human lane."""
    app.intervals[:] = [
        Interval(ts("00:20:00"), ts("00:30:00"), "msh", None, track="human")]


# =============================================== 1. skip: one gesture


def test_skip_commits_every_previewed_span_in_ONE_gesture(app, _dialogs,
                                                          _overlap_spy):
    seed(app)
    app.current_class_var.set("sw")
    pts, spans = app._rule_preview_apply(rule("skip"))
    assert spans == 2, "the rule spans the window and the seed carves it"
    assert len(app._commit_spans) == 2
    depth = len(app.undo_stack)

    n = app._commit_rule_result(rule("skip"))

    assert n == spans, "the number committed is the number previewed"
    assert len(app.undo_stack) == depth + 1, "ONE undo entry, not two"
    assert len(intervals_on(app.intervals, "human")) == 3
    assert sorted(iv.label for iv in app.intervals) == ["msh", "sw", "sw"]
    assert _overlap_spy == [], "no second prompt for a rule commit"
    assert _dialogs == []
    assert app.status_var.get() == \
        "Added 2 sw interval(s) by rule (skip overlaps)"
    assert list(app._commit_spans) == [] and list(app.current_spans) == []


def test_one_undo_takes_the_whole_skip_commit_back(app):
    seed(app)
    app.current_class_var.set("sw")
    app._rule_preview_apply(rule("skip"))
    app._commit_rule_result(rule("skip"))
    app._undo()
    assert [(str(iv.start), str(iv.end), iv.label) for iv in app.intervals] \
        == [(str(ts("00:20:00")), str(ts("00:30:00")), "msh")]


# =============================================== 2. replace: one gesture


def test_replace_commits_the_span_it_previewed_and_carves_once(app, _dialogs,
                                                               _overlap_spy):
    seed(app)
    app.current_class_var.set("sw")
    pts, spans = app._rule_preview_apply(rule("replace"))
    assert spans == 1, "replace previews the whole hypothetical span"
    depth = len(app.undo_stack)

    n = app._commit_rule_result(rule("replace"))

    assert n == spans == 1
    assert len(app.undo_stack) == depth + 1
    assert [iv.label for iv in app.intervals] == ["sw"], \
        "the msh interval was replaced, not carved around"
    assert _overlap_spy == [], \
        "the policy was already chosen; the box must not be built"
    assert tops(app) == []
    assert _dialogs == []
    assert app.status_var.get() == \
        "Added 1 sw interval(s) by rule (replace overlaps)"


def test_one_undo_restores_what_replace_took(app):
    seed(app)
    before = [(str(iv.start), str(iv.end), iv.label) for iv in app.intervals]
    app.current_class_var.set("sw")
    app._rule_preview_apply(rule("replace"))
    app._commit_rule_result(rule("replace"))
    app._undo()
    assert [(str(iv.start), str(iv.end), iv.label)
            for iv in app.intervals] == before


def test_the_policy_the_box_chose_is_the_policy_that_lands(app):
    """The two policies must give DIFFERENT answers from one preview --
    otherwise the pins above prove nothing about the policy."""
    seed(app)
    app.current_class_var.set("sw")
    app._rule_preview_apply(rule("skip"))
    assert app._commit_rule_result(rule("skip")) == 2
    app._undo()
    app._rule_preview_apply(rule("replace"))
    assert app._commit_rule_result(rule("replace")) == 1


# =============================================== 3. the lock, and the lane


def test_a_locked_lane_refuses_the_rule_commit_and_drops_the_staging(
        locked, _dialogs, _overlap_spy):
    locked._set_active_track("model", announce=False, repaint=False)
    locked._commit_spans = [(ts("00:05:00"), ts("00:10:00"))]
    locked.current_spans = [(ts("00:05:00"), ts("00:10:00"))]

    assert locked._commit_rule_result(rule("replace")) == 0

    assert locked.intervals == [], "nothing built"
    assert "add refused" in locked.status_var.get()
    assert "Model" in locked.status_var.get()
    assert list(locked._commit_spans) == []
    assert list(locked.current_spans) == [], "and the yellow is gone"
    assert tops(locked) == [] and _dialogs == [] and _overlap_spy == []


def test_switching_lane_drops_the_staged_rule_and_says_so(app):
    seed(app)
    app._rule_preview_apply(rule("skip"))
    assert app._commit_spans and app.current_spans
    app._set_active_track("model")
    assert list(app._commit_spans) == []
    assert list(app.current_spans) == []
    assert app.status_var.get() == (
        "Active lane: Model -- rule preview cleared (it belonged to the "
        "lane you left)"), app.status_var.get()


def test_switching_lane_with_nothing_staged_says_nothing_new(app):
    app._set_active_track("model")
    assert app.status_var.get() == "Active lane: Model"


def test_a_commit_with_nothing_staged_builds_nothing(app, _dialogs):
    app._commit_spans = []
    app.current_spans = []
    assert app._commit_rule_result(rule("skip")) == 0
    assert app.intervals == []
    assert "0 interval(s)" in app.status_var.get()
    assert _dialogs == []


def test_the_count_comes_from_the_ADDER_not_from_len_spans(app, _dialogs):
    """DR3's contract, pinned. A gesture that changes nothing routes to
    the adder's "Nothing Added" branch and reports 0, where `len(spans)`
    would report 1 and the bar would claim an interval that is not
    there."""
    seed(app)
    app.current_class_var.set("msh")
    app._commit_spans = [(ts("00:20:00"), ts("00:30:00"))]
    app.current_spans = list(app._commit_spans)
    assert app._commit_rule_result(rule("replace")) == 0
    assert len(app.intervals) == 1
    assert app.status_var.get() == \
        "Added 0 msh interval(s) by rule (replace overlaps)"


def test_the_older_Add_door_still_reads_the_staging(app, _overlap_spy):
    """`_add_interval` is UNCHANGED: it still commits `_commit_spans` and
    it still prompts when they overlap. That door is what the box and
    two-click gestures come in by, and this pins that the pack did not
    quietly close it."""
    seed(app)
    app.current_class_var.set("sw")
    app._commit_spans = [(ts("00:40:00"), ts("00:45:00"))]
    app._add_interval()
    assert len(app.intervals) == 2
    assert _overlap_spy == [], "that span overlaps nothing"
