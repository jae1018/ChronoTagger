"""Pack M2 -- THE ACTIVE-LANE SCOPE, THE LOCK, AND THE SIDEBAR.

PART A of the pack is not a feature. It is three paths M1 shipped that
DESTROY DATA ON A LANE THE USER IS NOT LOOKING AT, reachable today by
anyone who calls `add_track_from_column`:

  `_apply_label_manager_result`  one OK in Manage Labels rewrote ALL 256
                                intervals on the locked `agent` lane to
                                UNKNOWN -- a label that lane does not
                                declare -- which left the session
                                unexportable while the autosave wrote the
                                wreckage happily.
  `_clear_intervals_in_range`   deleted 50 and truncated 2 intervals on
                                the locked lane while `region` was active.
  `_find_gaps_in_current_range` returned 0 gaps for an EMPTY lane, because
                                another lane covered the record, so "Label
                                Unassigned" was a silent no-op.

Each of those numbers is pinned below as the RED DIRECTION: the test
builds the same situation and asserts the OTHER lane is untouched.

The lock is the second half, at nine paths -- the eight the gather
enumerated as mutations plus the Clear Range warning box -- and
the shape of the refusal is part of the contract: a STATUS LINE and never
a modal, because `crud.py` already carries five `showwarning` calls on the
label hot path and the campaign's floor is zero unexpected modals.
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
                          "UNKNOWN": "#7f7f7f"}}
AGENT = {"id": "agent", "name": "Agent (C-MMAE)", "classes": ["0", "1"],
         "class_colors": {"0": "#111111", "1": "#222222"},
         "locked": True, "order": 1}
OPEN = dict(AGENT)
OPEN["locked"] = False

LAYOUT = {
    "nrows": 3, "ncols": 1,
    "areas": [
        {"key": "panel1", "row": 0, "col": 0, "role": "time"},
        {"key": "panel2", "row": 1, "col": 0, "role": "time"},
        {"key": "labels", "row": 2, "col": 0, "role": "labels"},
    ],
}


@pytest.fixture(autouse=True)
def _dialog_counter(monkeypatch):
    """EVERY dialog function, counted. Zero is an assertion in this file."""
    import tkinter.messagebox as mb
    import tkinter.filedialog as fd
    calls = []
    for kind in ("showinfo", "showwarning", "showerror", "askyesno",
                 "askyesnocancel", "askokcancel"):
        monkeypatch.setattr(
            mb, kind, lambda *a, _k=kind, **kw: calls.append(_k) or True)
    for kind in ("asksaveasfilename", "askopenfilename", "askdirectory"):
        monkeypatch.setattr(fd, kind, lambda *a, **kw: calls.append(kind) or "")
    return calls


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def ts(hhmmss):
    return pd.Timestamp(DAY + hhmmss)


def frame():
    idx = pd.date_range(DAY + "00:00:00", periods=120, freq="30s")
    return pd.DataFrame({"a": np.linspace(0, 1, len(idx)),
                         "rule": ["0"] * 60 + ["1"] * 60}, index=idx)


def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["a"])
    axs["panel2"].plot(df.index, df["a"])


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
    lbl = build([HUMAN, AGENT], tmp_path)
    yield lbl
    lbl.root.destroy()


@pytest.fixture
def unlocked(tmp_path):
    lbl = build([HUMAN, OPEN], tmp_path)
    yield lbl
    lbl.root.destroy()


def agent_ivs(n=8, label="0"):
    out = []
    for j in range(n):
        s = ts("00:%02d:00" % (j * 5))
        out.append(Interval(s, s + pd.Timedelta("4min"), label, None,
                            track="agent"))
    return out


def counts(lbl):
    out = {}
    for iv in lbl.intervals:
        out[iv.track] = out.get(iv.track, 0) + 1
    return out


def labels_on(lbl, track):
    return sorted({iv.label for iv in lbl.intervals if iv.track == track})


def result(classes, rename=None, reassign=None, colors=None):
    from chronotagger.labeler.dialogs.label_manager import LabelManagerResult
    return LabelManagerResult(
        classes=list(classes),
        class_colors=dict(colors or {c: "#123456" for c in classes}),
        rename_map=dict(rename or {}),
        reassign_map=dict(reassign or {}),
    )


# ============================== 1. Manage Labels touches ONE lane


def test_manage_labels_leaves_the_other_lanes_intervals_alone(unlocked):
    """THE HEADLINE PIN. The gather's unscoped run rewrote 256 of 256."""
    unlocked.intervals[:] = agent_ivs() + [
        Interval(ts("00:01:00"), ts("00:03:00"), "sw", None, track="human")]
    unlocked._apply_label_manager_result(
        result(["UNKNOWN", "solar_wind", "msh"], rename={"sw": "solar_wind"}))
    assert labels_on(unlocked, "agent") == ["0"], \
        "an OK in Manage Labels must not touch another lane's labels"
    assert labels_on(unlocked, "human") == ["solar_wind"]
    assert counts(unlocked) == {"agent": 8, "human": 1}
    assert unlocked.track_by_id("agent").classes == ["0", "1"]


def test_the_RENAME_pass_does_not_touch_a_shared_class_on_another_lane(
        tmp_path):
    """The rename pass is its own loop and its own mutant: two lanes both
    carrying `sw`, renamed on the active one only."""
    shared = dict(OPEN)
    shared["classes"] = ["sw"]
    shared["class_colors"] = {"sw": "#111111"}
    lbl = build([HUMAN, shared], tmp_path)
    try:
        lbl.intervals[:] = [
            Interval(ts("00:01:00"), ts("00:03:00"), "sw", None,
                     track="human"),
            Interval(ts("00:05:00"), ts("00:07:00"), "sw", None,
                     track="agent"),
        ]
        lbl._apply_label_manager_result(
            result(["UNKNOWN", "solar_wind", "msh"],
                   rename={"sw": "solar_wind"}))
        assert labels_on(lbl, "human") == ["solar_wind"]
        assert labels_on(lbl, "agent") == ["sw"], \
            "the rename pass may only touch the lane being edited"
    finally:
        lbl.root.destroy()


def test_the_unknown_fallback_does_not_flatten_the_other_lane(unlocked):
    """The third pass is the one that did the damage: every label not in
    the NEW class list became UNKNOWN, on every lane."""
    unlocked.intervals[:] = agent_ivs() + [
        Interval(ts("00:01:00"), ts("00:03:00"), "msh", None, track="human")]
    unlocked._apply_label_manager_result(result(["UNKNOWN", "sw"]))
    assert labels_on(unlocked, "agent") == ["0"]
    assert labels_on(unlocked, "human") == ["UNKNOWN"]


def test_the_usage_counts_the_dialog_is_given_are_the_active_lanes(unlocked,
                                                                  monkeypatch):
    seen = {}

    class FakeDialog(object):
        def __init__(self, parent=None, classes=None, class_colors=None,
                     usage_counts=None, reserved=None):
            seen["counts"] = dict(usage_counts or {})
            self.result = None

    import chronotagger.labeler.mixins.labels as labels_mod
    monkeypatch.setattr(labels_mod, "LabelManagerDialog", FakeDialog)
    # Tk's wait_window wants a real widget (it reads dlg._w), and the fake
    # is deliberately not one.
    monkeypatch.setattr(unlocked.root, "wait_window", lambda w=None: None)
    unlocked.intervals[:] = [
        Interval(ts("00:01:00"), ts("00:03:00"), "sw", None, track="human"),
        Interval(ts("00:05:00"), ts("00:07:00"), "sw", None, track="agent"),
    ]
    unlocked._open_label_manager()
    assert seen["counts"]["sw"] == 1, "the other lane's `sw` is not this " \
                                     "lane's usage"


def test_manage_labels_on_a_locked_lane_is_refused_with_no_modal(
        app, _dialog_counter):
    app.intervals[:] = agent_ivs()
    app._set_active_track("agent", announce=False, repaint=False)
    before = list(app.track_by_id("agent").classes)
    app._apply_label_manager_result(result(["x", "y"]))
    assert app.track_by_id("agent").classes == before
    assert labels_on(app, "agent") == ["0"]
    assert "locked" in app.status_var.get()
    assert "edit the label schema refused" in app.status_var.get()
    assert _dialog_counter == []


# ============================== 2. clear range touches ONE lane


def test_clear_range_clears_the_active_lane_only(unlocked):
    """The gather's unscoped run deleted 50 and truncated 2 on the other
    lane. Here: the other lane's count does not move at all."""
    unlocked.intervals[:] = agent_ivs() + [
        Interval(ts("00:01:00"), ts("00:03:00"), "sw", None, track="human")]
    got = unlocked._clear_intervals_in_range(ts("00:00:00"), ts("00:40:00"))
    assert got["deleted"] == 1 and got["truncated"] == 0
    assert counts(unlocked) == {"agent": 8}


def test_clear_range_truncates_and_splits_on_the_active_lane_only(unlocked):
    unlocked.intervals[:] = [
        Interval(ts("00:00:00"), ts("00:30:00"), "sw", None, track="human"),
        Interval(ts("00:00:00"), ts("00:30:00"), "0", None, track="agent"),
    ]
    got = unlocked._clear_intervals_in_range(ts("00:10:00"), ts("00:20:00"))
    assert got["split"] == 1
    agent = [iv for iv in unlocked.intervals if iv.track == "agent"]
    assert len(agent) == 1
    assert (agent[0].start, agent[0].end) == (ts("00:00:00"), ts("00:30:00"))


def test_the_clear_range_ANALYSIS_counts_the_active_lane_only(unlocked):
    """The dialog and the edit must agree, or the dialog promises 50
    deletions and one happens."""
    unlocked.intervals[:] = agent_ivs() + [
        Interval(ts("00:01:00"), ts("00:03:00"), "sw", None, track="human")]
    a = unlocked._analyze_intervals_in_range(ts("00:00:00"), ts("00:40:00"))
    assert a["to_delete"] == 1 and a["total_affected"] == 1
    unlocked._set_active_track("agent", announce=False, repaint=False)
    b = unlocked._analyze_intervals_in_range(ts("00:00:00"), ts("00:40:00"))
    assert b["total_affected"] == 8


def test_clear_range_on_a_locked_lane_is_refused_and_changes_nothing(
        app, _dialog_counter):
    app.intervals[:] = agent_ivs()
    app._set_active_track("agent", announce=False, repaint=False)
    got = app._clear_intervals_in_range(ts("00:00:00"), ts("00:40:00"))
    assert got == {"deleted": 0, "truncated": 0, "split": 0,
                   "total_affected": 0}
    assert counts(app) == {"agent": 8}
    assert "clear range refused" in app.status_var.get()
    assert _dialog_counter == []


# ============================== 3. a gap is a gap ON A LANE


def test_gaps_are_found_on_the_active_lane_even_when_another_lane_covers_all(
        unlocked):
    """MEASURED RED: 0 gaps for a lane holding NO INTERVALS AT ALL."""
    unlocked.intervals[:] = [
        Interval(unlocked.t0, unlocked.t1, "0", None, track="agent")]
    gaps = unlocked._find_gaps_in_current_range()
    assert len(gaps) == 1
    assert gaps[0] == (unlocked.t0, unlocked.t1)
    unlocked._set_active_track("agent", announce=False, repaint=False)
    assert unlocked._find_gaps_in_current_range() == []


def test_fill_gaps_on_a_locked_lane_is_refused(app, _dialog_counter):
    app.intervals[:] = agent_ivs()
    app._set_active_track("agent", announce=False, repaint=False)
    n = len(app.intervals)
    app._assign_gaps_to_label([(ts("00:41:00"), ts("00:45:00"))], "0")
    assert len(app.intervals) == n
    assert "fill gaps refused" in app.status_var.get()
    assert _dialog_counter == []


def test_a_gap_fill_still_works_on_an_unlocked_lane(unlocked):
    n = len(unlocked.intervals)
    unlocked._assign_gaps_to_label([(ts("00:10:00"), ts("00:12:00"))], "sw")
    assert len(unlocked.intervals) == n + 1
    assert unlocked.intervals[-1].track == "human"


# ============================== 4. the rest of the nine


def test_add_on_a_locked_lane_is_refused(app, _dialog_counter):
    app._set_active_track("agent", announce=False, repaint=False)
    app.current_selection = (ts("00:02:00"), ts("00:05:00"))
    app._add_interval()
    assert app.intervals == []
    assert "add refused" in app.status_var.get()
    assert _dialog_counter == []


def test_the_rules_commit_on_a_locked_lane_is_refused(app):
    """AMENDED by Pack M2.6, not deleted: the rule commit moved doors.

    Pack M2's sentence was "the rules engine sets _commit_spans and the
    user presses the same Add button, so one guard covers both doors".
    Since M2.6 the By-Rule dialog's OK COMMITS, through
    `_commit_rule_result`, and that is the door a rule comes in by -- so
    it is pinned here first. The OLDER door is pinned straight after it,
    unchanged in substance: `_add_interval` still reads `_commit_spans`
    and still refuses on a locked lane, which is what keeps the box and
    two-click gestures honest.
    """
    from chronotagger.labeler.dialogs.label_by_rule import (
        LabelByRuleResult, RuleCondition)
    app._set_active_track("agent", announce=False, repaint=False)
    app._commit_spans = [(ts("00:02:00"), ts("00:05:00")),
                         (ts("00:10:00"), ts("00:12:00"))]
    res = LabelByRuleResult(
        conditions=[RuleCondition(column="a", op=">=", value=0.0)],
        combine_mode="AND", nan_as_true=False, overlap_policy="replace",
        scope="window")
    assert app._commit_rule_result(res) == 0
    assert app.intervals == []
    assert "add refused" in app.status_var.get()
    assert list(app._commit_spans) == [], "a refusal leaves nothing staged"

    # the older door, unchanged
    app._commit_spans = [(ts("00:02:00"), ts("00:05:00")),
                         (ts("00:10:00"), ts("00:12:00"))]
    app._add_interval()
    assert app.intervals == []
    assert "add refused" in app.status_var.get()


def test_relabel_and_delete_judge_the_intervals_OWN_lane(app,
                                                         _dialog_counter):
    """A click on a locked lane SELECTS without activating, so the selected
    interval can be on a lane that is not the active one."""
    app.intervals[:] = agent_ivs()
    app.selected_interval = app.intervals[0]
    assert app.active_track_id == "human"
    app.current_class_var.set("msh")
    app._relabel_interval()
    assert app.intervals[0].label == "0"
    assert "relabel refused" in app.status_var.get()
    app._delete_interval()
    assert len(app.intervals) == 8
    assert "delete refused" in app.status_var.get()
    assert _dialog_counter == []


def _locked_band_press(app):
    """The event a plain left click INSIDE the locked agent band produces."""
    from matplotlib.backend_bases import MouseEvent
    from chronotagger.core.lanes import lane_band, pad_for
    ax = app.active_pane.strip_ax
    bb = ax.get_window_extent()
    iv = app.selected_interval
    xf = float((iv.start + (iv.end - iv.start) / 2 - app.t0)
               / (app.t1 - app.t0))
    row = app.active_pane._strip_lane_ids.index("agent")
    lo, hi = lane_band(row, app.active_pane._strip_lane_count,
                       pad_for(app.active_pane._strip_lane_count))
    ev = MouseEvent("button_press_event", ax.figure.canvas,
                    int(round(bb.x0 + xf * bb.width)),
                    int(round(bb.y0 + 0.5 * (lo + hi) * bb.height)),
                    button=1)
    ev.inaxes = ax
    return ev


def _locked_setup(app):
    app.intervals[:] = agent_ivs()
    app.selected_interval = app.intervals[1]
    app._update_plot()
    return _locked_band_press(app)


def test_a_drag_on_a_locked_lane_never_starts(app):
    """AMENDED by Pack M2.5: the press no longer writes the refusal.

    _drag_mode is still never set -- that is the whole contract -- but the
    sentence the user reads after a plain press is now the confirmation of
    the selection, with the lock named. The refusal arrives on MOTION; the
    two pins below cover both halves.
    """
    ev = _locked_setup(app)
    app._drag_mode = None
    app._on_strip_press(ev, app.active_pane)
    assert app._drag_mode is None
    assert "lane locked (Ctrl+L to unlock)" in app.status_var.get()
    assert "drag refused" not in app.status_var.get()


def test_a_press_on_a_locked_band_confirms_the_selection(app,
                                                         _dialog_counter):
    """Pack M2.5 item 8: press only, no motion. The user reads what they
    selected AND that the lane is locked, in one line, with no dialog."""
    ev = _locked_setup(app)
    iv = app.selected_interval
    app._on_strip_press(ev, app.active_pane)
    want = ("Selected: %s [%s -> %s] -- lane locked (Ctrl+L to unlock)"
            % (iv.label, iv.start.strftime("%H:%M:%S"),
               iv.end.strftime("%H:%M:%S")))
    assert app.status_var.get() == want
    assert app.selected_interval is iv
    assert app._drag_mode is None
    assert _dialog_counter == []


def test_the_drag_refusal_arrives_on_MOTION_and_only_once(app,
                                                          _dialog_counter):
    """Pack M2.5 item 8: the existing sentence, on the gesture it is about.

    One press then one motion with the button still held writes the drag
    refusal exactly once; a second motion writes nothing more.
    """
    ev = _locked_setup(app)
    app._on_strip_press(ev, app.active_pane)
    assert app._lock_refusal_owed is app.selected_interval
    app._on_strip_motion(ev, app.active_pane)
    assert app.status_var.get() == ("track 'Agent (C-MMAE)' is locked "
                                    "(press Ctrl+L to unlock) -- drag "
                                    "refused")
    assert app._lock_refusal_owed is None
    assert app._drag_mode is None
    app.status_var.set("SENTINEL")
    app._on_strip_motion(ev, app.active_pane)
    assert app.status_var.get() == "SENTINEL"
    assert _dialog_counter == []


def test_a_press_and_release_with_no_motion_never_refuses(app):
    """The release clears what the press owed, so a later hover over the
    same band is a hover and not a refusal."""
    ev = _locked_setup(app)
    app._on_strip_press(ev, app.active_pane)
    app._on_strip_release(ev, app.active_pane)
    assert app._lock_refusal_owed is None
    app.status_var.set("SENTINEL")
    app._on_strip_motion(ev, app.active_pane)
    assert app.status_var.get() == "SENTINEL"


def test_a_motion_that_has_left_the_band_does_not_spend_the_refusal(app):
    """DR7's second condition, pinned.

    `mode is not None` is the "still on that band" test: a motion above the
    top band after a press on the locked band leaves the refusal OWED and
    the status bar untouched, so the sentence still arrives on the gesture
    it is about.
    """
    ev = _locked_setup(app)
    app._on_strip_press(ev, app.active_pane)
    owed = app._lock_refusal_owed
    assert owed is app.selected_interval
    off = _locked_band_press(app)
    bb = app.active_pane.strip_ax.get_window_extent()
    off.y = int(round(bb.y0 + 0.999 * bb.height))
    assert app._hit_test_selected(off) is None
    app.status_var.set("SENTINEL")
    app._on_strip_motion(off, app.active_pane)
    assert app.status_var.get() == "SENTINEL"
    assert app._lock_refusal_owed is owed
    app._on_strip_motion(ev, app.active_pane)
    assert "drag refused" in app.status_var.get()


def test_an_unlocked_lane_still_edits(unlocked):
    unlocked.current_selection = (ts("00:02:00"), ts("00:05:00"))
    unlocked.current_class_var.set("sw")
    unlocked._add_interval()
    assert len(unlocked.intervals) == 1
    unlocked.selected_interval = unlocked.intervals[0]
    unlocked.current_class_var.set("msh")
    unlocked._relabel_interval()
    assert unlocked.intervals[0].label == "msh"
    unlocked._delete_interval()
    assert unlocked.intervals == []


def test_the_ingest_is_NOT_lock_guarded(app):
    """The ingest is what CREATES a locked lane. A lock that blocked it
    would make the flag impossible to use for what it is for."""
    n = len(app.intervals)
    app.add_track_from_column("rule", "agent", gap_tolerance="10min")
    assert len(app.intervals) > n
    assert app.track_by_id("agent").locked is True


# ============================== 5. the sidebar


def test_the_lane_column_shows_the_lanes_display_name(unlocked):
    unlocked.intervals[:] = [
        Interval(ts("00:01:00"), ts("00:03:00"), "sw", None, track="human")]
    unlocked._update_intervals_list()
    rows = unlocked.intervals_tree.get_children()
    assert len(rows) == 1
    vals = unlocked.intervals_tree.item(rows[0])["values"]
    assert len(vals) == 5
    assert str(vals[4]) == "Human"


def test_the_row_tag_is_per_lane(unlocked):
    unlocked.intervals[:] = [
        Interval(ts("00:01:00"), ts("00:03:00"), "sw", None, track="human")]
    unlocked._update_intervals_list()
    rows = unlocked.intervals_tree.get_children()
    assert tuple(unlocked.intervals_tree.item(rows[0])["tags"]) \
        == ("human|sw",)


def test_two_lanes_sharing_a_class_name_get_their_own_colours(tmp_path):
    """DR14, closed. Measured RED under M1: 1 row in the WRONG lane's
    colour and 2 with no colour at all, of 6."""
    shared = dict(AGENT)
    shared["locked"] = False
    shared["classes"] = ["sw"]
    shared["class_colors"] = {"sw": "#111111"}
    lbl = build([HUMAN, shared], tmp_path)
    try:
        lbl.intervals[:] = [
            Interval(ts("00:01:00"), ts("00:03:00"), "sw", None,
                     track="human"),
            Interval(ts("00:05:00"), ts("00:07:00"), "sw", None,
                     track="agent"),
        ]
        lbl.interval_track_scope_var.set("all")
        lbl._update_intervals_list()
        a = str(lbl.intervals_tree.tag_configure("human|sw", "background"))
        b = str(lbl.intervals_tree.tag_configure("agent|sw", "background"))
        assert a == "#4e79a7"
        assert b == "#111111"
        assert a != b
    finally:
        lbl.root.destroy()


def test_the_list_defaults_to_the_active_lane_and_the_filter_opens_it(
        unlocked):
    unlocked.intervals[:] = agent_ivs() + [
        Interval(ts("00:01:00"), ts("00:03:00"), "sw", None, track="human")]
    unlocked._update_intervals_list()
    assert len(unlocked.intervals_tree.get_children()) == 1
    assert unlocked._interval_track_scope() == "active"
    unlocked.interval_track_scope_var.set("all")
    unlocked._on_interval_track_scope_change()
    assert len(unlocked.intervals_tree.get_children()) == 9
    assert "every lane" in unlocked.status_var.get()


def test_the_filtered_rows_iid_still_indexes_the_right_interval(unlocked):
    """This is the bug Pack M0 fixed and the filter is what would have
    shipped it: the ordinal would have named a different lane's interval."""
    ivs = []
    for j in range(5):
        ivs.append(Interval(ts("00:%02d:00" % (j * 6)),
                            ts("00:%02d:30" % (j * 6)), "0", None,
                            track="agent"))
        ivs.append(Interval(ts("00:%02d:00" % (j * 6 + 2)),
                            ts("00:%02d:30" % (j * 6 + 2)), "sw", None,
                            track="human"))
    unlocked.intervals[:] = ivs
    unlocked._update_intervals_list()
    rows = list(unlocked.intervals_tree.get_children())
    assert len(rows) == 5
    for r in rows:
        iv = unlocked._interval_row_map[r]
        assert iv.track == "human"
        assert iv.start.strftime("%H:%M:%S") == \
            str(unlocked.intervals_tree.item(r)["values"][0])
        ordinal = int(unlocked.intervals_tree.item(r)["text"])
        assert unlocked.intervals[ordinal - 1] is not iv or ordinal == 1


def test_a_refill_keeps_the_selection_on_the_row_that_carries_it(unlocked):
    """DR6. The clear drops the Treeview's selection; the refill puts it
    back on the row whose iid still carries the selected interval."""
    unlocked.intervals[:] = [
        Interval(ts("00:%02d:00" % (j * 6)), ts("00:%02d:30" % (j * 6)),
                 "sw", None, track="human") for j in range(5)]
    unlocked._update_intervals_list()
    want = unlocked.intervals[3]
    unlocked.selected_interval = want
    unlocked._update_intervals_list()
    sel = unlocked.intervals_tree.selection()
    assert len(sel) == 1
    assert unlocked._interval_row_map[sel[0]] is want
    assert unlocked.selected_interval is want, \
        "the re-assert must not walk into the DESELECT branch"


def test_the_suppression_flag_makes_the_select_handler_a_no_op(unlocked):
    """DR6's other half, pinned the only way a headless test can.

    `<<TreeviewSelect>>` is dispatched by a RUNNING event loop, so no
    headless pin can watch the refill's own `selection_set` walk into
    `_on_interval_tree_select`. What IS testable is the flag's contract:
    with it set, the handler returns without touching the selection -- and
    that handler's first branch is a DESELECT toggle, which is exactly what
    would eat the refill's re-selection in the live app.
    """
    unlocked.intervals[:] = [
        Interval(ts("00:01:00"), ts("00:03:00"), "sw", None, track="human")]
    unlocked._update_intervals_list()
    row = unlocked.intervals_tree.get_children()[0]
    want = unlocked._interval_row_map[row]
    unlocked.selected_interval = want
    unlocked.intervals_tree.selection_set(row)
    unlocked._suppress_tree_select = True
    try:
        unlocked._on_interval_tree_select(None)
    finally:
        unlocked._suppress_tree_select = False
    assert unlocked.selected_interval is want, \
        "with the flag set the handler must not toggle the selection off"
    # and WITHOUT the flag the same call is the user's deselect gesture
    unlocked._on_interval_tree_select(None)
    assert unlocked.selected_interval is None


def test_a_selection_the_filter_hides_is_named_on_the_status_bar(unlocked):
    unlocked.intervals[:] = agent_ivs() + [
        Interval(ts("00:01:00"), ts("00:03:00"), "sw", None, track="human")]
    unlocked.selected_interval = unlocked.intervals[0]
    unlocked._update_intervals_list()
    assert unlocked.intervals_tree.selection() == ()
    assert "Agent (C-MMAE)" in unlocked.status_var.get()
    assert unlocked.selected_interval is not None, \
        "it is still SELECTED -- Delete still works on it"


# ============================== 6. persistence, unchanged


def test_the_lane_view_state_round_trips_with_no_new_payload_key(app,
                                                                 tmp_path):
    app.intervals[:] = agent_ivs()
    app.track_by_id("agent").visible = False
    app._set_active_track("human", announce=False, repaint=False)
    app._save_autosave()
    import glob
    import json
    files = glob.glob(str(tmp_path / "*.json"))
    assert files
    payload = json.load(open(sorted(files)[0]))
    assert sorted(payload.keys()) == ["active_track", "intervals",
                                      "label_stats", "metadata", "tracks",
                                      "version"]
    assert payload["version"] == 2
    assert payload["active_track"] == "human"
    rows = {t["id"]: t for t in payload["tracks"]}
    assert rows["agent"]["visible"] is False
    assert rows["agent"]["locked"] is True


def test_a_refill_keeps_the_selection_once_Tk_DELIVERS_the_event(unlocked):
    """v2's BLOCKER PIN. DR6, through a REAL Tk event queue.

    ttk sends <<TreeviewSelect>> with Tcl_QueueEvent, so it arrives on a
    LATER turn of the loop, not inside selection_set. `root.update()`
    drains that queue without a mainloop, which is all this needs -- so
    this IS pinnable headlessly, and v1's claim that it was not is what
    let the defect through. With the suppression released synchronously
    the queued event walked into the handler's DESELECT branch:
    selected_interval None, "Interval deselected" on the status bar, and
    the next Delete popped "No Selection".
    """
    unlocked.intervals[:] = [
        Interval(ts("00:%02d:00" % (j * 6)), ts("00:%02d:30" % (j * 6)),
                 "sw", None, track="human") for j in range(5)]
    unlocked._update_intervals_list()
    want = unlocked.intervals[3]
    unlocked.selected_interval = want
    unlocked._update_intervals_list()
    unlocked.root.update()
    assert unlocked.selected_interval is want, \
        "the queued <<TreeviewSelect>> deselected what the refill kept"
    sel = unlocked.intervals_tree.selection()
    assert len(sel) == 1 and unlocked._interval_row_map[sel[0]] is want


def test_delete_still_works_after_a_repaint(unlocked):
    """The consequence, end to end: select, repaint, delete."""
    unlocked.intervals[:] = [
        Interval(ts("00:%02d:00" % (j * 6)), ts("00:%02d:30" % (j * 6)),
                 "sw", None, track="human") for j in range(5)]
    unlocked._update_plot()
    unlocked.root.update()
    unlocked.selected_interval = unlocked.intervals[2]
    unlocked._update_plot()
    unlocked.root.update()
    n0 = len(unlocked.intervals)
    unlocked._delete_interval()
    assert len(unlocked.intervals) == n0 - 1, \
        "the repaint dropped the selection, so Delete did nothing"


def test_the_clear_range_CONFIRMATION_is_never_built_on_a_locked_lane(
        app, _dialog_counter, monkeypatch):
    """v2 fold F6. v1 guarded the clear but not the dialog that OFFERS it:
    a locked active lane got the whole "31 will be deleted, 2 truncated"
    modal, the Yes button changed nothing, and on_confirm's own status line
    then overwrote the refusal with "Cleared intervals: " and an empty
    list. Nothing may be built at all."""
    import tkinter as tk
    built = []
    real = tk.Toplevel

    class Spy(real):
        def __init__(self, *a, **kw):
            built.append(1)
            real.__init__(self, *a, **kw)

    monkeypatch.setattr(tk, "Toplevel", Spy)
    app.intervals[:] = agent_ivs()
    app._set_active_track("agent", announce=False, repaint=False)
    app._show_clear_confirmation(ts("00:00:00"), ts("00:40:00"), "range")
    assert built == [], "a modal was offered for an edit that is refused"
    assert "clear range refused" in app.status_var.get()
    assert counts(app) == {"agent": 8}
    assert _dialog_counter == []


def test_the_ingest_refusal_names_real_data_gaps_as_a_cause():
    """The message, not the guard: on the user's own frame every tolerance
    from 5 min to 45 min is refused BECAUSE THE RECORD IS HOLEY, and the
    old text blamed a tolerance that was too small."""
    from chronotagger.core.ingest import intervals_from_column
    # A HOLEY RECORD, which is the case the old message could not name:
    # twelve 5-sample blocks at a 1-minute cadence, two hours apart, one
    # label per PAIR of blocks. Every run therefore spans exactly one real
    # hole -- 6 runs, 6 splits -- and a 30-minute tolerance is ABOVE the
    # cadence and BELOW the holes, so the tolerance is right and the data
    # is what is interrupted.
    idx = []
    values = []
    base = pd.Timestamp(DAY + "00:00:00")
    for block in range(12):
        idx += list(pd.date_range(base + pd.Timedelta(hours=2 * block),
                                  periods=5, freq="1min"))
        values += ["abcdef"[block // 2]] * 5
    with pytest.raises(ValueError) as e:
        intervals_from_column(pd.DatetimeIndex(idx), values, "t",
                              gap_tolerance=pd.Timedelta("30min"))
    msg = str(e.value)
    assert "would split 6 of 6 runs" in msg
    assert "TWO things look like this and the fix is different for each" \
        in msg
    assert "The RECORD IS HOLEY" in msg
    assert "the answer is an explicit, LARGER" in msg
