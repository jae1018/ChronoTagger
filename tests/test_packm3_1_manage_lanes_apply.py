"""Pack M3.1, item D -- THE BUTTON, THE OPENER, AND THE ONE-GESTURE APPLY.

A `Manage Lanes...` button in the sidebar's Lane box, under the
visible / locked checkboxes. It opens EVEN WHEN THE ACTIVE LANE IS
LOCKED -- unlike Manage Labels, whose lock guard refuses at the door --
because this box is where you unlock.

THE APPLY IS ONE UNDO STEP. Adds, renames, reorders, lock and
visibility changes, deletes WITH their intervals and column imports
WITH theirs all land inside ONE gesture, so ONE Ctrl+Z restores every
one of them -- a deleted lane comes back with its intervals -- and one
Ctrl+Y re-applies the lot. `GestureCommand` already snapshots the whole
lane table beside the interval list, so no new command class was needed.

ALL OR NOTHING: the staged end state is validated BEFORE it is
published, and a refusal changes nothing at all -- including a staged
import that fails inside the gesture, and including an end state that
would leave NO lane visible (a hide and a delete are two presses the
box allows one at a time).

A LANE MADE IN THE BOX IS COLOURED exactly as a lane a driver
declares: the box builds a bare `Track` and `Track.from_dict` copies
what it is given, so the colours are filled in from the palette at
this door.

THE UNSAVED-CHANGES RULE, which AMENDS Pack M3.0's DR2: EVERY lane-list
write marks the session modified -- the Manage Lanes OK, Ctrl+L, Ctrl+H
and un-hiding from the sidebar's Lane list. Pack M3.0 put the flag back
and a lock could be lost by closing the window. The amended Pack M3.0
pin is `tests/test_packm3_0_lane_undo.py
::test_a_lock_still_does_not_mark_the_session_modified`.
"""

import tkinter as tk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from chronotagger.core.models import Interval
from chronotagger.labeler.dialogs.manage_lanes import ManageLanesResult

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


@pytest.fixture(autouse=True)
def boxes(monkeypatch):
    """No box may reach the screen, whatever runner is used."""
    import tkinter.messagebox as mb
    import tkinter.filedialog as fdlg
    import tkinter.simpledialog as sdlg
    seen = []
    for name in ("showerror", "showinfo", "showwarning"):
        monkeypatch.setattr(
            mb, name,
            lambda *a, _n=name, **k: seen.append((_n,) + tuple(a[:1])))
    for name in ("askyesno", "askyesnocancel", "askokcancel",
                 "askretrycancel"):
        monkeypatch.setattr(
            mb, name,
            lambda *a, _n=name, **k: (seen.append((_n,) + tuple(a[:1]))
                                      or False))
    for name in ("askopenfilename", "asksaveasfilename", "askdirectory"):
        monkeypatch.setattr(fdlg, name, lambda *a, **k: "")
    for name in ("askstring", "askinteger", "askfloat"):
        monkeypatch.setattr(sdlg, name, lambda *a, **k: None)
    return seen


@pytest.fixture(autouse=True)
def _no_modal_blocking(monkeypatch):
    """Nothing in this module may block on a window.

    `wait_visibility` and `grab_set` HANG against a withdrawn parent,
    and `wait_window` would wait for a box nobody can click. The default
    here CLOSES the box, which is Cancel; a pin that wants to drive one
    replaces `wait_window` itself.
    """
    monkeypatch.setattr(tk.Misc, "wait_visibility",
                        lambda self, *a, **k: None)
    monkeypatch.setattr(tk.Misc, "grab_set", lambda self, *a, **k: None)

    def _dont_wait(self, win=None, *a, **k):
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass
    monkeypatch.setattr(tk.Misc, "wait_window", _dont_wait)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def ts(hhmmss):
    return pd.Timestamp(DAY + hhmmss)


def frame():
    idx = pd.date_range(DAY + "00:00:00", periods=120, freq="30s")
    return pd.DataFrame({"a": np.linspace(0.0, 1.0, len(idx))}, index=idx)


def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["a"])
    axs["panel2"].plot(df.index, df["a"])


@pytest.fixture
def app(tmp_path):
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(
        df=frame(), plot_fn=plot_fn, layout_spec=dict(LAYOUT),
        window=pd.Timedelta("60min"), autosave_folder=str(tmp_path),
        tracks=[dict(REGION), dict(WAKE), dict(AGENT)])
    lbl._build_gui()
    lbl.root.withdraw()
    lbl.intervals.append(Interval(ts("00:05:00"), ts("00:10:00"), "sw",
                                  None, track="region"))
    lbl.intervals.append(Interval(ts("00:20:00"), ts("00:25:00"), "msh",
                                  None, track="region"))
    lbl.intervals.append(Interval(ts("00:40:00"), ts("00:45:00"), "umbra",
                                  None, track="wake"))
    lbl._update_plot()
    for _ in range(2):
        lbl.root.update()
    yield lbl
    try:
        lbl.root.destroy()
    except Exception:
        pass


def ids(app):
    return [t.id for t in app.tracks]


def result_from(app, **kw):
    """A result naming the app's CURRENT table, plus the edits in kw."""
    lanes = kw.pop("lanes", None)
    if lanes is None:
        lanes = [t.to_dict() for t in app.tracks]
    return ManageLanesResult(lanes=lanes, **kw)


def walk(w):
    out = [w]
    i = 0
    while i < len(out):
        try:
            out.extend(out[i].winfo_children())
        except Exception:
            pass
        i += 1
    return out


# ==================================================== 1. the button


def test_the_sidebar_carries_a_manage_lanes_button(app):
    assert hasattr(app, "manage_lanes_btn")
    assert str(app.manage_lanes_btn.cget("text")) == "Manage Lanes..."
    assert app.manage_lanes_btn.winfo_exists()


def test_the_button_sits_in_the_lane_frame(app):
    parent = app.manage_lanes_btn.nametowidget(
        app.manage_lanes_btn.winfo_parent())
    assert str(parent.cget("text")) == "Lane"
    texts = []
    for child in walk(parent):
        try:
            texts.append(str(child.cget("text")))
        except Exception:
            pass
    assert "visible" in texts and "locked" in texts
    assert "Ctrl+Up / Ctrl+Down switch lanes" in texts


def test_pressing_the_button_really_opens_the_box(app, monkeypatch):
    """The command is the opener, bound at build time."""
    seen = {}

    def waiter(self, win=None, *a, **k):
        seen["title"] = win.title()
        win._on_cancel()
    monkeypatch.setattr(tk.Misc, "wait_window", waiter)
    app.manage_lanes_btn.invoke()
    assert seen.get("title") == "Manage Lanes"


# ==================================================== 2. the opener


def test_the_opener_opens_on_a_LOCKED_active_lane(app, monkeypatch):
    """This is where you unlock; a lock guard would lock the door."""
    app._set_active_track("agent")
    assert app.active_track.locked is True
    seen = {}

    def waiter(self, win=None, *a, **k):
        seen["title"] = win.title()
        seen["rows"] = [win.tree.item(i, "values")
                        for i in win.tree.get_children()]
        win._on_cancel()
    monkeypatch.setattr(tk.Misc, "wait_window", waiter)
    app._open_manage_lanes()
    assert seen.get("title") == "Manage Lanes"
    assert [r[1] for r in seen["rows"]] == ["region", "wake", "agent"]


def test_the_opener_hands_the_box_the_interval_counts(app, monkeypatch):
    seen = {}

    def waiter(self, win=None, *a, **k):
        seen["counts"] = dict(win._counts)
        win._on_cancel()
    monkeypatch.setattr(tk.Misc, "wait_window", waiter)
    app._open_manage_lanes()
    assert seen["counts"] == {"region": 2, "wake": 1, "agent": 0}


def test_a_cancelled_box_changes_nothing(app, monkeypatch):
    before = ([t.to_dict() for t in app.tracks], len(app.intervals))
    n0 = len(app.undo_stack)
    monkeypatch.setattr(tk.Misc, "wait_window",
                        lambda self, win=None, *a, **k: win._on_cancel())
    app._open_manage_lanes()
    assert ([t.to_dict() for t in app.tracks],
            len(app.intervals)) == before
    assert len(app.undo_stack) == n0


# ==================================================== 3. the apply


def test_the_whole_ok_is_ONE_undo_step(app):
    """Add, rename, reorder, hide and delete-with-intervals: one press."""
    lanes = [dict(WAKE), dict(AGENT),
             {"id": "fresh", "name": "Fresh", "classes": ["p", "q"]}]
    lanes[0]["name"] = "Wake 2"
    lanes[1]["visible"] = False
    n0 = len(app.undo_stack)
    assert app._apply_manage_lanes_result(result_from(
        app, lanes=lanes, added=["fresh"], deleted=["region"],
        deleted_intervals=2, renamed={"wake": "Wake 2"},
        reordered=True, flagged=True)) is True
    assert len(app.undo_stack) - n0 == 1
    assert ids(app) == ["wake", "agent", "fresh"]
    assert [t.order for t in app.tracks] == [0, 1, 2]
    assert app.track_by_id("wake").name == "Wake 2"
    assert len(app.intervals) == 1
    app._undo()
    assert ids(app) == ["region", "wake", "agent"]
    assert len(app.intervals) == 3, \
        "a deleted lane comes back WITH its intervals"
    assert app.track_by_id("wake").name == "Wake (umbra)"
    assert app.status_var.get() == "Undo: manage lanes"
    app._redo()
    assert ids(app) == ["wake", "agent", "fresh"]
    assert len(app.intervals) == 1
    assert app.status_var.get() == "Redo: manage lanes"


def test_a_delete_never_leaves_a_stray_interval(app):
    from chronotagger.core.tracks import stray_tracks
    app._apply_manage_lanes_result(result_from(
        app, lanes=[dict(WAKE), dict(AGENT)], deleted=["region"],
        deleted_intervals=2))
    assert stray_tracks(app.intervals, app.tracks) == []


def test_deleting_the_active_lane_lands_on_the_first_VISIBLE_lane(app):
    assert app.active_track_id == "region"
    lanes = [dict(WAKE), dict(AGENT)]
    lanes[0]["visible"] = False
    app._apply_manage_lanes_result(result_from(
        app, lanes=lanes, deleted=["region"], deleted_intervals=2,
        flagged=True))
    assert app.active_track_id == "agent"


def test_hiding_the_active_lane_moves_it(app):
    lanes = [t.to_dict() for t in app.tracks]
    lanes[0]["visible"] = False
    app._apply_manage_lanes_result(result_from(app, lanes=lanes,
                                               flagged=True))
    assert app.active_track_id == "wake", \
        "a gesture may not write to a lane the user cannot see"


def test_a_moved_active_lane_is_in_the_SAME_sentence(app):
    """Pack M2.6's doctrine: one sentence, not two.

    `_reconcile_active_track` writes its own line about the move and
    the OK's own line would write straight over it, so the user would
    never learn that the next Add lands somewhere else.
    """
    app._apply_manage_lanes_result(result_from(
        app, lanes=[dict(WAKE), dict(AGENT)], deleted=["region"],
        deleted_intervals=2))
    assert app.status_var.get() == (
        "Lanes updated: 1 deleted (2 intervals) -- active lane is now "
        "'Wake (umbra)'")


def test_an_ok_that_does_not_move_the_active_lane_says_nothing_about_it(
        app):
    lanes = [t.to_dict() for t in app.tracks] + [
        {"id": "fresh", "name": "Fresh", "classes": ["p"]}]
    app._apply_manage_lanes_result(result_from(app, lanes=lanes,
                                               added=["fresh"]))
    assert app.status_var.get() == "Lanes updated: 1 added"


def test_the_class_dropdown_follows(app):
    lanes = [dict(WAKE), dict(AGENT)]
    app._apply_manage_lanes_result(result_from(
        app, lanes=lanes, deleted=["region"], deleted_intervals=2))
    assert list(app.class_combo["values"]) == ["umbra"]
    assert app.current_class_var.get() == "umbra"


def test_the_sidebar_lane_list_refills(app):
    lanes = [t.to_dict() for t in app.tracks] + [
        {"id": "fresh", "name": "Fresh", "classes": ["p"]}]
    app._apply_manage_lanes_result(result_from(app, lanes=lanes,
                                               added=["fresh"]))
    assert list(app.lane_combo["values"])[-1] == "Fresh"


def test_the_strip_resizes_with_the_new_lane_count(app):
    from chronotagger.core.lanes import labels_row_height
    pane = app.panes[0]
    row = pane.strip_ax.get_subplotspec().rowspan.start
    lanes = [t.to_dict() for t in app.tracks] + [
        {"id": "fresh", "name": "Fresh", "classes": ["p"]}]
    app._apply_manage_lanes_result(result_from(app, lanes=lanes,
                                               added=["fresh"]))
    hrs = list(pane.strip_ax.get_subplotspec().get_gridspec()
               .get_height_ratios())
    assert hrs[row] == pytest.approx(labels_row_height(4))


def test_a_lane_the_box_made_is_coloured_from_the_palette(app):
    """THE COLOUR RULE AT THIS DOOR, not only at the model's.

    The box builds a bare `Track` and `Track.from_dict` copies exactly
    what it is given, so without this the new lane arrives with NO
    colours and every band on it, its legend entry and its focus ring
    paint the `#cccccc` fallback -- grey, beside a driver's coloured
    lanes. A lane made in the box is coloured exactly as a lane a
    driver declares.
    """
    lanes = [t.to_dict() for t in app.tracks] + [
        {"id": "fresh", "name": "Fresh", "classes": ["p", "q"]}]
    assert app._apply_manage_lanes_result(result_from(
        app, lanes=lanes, added=["fresh"])) is True
    assert app.track_by_id("fresh").class_colors == {
        "p": app.DEFAULT_COLORS[0], "q": app.DEFAULT_COLORS[1]}


def test_a_staged_import_is_coloured_too(app):
    """The other door into a new lane: `From column...`."""
    app._ingest_staged_column = lambda spec: 0
    lanes = [t.to_dict() for t in app.tracks] + [
        {"id": "imp", "name": "Imported", "classes": ["msh", "sw"]}]
    assert app._apply_manage_lanes_result(result_from(
        app, lanes=lanes, added=["imp"],
        imports=[{"id": "imp", "column": "a"}])) is True
    assert app.track_by_id("imp").class_colors == {
        "msh": app.DEFAULT_COLORS[0], "sw": app.DEFAULT_COLORS[1]}


def test_colours_a_lane_ALREADY_HAS_are_left_alone(app):
    """Only an EMPTY colour map is filled in."""
    before = dict(app.track_by_id("region").class_colors)
    lanes = [t.to_dict() for t in app.tracks] + [
        {"id": "fresh", "name": "Fresh", "classes": ["p"]}]
    app._apply_manage_lanes_result(result_from(app, lanes=lanes,
                                               added=["fresh"]))
    assert app.track_by_id("region").class_colors == before


def test_the_bar_names_what_happened(app):
    app._apply_manage_lanes_result(result_from(
        app, lanes=[dict(WAKE), dict(AGENT),
                    {"id": "fresh", "name": "Fresh", "classes": ["p"]}],
        added=["fresh"], deleted=["region"], deleted_intervals=41))
    assert app.status_var.get().startswith(
        "Lanes updated: 1 added, 1 deleted (2 intervals)")


def test_one_interval_is_singular_in_the_bar(app):
    del app.intervals[1:]
    app._apply_manage_lanes_result(result_from(
        app, lanes=[dict(WAKE), dict(AGENT)], deleted=["region"]))
    assert "1 deleted (1 interval)" in app.status_var.get()


# ==================================================== 4. all or nothing


def test_an_ok_that_changes_nothing_pushes_nothing_and_says_nothing(app):
    app.status_var.set("BEFORE")
    app.modified = False
    n0 = len(app.undo_stack)
    assert app._apply_manage_lanes_result(result_from(app)) is False
    assert len(app.undo_stack) == n0
    assert app.status_var.get() == "BEFORE"
    assert app.modified is False


def test_a_table_rule_break_publishes_nothing(app):
    before = [t.to_dict() for t in app.tracks]
    n0 = len(app.undo_stack)
    assert app._apply_manage_lanes_result(result_from(
        app, lanes=[{"id": "a", "name": "A", "classes": []}],
        added=["a"], deleted=["region", "wake", "agent"])) is False
    assert [t.to_dict() for t in app.tracks] == before
    assert len(app.undo_stack) == n0
    assert "changed nothing" in app.status_var.get()


def test_an_end_state_with_no_visible_lane_is_refused(app):
    """Two presses the box allows ONE AT A TIME.

    `Show/Hide` refuses to hide the last visible lane one press at a
    time, but a hide and a DELETE are two presses it allows separately
    that together leave every lane hidden: a blank Labels strip, and a
    gesture still writing to a lane the user cannot see. The rule is
    checked on the staged END STATE, where a delete goes too.
    """
    before = [t.to_dict() for t in app.tracks]
    n0 = len(app.undo_stack)
    lanes = [dict(WAKE)]
    lanes[0]["visible"] = False
    assert app._apply_manage_lanes_result(result_from(
        app, lanes=lanes, deleted=["region", "agent"],
        deleted_intervals=2, flagged=True)) is False
    assert [t.to_dict() for t in app.tracks] == before
    assert len(app.intervals) == 3
    assert len(app.undo_stack) == n0
    assert app.status_var.get() == (
        "Manage Lanes changed nothing: that would leave no lane visible")


def test_a_staged_import_that_fails_changes_nothing_and_says_why(app):
    """ALL OR NOTHING, INCLUDING THE STAGED IMPORTS.

    `_gesture` is transactional and rolls the model back on its own,
    but the exception used to travel on out of the Tk button callback:
    a traceback on stderr, nothing on the bar, and the user left
    looking at the line that was there before.
    """
    def _raise(spec):
        raise ValueError("no column 'nope' in the frame")
    app._ingest_staged_column = _raise
    app.status_var.set("BEFORE")
    before = ([t.to_dict() for t in app.tracks], len(app.intervals))
    n0 = len(app.undo_stack)
    lanes = [t.to_dict() for t in app.tracks] + [
        {"id": "nope", "name": "Nope", "classes": ["x"]}]
    assert app._apply_manage_lanes_result(result_from(
        app, lanes=lanes, added=["nope"],
        imports=[{"id": "nope", "column": "nope"}])) is False
    assert ([t.to_dict() for t in app.tracks],
            len(app.intervals)) == before
    assert len(app.undo_stack) == n0
    assert app.status_var.get() == (
        "Manage Lanes changed nothing: no column 'nope' in the frame")


def test_a_duplicate_display_name_the_box_made_is_refused(app):
    lanes = [t.to_dict() for t in app.tracks]
    lanes.append({"id": "twin", "name": "region (HUMAN)",
                  "classes": ["x"]})
    assert app._apply_manage_lanes_result(result_from(
        app, lanes=lanes, added=["twin"])) is False
    assert ids(app) == ["region", "wake", "agent"]
    assert "two lanes are called" in app.status_var.get()


def test_a_duplicate_name_that_was_ALREADY_THERE_is_not_refused(app):
    """The fence: existing files and driver tables keep working."""
    app.tracks[1].name = "Region (human)"
    lanes = [t.to_dict() for t in app.tracks]
    lanes.append({"id": "fresh", "name": "Fresh", "classes": ["x"]})
    assert app._apply_manage_lanes_result(result_from(
        app, lanes=lanes, added=["fresh"])) is True
    assert ids(app) == ["region", "wake", "agent", "fresh"]


# ==================================================== 5. modified


def test_the_manage_lanes_ok_marks_the_session_modified(app):
    app.modified = False
    app._apply_manage_lanes_result(result_from(
        app, lanes=[t.to_dict() for t in app.tracks] + [
            {"id": "fresh", "name": "Fresh", "classes": ["p"]}],
        added=["fresh"]))
    assert app.modified is True


def test_ctrl_l_ctrl_h_and_the_lane_list_mark_the_session_modified(app):
    """The AMENDMENT to Pack M3.0's DR2, in one pin.

    Pack M3.0 put `modified` back after a lane write, so a lock survived
    only until the window closed and nothing asked. J.E. ruled that
    every lane-list write is unsaved work.
    """
    app.modified = False
    app._toggle_active_lane_locked()
    assert app.modified is True, "Ctrl+L"

    app.modified = False
    app._toggle_active_lane_visible()
    assert app.modified is True, "Ctrl+H"

    app.modified = False
    app.lane_var.set("Region (human) (hidden)")
    app.lane_combo.current(0)
    app._on_lane_combo_change()
    assert app.modified is True, "un-hiding from the sidebar's Lane list"

    # And PART B's six edit methods, which go through the same wrapper.
    app.modified = False
    app._lane_add("Fourth lane", ["p", "q"])
    assert app.modified is True, "_lane_add"
    app.modified = False
    app._lane_rename("wake", "Wake 2")
    assert app.modified is True, "_lane_rename"


def test_a_refused_hide_still_marks_nothing(app):
    """A refusal writes nothing, so there is nothing to save."""
    app._lane_set_visible("wake", False)
    app._lane_set_visible("agent", False)
    app.modified = False
    app._toggle_active_lane_visible()
    assert app.modified is False
    assert app.track_by_id("region").visible is True
