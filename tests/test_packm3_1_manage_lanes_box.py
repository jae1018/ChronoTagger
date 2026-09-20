"""Pack M3.1, item C -- THE MANAGE LANES BOX ITSELF.

A STANDALONE dialog, modelled on LabelManagerDialog: working copies
staged inside it, a result object, OK / Cancel, WM_DELETE_WINDOW =
Cancel, centred over the window that opened it, modal through transient
+ grab_set, and the CALLER waits with `wait_window`.

IT NEVER TOUCHES THE APP. It is handed the lane table (copied), a
per-lane interval count and two OPTIONAL callables, and returns the
staged END STATE plus what was deleted and what was imported. These pins
build it with plain lists and two plain functions and no labeler at all
-- which is also the shape Pack M3.2's wizard will use.

Every button is INVOKED through the widget, not called as a method, and
every sub-box is answered through a patched `wait_window`, because the
thing that must be pinned is that the buttons are wired to what they
say they do.

HEADLESS SAFETY. `wait_visibility` and `grab_set` HANG against a
withdrawn parent, so both are neutered here, and every message box is
stubbed by this module itself -- GitHub CI runs plain `pytest tests/`
with no global stub.
"""

import tkinter as tk

import pytest

from chronotagger.core.tracks import Track
from chronotagger.labeler.dialogs.manage_lanes import (ManageLanesDialog,
                                                       ManageLanesResult,
                                                       split_classes)

LANES = [
    {"id": "region", "name": "Region (human)",
     "classes": ["sw", "msh", "UNKNOWN"], "order": 0},
    {"id": "wake", "name": "Wake (umbra)", "classes": ["umbra"],
     "order": 1},
    {"id": "agent", "name": "Agent (C-MMAE)", "classes": ["0", "1"],
     "locked": True, "order": 2},
]
COUNTS = {"region": 41, "wake": 0, "agent": 256}


class _Seen(list):
    """A list of the boxes that were asked for, with the answer on it."""

    answer = None


@pytest.fixture(autouse=True)
def boxes(monkeypatch):
    """Every box is RECORDED, never shown. Answers come from `answer`."""
    import tkinter.messagebox as mb
    import tkinter.filedialog as fdlg
    import tkinter.simpledialog as sdlg
    seen = _Seen()
    answer = {"yes": True}
    seen.answer = answer
    for name in ("showerror", "showinfo", "showwarning"):
        monkeypatch.setattr(
            mb, name,
            lambda *a, _n=name, **k: seen.append((_n,) + tuple(a[:1])))
    for name in ("askyesno", "askyesnocancel", "askokcancel",
                 "askretrycancel"):
        monkeypatch.setattr(
            mb, name,
            lambda *a, _n=name, **k: (seen.append((_n,) + tuple(a[:2]))
                                      or answer["yes"]))
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
    here CLOSES the box, which is Cancel; `answer_next` replaces it.
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


@pytest.fixture(scope="module")
def _root():
    """ONE Tk interpreter for the whole module.

    Creating and destroying one per test exhausted something in Tcl on
    this machine -- `Can't find a usable init.tcl` after about thirty
    roots -- and the box needs a parent, not a fresh interpreter.
    """
    r = tk.Tk()
    r.withdraw()
    yield r
    try:
        r.destroy()
    except Exception:
        pass


@pytest.fixture
def root(_root):
    """The shared root, swept clean of any box a test forgot."""
    yield _root
    for child in list(_root.winfo_children()):
        try:
            child.destroy()
        except Exception:
            pass


def pump(widget, n=4):
    for _ in range(n):
        try:
            widget.update()
        except Exception:
            return


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


def button(widget, text):
    for child in walk(widget):
        try:
            if child.winfo_class() in ("TButton", "Button") and \
                    str(child.cget("text")) == text:
                return child
        except Exception:
            continue
    return None


def press(widget, text):
    b = button(widget, text)
    assert b is not None, "no button %r" % (text,)
    b.invoke()


def answer_next(monkeypatch, root, fill):
    """Replace wait_window so the next sub-box is driven and closed."""
    seen = []

    def waiter(self, win=None, *a, **k):
        seen.append(win)
        pump(root)
        fill(win)
    monkeypatch.setattr(tk.Misc, "wait_window", waiter)
    return seen


def no_wait(monkeypatch):
    monkeypatch.setattr(tk.Misc, "wait_window",
                        lambda self, win=None, *a, **k: None)


def open_box(root, lanes=None, counts=None, **kw):
    dlg = ManageLanesDialog(parent=root, lanes=lanes or LANES,
                            interval_counts=counts or COUNTS, **kw)
    pump(root)
    return dlg


def table(dlg):
    return [dlg.tree.item(i, "values") for i in dlg.tree.get_children()]


# ==================================================== 1. what it shows


def test_the_window_is_called_manage_lanes(root):
    dlg = open_box(root)
    assert dlg.title() == "Manage Lanes"
    dlg.destroy()


def test_the_six_ruled_columns_in_order(root):
    dlg = open_box(root)
    assert list(dlg.tree.cget("columns")) == [
        "Name", "Id", "Classes", "Intervals", "Locked", "Visible"]
    assert table(dlg) == [
        ("Region (human)", "region", "3", "41", "no", "yes"),
        ("Wake (umbra)", "wake", "1", "0", "no", "yes"),
        ("Agent (C-MMAE)", "agent", "2", "256", "yes", "yes"),
    ]
    dlg.destroy()


def test_the_ruled_buttons_are_there(root):
    dlg = open_box(root)
    for text in ("Add...", "Rename...", "Move up", "Move down",
                 "Lock/Unlock", "Show/Hide", "Delete...", "Cancel", "OK"):
        assert button(dlg, text) is not None, text
    assert button(dlg, "From column...") is None, \
        "the column button is built only when a column callable arrives"
    dlg.destroy()


def test_the_column_button_appears_with_its_callable(root):
    dlg = open_box(root, columns_fn=lambda: [], preview_fn=lambda *a: {})
    assert button(dlg, "From column...") is not None
    dlg.destroy()


def test_it_copies_the_rows_it_is_given(root):
    rows = [Track.from_dict(d) for d in LANES]
    dlg = open_box(root, lanes=rows)
    dlg._rows[0].name = "changed inside the box"
    assert rows[0].name == "Region (human)", \
        "the caller's rows are not the box's rows"
    dlg.destroy()


def test_it_needs_no_labeler_and_no_counts(root):
    """The shape Pack M3.2's wizard will construct."""
    dlg = ManageLanesDialog(parent=root, lanes=[dict(LANES[0])],
                            interval_counts=None)
    pump(root)
    assert table(dlg) == [("Region (human)", "region", "3", "0", "no",
                           "yes")]
    dlg.destroy()


# ==================================================== 2. the actions


def test_add_stages_a_lane_with_an_id_made_from_the_name(
        root, monkeypatch):
    dlg = open_box(root)

    def fill(win):
        win.name_var.set("Wake (umbra) 2")
        pump(root)
        assert win.id_var.get() == "wake_umbra_2", \
            "the Id follows the Name until the user edits it"
        win.classes_var.set(" a , b ,, a ")
        press(win, "OK")
        assert win.result is None, "a duplicate class is refused"
        win.classes_var.set(" a , b ,, c ")
        press(win, "OK")
    answer_next(monkeypatch, root, fill)
    press(dlg, "Add...")
    assert [r[1] for r in table(dlg)] == ["region", "wake", "agent",
                                          "wake_umbra_2"]
    assert table(dlg)[-1] == ("Wake (umbra) 2", "wake_umbra_2", "3", "0",
                              "no", "yes")
    dlg.destroy()


def test_the_id_stops_following_once_it_is_edited(root, monkeypatch):
    dlg = open_box(root)

    def fill(win):
        win.name_var.set("First")
        pump(root)
        assert win.id_var.get() == "first"
        win.id_var.set("mine")
        win._id_edited()
        win.name_var.set("Second")
        pump(root)
        assert win.id_var.get() == "mine"
        win.classes_var.set("x")
        press(win, "OK")
    answer_next(monkeypatch, root, fill)
    press(dlg, "Add...")
    assert table(dlg)[-1][1] == "mine"
    dlg.destroy()


def test_add_refuses_an_empty_name_a_duplicate_name_and_a_bad_id(
        root, monkeypatch, boxes):
    dlg = open_box(root)

    def fill(win):
        press(win, "OK")
        win.name_var.set("Wake (umbra)")
        win.classes_var.set("x")
        press(win, "OK")
        win.name_var.set("Fine")
        win.id_var.set("no/slashes")
        win._id_edited()
        press(win, "OK")
        win.id_var.set("region")
        press(win, "OK")
        assert win.result is None
        press(win, "Cancel")
    answer_next(monkeypatch, root, fill)
    press(dlg, "Add...")
    kinds = [b[1] for b in boxes]
    assert kinds == ["Invalid name", "Duplicate name", "Invalid id",
                     "Duplicate id"]
    assert [r[1] for r in table(dlg)] == ["region", "wake", "agent"]
    dlg.destroy()


def test_rename_changes_the_name_and_never_the_id(root, monkeypatch):
    dlg = open_box(root)
    dlg.tree.selection_set("wake")
    pump(root)

    def fill(win):
        win.name_var.set("Wake 2")
        press(win, "OK")
    answer_next(monkeypatch, root, fill)
    press(dlg, "Rename...")
    assert table(dlg)[1] == ("Wake 2", "wake", "1", "0", "no", "yes")
    dlg.destroy()


def test_rename_refuses_empty_and_duplicate(root, monkeypatch, boxes):
    dlg = open_box(root)
    dlg.tree.selection_set("wake")
    pump(root)

    def fill(win):
        win.name_var.set("   ")
        press(win, "OK")
        win.name_var.set("region (HUMAN)")
        press(win, "OK")
        assert win.result is None
        press(win, "Cancel")
    answer_next(monkeypatch, root, fill)
    press(dlg, "Rename...")
    assert [b[1] for b in boxes] == ["Invalid name", "Duplicate name"]
    assert table(dlg)[1][0] == "Wake (umbra)"
    dlg.destroy()


def test_move_up_and_down_do_not_wrap(root, monkeypatch):
    no_wait(monkeypatch)
    dlg = open_box(root)
    dlg.tree.selection_set("region")
    pump(root)
    press(dlg, "Move up")
    assert [r[1] for r in table(dlg)] == ["region", "wake", "agent"]
    dlg.tree.selection_set("agent")
    pump(root)
    press(dlg, "Move up")
    assert [r[1] for r in table(dlg)] == ["region", "agent", "wake"]
    press(dlg, "Move down")
    assert [r[1] for r in table(dlg)] == ["region", "wake", "agent"]
    dlg.destroy()


def test_lock_unlock_and_show_hide_toggle_the_selected_lane(
        root, monkeypatch):
    no_wait(monkeypatch)
    dlg = open_box(root)
    dlg.tree.selection_set("agent")
    pump(root)
    press(dlg, "Lock/Unlock")
    assert table(dlg)[2][4] == "no"
    press(dlg, "Show/Hide")
    assert table(dlg)[2][5] == "no"
    press(dlg, "Show/Hide")
    assert table(dlg)[2][5] == "yes"
    dlg.destroy()


def test_hiding_the_last_visible_lane_is_refused(root, monkeypatch,
                                                 boxes):
    no_wait(monkeypatch)
    dlg = open_box(root)
    for tid in ("wake", "agent"):
        dlg.tree.selection_set(tid)
        pump(root)
        press(dlg, "Show/Hide")
    dlg.tree.selection_set("region")
    pump(root)
    press(dlg, "Show/Hide")
    assert table(dlg)[0][5] == "yes"
    assert boxes[-1][0] == "showwarning"
    assert boxes[-1][1] == "Cannot hide"
    dlg.destroy()


# ==================================================== 3. delete


def test_delete_confirms_with_the_count(root, monkeypatch, boxes):
    no_wait(monkeypatch)
    dlg = open_box(root)
    dlg.tree.selection_set("region")
    pump(root)
    press(dlg, "Delete...")
    assert boxes[-1] == ("askyesno", "Delete lane",
                         "Delete 'Region (human)' and its 41 intervals?")
    assert [r[1] for r in table(dlg)] == ["wake", "agent"]
    dlg.destroy()


def test_one_interval_is_singular_and_none_drops_the_clause(
        root, monkeypatch, boxes):
    no_wait(monkeypatch)
    dlg = open_box(root, counts={"region": 1, "wake": 0, "agent": 2})
    dlg.tree.selection_set("region")
    pump(root)
    press(dlg, "Delete...")
    assert boxes[-1][2] == "Delete 'Region (human)' and its 1 interval?"
    dlg.tree.selection_set("wake")
    pump(root)
    press(dlg, "Delete...")
    assert boxes[-1][2] == "Delete 'Wake (umbra)'?"
    dlg.destroy()


def test_answering_no_deletes_nothing(root, monkeypatch, boxes):
    no_wait(monkeypatch)
    boxes.answer["yes"] = False
    dlg = open_box(root)
    dlg.tree.selection_set("region")
    pump(root)
    press(dlg, "Delete...")
    assert [r[1] for r in table(dlg)] == ["region", "wake", "agent"]
    dlg.destroy()


def test_delete_refuses_a_locked_lane_and_the_last_lane(
        root, monkeypatch, boxes):
    no_wait(monkeypatch)
    dlg = open_box(root)
    dlg.tree.selection_set("agent")
    pump(root)
    press(dlg, "Delete...")
    assert boxes[-1][:2] == ("showwarning", "Cannot delete")
    assert [r[1] for r in table(dlg)] == ["region", "wake", "agent"]
    dlg.tree.selection_set("region")
    pump(root)
    press(dlg, "Delete...")
    dlg.tree.selection_set("wake")
    pump(root)
    press(dlg, "Delete...")
    assert [r[1] for r in table(dlg)] == ["agent"]
    dlg.tree.selection_set("agent")
    pump(root)
    press(dlg, "Delete...")
    assert [r[1] for r in table(dlg)] == ["agent"]
    dlg.destroy()


# ==================================================== 4. OK / Cancel


def test_ok_reports_the_staged_end_state(root, monkeypatch):
    dlg = open_box(root)

    def fill(win):
        win.name_var.set("Fresh")
        win.classes_var.set("x,y")
        press(win, "OK")
    answer_next(monkeypatch, root, fill)
    press(dlg, "Add...")
    no_wait(monkeypatch)
    dlg.tree.selection_set("region")
    pump(root)
    press(dlg, "Delete...")
    press(dlg, "OK")
    res = dlg.result
    assert isinstance(res, ManageLanesResult)
    assert res.added == ["fresh"]
    assert res.deleted == ["region"]
    assert res.deleted_intervals == 41
    assert [d["id"] for d in res.lanes] == ["wake", "agent", "fresh"]
    assert [d["order"] for d in res.lanes] == [0, 1, 2]
    assert res.is_empty() is False


def test_a_lane_added_then_deleted_is_in_neither_list(root, monkeypatch):
    dlg = open_box(root)

    def fill(win):
        win.name_var.set("Ghost")
        win.classes_var.set("x")
        press(win, "OK")
    answer_next(monkeypatch, root, fill)
    press(dlg, "Add...")
    no_wait(monkeypatch)
    dlg.tree.selection_set("ghost")
    pump(root)
    press(dlg, "Delete...")
    press(dlg, "OK")
    assert dlg.result.added == [] and dlg.result.deleted == []
    assert dlg.result.is_empty() is True


def test_ok_with_no_change_is_empty(root, monkeypatch):
    no_wait(monkeypatch)
    dlg = open_box(root)
    press(dlg, "OK")
    assert dlg.result.is_empty() is True
    assert dlg.result.added == [] and dlg.result.deleted == []
    assert dlg.result.renamed == {} and dlg.result.imports == []
    assert dlg.result.reordered is False and dlg.result.flagged is False


def test_a_reorder_alone_is_not_empty(root, monkeypatch):
    no_wait(monkeypatch)
    dlg = open_box(root)
    dlg.tree.selection_set("agent")
    pump(root)
    press(dlg, "Move up")
    press(dlg, "OK")
    assert dlg.result.reordered is True
    assert dlg.result.is_empty() is False


def test_a_flag_alone_is_not_empty(root, monkeypatch):
    no_wait(monkeypatch)
    dlg = open_box(root)
    dlg.tree.selection_set("agent")
    pump(root)
    press(dlg, "Lock/Unlock")
    press(dlg, "OK")
    assert dlg.result.flagged is True
    assert dlg.result.is_empty() is False


def test_cancel_discards_everything_including_a_staged_delete(
        root, monkeypatch):
    dlg = open_box(root)

    def fill(win):
        win.name_var.set("Never")
        win.classes_var.set("x")
        press(win, "OK")
    answer_next(monkeypatch, root, fill)
    press(dlg, "Add...")
    no_wait(monkeypatch)
    dlg.tree.selection_set("region")
    pump(root)
    press(dlg, "Delete...")
    press(dlg, "Cancel")
    assert dlg.result is None


def test_closing_the_window_is_cancel(root, monkeypatch):
    no_wait(monkeypatch)
    dlg = open_box(root)
    assert dlg.protocol("WM_DELETE_WINDOW")
    dlg._on_cancel()
    assert dlg.result is None


def test_the_box_is_centred_over_its_parent(root, monkeypatch):
    no_wait(monkeypatch)
    seen = {}
    import chronotagger.labeler.dialogs._placement as P
    real = P.center_on_parent

    def spy(dialog, parent, width=None, height=None):
        seen["parent"] = parent
        return real(dialog, parent, width, height)
    monkeypatch.setattr(P, "center_on_parent", spy)
    dlg = open_box(root)
    assert seen.get("parent") is root
    dlg.destroy()


# ==================================================== 5. the small rules


@pytest.mark.parametrize("text,want", [
    ("a,b,c", ["a", "b", "c"]),
    (" a , b ,, c ", ["a", "b", "c"]),
    ("one", ["one"]),
])
def test_the_class_list_is_trimmed_and_empties_dropped(text, want):
    got, err = split_classes(text)
    assert (got, err) == (want, "")


@pytest.mark.parametrize("text", ["", "   ", ",,,"])
def test_an_empty_class_list_is_refused(text):
    got, err = split_classes(text)
    assert got == [] and "at least one class" in err


def test_a_duplicate_class_is_refused():
    got, err = split_classes("a, A")
    assert got == [] and "listed twice" in err
