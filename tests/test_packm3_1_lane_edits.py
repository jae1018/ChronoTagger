"""Pack M3.1, item B -- THE LANE-LIST EDIT FUNCTIONS, AND ONE VALIDATOR.

There were NONE at base, measured: no add, no rename, no reorder, no
delete anywhere in the tree. A lane could only be DECLARED in a driver
script, and the only lane that ever appeared after launch came from
`add_track_from_column`, which nothing on screen called.

And the three table rules -- not empty, unique ids, at least one class
per lane -- were written TWICE, in `_build_track_table` (app.py) and in
`validate_track_table` (io_export.py). This pack adds a third door, so
they are factored into `core.tracks.check_track_table`, which keeps BOTH
voices byte for byte: `what` selects the file's voice, its absence the
constructor's.

THE NEW RULE, and its fence. Display names must be unique --
case-insensitively, whitespace-trimmed -- FOR EDITS MADE THROUGH THE
MANAGE LANES BOX ONLY. Files and driver tables that already carry two
lanes with one name are NOT refused: they have always been allowed, and
since Pack M3.0 the sidebar's Lane list tells them apart by position.

THE ID RULE, ratified: made from the name (lower case, every run of
characters outside [a-z0-9] one underscore, trimmed, cut to 32, empty
gives `lane`, a collision takes _2, _3, ...), shown and EDITABLE in the
Add box, then FROZEN forever. No function in this pack changes an id.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from chronotagger.core.models import Interval
from chronotagger.core.tracks import (Track, check_track_table,
                                      duplicate_lane_name,
                                      lane_delete_refusal,
                                      lane_display_key, lane_hide_refusal,
                                      lane_id_from_name, lane_rows_add,
                                      lane_rows_delete, lane_rows_move,
                                      renumber_lane_rows)

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
    lbl._update_plot()
    return lbl


def rows(app):
    return [t.id for t in app.tracks]


# ================================================== 1. the id rule


@pytest.mark.parametrize("name,want", [
    ("Wake (umbra)", "wake_umbra"),
    ("Region (human)", "region_human"),
    ("Agent (C-MMAE)", "agent_c_mmae"),
    ("   ", "lane"),
    ("!!!", "lane"),
    ("", "lane"),
    ("Already_fine-1", "already_fine_1"),
    ("A" * 60, "a" * 32),
])
def test_the_id_is_made_from_the_name(name, want):
    assert lane_id_from_name(name) == want


def test_a_colliding_id_takes_a_number():
    assert lane_id_from_name("Wake", ["wake"]) == "wake_2"
    assert lane_id_from_name("Wake", ["wake", "wake_2"]) == "wake_3"


def test_a_made_id_is_always_a_legal_id():
    from chronotagger.core.tracks import validate_track_id
    for name in ("Wake (umbra)", "a/b.c", "  ", "x" * 99, "3 lanes!"):
        validate_track_id(lane_id_from_name(name))


def test_no_function_in_this_pack_changes_an_id(app):
    """The id is FROZEN once the lane exists. RULED."""
    before = rows(app)
    app._lane_rename("wake", "Something else")
    app._lane_move("wake", -1)
    app._lane_set_locked("wake", True)
    app._lane_set_visible("wake", False)
    assert sorted(rows(app)) == sorted(before)


# ================================ 2. ONE validator, TWO voices


def test_the_constructor_voice_is_unchanged():
    with pytest.raises(ValueError) as e:
        check_track_table([])
    assert "must contain at least one track" in str(e.value)
    with pytest.raises(ValueError) as e:
        check_track_table([Track(id="a", classes=["x"]),
                           Track(id="a", classes=["x"])])
    assert "duplicate track id" in str(e.value)
    with pytest.raises(ValueError) as e:
        check_track_table([Track(id="a")])
    assert "no classes" in str(e.value)


def test_the_file_voice_is_unchanged():
    with pytest.raises(ValueError) as e:
        check_track_table([], "a session")
    assert "has an empty track table" in str(e.value)
    with pytest.raises(ValueError) as e:
        check_track_table([Track(id="dup", classes=["x"]),
                           Track(id="dup", classes=["x"])], "a session")
    assert "two tracks with the id 'dup'" in str(e.value)
    with pytest.raises(ValueError) as e:
        check_track_table([Track(id="a")], "a session")
    assert "no classes" in str(e.value)


def test_both_shipped_doors_now_go_through_the_one_checker():
    """No third copy: the two old entry points delegate."""
    from chronotagger.labeler.mixins.io_export import validate_track_table
    rowset = [Track(id="a", classes=["x"])]
    assert validate_track_table(rowset, "a session") is rowset


def test_display_names_are_NOT_checked_by_the_table_rules():
    """The fence: a file or a driver may hold two lanes with one name."""
    rowset = [Track(id="a", name="Same", classes=["x"]),
              Track(id="b", name="Same", classes=["x"])]
    assert check_track_table(rowset) is rowset


# ================================ 3. the new display-name rule


def test_a_duplicate_display_name_is_found_only_when_the_box_made_it():
    rowset = [Track(id="a", name=" Wake  (umbra) ", classes=["x"]),
              Track(id="b", name="wake (UMBRA)", classes=["x"])]
    assert duplicate_lane_name(rowset, only_ids=set()) is None, \
        "a pre-existing clash is left alone"
    found = duplicate_lane_name(rowset, only_ids={"b"})
    assert found is not None and found[0] == "b"
    assert duplicate_lane_name(rowset, only_ids=None)[0] == "b"


def test_the_comparison_trims_and_lowercases():
    assert lane_display_key("  Wake   (umbra) ") == \
        lane_display_key("wake (UMBRA)")


# ================================ 4. the pure table operations


def test_add_appends_at_the_end_and_colours_from_the_palette():
    rowset = [Track(id="a", name="A", classes=["x"])]
    row = lane_rows_add(rowset, "Wake (umbra)", ["umbra", "penumbra"],
                        palette=["#111111", "#222222"])
    assert row.id == "wake_umbra"
    assert row.order == 1 and rowset[-1] is row
    assert row.class_colors == {"umbra": "#111111",
                                "penumbra": "#222222"}
    assert row.locked is False and row.visible is True


def test_move_does_not_wrap_and_renumbers():
    rowset = [Track(id=c, name=c, classes=["x"]) for c in "abc"]
    renumber_lane_rows(rowset)
    assert lane_rows_move(rowset, "a", -1) is False
    assert lane_rows_move(rowset, "c", +1) is False
    assert lane_rows_move(rowset, "c", -1) is True
    assert [t.id for t in rowset] == ["a", "c", "b"]
    assert [t.order for t in rowset] == [0, 1, 2]


def test_delete_takes_the_intervals_with_it():
    rowset = [Track(id=c, name=c, classes=["x"]) for c in "ab"]
    ivs = [Interval(ts("00:00:00"), ts("00:01:00"), "x", None, track="a"),
           Interval(ts("00:02:00"), ts("00:03:00"), "x", None, track="b")]
    assert lane_rows_delete(rowset, ivs, "a") == 1
    assert [t.id for t in rowset] == ["b"]
    assert [iv.track for iv in ivs] == ["b"], \
        "a stray interval blocks every export; the two move together"


def test_the_two_refusals_say_why_in_plain_words():
    one = [Track(id="a", name="Only", classes=["x"])]
    assert "only lane" in lane_delete_refusal(one, "a")
    two = [Track(id="a", name="A", classes=["x"], locked=True),
           Track(id="b", name="B", classes=["x"])]
    assert "unlock it first" in lane_delete_refusal(two, "a")
    assert lane_delete_refusal(two, "b") == ""
    vis = [Track(id="a", name="A", classes=["x"]),
           Track(id="b", name="B", classes=["x"], visible=False)]
    assert "only lane you can see" in lane_hide_refusal(vis, "a")
    assert lane_hide_refusal(vis, "b") == ""


# ================================ 5. the mixin methods, on the app


def test_add_is_one_undo_step(app):
    """Whether it marks the session is PART D's subject, not this one.

    These six methods go through `_lane_gesture`, and whether THAT puts
    the `modified` flag back is exactly the Pack M3.0 DR2 amendment that
    PART D carries and that J.E. can still flip. It is measured in
    `tests/test_packm3_1_manage_lanes_apply.py`, in one place.
    """
    n0 = len(app.undo_stack)
    row = app._lane_add("Fourth lane", ["p", "q"])
    assert row is not None and row.id == "fourth_lane"
    assert rows(app) == ["region", "wake", "agent", "fourth_lane"]
    assert len(app.undo_stack) - n0 == 1
    app._undo()
    assert rows(app) == ["region", "wake", "agent"]


def test_rename_touches_the_name_and_nothing_else(app):
    assert app._lane_rename("wake", "  Wake   2 ") is True
    row = app.track_by_id("wake")
    assert row.name == "Wake 2"
    assert row.id == "wake"
    assert app._lane_rename("wake", "Wake 2") is False, "a no-op is a no-op"
    app._undo()
    assert app.track_by_id("wake").name == "Wake (umbra)"


def test_move_is_one_undo_step(app):
    assert app._lane_move("agent", -1) is True
    assert rows(app) == ["region", "agent", "wake"]
    assert [t.order for t in app.tracks] == [0, 1, 2]
    app._undo()
    assert rows(app) == ["region", "wake", "agent"]


def test_lock_and_visible_can_reach_any_lane(app):
    """Not only the ACTIVE one, which is all Ctrl+L and Ctrl+H reach."""
    assert app.active_track_id == "region"
    assert app._lane_set_locked("wake", True) is True
    assert app.track_by_id("wake").locked is True
    assert app._lane_set_visible("wake", False) is True
    assert app.track_by_id("wake").visible is False
    app._undo()
    assert app.track_by_id("wake").visible is True


def test_hiding_the_last_visible_lane_is_refused(app):
    app._lane_set_visible("wake", False)
    app._lane_set_visible("agent", False)
    n0 = len(app.undo_stack)
    assert app._lane_set_visible("region", False) is False
    assert app.track_by_id("region").visible is True
    assert len(app.undo_stack) == n0, "a refusal pushes nothing"
    assert "only lane you can see" in app.status_var.get()


def test_delete_takes_the_intervals_and_moves_the_active_lane(app):
    assert len(app.intervals) == 2
    n = app._lane_delete("region")
    assert n == 2
    assert rows(app) == ["wake", "agent"]
    assert app.intervals == []
    assert app.active_track_id == "wake", \
        "deleting the active lane lands on the first VISIBLE lane"
    app._undo()
    assert rows(app) == ["region", "wake", "agent"]
    assert len(app.intervals) == 2


def test_delete_refuses_a_locked_lane_and_the_last_lane(app):
    n0 = len(app.undo_stack)
    assert app._lane_delete("agent") == -1
    assert "unlock it first" in app.status_var.get()
    assert len(app.undo_stack) == n0
    app._lane_delete("region")
    app._lane_delete("wake")
    app._lane_set_locked("agent", False)
    assert app._lane_delete("agent") == -1
    assert "only lane" in app.status_var.get()


def test_an_add_that_breaks_a_table_rule_changes_nothing(app):
    n0 = len(app.undo_stack)
    assert app._lane_add("Nothing", []) is None
    assert rows(app) == ["region", "wake", "agent"]
    assert len(app.undo_stack) == n0
    assert "cannot add that lane" in app.status_var.get()
