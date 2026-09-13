"""Pack M0 (sidebar integrity) regression tests.

Three rulings, one pin group each.

R1 THE ORDINAL BUG.  The sidebar's clicked row used to resolve to an
interval by the 1-based ORDINAL printed in the "#" column
(`events/base.py:120` reading what `stats.py:27` wrote).  That is right
only while the list shows every interval in list order.  Scope the list,
filter it or sort it and the ordinal lands on a DIFFERENT interval: the
user clicks one row, another interval is selected, and the next `d`
deletes that one, silently.  Reproduced in
`edit_pack/evidence/probes/ml_g4_07_points_and_rowmap.py` part 2.  After
this pack the row's identity is its Treeview iid and the "#" column is
display only.

R2 SCOPE TOGGLE.  A "Show: all / window" control scopes the list to
[t0, t1] using the Labels strip's own window test.  The DEFAULT IS
"all", because an interval lying past the end of the data is
navigation-unreachable (campaign finding C13-3) and this list is the
only way to reach it.

R3 tag_configure HYGIENE.  Once per class per refresh, not once per row.

WHAT WOULD HAVE CAUGHT THE BUG ON THE SHIPPED TREE: three of these pins
are red at `refactor` @ b5431d5 without any of the new machinery --
`test_a_row_the_refill_did_not_write_selects_nothing_at_all`,
`test_no_source_file_indexes_the_interval_list_with_a_row_ordinal` and
`test_the_sidebar_exposes_a_show_all_window_control`.  The rest need the
scope filter to exist before the defect can be triggered at all, which
is why the pack ships the fix and the trigger together.
"""

from pathlib import Path

import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import chronotagger
from chronotagger.core.models import Interval

REPO = Path(chronotagger.__file__).resolve().parents[2]
SRC = REPO / "src" / "chronotagger"
STRIP_BANDS_GID = "chronotagger:strip-bands"

DAY = "2015-01-03 "


@pytest.fixture(autouse=True)
def _stub_dialogs(monkeypatch):
    """STANDING RULE (Pack 3): stub every dialog on any reachable path."""
    import tkinter.messagebox as mb
    import tkinter.filedialog as fd
    calls = []
    for kind in ("showinfo", "showwarning", "showerror", "askyesno",
                 "askyesnocancel", "askokcancel"):
        monkeypatch.setattr(
            mb, kind, lambda *a, _k=kind, **kw: calls.append(_k) or True)
    for kind in ("asksaveasfilename", "askopenfilename", "askdirectory"):
        monkeypatch.setattr(fd, kind, lambda *a, **kw: "")
    return calls


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------- helpers

def ts(hhmmss):
    return pd.Timestamp(DAY + hhmmss)


def spread_intervals():
    """Six one-minute intervals, ten minutes apart, in list order.

    Real indices 0..5 start at 00:01, 00:11, 00:21, 00:31, 00:41, 00:51.
    The test window below shows indices 2, 3 and 4 -- a run that does NOT
    start at zero, which is what makes the ordinal defect visible.
    """
    out = []
    for n, minute in enumerate((1, 11, 21, 31, 41, 51)):
        out.append(Interval(ts("00:%02d:00" % minute),
                            ts("00:%02d:00" % (minute + 1)),
                            ("A", "B", "C")[n % 3], None))
    return out


def scope_window(lbl):
    """00:20:00 .. 00:50:00 -- real indices 2, 3, 4 are inside it."""
    lbl.t0 = ts("00:20:00")
    lbl.t1 = ts("00:50:00")


def rows(lbl):
    return list(lbl.intervals_tree.get_children())


def row_start(lbl, iid):
    """The Start string this row is SHOWING, read back from the widget."""
    return lbl.intervals_tree.item(iid)["values"][0]


def row_label(lbl, iid):
    """The Label string this row is SHOWING, read back from the widget."""
    return str(lbl.intervals_tree.item(iid)["values"][2])


def row_text(lbl, iid):
    return str(lbl.intervals_tree.item(iid)["text"])


def click(lbl, iid):
    """Select one row the way the user does, then run the handler.

    `<<TreeviewSelect>>` is only dispatched by a running event loop and
    this root is withdrawn with no mainloop, so the handler is called
    explicitly -- the same shape probe ml_g4_07 part 2 used.
    """
    lbl.intervals_tree.selection_set(iid)
    lbl._on_interval_tree_select(None)


def scoped(lbl):
    """Put the list in window scope through the product's own control."""
    lbl.interval_scope_var.set("window")
    lbl._on_interval_scope_change()


# ===================================================================== R1


def test_clicking_a_scoped_row_selects_the_interval_that_row_shows(labeler):
    """THE HEADLINE PIN.  Row 3 of a scoped list is real index 4."""
    labeler.intervals[:] = spread_intervals()
    scope_window(labeler)
    scoped(labeler)

    shown = rows(labeler)
    assert len(shown) == 3, "window 00:20-00:50 holds exactly three"
    assert [row_text(labeler, r) for r in shown] == ["1", "2", "3"]

    for r in shown:
        labeler.selected_interval = None
        click(labeler, r)
        assert labeler.selected_interval is not None
        assert (labeler.selected_interval.start.strftime("%H:%M:%S")
                == row_start(labeler, r))

    # and spell the third row out, because that is the case the probe ran
    labeler.selected_interval = None
    click(labeler, shown[2])
    assert labeler.selected_interval.start == ts("00:41:00")
    assert labeler.selected_interval is labeler.intervals[4]


def test_a_sorted_scoped_list_still_selects_the_interval_the_row_shows(labeler):
    """Reorder the DISPLAY with tree.move -- what any sort would do."""
    labeler.intervals[:] = spread_intervals()
    scope_window(labeler)
    scoped(labeler)

    shown = rows(labeler)
    for pos, iid in enumerate(reversed(shown)):
        labeler.intervals_tree.move(iid, "", pos)
    reordered = rows(labeler)
    assert reordered == list(reversed(shown))

    labeler.selected_interval = None
    click(labeler, reordered[0])
    assert labeler.selected_interval is labeler.intervals[4]
    assert (labeler.selected_interval.start.strftime("%H:%M:%S")
            == row_start(labeler, reordered[0]))


def test_d_deletes_the_interval_the_clicked_row_showed(labeler):
    """Click row 3 of the scoped list, press `d`, lose 00:41 and nothing else."""
    labeler.intervals[:] = spread_intervals()
    scope_window(labeler)
    scoped(labeler)

    shown = rows(labeler)
    click(labeler, shown[2])
    assert labeler.selected_interval.start == ts("00:41:00")

    labeler._delete_interval()

    starts = [iv.start.strftime("%H:%M:%S") for iv in labeler.intervals]
    assert starts == ["00:01:00", "00:11:00", "00:21:00",
                      "00:31:00", "00:51:00"]
    assert len(labeler.intervals) == 5


def test_every_row_resolves_to_the_interval_its_own_columns_show(labeler):
    """Position independence, under both scopes, for every row."""
    labeler.intervals[:] = spread_intervals()
    scope_window(labeler)

    for mode in ("all", "window"):
        labeler.interval_scope_var.set(mode)
        labeler._on_interval_scope_change()
        for r in rows(labeler):
            labeler.selected_interval = None
            click(labeler, r)
            assert labeler.selected_interval is not None, mode
            assert (labeler.selected_interval.start.strftime("%H:%M:%S")
                    == row_start(labeler, r)), mode
            assert labeler.selected_interval.label == row_label(labeler, r), mode


def test_the_row_id_is_the_index_into_the_full_interval_list(labeler):
    """The identity is structural, not a coincidence of ordering."""
    labeler.intervals[:] = spread_intervals()
    scope_window(labeler)
    scoped(labeler)
    assert rows(labeler) == ["iv:2", "iv:3", "iv:4"]
    assert labeler._interval_row_map["iv:4"] is labeler.intervals[4]

    labeler.interval_scope_var.set("all")
    labeler._on_interval_scope_change()
    assert rows(labeler) == ["iv:%d" % i for i in range(6)]
    for i in range(6):
        assert labeler._interval_row_map["iv:%d" % i] is labeler.intervals[i]


def test_a_row_the_refill_did_not_write_selects_nothing_at_all(labeler):
    """RED ON THE SHIPPED TREE.  A foreign row must not be guessed at.

    An imposter row carrying the text "3" is exactly the shape any
    filtered or sorted builder produces.  On b5431d5 the handler reads
    that text and selects self.intervals[2].  After this pack the iid is
    unknown, so the click resolves to nothing and the selection that was
    already in hand is left alone.
    """
    labeler.intervals[:] = spread_intervals()
    labeler._update_intervals_list()
    labeler.selected_interval = labeler.intervals[0]

    labeler.intervals_tree.insert(
        "", "end", iid="imposter", text="3",
        values=("99:99:99", "99:99:99", "A", "0"))
    click(labeler, "imposter")

    assert labeler.selected_interval is labeler.intervals[0]


def test_no_source_file_indexes_the_interval_list_with_a_row_ordinal():
    """RED ON THE SHIPPED TREE.  The "#" column is display only.

    A source pin, deliberately: it is the cheap one that stops the defect
    being re-introduced by a future list builder, and it is red at
    b5431d5 on `events/base.py:120` without any new machinery at all.
    """
    if not SRC.is_dir():
        pytest.skip("installed without the source tree beside it")
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            if ".item(" in line and '["text"]' in line:
                offenders.append("%s:%d" % (path.name, n))
    assert offenders == []


# ===================================================================== R2


def test_the_sidebar_exposes_a_show_all_window_control(labeler):
    """RED ON THE SHIPPED TREE.  The control exists and defaults to all."""
    assert labeler.interval_scope_var.get() == "all"
    assert labeler.interval_scope_all_btn.cget("value") == "all"
    assert labeler.interval_scope_window_btn.cget("value") == "window"
    assert str(labeler.interval_scope_all_btn.cget("text")) == "all"
    assert str(labeler.interval_scope_window_btn.cget("text")) == "window"


def test_the_default_scope_shows_every_interval_the_session_holds(labeler):
    """R5: zero behaviour change at default settings."""
    labeler.intervals[:] = spread_intervals()
    scope_window(labeler)
    labeler._update_intervals_list()

    shown = rows(labeler)
    assert len(shown) == len(labeler.intervals) == 6
    assert [row_text(labeler, r) for r in shown] == ["1", "2", "3",
                                                     "4", "5", "6"]
    assert ([row_start(labeler, r) for r in shown]
            == [iv.start.strftime("%H:%M:%S") for iv in labeler.intervals])


def test_window_scope_hides_the_intervals_outside_the_window(labeler):
    labeler.intervals[:] = spread_intervals()
    scope_window(labeler)
    scoped(labeler)
    assert ([row_start(labeler, r) for r in rows(labeler)]
            == ["00:21:00", "00:31:00", "00:41:00"])

    labeler.interval_scope_var.set("all")
    labeler._on_interval_scope_change()
    assert len(rows(labeler)) == 6


def test_the_scope_buttons_refill_the_list_when_invoked(labeler):
    """Drive the WIDGET, not the variable: the command must be wired."""
    labeler.intervals[:] = spread_intervals()
    scope_window(labeler)
    labeler._update_intervals_list()
    assert len(rows(labeler)) == 6

    labeler.interval_scope_window_btn.invoke()
    assert labeler.interval_scope_var.get() == "window"
    assert len(rows(labeler)) == 3

    labeler.interval_scope_all_btn.invoke()
    assert labeler.interval_scope_var.get() == "all"
    assert len(rows(labeler)) == 6


def test_the_hash_column_renumbers_from_one_when_the_list_is_scoped(labeler):
    """The "#" column numbers the rows you can see, 1..n."""
    labeler.intervals[:] = spread_intervals()
    scope_window(labeler)
    scoped(labeler)
    assert [row_text(labeler, r) for r in rows(labeler)] == ["1", "2", "3"]


def test_the_list_and_the_labels_strip_agree_about_the_window(labeler):
    """Same predicate, cross-checked against the painted strip.

    All four boundary cases at once, against a window of 00:20 .. 00:50:
    an interval that ENDS exactly at t0 (out), one that STARTS exactly at
    t0 (in), one that ENDS exactly at t1 (in) and one that STARTS exactly
    at t1 (out).  Half-open, so the two that merely touch are out.
    """
    labeler.intervals[:] = [
        Interval(ts("00:10:00"), ts("00:20:00"), "A", None),   # ends at t0
        Interval(ts("00:20:00"), ts("00:22:00"), "B", None),   # starts at t0
        Interval(ts("00:45:00"), ts("00:50:00"), "C", None),   # ends at t1
        Interval(ts("00:50:00"), ts("00:55:00"), "A", None),   # starts at t1
    ]
    scope_window(labeler)
    scoped(labeler)
    labeler._update_strip()

    faces = 0
    for coll in labeler.strip_ax.collections:
        if (coll.get_gid() or "") == STRIP_BANDS_GID:
            faces = len(coll.get_paths())
    assert faces == 2
    assert len(rows(labeler)) == 2
    assert ([row_start(labeler, r) for r in rows(labeler)]
            == ["00:20:00", "00:45:00"])


def test_an_interval_past_the_end_of_the_data_stays_reachable_under_all(labeler):
    """Campaign finding C13-3, and the whole reason "all" is the default.

    Navigation clamps t1 to data_end, so an interval that begins after the
    last sample can never be put on screen.  The list is the only way to
    select or delete it -- so the default must show it, and window scope
    must be an explicit choice that hides it.
    """
    past_end = Interval(labeler.data_end + pd.Timedelta("1h"),
                        labeler.data_end + pd.Timedelta("2h"), "A", None)
    labeler.intervals[:] = spread_intervals() + [past_end]
    labeler._update_intervals_list()

    shown = rows(labeler)
    assert len(shown) == 7
    labeler.selected_interval = None
    click(labeler, shown[-1])
    assert labeler.selected_interval is past_end

    labeler._delete_interval()
    assert past_end not in labeler.intervals
    assert len(labeler.intervals) == 6

    # and under window scope it is hidden, whatever window you are on
    labeler.intervals[:] = spread_intervals() + [past_end]
    scope_window(labeler)
    scoped(labeler)
    assert past_end.start.strftime("%H:%M:%S") not in [
        row_start(labeler, r) for r in rows(labeler)]


def test_scoping_the_list_does_not_scope_the_statistics(labeler):
    """Coverage always describes the whole session."""
    labeler.intervals[:] = spread_intervals()
    scope_window(labeler)

    labeler._update_intervals_list()
    all_stats = labeler.stats_text.get("1.0", "end")
    scoped(labeler)
    win_stats = labeler.stats_text.get("1.0", "end")

    assert "Total Intervals: 6" in all_stats
    assert all_stats == win_stats


def test_scoping_the_list_does_not_deselect_what_is_off_window(labeler):
    """The scope change is a VIEW change, not an edit -- at model level.

    HONEST LIMIT, measured in `packm0_draft_7_liveselect.py` on BOTH
    trees: in the RUNNING application a refill deselects anyway. The
    clear pass empties the Treeview selection, Tk dispatches an empty
    `<<TreeviewSelect>>` on the next turn of the event loop, and the
    app's own binding turns that into `selected_interval = None`
    (`events/base.py:114-117`). That is PRE-EXISTING -- it happens at
    b5431d5 for every `_update_plot` -- and this pack changes nothing
    about it (DR6). What this pin holds is that
    `_on_interval_scope_change` itself does not touch the selection,
    which is what a fix would build on.
    """
    labeler.intervals[:] = spread_intervals()
    scope_window(labeler)
    labeler._update_intervals_list()

    click(labeler, "iv:0")                     # 00:01, outside the window
    assert labeler.selected_interval is labeler.intervals[0]

    scoped(labeler)
    assert "iv:0" not in rows(labeler)
    assert labeler.selected_interval.start == ts("00:01:00")

    labeler._delete_interval()
    assert [iv.start.strftime("%H:%M:%S") for iv in labeler.intervals] == [
        "00:11:00", "00:21:00", "00:31:00", "00:41:00", "00:51:00"]


def test_the_scope_change_says_what_it_did(labeler):
    labeler.intervals[:] = spread_intervals()
    scope_window(labeler)
    scoped(labeler)
    assert labeler.status_var.get() == (
        "Interval list scoped to this window: 3 of 6 shown")
    labeler.interval_scope_var.set("all")
    labeler._on_interval_scope_change()
    assert labeler.status_var.get() == "Interval list showing all 6 intervals"


# ===================================================================== R3


def test_tag_configure_runs_once_per_class_not_once_per_row(labeler,
                                                            monkeypatch):
    """R3.  21 rows, 3 distinct labels -> 3 calls, not 21."""
    labeler.intervals[:] = [
        Interval(ts("00:%02d:00" % (2 * n)), ts("00:%02d:30" % (2 * n)),
                 ("A", "B", "C")[n % 3], None)
        for n in range(21)
    ]
    calls = []
    real = labeler.intervals_tree.tag_configure

    def counting(tagname, *a, **kw):
        calls.append(tagname)
        return real(tagname, *a, **kw)

    monkeypatch.setattr(labeler.intervals_tree, "tag_configure", counting)
    labeler._update_intervals_list()

    assert len(rows(labeler)) == 21
    assert len(calls) == 3
    assert sorted(calls) == ["A", "B", "C"]


def test_a_label_with_no_colour_of_its_own_still_gets_the_grey_default(labeler):
    """R5.  An interval whose class left the schema keeps its old look.

    The label set is built from the rows on screen, not from self.classes,
    so `orphan` is still configured -- with the "#cccccc" that
    class_colors.get() always handed it.
    """
    labeler.intervals[:] = [
        Interval(ts("00:01:00"), ts("00:02:00"), "A", None),
        Interval(ts("00:03:00"), ts("00:04:00"), "orphan", None),
    ]
    assert "orphan" not in labeler.class_colors
    labeler._update_intervals_list()
    assert (str(labeler.intervals_tree.tag_configure("orphan", "background"))
            == "#cccccc")


def test_tag_configure_is_not_called_for_a_class_with_no_rows(labeler,
                                                              monkeypatch):
    """Unchanged from before: an empty class configures nothing."""
    labeler.intervals[:] = [
        Interval(ts("00:01:00"), ts("00:02:00"), labeler.classes[0], None)]
    calls = []
    real = labeler.intervals_tree.tag_configure

    def counting(tagname, *a, **kw):
        calls.append(tagname)
        return real(tagname, *a, **kw)

    monkeypatch.setattr(labeler.intervals_tree, "tag_configure", counting)
    labeler._update_intervals_list()
    assert calls == [labeler.classes[0]]


# ================================================== the no-regression fence


def test_an_unfiltered_refill_is_what_it_always_was(labeler):
    """R5 in one assertion: default scope, row for row, column for column."""
    labeler.intervals[:] = spread_intervals()
    labeler._update_intervals_list()

    got = []
    for r in rows(labeler):
        item = labeler.intervals_tree.item(r)
        got.append((str(item["text"]), tuple(str(v) for v in item["values"]),
                    tuple(item["tags"])))

    want = []
    for i, iv in enumerate(labeler.intervals):
        want.append((str(i + 1),
                     (iv.start.strftime("%H:%M:%S"),
                      iv.end.strftime("%H:%M:%S"),
                      iv.label,
                      str(iv.end - iv.start).split(".")[0]),
                     (iv.label,)))

    assert got == want


def test_a_labeler_with_no_scope_control_still_lists_everything(labeler):
    """A host built without the sidebar reads as "all" and never raises."""
    labeler.intervals[:] = spread_intervals()
    scope_window(labeler)
    del labeler.interval_scope_var
    assert labeler._interval_list_scope() == "all"
    labeler._update_intervals_list()
    assert len(rows(labeler)) == 6
