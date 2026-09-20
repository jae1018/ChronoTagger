"""Pack M3.1, item E -- A LANE OUT OF A DATA COLUMN.

`add_track_from_column` has existed since Pack M1 and nothing on screen
called it: it was reachable only from a driver script. This is its door,
and it is the last PART of the pack on purpose -- cut it and everything
else still applies and still passes.

THE INGEST IS FACTORED, THE BEHAVIOUR IS NOT. `add_track_from_column`
opened its own gesture, which is exactly what a staged import must not
do inside the Manage Lanes OK. The part that only COMPUTES is
`_column_ingest_plan` -- which is also what the box's preview line calls
-- and the part that only WRITES is `_publish_column_ingest`. The public
method still opens the same gesture with the same name, raises the same
errors and writes the same status line.

WHAT THE BOX OFFERS: every column with at most 20 distinct non-null
values that are text or whole numbers (bool counts; a float column whose
every value is integral counts, because a rule label that arrived with
holes is float64). The gap tolerance is PRE-FILLED with the automatic
value the ingest would use and is editable. The preview says "would make
N intervals in M classes" without touching anything, and a refusal says
why and suggests a bigger tolerance.
"""

import tkinter as tk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from chronotagger.core.ingest import column_is_labelish
from chronotagger.labeler.dialogs._from_column import (FromColumnDialog,
                                                       refusal_line)

DAY = "2015-01-03 "

REGION = {"id": "region", "name": "Region (human)",
          "classes": ["sw", "msh", "UNKNOWN"], "order": 0}

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
    """Every box is RECORDED, never shown."""
    import tkinter.messagebox as mb
    import tkinter.filedialog as fdlg
    import tkinter.simpledialog as sdlg
    seen = []
    for name in ("showerror", "showinfo", "showwarning"):
        monkeypatch.setattr(
            mb, name,
            lambda *a, _n=name, **k: seen.append((_n,) + tuple(a[:2])))
    for name in ("askyesno", "askyesnocancel", "askokcancel",
                 "askretrycancel"):
        monkeypatch.setattr(
            mb, name,
            lambda *a, _n=name, **k: (seen.append((_n,) + tuple(a[:2]))
                                      or True))
    for name in ("askopenfilename", "asksaveasfilename", "askdirectory"):
        monkeypatch.setattr(fdlg, name, lambda *a, **k: "")
    for name in ("askstring", "askinteger", "askfloat"):
        monkeypatch.setattr(sdlg, name, lambda *a, **k: None)
    return seen


@pytest.fixture(autouse=True)
def _no_modal_blocking(monkeypatch):
    """Nothing in this module may block on a window. See the apply pins."""
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


def frame():
    idx = pd.date_range(DAY + "00:00:00", periods=240, freq="30s")
    return pd.DataFrame({
        "a": np.linspace(0.0, 1.0, len(idx)),
        "rule": ["sw"] * 80 + ["msh"] * 80 + ["sw"] * 80,
        "flag": np.array([0] * 120 + [1] * 120),
        "holey": np.array([0.0] * 80 + [np.nan] * 80 + [1.0] * 80),
        "yes": np.array([True] * 120 + [False] * 120),
    }, index=idx)


def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["a"])
    axs["panel2"].plot(df.index, df["a"])


@pytest.fixture
def app(tmp_path):
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(
        df=frame(), plot_fn=plot_fn, layout_spec=dict(LAYOUT),
        window=pd.Timedelta("60min"), autosave_folder=str(tmp_path),
        tracks=[dict(REGION)])
    lbl._build_gui()
    lbl.root.withdraw()
    lbl._update_plot()
    for _ in range(2):
        lbl.root.update()
    yield lbl
    try:
        lbl.root.destroy()
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


def press(widget, text):
    for child in walk(widget):
        try:
            if child.winfo_class() in ("TButton", "Button") and \
                    str(child.cget("text")) == text:
                child.invoke()
                return
        except Exception:
            continue
    raise AssertionError("no button %r" % (text,))


# ==================================================== 1. the rule


@pytest.mark.parametrize("values,want", [
    (pd.Series(["a", "b"]), True),
    (pd.Series([0, 1, 2]), True),
    (pd.Series([True, False]), True),
    (pd.Series([0.0, 1.0, 2.0]), True),
    (pd.Series([0.5, 1.5]), False),
    (pd.Series(pd.to_datetime(["2015-01-01", "2015-01-02"])), False),
])
def test_what_counts_as_a_label_column(values, want):
    assert column_is_labelish(values) is want


def test_the_offered_columns_are_the_labelish_short_ones(app):
    got = {c["name"] for c in app._lane_column_choices()}
    assert got == {"rule", "flag", "holey", "yes"}, \
        "'a' has 240 distinct float values and is data, not labels"


def test_each_column_carries_its_distinct_count_and_the_auto_tolerance(
        app):
    from chronotagger.core.ingest import default_gap_tolerance
    by = {c["name"]: c for c in app._lane_column_choices()}
    assert by["rule"]["distinct"] == 2
    assert by["flag"]["distinct"] == 2
    assert by["holey"]["distinct"] == 2, "NaN is not a distinct value"
    assert by["rule"]["default_gap_tolerance"] == \
        str(default_gap_tolerance(app.df.index))


# ==================================================== 2. the preview


def test_the_preview_touches_nothing(app):
    before = ([t.to_dict() for t in app.tracks], len(app.intervals),
              len(app.undo_stack), list(app.df.columns))
    res = app._lane_column_preview("rule", "1h")
    assert res["ok"] is True
    assert res["n_intervals"] == 3 and res["n_classes"] == 2
    assert res["classes"] == ["msh", "sw"]
    assert ([t.to_dict() for t in app.tracks], len(app.intervals),
            len(app.undo_stack), list(app.df.columns)) == before


def test_a_refused_ingest_comes_back_as_words_not_an_exception(app):
    idx = app.df.index
    vals = ["a" if (i // 2) % 2 else "b" for i in range(len(idx))]
    app.df["choppy"] = vals
    res = app._lane_column_preview("choppy", "1ns")
    assert res["ok"] is False
    assert "gap tolerance" in res["why"]
    assert "Try a bigger gap tolerance." in refusal_line(res["why"])


def test_an_unknown_column_is_words_too(app):
    res = app._lane_column_preview("nope")
    assert res["ok"] is False and "no column" in res["why"]


def test_the_refusal_line_is_one_sentence():
    long = ("the gap tolerance 0 days 00:27:31 would split 27 of 239 "
            "runs (11.3 %), which is more than the 10 % this ingest "
            "accepts. TWO things look like this and the fix is "
            "different for each.")
    line = refusal_line(long)
    assert line.endswith("Try a bigger gap tolerance.")
    assert "TWO things" not in line


# ==================================================== 3. the factoring


def test_the_plan_and_the_publish_are_separate(app):
    before = ([t.to_dict() for t in app.tracks], len(app.intervals))
    plan = app._column_ingest_plan("rule", "rule", gap_tolerance="1h")
    assert len(plan["intervals"]) == 3
    assert plan["source"] == "column:rule"
    assert ([t.to_dict() for t in app.tracks],
            len(app.intervals)) == before, \
        "the plan writes nothing at all"


def test_the_public_method_still_opens_its_own_gesture(app):
    n0 = len(app.undo_stack)
    row = app.add_track_from_column("rule", "rule", name="Rule",
                                    gap_tolerance="1h")
    assert row.id == "rule" and row.locked is True
    assert len(app.undo_stack) - n0 == 1
    assert app.undo_stack[-1].name == "import track rule from column:rule"
    assert "Imported 3 interval(s) into track 'rule'" in \
        app.status_var.get()
    app._undo()
    assert [t.id for t in app.tracks] == ["region"]
    assert app.intervals == []


def test_the_public_method_still_refuses_an_unknown_class(app):
    app.add_track_from_column("rule", "rule", gap_tolerance="1h")
    with pytest.raises(ValueError) as e:
        app.add_track_from_column("flag", "rule", gap_tolerance="1h")
    assert "does not contain" in str(e.value)


def test_a_staged_import_runs_inside_an_OUTER_gesture(app):
    """No nesting, no second undo entry."""
    from chronotagger.core.tracks import Track
    n0 = len(app.undo_stack)
    with app._gesture("manage lanes"):
        app.tracks.append(Track(id="rule", name="Rule",
                                classes=["msh", "sw"], locked=True,
                                order=1))
        n = app._ingest_staged_column({"id": "rule", "column": "rule",
                                       "gap_tolerance": "1h",
                                       "classes": ["msh", "sw"]})
    assert n == 3
    assert len(app.undo_stack) - n0 == 1
    assert app.undo_stack[-1].name == "manage lanes"
    app._undo()
    assert [t.id for t in app.tracks] == ["region"]
    assert app.intervals == []


# ==================================================== 4. the box


def open_col(app, monkeypatch):
    dlg = FromColumnDialog(app.root, columns_fn=app._lane_column_choices,
                           preview_fn=app._lane_column_preview,
                           existing_ids=[t.id for t in app.tracks],
                           existing_names=[t.name for t in app.tracks])
    pump(app.root)
    return dlg


def test_the_box_lists_the_columns_with_their_counts(app, monkeypatch):
    dlg = open_col(app, monkeypatch)
    rows = [dlg.tree.item(i, "values") for i in dlg.tree.get_children()]
    assert ("rule", 2) in [(r[0], int(r[1])) for r in rows]
    assert "a" not in [r[0] for r in rows]
    dlg.destroy()


def test_picking_a_column_fills_the_name_the_id_and_the_preview(
        app, monkeypatch):
    dlg = open_col(app, monkeypatch)
    dlg.tree.selection_set("rule")
    pump(app.root)
    assert dlg.name_var.get() == "rule"
    assert dlg.id_var.get() == "rule"
    assert dlg.preview_var.get() == "would make 3 intervals in 2 classes"
    dlg.destroy()


def test_the_tolerance_is_prefilled_and_editable(app, monkeypatch):
    from chronotagger.core.ingest import default_gap_tolerance
    dlg = open_col(app, monkeypatch)
    assert dlg.tol_var.get() == str(default_gap_tolerance(app.df.index))
    dlg.tree.selection_set("rule")
    pump(app.root)
    dlg.tol_var.set("1h")
    assert dlg.refresh_preview()["ok"] is True
    dlg.destroy()


def test_locked_is_pre_ticked(app, monkeypatch):
    dlg = open_col(app, monkeypatch)
    assert dlg.locked_var.get() is True
    dlg.destroy()


def test_a_refusal_shows_why_and_ok_is_refused(app, monkeypatch, boxes):
    app.df["choppy"] = ["a" if (i // 2) % 2 else "b"
                        for i in range(len(app.df.index))]
    dlg = open_col(app, monkeypatch)
    dlg.tree.selection_set("choppy")
    pump(app.root)
    dlg.tol_var.set("1ns")
    dlg.refresh_preview()
    assert "Try a bigger gap tolerance." in dlg.preview_var.get()
    press(dlg, "OK")
    assert dlg.result is None
    assert boxes[-1][:2] == ("showerror", "Cannot import that column")
    dlg.destroy()


def test_ok_stages_a_description_and_imports_nothing(app, monkeypatch):
    before = ([t.id for t in app.tracks], len(app.intervals))
    dlg = open_col(app, monkeypatch)
    dlg.tree.selection_set("rule")
    pump(app.root)
    dlg.tol_var.set("1h")
    dlg.name_var.set("Rule label")
    dlg.refresh_preview()
    press(dlg, "OK")
    assert dlg.result == {
        "id": "rule_label", "name": "Rule label", "column": "rule",
        "gap_tolerance": "1h", "locked": True,
        "classes": ["msh", "sw"], "n_intervals": 3}
    assert ([t.id for t in app.tracks], len(app.intervals)) == before


def test_ok_refuses_a_duplicate_name_and_a_duplicate_id(
        app, monkeypatch, boxes):
    dlg = open_col(app, monkeypatch)
    dlg.tree.selection_set("rule")
    pump(app.root)
    dlg.name_var.set("Region (human)")
    press(dlg, "OK")
    assert boxes[-1][1] == "Duplicate name"
    dlg.name_var.set("Fine")
    dlg.id_var.set("region")
    dlg._id_edited()
    press(dlg, "OK")
    assert boxes[-1][1] == "Duplicate id"
    assert dlg.result is None
    dlg.destroy()


def test_it_needs_no_labeler_at_all(app, monkeypatch):
    """Two plain functions is the whole contract."""
    dlg = FromColumnDialog(
        app.root,
        columns_fn=lambda: [{"name": "c", "distinct": 3,
                             "default_gap_tolerance": "1h"}],
        preview_fn=lambda col, tol: {"ok": True, "why": "",
                                     "n_intervals": 7, "n_classes": 1,
                                     "classes": ["x"]})
    pump(app.root)
    dlg.tree.selection_set("c")
    pump(app.root)
    assert dlg.preview_var.get() == "would make 7 intervals in 1 class"
    dlg.destroy()


# ==================================================== 5. end to end


def test_a_staged_import_lands_inside_the_manage_lanes_gesture(
        app, monkeypatch):
    from chronotagger.labeler.dialogs.manage_lanes import (
        ManageLanesResult)
    lanes = [t.to_dict() for t in app.tracks]
    lanes.append({"id": "rule_label", "name": "Rule label",
                  "classes": ["msh", "sw"], "locked": True, "order": 1})
    n0 = len(app.undo_stack)
    assert app._apply_manage_lanes_result(ManageLanesResult(
        lanes=lanes, added=["rule_label"],
        imports=[{"id": "rule_label", "name": "Rule label",
                  "column": "rule", "gap_tolerance": "1h",
                  "locked": True, "classes": ["msh", "sw"],
                  "n_intervals": 3}])) is True
    assert len(app.undo_stack) - n0 == 1
    assert len(app.intervals) == 3
    assert app.track_by_id("rule_label").locked is True
    assert app.status_var.get() == \
        "Lanes updated: 1 added, 1 imported (3 intervals)"
    app._undo()
    assert [t.id for t in app.tracks] == ["region"]
    assert app.intervals == []


def test_the_box_gets_the_columns_through_the_opener(app, monkeypatch):
    seen = {}

    def waiter(self, win=None, *a, **k):
        seen["has_button"] = any(
            str(getattr(c, "cget", lambda *_: "")("text")) ==
            "From column..."
            for c in walk(win) if c.winfo_class() in ("TButton", "Button"))
        seen["columns"] = [d["name"] for d in win._columns_fn()]
        win._on_cancel()
    monkeypatch.setattr(tk.Misc, "wait_window", waiter)
    app._open_manage_lanes()
    assert seen["has_button"] is True
    assert "rule" in seen["columns"]
