"""Pack M2.8 -- DELETING A CLASS THAT IS IN USE.

THE DEFECT (session-5 S5.4, measured on 82acc7a by probe_s5_lowdown Q2).
`LabelManagerDialog._on_delete` opens `_ReassignDialog` and reads
`dlg.result` on the very next line. `_ReassignDialog.__init__` ends at
`wait_visibility()` / `focus()` and RETURNS, so the read happens the
instant the box appears: the answer is always None, `_on_delete` returns
having changed nothing, and whatever the user then picks in the box goes
nowhere. `git log -S wait_window` on that file finds no commit. Deleting
a class the active lane's intervals use has never worked.

THE FIX is one line: `self.wait_window(dlg)` before the read.

HOW THESE PINS ANSWER THE BOX. A modal cannot be waited on inside a
head-less suite, so `tkinter.Misc.wait_window` is patched to a function
that ANSWERS it -- sets the box's variable and calls its OK, or calls its
Cancel -- which is exactly what a pair of hands does and is the only
shape that can tell "waited and got an answer" from "never waited".
`wait_visibility` and `grab_set` are patched to no-ops as well: against a
withdrawn parent they HANG.

THE LAST GROUP goes end to end through `_apply_label_manager_result`,
which is where the reassignment actually lands on the intervals. That
method has always worked; this is the first pack in which anything can
reach it by this route.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tkinter as tk

from chronotagger.core.models import Interval

DAY = "2015-01-03 "

CLASSES = ["UNKNOWN", "sw", "msh"]
COLORS = {"UNKNOWN": "#7f7f7f", "sw": "#4e79a7", "msh": "#f28e2b"}

HUMAN = {"id": "human", "name": "Human", "classes": list(CLASSES),
         "class_colors": dict(COLORS), "order": 0}
MODEL = {"id": "model", "name": "Model", "classes": ["sw", "0"],
         "class_colors": {"sw": "#111111", "0": "#222222"}, "order": 1}

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
    idx = pd.date_range(DAY + "00:00:00", periods=240, freq="30s")
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
    import tkinter.simpledialog as sd
    calls = []
    for kind in ("showinfo", "showwarning", "showerror", "askyesno",
                 "askyesnocancel", "askokcancel"):
        monkeypatch.setattr(
            mb, kind, lambda *a, _k=kind, **kw: calls.append(_k) or True)
    for kind in ("askstring", "askinteger", "askfloat"):
        monkeypatch.setattr(sd, kind, lambda *a, **kw: None)
    return calls


@pytest.fixture(autouse=True)
def _no_modal_blocking(monkeypatch):
    """wait_visibility and grab_set HANG against a withdrawn parent."""
    monkeypatch.setattr(tk.Misc, "wait_visibility", lambda *a, **k: None)
    monkeypatch.setattr(tk.Misc, "grab_set", lambda *a, **k: None)


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


def make_dialog(app, usage):
    from chronotagger.labeler.dialogs.label_manager import LabelManagerDialog
    dlg = LabelManagerDialog(
        parent=app.root, classes=list(CLASSES), class_colors=dict(COLORS),
        usage_counts=dict(usage), reserved={"UNKNOWN"})
    return dlg


def answer(monkeypatch, how, target="msh"):
    """Patch wait_window into the user's hands. Returns the call log."""
    seen = []

    def _wait(self, window=None):
        seen.append(window)
        if window is None:
            return
        if how == "ok":
            window._var.set(target)
            window._ok()
        else:
            window._cancel()

    monkeypatch.setattr(tk.Misc, "wait_window", _wait)
    return seen


# ==================================== 1. the box is waited on, and obeyed


def test_on_delete_waits_for_the_box(app, monkeypatch):
    seen = answer(monkeypatch, "ok")
    dlg = make_dialog(app, {"sw": 3})
    try:
        dlg._selected_name = lambda: "sw"
        dlg._on_delete()
        assert len(seen) == 1, "_on_delete did not wait for the box"
        assert seen[0].__class__.__name__ == "_ReassignDialog"
    finally:
        dlg.destroy()


def test_ok_deletes_the_class_and_records_the_reassignment(app, monkeypatch):
    answer(monkeypatch, "ok", target="msh")
    dlg = make_dialog(app, {"sw": 3})
    try:
        dlg._selected_name = lambda: "sw"
        dlg._on_delete()
        assert "sw" not in dlg._classes
        assert dlg._classes == ["UNKNOWN", "msh"]
        assert dict(dlg._reassign_map) == {"sw": "msh"}
    finally:
        dlg.destroy()


def test_cancel_leaves_the_class_and_the_map_alone(app, monkeypatch):
    answer(monkeypatch, "cancel")
    dlg = make_dialog(app, {"sw": 3})
    try:
        dlg._selected_name = lambda: "sw"
        dlg._on_delete()
        assert dlg._classes == list(CLASSES)
        assert dict(dlg._reassign_map) == {}
    finally:
        dlg.destroy()


def test_an_unused_class_deletes_with_no_box_at_all(app, monkeypatch):
    seen = answer(monkeypatch, "ok")
    dlg = make_dialog(app, {"sw": 0, "msh": 0})
    try:
        dlg._selected_name = lambda: "msh"
        dlg._on_delete()
        assert "msh" not in dlg._classes
        assert dict(dlg._reassign_map) == {}
        assert seen == [], "an unused class must not open the reassign box"
    finally:
        dlg.destroy()


def test_the_boxs_variable_names_its_master(app, monkeypatch):
    """EDIT 616, held the only way that bites.

    With ONE root, an unmastered `tk.StringVar` lands on the same
    interpreter as a mastered one, so comparing `_root` proves nothing.
    What separates them is the QUESTION they ask: a var with no master
    calls `tkinter._get_default_root`, and one that names its master
    never does. Take the default root away and the unmastered var cannot
    be built at all.
    """
    import tkinter
    from chronotagger.labeler.dialogs import label_manager as LM
    built = []
    monkeypatch.setattr(tk.Misc, "wait_window",
                        lambda self, w=None: built.append(w))
    dlg = make_dialog(app, {"sw": 3})
    try:
        def _no_default_root(*a, **k):
            raise RuntimeError("something asked for the DEFAULT root")

        monkeypatch.setattr(tkinter, "_get_default_root", _no_default_root)
        dlg._selected_name = lambda: "sw"
        dlg._on_delete()
        assert len(built) == 1
        box = built[0]
        assert isinstance(box, LM._ReassignDialog)
        assert box._var._root is box._root()
        box._cancel()
    finally:
        dlg.destroy()


# ==================================== 2. end to end, onto the intervals


def test_the_reassignment_lands_on_the_active_lanes_intervals(app):
    from chronotagger.labeler.dialogs.label_manager import LabelManagerResult
    app.intervals[:] = [
        Interval(ts("00:05:00"), ts("00:08:00"), "sw", None, track="human"),
        Interval(ts("00:10:00"), ts("00:12:00"), "msh", None, track="human"),
        Interval(ts("00:20:00"), ts("00:22:00"), "sw", None, track="model"),
    ]
    app.undo_stack.append(object())
    assert app.active_track_id == "human"

    app._apply_label_manager_result(LabelManagerResult(
        classes=["UNKNOWN", "msh"],
        class_colors={"UNKNOWN": "#7f7f7f", "msh": "#f28e2b"},
        rename_map={},
        reassign_map={"sw": "msh"}))

    human = [iv for iv in app.intervals if iv.track == "human"]
    assert [iv.label for iv in human] == ["msh", "msh"]
    assert app.classes == ["UNKNOWN", "msh"]
    assert "sw" not in app.classes

    model = [iv for iv in app.intervals if iv.track == "model"]
    assert [iv.label for iv in model] == ["sw"], \
        "the other lane's intervals and vocabulary are not this dialog's"
    assert app.track_by_id("model").classes == ["sw", "0"]

    assert app.undo_stack == [] and app.redo_stack == []
    assert app.status_var.get() == "Labels updated"
