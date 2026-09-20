"""Pack M2.8 -- F1 HELP: ESCAPE CLOSES IT, AND THE TABLE IS TRUE.

TWO DEFECTS, both small and both measured.

S5.2. `mixins/help.py` binds `<Escape>` on the Help Toplevel and never
gives that window keyboard focus. `grab_set()` takes the POINTER; the
keyboard stays where the window manager left it, which is the main
window. Live in computer-use session 5: Help opened, Escape left it
open, the Close button worked.

THE BACKLOG ITEM. `KEYBOARD_SHORTCUTS` listed
`("Ctrl+Y / Shift+Backspace", "Redo")`, and Shift+BackSpace UNDOES --
`mixins/events/keyboard.py` tests plain BackSpace without looking at
Shift, and has since long before Pack M2 (Pack M2.5 DR14, on purpose).
J.E. ruled the TABLE changes and the KEYS do not, so the last pin here
presses a synthetic Shift+BackSpace and asserts it still undoes.

THIS MODULE OPENS A REAL Toplevel and withdraws it on the next line.
That is deliberate: the focus call and the Escape binding are properties
of the window the product builds, not of a mock. `focus_force` is spied
rather than executed, so nothing steals focus from the machine running
the suite.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tkinter as tk

from chronotagger.core.models import Interval
from chronotagger.labeler.mixins.help import KEYBOARD_SHORTCUTS

DAY = "2015-01-03 "

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
    for kind in ("showinfo", "showwarning", "showerror", "askyesno",
                 "askyesnocancel", "askokcancel"):
        monkeypatch.setattr(mb, kind, lambda *a, **kw: True)


@pytest.fixture(autouse=True)
def _no_real_focus_or_grab(monkeypatch):
    """Spy the focus call and neuter grab_set: a grab against a withdrawn
    parent is the thing that hangs a head-less run."""
    forced = []
    monkeypatch.setattr(tk.Misc, "grab_set", lambda *a, **k: None)
    monkeypatch.setattr(tk.Misc, "focus_force",
                        lambda self: forced.append(self))
    return forced


@pytest.fixture
def app(tmp_path):
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(
        df=frame(), plot_fn=plot_fn, layout_spec=dict(LAYOUT),
        window=pd.Timedelta("60min"), autosave_folder=str(tmp_path))
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    yield lbl
    lbl.root.destroy()


def open_help(app, monkeypatch):
    """Open Help, capture the callback it binds to <Escape>, withdraw it."""
    bound = {}
    real_bind = tk.Toplevel.bind

    def spy_bind(self, sequence=None, func=None, add=None):
        if sequence and func is not None:
            bound[sequence] = func
        return real_bind(self, sequence, func, add)

    monkeypatch.setattr(tk.Toplevel, "bind", spy_bind)
    app._open_help_dialog()
    win = app._help_window
    win.withdraw()
    return win, bound


# ==================================== 1. the window takes the keyboard


def test_opening_help_focuses_the_new_window(app, monkeypatch,
                                             _no_real_focus_or_grab):
    win, _ = open_help(app, monkeypatch)
    try:
        assert _no_real_focus_or_grab == [win], \
            "the Help window was built without being given the keyboard"
    finally:
        win.destroy()


def test_f1_on_an_open_help_window_focuses_it_again(app, monkeypatch,
                                                    _no_real_focus_or_grab):
    win, _ = open_help(app, monkeypatch)
    try:
        del _no_real_focus_or_grab[:]
        app._open_help_dialog()
        assert app._help_window is win, "a second window was built"
        assert _no_real_focus_or_grab == [win], \
            "F1 lifted the open window without focusing it"
    finally:
        win.destroy()


def test_escapes_binding_destroys_the_help_window(app, monkeypatch,
                                                  _no_real_focus_or_grab):
    win, bound = open_help(app, monkeypatch)
    assert "<Escape>" in bound, "Help bound no <Escape> at all"
    bound["<Escape>"](None)
    assert not tk.Toplevel.winfo_exists(win), \
        "the <Escape> binding did not destroy the Help window"


# ==================================== 2. the table matches the bindings


def test_the_help_table_says_shift_backspace_undoes():
    rows = dict(KEYBOARD_SHORTCUTS)
    assert "Ctrl+Z / Backspace / Shift+Backspace" in rows
    assert rows["Ctrl+Z / Backspace / Shift+Backspace"] == "Undo"
    assert rows["Ctrl+Y"] == "Redo"
    assert "Ctrl+Y / Shift+Backspace" not in rows, \
        "the old row claimed Shift+Backspace redoes"
    assert "Ctrl+Z / Backspace" not in rows


def test_shift_backspace_still_undoes(app):
    """R4's other half: the TABLE changed and the KEY did not."""
    app.current_class_var.set("UNKNOWN")
    app.current_selection = (ts("00:05:00"), ts("00:08:00"))
    app._add_interval()
    assert len(app.intervals) == 1

    evt = type("E", (), {"keysym": "BackSpace", "state": 0x1, "char": ""})()
    app._on_key_press(evt)
    assert app.intervals == [], \
        "Shift+BackSpace must still UNDO, whatever the table used to say"
    assert len(app.redo_stack) == 1
