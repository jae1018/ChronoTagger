"""Pack M2.5 -- ONE Tk INTERPRETER, HOWEVER THE WINDOW IS BUILT.

THE LIVE DEFECT. `test_drivers/drive_multilabel_thb.py` builds the window
itself (`app._build_gui()`), ingests the agent lane, sets the window, and
THEN calls `app.run()` -- and `run()` called `_build_gui()` a second time,
so the process ended up with TWO `tk.Tk()` interpreters. Tk keeps
`tk._default_root` pointing at the FIRST one for the life of the process,
and every `tk.StringVar()` / `BooleanVar()` built without an explicit
master lands in `_default_root`. The widgets the user sees are children of
the SECOND root. Measured on the committed Pack M2 tree: 9 of the app's 10
Tk variables landed in the hidden first interpreter, the Lane combobox's
own textvariable raised TclError when read in the interpreter its widget
lives in, and a fresh masterless `StringVar` -- which is what every dialog
in this tree builds -- landed in the hidden one too.

WHAT THE USER SAW. A blank Lane dropdown. Overlap Detected radio buttons
that draw as selected while the hint never changes and Confirm stays grey
(`dialogs/overlap_resolution.py::_on_policy_change` reads `''` and
returns). Label by Rule radios that never select and a preview stuck on
'skip'. The Save Changes? box opening BEHIND the window.

THE FIX, in two halves. `_build_gui()` returns at once when the window it
would build is already there, so calling it twice is safe and the driver
keeps calling `run()` (it wants the autosave-recovery prompt). And, belt
and braces, the nine sidebar variables now pass `master=self.root` the way
`current_class_var` already did, so they bind to the labeler's own
interpreter even in a process that has another Tk root in it first.

Every test here builds a REAL Tk root, because what is pinned is which
interpreter a variable landed in.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tkinter as tk
from tkinter import ttk

DAY = "2015-01-03 "

# The nine that had no master. current_class_var (controls.py) always had
# one and is the control in the pin below.
SIDEBAR_VARS = ("lane_var", "lane_visible_var", "lane_locked_var",
                "interval_scope_var", "interval_track_scope_var",
                "snap_var", "overlays_var", "highlight_points_var",
                "status_var")

TWO = [
    {"id": "human", "name": "Human", "classes": ["UNKNOWN", "sw"],
     "class_colors": {"UNKNOWN": "#7f7f7f", "sw": "#4e79a7"}, "order": 0},
    {"id": "model", "name": "Model", "classes": ["0", "1"],
     "class_colors": {"0": "#111111", "1": "#222222"}, "order": 1},
]

LAYOUT = {
    "nrows": 2, "ncols": 1,
    "areas": [
        {"key": "panel1", "row": 0, "col": 0, "role": "time"},
        {"key": "labels", "row": 1, "col": 0, "role": "labels"},
    ],
}


@pytest.fixture(autouse=True)
def _dialog_counter(monkeypatch):
    """STANDING RULE (Pack 3): every dialog stubbed, and counted."""
    import tkinter.messagebox as mb
    import tkinter.filedialog as fd
    import tkinter.simpledialog as sd
    calls = []
    for kind in ("showinfo", "showwarning", "showerror", "askyesno",
                 "askyesnocancel", "askokcancel"):
        monkeypatch.setattr(
            mb, kind, lambda *a, _k=kind, **kw: calls.append(_k) or True)
    for kind in ("asksaveasfilename", "askopenfilename", "askdirectory"):
        monkeypatch.setattr(fd, kind, lambda *a, **kw: "")
    for kind in ("askstring", "askinteger", "askfloat"):
        monkeypatch.setattr(sd, kind, lambda *a, **kw: None)
    return calls


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def frame():
    idx = pd.date_range(DAY + "00:00:00", periods=60, freq="30s")
    return pd.DataFrame({"a": np.linspace(0, 1, len(idx))}, index=idx)


def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["a"])


def make(tmp_path):
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(
        df=frame(), plot_fn=plot_fn, layout_spec=dict(LAYOUT),
        window=pd.Timedelta("15min"), autosave_folder=str(tmp_path),
        tracks=[dict(t) for t in TWO])
    lbl._build_gui()
    lbl.root.withdraw()
    lbl._update_plot()
    return lbl


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """A labeler built in a process whose default root is THIS labeler's.

    The suite runs 890 other tests in the same process and some of them
    build roots, so `_default_root` is cleared first: that is the state a
    real launch is in, and it is the state the defect needs to be visible
    in at all.
    """
    monkeypatch.setattr(tk, "_default_root", None, raising=False)
    lbl = make(tmp_path)
    assert tk._default_root is lbl.root
    yield lbl
    lbl.root.destroy()


def variables(app):
    return dict((n, v) for n, v in vars(app).items()
                if isinstance(v, tk.Variable))


def widgets_of(widget, cls, out):
    for ch in widget.winfo_children():
        if isinstance(ch, cls):
            out.append(ch)
        widgets_of(ch, cls, out)
    return out


# ---------------------------------------------------------------- the guard

def test_a_second_build_gui_does_not_make_a_second_interpreter(fresh,
                                                               monkeypatch):
    from chronotagger.labeler.mixins.view_build import window as window_mod
    made = []
    real = window_mod._new_tk_root
    monkeypatch.setattr(window_mod, "_new_tk_root",
                        lambda *a, **k: made.append(1) or real(*a, **k))
    root = fresh.root
    fresh._build_gui()
    assert made == [], "a second _build_gui() built a second tk.Tk()"
    assert fresh.root is root
    assert tk._default_root is fresh.root


def test_run_after_a_manual_build_builds_nothing_and_still_recovers(
        fresh, monkeypatch):
    """The driver's real launch path: build, ingest, then run().

    run() must still do its autosave check -- that is why the driver calls
    it -- and must not rebuild the window underneath it.
    """
    from chronotagger.labeler.mixins.view_build import window as window_mod
    made = []
    real = window_mod._new_tk_root
    monkeypatch.setattr(window_mod, "_new_tk_root",
                        lambda *a, **k: made.append(1) or real(*a, **k))
    checked = []
    monkeypatch.setattr(type(fresh), "_check_autosave",
                        lambda self: checked.append(1) or None)
    monkeypatch.setattr(tk.Tk, "mainloop", lambda self, *a, **k: None)
    root = fresh.root
    fresh.run()
    assert made == []
    assert checked == [1], "run() must still offer autosave recovery"
    assert fresh.root is root
    assert tk._default_root is root


def test_every_tk_variable_on_the_app_is_in_the_visible_root(fresh):
    fresh._build_gui()
    vs = variables(fresh)
    assert len(vs) >= 10, sorted(vs)
    wrong = sorted(n for n, v in vs.items() if v._tk is not fresh.root.tk)
    assert wrong == [], wrong


def test_the_lane_dropdowns_own_variable_reads_in_its_widgets_interpreter(
        fresh):
    """The blank Lane dropdown, as one assertion.

    A Combobox whose textvariable was created in another interpreter has
    no such Tcl variable in its own, so reading the name the widget is
    actually bound to raises -- which is why the box drew empty.
    """
    fresh._build_gui()
    combo = fresh.lane_combo
    name = str(combo.cget("textvariable"))
    assert combo.tk.globalgetvar(name) == fresh.lane_var.get()
    assert fresh.lane_var.get() == "Human"


# ------------------------------------------------- the nine explicit masters

def test_the_sidebar_variables_bind_to_the_labelers_own_root(tmp_path,
                                                             monkeypatch):
    """Belt and braces: another Tk root exists FIRST, as in the wizard.

    With no `master=`, a Tk variable binds to `tk._default_root`, which in
    this process is the decoy -- not the window the user is looking at.

    The decoy is built through `_new_tk_root()` and not through a bare
    `tk.Tk()`. Pack 6 R10 measured the transient startup TclError
    ("Can't find a usable init.tcl") on this machine in 70 % of
    single-module runs and shipped a bounded retry for it, and
    `tests/test_pack8_wizard_ux.py` (R16) already pins that nothing calls
    `tk.Tk()` directly. Measured at the seal: a bare decoy failed 6 of 40
    ISOLATED runs of this module, always on that line; the retried one
    failed 0 of 50.
    """
    from chronotagger.labeler.mixins.view_build.window import _new_tk_root
    decoy = _new_tk_root()
    decoy.withdraw()
    monkeypatch.setattr(tk, "_default_root", decoy, raising=False)
    lbl = None
    try:
        lbl = make(tmp_path)
        assert tk._default_root is decoy and lbl.root is not decoy
        for name in SIDEBAR_VARS + ("current_class_var",):
            var = getattr(lbl, name)
            assert var._tk is lbl.root.tk, name
            assert var._tk is not decoy.tk, name
    finally:
        if lbl is not None:
            lbl.root.destroy()
        decoy.destroy()


# ------------------------------------------------------ a dialog's own var

def test_the_overlap_dialogs_radio_buttons_reach_its_own_variable(
        fresh, monkeypatch):
    """The Overlap Detected box: radios drew selected, Confirm stayed grey.

    The dialog builds `tk.StringVar(value="")` with NO master, so with two
    interpreters it read '' forever, `_on_policy_change` returned at its
    first line, the hint never changed and Confirm was never enabled.
    """
    from chronotagger.labeler.dialogs.overlap_resolution import (
        OverlapResolutionDialog)
    fresh._build_gui()
    noop = lambda *a, **k: None
    monkeypatch.setattr(tk.Misc, "wait_visibility", noop)
    monkeypatch.setattr(tk.Misc, "wait_window", noop)
    monkeypatch.setattr(tk.Misc, "grab_set", noop)
    seen = []
    dlg = OverlapResolutionDialog(
        parent=fresh.root, overlap_count=2,
        on_policy_selected=lambda p: (seen.append(p) or (17, 1)))
    try:
        assert dlg._policy_var._tk is fresh.root.tk
        radios = widgets_of(dlg, ttk.Radiobutton, [])
        want = [r for r in radios if str(r.cget("value")) == "replace"]
        assert len(want) == 1, [str(r.cget("value")) for r in radios]
        want[0].invoke()
        assert seen == ["replace"], seen
        assert dlg._policy_var.get() == "replace"
        assert "Will create 1 interval" in dlg._feedback.cget("text")
        assert str(dlg._confirm_btn.cget("state")) == "normal"
    finally:
        dlg.destroy()


def test_a_destroyed_window_can_still_be_rebuilt(fresh, monkeypatch):
    """DR13: winfo_exists() RAISES on a DESTROYED root instead of answering
    False, which is why the guard asks inside try/except -- a window the
    user closed must still be rebuildable."""
    old = fresh.root
    old.destroy()
    fresh._build_gui()
    assert fresh.root is not old
    assert bool(fresh.root.winfo_exists())
    fresh.root.withdraw()
