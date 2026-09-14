"""Pack M2.5 -- A PLAIN SHORTCUT IS DEAD WHILE CONTROL IS HELD.

THE LIVE DEFECT. Press Ctrl+A in the Start box to select the text you are
about to retype. Tk's own <<SelectAll>> selects it -- and ChronoTagger's
global Add shortcut fires too, so a "No Selection" modal ("Select a time
range first (drag or click x2).") lands on top of the window. Found twice
in the live window. The focus-aware early exit in `_on_key_press` returns
only for an UNMODIFIED key while an editable widget has focus, and the
shortcut branch below it never looked at the modifier at all.

THE CENSUS this file pins (`events/keyboard.py`, as shipped in Pack M2).

  PLAIN shortcuts -- fired on any modifier state, now Control-free only:
    1..9           pick the class at that position
    n / N / Right  next window
    p / P / Left   previous window
    a / A / Return Add
    d / D / Delete Delete
    u / U          pick UNKNOWN
    BackSpace      Undo

  CONTROL-DEFINED shortcuts -- unchanged by this pack:
    Ctrl+Down / Ctrl+Up   cycle the active lane
    Ctrl+L                lock / unlock the active lane
    Ctrl+H                hide / show the active lane
    Ctrl+S                save the session
    Ctrl+E                export
    Ctrl+Z                undo
    Ctrl+Y                redo

  Escape is handled ABOVE the focus guard and is deliberately untouched.
  NOT A CHANGE HERE, recorded so nobody reads it as one: Shift+BackSpace
  reaches Undo, not Redo, because the BackSpace branch matches first. That
  is how Pack M2 shipped and this pack does not touch it.

  TWO BRANCHES, NOT ONE. BackSpace also reaches the REDO branch, which
  fires whenever Shift is held, so the guard has to be on both: guarding
  only the Undo branch moves Ctrl+Shift+BackSpace from UNDO to REDO instead
  of silencing it -- still a data command fired from inside a text box. The
  last pin before the live-symptom pair is the one that holds it.

The collisions the guard closes are not only Ctrl+A: Ctrl+D deleted the
selected interval, Ctrl+N / Ctrl+P and Ctrl+Left / Ctrl+Right moved the
window, Ctrl+U rewrote the class, Ctrl+BackSpace ("delete the previous
word" in every text box on this platform) undid the last data edit, and
Ctrl+1..9 changed the class while the Tk-level binding changed the pane
tab -- one keystroke, two effects.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

DAY = "2015-01-03 "
CTRL = 0x4
SHIFT = 0x1

TWO = [
    {"id": "human", "name": "Human", "classes": ["UNKNOWN", "sw", "msh"],
     "class_colors": {"UNKNOWN": "#7f7f7f", "sw": "#4e79a7",
                      "msh": "#f28e2b"}, "order": 0},
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

# (keysym, what it does when Control is NOT held). "class:<value>" means
# the class dropdown ends on that value; anything else is a method name.
PLAIN = [
    ("1", "class:UNKNOWN"),
    ("2", "class:sw"),
    ("a", "_add_interval"),
    ("A", "_add_interval"),
    ("Return", "_add_interval"),
    ("d", "_delete_interval"),
    ("D", "_delete_interval"),
    ("Delete", "_delete_interval"),
    ("n", "_next_window"),
    ("N", "_next_window"),
    ("Right", "_next_window"),
    ("p", "_prev_window"),
    ("P", "_prev_window"),
    ("Left", "_prev_window"),
    ("u", "class:UNKNOWN"),
    ("U", "class:UNKNOWN"),
    ("BackSpace", "_undo"),
]

CONTROL = [
    ("Down", "_cycle_active_lane"),
    ("Up", "_cycle_active_lane"),
    ("l", "_toggle_active_lane_locked"),
    ("L", "_toggle_active_lane_locked"),
    ("h", "_toggle_active_lane_visible"),
    ("H", "_toggle_active_lane_visible"),
    ("s", "_save_session"),
    ("S", "_save_session"),
    ("e", "_export_intervals"),
    ("E", "_export_intervals"),
    ("z", "_undo"),
    ("y", "_redo"),
]

SPIED = ("_add_interval", "_delete_interval", "_next_window",
         "_prev_window", "_undo", "_redo", "_save_session",
         "_export_intervals", "_cycle_active_lane",
         "_toggle_active_lane_locked", "_toggle_active_lane_visible")


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
        monkeypatch.setattr(fd, kind, lambda *a, **kw: calls.append(kind) or "")
    for kind in ("askstring", "askinteger", "askfloat"):
        monkeypatch.setattr(sd, kind, lambda *a, **kw: calls.append(kind))
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


@pytest.fixture
def app(tmp_path):
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(
        df=frame(), plot_fn=plot_fn, layout_spec=dict(LAYOUT),
        window=pd.Timedelta("15min"), autosave_folder=str(tmp_path),
        tracks=[dict(t) for t in TWO])
    lbl._build_gui()
    lbl.root.withdraw()
    lbl._update_plot()
    yield lbl
    lbl.root.destroy()


class Key(object):
    def __init__(self, keysym, state=0):
        self.keysym = keysym
        self.state = state


def spy(app, monkeypatch):
    hits = []
    for name in SPIED:
        monkeypatch.setattr(app, name,
                            lambda *a, _n=name, **k: hits.append(_n))
    return hits


def press(app, keysym, state=0):
    app._on_key_press(Key(keysym, state))


# ------------------------------------- one pin per plain shortcut, Ctrl held

@pytest.mark.parametrize("keysym,_action", PLAIN)
def test_a_plain_shortcut_does_nothing_while_control_is_held(
        app, monkeypatch, _dialog_counter, keysym, _action):
    hits = spy(app, monkeypatch)
    app.current_class_var.set("msh")
    app.status_var.set("SENTINEL")
    press(app, keysym, CTRL)
    assert hits == [], (keysym, hits)
    assert app.current_class_var.get() == "msh", keysym
    assert app.status_var.get() == "SENTINEL", (keysym,
                                                app.status_var.get())
    assert _dialog_counter == [], (keysym, _dialog_counter)


# --------------------------------------- the same shortcuts, Control-free

def test_every_plain_shortcut_still_fires_without_control(app, monkeypatch):
    hits = spy(app, monkeypatch)
    for keysym, action in PLAIN:
        del hits[:]
        app.current_class_var.set("msh")
        press(app, keysym, 0)
        if action.startswith("class:"):
            assert app.current_class_var.get() == action.split(":")[1], keysym
            assert hits == [], (keysym, hits)
        else:
            assert hits == [action], (keysym, hits)


def test_the_control_defined_shortcuts_are_unchanged(app, monkeypatch):
    hits = spy(app, monkeypatch)
    for keysym, action in CONTROL:
        del hits[:]
        press(app, keysym, CTRL)
        assert hits == [action], (keysym, hits)


def test_shift_backspace_still_reaches_undo_as_pack_m2_shipped_it(
        app, monkeypatch):
    """Recorded, not changed: the BackSpace branch matches before the Redo
    branch, so Shift+BackSpace undoes. Out of this pack's scope."""
    hits = spy(app, monkeypatch)
    press(app, "BackSpace", SHIFT)
    assert hits == ["_undo"], hits


def test_ctrl_shift_backspace_does_nothing_either(app, monkeypatch):
    """The one state the guard has to reach TWICE.

    BackSpace also reaches the Redo branch, which fires whenever Shift is
    held -- so guarding only the Undo branch moved Ctrl+Shift+BackSpace
    from UNDO to REDO instead of silencing it, and a data command still
    fired from inside a text box.
    """
    hits = spy(app, monkeypatch)
    press(app, "BackSpace", CTRL | SHIFT)
    assert hits == [], hits


# ------------------------------------------------------- the live symptom

def test_ctrl_a_with_nothing_selected_opens_no_dialog(app, _dialog_counter):
    """The exact live failure: Ctrl+A in a text box popped 'No Selection'."""
    app.current_selection = None
    app.status_var.set("SENTINEL")
    press(app, "a", CTRL)
    assert _dialog_counter == []
    assert app.intervals == []
    assert app.status_var.get() == "SENTINEL"


def test_ctrl_a_without_the_modifier_still_asks_for_a_selection(
        app, _dialog_counter):
    """The other half: plain 'a' with nothing selected is still the Add
    that tells the user to select something first."""
    app.current_selection = None
    press(app, "a", 0)
    assert _dialog_counter == ["showwarning"], _dialog_counter
