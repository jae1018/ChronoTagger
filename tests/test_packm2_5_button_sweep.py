"""Pack M2.5 -- EVERY BUTTON IN THE WINDOW, PRESSED ONCE.

The live launch and two computer-use sessions found their defects by
pressing things. This is the cheap, deterministic version of that: build
the feel-test driver's window headless, walk it for every button, and
invoke each one with the dialogs stubbed and counted and with
wait_window / wait_visibility / grab_set no-opped so a modal cannot block.
What it holds is not any one button's behaviour -- it is that NO button
raises, that the process still has exactly ONE Tk interpreter afterwards,
that no Toplevel is left behind, and that the window still repaints.

THE CENSUS, RE-MEASURED ON PACK M2.7 (py3.11, Agg, 1600x900 window,
3 panes, 3 lanes, 256 ingested agent intervals). Pack M2.6 made Undo and
Redo name the gesture and Pack M2.7 adds the Export Intervals... button,
so every number and half the status column moved: 38 buttons, 0
exceptions, 0 buttons that could not be invoked, 9 dialog calls (3
showwarning, 1 askopenfilename, 5 asksaveasfilename), 6 Toplevels built
and destroyed, one interpreter throughout:

  button              dialogs tops  what it left on the status bar
  Update Window       0       0     Window updated: 06:00:00 -> 12:00:00
  x2 (window)         0       0     Window: 03:00:00 -> 15:00:00
  /2 (window)         0       0     Window: 06:00:00 -> 12:00:00
  <- Prev             0       0     Window: 03:00:00 -> 09:00:00
  Next ->             0       0     Window: 06:00:00 -> 12:00:00
  x2 (step)           0       0     Step: 0 days 06:00:00
  /2 (step)           0       0     Step: 0 days 03:00:00
  Re-label            1       0     (No Selection warning; bar unchanged)
  Add                 1       0     (No Selection warning; bar unchanged)
  Undo                0       0     Undo: import track agent from
                                    column:prediction
  Manage...           0       1     (bar unchanged)
  Fill Gaps...        0       1     (bar unchanged)
  Delete              1       0     (No Selection warning; bar unchanged)
  Redo                0       0     Redo: import track agent from
                                    column:prediction
  By-Rule...          0       1     (bar unchanged)
  Clear...            0       1     (bar unchanged)
  Save Session        1       0     (asksaveasfilename -> "")
  Load Session        1       0     (askopenfilename -> "")
  Export Labels...    0       1     (bar unchanged)
  Export Intervals... 1       0     (asksaveasfilename -> ""; no file,
                                    no second dialog)
  Help (F1)           0       1     (bar unchanged)
  Reset Scale         0       0     Scales reset to auto
  Hide Panel          0       0     Scales reset to auto
  Home/Back/Forward/Subplots/Save  x3 panes, matplotlib's own toolbar;
                                   Save calls asksaveasfilename -> ""

The toolbar half of that list is matplotlib's and its size depends on the
matplotlib version, so the pin asserts the APP's own buttons by name and
then asserts the invariants over whatever else it found. TWENTY of the 23
are named in the module's `APP_BUTTONS`; the other three carry non-ASCII
text (the window `x2` / `/2` pair and Hide Panel's arrow) and are swept
without being named.

SKIPPED WITHOUT THE CASE FRAME -- see tests/test_packm2_5_live_launch.py.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import tkinter as tk
from tkinter import ttk

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVERS = os.path.join(REPO, "test_drivers")
C05_FRAME = os.path.join(REPO, "edit_pack", "campaign", "C05_cmmae",
                         "c05_frame.parquet")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(C05_FRAME),
    reason=("the feel-test driver reads edit_pack/campaign/C05_cmmae/"
            "c05_frame.parquet, which is gitignored campaign evidence and "
            "is absent on a clean checkout (CI included)"))

# The app's own buttons, by the text they carry. THREE more exist whose text
# is not ASCII (the window x2 / /2 pair and Hide Panel's arrow) and they are
# swept like the rest; they are simply not named here. Pack M2.7 adds
# "Export Intervals...", so it is 20 named + 3 = the 23 app buttons of the 38
# this sweep finds.
APP_BUTTONS = (
    "Update Window", "<- Prev", "Next ->", "x2", "/2", "Re-label", "Add",
    "Undo", "Redo", "Delete", "Manage...", "Fill Gaps...", "By-Rule...",
    "Clear...", "Save Session", "Load Session", "Export Labels...",
    "Export Intervals...", "Help (F1)", "Reset Scale",
)

DIALOG_KINDS = ("showinfo", "showwarning", "showerror", "askyesno",
                "askyesnocancel", "askokcancel", "asksaveasfilename",
                "askopenfilename", "askdirectory", "askstring",
                "askinteger", "askfloat")


@pytest.fixture
def counted(monkeypatch):
    """A GLOBAL dialog stub that COUNTS, forced before anything is pressed."""
    import tkinter.messagebox as mb
    import tkinter.filedialog as fd
    import tkinter.simpledialog as sd
    calls = []
    for kind in ("showinfo", "showwarning", "showerror", "askyesno",
                 "askyesnocancel", "askokcancel"):
        monkeypatch.setattr(
            mb, kind, lambda *a, _k=kind, **kw: calls.append(_k) or True)
    for kind in ("asksaveasfilename", "askopenfilename", "askdirectory"):
        monkeypatch.setattr(
            fd, kind, lambda *a, _k=kind, **kw: calls.append(_k) or "")
    for kind in ("askstring", "askinteger", "askfloat"):
        monkeypatch.setattr(sd, kind,
                            lambda *a, _k=kind, **kw: calls.append(_k))
    return calls


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def app(monkeypatch, tmp_path):
    if DRIVERS not in sys.path:
        sys.path.insert(0, DRIVERS)
    import drive_multilabel_thb as D
    monkeypatch.setattr(D, "CASE_DIR", str(tmp_path))
    monkeypatch.setattr(D, "AUTOSAVE_FOLDER",
                        os.path.join(str(tmp_path), "chronotagger_autosave"))
    monkeypatch.setattr(tk, "_default_root", None, raising=False)
    lbl = D.build(run=False)
    lbl.root.withdraw()
    yield lbl
    lbl.root.destroy()


def ascii_text(widget):
    try:
        return str(widget.cget("text")).encode("ascii", "replace").decode(
            "ascii")
    except Exception:
        return "<no text>"


def all_buttons(widget, out):
    for ch in widget.winfo_children():
        cls = ""
        try:
            cls = ch.winfo_class()
        except Exception:
            cls = ""
        if isinstance(ch, ttk.Button) or cls in ("TButton", "Button"):
            out.append(ch)
        all_buttons(ch, out)
    return out


def test_every_button_in_the_window_can_be_pressed_once(app, counted,
                                                        monkeypatch):
    assert tk._default_root is app.root
    found = all_buttons(app.root, [])
    texts = [ascii_text(b) for b in found]
    missing = [t for t in APP_BUTTONS if t not in texts]
    assert missing == [], missing
    assert len(found) >= len(APP_BUTTONS)

    noop = lambda *a, **k: None
    monkeypatch.setattr(tk.Misc, "wait_window", noop)
    monkeypatch.setattr(tk.Misc, "wait_visibility", noop)
    monkeypatch.setattr(tk.Misc, "grab_set", noop)

    outcomes = []
    errors = []
    tops_built = 0
    for btn, text in zip(found, texts):
        before = len(counted)
        known = set(map(str, app.root.winfo_children()))
        err = ""
        try:
            btn.invoke()
        except Exception as exc:            # pragma: no cover - the pin
            err = "%s: %s" % (type(exc).__name__, exc)
            errors.append((text, err))
        new_tops = [w for w in app.root.winfo_children()
                    if str(w) not in known and isinstance(w, tk.Toplevel)]
        tops_built += len(new_tops)
        for w in new_tops:
            w.destroy()
        outcomes.append((text, len(counted) - before, len(new_tops),
                         err or "-"))

    assert errors == [], outcomes
    # every dialog that opened was one of the stubbed entry points
    assert [c for c in counted if c not in DIALOG_KINDS] == [], counted
    # nothing modal left standing, one interpreter, window alive
    assert [w for w in app.root.winfo_children()
            if isinstance(w, tk.Toplevel)] == []
    assert tk._default_root is app.root
    assert bool(app.root.winfo_exists())
    assert tops_built >= 1, outcomes

    # and the window still repaints, with its three lanes
    app._update_plot()
    assert [p._strip_lane_count for p in app.panes] == [3, 3, 3]
