"""Withdrawn-root GUI harness (Pack 8 R17).

The gather measured 27 of 27 GUI-wired callables in `layout_builder/` and
`quickstart/` with ZERO test references.  Pack 7 tested the designer's
EMISSION half through a fake host with stub Tk variables, which reaches no
widget at all; this pack edits the widgets, so the pins need real ones.

WHAT THIS MODULE GUARANTEES, and why each rule is here:

  * ONE real, WITHDRAWN `tk.Tk` per test, built through Pack 6's bounded
    retry and destroyed in a `finally`.  A real root is what makes a real
    `Toplevel`, a real `ttk.Combobox` and a real `tk.Canvas` possible.
  * NOTHING IS EVER MAPPED ON SCREEN.  `tk.Toplevel.__init__` is wrapped
    so every Toplevel -- including subclasses defined long before this
    module was imported, which is why the wrap is on `__init__` and not
    on the class -- withdraws itself the instant it exists.
  * `messagebox`, `filedialog`, `simpledialog` and `colorchooser` are ALL
    stubbed.  The first is the burned lesson: an unstubbed messagebox
    self-creates a Tk root and lands a modal on the developer's live
    screen, blocking the run.  `simpledialog` and `colorchooser` are here
    because `LabelManagerDialog._on_add` / `_on_rename` / `_on_color`
    reach both, and this pack drives that dialog.
  * `grab_set`, `wait_visibility` and `wait_window` are no-ops.
    `wait_visibility` is not cosmetic: a WITHDRAWN window never becomes
    visible, and `LabelManagerDialog.__init__` calls it, so an unstubbed
    probe of that dialog hangs forever.
  * `transient` is deliberately NOT stubbed.  Five live call sites reach
    it (`label_manager.py:37,83`, `label_by_rule.py:214`,
    `overlap_resolution.py:59`, `events/selection.py:1725`) and every one
    is on a Toplevel this module has already withdrawn, where
    `transient()` is a no-op anyway -- measured: the harness maps
    nothing.  A stub would add a patch that changes nothing.  (This
    docstring claimed the stub existed until Pack 8 v2; the code never
    had it.  V2 F3.)
  * Everything is applied through `monkeypatch`, so it is undone at the
    end of every test and cannot leak into the rest of the suite.

Every stub RECORDS.  The recorder is the assertion surface: tests read
`gui.messages`, `gui.saves` and friends rather than guessing.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, simpledialog

import pytest

from chronotagger.labeler.mixins.view_build.window import _new_tk_root


class DialogRecorder:
    """Records every modal the code under test tried to open."""

    def __init__(self) -> None:
        self.root = None
        # (kind, title, message) for every messagebox call
        self.messages = []
        # (kind, kwargs) for every filedialog call
        self.saves = []
        self.opens = []
        self.strings = []
        self.colors = []
        self.waits = []
        # Queued answers.  A queue rather than a single value because one
        # flow can open the same dialog twice (save-as on a retry).
        self.save_returns = []
        self.open_returns = []
        self.string_returns = []
        self.color_returns = []
        self.yes = True
        # Hook fired instead of a real wait_window(widget); a test that
        # wants to DRIVE a modal sets this.
        self.on_wait_window = None

    def next_save(self) -> str:
        return self.save_returns.pop(0) if self.save_returns else ""

    def next_open(self) -> str:
        return self.open_returns.pop(0) if self.open_returns else ""

    def titles(self):
        return [title for _, title, _ in self.messages]


def _stub_messagebox(monkeypatch, rec):
    def make(kind, ret):
        def stub(*args, **kwargs):
            title = args[0] if args else kwargs.get("title", "")
            body = args[1] if len(args) > 1 else kwargs.get("message", "")
            rec.messages.append((kind, str(title), str(body)))
            return rec.yes if ret is None else ret
        return stub

    for name, ret in (("showerror", "ok"), ("showwarning", "ok"),
                      ("showinfo", "ok")):
        monkeypatch.setattr(messagebox, name, make(name, ret))
    for name in ("askyesno", "askokcancel", "askyesnocancel", "askretrycancel"):
        monkeypatch.setattr(messagebox, name, make(name, None))


def _stub_filedialog(monkeypatch, rec):
    def save(*args, **kwargs):
        rec.saves.append(("asksaveasfilename", dict(kwargs)))
        return rec.next_save()

    def open_(*args, **kwargs):
        rec.opens.append(("askopenfilename", dict(kwargs)))
        return rec.next_open()

    def directory(*args, **kwargs):
        rec.opens.append(("askdirectory", dict(kwargs)))
        return rec.next_open()

    monkeypatch.setattr(filedialog, "asksaveasfilename", save)
    monkeypatch.setattr(filedialog, "askopenfilename", open_)
    monkeypatch.setattr(filedialog, "askdirectory", directory)


def _stub_simpledialog(monkeypatch, rec):
    def askstring(*args, **kwargs):
        rec.strings.append((args, dict(kwargs)))
        return rec.string_returns.pop(0) if rec.string_returns else None

    monkeypatch.setattr(simpledialog, "askstring", askstring)


def _stub_colorchooser(monkeypatch, rec):
    def askcolor(*args, **kwargs):
        rec.colors.append((args, dict(kwargs)))
        if rec.color_returns:
            chosen = rec.color_returns.pop(0)
            return (None, chosen)
        return (None, None)

    monkeypatch.setattr(colorchooser, "askcolor", askcolor)


def _silence_windows(monkeypatch, rec):
    real_init = tk.Toplevel.__init__

    def hidden_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        try:
            self.withdraw()
        except tk.TclError:
            pass

    def wait_window(self, window=None):
        rec.waits.append(window)
        if rec.on_wait_window is not None:
            rec.on_wait_window(window)

    monkeypatch.setattr(tk.Toplevel, "__init__", hidden_init)
    monkeypatch.setattr(tk.Misc, "grab_set", lambda self: None)
    monkeypatch.setattr(tk.Misc, "wait_visibility", lambda self, w=None: None)
    monkeypatch.setattr(tk.Misc, "wait_window", wait_window)


@pytest.fixture
def gui(monkeypatch):
    """A withdrawn Tk root with every modal stubbed and recorded."""
    rec = DialogRecorder()
    _stub_messagebox(monkeypatch, rec)
    _stub_filedialog(monkeypatch, rec)
    _stub_simpledialog(monkeypatch, rec)
    _stub_colorchooser(monkeypatch, rec)
    _silence_windows(monkeypatch, rec)

    root = _new_tk_root()
    root.withdraw()
    rec.root = root
    try:
        yield rec
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass
