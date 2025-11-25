# src/chronotagger/labeler/mixins/help.py
"""
Help / shortcuts dialog mixin.

- F1 (and a 'Help' button) opens a modal dialog listing keyboard & mouse commands.
- Definitions live in two tuples so they're easy to keep in sync with bindings.
"""

from __future__ import annotations
from typing import Iterable, Tuple
import tkinter as tk
from tkinter import ttk


# Keep these in ONE place to avoid drift with the actual bindings.
KEYBOARD_SHORTCUTS: Tuple[Tuple[str, str], ...] = (
    ("1–9",               "Select class by index"),
    ("n / right-arrow",   "Next window"),
    ("p / left-arrow",    "Previous window"),
    ("a / Enter",         "Add label (use current class)"),
    ("d / Delete",        "Delete selected interval"),
    ("u",                 "Quick-select UNKNOWN class (if present)"),
    ("Ctrl+S",            "Save session"),
    ("Ctrl+E",            "Export intervals"),
    ("Ctrl+Z / Backspace","Undo"),
    ("Ctrl+Y / Shift+Backspace", "Redo"),
)

MULTI_PANE_SHORTCUTS: Tuple[Tuple[str, str], ...] = (
    ("Ctrl+Tab",          "Next tab"),
    ("Ctrl+Shift+Tab",    "Previous tab"),
    ("Ctrl+1...9",        "Jump to tab 1-9"),
    ("Ctrl+0",            "Jump to tab 10"),
    ("Right-click tab",   "Show tab menu (rename, refresh)"),
)

MOUSE_CONTROLS: Tuple[Tuple[str, str], ...] = (
    ("Left-click (time axes)",          "Two-click interval selection (click start, click end)"),
    ("Left-drag (time axes)",           "Box-select points → contiguous blocks previewed"),
    ("Click interval (strip)",          "Select interval"),
    ("Wheel",                           "Zoom around cursor"),
    ("Shift + Wheel",                   "Pan left/right"),
)


class HelpMixin:
    _help_window: tk.Toplevel | None = None

    def _open_help_dialog(self, _evt=None) -> None:
        if self._help_window and tk.Toplevel.winfo_exists(self._help_window):
            self._help_window.lift()
            return

        win = tk.Toplevel(self.root)
        self._help_window = win
        win.title("ChronoTagger — Shortcuts & Help")
        win.transient(self.root)
        win.grab_set()
        win.resizable(False, False)

        container = ttk.Frame(win, padding=12)
        container.grid(row=0, column=0, sticky="nsew")

        # Title / subtitle
        title = ttk.Label(container, text="Shortcuts", font=("Segoe UI", 12, "bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 6))

        nb = ttk.Notebook(container)
        nb.grid(row=1, column=0, sticky="nsew")

        kb_frame = ttk.Frame(nb, padding=8)
        mouse_frame = ttk.Frame(nb, padding=8)
        nb.add(kb_frame, text="Keyboard")
        nb.add(mouse_frame, text="Mouse")

        self._build_table(kb_frame, ("Keys", "Action"), KEYBOARD_SHORTCUTS)
        self._build_table(mouse_frame, ("Control", "Action"), MOUSE_CONTROLS)

        # Add multi-pane shortcuts tab if in multi-pane mode
        if getattr(self, 'multi_pane_mode', False):
            multi_pane_frame = ttk.Frame(nb, padding=8)
            nb.add(multi_pane_frame, text="Multi-Pane")
            self._build_table(multi_pane_frame, ("Keys", "Action"), MULTI_PANE_SHORTCUTS)

        # Footer
        footer = ttk.Frame(container)
        footer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(
            footer,
            text="Tip: press F1 any time to open this dialog.",
            foreground="#666",
        ).grid(row=0, column=0, sticky="w")
        tk.Button(footer, text="Close", command=win.destroy).grid(row=0, column=1, sticky="e")

        # Close on Esc / window X
        win.bind("<Escape>", lambda e: win.destroy())
        win.protocol("WM_DELETE_WINDOW", win.destroy)

        # Center on parent
        win.update_idletasks()
        px = self.root.winfo_rootx() + (self.root.winfo_width() - win.winfo_width()) // 2
        py = self.root.winfo_rooty() + (self.root.winfo_height() - win.winfo_height()) // 2
        win.geometry(f"+{max(px,0)}+{max(py,0)}")

    def _build_table(
        self,
        parent: tk.Misc,
        headings: Tuple[str, str],
        rows: Iterable[Tuple[str, str]],
    ) -> None:
        tree = ttk.Treeview(parent, columns=("c1", "c2"), show="headings", height=10)
        tree.grid(row=0, column=0, sticky="nsew")
        tree.heading("c1", text=headings[0])
        tree.heading("c2", text=headings[1])
        tree.column("c1", width=190, anchor="w")
        tree.column("c2", width=420, anchor="w")

        for k, desc in rows:
            tree.insert("", "end", values=(k, desc))

        sb = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.grid(row=0, column=1, sticky="ns")

        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
