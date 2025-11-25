# src/chronotagger/labeler/dialogs/label_manager.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import tkinter as tk
from tkinter import ttk, messagebox, colorchooser, simpledialog


@dataclass
class LabelManagerResult:
    classes: List[str]
    class_colors: Dict[str, str]
    rename_map: Dict[str, str]
    reassign_map: Dict[str, str]  # deleted_label -> target_label


class _ReassignDialog(tk.Toplevel):
    """Small modal used when deleting a label that has usage."""
    def __init__(self, parent, deleting: str, choices: List[str]) -> None:
        super().__init__(parent)
        self.title("Reassign intervals")
        self.resizable(False, False)
        self.result: Optional[str] = None

        ttk.Label(self, text=f"'{deleting}' is used. Reassign its intervals to:").pack(padx=10, pady=(10, 5))
        self._var = tk.StringVar(value=choices[0] if choices else "")
        self._menu = ttk.OptionMenu(self, self._var, self._var.get(), *choices)
        self._menu.pack(padx=10, pady=5, fill=tk.X)

        btns = ttk.Frame(self)
        btns.pack(padx=10, pady=(5, 10), fill=tk.X)
        tk.Button(btns, text="Cancel", command=self._cancel).pack(side=tk.RIGHT, padx=5)
        tk.Button(btns, text="OK", command=self._ok).pack(side=tk.RIGHT)

        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_visibility()
        self.focus()

    def _ok(self):
        self.result = self._var.get()
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


class LabelManagerDialog(tk.Toplevel):
    """
    Modal dialog to manage the label schema (names, order, colors).
    Returns LabelManagerResult or None on cancel.
    """
    def __init__(
        self,
        parent: tk.Misc,
        classes: List[str],
        class_colors: Dict[str, str],
        usage_counts: Dict[str, int],
        reserved: Set[str] = frozenset({"UNKNOWN"}),
    ) -> None:
        super().__init__(parent)
        self.title("Manage Labels")
        self.resizable(False, False)

        # Working copies
        self._reserved = {s.upper() for s in reserved}
        self._classes: List[str] = list(classes)
        self._colors: Dict[str, str] = dict(class_colors)
        self._counts: Dict[str, int] = {k: int(usage_counts.get(k, 0)) for k in self._classes}

        # Track semantic changes
        self._rename_map: Dict[str, str] = {}
        self._reassign_map: Dict[str, str] = {}

        # UI
        self._build_ui()

        self.result: Optional[LabelManagerResult] = None
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.wait_visibility()
        self.focus()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        cols = ("Name", "Color", "Used")
        self.tree = ttk.Treeview(main, columns=cols, show="headings", height=10, selectmode="browse")
        for c in cols:
            self.tree.heading(c, text=c)
        self.tree.column("Name", width=180)
        self.tree.column("Color", width=90)
        self.tree.column("Used", width=60, anchor=tk.E)
        self.tree.grid(row=0, column=0, rowspan=6, sticky="nsew")

        # Scrollbar
        sb = ttk.Scrollbar(main, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.grid(row=0, column=1, rowspan=6, sticky="ns", padx=(4, 0))

        # Buttons
        tk.Button(main, text="Add…", command=self._on_add).grid(row=0, column=2, sticky="ew", padx=(12, 0), pady=(0, 4))
        tk.Button(main, text="Rename…", command=self._on_rename).grid(row=1, column=2, sticky="ew", padx=(12, 0), pady=4)
        tk.Button(main, text="Change color…", command=self._on_color).grid(row=2, column=2, sticky="ew", padx=(12, 0), pady=4)
        tk.Button(main, text="Move up", command=lambda: self._move(-1)).grid(row=3, column=2, sticky="ew", padx=(12, 0), pady=4)
        tk.Button(main, text="Move down", command=lambda: self._move(+1)).grid(row=4, column=2, sticky="ew", padx=(12, 0), pady=4)
        tk.Button(main, text="Delete…", command=self._on_delete).grid(row=5, column=2, sticky="ew", padx=(12, 0), pady=(4, 0))

        # Footer
        footer = ttk.Frame(main)
        footer.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        footer.columnconfigure(0, weight=1)
        tk.Button(footer, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT, padx=5)
        tk.Button(footer, text="OK", command=self._on_ok).pack(side=tk.RIGHT)

        main.columnconfigure(0, weight=1)
        self._refresh_tree()

    def _refresh_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for name in self._classes:
            color = self._colors.get(name, "#cccccc")
            used = self._counts.get(name, 0)
            self.tree.insert("", "end", values=(name, color, used))

    def _selected_name(self) -> Optional[str]:
        sel = self.tree.selection()
        if not sel:
            return None
        (name, _, _) = self.tree.item(sel[0], "values")
        return name

    # ---------- actions ----------
    def _on_add(self) -> None:
        name = simpledialog.askstring("New label", "Label name:", parent=self)
        if not name:
            return
        name = name.strip()
        if not name:
            messagebox.showerror("Invalid name", "Name cannot be empty.", parent=self)
            return
        if any(n.lower() == name.lower() for n in self._classes):
            messagebox.showerror("Duplicate", f"A label named '{name}' already exists.", parent=self)
            return

        # Choose color
        _rgb, hexcolor = colorchooser.askcolor(title=f"Choose color for '{name}'", parent=self)
        if not hexcolor:
            hexcolor = "#9aa5b1"  # sensible default if user cancels

        self._classes.append(name)
        self._colors[name] = hexcolor
        self._counts.setdefault(name, 0)
        self._refresh_tree()

    def _on_rename(self) -> None:
        cur = self._selected_name()
        if not cur:
            return
        if cur.upper() in self._reserved:
            messagebox.showwarning("Reserved", f"'{cur}' is reserved and cannot be renamed.", parent=self)
            return

        new = simpledialog.askstring("Rename label", f"New name for '{cur}':", parent=self, initialvalue=cur)
        if not new:
            return
        new = new.strip()
        if not new or any(n.lower() == new.lower() for n in self._classes if n != cur):
            messagebox.showerror("Invalid name", f"'{new}' is not allowed or already exists.", parent=self)
            return

        # Apply rename locally
        idx = self._classes.index(cur)
        self._classes[idx] = new
        self._colors[new] = self._colors.pop(cur, "#cccccc")
        self._counts[new] = self._counts.pop(cur, 0)

        # Track semantic rename
        # If 'cur' was already renamed earlier, map original->new through chain
        original = next((k for k, v in self._rename_map.items() if v == cur), cur)
        self._rename_map[original] = new

        # If 'cur' was a reassign target for others, retarget to 'new'
        for k, v in list(self._reassign_map.items()):
            if v == cur:
                self._reassign_map[k] = new

        self._refresh_tree()
        # Reselect renamed row
        for item in self.tree.get_children():
            if self.tree.item(item, "values")[0] == new:
                self.tree.selection_set(item)
                break

    def _on_color(self) -> None:
        cur = self._selected_name()
        if not cur:
            return
        _rgb, hexcolor = colorchooser.askcolor(title=f"Choose color for '{cur}'", parent=self, initialcolor=self._colors.get(cur, "#cccccc"))
        if not hexcolor:
            return
        self._colors[cur] = hexcolor
        self._refresh_tree()

    def _move(self, direction: int) -> None:
        cur = self._selected_name()
        if not cur:
            return
        i = self._classes.index(cur)
        j = i + direction
        if j < 0 or j >= len(self._classes):
            return
        self._classes[i], self._classes[j] = self._classes[j], self._classes[i]
        self._refresh_tree()
        # Reselect moved item
        for item in self.tree.get_children():
            if self.tree.item(item, "values")[0] == cur:
                self.tree.selection_set(item)
                break

    def _on_delete(self) -> None:
        cur = self._selected_name()
        if not cur:
            return
        if cur.upper() in self._reserved:
            messagebox.showwarning("Reserved", f"'{cur}' is reserved and cannot be deleted.", parent=self)
            return

        used = self._counts.get(cur, 0)
        if used > 0:
            choices = [n for n in self._classes if n != cur]
            if not choices:
                messagebox.showerror("Cannot delete", "No other label to reassign intervals to.", parent=self)
                return
            dlg = _ReassignDialog(self, deleting=cur, choices=choices)
            if dlg.result is None:
                return
            self._reassign_map[cur] = dlg.result

        # remove locally
        self._classes.remove(cur)
        self._colors.pop(cur, None)
        self._counts.pop(cur, None)
        self._refresh_tree()

    def _on_ok(self) -> None:
        self.result = LabelManagerResult(
            classes=list(self._classes),
            class_colors=dict(self._colors),
            rename_map=dict(self._rename_map),
            reassign_map=dict(self._reassign_map),
        )
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.destroy()
