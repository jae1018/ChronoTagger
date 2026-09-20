# src/chronotagger/labeler/dialogs/_from_column.py
"""Make a lane out of a data column. Pack M3.1.

`add_track_from_column` has existed since Pack M1 and nothing on screen
called it: it was reachable only from a driver script. This is its door.

IT NEVER SEES THE APP. The column list and the preview arrive as two
CALLABLES -- `columns_fn()` and `preview_fn(column, gap_tolerance)` --
so the box can be built and driven with two plain functions, and so
Pack M3.2's wizard can hand it a different pair.

NOTHING IS IMPORTED HERE EITHER. OK stages a description of the import;
the Manage Lanes OK runs it, inside the one gesture that also carries
the adds, the renames and the deletes, so ONE Ctrl+Z takes the lane and
its intervals back out together.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import tkinter as tk
from tkinter import ttk, messagebox

from chronotagger.core.tracks import (lane_display_key, lane_id_from_name,
                                      validate_track_id)


def refusal_line(msg) -> str:
    """One plain sentence out of the ingest's own refusal.

    The ingest's message is a paragraph -- it names the two different
    causes and what to do about each -- and a paragraph does not belong
    in a one-line preview. The FIRST sentence carries the numbers, which
    is the part that changes with the tolerance.
    """
    head = str(msg or "").split(". ")[0].strip()
    if head and not head.endswith("."):
        head += "."
    if "gap tolerance" in head:
        head += " Try a bigger gap tolerance."
    return head


class FromColumnDialog(tk.Toplevel):
    """Pick a column, a gap tolerance and a name; see what it would make."""

    def __init__(
        self,
        parent,
        columns_fn: Callable[[], List[dict]],
        preview_fn: Optional[Callable[..., dict]] = None,
        existing_ids=(),
        existing_names=(),
        title: str = "Lane From Column",
    ) -> None:
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.result: Optional[dict] = None
        self._preview_fn = preview_fn
        self._ids = [str(i) for i in existing_ids]
        self._names = {lane_display_key(n) for n in existing_names}
        self._id_touched = False
        self._last = {"ok": False, "why": "", "n_intervals": 0,
                      "n_classes": 0, "classes": []}
        try:
            self._columns = list(columns_fn() or [])
        except Exception:
            self._columns = []

        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="Column:").grid(row=0, column=0, sticky="nw")
        self.tree = ttk.Treeview(main, columns=("Column", "Values"),
                                 show="headings", height=8,
                                 selectmode="browse")
        self.tree.heading("Column", text="Column")
        self.tree.heading("Values", text="Distinct values")
        self.tree.column("Column", width=200)
        self.tree.column("Values", width=110, anchor=tk.E)
        self.tree.grid(row=0, column=1, sticky="ew", pady=2)
        for c in self._columns:
            self.tree.insert("", "end", iid=c["name"],
                             values=(c["name"], c["distinct"]))
        if not self._columns:
            ttk.Label(main, foreground="#666666",
                      text="no column in this frame has 20 or fewer "
                           "text or whole-number values").grid(
                row=1, column=1, sticky="w")

        self.tol_var = tk.StringVar(
            master=self,
            value=(self._columns[0]["default_gap_tolerance"]
                   if self._columns else ""))
        ttk.Label(main, text="Gap tolerance:").grid(row=2, column=0,
                                                    sticky="w",
                                                    pady=(6, 0))
        self.tol_entry = ttk.Entry(main, textvariable=self.tol_var,
                                   width=34)
        self.tol_entry.grid(row=2, column=1, sticky="ew", pady=(6, 2))
        ttk.Label(main, text="the automatic value; '1h' or '15min' also "
                             "work", foreground="#666666").grid(
            row=3, column=1, sticky="w")

        self.name_var = tk.StringVar(master=self, value="")
        ttk.Label(main, text="Name:").grid(row=4, column=0, sticky="w",
                                           pady=(6, 0))
        self.name_entry = ttk.Entry(main, textvariable=self.name_var,
                                    width=34)
        self.name_entry.grid(row=4, column=1, sticky="ew", pady=(6, 2))

        self.id_var = tk.StringVar(master=self, value="")
        ttk.Label(main, text="Id:").grid(row=5, column=0, sticky="w")
        self.id_entry = ttk.Entry(main, textvariable=self.id_var, width=34)
        self.id_entry.grid(row=5, column=1, sticky="ew", pady=2)
        ttk.Label(main, text="made from the name; frozen after OK",
                  foreground="#666666").grid(row=6, column=1, sticky="w")

        self.locked_var = tk.BooleanVar(master=self, value=True)
        ttk.Checkbutton(main, text="Locked (it came from a file)",
                        variable=self.locked_var).grid(
            row=7, column=1, sticky="w", pady=(6, 0))

        self.preview_var = tk.StringVar(master=self,
                                        value="pick a column")
        ttk.Label(main, textvariable=self.preview_var,
                  wraplength=360, justify=tk.LEFT).grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(8, 0))

        footer = ttk.Frame(main)
        footer.grid(row=9, column=0, columnspan=2, sticky="ew",
                    pady=(12, 0))
        ttk.Button(footer, text="Cancel",
                   command=self._on_cancel).pack(side=tk.RIGHT, padx=5)
        ttk.Button(footer, text="OK",
                   command=self._on_ok).pack(side=tk.RIGHT)
        ttk.Button(footer, text="Preview",
                   command=self.refresh_preview).pack(side=tk.LEFT)
        main.columnconfigure(1, weight=1)

        self.name_var.trace_add("write", self._name_changed)
        self.id_entry.bind("<KeyRelease>", self._id_edited)
        self.id_entry.bind("<<Paste>>", self._id_edited)
        self.tree.bind("<<TreeviewSelect>>", self._column_picked)
        self.tol_entry.bind("<Return>", lambda e: self.refresh_preview())

        from ._placement import center_on_parent
        center_on_parent(self, parent)
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.wait_visibility()
        self.focus()

    # ---------- the pieces that follow each other ----------
    def _id_edited(self, _evt=None) -> None:
        self._id_touched = True

    def _name_changed(self, *_a) -> None:
        if self._id_touched:
            return
        try:
            self.id_var.set(lane_id_from_name(self.name_var.get(),
                                              self._ids))
        except Exception:
            pass

    def selected_column(self) -> Optional[str]:
        sel = self.tree.selection()
        return sel[0] if sel else None

    def _column_picked(self, _evt=None) -> None:
        col = self.selected_column()
        if not col:
            return
        if not str(self.name_var.get() or "").strip():
            self.name_var.set(col)
        for c in self._columns:
            if c["name"] == col and not str(self.tol_var.get() or "").strip():
                self.tol_var.set(c["default_gap_tolerance"])
        self.refresh_preview()

    # ---------- the preview ----------
    def refresh_preview(self) -> dict:
        col = self.selected_column()
        if not col:
            self._last = {"ok": False, "why": "pick a column",
                          "n_intervals": 0, "n_classes": 0, "classes": []}
            self.preview_var.set("pick a column")
            return self._last
        if self._preview_fn is None:
            self._last = {"ok": False, "why": "no preview is available",
                          "n_intervals": 0, "n_classes": 0, "classes": []}
            self.preview_var.set("no preview is available")
            return self._last
        tol = str(self.tol_var.get() or "").strip() or None
        try:
            res = dict(self._preview_fn(col, tol) or {})
        except Exception as exc:
            res = {"ok": False, "why": str(exc), "n_intervals": 0,
                   "n_classes": 0, "classes": []}
        self._last = res
        if res.get("ok"):
            n, m = int(res.get("n_intervals", 0)), int(
                res.get("n_classes", 0))
            self.preview_var.set(
                "would make %d interval%s in %d class%s"
                % (n, "" if n == 1 else "s", m, "" if m == 1 else "es"))
        else:
            self.preview_var.set(refusal_line(res.get("why")))
        return res

    # ---------- OK / Cancel ----------
    def _on_ok(self) -> None:
        col = self.selected_column()
        if not col:
            messagebox.showerror("No column", "Pick a column first.",
                                 parent=self)
            return
        res = self.refresh_preview()
        if not res.get("ok"):
            messagebox.showerror("Cannot import that column",
                                 refusal_line(res.get("why")),
                                 parent=self)
            return
        name = " ".join(str(self.name_var.get() or "").split()) or col
        if lane_display_key(name) in self._names:
            messagebox.showerror(
                "Duplicate name",
                "A lane called '%s' already exists. Give this one a name "
                "of its own so the Lane list can tell them apart."
                % (name,), parent=self)
            return
        tid = str(self.id_var.get() or "").strip() or \
            lane_id_from_name(name, self._ids)
        try:
            validate_track_id(tid)
        except ValueError as exc:
            messagebox.showerror("Invalid id", str(exc), parent=self)
            return
        if tid in self._ids:
            messagebox.showerror(
                "Duplicate id",
                "The id '%s' is already taken. An id names an export "
                "column and a label-map file, so it has to be unique."
                % (tid,), parent=self)
            return
        self.result = {
            "id": tid,
            "name": name,
            "column": col,
            "gap_tolerance": (str(self.tol_var.get() or "").strip()
                              or None),
            "locked": bool(self.locked_var.get()),
            "classes": list(res.get("classes") or []),
            "n_intervals": int(res.get("n_intervals", 0)),
        }
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.destroy()
