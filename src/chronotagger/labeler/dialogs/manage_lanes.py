# src/chronotagger/labeler/dialogs/manage_lanes.py
"""The Manage Lanes box. Pack M3.1.

Until this pack a lane could only be DECLARED in a driver script. The
app could lock one, hide one and switch between them, and that was the
whole of it: no add, no rename, no reorder, no delete, and the only
lane that ever appeared after launch came from `add_track_from_column`,
which nothing on screen calls.

THE SHAPE IS LabelManagerDialog's, deliberately. Working copies staged
INSIDE the dialog, a result object, OK / Cancel, WM_DELETE_WINDOW =
Cancel, centred over the window that opened it, modal through
transient + grab_set, and the caller waits with `wait_window`. Nothing
outside this box changes until OK, and Cancel discards everything --
staged deletes and staged column imports included.

IT NEVER TOUCHES THE APP. It is handed the lane table (copied), a
per-lane interval count, and two OPTIONAL callables for the column
feature. That is what lets Pack M3.2's wizard construct the same class
with no labeler at all, zero counts and its own extra button; the
constructor takes its arguments KEYWORD-ONLY after `interval_counts`
precisely so a later keyword can be added without touching this
pack's call site.

WHAT IT RETURNS is the staged END STATE plus what was deleted and what
was imported -- not a list of operations. The caller validates that end
state and publishes it in ONE gesture, so ONE Ctrl+Z puts everything
back, a deleted lane with its intervals included.

THE RULES ARE NOT WRITTEN HERE. Which lane may be deleted, which may be
hidden, what id a name makes and how `order` is renumbered all live in
`chronotagger.core.tracks`, because the caller re-checks the end state
with the same functions and two copies of one rule is how they stop
agreeing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import tkinter as tk
from tkinter import ttk, messagebox

from chronotagger.core.tracks import (Track, lane_delete_refusal,
                                      lane_display_key, lane_hide_refusal,
                                      lane_id_from_name, lane_rows_move,
                                      renumber_lane_rows, track_ids,
                                      validate_track_id)


@dataclass
class ManageLanesResult:
    """The staged END STATE, and what it cost to get there.

    lanes              every lane, in PAINTED order, as `Track.to_dict`
    added              ids that did not exist when the box opened
    renamed            {id: new display name} for lanes that already
                       existed and whose name changed
    deleted            ids that existed when the box opened and do not
                       exist now
    deleted_intervals  how many intervals go with those lanes
    imports            staged column imports, each a dict the caller
                       feeds to the ingest inside the one gesture
    reordered          True when the lanes that already existed now sit
                       in a different order
    flagged            True when a lane that already existed changed its
                       locked or visible flag
    """

    lanes: List[dict] = field(default_factory=list)
    added: List[str] = field(default_factory=list)
    renamed: Dict[str, str] = field(default_factory=dict)
    deleted: List[str] = field(default_factory=list)
    deleted_intervals: int = 0
    imports: List[dict] = field(default_factory=list)
    reordered: bool = False
    flagged: bool = False

    def is_empty(self) -> bool:
        """True when OK would change nothing at all."""
        return not (self.added or self.renamed or self.deleted
                    or self.imports or self.reordered or self.flagged)


def split_classes(text):
    """A comma-separated class list, trimmed, empties dropped.

    Returns (classes, error).  `error` is a plain sentence or "".
    """
    names = [p.strip() for p in str(text or "").split(",")]
    names = [n for n in names if n]
    if not names:
        return [], "Give the lane at least one class, separated by commas."
    seen = set()
    for n in names:
        if n.lower() in seen:
            return [], "The class '%s' is listed twice." % (n,)
        seen.add(n.lower())
    return names, ""


class _AddLaneDialog(tk.Toplevel):
    """Name, Id and Classes for ONE new lane.

    The Id is filled in FROM THE NAME as the user types, and stops
    following the moment he edits the Id field himself -- a name is
    what he is thinking about and an id is a detail he may or may not
    care about, and the one thing that must not happen is the id
    quietly changing under an id he chose.
    """

    def __init__(self, parent, existing_ids=(), existing_names=(),
                 title="Add Lane") -> None:
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.result: Optional[dict] = None
        self._ids = [str(i) for i in existing_ids]
        self._names = {lane_display_key(n) for n in existing_names}
        self._id_touched = False

        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        self.name_var = tk.StringVar(master=self, value="")
        self.id_var = tk.StringVar(master=self, value="")
        self.classes_var = tk.StringVar(master=self, value="")

        ttk.Label(main, text="Name:").grid(row=0, column=0, sticky="w")
        self.name_entry = ttk.Entry(main, textvariable=self.name_var,
                                    width=34)
        self.name_entry.grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(main, text="Id:").grid(row=1, column=0, sticky="w")
        self.id_entry = ttk.Entry(main, textvariable=self.id_var, width=34)
        self.id_entry.grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Label(main, text="made from the name; frozen after OK",
                  foreground="#666666").grid(row=2, column=1, sticky="w")

        ttk.Label(main, text="Classes:").grid(row=3, column=0, sticky="w",
                                              pady=(6, 0))
        self.classes_entry = ttk.Entry(main, textvariable=self.classes_var,
                                       width=34)
        self.classes_entry.grid(row=3, column=1, sticky="ew", pady=(6, 2))
        ttk.Label(main, text="comma-separated, at least one",
                  foreground="#666666").grid(row=4, column=1, sticky="w")

        footer = ttk.Frame(main)
        footer.grid(row=5, column=0, columnspan=2, sticky="ew",
                    pady=(12, 0))
        ttk.Button(footer, text="Cancel",
                   command=self._on_cancel).pack(side=tk.RIGHT, padx=5)
        ttk.Button(footer, text="OK",
                   command=self._on_ok).pack(side=tk.RIGHT)
        main.columnconfigure(1, weight=1)

        self.name_var.trace_add("write", self._name_changed)
        self.id_entry.bind("<KeyRelease>", self._id_edited)
        self.id_entry.bind("<<Paste>>", self._id_edited)

        from ._placement import center_on_parent
        center_on_parent(self, parent)
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.wait_visibility()
        self.focus()

    # ---------- the id follows the name until it does not ----------
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

    # ---------- OK / Cancel ----------
    def _on_ok(self) -> None:
        name = " ".join(str(self.name_var.get() or "").split())
        if not name:
            messagebox.showerror("Invalid name", "Give the lane a name.",
                                 parent=self)
            return
        if lane_display_key(name) in self._names:
            messagebox.showerror(
                "Duplicate name",
                "A lane called '%s' already exists. Give this one a name "
                "of its own so the Lane list can tell them apart."
                % (name,), parent=self)
            return
        tid = str(self.id_var.get() or "").strip()
        if not tid:
            tid = lane_id_from_name(name, self._ids)
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
        classes, err = split_classes(self.classes_var.get())
        if err:
            messagebox.showerror("Classes", err, parent=self)
            return
        self.result = {"id": tid, "name": name, "classes": classes}
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.destroy()


class _RenameLaneDialog(tk.Toplevel):
    """A new DISPLAY NAME for one lane. The id never moves."""

    def __init__(self, parent, current, other_names=(),
                 title="Rename Lane") -> None:
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.result: Optional[str] = None
        self._others = {lane_display_key(n) for n in other_names}

        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)
        ttk.Label(main, text="New name for '%s':" % (current,)).pack(
            anchor=tk.W)
        self.name_var = tk.StringVar(master=self, value=str(current or ""))
        self.entry = ttk.Entry(main, textvariable=self.name_var, width=34)
        self.entry.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(main, text="the id does not change",
                  foreground="#666666").pack(anchor=tk.W)

        footer = ttk.Frame(main)
        footer.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(footer, text="Cancel",
                   command=self._on_cancel).pack(side=tk.RIGHT, padx=5)
        ttk.Button(footer, text="OK",
                   command=self._on_ok).pack(side=tk.RIGHT)

        from ._placement import center_on_parent
        center_on_parent(self, parent)
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.wait_visibility()
        self.focus()

    def _on_ok(self) -> None:
        name = " ".join(str(self.name_var.get() or "").split())
        if not name:
            messagebox.showerror("Invalid name", "Give the lane a name.",
                                 parent=self)
            return
        if lane_display_key(name) in self._others:
            messagebox.showerror(
                "Duplicate name",
                "A lane called '%s' already exists. Give this one a name "
                "of its own so the Lane list can tell them apart."
                % (name,), parent=self)
            return
        self.result = name
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.destroy()


class ManageLanesDialog(tk.Toplevel):
    """Add, rename, reorder, lock, hide and delete lanes. Staged."""

    def __init__(
        self,
        parent: tk.Misc,
        lanes,
        interval_counts=None,
        *,
        columns_fn: Optional[Callable[[], List[dict]]] = None,
        preview_fn: Optional[Callable[..., dict]] = None,
        title: str = "Manage Lanes",
    ) -> None:
        super().__init__(parent)
        self.title(title)

        # Working copies. `Track.from_dict` accepts a Track and copies
        # it, so neither the caller's rows nor its objects are touched.
        self._rows: List[Track] = [Track.from_dict(t) for t in (lanes or [])]
        renumber_lane_rows(self._rows)
        self._start_ids = [t.id for t in self._rows]
        self._start_names = {t.id: t.name for t in self._rows}
        self._start_order = list(self._start_ids)
        self._start_flags = {t.id: (bool(t.locked), bool(t.visible))
                             for t in self._rows}
        self._counts: Dict[str, int] = {
            str(k): int(v) for k, v in dict(interval_counts or {}).items()}
        self._imports: List[dict] = []
        self._columns_fn = columns_fn
        self._preview_fn = preview_fn

        self._build_ui()

        self.result: Optional[ManageLanesResult] = None
        from ._placement import center_on_parent
        center_on_parent(self, parent)
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.wait_visibility()
        self.focus()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        cols = ("Name", "Id", "Classes", "Intervals", "Locked", "Visible")
        self.tree = ttk.Treeview(main, columns=cols, show="headings",
                                 height=10, selectmode="browse")
        for c in cols:
            self.tree.heading(c, text=c)
        self.tree.column("Name", width=170)
        self.tree.column("Id", width=110)
        self.tree.column("Classes", width=70, anchor=tk.E)
        self.tree.column("Intervals", width=80, anchor=tk.E)
        self.tree.column("Locked", width=60, anchor=tk.CENTER)
        self.tree.column("Visible", width=60, anchor=tk.CENTER)
        self.tree.grid(row=0, column=0, rowspan=9, sticky="nsew")

        sb = ttk.Scrollbar(main, orient=tk.VERTICAL,
                           command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.grid(row=0, column=1, rowspan=9, sticky="ns", padx=(4, 0))

        self.buttons: Dict[str, ttk.Button] = {}
        specs = [("add", "Add...", self._on_add)]
        if self._columns_fn is not None:
            specs.append(("column", "From column...", self._on_from_column))
        specs += [
            ("rename", "Rename...", self._on_rename),
            ("up", "Move up", lambda: self._on_move(-1)),
            ("down", "Move down", lambda: self._on_move(+1)),
            ("lock", "Lock/Unlock", self._on_lock),
            ("show", "Show/Hide", self._on_show_hide),
            ("delete", "Delete...", self._on_delete),
        ]
        for i, (key, text, cmd) in enumerate(specs):
            b = ttk.Button(main, text=text, command=cmd)
            b.grid(row=i, column=2, sticky="ew", padx=(12, 0), pady=2)
            self.buttons[key] = b

        footer = ttk.Frame(main)
        footer.grid(row=9, column=0, columnspan=3, sticky="ew",
                    pady=(10, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Button(footer, text="Cancel",
                   command=self._on_cancel).pack(side=tk.RIGHT, padx=5)
        ttk.Button(footer, text="OK",
                   command=self._on_ok).pack(side=tk.RIGHT)

        main.columnconfigure(0, weight=1)
        main.rowconfigure(8, weight=1)
        self._refresh_tree()

    def _refresh_tree(self, select=None) -> None:
        keep = select or self._selected_id()
        self.tree.delete(*self.tree.get_children())
        for row in self._rows:
            self.tree.insert(
                "", "end", iid=row.id,
                values=(row.name or row.id, row.id, len(row.classes),
                        self._counts.get(row.id, 0),
                        "yes" if row.locked else "no",
                        "yes" if row.visible else "no"))
        if keep and self.tree.exists(keep):
            self.tree.selection_set(keep)
            self.tree.focus(keep)
        elif self._rows:
            self.tree.selection_set(self._rows[0].id)
            self.tree.focus(self._rows[0].id)

    def _selected_id(self) -> Optional[str]:
        sel = self.tree.selection()
        return sel[0] if sel else None

    def _selected_row(self) -> Optional[Track]:
        tid = self._selected_id()
        return next((t for t in self._rows if t.id == tid), None)

    def _other_names(self, except_id=None):
        return [t.name or t.id for t in self._rows if t.id != except_id]

    # ---------- actions ----------
    def _on_add(self) -> None:
        dlg = _AddLaneDialog(self, existing_ids=track_ids(self._rows),
                             existing_names=self._other_names())
        self.wait_window(dlg)
        if not dlg.result:
            return
        row = Track(id=dlg.result["id"], name=dlg.result["name"],
                    classes=list(dlg.result["classes"]),
                    order=len(self._rows))
        self._rows.append(row)
        renumber_lane_rows(self._rows)
        self._counts.setdefault(row.id, 0)
        self._refresh_tree(select=row.id)

    def _on_from_column(self) -> None:
        """Only ever built when a column callable was handed in."""
        if self._columns_fn is None:
            return
        from ._from_column import FromColumnDialog
        dlg = FromColumnDialog(self, columns_fn=self._columns_fn,
                               preview_fn=self._preview_fn,
                               existing_ids=track_ids(self._rows),
                               existing_names=self._other_names())
        self.wait_window(dlg)
        if not dlg.result:
            return
        r = dlg.result
        row = Track(id=r["id"], name=r["name"], classes=list(r["classes"]),
                    locked=bool(r.get("locked", True)),
                    order=len(self._rows))
        self._rows.append(row)
        renumber_lane_rows(self._rows)
        self._counts[row.id] = int(r.get("n_intervals", 0))
        self._imports.append(r)
        self._refresh_tree(select=row.id)

    def _on_rename(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        dlg = _RenameLaneDialog(self, row.name or row.id,
                                other_names=self._other_names(row.id))
        self.wait_window(dlg)
        if not dlg.result:
            return
        row.name = dlg.result
        self._refresh_tree(select=row.id)

    def _on_move(self, delta) -> None:
        tid = self._selected_id()
        if not tid:
            return
        if lane_rows_move(self._rows, tid, delta):
            self._refresh_tree(select=tid)

    def _on_lock(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        row.locked = not bool(row.locked)
        self._refresh_tree(select=row.id)

    def _on_show_hide(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        if row.visible:
            why = lane_hide_refusal(self._rows, row.id)
            if why:
                messagebox.showwarning("Cannot hide", "Lane %s." % (why,),
                                       parent=self)
                return
        row.visible = not bool(row.visible)
        self._refresh_tree(select=row.id)

    def _on_delete(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        why = lane_delete_refusal(self._rows, row.id)
        if why:
            messagebox.showwarning("Cannot delete", "Lane %s." % (why,),
                                   parent=self)
            return
        n = int(self._counts.get(row.id, 0))
        name = row.name or row.id
        if n:
            question = ("Delete '%s' and its %d interval%s?"
                        % (name, n, "" if n == 1 else "s"))
        else:
            question = "Delete '%s'?" % (name,)
        if not messagebox.askyesno("Delete lane", question, parent=self):
            return
        self._rows = [t for t in self._rows if t.id != row.id]
        renumber_lane_rows(self._rows)
        self._imports = [d for d in self._imports if d["id"] != row.id]
        self._refresh_tree()

    # ---------- OK / Cancel ----------
    def _on_ok(self) -> None:
        renumber_lane_rows(self._rows)
        now = [t.id for t in self._rows]
        started = set(self._start_ids)
        added = [i for i in now if i not in started]
        deleted = [i for i in self._start_ids if i not in set(now)]
        renamed = {t.id: t.name for t in self._rows
                   if t.id in started
                   and t.name != self._start_names.get(t.id)}
        flagged = any(
            t.id in started
            and self._start_flags.get(t.id) != (bool(t.locked),
                                                bool(t.visible))
            for t in self._rows)
        kept_order = [i for i in now if i in started]
        reordered = kept_order != [i for i in self._start_order
                                   if i in set(now)]
        self.result = ManageLanesResult(
            lanes=[t.to_dict() for t in self._rows],
            added=added,
            renamed=renamed,
            deleted=deleted,
            deleted_intervals=sum(int(self._counts.get(i, 0))
                                  for i in deleted),
            imports=[dict(d) for d in self._imports
                     if d["id"] in set(now)],
            reordered=reordered,
            flagged=flagged,
        )
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.destroy()
