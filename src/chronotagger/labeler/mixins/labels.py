# src/chronotagger/labeler/mixins/labels.py
from __future__ import annotations

from typing import Dict
import tkinter as tk
from tkinter import messagebox

from ..dialogs.label_manager import LabelManagerDialog, LabelManagerResult


class LabelsMixin:
    """Label schema management (add/rename/delete/reorder/change color)."""

    def _open_label_manager(self) -> None:
        # Compute usage counts for nicer UX
        counts: Dict[str, int] = {name: 0 for name in self.classes}
        for iv in self.intervals:
            if iv.label in counts:
                counts[iv.label] += 1

        dlg = LabelManagerDialog(
            parent=self.root,               # type: ignore[arg-type]
            classes=self.classes,
            class_colors=self.class_colors,
            usage_counts=counts,
            reserved={"UNKNOWN"},
        )
        self.root.wait_window(dlg)          # type: ignore[union-attr]

        if dlg.result is None:
            return

        self._apply_label_manager_result(dlg.result)

    # ---- apply result (single place to keep logic tidy) ----
    def _apply_label_manager_result(self, res: LabelManagerResult) -> None:
        # Rename pass
        if res.rename_map:
            for iv in self.intervals:
                if iv.label in res.rename_map:
                    iv.label = res.rename_map[iv.label]

        # Reassign pass (deleted labels)
        if res.reassign_map:
            for iv in self.intervals:
                if iv.label in res.reassign_map:
                    iv.label = res.reassign_map[iv.label]

        # Final cleanup: any labels not present -> UNKNOWN (if exists) or first
        fallback = "UNKNOWN" if "UNKNOWN" in res.classes else (res.classes[0] if res.classes else None)
        if fallback:
            for iv in self.intervals:
                if iv.label not in res.classes:
                    iv.label = fallback

        # Update schema
        self.classes = list(res.classes)
        self.class_colors = dict(res.class_colors)

        # Update widgets safely
        if self.class_combo is not None and self.current_class_var is not None:
            self.class_combo["values"] = self.classes
            if self.current_class_var.get() not in self.classes:
                # Try to keep continuity: if fallback exists use it, else first
                new_sel = self.current_class_var.get()
                if new_sel not in self.classes:
                    new_sel = fallback or (self.classes[0] if self.classes else "")
                self.current_class_var.set(new_sel)

        # Redraw everything (sidebar tags, strip colors, etc.)
        self._update_plot()
        # Optional: toast
        if self.status_var is not None:
            self.status_var.set("Labels updated")
