# src/chronotagger/labeler/mixins/labels.py
from __future__ import annotations

from typing import Dict
import tkinter as tk
from tkinter import messagebox

from ..dialogs.label_manager import LabelManagerDialog, LabelManagerResult


class LabelsMixin:
    """Label schema management (add/rename/delete/reorder/change color)."""

    def _open_label_manager(self) -> None:
        # Compute usage counts for nicer UX.
        # Pack M2: counted on the ACTIVE lane only. This dialog edits ONE
        # lane's vocabulary -- self.classes is a property over the active
        # row -- so a count that included another lane's intervals would
        # tell the user a class is in use when the lane he is editing does
        # not use it, and the reassign step would then move intervals he
        # cannot see.
        from chronotagger.core.tracks import active_id_of, intervals_on
        _active_track = active_id_of(self)
        _mine = intervals_on(self.intervals, _active_track)
        counts: Dict[str, int] = {name: 0 for name in self.classes}
        for iv in _mine:
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
        # Pack M2 R1, the worst of the three cross-lane destruction paths
        # M1 shipped. MEASURED on the real C05 session with three lanes:
        # one OK in this dialog rewrote ALL 256 intervals on the LOCKED
        # `agent` lane to UNKNOWN -- a label that lane does not declare --
        # while the agent Track row still said ['0','1','2','3']. After
        # that, 256 of 256 intervals carried an off-vocabulary label and
        # BOTH programmatic exporters refused the session, while
        # _save_autosave wrote the wreckage happily. CHRONOTAGGER_STRICT=1
        # did not catch it either: the membership clause checks the TRACK,
        # not the LABEL.
        #
        # TWO fixes, and they are different things. The ACTIVE-LANE SCOPE
        # below is the correctness fix: this dialog edits one lane's
        # vocabulary, so it may only touch that lane's intervals. The LOCK
        # REFUSAL is the policy fix: a locked lane refuses the edit
        # outright, on the status bar, with no modal.
        from chronotagger.core.lanes import refuse_if_locked
        if refuse_if_locked(self, what="edit the label schema"):
            return

        from chronotagger.core.tracks import active_id_of, intervals_on
        active_track = active_id_of(self)
        mine = intervals_on(self.intervals, active_track)

        # Renames, reassignments, and class-set changes mutate interval
        # labels in place, outside the gesture/undo system. Undoing past
        # such an edit would restore intervals carrying labels the schema
        # no longer contains (which the ML export silently maps to -1),
        # so those edits invalidate the undo history. A pure recolor or
        # reorder changes no interval and keeps the history (fold V2-M2).
        if res.rename_map or res.reassign_map or set(res.classes) != set(self.classes):
            self.undo_stack.clear()
            self.redo_stack.clear()

        # Rename pass
        if res.rename_map:
            for iv in mine:
                if iv.label in res.rename_map:
                    iv.label = res.rename_map[iv.label]

        # Reassign pass (deleted labels)
        if res.reassign_map:
            for iv in mine:
                if iv.label in res.reassign_map:
                    iv.label = res.reassign_map[iv.label]

        # Final cleanup: any labels not present -> UNKNOWN (if exists) or first
        fallback = "UNKNOWN" if "UNKNOWN" in res.classes else (res.classes[0] if res.classes else None)
        if fallback:
            for iv in mine:
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

        # Schema edits are real modifications: mark, sync, and autosave.
        self.modified = True
        self.sync_manager.sync_intervals_changed()
        self._save_autosave()

        # Redraw everything (sidebar tags, strip colors, etc.)
        self._update_plot()
        # Optional: toast
        if self.status_var is not None:
            self.status_var.set("Labels updated")
