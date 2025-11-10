"""
Command pattern methods for interval operations.

This module provides the command execution, undo, and redo functionality
for interval operations in the ChronoTagger labeler.
"""

from __future__ import annotations
from chronotagger.core.commands import Command


class IntervalCommandsMixin:
    """Mixin providing command pattern methods for undo/redo support."""

    # ---- core operations ----
    def _execute_command(self, cmd: Command) -> None:
        cmd.execute()
        self.undo_stack.append(cmd)
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.modified = True

    def _undo(self) -> None:
        if not self.undo_stack:
            self.status_var.set("Nothing to undo")  # type: ignore[union-attr]
            return
        cmd = self.undo_stack.pop()
        cmd.undo()
        self.redo_stack.append(cmd)

        # Sync intervals across all panes
        self.sync_manager.sync_intervals_changed()

        # Check if the currently selected interval still exists after undo
        if hasattr(self, 'selected_interval') and self.selected_interval is not None:
            if self.selected_interval not in self.intervals:
                self.selected_interval = None
                if hasattr(self, '_clear_selected_interval_highlights'):
                    self._clear_selected_interval_highlights()

        self.status_var.set("Undo")  # type: ignore[union-attr]
        self._update_plot()
        self._maybe_autosave()

    def _redo(self) -> None:
        if not self.redo_stack:
            self.status_var.set("Nothing to redo")  # type: ignore[union-attr]
            return
        cmd = self.redo_stack.pop()
        cmd.execute()
        self.undo_stack.append(cmd)

        # Sync intervals across all panes
        self.sync_manager.sync_intervals_changed()

        # Check if the currently selected interval still exists after redo
        if hasattr(self, 'selected_interval') and self.selected_interval is not None:
            if self.selected_interval not in self.intervals:
                self.selected_interval = None
                if hasattr(self, '_clear_selected_interval_highlights'):
                    self._clear_selected_interval_highlights()

        self.status_var.set("Redo")  # type: ignore[union-attr]
        self._update_plot()
        self._maybe_autosave()
