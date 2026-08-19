"""
Gesture-based undo/redo for interval operations.

Every user gesture is wrapped in self._gesture(...), which captures
value-snapshots of the interval list before and after the gesture and
pushes ONE GestureCommand onto the undo stack. _execute_command() runs
bare operations; when called outside an explicit gesture it wraps
itself in one, so single-command call sites need no changes.

Strict mode: with CHRONOTAGGER_STRICT=1 in the environment, a
no-overlap invariant is checked on every list-changing gesture, undo,
and redo -- BEFORE the result is committed to the stacks and before
anything is autosaved. A violation rolls the interval list back and
raises IntervalInvariantError. tests/conftest.py turns strict mode on
for the whole suite.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from chronotagger.core.commands import (
    Command,
    GestureCommand,
    IntervalInvariantError,
    copy_intervals,
)


class IntervalCommandsMixin:
    """Mixin providing gesture-based undo/redo support."""

    # ---- gesture wrapper ----
    @contextmanager
    def _gesture(self, name: str = ""):
        """
        Group every interval mutation inside the block into one undo
        entry. Nested calls are absorbed by the outermost gesture. A
        gesture that changes nothing pushes no entry and leaves the
        redo stack alone. Gestures are transactional: if the body
        raises, or the strict-mode invariant rejects the result, the
        interval list is rolled back to its pre-gesture state and the
        exception re-raised -- nothing is committed.
        """
        depth = getattr(self, "_gesture_depth", 0)
        if depth > 0:
            self._gesture_depth = depth + 1
            try:
                yield
            finally:
                self._gesture_depth -= 1
            return

        self._gesture_depth = 1
        before = None
        try:
            before = copy_intervals(self.intervals)
            yield
            after = copy_intervals(self.intervals)
            if after != before:
                # Validate BEFORE committing anything: a violation must
                # be rolled back, never recorded (fold V2-M3/V3-M1).
                self._check_interval_invariants()
        except BaseException:
            # A failing gesture (body raise OR invariant violation) must
            # not leave half-applied mutations: roll back, then re-raise.
            if before is not None:
                self.intervals[:] = copy_intervals(before)
            raise
        finally:
            self._gesture_depth = 0
            # Keep the selection honest: a merge inside the gesture may
            # have consumed the selected object (fold V3-M2).
            self._repoint_selected_interval()
        if after == before:
            return
        self.undo_stack.append(GestureCommand(self, name, before, after))
        if len(self.undo_stack) > self.max_undo:
            del self.undo_stack[0 : len(self.undo_stack) - self.max_undo]
        self.redo_stack.clear()
        self.modified = True

    # ---- core operations ----
    def _execute_command(self, cmd: Command) -> None:
        if getattr(self, "_gesture_depth", 0) > 0:
            cmd.execute()
            return
        with self._gesture(type(cmd).__name__):
            cmd.execute()

    # ---- invariants ----
    def _check_interval_invariants(self) -> None:
        """
        Strict-mode invariant: no two intervals may overlap. Half-open
        semantics -- exact adjacency is legal and must not trip this.
        """
        if os.environ.get("CHRONOTAGGER_STRICT") != "1":
            return
        ivs = sorted(self.intervals, key=lambda iv: (iv.start, iv.end))
        for a, b in zip(ivs, ivs[1:]):
            if a.end > b.start:
                raise IntervalInvariantError(
                    f"overlapping intervals: "
                    f"[{a.start}, {a.end}) {a.label!r} / "
                    f"[{b.start}, {b.end}) {b.label!r}"
                )

    # ---- selection upkeep ----
    def _repoint_selected_interval(self) -> None:
        """
        After a snapshot restore the list holds fresh copies. Re-point
        selected_interval at the value-equal object now in the list, or
        clear the selection if its value is gone.
        """
        sel = getattr(self, "selected_interval", None)
        if sel is None:
            return
        for iv in self.intervals:
            if iv == sel:
                self.selected_interval = iv
                return
        self.selected_interval = None
        if hasattr(self, '_clear_selected_interval_highlights'):
            self._clear_selected_interval_highlights()

    def _undo(self) -> None:
        if not self.undo_stack:
            self.status_var.set("Nothing to undo")  # type: ignore[union-attr]
            return
        cmd = self.undo_stack.pop()
        cmd.undo()
        try:
            self._check_interval_invariants()
        except BaseException:
            # Do not commit a bad restore: put things back and re-raise.
            cmd.execute()
            self.undo_stack.append(cmd)
            raise
        self.redo_stack.append(cmd)
        self.modified = True

        # Sync intervals across all panes
        self.sync_manager.sync_intervals_changed()

        self._repoint_selected_interval()

        self.status_var.set("Undo")  # type: ignore[union-attr]
        self._update_plot()
        self._save_autosave()

    def _redo(self) -> None:
        if not self.redo_stack:
            self.status_var.set("Nothing to redo")  # type: ignore[union-attr]
            return
        cmd = self.redo_stack.pop()
        cmd.execute()
        try:
            self._check_interval_invariants()
        except BaseException:
            # Do not commit a bad restore: put things back and re-raise.
            cmd.undo()
            self.redo_stack.append(cmd)
            raise
        self.undo_stack.append(cmd)
        self.modified = True

        # Sync intervals across all panes
        self.sync_manager.sync_intervals_changed()

        self._repoint_selected_interval()

        self.status_var.set("Redo")  # type: ignore[union-attr]
        self._update_plot()
        self._save_autosave()
