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

from contextlib import contextmanager

from chronotagger.core.commands import (
    Command,
    GestureCommand,
    check_interval_invariants,
    copy_intervals,
)
from chronotagger.core.tracks import copy_tracks, snapshot_tracks


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
        # Pack M1: hoisted beside `before` on purpose. The rollback
        # branch below reads it, and a prototype that assigned it only
        # after the interval copy raised UnboundLocalError on every host
        # that has no track table.
        tracks_before = None
        try:
            before = copy_intervals(self.intervals)
            tracks_before = snapshot_tracks(self)
            yield
            after = copy_intervals(self.intervals)
            tracks_after = snapshot_tracks(self)
            # Pack M1: the no-op test compares the PAIR. Creating a
            # track adds no interval, so under an intervals-only test it
            # would push NO undo entry and be silently non-undoable.
            if (after, tracks_after) != (before, tracks_before):
                # Validate BEFORE committing anything: a violation must
                # be rolled back, never recorded (fold V2-M3/V3-M1).
                self._check_interval_invariants()
        except BaseException:
            # A failing gesture (body raise OR invariant violation) must
            # not leave half-applied mutations: roll back, then re-raise.
            if before is not None:
                self.intervals[:] = copy_intervals(before)
            if tracks_before is not None:
                self.tracks[:] = copy_tracks(tracks_before)
            raise
        finally:
            self._gesture_depth = 0
            # Keep the selection honest: a merge inside the gesture may
            # have consumed the selected object (fold V3-M2).
            self._repoint_selected_interval()
        if (after, tracks_after) == (before, tracks_before):
            return
        self.undo_stack.append(GestureCommand(self, name, before, after,
                                              tracks_before, tracks_after))
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
        """Strict-mode invariants on the LIVE interval list and table.

        Calls the module-level function rather than the sibling method
        below ON PURPOSE: the GUI-free hosts in tests/ bind a NAMED LIST
        of mixin methods, and a method that calls a sibling would force
        every one of those lists to grow an entry. The same reason
        write_label_map_sidecar is a free function.
        """
        check_interval_invariants(self.intervals,
                                  getattr(self, "tracks", None))

    def _check_interval_invariants_on(self, intervals, table=None) -> None:
        """Strict-mode invariants on the set being INSTALLED.

        Pack M1 moved the body out to
        core/commands.check_interval_invariants, so that a load path can
        validate an interval set it has not published yet and so that a
        GUI-free host binding a named list of mixin methods needs no new
        entry for it. The two clauses -- MEMBERSHIP and per-track
        non-overlap -- and the reason the `table` argument must be the
        table being installed are documented there.
        """
        check_interval_invariants(intervals, table)

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

        # Pack M2.6, two things on one line of screen. NAME THE GESTURE:
        # "Undo" alone never said WHAT came back, and the gesture's label
        # has existed since Pack 1 -- it was simply never printed.
        # RECONCILE: this undo may have removed the very lane
        # `_active_track_id` names (an ingest creates its lane inside one
        # gesture), and when it did, the reconcile's own sentence is the
        # one that stays on the bar, because a lane disappearing under
        # the user's hands outranks the name of what he undid.
        # getattr, because the GUI-free hosts in tests/ bind a NAMED LIST
        # of mixin methods and must not be forced to grow an entry.
        _name = getattr(cmd, "name", "") or ""
        # A command pushed WITHOUT a gesture wrapper is named
        # `type(cmd).__name__` by _execute_command above, so Delete,
        # Re-label and a drag-resize would put "DeleteIntervalCommand" on
        # the status bar. An internal class name is not a sentence: drop
        # it and say plain "Undo", which is what those three said before.
        if _name.endswith("Command"):
            _name = ""
        self.status_var.set(  # type: ignore[union-attr]
            ("Undo: %s" % _name) if _name else "Undo")
        _rec = getattr(self, "_reconcile_active_track", None)
        if callable(_rec):
            _rec()
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

        # Pack M2.6, the same two things as _undo above: name the gesture,
        # then reconcile a stale active lane (a redo can put a lane BACK,
        # and it can also redo the removal of one).
        _name = getattr(cmd, "name", "") or ""
        # See _undo: a bare command's "name" is its class name.
        if _name.endswith("Command"):
            _name = ""
        self.status_var.set(  # type: ignore[union-attr]
            ("Redo: %s" % _name) if _name else "Redo")
        _rec = getattr(self, "_reconcile_active_track", None)
        if callable(_rec):
            _rec()
        self._update_plot()
        self._save_autosave()
