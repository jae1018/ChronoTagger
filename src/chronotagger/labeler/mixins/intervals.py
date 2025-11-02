"""
Intervals manipulation & undo/redo mixin.
"""

# ---- intervals.py (TOP OF FILE) ----
from __future__ import annotations
from typing import List, Tuple
import pandas as pd
from tkinter import messagebox

from chronotagger.core.models import Interval
from chronotagger.core.commands import (
    Command,
    AddIntervalCommand,
    DeleteIntervalCommand,
    RelabelIntervalCommand,
)



class IntervalsMixin:
    # ---- user actions ----
    def _add_interval(self) -> None:
        """
        Add one or more intervals:
          • If self.current_spans is non-empty -> add each span (multi-add).
          • Else if self.current_selection is set -> add that single span.
          • Else warn the user.
        """
        # Multi-span from box-select: prefer half-open commit spans if present
        commit_spans = getattr(self, "_commit_spans", []) or []
        preview_spans = getattr(self, "current_spans", []) or []
        
        if commit_spans or preview_spans:
            spans_to_add = commit_spans if commit_spans else self._normalize_preview_spans_to_half_open(preview_spans)
            label = self.current_class_var.get()  # type: ignore[union-attr]
            count = 0
            for s, e in spans_to_add:
                if e <= s:
                    continue
                self._execute_command(AddIntervalCommand(self, Interval(s, e, label)))
                count += 1
        
            # Clear selection state
            if commit_spans:
                self._commit_spans.clear()
            self.current_spans.clear()
            self.current_selection = None
        
            if count > 0:
                self.status_var.set(f"Added {count} {label} interval(s)")  # type: ignore[union-attr]
                self._update_plot()
                self._maybe_autosave()
            else:
                messagebox.showwarning("No Selection", "Box contained no valid points/spans.")
            return

        # Single-span (two-click or drag full-height)
        if not self.current_selection:
            messagebox.showwarning("No Selection", "Select a time range first (drag or click×2).")
            return

        s, e = self.current_selection
        label = self.current_class_var.get()  # type: ignore[union-attr]
        self._execute_command(AddIntervalCommand(self, Interval(s, e, label)))
        self.current_selection = None
        self.status_var.set(f"Added {label} interval")  # type: ignore[union-attr]
        self._update_plot()
        self._maybe_autosave()
        
    def _normalize_preview_spans_to_half_open(
        self, spans: List[Tuple[pd.Timestamp, pd.Timestamp]]
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Convert preview spans that end AT the last included sample into half-open:
          [s, e]  -> [s, next(e))   (or e+1ns if e is the last sample)
        """
        out: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        idx = self.df.index
    
        for s, e in spans:
            try:
                loc = idx.get_loc(e)
                # handle duplicate timestamps (slice) or scalar int
                if isinstance(loc, slice):
                    j = loc.stop - 1
                else:
                    j = int(loc)
                if j + 1 < len(idx):
                    e2 = idx[j + 1]
                else:
                    e2 = e + pd.Timedelta(nanoseconds=1)
                out.append((s, e2))
            except KeyError:
                # e not exactly on a sample -> keep as-is (already half-open-ish)
                out.append((s, e))
            except Exception:
                out.append((s, e))
        return out

    def _relabel_interval(self) -> None:
        if not self.selected_interval:
            messagebox.showwarning("No Selection", "Select an interval (strip or list) first.")
            return
        new_label = self.current_class_var.get()  # type: ignore[union-attr]
        cmd = RelabelIntervalCommand(self, self.selected_interval, new_label)
        self._execute_command(cmd)
        self.status_var.set(f"Relabeled → {new_label}")  # type: ignore[union-attr]
        self._update_plot()
        self._maybe_autosave()

    def _delete_interval(self) -> None:
        if not self.selected_interval:
            messagebox.showwarning("No Selection", "Select an interval to delete.")
            return
        cmd = DeleteIntervalCommand(self, self.selected_interval)
        self._execute_command(cmd)
        self.selected_interval = None
        self.status_var.set("Deleted interval")  # type: ignore[union-attr]
        self._update_plot()
        self._maybe_autosave()

    def _assign_remainder(self) -> None:
        """Label unlabeled gaps in the current window as UNKNOWN."""
        if "UNKNOWN" not in self.classes:
            messagebox.showwarning("No UNKNOWN Class", "UNKNOWN class is not defined.")
            return

        covered: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for iv in self.intervals:
            if iv.end <= self.t0 or iv.start >= self.t1:
                continue
            covered.append((max(iv.start, self.t0), min(iv.end, self.t1)))

        covered.sort()
        # Merge overlaps
        merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for s, e in covered:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # Find gaps
        gaps: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        cur = self.t0
        for s, e in merged:
            if cur < s:
                gaps.append((cur, s))
            cur = max(cur, e)
        if cur < self.t1:
            gaps.append((cur, self.t1))

        # Add UNKNOWN intervals
        for s, e in gaps:
            self._execute_command(AddIntervalCommand(self, Interval(s, e, "UNKNOWN")))

        self.status_var.set(f"Assigned {len(gaps)} UNKNOWN intervals")  # type: ignore[union-attr]
        self._update_plot()
        self._maybe_autosave()

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
        self.status_var.set("Redo")  # type: ignore[union-attr]
        self._update_plot()
        self._maybe_autosave()

    def _remove_overlapping_intervals(
        self, new_interval: Interval
    ) -> Tuple[List[Interval], List[Interval]]:
        """
        Remove/trim intervals that overlap `new_interval`.

        Returns
        -------
        removed : list[Interval]
            Original intervals removed due to overlap.
        trims : list[Interval]
            New trimmed intervals added back (non-overlapping parts).
        """
        removed: List[Interval] = []
        trims: List[Interval] = []

        for iv in self.intervals[:]:
            if not iv.overlaps(new_interval):
                continue

            self.intervals.remove(iv)
            removed.append(iv)

            if iv.start < new_interval.start:
                trims.append(Interval(iv.start, new_interval.start, iv.label, iv.notes))

            if iv.end > new_interval.end:
                trims.append(Interval(new_interval.end, iv.end, iv.label, iv.notes))

        self.intervals.extend(trims)
        return removed, trims

    def _sort_and_merge_intervals(self) -> None:
        """Sort by start and merge adjacent intervals with the same label."""
        if not self.intervals:
            return
        self.intervals.sort(key=lambda x: x.start)
        merged = [self.intervals[0]]
        for iv in self.intervals[1:]:
            last = merged[-1]
            if iv.start == last.end and iv.label == last.label:
                last.end = iv.end
            else:
                merged.append(iv)
        self.intervals = merged

    def _snap_to_samples(
        self, t_start: pd.Timestamp, t_end: pd.Timestamp
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Snap timestamps to nearest samples within the current window."""
        sub = self.df.loc[self.t0:self.t1]
        if len(sub.index) == 0:
            return t_start, t_end
        idx_start = sub.index[sub.index.get_indexer([t_start], method="nearest")[0]]
        idx_end = sub.index[sub.index.get_indexer([t_end], method="nearest")[0]]
        return idx_start, idx_end
    
    def _clear_all_intervals(self) -> None:
        """Delete every interval, reset selection/undo stacks, and refresh UI."""
        if not self.intervals:
            return
        from tkinter import messagebox  # local import to keep mixin self-contained
        if messagebox.askyesno("Clear All", f"Delete all {len(self.intervals)} intervals?"):
            self.intervals.clear()
            self.selected_interval = None
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.modified = True
            # Refresh UI
            self._update_plot()
            self._update_intervals_list()
            if self.status_var is not None:
                self.status_var.set("All intervals cleared")
