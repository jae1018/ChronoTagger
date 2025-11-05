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
    ResizeIntervalCommand,
)



class IntervalsMixin:
    
    # ---- user actions ----
    def _add_interval(self) -> None:
        """
        Add one or more intervals:
          • If self._commit_spans is non-empty -> add each span exactly as provided
            (already policy-processed by dialogs).
          • Else if self.current_spans is set -> normalize to half-open and apply policy here.
          • Else warn the user.
        """
        commit_spans = getattr(self, "_commit_spans", []) or []
        preview_spans = getattr(self, "current_spans", []) or []
    
        label = self.current_class_var.get()  # type: ignore[union-attr]
        policy = getattr(self, "_overlap_policy", "skip")
        count = 0
    
        # === 1) Rule-driven path: commit spans are authoritative ===
        if commit_spans:
            # If you later implement "replace", you may carve here. For now, we just add.
            for s, e in commit_spans:
                if e <= s:
                    continue
                self._execute_command(AddIntervalCommand(self, Interval(s, e, label)))
                count += 1
    
            # Clear selection state
            self._commit_spans.clear()
            self.current_spans.clear()
            self.current_selection = None
        
        # Clear point highlights
        if hasattr(self, '_clear_selected_point_highlights'):
            self._clear_selected_point_highlights()
    
            if count > 0:
                self.status_var.set(f"Added {count} {label} interval(s)")  # type: ignore[union-attr]
                self._update_plot()
                self._maybe_autosave()
            else:
                # More informative message if policy produced emptiness
                from tkinter import messagebox
                messagebox.showwarning(
                    "No Selection",
                    "No valid spans after applying the overlap policy."
                )
            return
    
        # === 2) Box-select / two-click path ===
        if preview_spans:
            spans_to_add = self._normalize_preview_spans_to_half_open(preview_spans)
    
            final_spans: List[Tuple[pd.Timestamp, pd.Timestamp]]
            if policy == "skip":
                final_spans = []
                for s, e in spans_to_add:
                    final_spans.extend(self._subtract_overlaps_from_span(s, e))
            else:
                # Future: handle "replace" by carving, for now just add as-is
                final_spans = spans_to_add
    
            for s, e in final_spans:
                if e <= s:
                    continue
                self._execute_command(AddIntervalCommand(self, Interval(s, e, label)))
                count += 1
    
            self.current_spans.clear()
            self.current_selection = None
            
            # Clear point highlights
            if hasattr(self, '_clear_selected_point_highlights'):
                self._clear_selected_point_highlights()
    
            if count > 0:
                self.status_var.set(f"Added {count} {label} interval(s)")  # type: ignore[union-attr]
                self._update_plot()
                self._maybe_autosave()
            else:
                from tkinter import messagebox
                messagebox.showwarning("No Selection", "Box contained no valid points/spans.")
            return
    
        # === 3) Single-span path ===
        if not self.current_selection:
            from tkinter import messagebox
            messagebox.showwarning("No Selection", "Select a time range first (drag or click×2).")
            return
    
        s, e = self.current_selection
        self._execute_command(AddIntervalCommand(self, Interval(s, e, label)))
        self.current_selection = None
        
        # Clear point highlights
        if hasattr(self, '_clear_selected_point_highlights'):
            self._clear_selected_point_highlights()
        
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
        
        # Find gaps and assign to UNKNOWN
        gaps = self._find_gaps_in_current_range()
        
        if not gaps:
            self.status_var.set("No unassigned time in current window")  # type: ignore[union-attr]
            return
        
        # Create UNKNOWN intervals for each gap
        for s, e in gaps:
            self._execute_command(AddIntervalCommand(self, Interval(s, e, "UNKNOWN")))

        self.status_var.set(f"Assigned {len(gaps)} UNKNOWN intervals")  # type: ignore[union-attr]
        self._update_plot()
        self._maybe_autosave()
    
    def _find_gaps_in_current_range(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Find all unassigned time gaps in the current window [t0, t1].
        
        Returns a list of (start, end) tuples representing time periods
        that have no interval coverage in the current time range.
        
        Returns:
            List of (pd.Timestamp, pd.Timestamp) tuples for each gap
        """
        # Collect covered intervals in current window
        covered: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for iv in self.intervals:
            if iv.end <= self.t0 or iv.start >= self.t1:
                continue
            covered.append((max(iv.start, self.t0), min(iv.end, self.t1)))

        covered.sort()
        
        # Merge overlapping intervals
        merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for s, e in covered:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # Find gaps between merged intervals
        gaps: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        cur = self.t0
        for s, e in merged:
            if cur < s:
                gaps.append((cur, s))
            cur = max(cur, e)
        if cur < self.t1:
            gaps.append((cur, self.t1))
        
        return gaps
    
    def _open_label_unassigned_dialog(self) -> None:
        """
        Open dialog to select label for unassigned points in current range.
        
        Presents a scrollable listbox of existing labels, shows the current
        time range, and calculates how many intervals will be created.
        User selects a label and confirms to assign all gaps to that label.
        """
        import tkinter as tk
        from tkinter import ttk
        
        # Validate: check if any labels exist
        if not self.classes:
            messagebox.showwarning(
                "No Labels",
                "Please create labels first using 'Manage Labels'."
            )
            return
        
        # Find gaps to calculate count
        gaps = self._find_gaps_in_current_range()
        
        # If no gaps, inform user and return
        if not gaps:
            messagebox.showinfo(
                "No Gaps",
                "All points in the current time range are already labeled."
            )
            return
        
        # Create modal dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("Label Unassigned Points")
        dialog.geometry("400x350")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        
        # Main content frame
        content = ttk.Frame(dialog, padding=15)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Instruction label
        ttk.Label(
            content,
            text="Select label for unassigned points:",
            font=("TkDefaultFont", 10, "bold")
        ).pack(pady=(0, 10))
        
        # Listbox with scrollbar frame
        listbox_frame = ttk.Frame(content)
        listbox_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Listbox
        listbox = tk.Listbox(
            listbox_frame,
            yscrollcommand=scrollbar.set,
            font=("TkDefaultFont", 10),
            height=8,
            selectmode=tk.SINGLE,
            activestyle="dotbox"
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        # Populate listbox with labels
        for label in self.classes:
            listbox.insert(tk.END, label)
        
        # Pre-select first label and ensure it's visible
        listbox.selection_set(0)
        listbox.see(0)
        listbox.focus_set()
        
        # Separator
        ttk.Separator(content, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 10))
        
        # Info frame for time range and gap count
        info_frame = ttk.Frame(content)
        info_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Time range display
        time_range_text = f"Time range: {self.t0.strftime('%Y-%m-%d %H:%M:%S')} → {self.t1.strftime('%Y-%m-%d %H:%M:%S')}"
        ttk.Label(
            info_frame,
            text=time_range_text,
            font=("TkDefaultFont", 9),
            foreground="#666"
        ).pack(anchor=tk.W)
        
        # Gap count display
        gap_count_text = f"Will create {len(gaps)} interval(s)"
        ttk.Label(
            info_frame,
            text=gap_count_text,
            font=("TkDefaultFont", 9, "bold"),
            foreground="#0066cc"
        ).pack(anchor=tk.W, pady=(5, 0))
        
        # Button frame
        button_frame = ttk.Frame(content)
        button_frame.pack()
        
        def on_ok():
            """Handle OK button: assign gaps to selected label."""
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a label.")
                return
            
            selected_label = listbox.get(selection[0])
            dialog.destroy()
            
            # Assign all gaps to the selected label
            self._assign_gaps_to_label(gaps, selected_label)
        
        def on_cancel():
            """Handle Cancel button: close dialog without action."""
            dialog.destroy()
        
        # OK button
        ok_button = ttk.Button(
            button_frame,
            text="OK",
            command=on_ok,
            width=10
        )
        ok_button.pack(side=tk.LEFT, padx=5)
        
        # Cancel button
        cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=on_cancel,
            width=10
        )
        cancel_button.pack(side=tk.LEFT, padx=5)
        
        # Keyboard shortcuts
        dialog.bind("<Return>", lambda e: on_ok())
        dialog.bind("<Escape>", lambda e: on_cancel())
        
        # Double-click to select and confirm
        listbox.bind("<Double-Button-1>", lambda e: on_ok())
        
        # Center dialog on parent window
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
    
    def _assign_gaps_to_label(self, gaps: List[Tuple[pd.Timestamp, pd.Timestamp]], label: str) -> None:
        """
        Assign all gaps to the specified label.
        
        Args:
            gaps: List of (start, end) tuples representing unassigned time periods
            label: The label to assign to all gaps
        """
        if not gaps:
            return
        
        # Create intervals for each gap with the specified label
        for s, e in gaps:
            self._execute_command(AddIntervalCommand(self, Interval(s, e, label)))
        
        # Update UI
        self.status_var.set(f"Assigned {len(gaps)} interval(s) to {label}")  # type: ignore[union-attr]
        self._update_plot()
        self._maybe_autosave()
        
    def _carve_existing_for_new_span(self, s: pd.Timestamp, e: pd.Timestamp) -> None:
        """
        Modify existing intervals so that [s, e) becomes free space:
          - Fully covered intervals are deleted.
          - Left/right edge overlaps are resized.
          - Middle overlaps (new span cuts an interval in two) are split by
            deleting the original and adding two trimmed intervals.
        All changes are executed via commands for proper undo/redo.
        """
        if e <= s:
            return

        # Iterate over a stable snapshot; commands mutate self.intervals
        existing = list(self.intervals)
        for iv in existing:
            # no overlap
            if iv.end <= s or iv.start >= e:
                continue

            # Case A: fully covered by [s,e) -> delete
            if iv.start >= s and iv.end <= e:
                self._execute_command(DeleteIntervalCommand(self, iv))
                continue

            # Case B: overlap on the right edge (keep left)
            if iv.start < s <= iv.end <= e:
                self._execute_command(ResizeIntervalCommand(self, iv, iv.start, s))
                continue

            # Case C: overlap on the left edge (keep right)
            if s <= iv.start < e < iv.end:
                self._execute_command(ResizeIntervalCommand(self, iv, e, iv.end))
                continue

            # Case D: [s,e) strictly inside iv => split into left + right
            if iv.start < s and iv.end > e:
                # delete original
                self._execute_command(DeleteIntervalCommand(self, iv))
                # add left + right fragments with same label/notes
                left_start, left_end = iv.start, s
                right_start, right_end = e, iv.end
                from chronotagger.core.models import Interval
                self._execute_command(AddIntervalCommand(self, Interval(left_start, left_end, iv.label, iv.notes)))
                self._execute_command(AddIntervalCommand(self, Interval(right_start, right_end, iv.label, iv.notes)))
                continue
            
    def _commit_to_preview_spans(
        self, spans: List[Tuple[pd.Timestamp, pd.Timestamp]]
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Convert half-open [s, e) commit spans into preview spans that end AT
        the last included sample (s, e_last_included] so they render correctly
        in the strip / overlays.
        """
        out: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        idx = self.df.index
        for s, e in spans:
            j = idx.searchsorted(e, side="left") - 1
            if j >= 0:
                out.append((s, pd.Timestamp(idx[j])))
        return out
    
    
    def _apply_overlap_policy_to_spans(
        self, spans: List[Tuple[pd.Timestamp, pd.Timestamp]], policy: str
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Return spans after applying the requested overlap policy against
        current self.intervals.
    
        policy:
          - "skip"     -> remove any portions that overlap existing intervals
          - "replace"  -> (preview) leave spans as-is; carving happens when we add
          - anything else -> passthrough
        """
        policy = (policy or "").lower()
        if policy == "skip":
            pieces: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
            for s, e in spans:
                pieces.extend(self._subtract_overlaps_from_span(s, e))
            return pieces
        # For preview, "replace" shows what you'll add; carving is done at add-time.
        return spans

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
        
    def _subtract_overlaps_from_span(
        self,
        s: pd.Timestamp,
        e: pd.Timestamp,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Given a candidate half-open span [s, e), subtract any currently labeled
        intervals and return a list of non-overlapping subspans inside [s, e).
        """
        out: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        if e <= s:
            return out

        # Collect overlaps with existing intervals
        overlaps = [iv for iv in self.intervals if not (iv.end <= s or iv.start >= e)]
        overlaps.sort(key=lambda iv: iv.start)

        cur = s
        for iv in overlaps:
            if iv.start > cur:
                left_end = min(iv.start, e)
                if left_end > cur:
                    out.append((cur, left_end))
            if iv.end > cur:
                cur = max(cur, iv.end)
            if cur >= e:
                break

        if cur < e:
            out.append((cur, e))

        return out

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
