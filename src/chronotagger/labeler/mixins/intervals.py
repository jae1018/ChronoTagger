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
        current_selection = getattr(self, "current_selection", None)
    
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
                    subtracted = self._subtract_overlaps_from_span(s, e)
                    final_spans.extend(subtracted)
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
        if not current_selection:
            from tkinter import messagebox
            messagebox.showwarning("No Selection", "Select a time range first (drag or click×2).")
            return
    
        s, e = current_selection
        if e <= s:
            from tkinter import messagebox
            messagebox.showwarning("Invalid Selection", "End time must be after start time.")
            return
            
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
        # Clear interval highlights when interval is deleted
        if hasattr(self, '_clear_selected_interval_highlights'):
            self._clear_selected_interval_highlights()
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
        dialog.geometry("420x380")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        
        # Main content frame
        content = ttk.Frame(dialog, padding=20)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Instruction label
        ttk.Label(
            content,
            text="Select label for unassigned points:",
            font=("TkDefaultFont", 11, "bold")
        ).pack(pady=(0, 12))
        
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
            height=10,
            selectmode=tk.SINGLE,
            activestyle="dotbox",
            relief=tk.SOLID,
            borderwidth=1
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
        info_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Time range display
        ttk.Label(
            info_frame,
            text="Time range:",
            font=("TkDefaultFont", 9, "bold")
        ).pack(anchor=tk.W)
        
        time_range_text = f"{self.t0.strftime('%Y-%m-%d %H:%M:%S')} → {self.t1.strftime('%Y-%m-%d %H:%M:%S')}"
        ttk.Label(
            info_frame,
            text=time_range_text,
            font=("TkDefaultFont", 9),
            foreground="#555"
        ).pack(anchor=tk.W, padx=(10, 0), pady=(2, 8))
        
        # Gap count display
        gap_count_text = f"Will create {len(gaps)} interval(s)"
        ttk.Label(
            info_frame,
            text=gap_count_text,
            font=("TkDefaultFont", 10, "bold"),
            foreground="#0066cc"
        ).pack(anchor=tk.W)
        
        # Button frame
        button_frame = ttk.Frame(content)
        button_frame.pack(pady=(5, 0))
        
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
            width=15
        )
        ok_button.pack(side=tk.LEFT, padx=8)
        
        # Cancel button
        cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=on_cancel,
            width=15
        )
        cancel_button.pack(side=tk.LEFT, padx=8)
        
        # Keyboard shortcuts
        dialog.bind("<Return>", lambda e: on_ok())
        dialog.bind("<Escape>", lambda e: on_cancel())
        
        # Double-click to select and confirm
        listbox.bind("<Double-Button-1>", lambda e: on_ok())
        
        # Center dialog on parent window
        dialog.update_idletasks()
        
        # Calculate position relative to root window
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_width = self.root.winfo_width()
        root_height = self.root.winfo_height()
        
        dialog_width = dialog.winfo_reqwidth()
        dialog_height = dialog.winfo_reqheight()
        
        x = root_x + (root_width - dialog_width) // 2
        y = root_y + (root_height - dialog_height) // 2
        
        dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
        
        # Bring dialog to front
        dialog.lift()
        dialog.focus_set()
    
    def _show_clear_confirmation(
        self, t0: pd.Timestamp, t1: pd.Timestamp, mode_name: str
    ) -> None:
        """
        Show confirmation dialog with detailed breakdown of affected intervals.
        
        Displays warning message with:
        - Time range being cleared
        - Count of intervals that will be deleted
        - Count of intervals that will be truncated
        - Count of intervals that will be split
        - Reminder about undo support
        
        Args:
            t0: Start of clear range
            t1: End of clear range
            mode_name: Descriptive name of the mode (for display)
        """
        import tkinter as tk
        from tkinter import ttk
        
        # Analyze what will be affected
        analysis = self._analyze_intervals_in_range(t0, t1)
        
        # If nothing will be affected, inform user
        if analysis['total_affected'] == 0:
            messagebox.showinfo(
                "No Intervals to Clear",
                f"The selected range:\n{t0.strftime('%Y-%m-%d %H:%M:%S')} → {t1.strftime('%Y-%m-%d %H:%M:%S')}\n\ncontains no intervals."
            )
            return
        
        # Create confirmation dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Confirm Clear Intervals")
        dialog.geometry("400x280")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        
        # Main content frame
        content = ttk.Frame(dialog, padding=20)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Warning icon and title
        title_frame = ttk.Frame(content)
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(
            title_frame,
            text="⚠️",
            font=("TkDefaultFont", 24)
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(
            title_frame,
            text="Warning",
            font=("TkDefaultFont", 14, "bold")
        ).pack(side=tk.LEFT)
        
        # Separator
        ttk.Separator(content, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 15))
        
        # Mode label
        ttk.Label(
            content,
            text=f"Mode: {mode_name}",
            font=("TkDefaultFont", 9, "bold")
        ).pack(anchor=tk.W, pady=(0, 5))
        
        # Time range display
        ttk.Label(
            content,
            text="Time range:",
            font=("TkDefaultFont", 9, "bold")
        ).pack(anchor=tk.W)
        
        range_text = f"{t0.strftime('%Y-%m-%d %H:%M:%S')} → {t1.strftime('%Y-%m-%d %H:%M:%S')}"
        ttk.Label(
            content,
            text=range_text,
            font=("TkDefaultFont", 9)
        ).pack(anchor=tk.W, padx=(10, 0), pady=(2, 10))
        
        # Affected intervals breakdown
        ttk.Label(
            content,
            text=f"This will affect {analysis['total_affected']} interval(s):",
            font=("TkDefaultFont", 9, "bold")
        ).pack(anchor=tk.W, pady=(0, 5))
        
        # Breakdown details
        details_frame = ttk.Frame(content)
        details_frame.pack(fill=tk.X, padx=(10, 0))
        
        if analysis['to_delete'] > 0:
            ttk.Label(
                details_frame,
                text=f"• {analysis['to_delete']} will be deleted (fully inside range)",
                font=("TkDefaultFont", 9)
            ).pack(anchor=tk.W, pady=2)
        
        if analysis['to_truncate'] > 0:
            ttk.Label(
                details_frame,
                text=f"• {analysis['to_truncate']} will be truncated (partially overlap)",
                font=("TkDefaultFont", 9)
            ).pack(anchor=tk.W, pady=2)
        
        if analysis['to_split'] > 0:
            ttk.Label(
                details_frame,
                text=f"• {analysis['to_split']} will be split (spans entire range)",
                font=("TkDefaultFont", 9)
            ).pack(anchor=tk.W, pady=2)
        
        # Undo reminder
        ttk.Label(
            content,
            text="This action can be undone using the 'Undo' button in the top bar.",
            font=("TkDefaultFont", 9, "italic"),
            foreground="#0066cc"
        ).pack(anchor=tk.W, pady=(15, 0))
        
        # Confirmation question
        ttk.Label(
            content,
            text="Are you sure?",
            font=("TkDefaultFont", 10, "bold")
        ).pack(pady=(15, 15))
        
        # Button frame
        button_frame = ttk.Frame(content)
        button_frame.pack()
        
        def on_confirm():
            """Execute the clear operation."""
            dialog.destroy()
            
            # Perform the clear operation
            results = self._clear_intervals_in_range(t0, t1)
            
            # Build status message
            parts = []
            if results['deleted'] > 0:
                parts.append(f"{results['deleted']} deleted")
            if results['truncated'] > 0:
                parts.append(f"{results['truncated']} truncated")
            if results['split'] > 0:
                parts.append(f"{results['split']} split")
            
            status_msg = "Cleared intervals: " + ", ".join(parts)
            self.status_var.set(status_msg)  # type: ignore[union-attr]
            
            # Update UI
            self._update_plot()
            self._maybe_autosave()
        
        def on_cancel():
            """Close dialog without action."""
            dialog.destroy()
        
        # Yes button (styled as warning)
        yes_button = ttk.Button(
            button_frame,
            text="Yes, Clear",
            command=on_confirm,
            width=12
        )
        yes_button.pack(side=tk.LEFT, padx=5)
        
        # Cancel button
        cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=on_cancel,
            width=12
        )
        cancel_button.pack(side=tk.LEFT, padx=5)
        
        # Keyboard shortcuts
        dialog.bind("<Escape>", lambda e: on_cancel())
        # Note: Don't bind Enter to avoid accidental confirmation
        
        # Center dialog on parent
        dialog.update_idletasks()
        
        # Calculate position relative to root window
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_width = self.root.winfo_width()
        root_height = self.root.winfo_height()
        
        dialog_width = dialog.winfo_reqwidth()
        dialog_height = dialog.winfo_reqheight()
        
        x = root_x + (root_width - dialog_width) // 2
        y = root_y + (root_height - dialog_height) // 2
        
        dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
        
        # Bring dialog to front
        dialog.lift()
        dialog.focus_set()
    
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
    
    def _analyze_intervals_in_range(
        self, t0: pd.Timestamp, t1: pd.Timestamp
    ) -> dict:
        """
        Analyze how intervals will be affected by clearing range [t0, t1].
        
        Categorizes intervals by how they overlap with the clear range:
        - Fully inside: will be deleted
        - Left overlap: will be truncated (keep left part)
        - Right overlap: will be truncated (keep right part)
        - Spanning: will be split (keep left and right parts)
        - No overlap: will be kept
        
        Args:
            t0: Start of range to clear
            t1: End of range to clear
        
        Returns:
            Dictionary with:
            - 'to_delete': count of intervals fully inside range
            - 'to_truncate': count of intervals partially overlapping
            - 'to_split': count of intervals spanning entire range
            - 'total_affected': total count of affected intervals
        """
        to_delete = 0
        to_truncate = 0
        to_split = 0
        
        for iv in self.intervals:
            # No overlap - skip
            if iv.end <= t0 or iv.start >= t1:
                continue
            
            # Fully inside range - will be deleted
            if iv.start >= t0 and iv.end <= t1:
                to_delete += 1
                continue
            
            # Left overlap - will be truncated
            if iv.start < t0 and iv.end > t0 and iv.end <= t1:
                to_truncate += 1
                continue
            
            # Right overlap - will be truncated
            if iv.start >= t0 and iv.start < t1 and iv.end > t1:
                to_truncate += 1
                continue
            
            # Spans entire range - will be split
            if iv.start < t0 and iv.end > t1:
                to_split += 1
                continue
        
        total_affected = to_delete + to_truncate + to_split
        
        return {
            'to_delete': to_delete,
            'to_truncate': to_truncate,
            'to_split': to_split,
            'total_affected': total_affected
        }
    
    def _clear_intervals_in_range(
        self, t0: pd.Timestamp, t1: pd.Timestamp
    ) -> dict:
        """
        Clear or truncate intervals in the specified range [t0, t1].
        
        Handles five overlap cases:
        1. No overlap: Keep interval unchanged
        2. Fully inside: Delete interval
        3. Left overlap: Truncate to keep left part (before t0)
        4. Right overlap: Truncate to keep right part (after t1)
        5. Spanning: Split into left and right parts (before t0 and after t1)
        
        All operations use commands for proper undo/redo support.
        
        Args:
            t0: Start of range to clear
            t1: End of range to clear
        
        Returns:
            Dictionary with counts: {'deleted': N, 'truncated': M, 'split': K}
        """
        deleted = 0
        truncated = 0
        split = 0
        
        # Snapshot intervals (commands will modify the list)
        intervals_to_process = list(self.intervals)
        
        for iv in intervals_to_process:
            # Case 1: No overlap - skip
            if iv.end <= t0 or iv.start >= t1:
                continue
            
            # Case 2: Fully inside range - delete
            if iv.start >= t0 and iv.end <= t1:
                self._execute_command(DeleteIntervalCommand(self, iv))
                deleted += 1
                continue
            
            # Case 3: Left overlap - keep left part (truncate at t0)
            if iv.start < t0 and iv.end > t0 and iv.end <= t1:
                self._execute_command(ResizeIntervalCommand(self, iv, iv.start, t0))
                truncated += 1
                continue
            
            # Case 4: Right overlap - keep right part (truncate at t1)
            if iv.start >= t0 and iv.start < t1 and iv.end > t1:
                self._execute_command(ResizeIntervalCommand(self, iv, t1, iv.end))
                truncated += 1
                continue
            
            # Case 5: Spans entire range - split into left + right parts
            if iv.start < t0 and iv.end > t1:
                # Delete original interval
                self._execute_command(DeleteIntervalCommand(self, iv))
                
                # Add left part (before clear range)
                left_interval = Interval(iv.start, t0, iv.label, iv.notes)
                self._execute_command(AddIntervalCommand(self, left_interval))
                
                # Add right part (after clear range)
                right_interval = Interval(t1, iv.end, iv.label, iv.notes)
                self._execute_command(AddIntervalCommand(self, right_interval))
                
                split += 1
                continue
        
        return {
            'deleted': deleted,
            'truncated': truncated,
            'split': split
        }
    
    def _open_clear_intervals_dialog(self) -> None:
        """
        Open dialog to select range for clearing intervals.
        
        Presents three options:
        1. Current time range (default)
        2. Custom time range (user enters start/end)
        3. Entire dataset
        
        After range selection, shows confirmation dialog with details.
        """
        import tkinter as tk
        from tkinter import ttk
        
        # Check if any intervals exist
        if not self.intervals:
            messagebox.showinfo("No Intervals", "There are no intervals to clear.")
            return
        
        # Create modal dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Clear Intervals")
        dialog.geometry("380x250")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        
        # Main content frame
        content = ttk.Frame(dialog, padding=15)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Title label
        ttk.Label(
            content,
            text="Select range to clear intervals:",
            font=("TkDefaultFont", 10, "bold")
        ).pack(pady=(0, 15))
        
        # Radio button variable
        range_var = tk.StringVar(value="current")
        
        # Option 1: Current time range
        current_frame = ttk.Frame(content)
        current_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(
            current_frame,
            text="Current time range",
            variable=range_var,
            value="current"
        ).pack(anchor=tk.W)
        
        current_range_text = f"{self.t0.strftime('%Y-%m-%d %H:%M:%S')} → {self.t1.strftime('%Y-%m-%d %H:%M:%S')}"
        ttk.Label(
            current_frame,
            text=f"    {current_range_text}",
            font=("TkDefaultFont", 9),
            foreground="#666"
        ).pack(anchor=tk.W, padx=(20, 0))
        
        # Option 2: Custom time range
        custom_frame = ttk.Frame(content)
        custom_frame.pack(fill=tk.X, pady=10)
        
        ttk.Radiobutton(
            custom_frame,
            text="Custom time range",
            variable=range_var,
            value="custom"
        ).pack(anchor=tk.W)
        
        # Custom entry fields
        custom_entry_frame = ttk.Frame(custom_frame)
        custom_entry_frame.pack(fill=tk.X, padx=(20, 0), pady=(5, 0))
        
        ttk.Label(custom_entry_frame, text="From:").grid(row=0, column=0, sticky=tk.W, pady=2)
        from_entry = ttk.Entry(custom_entry_frame, width=30)
        from_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        from_entry.insert(0, str(self.t0))
        
        ttk.Label(custom_entry_frame, text="To:").grid(row=1, column=0, sticky=tk.W, pady=2)
        to_entry = ttk.Entry(custom_entry_frame, width=30)
        to_entry.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        to_entry.insert(0, str(self.t1))
        
        # Initially disable custom entries
        from_entry.config(state="disabled")
        to_entry.config(state="disabled")
        
        def on_range_change():
            """Enable/disable custom entries based on selection."""
            if range_var.get() == "custom":
                from_entry.config(state="normal")
                to_entry.config(state="normal")
            else:
                from_entry.config(state="disabled")
                to_entry.config(state="disabled")
        
        # Bind radio button changes
        range_var.trace('w', lambda *args: on_range_change())
        
        # Option 3: Entire dataset
        entire_frame = ttk.Frame(content)
        entire_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(
            entire_frame,
            text="Entire dataset",
            variable=range_var,
            value="entire"
        ).pack(anchor=tk.W)
        
        ttk.Label(
            entire_frame,
            text="    (clears all intervals in the dataset)",
            font=("TkDefaultFont", 9),
            foreground="#666"
        ).pack(anchor=tk.W, padx=(20, 0))
        
        # Button frame
        button_frame = ttk.Frame(content)
        button_frame.pack(pady=(20, 0))
        
        def on_next():
            """Validate selection and show confirmation dialog."""
            mode = range_var.get()
            
            if mode == "current":
                t0, t1 = self.t0, self.t1
                mode_name = "Current Time Range"
            
            elif mode == "custom":
                try:
                    t0 = pd.Timestamp(from_entry.get())
                    t1 = pd.Timestamp(to_entry.get())
                    
                    if t1 <= t0:
                        messagebox.showerror(
                            "Invalid Range",
                            "End time must be after start time."
                        )
                        return
                    
                    mode_name = "Custom Range"
                
                except Exception as e:
                    messagebox.showerror(
                        "Invalid Timestamps",
                        f"Please enter valid timestamps.\n\nError: {str(e)}"
                    )
                    return
            
            elif mode == "entire":
                if len(self.df.index) > 0:
                    t0 = self.df.index.min()
                    t1 = self.df.index.max()
                else:
                    t0, t1 = self.t0, self.t1
                mode_name = "Entire Dataset"
            
            else:
                return
            
            # Close this dialog and show confirmation
            dialog.destroy()
            self._show_clear_confirmation(t0, t1, mode_name)
        
        def on_cancel():
            """Close dialog without action."""
            dialog.destroy()
        
        # Next button
        next_button = ttk.Button(
            button_frame,
            text="Next",
            command=on_next,
            width=10
        )
        next_button.pack(side=tk.LEFT, padx=5)
        
        # Cancel button
        cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=on_cancel,
            width=10
        )
        cancel_button.pack(side=tk.LEFT, padx=5)
        
        # Keyboard shortcuts
        dialog.bind("<Return>", lambda e: on_next())
        dialog.bind("<Escape>", lambda e: on_cancel())
        
        # Center dialog on parent
        dialog.update_idletasks()
        
        # Calculate position relative to root window
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_width = self.root.winfo_width()
        root_height = self.root.winfo_height()
        
        dialog_width = dialog.winfo_reqwidth()
        dialog_height = dialog.winfo_reqheight()
        
        x = root_x + (root_width - dialog_width) // 2
        y = root_y + (root_height - dialog_height) // 2
        
        dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
        
        # Bring dialog to front
        dialog.lift()
        dialog.focus_set()
        
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
        
        # Check if the currently selected interval still exists after redo
        if hasattr(self, 'selected_interval') and self.selected_interval is not None:
            if self.selected_interval not in self.intervals:
                self.selected_interval = None
                if hasattr(self, '_clear_selected_interval_highlights'):
                    self._clear_selected_interval_highlights()
        
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
            # Clear interval highlights when all intervals are cleared
            if hasattr(self, '_clear_selected_interval_highlights'):
                self._clear_selected_interval_highlights()
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.modified = True
            # Refresh UI
            self._update_plot()
            self._update_intervals_list()
            if self.status_var is not None:
                self.status_var.set("All intervals cleared")
