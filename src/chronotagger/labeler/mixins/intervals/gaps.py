"""
Gap detection and labeling mixin for intervals.

This module provides functionality for:
- Finding unlabeled gaps in the current time range
- Assigning labels to unassigned time periods
- Interactive dialog for selecting labels for gaps
"""

from __future__ import annotations
from typing import List, Tuple
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk

from chronotagger.core.models import Interval
from chronotagger.core.commands import AddIntervalCommand


class IntervalGapsMixin:
    """Mixin providing gap detection and labeling functionality."""

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
        self._save_autosave()

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
        ok_button = tk.Button(
            button_frame,
            text="OK",
            command=on_ok,
            width=15
        )
        ok_button.pack(side=tk.LEFT, padx=8)

        # Cancel button
        cancel_button = tk.Button(
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
        self._save_autosave()
