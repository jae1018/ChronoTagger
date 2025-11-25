"""
Overlap resolution dialog for manual interval additions.

Shows when user clicks "Add" and the preview overlaps with existing intervals.
Allows user to choose Skip or Replace policy, previews the result, then confirms.
"""

from __future__ import annotations

from typing import Optional, Callable

import tkinter as tk
from tkinter import ttk


class OverlapResolutionDialog(tk.Toplevel):
    """
    Modal dialog to resolve overlaps when adding intervals manually.
    
    Workflow:
    1. Show overlap warning with count
    2. User picks Skip or Replace (enables Confirm button)
    3. Preview updates to show actual result
    4. User confirms or cancels
    
    Usage:
        dialog = OverlapResolutionDialog(
            parent=root,
            overlap_count=3,
            on_policy_selected=update_preview_callback
        )
        
        # After dialog closes:
        if dialog.policy:
            # User confirmed with "skip" or "replace"
            apply_policy(dialog.policy)
        else:
            # User canceled
            clear_preview()
    """
    
    def __init__(
        self,
        parent: tk.Misc,
        overlap_count: int,
        on_policy_selected: Callable[[str], tuple[int, int]],
    ) -> None:
        """
        Create the overlap resolution dialog.
        
        Args:
            parent: Parent window
            overlap_count: Number of existing intervals that overlap with preview
            on_policy_selected: Callback(policy) -> (num_points, num_intervals)
                Called when user selects Skip or Replace to update preview
        """
        super().__init__(parent)
        self.title("Overlap Detected")
        self.transient(parent)
        self.grab_set()
        self.resizable(False, False)
        
        self._overlap_count = overlap_count
        self._on_policy_cb = on_policy_selected
        
        # State
        self._policy_var = tk.StringVar(value="")  # "" | "skip" | "replace"
        self._last_counts: Optional[tuple[int, int]] = None  # (points, intervals)
        
        # Build UI
        self._build_ui()
        
        # Result
        self.policy: Optional[str] = None
        
        # Finalize
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.wait_visibility()
        self.focus()
    
    def _build_ui(self):
        """Build the dialog UI."""
        main = ttk.Frame(self, padding=15)
        main.pack(fill="both", expand=True)
        
        row = 0
        
        # ─── Warning message ───
        warning_frame = ttk.Frame(main)
        warning_frame.grid(row=row, column=0, sticky="ew", pady=(0, 15))
        
        # Warning icon (⚠)
        icon_label = ttk.Label(
            warning_frame, 
            text="⚠️", 
            font=("", 24)
        )
        icon_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Warning text
        msg_frame = ttk.Frame(warning_frame)
        msg_frame.pack(side=tk.LEFT, fill="both", expand=True)
        
        ttk.Label(
            msg_frame,
            text="Selection overlaps with existing intervals",
            font=("", 10, "bold")
        ).pack(anchor="w")
        
        count_msg = f"Overlaps with {self._overlap_count} interval"
        if self._overlap_count > 1:
            count_msg += "s"
        
        ttk.Label(
            msg_frame,
            text=count_msg,
            foreground="#555"
        ).pack(anchor="w", pady=(2, 0))
        
        row += 1
        
        # ─── Policy selection ───
        policy_frame = ttk.LabelFrame(main, text="How to handle overlaps:", padding=10)
        policy_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Radiobutton(
            policy_frame,
            text="Skip overlaps (only label unassigned regions)",
            value="skip",
            variable=self._policy_var,
            command=self._on_policy_change
        ).pack(anchor="w", pady=4)
        
        ttk.Radiobutton(
            policy_frame,
            text="Replace overlaps (delete conflicting intervals)",
            value="replace",
            variable=self._policy_var,
            command=self._on_policy_change
        ).pack(anchor="w", pady=4)
        
        row += 1
        
        # ─── Preview feedback ───
        self._feedback = ttk.Label(
            main, 
            text="Select a policy to preview result",
            foreground="#555",
            font=("", 9, "italic")
        )
        self._feedback.grid(row=row, column=0, sticky="w", pady=(0, 15))
        
        row += 1
        
        # ─── Buttons ───
        btns = ttk.Frame(main)
        btns.grid(row=row, column=0, sticky="e")
        
        ttk.Button(
            btns, 
            text="Cancel", 
            command=self._on_cancel
        ).pack(side=tk.RIGHT, padx=(8, 0))
        
        self._confirm_btn = ttk.Button(
            btns,
            text="Confirm",
            command=self._on_confirm,
            state="disabled"  # Enabled after policy selection
        )
        self._confirm_btn.pack(side=tk.RIGHT)
        
        # Configure grid
        main.columnconfigure(0, weight=1)
    
    def _on_policy_change(self):
        """Handle policy selection - update preview and enable Confirm."""
        policy = self._policy_var.get()
        if not policy:
            return
        
        try:
            # Call preview callback with selected policy
            pts, spans = self._on_policy_cb(policy)
            self._last_counts = (pts, spans)
            
            # Update feedback message
            if spans == 0:
                self._feedback.config(
                    text="⚠️ Nothing to add after applying policy",
                    foreground="#d9534f"
                )
                self._confirm_btn.configure(state="disabled")
            else:
                policy_name = "skipping overlaps" if policy == "skip" else "replacing overlaps"
                self._feedback.config(
                    text=f"Will create {spans} interval{'s' if spans != 1 else ''} after {policy_name}",
                    foreground="#5cb85c",
                    font=("", 9, "normal")
                )
                self._confirm_btn.configure(state="normal")
            
        except Exception as e:
            self._feedback.config(
                text=f"Error: {str(e)}",
                foreground="#d9534f"
            )
            self._confirm_btn.configure(state="disabled")
    
    def _on_confirm(self):
        """User confirmed - store policy and close."""
        policy = self._policy_var.get()
        if not policy:
            return
        
        # Check if policy eliminated all intervals
        if self._last_counts and self._last_counts[1] == 0:
            # Nothing to add - should already be disabled, but double-check
            return
        
        self.policy = policy
        self.destroy()
    
    def _on_cancel(self):
        """User canceled - no policy stored."""
        self.policy = None
        self.destroy()
