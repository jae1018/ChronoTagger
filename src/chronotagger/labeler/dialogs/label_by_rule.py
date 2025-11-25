"""
Label by Rule dialog with support for multiple conditions combined with AND/OR logic.

Key Features:
- Single condition (backward compatible with original behavior)
- Multiple conditions (new) combined with AND or OR
- Clean, modular condition row management
- Robust validation and error handling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import tkinter as tk
from tkinter import ttk, messagebox


@dataclass
class RuleCondition:
    """
    Single condition in a rule.
    
    Examples:
        RuleCondition(column="X_GSE", op="<", value=0.0)
        RuleCondition(column="BX", op=">=", value=5.0)
    """
    column: str
    op: str      # "<", "<=", "==", ">=", ">", "!="
    value: float


@dataclass
class LabelByRuleResult:
    """
    Complete rule specification with one or more conditions.
    
    Attributes:
        conditions: List of conditions to evaluate
        combine_mode: How to combine conditions - "AND" (all must be true) or "OR" (any must be true)
        nan_as_true: Whether NaN values should be treated as True
        overlap_policy: "skip" (don't cover existing) or "replace" (carve existing, keep new)
        scope: "window" (current view), "dataset" (all data), or "custom" (date range)
        custom_start: Start time for custom scope (ISO format string)
        custom_end: End time for custom scope (ISO format string)
    """
    conditions: List[RuleCondition]
    combine_mode: str  # "AND" | "OR"
    nan_as_true: bool
    overlap_policy: str   # "skip" | "replace"
    scope: str            # "window" | "dataset" | "custom"
    custom_start: Optional[str] = None
    custom_end: Optional[str] = None


class ConditionRow:
    """
    Manages UI widgets for a single condition row.
    
    Encapsulates the column dropdown, operator dropdown, value entry, and optional remove button.
    Makes it easy to add/remove condition rows dynamically.
    """
    
    def __init__(
        self,
        parent: ttk.Frame,
        row_number: int,
        numeric_columns: List[str],
        on_remove: Optional[Callable[[], None]] = None,
        initial_column: Optional[str] = None,
        initial_op: str = ">=",
        initial_value: str = "0",
    ):
        """
        Create a condition row.
        
        Args:
            parent: Parent frame to place widgets in
            row_number: Display number for the row (1, 2, 3, ...)
            numeric_columns: Available columns for the dropdown
            on_remove: Callback when remove button is clicked (None = no button)
            initial_column: Default column (uses first if None)
            initial_op: Default operator
            initial_value: Default value
        """
        self.frame = ttk.LabelFrame(parent, text=f"Condition {row_number}", padding=8)
        self.row_number = row_number
        
        # State variables
        self.col_var = tk.StringVar(value=initial_column or numeric_columns[0])
        self.op_var = tk.StringVar(value=initial_op)
        self.val_var = tk.StringVar(value=initial_value)
        
        # Build UI
        inner = ttk.Frame(self.frame)
        inner.pack(fill="both", expand=True)
        
        # Column selection
        col_row = ttk.Frame(inner)
        col_row.pack(fill="x", pady=2)
        ttk.Label(col_row, text="Column:", width=10).pack(side=tk.LEFT, padx=(0, 6))
        self.col_combo = ttk.Combobox(
            col_row,
            textvariable=self.col_var,
            values=numeric_columns,
            state="readonly",
            width=20
        )
        self.col_combo.pack(side=tk.LEFT, fill="x", expand=True)
        
        # Operator and Value on same row
        op_val_row = ttk.Frame(inner)
        op_val_row.pack(fill="x", pady=2)
        
        ttk.Label(op_val_row, text="Operator:", width=10).pack(side=tk.LEFT, padx=(0, 6))
        self.op_combo = ttk.Combobox(
            op_val_row,
            textvariable=self.op_var,
            values=["<", "<=", "==", ">=", ">", "!="],
            state="readonly",
            width=6
        )
        self.op_combo.pack(side=tk.LEFT, padx=(0, 12))
        
        ttk.Label(op_val_row, text="Value:").pack(side=tk.LEFT, padx=(0, 6))
        self.val_entry = ttk.Entry(op_val_row, textvariable=self.val_var, width=12)
        self.val_entry.pack(side=tk.LEFT, fill="x", expand=True)
        
        # Optional remove button
        if on_remove is not None:
            btn_row = ttk.Frame(inner)
            btn_row.pack(fill="x", pady=(4, 0))
            self.remove_btn = tk.Button(
                btn_row,
                text="− Remove",
                command=on_remove,
                width=10
            )
            self.remove_btn.pack(side=tk.RIGHT)
        else:
            self.remove_btn = None
    
    def pack(self, **kwargs):
        """Pack the frame into parent."""
        self.frame.pack(**kwargs)
    
    def destroy(self):
        """Remove this row from the UI."""
        self.frame.destroy()
    
    def get_condition(self) -> Optional[RuleCondition]:
        """
        Extract condition from this row.
        
        Returns:
            RuleCondition if valid, None if validation fails
        """
        try:
            col = self.col_var.get().strip()
            op = self.op_var.get().strip()
            val = float(self.val_var.get())
            
            if not col or op not in {"<", "<=", "==", ">=", ">", "!="}:
                return None
                
            return RuleCondition(column=col, op=op, value=val)
        except ValueError:
            return None


class LabelByRuleDialog(tk.Toplevel):
    """
    Modal dialog to define one or more rule conditions and preview/confirm them.
    
    Features:
    - Add multiple conditions (Column, Operator, Value)
    - Combine conditions with AND (all must be true) or OR (any must be true)
    - Preview results before applying
    - Backward compatible: single condition works exactly like before
    
    Usage:
        dialog = LabelByRuleDialog(
            parent=root,
            numeric_columns=df.select_dtypes(include=[np.number]).columns.tolist(),
            on_preview=my_preview_func,
            on_clear_preview=my_clear_func
        )
        
        # After dialog closes:
        if dialog.result:
            # Use dialog.result.conditions, combine_mode, etc.
            pass
    """
    
    def __init__(
        self,
        parent: tk.Misc,
        numeric_columns: List[str],
        on_preview: Callable[[LabelByRuleResult], tuple[int, int]],
        on_clear_preview: Callable[[], None],
    ) -> None:
        """
        Create the Label by Rule dialog.
        
        Args:
            parent: Parent window
            numeric_columns: List of numeric column names from dataframe
            on_preview: Callback(result) -> (num_points, num_spans) to preview rule
            on_clear_preview: Callback to clear preview from UI
        """
        super().__init__(parent)
        self.title("Label by Rule")
        self.transient(parent)
        self.grab_set()
        self.resizable(False, False)

        if not numeric_columns:
            messagebox.showerror(
                "No Numeric Columns",
                "Your DataFrame has no numeric columns.",
                parent=parent
            )
            self.result = None
            self.destroy()
            return

        self._numeric_columns = numeric_columns
        self._on_preview_cb = on_preview
        self._on_clear_preview_cb = on_clear_preview
        self._last_counts: Optional[tuple[int, int]] = None  # (points, spans)
        
        # Condition rows (start with 1)
        self._condition_rows: List[ConditionRow] = []
        self._next_row_num = 1

        # --- State vars ---
        self._combine_var = tk.StringVar(value="AND")  # NEW: AND or OR
        self._nan_true_var = tk.BooleanVar(value=False)
        self._policy_var = tk.StringVar(value="skip")      # "skip" | "replace"
        self._scope_var = tk.StringVar(value="window")     # "window" | "dataset" | "custom"
        self._cust_start_var = tk.StringVar(value="")
        self._cust_end_var = tk.StringVar(value="")

        # --- Build UI ---
        self._build_ui()
        
        # Finalize
        self.result: Optional[LabelByRuleResult] = None
        self.bind("<Return>", lambda e: self._on_preview())
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.wait_visibility()
        self.focus()

    def _build_ui(self):
        """Build the complete dialog UI."""
        main = ttk.Frame(self, padding=10)
        main.pack(fill="both", expand=True)
        
        row = 0
        
        # ─── Combine Mode (NEW) ───
        combine_frame = ttk.LabelFrame(main, text="Combine Conditions", padding=6)
        combine_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 8))
        
        combine_inner = ttk.Frame(combine_frame)
        combine_inner.pack()
        ttk.Radiobutton(
            combine_inner,
            text="AND (all conditions must be true)",
            value="AND",
            variable=self._combine_var
        ).pack(side=tk.LEFT, padx=8)
        ttk.Radiobutton(
            combine_inner,
            text="OR (any condition must be true)",
            value="OR",
            variable=self._combine_var
        ).pack(side=tk.LEFT, padx=8)
        
        row += 1
        
        # ─── Conditions Container (scrollable if needed) ───
        conditions_frame = ttk.LabelFrame(main, text="Conditions", padding=6)
        conditions_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 6))
        
        # Scrollable frame for conditions (in case user adds many)
        self._conditions_canvas = tk.Canvas(conditions_frame, height=200, highlightthickness=0)
        self._conditions_inner = ttk.Frame(self._conditions_canvas)
        
        scrollbar = ttk.Scrollbar(
            conditions_frame,
            orient="vertical",
            command=self._conditions_canvas.yview
        )
        self._conditions_canvas.configure(yscrollcommand=scrollbar.set)
        
        self._conditions_canvas.pack(side=tk.LEFT, fill="both", expand=True)
        scrollbar.pack(side=tk.RIGHT, fill="y")
        
        self._canvas_window = self._conditions_canvas.create_window(
            (0, 0),
            window=self._conditions_inner,
            anchor="nw"
        )
        
        # Update scroll region when inner frame changes size
        self._conditions_inner.bind(
            "<Configure>",
            lambda e: self._conditions_canvas.configure(scrollregion=self._conditions_canvas.bbox("all"))
        )
        
        # Add first condition row
        self._add_condition_row()
        
        # Add Condition button
        add_btn_frame = ttk.Frame(conditions_frame)
        add_btn_frame.pack(fill="x", pady=(6, 0))
        self._add_btn = tk.Button(
            add_btn_frame,
            text="+ Add Condition",
            command=self._add_condition_row
        )
        self._add_btn.pack(anchor="center")
        
        row += 1
        
        # ─── NaN Policy ───
        nan_frame = ttk.LabelFrame(main, text="NaNs should be regarded as…", padding=6)
        nan_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=10, pady=6)
        ttk.Radiobutton(nan_frame, text="True", value=True, variable=self._nan_true_var)\
            .pack(side=tk.LEFT, padx=8, pady=4)
        ttk.Radiobutton(nan_frame, text="False", value=False, variable=self._nan_true_var)\
            .pack(side=tk.LEFT, padx=8, pady=4)

        row += 1
        
        # ─── Overlap Policy ───
        pol = ttk.LabelFrame(main, text="When new spans overlap existing intervals…", padding=6)
        pol.grid(row=row, column=0, columnspan=2, sticky="ew", padx=10, pady=6)
        ttk.Radiobutton(
            pol,
            text="Skip overlaps (do not cover existing)",
            value="skip",
            variable=self._policy_var
        ).pack(anchor="w", padx=8, pady=2)
        ttk.Radiobutton(
            pol,
            text="Replace overlaps (carve existing, keep new)",
            value="replace",
            variable=self._policy_var
        ).pack(anchor="w", padx=8, pady=2)

        row += 1
        
        # ─── Scope ───
        scope = ttk.LabelFrame(main, text="Apply to", padding=6)
        scope.grid(row=row, column=0, columnspan=2, sticky="ew", padx=10, pady=6)
        
        rb_row = ttk.Frame(scope)
        rb_row.pack(fill="x", padx=6, pady=4)
        ttk.Radiobutton(rb_row, text="Current window", value="window", variable=self._scope_var)\
            .pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(rb_row, text="Entire dataset", value="dataset", variable=self._scope_var)\
            .pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(rb_row, text="Custom range", value="custom", variable=self._scope_var)\
            .pack(side=tk.LEFT, padx=6)

        cust = ttk.Frame(scope)
        cust.pack(fill="x", padx=12, pady=(4, 8))
        ttk.Label(cust, text="Start:").pack(side=tk.LEFT, padx=(0, 6))
        self._cust_start = ttk.Entry(cust, textvariable=self._cust_start_var, width=22)
        self._cust_start.pack(side=tk.LEFT, padx=(0, 18))
        ttk.Label(cust, text="End:").pack(side=tk.LEFT, padx=(0, 6))
        self._cust_end = ttk.Entry(cust, textvariable=self._cust_end_var, width=22)
        self._cust_end.pack(side=tk.LEFT)

        def _toggle_custom(*_):
            on = (self._scope_var.get() == "custom")
            state = "normal" if on else "disabled"
            self._cust_start.configure(state=state)
            self._cust_end.configure(state=state)
        
        self._scope_var.trace_add("write", _toggle_custom)
        _toggle_custom()

        row += 1
        
        # ─── Feedback ───
        self._feedback = ttk.Label(main, text="No preview yet.", foreground="#555")
        self._feedback.grid(row=row, column=0, columnspan=2, sticky="w", padx=10, pady=6)

        row += 1
        
        # ─── Buttons ───
        btns = ttk.Frame(main)
        btns.grid(row=row, column=0, columnspan=2, sticky="e", padx=10, pady=(10, 0))
        
        self._preview_btn = tk.Button(btns, text="Preview", command=self._on_preview)
        self._preview_btn.pack(side=tk.LEFT, padx=4)
        
        tk.Button(btns, text="Clear preview", command=self._on_clear).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT, padx=4)
        
        self._ok_btn = tk.Button(btns, text="OK", command=self._on_ok)
        self._ok_btn.pack(side=tk.RIGHT, padx=4)
        
        # Require a preview before allowing OK
        self._ok_btn.configure(state="disabled")
        
        # Configure grid weights
        main.columnconfigure(1, weight=1)

    def _add_condition_row(self):
        """Add a new condition row to the dialog."""
        row_num = self._next_row_num
        self._next_row_num += 1
        
        # Only first row has no remove button
        show_remove = len(self._condition_rows) > 0
        
        def on_remove():
            self._remove_condition_row(row)
        
        row = ConditionRow(
            parent=self._conditions_inner,
            row_number=row_num,
            numeric_columns=self._numeric_columns,
            on_remove=on_remove if show_remove else None,
        )
        
        row.pack(fill="x", padx=4, pady=4)
        self._condition_rows.append(row)
        
        # Update canvas scroll region
        self._conditions_canvas.update_idletasks()
        self._conditions_canvas.configure(scrollregion=self._conditions_canvas.bbox("all"))

    def _remove_condition_row(self, row: ConditionRow):
        """Remove a condition row from the dialog."""
        if len(self._condition_rows) <= 1:
            messagebox.showwarning(
                "Cannot Remove",
                "At least one condition is required.",
                parent=self
            )
            return
        
        row.destroy()
        self._condition_rows.remove(row)
        
        # Update canvas scroll region
        self._conditions_canvas.update_idletasks()
        self._conditions_canvas.configure(scrollregion=self._conditions_canvas.bbox("all"))
        
        # Renumber remaining rows
        for i, r in enumerate(self._condition_rows, start=1):
            r.frame.configure(text=f"Condition {i}")

    def _collect(self) -> Optional[LabelByRuleResult]:
        """
        Collect all conditions and settings from the dialog.
        
        Returns:
            LabelByRuleResult if valid, None if validation fails
        """
        # Collect all conditions
        conditions = []
        for i, row in enumerate(self._condition_rows, start=1):
            cond = row.get_condition()
            if cond is None:
                messagebox.showerror(
                    "Invalid Condition",
                    f"Condition {i} has invalid or missing values. "
                    f"Please check column, operator, and value.",
                    parent=self
                )
                return None
            conditions.append(cond)
        
        if not conditions:
            messagebox.showerror(
                "No Conditions",
                "Please add at least one condition.",
                parent=self
            )
            return None
        
        # Validate scope
        scope = self._scope_var.get()
        cs = self._cust_start_var.get().strip() or None
        ce = self._cust_end_var.get().strip() or None
        if scope == "custom" and (not cs or not ce):
            messagebox.showerror(
                "Custom Range",
                "Please enter both Start and End for the custom scope.",
                parent=self
            )
            return None

        return LabelByRuleResult(
            conditions=conditions,
            combine_mode=self._combine_var.get(),
            nan_as_true=bool(self._nan_true_var.get()),
            overlap_policy=self._policy_var.get() or "skip",
            scope=scope,
            custom_start=cs,
            custom_end=ce,
        )

    def _on_preview(self) -> None:
        """Preview the rule results."""
        res = self._collect()
        if not res:
            return
        
        try:
            pts, spans = self._on_preview_cb(res)
            self._last_counts = (pts, spans)

            scope_txt = {
                "window":  "current window",
                "dataset": "entire dataset",
                "custom":  "custom range"
            }[res.scope]
            
            pol_txt = res.overlap_policy
            mode_txt = res.combine_mode
            cond_count = len(res.conditions)

            # Build descriptive feedback
            if cond_count == 1:
                self._feedback.config(
                    text=f"Preview ({scope_txt}; policy={pol_txt}): {pts} points → {spans} spans"
                )
            else:
                self._feedback.config(
                    text=f"Preview ({cond_count} conditions with {mode_txt}; {scope_txt}; "
                         f"policy={pol_txt}): {pts} points → {spans} spans"
                )

            # Enable OK only if there is something to add after policy
            self._ok_btn.configure(state=("normal" if spans > 0 else "disabled"))
            
        except Exception as e:
            messagebox.showerror("Preview Failed", str(e), parent=self)

    def _on_clear(self) -> None:
        """Clear the preview."""
        try:
            self._on_clear_preview_cb()
            self._last_counts = None
            self._feedback.config(text="Preview cleared.")
            self._ok_btn.configure(state="disabled")
        except Exception as e:
            messagebox.showerror("Clear Failed", str(e), parent=self)

    def _on_ok(self) -> None:
        """Accept the rule and close the dialog."""
        if self._last_counts is None:
            self._on_preview()
            if self._last_counts is None:
                return

        # If policy trimmed everything away, keep the dialog open
        if self._last_counts and self._last_counts[1] == 0:
            messagebox.showwarning(
                "Nothing to Add",
                "No spans remain after applying the overlap policy. "
                "Adjust the rule or policy, then Preview again.",
                parent=self,
            )
            return

        self.result = self._collect()
        if self.result is None:
            return
        self.destroy()

    def _on_cancel(self) -> None:
        """Cancel and close the dialog."""
        # Clear any active preview before closing
        try:
            self._on_clear_preview_cb()
        except Exception:
            pass  # Silently ignore errors during cleanup
        
        self.result = None
        self.destroy()
