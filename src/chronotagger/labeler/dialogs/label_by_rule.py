from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import tkinter as tk
from tkinter import ttk, messagebox


@dataclass
class LabelByRuleResult:
    column: str
    op: str               # "<", "<=", "==", ">=", ">", "!="
    value: float
    nan_as_true: bool
    overlap_policy: str   # "skip" | "replace"
    scope: str            # "window" | "dataset" | "custom"
    custom_start: Optional[str] = None   # ISO-ish strings; parsed by controller
    custom_end: Optional[str] = None


class LabelByRuleDialog(tk.Toplevel):
    """
    Modal dialog to define a single rule and preview/confirm it.

    Calls the provided callbacks (non-mutating):
      - on_preview(result) -> (points:int, spans:int) across the chosen scope
      - on_clear_preview()

    Sets self.result when OK is pressed. Actual labeling happens when the user
    clicks the app's "Add Label" button (uses _commit_spans prepared on preview).
    """
    def __init__(
        self,
        parent: tk.Misc,
        numeric_columns: List[str],
        on_preview: Callable[[LabelByRuleResult], tuple[int, int]],
        on_clear_preview: Callable[[], None],
    ) -> None:
        super().__init__(parent)
        self.title("Label by Rule")
        self.transient(parent)
        self.grab_set()
        self.resizable(False, False)

        if not numeric_columns:
            messagebox.showerror("No Numeric Columns", "Your DataFrame has no numeric columns.", parent=parent)
            self.result = None
            self.destroy()
            return

        self._on_preview_cb = on_preview
        self._on_clear_preview_cb = on_clear_preview
        self._last_counts: Optional[tuple[int, int]] = None  # (points, spans)

        # --- State vars ---
        self._col_var = tk.StringVar(value=numeric_columns[0])
        self._op_var = tk.StringVar(value=">=")
        self._val_var = tk.StringVar(value="0")
        self._nan_true_var = tk.BooleanVar(value=False)

        self._policy_var = tk.StringVar(value="skip")      # "skip" | "replace"
        self._scope_var = tk.StringVar(value="window")     # "window" | "dataset" | "custom"
        self._cust_start_var = tk.StringVar(value="")
        self._cust_end_var = tk.StringVar(value="")

        # --- UI ---
        pad = {"padx": 10, "pady": 6}
        main = ttk.Frame(self, padding=10)
        main.grid(row=0, column=0, sticky="nsew")

        # Row 0: Column
        row = 0
        ttk.Label(main, text="Column:").grid(row=row, column=0, sticky="w", **pad)
        self._col_combo = ttk.Combobox(
            main, textvariable=self._col_var, values=numeric_columns, state="readonly", width=28
        )
        self._col_combo.grid(row=row, column=1, sticky="ew", **pad)

        # Row 1: Operator + Value
        row += 1
        op_row = ttk.Frame(main)
        op_row.grid(row=row, column=0, columnspan=2, sticky="ew", **pad)
        ttk.Label(op_row, text="Operator:").pack(side=tk.LEFT, padx=(0, 8))
        self._op_combo = ttk.Combobox(
            op_row, textvariable=self._op_var,
            values=["<", "<=", "==", ">=", ">", "!="], state="readonly", width=8
        )
        self._op_combo.pack(side=tk.LEFT)
        ttk.Label(op_row, text="Value:").pack(side=tk.LEFT, padx=(18, 8))
        self._val_entry = ttk.Entry(op_row, textvariable=self._val_var, width=16)
        self._val_entry.pack(side=tk.LEFT)

        # Row 2: NaN policy
        row += 1
        nan_frame = ttk.LabelFrame(main, text="NaNs should be regarded as…")
        nan_frame.grid(row=row, column=0, columnspan=2, sticky="ew", **pad)
        ttk.Radiobutton(nan_frame, text="True", value=True, variable=self._nan_true_var)\
            .pack(side=tk.LEFT, padx=8, pady=4)
        ttk.Radiobutton(nan_frame, text="False", value=False, variable=self._nan_true_var)\
            .pack(side=tk.LEFT, padx=8, pady=4)

        # Row 3: Overlap policy
        row += 1
        pol = ttk.LabelFrame(main, text="When new spans overlap existing intervals…")
        pol.grid(row=row, column=0, columnspan=2, sticky="ew", **pad)
        ttk.Radiobutton(pol, text="Skip overlaps (do not cover existing)", value="skip", variable=self._policy_var)\
            .pack(anchor="w", padx=8, pady=2)
        ttk.Radiobutton(pol, text="Replace overlaps (carve existing, keep new)", value="replace", variable=self._policy_var)\
            .pack(anchor="w", padx=8, pady=2)

        # Row 4: Scope
        row += 1
        scope = ttk.LabelFrame(main, text="Apply to")
        scope.grid(row=row, column=0, columnspan=2, sticky="ew", **pad)
        rb_row = ttk.Frame(scope); rb_row.pack(fill="x", padx=6, pady=4)
        ttk.Radiobutton(rb_row, text="Current window", value="window", variable=self._scope_var)\
            .pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(rb_row, text="Entire dataset", value="dataset", variable=self._scope_var)\
            .pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(rb_row, text="Custom range", value="custom", variable=self._scope_var)\
            .pack(side=tk.LEFT, padx=6)

        cust = ttk.Frame(scope); cust.pack(fill="x", padx=12, pady=(4, 8))
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

        # Row 5: Feedback
        row += 1
        self._feedback = ttk.Label(main, text="No preview yet.", foreground="#555")
        self._feedback.grid(row=row, column=0, columnspan=2, sticky="w", **pad)

        # Row 6: Buttons
        btns = ttk.Frame(main)
        btns.grid(row=row, column=0, columnspan=2, sticky="e", padx=10, pady=(10, 0))
        self._preview_btn = ttk.Button(btns, text="Preview", command=self._on_preview)
        self._preview_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Clear preview", command=self._on_clear).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT, padx=4)
        self._ok_btn = ttk.Button(btns, text="OK", command=self._on_ok)
        self._ok_btn.pack(side=tk.RIGHT, padx=4)
        # Require a preview before allowing OK
        self._ok_btn.configure(state="disabled")

        # Finalize
        self.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

        self.result: Optional[LabelByRuleResult] = None
        self.bind("<Return>", lambda e: self._on_preview())
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.wait_visibility()
        self.focus()

    # --- actions ---
    def _collect(self) -> Optional[LabelByRuleResult]:
        try:
            val = float(self._val_var.get())
        except Exception:
            messagebox.showerror("Invalid Value", "Please enter a numeric value.", parent=self)
            return None

        col = (self._col_var.get() or "").strip()
        if not col:
            messagebox.showerror("Missing Column", "Please choose a column.", parent=self)
            return None

        op = (self._op_var.get() or "").strip()
        if op not in {"<", "<=", "==", ">=", ">", "!="}:
            messagebox.showerror("Invalid Operator", "Choose a valid operator.", parent=self)
            return None

        scope = self._scope_var.get()
        cs = self._cust_start_var.get().strip() or None
        ce = self._cust_end_var.get().strip() or None
        if scope == "custom" and (not cs or not ce):
            messagebox.showerror("Custom Range", "Please enter both Start and End for the custom scope.", parent=self)
            return None

        return LabelByRuleResult(
            column=col,
            op=op,
            value=val,
            nan_as_true=bool(self._nan_true_var.get()),
            overlap_policy=self._policy_var.get() or "skip",
            scope=scope,
            custom_start=cs,
            custom_end=ce,
        )

    def _on_preview(self) -> None:
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
            pol_txt = (res.overlap_policy or "skip")

            # Text reflects the chosen overlap policy
            self._feedback.config(
                text=f"Preview ({scope_txt}; policy={pol_txt}): {pts} points \u2192 {spans} spans"
            )

            # Enable OK only if there is something to add after policy
            self._ok_btn.configure(state=("normal" if spans > 0 else "disabled"))
        except Exception as e:
            messagebox.showerror("Preview Failed", str(e), parent=self)

    def _on_clear(self) -> None:
        try:
            self._on_clear_preview_cb()
            self._last_counts = None
            self._feedback.config(text="Preview cleared.")
            self._ok_btn.configure(state="disabled")  # <-- add this
        except Exception as e:
            messagebox.showerror("Clear Failed", str(e), parent=self)

    def _on_ok(self) -> None:
        if self._last_counts is None:
            self._on_preview()
            if self._last_counts is None:
                return

        # If policy trimmed everything away, keep the dialog open and inform the user
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
        self.result = None
        self.destroy()