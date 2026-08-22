"""
Session I/O and export mixin.

All persistence writes go through utils.atomic_io: the target file
always holds either the complete old content or the complete new
content, never a truncated hybrid (Pack 2).
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List

import hashlib
import json
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np

from chronotagger.core.models import Interval
from ..utils.atomic_io import atomic_write_json, atomic_write_path


def _norm_iso(ts: pd.Timestamp) -> str:
    """tz-normalized isoformat: tz-aware timestamps are converted to
    UTC and made naive, so a naive frame and its UTC-localized twin
    produce IDENTICAL strings. Used by BOTH the fingerprint and the
    saved/compared time_range, so the identity check can never
    contradict the fingerprint about timezones (fold V1/V2/V3 tz)."""
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.isoformat()


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """
    12-hex identity used in the autosave filename: sha1 of sorted
    column names + tz-normalized index bounds + row count (grill Q1,
    recipe R1). Stable across column reorder, dtype casts, and value
    edits; changes when columns are added/renamed or the time range
    changes. HONEST LIMIT (evidence map 3b): two datasets with
    identical column names AND identical time coverage (e.g. two
    spacecraft through a shared loader) share a fingerprint -- the
    source_name comparison in _check_autosave is the guard for that
    case, when a source name is known.
    """
    key = "|".join([
        ",".join(sorted(str(c) for c in df.columns)),
        _norm_iso(df.index[0]),
        _norm_iso(df.index[-1]),
        str(len(df.index)),
    ])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


class IOExportMixin:
    def _dataset_fingerprint(self) -> str:
        """Fingerprint of the currently loaded DataFrame (see
        dataset_fingerprint)."""
        return dataset_fingerprint(self.df)

    # ---- Public convenience wrappers ----
    def save(self, path: Optional[str] = None) -> None:
        self._save_session(path)

    def load(self, path: str) -> None:
        self._load_session(path)

    def export_intervals(self, path: str, fmt: str = "parquet") -> None:
        if not self.intervals:
            # Scripts must fail loudly: a print-and-return leaves a
            # pipeline with no exception and no file (grill Q7).
            raise ValueError("No intervals to export.")
        rows = [
            {"start": iv.start, "end": iv.end, "label": iv.label, "notes": iv.notes}
            for iv in self.intervals
        ]
        df_export = pd.DataFrame(rows)
        if fmt.lower() == "parquet":
            atomic_write_path(path, lambda p: df_export.to_parquet(p, index=False))
        else:
            atomic_write_path(path, lambda p: df_export.to_csv(p, index=False))
        print(f"Exported intervals to {path}")

    def export_per_sample(
        self, path: str, fmt: str = "parquet", label_on_uncovered: Optional[str] = "UNKNOWN"
    ) -> None:
        if not self.intervals:
            raise ValueError("No intervals to export.")
        labels: List[Optional[str]] = []
        for ts in self.df.index:
            lbl = None
            for iv in self.intervals:
                if iv.contains(ts):
                    lbl = iv.label
                    break
            labels.append(lbl if lbl is not None else label_on_uncovered)

        df_export = pd.DataFrame({"label": labels}, index=self.df.index)
        if fmt.lower() == "parquet":
            atomic_write_path(path, lambda p: df_export.to_parquet(p))
        else:
            atomic_write_path(path, lambda p: df_export.to_csv(p))
        print(f"Exported per-sample labels to {path}")

    # ---- GUI-connected ops ----
    def _save_session(self, path: Optional[str] = None) -> bool:
        """Save the session JSON atomically. Returns True only if the
        file was actually written (False on dialog cancel or failure) --
        _on_closing depends on this to never close on a failed save."""
        target = Path(path) if path else None
        if target is None:
            chosen = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            if not chosen:
                return False
            target = Path(chosen)

        data = {
            "version": 1,
            "classes": self.classes,
            "class_colors": self.class_colors,
            "window": str(self.window),
            "step": str(self.step),
            "data_start": self.data_start.isoformat(),
            "data_end": self.data_end.isoformat(),
            "intervals": [iv.to_dict() for iv in self.intervals],
            "layout_spec": self.layout_spec,  # Save layout configuration
            # Multi-pane metadata
            "multi_pane_mode": getattr(self, 'multi_pane_mode', False),
            "active_pane_idx": getattr(self, 'active_pane_idx', 0) if getattr(self, 'multi_pane_mode', False) else 0,
            "panes": [
                {
                    "title": pane.title,
                    "layout_spec": getattr(pane, 'layout_spec', None),
                }
                for pane in getattr(self, 'panes', [])
            ] if getattr(self, 'multi_pane_mode', False) else [],
        }
        try:
            atomic_write_json(target, data)
        except Exception as e:
            # GUI session: surface in a dialog and report failure.
            # Headless/library use: RE-RAISE -- a modal here would hang
            # a display-less script forever, and swallowing would be
            # the exact silent-failure shape this pack exists to kill
            # (fold V3-M: executed, it hangs).
            if getattr(self, 'root', None) is None:
                raise
            messagebox.showerror(
                "Save Failed",
                f"Could not save session:\n{e}\n\n"
                f"The previous file (if any) is unchanged.")
            if getattr(self, 'status_var', None) is not None:
                self.status_var.set("Save failed")
            return False

        self.modified = False
        if getattr(self, 'status_var', None) is not None:
            self.status_var.set(f"Saved to {target}")
        return True

    def _load_session(self, path: Optional[str] = None) -> None:
        if path is None:
            chosen = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not chosen:
                return
            path = chosen

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check if session has layout_spec (backward compatibility)
        saved_layout = data.get("layout_spec", None)
        
        if saved_layout is not None and self.layout_spec is not None:
            # Validate against current layout
            if not self._layouts_compatible(saved_layout, self.layout_spec):
                # Show warning dialog
                result = messagebox.askyesno(
                    "Layout Mismatch",
                    "This session was saved with a different layout configuration.\n\n"
                    f"Saved:   {self._describe_layout(saved_layout)}\n"
                    f"Current: {self._describe_layout(self.layout_spec)}\n\n"
                    "Loading this session may cause display issues.\n"
                    "Continue anyway?",
                    icon='warning'
                )
                if not result:
                    self.status_var.set("Load cancelled - layout mismatch")  # type: ignore[union-attr]
                    return  # User cancelled

        self.classes = list(data["classes"])
        self.class_colors = dict(data["class_colors"])
        self.window = pd.Timedelta(data["window"])
        self.step = pd.Timedelta(data["step"])
        self.intervals = [Interval.from_dict(d) for d in data["intervals"]]

        # A loaded session invalidates the undo history and any selection
        # made against the previous session's interval objects. In strict
        # mode, validate the loaded set NOW so a corrupt session file is
        # blamed on the load, not on the user's next gesture (fold V3-M3).
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.selected_interval = None
        if hasattr(self, '_clear_selected_interval_highlights'):
            self._clear_selected_interval_highlights()
        self._check_interval_invariants()

        self.modified = False

        if self.class_combo is not None and self.current_class_var is not None:
            self.class_combo["values"] = self.classes
            if self.current_class_var.get() not in self.classes:
                self.current_class_var.set(self.classes[0])

        if self.start_time_entry and self.end_time_entry and self.step_entry:
            self.start_time_entry.delete(0, tk.END)
            self.start_time_entry.insert(0, str(self.t0))
            self.end_time_entry.delete(0, tk.END)
            self.end_time_entry.insert(0, str(self.t1))
            self.step_entry.delete(0, tk.END)
            self.step_entry.insert(0, str(self.step))

        # Restore active tab if multi-pane
        if getattr(self, 'multi_pane_mode', False) and "active_pane_idx" in data:
            idx = data["active_pane_idx"]
            if hasattr(self, 'notebook') and hasattr(self, 'panes'):
                if 0 <= idx < len(self.panes):
                    self.active_pane_idx = idx
                    self.notebook.select(idx)

        # Restore pane titles if saved
        if getattr(self, 'multi_pane_mode', False) and "panes" in data:
            saved_panes = data["panes"]
            for i, saved_pane in enumerate(saved_panes):
                if i < len(self.panes) and "title" in saved_pane:
                    self.panes[i].title = saved_pane["title"]
                    if hasattr(self, 'notebook'):
                        self.notebook.tab(i, text=saved_pane["title"])

        self._update_plot()
        self.status_var.set(f"Loaded from {path}")  # type: ignore[union-attr]
        
    def _compute_label_id_series(self) -> pd.Series:
        """
        Build a vectorized per-sample label_id series aligned to self.df.index.
    
        Mapping:
          classes -> ids  (0..N-1) using current self.classes order
          Unlabeled samples -> -1
    
        Uses the smallest feasible integer dtype to keep CSVs compact.
        """
        import pandas as pd
    
        # Stable, deterministic mapping from current classes ordering
        label_to_id = {label: i for i, label in enumerate(self.classes)}
        unknown_id = -1
    
        # Smallest int dtype that fits the number of classes
        n = len(label_to_id)
        if n <= np.iinfo(np.int8).max:
            dtype = np.int8
        elif n <= np.iinfo(np.int16).max:
            dtype = np.int16
        else:
            dtype = np.int32
    
        idx = self.df.index
        ids = np.full(len(idx), fill_value=unknown_id, dtype=dtype)
    
        # NOTE: intervals are non-overlapping by construction
        for iv in self.intervals:
            s = idx.searchsorted(iv.start, side="left")
            e = idx.searchsorted(iv.end, side="left")
            if s < e:
                ids[s:e] = label_to_id.get(iv.label, unknown_id)
    
        return pd.Series(ids, index=idx, name="label_id")

    def _export_labels_dialog(self) -> None:
        """
        Enhanced modal with live preview that lets the user choose:
          - Scope: Full dataset  |  Selected intervals only
          - Content: Index + labels (CSV)  |  Full DF + labels (CSV)
        
        Shows real-time preview of first 10 rows and estimated total.
        Writes a CSV plus a sidecar '<chosen_name>_label_map.json'.
        """
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
    
        if self.df is None or len(self.df.index) == 0:
            messagebox.showwarning("No Data", "There is no data to export.")
            return

        # Orphan labels would render as -1 in the preview and the CSV;
        # refuse at the door so the preview never lies (grill Q4).
        orphans = sorted({iv.label for iv in self.intervals} - set(self.classes))
        if orphans:
            messagebox.showerror(
                "Export Blocked",
                "These interval labels are not in the current label schema "
                "and would be exported as -1 (unlabeled):\n\n"
                f"  {', '.join(orphans)}\n\n"
                "Fix them via Manage Labels..., then export again.")
            return
    
        # Modal container - larger size for side-by-side layout
        dlg = tk.Toplevel(self.root)
        dlg.title("Export Labels")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(True, True)
        dlg.geometry("800x500")  # Wider for preview
    
        # Main frame with two sides
        main_frame = ttk.Frame(dlg)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side: Options (30% width)
        options_frame = ttk.Frame(main_frame)
        options_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Right side: Preview (70% width)
        preview_frame = ttk.LabelFrame(main_frame, text="Preview")
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
        # --- Scope Options ---
        scope_var = tk.StringVar(value="full")
        scope_grp = ttk.LabelFrame(options_frame, text="Scope")
        scope_grp.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Radiobutton(
            scope_grp, text="Full dataset (unlabeled = -1)",
            variable=scope_var, value="full"
        ).pack(anchor="w", pady=2, padx=5)
        
        ttk.Radiobutton(
            scope_grp, text="Selected intervals only",
            variable=scope_var, value="selected"
        ).pack(anchor="w", pady=2, padx=5)
    
        # --- Content Options ---
        content_var = tk.StringVar(value="index_labels_csv")
        content_grp = ttk.LabelFrame(options_frame, text="Content")
        content_grp.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Radiobutton(
            content_grp, text="Index + labels (CSV)",
            variable=content_var, value="index_labels_csv"
        ).pack(anchor="w", pady=2, padx=5)
        
        ttk.Radiobutton(
            content_grp, text="Full DataFrame + labels (CSV)",
            variable=content_var, value="full_df_labels_csv"
        ).pack(anchor="w", pady=2, padx=5)
    
        # --- Preview Panel ---
        preview_text = tk.Text(
            preview_frame, 
            font=("Courier", 9), 
            state="disabled",
            wrap=tk.NONE,
            bg="#f8f8f8"
        )
        
        # Add scrollbars to preview
        preview_scroll_y = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=preview_text.yview)
        preview_scroll_x = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=preview_text.xview)
        preview_text.configure(yscrollcommand=preview_scroll_y.set, xscrollcommand=preview_scroll_x.set)
        
        preview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        preview_scroll_y.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        preview_scroll_x.pack(side=tk.BOTTOM, fill=tk.X, padx=(5, 0))
    
        # --- Action Buttons ---
        btns = ttk.Frame(options_frame)
        btns.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
    
        def do_export_and_close():
            # Ask for CSV location
            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save labels CSV as…",
            )
            if not path:
                return
            try:
                # _export_labels_do returns True only when the CSV (and
                # sidecar) were actually written; its early-return paths
                # have already told the user why (grill Q7).
                if self._export_labels_do(path, scope_var.get(), content_var.get()):
                    messagebox.showinfo("Export Complete", f"Exported:\n{path}")
                    dlg.destroy()
            except Exception as e:
                messagebox.showerror("Export Failed", f"{e}")
    
        tk.Button(btns, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT, padx=(5, 0))
        tk.Button(btns, text="Export", command=do_export_and_close).pack(side=tk.RIGHT)
        
        # --- Preview Update Functions ---
        def update_preview():
            """Update preview based on current dialog options."""
            try:
                preview_df, total_estimate, info = self._generate_export_preview(
                    scope_var.get(), 
                    content_var.get(), 
                    limit=10
                )
                preview_content = self._format_dataframe_preview(preview_df, total_estimate, info)
                
                # Update preview text
                preview_text.config(state="normal")
                preview_text.delete(1.0, tk.END)
                preview_text.insert(1.0, preview_content)
                preview_text.config(state="disabled")
                
            except Exception as e:
                # Show error in preview
                error_msg = f"Preview Error:\n{str(e)}\n\nThis might indicate no data matches your selection."
                preview_text.config(state="normal")
                preview_text.delete(1.0, tk.END)
                preview_text.insert(1.0, error_msg)
                preview_text.config(state="disabled")
        
        # Connect option changes to preview updates
        scope_var.trace_add("write", lambda *args: update_preview())
        content_var.trace_add("write", lambda *args: update_preview())
        
        # Initial preview
        dlg.after(100, update_preview)  # Small delay to ensure UI is ready
        
        # Center over parent
        dlg.update_idletasks()
        if self.root is not None:
            rx = self.root.winfo_rootx()
            ry = self.root.winfo_rooty()
            rw = self.root.winfo_width()
            rh = self.root.winfo_height()
            dw = dlg.winfo_width()
            dh = dlg.winfo_height()
            dlg.geometry(f"+{rx + (rw - dw)//2}+{ry + (rh - dh)//2}")

    def _export_labels_do(self, csv_path: str, scope: str, content: str) -> bool:
        """
        Core writer for labels CSV + sidecar label_map.json.
    
        Parameters
        ----------
        csv_path : str
            Destination CSV path (chosen by user).
        scope : {"full","selected"}
            "full"     → all rows included, unlabeled rows get -1.
            "selected" → only rows that fall within labeled intervals (label_id != -1).
        content : {"index_labels_csv","full_df_labels_csv"}
            Index + labels only, or full DF with labels appended.

        Returns True only if the CSV (and sidecar) were written.
        """
        import json
        from pathlib import Path
        import pandas as pd

        # Refuse to export labels the schema does not contain: they
        # would silently collapse to -1 (= unlabeled) in the CSV and be
        # absent from the sidecar -- corrupted training data with no
        # warning (grill Q4; reproduced in the Pack 2 evidence map).
        orphans = sorted({iv.label for iv in self.intervals} - set(self.classes))
        if orphans:
            from tkinter import messagebox
            messagebox.showerror(
                "Export Blocked",
                "These interval labels are not in the current label schema "
                "and would be exported as -1 (unlabeled):\n\n"
                f"  {', '.join(orphans)}\n\n"
                "Fix them via Manage Labels..., then export again.")
            return False

        # Build label_id column
        label_id = self._compute_label_id_series()
    
        if scope == "selected":
            mask = label_id.values != -1
            if not mask.any():
                from tkinter import messagebox
                messagebox.showwarning("No Labeled Samples", "There are no labeled samples in the current data.")
                return False
            idx = self.df.index[mask]
            label_id = label_id.loc[idx]
            df_source = self.df.loc[idx]
        else:
            # full dataset (unlabeled = -1)
            df_source = self.df
    
        # Assemble output frame
        if content == "index_labels_csv":
            out = pd.DataFrame({"label_id": label_id}, index=label_id.index)
            out.index.name = "time"
        else:  # "full_df_labels_csv"
            out = df_source.copy()
            out["label_id"] = label_id.astype(label_id.dtype)
            if out.index.name is None:
                out.index.name = "time"
    
        # Write CSV atomically (complete file or no change, never a
        # valid-looking truncated training set)
        atomic_write_path(csv_path, lambda p: out.to_csv(p))
    
        # Write sidecar mapping, atomically, AFTER the CSV -- so the only
        # possible partial state is "complete CSV, old/missing sidecar",
        # which is reported honestly below.
        # NOTE: Mapping follows the current class ordering for stability.
        label_to_id = {label: i for i, label in enumerate(self.classes)}
        sidecar = Path(csv_path).with_name(Path(csv_path).stem + "_label_map.json")
        try:
            atomic_write_json(sidecar, label_to_id)
        except Exception as e:
            raise RuntimeError(
                f"The labels CSV was written to {csv_path}, but the label "
                f"map sidecar failed: {e}") from e
        return True

    def _generate_export_preview(self, scope: str, content: str, limit: int = 10):
        """
        Generate efficient preview of export data using only first few rows.
        
        Parameters
        ----------
        scope : {"full", "selected"}
            Export scope selection
        content : {"index_labels_csv", "full_df_labels_csv"}
            Content type selection  
        limit : int
            Maximum number of rows to include in preview
            
        Returns
        -------
        tuple
            (preview_df, total_estimate, info_dict)
        """
        import pandas as pd
        
        # Build label_id efficiently for preview
        if scope == "full":
            # Take first `limit` rows from original DataFrame
            preview_input = self.df.head(limit)
            total_estimate = len(self.df)
            info = {
                "scope_desc": "Full dataset",
                "total_unlabeled": "Some rows may be unlabeled (-1)"
            }
        elif scope == "selected":
            # Get first `limit` rows that fall within labeled intervals
            preview_input, total_estimate = self._get_first_labeled_rows(limit)
            if len(preview_input) == 0:
                raise ValueError("No labeled intervals found in current data range")
            info = {
                "scope_desc": "Selected intervals only", 
                "total_unlabeled": "All rows are labeled"
            }
        else:
            raise ValueError(f"Unknown scope: {scope}")
        
        # Generate label_id series for preview data
        label_id = self._compute_label_id_series_for_subset(preview_input)
        
        # Apply content formatting
        if content == "index_labels_csv":
            preview_df = pd.DataFrame({"label_id": label_id}, index=label_id.index)
            preview_df.index.name = "time"
            info["content_desc"] = "Index + labels only"
        elif content == "full_df_labels_csv":
            preview_df = preview_input.copy()
            preview_df["label_id"] = label_id.astype(label_id.dtype)
            if preview_df.index.name is None:
                preview_df.index.name = "time"
            info["content_desc"] = "Full DataFrame + labels"
        else:
            raise ValueError(f"Unknown content type: {content}")
            
        return preview_df, total_estimate, info
    
    def _get_first_labeled_rows(self, limit: int = 10):
        """
        Get first N rows that fall within labeled intervals.
        
        Parameters
        ----------
        limit : int
            Maximum number of rows to return
            
        Returns
        -------
        tuple
            (preview_df, total_labeled_count)
        """
        import pandas as pd
        
        if not self.intervals:
            return pd.DataFrame(), 0
        
        collected_rows = []
        total_labeled_count = 0
        
        # Sort intervals by start time for consistent preview
        sorted_intervals = sorted(self.intervals, key=lambda iv: iv.start)
        
        for interval in sorted_intervals:
            # Calculate total labeled count (for estimation)
            interval_mask = (self.df.index >= interval.start) & (self.df.index <= interval.end)
            interval_size = interval_mask.sum()
            total_labeled_count += interval_size
            
            # For preview, only collect what we need
            if len(collected_rows) < limit:
                interval_df = self.df.loc[interval_mask]
                remaining_needed = limit - len(collected_rows)
                
                if len(interval_df) > 0:
                    rows_to_take = min(remaining_needed, len(interval_df))
                    collected_rows.append(interval_df.iloc[:rows_to_take])
        
        if collected_rows:
            preview_df = pd.concat(collected_rows)
        else:
            preview_df = pd.DataFrame()
        
        return preview_df, total_labeled_count
    
    def _compute_label_id_series_for_subset(self, subset_df):
        """
        Compute label_id series for a subset of data efficiently.
        
        Parameters
        ----------
        subset_df : pd.DataFrame
            Subset of self.df to compute labels for
            
        Returns
        -------
        pd.Series
            Label ID series aligned with subset_df.index
        """
        import pandas as pd
        import numpy as np
        
        # Stable mapping from current classes
        label_to_id = {label: i for i, label in enumerate(self.classes)}
        unknown_id = -1
        
        # Determine dtype
        n = len(label_to_id)
        if n <= np.iinfo(np.int8).max:
            dtype = np.int8
        elif n <= np.iinfo(np.int16).max:
            dtype = np.int16
        else:
            dtype = np.int32
        
        # Initialize with unknown
        ids = np.full(len(subset_df), fill_value=unknown_id, dtype=dtype)
        
        # Apply interval labels
        for i, ts in enumerate(subset_df.index):
            for iv in self.intervals:
                if iv.contains(ts):
                    ids[i] = label_to_id.get(iv.label, unknown_id)
                    break
        
        return pd.Series(ids, index=subset_df.index, name="label_id")
    
    def _format_dataframe_preview(self, preview_df, total_estimate: int, info: dict) -> str:
        """
        Format DataFrame for display in preview text widget.
        
        Parameters
        ----------
        preview_df : pd.DataFrame
            Preview data to format
        total_estimate : int
            Estimated total rows in full export
        info : dict
            Additional information about the export
            
        Returns
        -------
        str
            Formatted text for preview display
        """
        if preview_df.empty:
            return (
                "No data to preview.\n\n"
                "This might occur if:\n"
                "• No intervals are labeled\n"
                "• No data exists in the current time range\n"
                "• Selected intervals don't contain any data points"
            )
        
        lines = []
        
        # Header with summary
        if len(preview_df) < total_estimate:
            lines.append(f"Preview (first {len(preview_df)} of ~{total_estimate:,} rows):")
        else:
            lines.append(f"Preview (all {len(preview_df)} rows):")
        lines.append("")
        
        # Data preview - limit columns for readability
        display_df = preview_df.copy()
        columns_truncated = False
        
        # For wide DataFrames, show only first few columns + label_id
        max_cols = 6
        if len(display_df.columns) > max_cols:
            # Keep label_id if it exists, otherwise just take first max_cols-1
            if 'label_id' in display_df.columns:
                other_cols = [col for col in display_df.columns if col != 'label_id']
                cols_to_show = other_cols[:max_cols-1] + ['label_id']
            else:
                cols_to_show = list(display_df.columns[:max_cols])
            
            display_df = display_df[cols_to_show]
            columns_truncated = True
            lines.append(f"(Showing {len(cols_to_show)} of {len(preview_df.columns)} columns)")
            lines.append("")
        
        # Add visual separator before DataFrame
        lines.append("─" * 60)  # Horizontal line
        
        # Format the DataFrame with manual column truncation indication
        df_str = display_df.to_string(max_rows=20, max_cols=max_cols)
        
        # Add ellipsis to column headers if truncated
        if columns_truncated:
            df_lines = df_str.split('\n')
            if len(df_lines) > 0:
                # Add "..." to the header line
                header_line = df_lines[0]
                if not header_line.endswith('...'):
                    df_lines[0] = header_line + "  ..."
                
                # Add "..." to each data row
                for i in range(1, len(df_lines)):
                    if df_lines[i].strip() and not df_lines[i].endswith('...'):
                        df_lines[i] = df_lines[i] + "  ..."
                
                df_str = '\n'.join(df_lines)
        
        lines.append(df_str)
        
        # Add row continuation indicator if we have more rows
        if len(preview_df) < total_estimate:
            lines.append("...")
            lines.append(f"(+ {total_estimate - len(preview_df):,} more rows)")
        
        # Add visual separator after DataFrame
        lines.append("─" * 60)  # Horizontal line
        lines.append("")
        
        # Summary information
        lines.append("Export Settings:")
        lines.append(f"  Scope: {info['scope_desc']}")
        lines.append(f"  Content: {info['content_desc']}")
        if total_estimate > 1000:
            est_size_mb = total_estimate * len(display_df.columns) * 20 / (1024 * 1024)  # Rough estimate
            lines.append(f"  Estimated file size: ~{est_size_mb:.1f} MB")
        
        # Label mapping preview
        if hasattr(self, 'classes') and self.classes:
            lines.append("")
            lines.append("Label ID Mapping:")
            for i, label in enumerate(self.classes[:8]):  # Show first 8 labels
                lines.append(f"  {i}: {label}")
            if len(self.classes) > 8:
                lines.append(f"  ... and {len(self.classes) - 8} more")
            lines.append(f"  -1: UNLABELED")
        
        return "\n".join(lines)


    def _export_intervals(self) -> None:
        if not self.intervals:
            messagebox.showwarning("No Data", "No intervals to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Parquet files", "*.parquet"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        rows = [
            {"start": iv.start, "end": iv.end, "label": iv.label, "notes": iv.notes}
            for iv in self.intervals
        ]
        df_export = pd.DataFrame(rows)
        try:
            if path.lower().endswith(".parquet"):
                atomic_write_path(path, lambda p: df_export.to_parquet(p, index=False))
            else:
                atomic_write_path(path, lambda p: df_export.to_csv(p, index=False))
        except Exception as e:
            messagebox.showerror(
                "Export Failed",
                f"Could not export intervals:\n{e}\n\n"
                f"The previous file (if any) is unchanged.")
            return
        self.status_var.set(f"Exported to {path}")  # type: ignore[union-attr]
        messagebox.showinfo("Export Complete", f"Intervals exported to {path}")

    def _save_autosave(self) -> None:
        """Atomically save current state (intervals + label schema +
        dataset identity) to the fingerprinted autosave file, keeping
        one .bak generation."""
        from datetime import datetime

        # Use instance autosave file path
        autosave_path = self.autosave_file

        # Calculate statistics
        label_stats = {}
        total_duration = 0
        for interval in self.intervals:
            label = interval.label
            duration_hours = (interval.end - interval.start).total_seconds() / 3600

            if label not in label_stats:
                label_stats[label] = {'count': 0, 'duration_hours': 0}
            label_stats[label]['count'] += 1
            label_stats[label]['duration_hours'] += duration_hours
            total_duration += duration_hours

        # Calculate coverage percentage
        data_duration = (self.data_end - self.data_start).total_seconds() / 3600
        coverage_percent = (total_duration / data_duration * 100) if data_duration > 0 else 0

        # Build autosave data structure
        autosave_data = {
            'metadata': {
                'data_columns': [str(c) for c in self.df.columns],  # For matching validation
                'dtypes': {str(c): str(t) for c, t in self.df.dtypes.items()},
                'n_rows': int(len(self.df.index)),
                'fingerprint': self._dataset_fingerprint(),
                'source_name': getattr(self, 'source_name', None),
                'autosave_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_intervals': len(self.intervals),
                'coverage_percent': round(coverage_percent, 1),
                # tz-NORMALIZED, matching the fingerprint and the load-time
                # comparison, so a naive/UTC-localized pair of the same
                # dataset can never self-flag as a mismatch (fold tz).
                'time_range': {
                    'start': _norm_iso(self.data_start),
                    'end': _norm_iso(self.data_end)
                }
            },
            # Label schema travels with the intervals so recovery can
            # restore it -- without this, recovered labels outside the
            # session's default schema silently export as -1 (grill Q4).
            'classes': list(self.classes),
            'class_colors': dict(self.class_colors),
            'intervals': [iv.to_dict() for iv in self.intervals],  # Convert Interval objects to dicts
            'label_stats': label_stats
        }

        # Atomic write; keep one .bak generation as the last line of
        # defence (a crash mid-write can no longer destroy recovery).
        try:
            atomic_write_json(autosave_path, autosave_data, backup=True)
        except Exception as e:
            if getattr(self, 'status_var', None) is not None:
                self.status_var.set(f"Autosave failed: {e}")

    def _check_autosave(self):
        """
        Look for THIS dataset's autosave (fingerprinted filename) and
        return its parsed contents plus identity annotations, or None.

        Clean break (Pack 2, grill Q2): only the fingerprinted JSON
        name is consulted. Pre-fingerprint files and the old pickle
        format are never read. If the main file is corrupt, the .bak
        generation written by _save_autosave is tried before giving up.
        """
        from chronotagger.core.models import Interval

        main = self.autosave_file
        bak = main.with_name(main.name + ".bak")
        main_existed = main.exists()

        # The .bak is consulted ONLY when the main file exists but is
        # unreadable. If the user deliberately deleted the named main
        # file, the .bak must not resurrect it (fold V3).
        candidates = [main]
        if main_existed:
            candidates.append(bak)

        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                with open(candidate, 'r', encoding='utf-8') as f:
                    autosave_data = json.load(f)
                # Convert interval dicts back to Interval objects
                autosave_data['intervals'] = [
                    Interval.from_dict(d) for d in autosave_data.get('intervals', [])
                ]
            except Exception as e:
                # A corrupt candidate is worth telling the user about --
                # silently pretending no autosave exists converted
                # "recoverable" into "lost" before this pack.
                if getattr(self, 'status_var', None) is not None:
                    self.status_var.set(
                        f"Autosave unreadable ({candidate.name}): {e}")
                continue

            # Identity check against the currently loaded DataFrame.
            # Wrapped in its own try/except: a parseable-but-wrong-SHAPED
            # file must degrade to a warning, never kill the app at
            # launch (fold V2-M: {"metadata": 5} used to raise out of
            # run()). Everything here validates what _save_autosave
            # writes: fingerprint (exact identity, subsumes columns +
            # bounds + count), then human-readable diffs for the dialog.
            warns = []
            try:
                metadata = autosave_data.get('metadata', {})
                if not isinstance(metadata, dict):
                    metadata = {}
                    warns.append("WARNING: autosave metadata is malformed")

                saved_fp = metadata.get('fingerprint')
                if saved_fp and str(saved_fp) != self._dataset_fingerprint():
                    warns.append("WARNING: dataset fingerprint differs from current data")

                saved_columns = [str(c) for c in (metadata.get('data_columns') or [])]
                current_columns = [str(c) for c in self.df.columns]
                if saved_columns and sorted(saved_columns) != sorted(current_columns):
                    from collections import Counter
                    saved_c = Counter(saved_columns)
                    cur_c = Counter(current_columns)
                    missing = sorted((saved_c - cur_c).keys())
                    extra = sorted((cur_c - saved_c).keys())
                    warns.append("WARNING: saved columns differ from current data")
                    if missing:
                        warns.append(f"  only in autosave: {', '.join(missing)}")
                    if extra:
                        warns.append(f"  only in current:  {', '.join(extra)}")
                    if not missing and not extra:
                        warns.append("  (duplicate column name counts differ)")

                n_rows = metadata.get('n_rows')
                if n_rows is not None and int(n_rows) != len(self.df.index):
                    warns.append(
                        f"WARNING: saved row count ({n_rows}) differs from "
                        f"current data ({len(self.df.index)})")

                tr = metadata.get('time_range')
                if not isinstance(tr, dict):
                    tr = {}
                if tr.get('start') and str(tr['start']) != _norm_iso(self.data_start):
                    warns.append("WARNING: saved time range differs from current data")
                elif tr.get('end') and str(tr['end']) != _norm_iso(self.data_end):
                    warns.append("WARNING: saved time range differs from current data")

                # Same-schema, same-window datasets (two spacecraft via a
                # shared loader) share a fingerprint -- the source name is
                # the tiebreaker when both sides know one (evidence 3b).
                saved_src = metadata.get('source_name')
                live_src = getattr(self, 'source_name', None)
                if saved_src and live_src and str(saved_src) != str(live_src):
                    warns.append(
                        "WARNING: autosave came from a different source file")
                    warns.append(f"  autosave: {saved_src}")
                    warns.append(f"  current:  {live_src}")
            except Exception:
                warns.append("WARNING: autosave metadata is malformed")

            lines = list(warns)
            if candidate is not candidates[0]:
                lines.append("NOTE: loaded from the .bak backup copy "
                             "(the main autosave file is corrupt)")
            autosave_data['_identity'] = {'mismatch': bool(warns), 'lines': lines}
            autosave_data['_loaded_path'] = str(candidate)
            return autosave_data

        # Clean break notice (not a fallback): if a pre-fingerprint
        # autosave sits in this folder, say so once instead of silently
        # ignoring what a returning user may believe is their session.
        legacy = self.autosave_folder / "chronotagger_autosave.json"
        if legacy.exists() and getattr(self, 'status_var', None) is not None:
            self.status_var.set(
                "Note: a pre-2.x autosave (chronotagger_autosave.json) "
                "exists here and is no longer read.")
        return None

    def _apply_recovered_autosave(self, autosave_data: dict) -> None:
        """
        Install a recovered autosave: the intervals plus the label
        schema they were made with. Invalidates the undo history and
        selection, validates in strict mode, and marks the session
        modified (recovered work is unsaved work). GUI refresh and pane
        sync are the caller's job.
        """
        self.intervals = list(autosave_data.get('intervals', []))
        saved_classes = autosave_data.get('classes')
        if saved_classes:
            self.classes = list(saved_classes)
            # Only replace colors when the payload actually carries them:
            # a schema without colors must not wipe the live map into
            # all-grey fallbacks (fold V1/V3).
            if 'class_colors' in autosave_data:
                self.class_colors = dict(autosave_data['class_colors'] or {})
            # getattr guards: this method is a named entry point and must
            # not assume the GUI widgets exist yet (fold V2).
            combo = getattr(self, 'class_combo', None)
            var = getattr(self, 'current_class_var', None)
            if combo is not None and var is not None:
                combo["values"] = self.classes
                if var.get() not in self.classes and self.classes:
                    var.set(self.classes[0])
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.selected_interval = None
        if hasattr(self, '_clear_selected_interval_highlights'):
            self._clear_selected_interval_highlights()
        self._check_interval_invariants()
        self.modified = True

    def _show_recovery_dialog(self, autosave_data):
        """
        Show recovery dialog with autosave information.

        Args:
            autosave_data: Dict containing autosave metadata and intervals

        Returns:
            str: User choice - 'recover', 'start_fresh', 'save_backup', or 'cancel'
        """
        import tkinter as tk
        from tkinter import ttk, messagebox
        from datetime import datetime
        import shutil

        # Identity annotations from _check_autosave (may be absent when
        # the dialog is driven directly, e.g. in tests)
        identity = autosave_data.get('_identity', {}) or {}

        # Create modal dialog. The BUTTONS are structurally protected
        # from overflow (packed bottom-first, see below), so the
        # geometry only decides how much info shows before clipping;
        # vertical resize is the escape hatch. Width covers the row
        # content at 150% DPI scaling (recheck: reqwidth 626 > 620).
        dialog = tk.Toplevel(self.root)
        dialog.title("Autosave Found")
        dialog.geometry("640x700")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center on screen
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        # Horizontally fixed for consistent appearance; vertically
        # resizable as the escape hatch against content overflow
        dialog.resizable(False, True)

        # Store result
        result = {'choice': None}

        # Main container with padding
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill='both', expand=True)

        # Reserve the button area FIRST (packed side='bottom' before
        # any content packs with expand=True): pack priority follows
        # pack order, so overflowing info content can never push the
        # buttons off-screen. The buttons themselves are created later,
        # after their callbacks are defined.
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side='bottom', fill='x', pady=(10, 0))

        top_button_frame = ttk.Frame(button_frame)
        top_button_frame.pack(fill='x', pady=(0, 5))

        bottom_button_frame = ttk.Frame(button_frame)
        bottom_button_frame.pack(fill='x')

        # Header with icon and title
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill='x', pady=(0, 15))

        # Title
        title_label = ttk.Label(
            header_frame,
            text=("Autosave Found -- Identity Mismatch"
                  if identity.get('mismatch')
                  else "Autosave Found for This Data File"),
            font=('Segoe UI', 12, 'bold')
        )
        title_label.pack()

        # Info container with light background
        info_frame = ttk.LabelFrame(main_frame, text="Session Information", padding="15")
        info_frame.pack(fill='both', expand=True, pady=(0, 15))

        metadata = autosave_data.get('metadata', {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        label_stats = autosave_data.get('label_stats', {}) or {}
        if not isinstance(label_stats, dict):
            label_stats = {}


        # Autosave file actually loaded (main or .bak); wraplength so a
        # long absolute path wraps instead of clipping at the fixed width
        loaded_path = autosave_data.get('_loaded_path')
        if loaded_path:
            ttk.Label(
                info_frame,
                text=f"Autosave File: {loaded_path}",
                font=('Segoe UI', 9),
                wraplength=560,
                justify='left'
            ).pack(anchor='w', pady=(0, 5))

        # Source dataset, if the wizard recorded one
        source_name = metadata.get('source_name')
        if source_name:
            ttk.Label(
                info_frame,
                text=f"Source Data: {source_name}",
                font=('Segoe UI', 9),
                wraplength=560,
                justify='left'
            ).pack(anchor='w', pady=(0, 5))

        # Dataset fingerprint (matches the 12-hex in the filename)
        saved_fp = metadata.get('fingerprint')
        if saved_fp:
            ttk.Label(
                info_frame,
                text=f"Dataset ID: {saved_fp}",
                font=('Segoe UI', 9)
            ).pack(anchor='w', pady=(0, 5))

        # Autosave date
        date_label = ttk.Label(
            info_frame,
            text=f"Autosave Date: {metadata.get('autosave_timestamp', 'unknown')}",
            font=('Segoe UI', 9)
        )
        date_label.pack(anchor='w', pady=(0, 5))

        # Saved time range (the strongest identity signal on disk)
        tr = metadata.get('time_range')
        if not isinstance(tr, dict):
            tr = {}
        if tr.get('start') and tr.get('end'):
            ttk.Label(
                info_frame,
                text=f"Time Range: {tr['start']}  to  {tr['end']}",
                font=('Segoe UI', 9),
                wraplength=560,
                justify='left'
            ).pack(anchor='w', pady=(0, 5))

        # Coverage
        coverage_label = ttk.Label(
            info_frame,
            text=f"Coverage: {metadata.get('coverage_percent', '?')}% of time range labeled",
            font=('Segoe UI', 9)
        )
        coverage_label.pack(anchor='w', pady=(0, 10))

        # Identity warnings (fingerprint / column diff / time-range /
        # source mismatch / loaded from .bak) -- rendered, not print()ed
        # to a console nobody sees
        if identity.get('lines'):
            for line in identity['lines']:
                ttk.Label(
                    info_frame,
                    text=line,
                    font=('Segoe UI', 9, 'bold'),
                    foreground='#8b2e2e',
                    wraplength=560,
                    justify='left'
                ).pack(anchor='w')
            if identity.get('mismatch'):
                ttk.Label(
                    info_frame,
                    text="Recovering into a different dataset is NOT recommended.",
                    font=('Segoe UI', 9, 'bold'),
                    foreground='#8b2e2e'
                ).pack(anchor='w', pady=(0, 8))

        # Separator
        ttk.Separator(info_frame, orient='horizontal').pack(fill='x', pady=(0, 10))

        # Intervals by label header
        intervals_header = ttk.Label(
            info_frame,
            text="Intervals by Label:",
            font=('Segoe UI', 9, 'bold')
        )
        intervals_header.pack(anchor='w', pady=(0, 5))

        # Create frame for interval list with padding
        intervals_container = ttk.Frame(info_frame)
        intervals_container.pack(fill='both', expand=True, pady=(0, 10))

        # Display each label's stats
        shown = 0
        for label, stats in sorted(label_stats.items()):
            if not isinstance(stats, dict):
                continue
            if shown >= 8:
                ttk.Label(
                    intervals_container,
                    text=f"  ... and {len(label_stats) - shown} more label(s)",
                    font=('Segoe UI', 9),
                    foreground='#666666'
                ).pack(anchor='w', pady=2)
                break
            shown += 1
            count = stats.get('count', 0)
            hours = stats.get('duration_hours', 0)

            interval_frame = ttk.Frame(intervals_container)
            interval_frame.pack(fill='x', pady=2)

            # Label name (left-aligned)
            label_text = ttk.Label(
                interval_frame,
                text=f"  {label}:",
                font=('Segoe UI', 9),
                width=20,
                anchor='w'
            )
            label_text.pack(side='left')

            # Stats (right side)
            stats_text = ttk.Label(
                interval_frame,
                text=f"{count} intervals ({hours:.1f} hours)",
                font=('Segoe UI', 9),
                foreground='#666666'
            )
            stats_text.pack(side='left')

        # Separator
        ttk.Separator(info_frame, orient='horizontal').pack(fill='x', pady=(5, 10))

        # Total summary (tolerant reads: a missing key must not kill the
        # app at launch)
        total_intervals = metadata.get('total_intervals',
                                       len(autosave_data.get('intervals', [])))
        total_hours = sum(s.get('duration_hours', 0) for s in label_stats.values())

        total_label = ttk.Label(
            info_frame,
            text=f"Total: {total_intervals} intervals covering {total_hours:.1f} hours",
            font=('Segoe UI', 9, 'bold')
        )
        total_label.pack(anchor='w')

        # Button callbacks
        def on_recover():
            result['choice'] = 'recover'
            dialog.destroy()

        def on_start_fresh():
            confirm = messagebox.askyesno(
                "Confirm Start Fresh",
                f"Are you sure you want to start fresh?\n\n"
                f"This will NOT load the autosave with {total_intervals} intervals.\n"
                f"The autosave file will remain and can be recovered later.",
                parent=dialog
            )
            if confirm:
                result['choice'] = 'start_fresh'
                dialog.destroy()

        def on_save_backup():
            # Copy the file that was ACTUALLY loaded (main or .bak), and
            # never overwrite an existing backup from the same second.
            src = autosave_data.get('_loaded_path') or str(self.autosave_file)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.autosave_folder / f'chronotagger_autosave_backup_{timestamp}.json'
            n = 1
            while backup_file.exists():
                backup_file = self.autosave_folder / f'chronotagger_autosave_backup_{timestamp}_{n}.json'
                n += 1

            try:
                shutil.copy(src, backup_file)
                messagebox.showinfo(
                    "Backup Saved",
                    f"Autosave backed up to:\n{backup_file.name}",
                    parent=dialog
                )
                result['choice'] = 'save_backup'
                dialog.destroy()
            except Exception as e:
                messagebox.showerror(
                    "Backup Failed",
                    f"Could not save backup:\n{e}",
                    parent=dialog
                )

        def on_cancel():
            result['choice'] = 'cancel'
            dialog.destroy()

        # Buttons go into the frames reserved at the top of this method
        # (042l): bottom-packed first, so they always stay on screen.

        # Top row buttons
        recover_btn = tk.Button(
            top_button_frame,
            text="Recover Session",
            command=on_recover,
            width=25
        )
        recover_btn.pack(side='left', expand=True, padx=(0, 5))

        fresh_btn = tk.Button(
            top_button_frame,
            text="Start Fresh",
            command=on_start_fresh,
            width=25
        )
        fresh_btn.pack(side='left', expand=True, padx=(5, 0))

        # Bottom row buttons
        backup_btn = tk.Button(
            bottom_button_frame,
            text="Save & Start Fresh",
            command=on_save_backup,
            width=25
        )
        backup_btn.pack(side='left', expand=True, padx=(0, 5))

        cancel_btn = tk.Button(
            bottom_button_frame,
            text="Exit ChronoTagger",
            command=on_cancel,
            width=25
        )
        cancel_btn.pack(side='left', expand=True, padx=(5, 0))

        # Closing the dialog with the window X means cancel (exit), the
        # same as Escape -- never the silent no-branch limbo it was (D3)
        dialog.protocol("WM_DELETE_WINDOW", on_cancel)

        # Default button: Recover -- unless the identity check flagged a
        # mismatch, in which case Start Fresh is the safe default.
        if identity.get('mismatch'):
            fresh_btn.focus_set()
            dialog.bind('<Return>', lambda e: on_start_fresh())
        else:
            recover_btn.focus_set()
            dialog.bind('<Return>', lambda e: on_recover())
        dialog.bind('<Escape>', lambda e: on_cancel())

        # Wait for dialog to close
        dialog.wait_window()

        return result['choice']

    def _on_closing(self) -> None:
        if self.modified:
            resp = messagebox.askyesnocancel("Save Changes?", "Save before closing?")
            if resp is None:
                return
            elif resp:
                # _save_session returns True only if the file was
                # actually written. On cancel/failure, offer the
                # close-anyway choice Q7 ruled -- never close silently
                # on a failed save, never trap the user either.
                if not self._save_session():
                    if not messagebox.askyesno(
                            "Close Anyway?",
                            "The session was not saved. Close anyway and "
                            "discard unsaved changes?"):
                        return
        self.root.destroy()  # type: ignore[union-attr]
    
    def _layouts_compatible(self, layout1: dict, layout2: dict) -> bool:
        """
        Check if two layout_specs are compatible.
        
        Args:
            layout1: First layout specification
            layout2: Second layout specification
            
        Returns:
            True if layouts are structurally compatible, False otherwise
        """
        if layout1 is None or layout2 is None:
            return True  # Can't validate, allow load
        
        # Compare key structural elements
        nrows_match = layout1.get("nrows") == layout2.get("nrows")
        ncols_match = layout1.get("ncols") == layout2.get("ncols")
        areas_match = len(layout1.get("areas", [])) == len(layout2.get("areas", []))
        
        return nrows_match and ncols_match and areas_match
    
    def _describe_layout(self, layout: dict) -> str:
        """
        Create human-readable layout description.
        
        Args:
            layout: Layout specification dictionary
            
        Returns:
            String describing the layout (e.g., "3 rows × 2 columns (4 panels)")
        """
        if layout is None:
            return "Unknown layout"
        
        nrows = layout.get("nrows", "?")
        ncols = layout.get("ncols", "?")
        n_panels = len(layout.get("areas", []))
        
        return f"{nrows} rows × {ncols} columns ({n_panels} panels)"
