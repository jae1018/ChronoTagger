"""
Session I/O and export mixin.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List

import json
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np

from chronotagger.core.models import Interval


class IOExportMixin:
    # ---- Public convenience wrappers ----
    def save(self, path: Optional[str] = None) -> None:
        self._save_session(path)

    def load(self, path: str) -> None:
        self._load_session(path)

    def export_intervals(self, path: str, fmt: str = "parquet") -> None:
        if not self.intervals:
            print("No intervals to export.")
            return
        rows = [
            {"start": iv.start, "end": iv.end, "label": iv.label, "notes": iv.notes}
            for iv in self.intervals
        ]
        df_export = pd.DataFrame(rows)
        if fmt.lower() == "parquet":
            df_export.to_parquet(path, index=False)
        else:
            df_export.to_csv(path, index=False)
        print(f"Exported intervals to {path}")

    def export_per_sample(
        self, path: str, fmt: str = "parquet", label_on_uncovered: Optional[str] = "UNKNOWN"
    ) -> None:
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
            df_export.to_parquet(path)
        else:
            df_export.to_csv(path)
        print(f"Exported per-sample labels to {path}")

    # ---- GUI-connected ops ----
    def _save_session(self, path: Optional[str] = None) -> None:
        target = Path(path) if path else None
        if target is None:
            chosen = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            if not chosen:
                return
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
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        self.modified = False
        self.status_var.set(f"Saved to {target}")  # type: ignore[union-attr]

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
                self._export_labels_do(path, scope_var.get(), content_var.get())
                messagebox.showinfo("Export Complete", f"Exported:\n{path}")
                dlg.destroy()
            except Exception as e:
                messagebox.showerror("Export Failed", f"{e}")
    
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(btns, text="Export", command=do_export_and_close).pack(side=tk.RIGHT)
        
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

    def _export_labels_do(self, csv_path: str, scope: str, content: str) -> None:
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
        """
        import json
        from pathlib import Path
        import pandas as pd
    
        # Build label_id column
        label_id = self._compute_label_id_series()
    
        if scope == "selected":
            mask = label_id.values != -1
            if not mask.any():
                from tkinter import messagebox
                messagebox.showwarning("No Labeled Samples", "There are no labeled samples in the current data.")
                return
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
    
        # Write CSV
        out.to_csv(csv_path)
    
        # Write sidecar mapping
        # NOTE: Mapping follows the current class ordering for stability.
        label_to_id = {label: i for i, label in enumerate(self.classes)}
        sidecar = Path(csv_path).with_name(Path(csv_path).stem + "_label_map.json")
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(label_to_id, f, indent=2)

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
        if path.endswith(".parquet"):
            df_export.to_parquet(path, index=False)
        else:
            df_export.to_csv(path, index=False)
        self.status_var.set(f"Exported to {path}")  # type: ignore[union-attr]
        messagebox.showinfo("Export Complete", f"Intervals exported to {path}")

    def _export_per_sample(self) -> None:
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

        labels: List[Optional[str]] = []
        for ts in self.df.index:
            lbl = None
            for iv in self.intervals:
                if iv.contains(ts):
                    lbl = iv.label
                    break
            labels.append(lbl if lbl is not None else "UNKNOWN")
        df_export = pd.DataFrame({"label": labels}, index=self.df.index)

        if path.endswith(".parquet"):
            df_export.to_parquet(path)
        else:
            df_export.to_csv(path)

        self.status_var.set(f"Exported to {path}")  # type: ignore[union-attr]
        messagebox.showinfo("Export Complete", f"Per-sample labels exported to {path}")

    def _save_autosave(self) -> None:
        """Save current state to autosave file with metadata."""
        import pickle
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
                'data_columns': list(self.df.columns),  # For matching validation
                'autosave_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_intervals': len(self.intervals),
                'coverage_percent': round(coverage_percent, 1),
                'time_range': {
                    'start': str(self.data_start),
                    'end': str(self.data_end)
                }
            },
            'intervals': self.intervals,  # List of Interval objects
            'label_stats': label_stats
        }

        # Save to file
        try:
            with open(autosave_path, 'wb') as f:
                pickle.dump(autosave_data, f)
        except Exception as e:
            print(f"Warning: Could not save autosave: {e}")

    def _check_autosave(self):
        """
        Check if autosave exists and matches current data.

        Returns:
            dict or None: Autosave data if exists and matches, None otherwise
        """
        import pickle

        # Use instance autosave file path
        if not self.autosave_file.exists():
            return None

        try:
            with open(self.autosave_file, 'rb') as f:
                autosave_data = pickle.load(f)

            # Validate: Check if data columns match
            saved_columns = autosave_data['metadata'].get('data_columns', [])
            current_columns = list(self.df.columns)

            if saved_columns != current_columns:
                print(f"Warning: Autosave columns don't match current data")
                print(f"  Saved: {saved_columns}")
                print(f"  Current: {current_columns}")
                # Still return it, let user decide in dialog

            return autosave_data

        except Exception as e:
            print(f"Warning: Could not load autosave: {e}")
            return None

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

        # Create modal dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Autosave Found")
        dialog.geometry("500x450")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center on screen
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        # Store result
        result = {'choice': None}

        # Header
        header = ttk.Label(dialog, text="Autosave Found for This Data File",
                           font=('Arial', 12, 'bold'))
        header.pack(pady=(15, 10))

        # Info frame
        info_frame = tk.Frame(dialog, bg='#f0f0f0', relief='sunken', bd=1)
        info_frame.pack(pady=10, padx=15, fill='both', expand=False)

        metadata = autosave_data['metadata']
        label_stats = autosave_data['label_stats']

        # Data file (use autosave folder name)
        data_file = self.autosave_folder.name if self.autosave_folder.name != '.' else 'Current directory'
        info_text = f"Autosave Folder: {data_file}\n"
        info_text += f"Autosave Date: {metadata['autosave_timestamp']}\n"
        info_text += f"Coverage: {metadata['coverage_percent']}% of time range labeled\n\n"
        info_text += "Intervals by Label:\n"

        for label, stats in sorted(label_stats.items()):
            count = stats['count']
            hours = stats['duration_hours']
            info_text += f"  {label}: {count} intervals ({hours:.1f} hours)\n"

        total_intervals = metadata['total_intervals']
        total_hours = sum(s['duration_hours'] for s in label_stats.values())
        info_text += f"\nTotal: {total_intervals} intervals covering {total_hours:.1f} hours"

        info_label = tk.Label(info_frame, text=info_text, justify='left',
                              bg='#f0f0f0', font=('Courier', 9), padx=10, pady=10)
        info_label.pack()

        # Button callbacks
        def on_recover():
            result['choice'] = 'recover'
            dialog.destroy()

        def on_start_fresh():
            # Confirm before discarding
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
            # Save backup with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f'chronotagger_autosave_backup_{timestamp}.pkl'
            backup_file = self.autosave_folder / backup_filename

            try:
                shutil.copy(self.autosave_file, backup_file)

                messagebox.showinfo("Backup Saved",
                                   f"Autosave backed up to:\n{backup_filename}",
                                   parent=dialog)
                result['choice'] = 'start_fresh'
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Backup Failed",
                                    f"Could not save backup:\n{e}",
                                    parent=dialog)

        def on_cancel():
            result['choice'] = 'cancel'
            dialog.destroy()

        # Buttons frame
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=15)

        # Buttons
        recover_btn = ttk.Button(btn_frame, text="Recover Session", command=on_recover, width=18)
        recover_btn.grid(row=0, column=0, padx=5, pady=5)

        fresh_btn = ttk.Button(btn_frame, text="Start Fresh", command=on_start_fresh, width=18)
        fresh_btn.grid(row=0, column=1, padx=5, pady=5)

        backup_btn = ttk.Button(btn_frame, text="Save & Start Fresh", command=on_save_backup, width=18)
        backup_btn.grid(row=1, column=0, padx=5, pady=5)

        cancel_btn = ttk.Button(btn_frame, text="Cancel", command=on_cancel, width=18)
        cancel_btn.grid(row=1, column=1, padx=5, pady=5)

        # Make Recover button default (highlighted)
        recover_btn.focus_set()

        # Bind Escape to cancel
        dialog.bind('<Escape>', lambda e: on_cancel())

        # Bind Enter to recover (default action)
        dialog.bind('<Return>', lambda e: on_recover())

        # Wait for dialog to close
        dialog.wait_window()

        return result['choice']

    def _maybe_autosave(self) -> None:
        """Legacy method for JSON session autosave. Now handled by _save_autosave()."""
        # This method is deprecated - automatic saves now use _save_autosave()
        # which saves to pickle format after every interval modification.
        # Keeping this as a no-op for backward compatibility.
        pass

    def _on_closing(self) -> None:
        if self.modified:
            resp = messagebox.askyesnocancel("Save Changes?", "Save before closing?")
            if resp is None:
                return
            elif resp:
                self._save_session()
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
