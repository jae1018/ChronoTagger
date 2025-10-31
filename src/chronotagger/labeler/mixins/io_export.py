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
        target = Path(path) if path else self.autosave_path
        if target is None:
            chosen = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            if not chosen:
                return
            target = Path(chosen)
            self.autosave_path = target

        data = {
            "version": 1,
            "classes": self.classes,
            "class_colors": self.class_colors,
            "window": str(self.window),
            "step": str(self.step),
            "data_start": self.data_start.isoformat(),
            "data_end": self.data_end.isoformat(),
            "intervals": [iv.to_dict() for iv in self.intervals],
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

        self.classes = list(data["classes"])
        self.class_colors = dict(data["class_colors"])
        self.window = pd.Timedelta(data["window"])
        self.step = pd.Timedelta(data["step"])
        self.intervals = [Interval.from_dict(d) for d in data["intervals"]]

        self.autosave_path = Path(path)
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
        Small modal that lets the user choose:
          - Scope: Full dataset  |  Selected intervals only
          - Content: Index + labels (CSV)  |  Full DF + labels (CSV)
    
        Writes a CSV plus a sidecar '<chosen_name>_label_map.json'.
        """
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
    
        if self.df is None or len(self.df.index) == 0:
            messagebox.showwarning("No Data", "There is no data to export.")
            return
    
        # Modal container
        dlg = tk.Toplevel(self.root)
        dlg.title("Export Labels")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)
        pad = {"padx": 10, "pady": 8}
    
        # --- Scope ---
        scope_var = tk.StringVar(value="full")
        scope_grp = ttk.LabelFrame(dlg, text="Scope")
        scope_grp.grid(row=0, column=0, sticky="ew", **pad)
        ttk.Radiobutton(scope_grp, text="Full dataset (unlabeled = -1)",
                        variable=scope_var, value="full").pack(anchor="w", pady=2)
        ttk.Radiobutton(scope_grp, text="Selected intervals only",
                        variable=scope_var, value="selected").pack(anchor="w", pady=2)
    
        # --- Content ---
        content_var = tk.StringVar(value="index_labels_csv")
        content_grp = ttk.LabelFrame(dlg, text="Content")
        content_grp.grid(row=1, column=0, sticky="ew", **pad)
        ttk.Radiobutton(content_grp, text="Index + labels (CSV)",
                        variable=content_var, value="index_labels_csv").pack(anchor="w", pady=2)
        ttk.Radiobutton(content_grp, text="Full DataFrame + labels (CSV)",
                        variable=content_var, value="full_df_labels_csv").pack(anchor="w", pady=2)
    
        # --- Actions ---
        btns = ttk.Frame(dlg)
        btns.grid(row=2, column=0, sticky="e", **pad)
    
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
    
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT, padx=6)
        ttk.Button(btns, text="Export", command=do_export_and_close).pack(side=tk.RIGHT)
    
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

    def _maybe_autosave(self) -> None:
        if self.autosave_path and self.modified:
            self._save_session(str(self.autosave_path))

    def _on_closing(self) -> None:
        if self.modified:
            resp = messagebox.askyesnocancel("Save Changes?", "Save before closing?")
            if resp is None:
                return
            elif resp:
                self._save_session()
        self.root.destroy()  # type: ignore[union-attr]
