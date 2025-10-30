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
