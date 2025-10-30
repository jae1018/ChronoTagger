"""
Navigation / range update mixin.
"""

from __future__ import annotations
import tkinter as tk
import pandas as pd


class NavigationMixin:
    def go_to_window(self, t0: pd.Timestamp) -> None:
        """Programmatically jump to a specific start time."""
        self.t0 = max(t0, self.data_start)
        self.t1 = min(self.t0 + self.window, self.data_end)
        if self.start_time_entry and self.end_time_entry:
            self.start_time_entry.delete(0, tk.END)
            self.start_time_entry.insert(0, str(self.t0))
            self.end_time_entry.delete(0, tk.END)
            self.end_time_entry.insert(0, str(self.t1))
        self._update_plot()

    def _prev_window(self) -> None:
        try:
            self.step = pd.Timedelta(self.step_entry.get())  # type: ignore[union-attr]
        except Exception:
            pass
        self.window = self.t1 - self.t0
        self.t0 = max(self.data_start, self.t0 - self.step)
        self.t1 = min(self.t0 + self.window, self.data_end)
        self._sync_entries_and_plot()

    def _next_window(self) -> None:
        try:
            self.step = pd.Timedelta(self.step_entry.get())  # type: ignore[union-attr]
        except Exception:
            pass
        self.window = self.t1 - self.t0
        self.t0 = self.t0 + self.step
        self.t1 = self.t0 + self.window
        if self.t1 > self.data_end:
            self.t1 = self.data_end
            self.t0 = max(self.t1 - self.window, self.data_start)
        self._sync_entries_and_plot()

    def _sync_entries_and_plot(self) -> None:
        self.start_time_entry.delete(0, tk.END)  # type: ignore[union-attr]
        self.start_time_entry.insert(0, str(self.t0))  # type: ignore[union-attr]
        self.end_time_entry.delete(0, tk.END)  # type: ignore[union-attr]
        self.end_time_entry.insert(0, str(self.t1))  # type: ignore[union-attr]
        self._update_plot()
        self.status_var.set(  # type: ignore[union-attr]
            f"Window: {self.t0.strftime('%H:%M:%S')} → {self.t1.strftime('%H:%M:%S')}"
        )
