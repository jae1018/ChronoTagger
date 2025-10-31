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
        
    # ---- Step helpers ----
    def _apply_step_entry(self) -> None:
        """Parse the step entry and store to self.step; normalize the entry text."""
        td = self._parse_step_entry()
        self._set_step(td)

    def _parse_step_entry(self) -> pd.Timedelta:
        """Return a parsed timedelta from the step entry, falling back to current self.step."""
        try:
            txt = self.step_entry.get()  # type: ignore[union-attr]
            # Accept pandas Timedelta strings, e.g., '30min', '1h', '00:30:00', '0 days 00:30:00'
            td = pd.to_timedelta(txt)
            # Guard against zero/negative
            if td <= pd.Timedelta(0):
                raise ValueError("Step must be > 0")
            return td
        except Exception:
            # Revert entry to current known-good step
            self.step_entry.delete(0, tk.END)  # type: ignore[union-attr]
            self.step_entry.insert(0, str(self.step))  # type: ignore[union-attr]
            return self.step

    def _set_step(self, td: pd.Timedelta) -> None:
        """Set self.step and normalize entry text."""
        # Clip to [1s, max_window]; max_window defined in app.__init__
        min_step = pd.Timedelta(seconds=1)
        clipped = max(min_step, min(td, getattr(self, "max_window", td)))
        self.step = clipped
        # Normalize the text; keep default pandas formatting
        self.step_entry.delete(0, tk.END)  # type: ignore[union-attr]
        self.step_entry.insert(0, str(self.step))  # type: ignore[union-attr]
        if self.status_var is not None:
            self.status_var.set(f"Step: {self.step}")  # type: ignore[union-attr]

    def _double_step(self) -> None:
        """Double the step duration."""
        base = self._parse_step_entry()
        self._set_step(base * 2)

    def _halve_step(self) -> None:
        """Halve the step duration."""
        base = self._parse_step_entry()
        # Avoid sub-second steps
        self._set_step(base / 2)
