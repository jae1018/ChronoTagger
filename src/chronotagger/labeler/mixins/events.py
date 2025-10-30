"""
Event handlers mixin.

Responsibilities:
- Keyboard shortcuts
- Rectangle selection callback
- Tree selection
- Strip click (pick_event)
- Time range entry updates
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
import matplotlib.dates as mdates
import pandas as pd


class EventsMixin:
    def _update_time_window(self) -> None:
        try:
            new_t0 = pd.to_datetime(self.start_time_entry.get())  # type: ignore[union-attr]
            new_t1 = pd.to_datetime(self.end_time_entry.get())    # type: ignore[union-attr]
            if new_t0 >= new_t1:
                messagebox.showerror("Invalid Range", "Start time must be before end time.")
                return
            self.t0 = max(new_t0, self.data_start)
            self.t1 = min(new_t1, self.data_end)

            self.start_time_entry.delete(0, tk.END)  # type: ignore[union-attr]
            self.start_time_entry.insert(0, str(self.t0))  # type: ignore[union-attr]
            self.end_time_entry.delete(0, tk.END)  # type: ignore[union-attr]
            self.end_time_entry.insert(0, str(self.t1))  # type: ignore[union-attr]

            self._update_plot()
            self.status_var.set(  # type: ignore[union-attr]
                f"Window updated: {self.t0.strftime('%H:%M:%S')} → {self.t1.strftime('%H:%M:%S')}"
            )
        except Exception as e:
            messagebox.showerror("Invalid Time Format", f"Could not parse time: {e}")

    def _on_interval_tree_select(self, _event) -> None:
        sel = self.intervals_tree.selection()  # type: ignore[union-attr]
        if not sel:
            self.selected_interval = None
            return
        item = sel[0]
        try:
            idx = int(self.intervals_tree.item(item)["text"]) - 1  # type: ignore[union-attr]
            if 0 <= idx < len(self.intervals):
                self.selected_interval = self.intervals[idx]
                iv = self.selected_interval
                self.status_var.set(  # type: ignore[union-attr]
                    f"Selected: {iv.label} [{iv.start.strftime('%H:%M:%S')} → {iv.end.strftime('%H:%M:%S')}]"
                )
                self._update_strip()
                self.canvas.draw()  # type: ignore[union-attr]
        except Exception:
            self.selected_interval = None

    def _on_rectangle_select(self, eclick, erelease) -> None:
        # Guard: selection outside data area yields None
        if eclick.xdata is None or erelease.xdata is None:
            return
        x1, x2 = sorted([eclick.xdata, erelease.xdata])

        def _to_naive_ts(x: float) -> pd.Timestamp:
            dt = mdates.num2date(x)
            if getattr(dt, "tzinfo", None) is not None:
                dt = dt.replace(tzinfo=None)
            return pd.Timestamp(dt)

        t_start, t_end = _to_naive_ts(x1), _to_naive_ts(x2)

        if self.snap_var.get():  # type: ignore[union-attr]
            t_start, t_end = self._snap_to_samples(t_start, t_end)

        self.current_selection = (t_start, t_end)
        self.status_var.set(  # type: ignore[union-attr]
            f"Selected: {t_start.strftime('%H:%M:%S')} → {t_end.strftime('%H:%M:%S')}"
        )
        self._update_strip()
        self.canvas.draw()  # type: ignore[union-attr]

    def _on_strip_click(self, event) -> None:
        if event.artist not in self.strip_ax.patches:  # type: ignore[union-attr]
            return
        if event.mouseevent.xdata is None:
            return
        dt = mdates.num2date(event.mouseevent.xdata)
        if getattr(dt, "tzinfo", None) is not None:
            dt = dt.replace(tzinfo=None)
        click_ts = pd.Timestamp(dt)

        for iv in self.intervals:
            if iv.contains(click_ts):
                self.selected_interval = iv
                self.status_var.set(  # type: ignore[union-attr]
                    f"Selected: {iv.label} [{iv.start.strftime('%H:%M:%S')} → {iv.end.strftime('%H:%M:%S')}]"
                )
                self._update_strip()
                self.canvas.draw()  # type: ignore[union-attr]
                break

    def _on_key_press(self, event) -> None:
        key = event.keysym

        # Class selection 1..9
        if key.isdigit() and int(key) > 0:
            idx = int(key) - 1
            if idx < len(self.classes):
                self.current_class_var.set(self.classes[idx])  # type: ignore[union-attr]
                self.status_var.set(f"Selected class: {self.classes[idx]}")  # type: ignore[union-attr]

        # Navigation
        elif key in ("n", "N", "Right"):
            self._next_window()
        elif key in ("p", "P", "Left"):
            self._prev_window()

        # Actions
        elif key in ("a", "A", "Return"):
            self._add_interval()
        elif key in ("d", "D", "Delete"):
            self._delete_interval()
        elif key in ("u", "U"):
            if "UNKNOWN" in self.classes:
                self.current_class_var.set("UNKNOWN")  # type: ignore[union-attr]
                self.status_var.set("Selected class: UNKNOWN")  # type: ignore[union-attr]

        # Save / export (Ctrl+S / Ctrl+E)
        elif key in ("s", "S") and event.state & 0x4:
            self._save_session()
        elif key in ("e", "E") and event.state & 0x4:
            self._export_intervals()

        # Undo / Redo (Ctrl+Z / Ctrl+Y) + Backspace ergonomics
        elif (key == "z" and event.state & 0x4) or key == "BackSpace":
            self._undo()
        elif (key == "y" and event.state & 0x4) or (key == "BackSpace" and event.state & 0x1):
            self._redo()
