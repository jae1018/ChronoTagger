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
        self._time_range_dirty = True  # Reset axis zooms
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
        self._time_range_dirty = True  # Reset axis zooms
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
    
    def _double_time_window(self) -> None:
        """
        Double the time window size, centering on current window.
        If bounds are hit, adjust both ends while preserving the new window size.
        """
        current_window = self.t1 - self.t0
        new_window = current_window * 2
        
        # Clamp to max window
        new_window = min(new_window, self.max_window)
        
        # Try to center on current window
        center = self.t0 + current_window / 2
        half = new_window / 2
        
        new_t0 = center - half
        new_t1 = center + half
        
        # Clamp to data bounds while preserving window size
        if new_t0 < self.data_start:
            new_t0 = self.data_start
            new_t1 = new_t0 + new_window
        if new_t1 > self.data_end:
            new_t1 = self.data_end
            new_t0 = new_t1 - new_window
        
        # Apply
        self.t0, self.t1 = new_t0, new_t1
        self.window = self.t1 - self.t0
        self._time_range_dirty = True
        self._sync_entries_and_plot()
    
    def _halve_time_window(self) -> None:
        """
        Halve the time window size, centering on current window.
        If bounds are hit, adjust both ends while preserving the new window size.
        """
        current_window = self.t1 - self.t0
        new_window = current_window / 2

        # Clamp to min window
        new_window = max(new_window, self.min_window)

        # Try to center on current window
        center = self.t0 + current_window / 2
        half = new_window / 2

        new_t0 = center - half
        new_t1 = center + half

        # Clamp to data bounds while preserving window size
        if new_t0 < self.data_start:
            new_t0 = self.data_start
            new_t1 = new_t0 + new_window
        if new_t1 > self.data_end:
            new_t1 = self.data_end
            new_t0 = new_t1 - new_window

        # Apply
        self.t0, self.t1 = new_t0, new_t1
        self.window = self.t1 - self.t0
        self._time_range_dirty = True
        self._sync_entries_and_plot()

    def _on_zoom_box_complete(self, eclick, erelease, pane) -> None:
        """
        Handle right-click drag zoom on time axes.

        Distinguishes between:
        - Short drag (< 10 pixels) = Cancel selection (existing behavior)
        - Long drag (>= 10 pixels) = Zoom to time range

        Args:
            eclick: Mouse click event at selection start
            erelease: Mouse release event at selection end
            pane: The TabPane where the zoom occurred
        """
        # Only process on active pane
        if pane is not self.active_pane:
            return

        # Check if this was a click vs drag (pixel-based threshold)
        dx_pixels = abs(erelease.x - eclick.x)

        if dx_pixels < 10:
            # Too small - treat as cancel click (existing behavior)
            if hasattr(self, '_cancel_active_selection'):
                self._cancel_active_selection()
            return

        # Extract time bounds from drag
        if eclick.xdata is None or erelease.xdata is None:
            return

        x0, x1 = float(eclick.xdata), float(erelease.xdata)
        x_lo, x_hi = sorted([x0, x1])

        # Convert matplotlib dates to timestamps
        import matplotlib.dates as mdates
        t_start = pd.Timestamp(mdates.num2date(x_lo)).tz_localize(None)
        t_end = pd.Timestamp(mdates.num2date(x_hi)).tz_localize(None)

        # Clamp to data bounds
        t_start = max(t_start, self.data_start)
        t_end = min(t_end, self.data_end)

        # Ensure start < end
        if t_start >= t_end:
            return  # Invalid range, ignore

        # Update time window
        self.t0 = t_start
        self.t1 = t_end
        self.window = self.t1 - self.t0
        self._time_range_dirty = True

        # Update the time range UI fields to reflect new window
        self._update_time_range_fields()

        # Redraw plot with new time range
        self._update_plot()

        # Update status
        if hasattr(self, 'status_var') and self.status_var is not None:
            self.status_var.set(
                f"Zoomed to: {t_start.strftime('%H:%M:%S')} → {t_end.strftime('%H:%M:%S')}"
            )

    def _update_time_range_fields(self) -> None:
        """Update Start/End text fields after programmatic time change."""
        if hasattr(self, 'start_time_entry') and self.start_time_entry is not None:
            self.start_time_entry.delete(0, tk.END)
            self.start_time_entry.insert(0, str(self.t0))
        if hasattr(self, 'end_time_entry') and self.end_time_entry is not None:
            self.end_time_entry.delete(0, tk.END)
            self.end_time_entry.insert(0, str(self.t1))
