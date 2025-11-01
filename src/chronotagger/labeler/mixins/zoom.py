# src/chronotagger/labeler/mixins/zoom.py
"""
Mouse-wheel zoom/pan mixin.

Usage:
- Wheel = zoom around cursor (20% per notch).
- Shift + Wheel = pan left/right (20% of window per notch).

Configuration (set in app.__init__):
- self.zoom_sensitivity: float (fraction per wheel step, default 0.2)
- self.pan_sensitivity: float (fraction of window per wheel step, default 0.2)
- self.min_window: pd.Timedelta
- self.max_window: pd.Timedelta
"""

from __future__ import annotations
from typing import Optional

import pandas as pd
import matplotlib.dates as mdates


class ZoomMixin:
    # ---------- helpers ----------

    def _ts_from_mpl_x(self, x: Optional[float]) -> pd.Timestamp:
        """Convert Matplotlib float-date to naive Timestamp; fallback to window center."""
        if x is None:
            return self.t0 + (self.t1 - self.t0) / 2
        dt = mdates.num2date(x)
        if getattr(dt, "tzinfo", None) is not None:
            dt = dt.replace(tzinfo=None)
        return pd.Timestamp(dt)

    def _clamp_to_bounds(self, t0: pd.Timestamp, t1: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Clamp proposed [t0, t1] to data bounds while preserving window length."""
        win = t1 - t0
        if t0 < self.data_start:
            t0 = self.data_start
            t1 = t0 + win
        if t1 > self.data_end:
            t1 = self.data_end
            t0 = t1 - win
        return t0, t1

    # ---------- wheel handler ----------

    def _on_scroll_zoom(self, event) -> None:
        """
        Matplotlib 'scroll_event' handler.
          - Wheel = zoom around cursor (self.zoom_sensitivity per notch).
          - Shift + Wheel = pan (self.pan_sensitivity * window per notch).
        Operates only on time axes (and the strip); ignores XY panes.
        """
        # Build the set of valid axes (time axes + strip)
        if getattr(self, "_time_axis_keys", None):
            valid_axes = [self.user_axes[k] for k in self._time_axis_keys if k in self.user_axes]
        else:
            valid_axes = list(self.user_axes.values())
        if self.strip_ax is not None:
            valid_axes.append(self.strip_ax)
    
        if event.inaxes not in valid_axes:
            return
    
        # Direction (+1 zoom in / pan left, -1 zoom out / pan right)
        direction = 1 if getattr(event, "button", None) == "up" else -1
        is_pan = isinstance(getattr(event, "key", None), str) and ("shift" in event.key)
    
        if is_pan:
            # Pan by a fraction of the current window
            win = self.t1 - self.t0
            dx = direction * self.pan_sensitivity * win
            new_t0 = self.t0 - dx
            new_t1 = self.t1 - dx
            new_t0, new_t1 = self._clamp_to_bounds(new_t0, new_t1)
        else:
            # Zoom around the cursor (fall back to window center)
            import matplotlib.dates as mdates
            center = self._ts_from_mpl_x(getattr(event, "xdata", None))
            win = self.t1 - self.t0
    
            # one notch scales window by (1 ± zoom_sensitivity)
            scale = (1.0 - self.zoom_sensitivity) if direction > 0 else (1.0 + self.zoom_sensitivity)
            new_win = win * scale
            if new_win < self.min_window:
                new_win = self.min_window
            if new_win > self.max_window:
                new_win = self.max_window
    
            half = new_win / 2
            new_t0 = center - half
            new_t1 = center + half
            new_t0, new_t1 = self._clamp_to_bounds(new_t0, new_t1)
    
        self.t0, self.t1 = new_t0, new_t1
        self._sync_entries_and_plot()
