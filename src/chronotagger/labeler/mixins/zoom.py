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
    
    # ---------- axis zoom methods ----------
    
    def _zoom_y_axis(self, event) -> None:
        """
        Zoom Y-axis of the plot under cursor.
        
        Zooms around the axis center (not cursor position) for consistent behavior.
        This ensures zooming out then in cancels perfectly.
        Fast redraw using canvas.draw_idle() without full replot.
        """
        ax = event.inaxes
        if ax is None:
            return
        
        direction = 1 if getattr(event, 'button', None) == 'up' else -1
        
        # Get current ylim
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        
        # Always zoom around axis center (not cursor) for consistent behavior
        center_y = (ymin + ymax) / 2
        
        # Calculate new limits
        zoom_factor = (1 - self.zoom_sensitivity) if direction > 0 else (1 + self.zoom_sensitivity)
        new_range = y_range * zoom_factor
        
        half = new_range / 2
        new_ymin = center_y - half
        new_ymax = center_y + half
        
        # Apply
        ax.set_ylim(new_ymin, new_ymax)
        
        # Track manual zoom
        if ax not in self._manual_zooms:
            self._manual_zooms[ax] = set()
        self._manual_zooms[ax].add('y')
        
        # Fast redraw (no replot)
        self.canvas.draw_idle()
    
    def _zoom_x_axis(self, event) -> None:
        """
        Zoom X-axis of cross-plot under cursor.
        
        Only works on cross-plots (role='not-time').
        Zooms around the axis center (not cursor position) for consistent behavior.
        """
        ax = event.inaxes
        if ax is None or not self._is_cross_plot_axis(ax):
            return
        
        direction = 1 if getattr(event, 'button', None) == 'up' else -1
        
        # Get current xlim
        xmin, xmax = ax.get_xlim()
        x_range = xmax - xmin
        
        # Always zoom around axis center (not cursor) for consistent behavior
        center_x = (xmin + xmax) / 2
        
        # Calculate new limits
        zoom_factor = (1 - self.zoom_sensitivity) if direction > 0 else (1 + self.zoom_sensitivity)
        new_range = x_range * zoom_factor
        
        half = new_range / 2
        new_xmin = center_x - half
        new_xmax = center_x + half
        
        # Apply
        ax.set_xlim(new_xmin, new_xmax)
        
        # Track manual zoom
        if ax not in self._manual_zooms:
            self._manual_zooms[ax] = set()
        self._manual_zooms[ax].add('x')
        
        # Fast redraw (no replot)
        self.canvas.draw_idle()
    
    def _zoom_both_axes(self, event) -> None:
        """
        Zoom both X and Y axes of cross-plot under cursor (simultaneously).
        
        Only works on cross-plots (role='not-time').
        Zooms around the axis centers (not cursor position) for consistent behavior.
        This is the default scroll behavior for cross-plots.
        """
        ax = event.inaxes
        if ax is None or not self._is_cross_plot_axis(ax):
            return
        
        direction = 1 if getattr(event, 'button', None) == 'up' else -1
        zoom_factor = (1 - self.zoom_sensitivity) if direction > 0 else (1 + self.zoom_sensitivity)
        
        # Zoom X-axis
        xmin, xmax = ax.get_xlim()
        x_range = xmax - xmin
        center_x = (xmin + xmax) / 2
        new_x_range = x_range * zoom_factor
        half_x = new_x_range / 2
        ax.set_xlim(center_x - half_x, center_x + half_x)
        
        # Zoom Y-axis
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        center_y = (ymin + ymax) / 2
        new_y_range = y_range * zoom_factor
        half_y = new_y_range / 2
        ax.set_ylim(center_y - half_y, center_y + half_y)
        
        # Track manual zoom
        if ax not in self._manual_zooms:
            self._manual_zooms[ax] = set()
        self._manual_zooms[ax].add('x')
        self._manual_zooms[ax].add('y')
        
        # Fast redraw (no replot)
        self.canvas.draw_idle()
    
    def _zoom_time_range(self, event) -> None:
        """
        Zoom time range, centered on current window.
        
        Called when scrolling over non-plot areas (strip, empty canvas).
        Triggers full replot and resets axis zoom to auto.
        """
        direction = 1 if getattr(event, 'button', None) == 'up' else -1
        
        # Center on current window (not cursor)
        center = self.t0 + (self.t1 - self.t0) / 2
        
        win = self.t1 - self.t0
        scale = (1.0 - self.zoom_sensitivity) if direction > 0 else (1.0 + self.zoom_sensitivity)
        new_win = win * scale
        
        # Clamp to min/max window
        new_win = max(self.min_window, min(new_win, self.max_window))
        
        half = new_win / 2
        new_t0 = center - half
        new_t1 = center + half
        
        # Clamp to data bounds
        new_t0, new_t1 = self._clamp_to_bounds(new_t0, new_t1)
        
        self.t0, self.t1 = new_t0, new_t1
        self._time_range_dirty = True
        self._sync_entries_and_plot()
    
    def _pan_time_range(self, event) -> None:
        """
        Pan time range left/right by a fraction of the window.
        
        Preserves existing Shift + Wheel behavior.
        """
        direction = 1 if getattr(event, 'button', None) == 'up' else -1
        
        # Pan by a fraction of the current window
        win = self.t1 - self.t0
        dx = direction * self.pan_sensitivity * win
        new_t0 = self.t0 - dx
        new_t1 = self.t1 - dx
        new_t0, new_t1 = self._clamp_to_bounds(new_t0, new_t1)
        
        self.t0, self.t1 = new_t0, new_t1
        self._time_range_dirty = True
        self._sync_entries_and_plot()
    
    def _is_cross_plot_axis(self, ax) -> bool:
        """
        Check if axis is a cross-plot (role='not-time').
        
        Returns:
            True if axis has role='not-time', False otherwise
        """
        for key, axis in self.user_axes.items():
            if axis is ax:
                role = self.axes_meta.get(key, {}).get('role', 'time').lower()
                return role == 'not-time'
        return False
    
    def _reset_all_yscales(self) -> None:
        """
        Reset all axis scales to auto limits.
        
        Called by "Reset Scale" button.
        Resets Y-axes for all plots, and X-axes for cross-plots.
        """
        # Reset to auto limits
        for ax, (ymin, ymax) in self._auto_ylims.items():
            ax.set_ylim(ymin, ymax)
        
        for ax, (xmin, xmax) in self._auto_xlims.items():
            ax.set_xlim(xmin, xmax)
        
        # Clear manual zoom tracking
        self._manual_zooms.clear()
        
        # Redraw
        self.canvas.draw_idle()
        
        # Update status
        if hasattr(self, 'status_var') and self.status_var is not None:
            self.status_var.set("Scales reset to auto")

    # ---------- wheel handler ----------

    def _on_scroll_zoom(self, event) -> None:
        """
        Matplotlib 'scroll_event' handler with split behavior:
        
        - Over a cross-plot (no modifiers) → Zoom both X and Y axes
        - Over a time plot (no modifiers) → Zoom Y-axis only
        - Over a cross-plot (Ctrl/Alt) → Zoom X-axis only
        - Not over a plot (no modifiers) → Zoom time range (centered)
        - Shift + Wheel → Pan time range left/right (existing behavior)
        
        The cursor location determines which zoom mode activates.
        """
        # Check for modifier keys
        key = getattr(event, 'key', None) or ''
        has_shift = 'shift' in key
        has_ctrl = 'control' in key or 'ctrl' in key
        has_alt = 'alt' in key
        
        # Shift + Wheel = Pan (preserve existing behavior)
        if has_shift:
            self._pan_time_range(event)
            return
        
        # Determine where the cursor is
        if event.inaxes in self.user_axes.values():
            # Over a DATA PLOT
            is_cross_plot = self._is_cross_plot_axis(event.inaxes)
            
            if is_cross_plot:
                # Cross-plot behavior
                if has_ctrl or has_alt:
                    # Modifier key → X-axis only
                    self._zoom_x_axis(event)
                else:
                    # No modifier → Both axes (simultaneous zoom)
                    self._zoom_both_axes(event)
            else:
                # Time plot → Y-axis only
                self._zoom_y_axis(event)
        else:
            # Not over a plot (strip, empty canvas, etc.) → Time zoom
            self._zoom_time_range(event)
