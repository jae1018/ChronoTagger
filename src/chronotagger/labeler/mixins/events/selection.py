"""
Selection handlers mixin.

Responsibilities:
- Rectangle/box selection callback
- Strip click (pick_event) for interval selection
- Right-click cancellation
- Point highlighting for selections
- Helper methods for selection processing
"""

from __future__ import annotations

from typing import Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas as pd
import numpy as np


class SelectionMixin:
    """Mixin providing selection-related event handlers and utilities."""

    def _on_rectangle_select(self, eclick, erelease, pane) -> None:
        """
        Handle both:
          • Full-height time selection (t0..t1)  -> single preview span (legacy)
          • Box selection (t/y bounded)          -> one or more preview spans from points

        Rule of thumb:
          If the selection covers ~entire y-range of the axis (>=95%), we treat it as time-only.
          Otherwise, we treat it as a box over data points on time-lane panels.

        Args:
            eclick: Mouse click event at selection start
            erelease: Mouse release event at selection end
            pane: The TabPane where the selection occurred
        """
        # Which selector triggered this?
        triggered_selector_key = None
        for key, selector in pane.rect_selectors.items():
            if pane.user_axes.get(key) is eclick.inaxes:
                triggered_selector_key = key
                break

        # Only process if we have a valid selector match
        if triggered_selector_key is None:
            return

        if eclick.xdata is None or erelease.xdata is None:
            return
        ax = getattr(eclick, "inaxes", None)
        if ax is None:
            return

        # If the two-click preview was armed by the initial mouse press,
        # cancel it so rectangle-select takes precedence.
        if getattr(self, "_two_click_active", False) or getattr(self, "_twoclick_motion_cid", None):
            self._clear_two_click_state(keep_selection=True)  # hides overlays + disconnects motion

        # Compute data-rect
        x1, x2 = float(eclick.xdata), float(erelease.xdata)
        y1 = float(getattr(eclick, "ydata", np.nan))
        y2 = float(getattr(erelease, "ydata", np.nan))
        x_lo, x_hi = (x1, x2) if x1 <= x2 else (x2, x1)
        y_lo, y_hi = (y1, y2) if y1 <= y2 else (y2, y1)

        # Heuristic: full-height band => treat as time-only
        try:
            ymin, ymax = ax.get_ylim()
            y_span = max(1e-12, float(ymax) - float(ymin))
            sel_span = max(0.0, min(float(ymax), y_hi) - max(float(ymin), y_lo))
            full_height = (sel_span / y_span) >= 0.95
        except Exception:
            full_height = True  # safest fallback

        # Reset arbitration flags from a completed drag
        self._dragging_box = False
        self._press_xy_px = None

        if full_height:
            # === TIME-ONLY path (backwards-compatible) ===
            import matplotlib.dates as mdates
            def _to_naive_ts(xf: float) -> pd.Timestamp:
                dt = mdates.num2date(xf)
                if getattr(dt, "tzinfo", None) is not None:
                    dt = dt.replace(tzinfo=None)
                return pd.Timestamp(dt)

            t_start, t_end = _to_naive_ts(x_lo), _to_naive_ts(x_hi)

            if self.snap_var.get():  # type: ignore[union-attr]
                t_start, t_end = self._snap_to_samples(t_start, t_end)

            self.current_spans.clear()
            self.current_selection = (min(t_start, t_end), max(t_start, t_end))
            self._commit_spans = []
            self.status_var.set(  # type: ignore[union-attr]
                f"Selected: {self.current_selection[0].strftime('%H:%M:%S')} → {self.current_selection[1].strftime('%H:%M:%S')}"
            )
            self._update_strip()

            # Show highlight overlays on selected points
            self._show_selected_point_highlights()

            self.canvas.draw_idle()  # type: ignore[union-attr]
            return

        # === BOX-SELECT path (points-in-rect over time lane axes OR not-time axes) ===
        import matplotlib.dates as mdates

        # CRITICAL: Only check data from the axis where the box was drawn
        # to avoid y-coordinate mismatches across different panels
        drag_ax = getattr(eclick, 'inaxes', None)

        # Validate that the drag occurred on a user axis
        if drag_ax is None:
            # No axis → abort
            self.current_spans.clear()
            self.current_selection = None
            self._commit_spans = []
            self._update_strip()
            self.canvas.draw_idle()
            return

        if drag_ax is pane.strip_ax:
            # Box drawn on strip → abort
            self.current_spans.clear()
            self.current_selection = None
            self._commit_spans = []
            self.status_var.set("Box selection not supported on labels strip")
            self._update_strip()
            self.canvas.draw_idle()
            return

        # Determine which axis key and role
        meta_key = None
        for k, a in pane.user_axes.items():
            if a is drag_ax:
                meta_key = k
                break

        if meta_key is None:
            # Not a known user axis, abort
            self.current_spans.clear()
            self.current_selection = None
            self._commit_spans = []
            self._update_strip()
            self.canvas.draw_idle()
            return

        # Get axis role from metadata
        role = pane.axes_meta.get(meta_key, {}).get("role", "time").lower()

        xlo, xhi = float(x_lo), float(x_hi)
        ylo, yhi = float(y_lo), float(y_hi)

        # Branch on axis role
        if role == "not-time":
            # === NOT-TIME AXIS path (position plots, phase space, etc.) ===
            # Use the triggered selector key directly
            exact_intervals = self._box_select_on_not_time_axis(drag_ax, xlo, xhi, ylo, yhi, triggered_selector_key)

            if not exact_intervals:
                self.current_spans.clear()
                self.current_selection = None
                self._commit_spans = []
                self.status_var.set("No points in selection")
                self._update_strip()
                self.canvas.draw_idle()
                return

            # APPLY LOCALIZED PADDING IMMEDIATELY for preview (user's request)
            padded_intervals = self._apply_localized_padding_to_intervals(exact_intervals)

            # Use padded intervals for both preview AND commit
            spans_commit = padded_intervals
            spans_preview = [(s, e) for s, e in padded_intervals]

            # Optional snap (preview only)
            if self.snap_var.get():
                snapped_prev = []
                for s, e in spans_preview:
                    ss, ee = self._snap_to_samples(s, e)
                    snapped_prev.append((ss, ee))
                spans_preview = snapped_prev

            self.current_selection = None
            self.current_spans = spans_preview
            self._commit_spans = spans_commit

            # UPDATE TIME OVERLAYS for multi-span preview with padded intervals
            self._update_time_overlays_for_multi_spans(spans_preview)

            # UPDATE STRIP PREVIEW with padded intervals
            import matplotlib.dates as mdates
            spans_float = [(mdates.date2num(s), mdates.date2num(e)) for s, e in spans_preview]
            self._draw_strip_preview_spans(spans_float)

            self.status_var.set(f"Selected {len(spans_preview)} block(s) from position axis")
            self._update_strip()

            # Show highlight overlays on selected points
            self._show_selected_point_highlights()

            self.canvas.draw_idle()
            return

        # === TIME AXIS path (original logic) ===
        # Validate it's actually a time-lane axis
        if not self._is_time_lane_axis(drag_ax):
            self.current_spans.clear()
            self.current_selection = None
            self._commit_spans = []
            self.status_var.set("Box selection only works on time or not-time axes")
            self._update_strip()
            self.canvas.draw_idle()
            return

        # Only check THIS axis
        time_axes = [drag_ax]

        xlo, xhi = float(x_lo), float(x_hi)
        ylo, yhi = float(y_lo), float(y_hi)

        # Collect timestamps from lines & scatters inside the box
        picked_ts: list[pd.Timestamp] = []

        for a in time_axes:
            # Line2D objects
            for ln in a.lines:
                try:
                    xs = np.asarray(ln.get_xdata(orig=False), dtype=float)
                    ys = np.asarray(ln.get_ydata(orig=False), dtype=float)
                    if xs.size != ys.size or xs.size == 0:
                        continue
                    m = (xs >= xlo) & (xs <= xhi) & (ys >= ylo) & (ys <= yhi)
                    if not m.any():
                        continue
                    xs_sel = xs[m]
                    # Convert selected xs (float days) to naive timestamps
                    for xf in xs_sel:
                        dt = mdates.num2date(float(xf))
                        if getattr(dt, "tzinfo", None) is not None:
                            dt = dt.replace(tzinfo=None)
                        picked_ts.append(pd.Timestamp(dt))
                except Exception:
                    continue

            # Scatter-style PathCollections
            for coll in a.collections:
                if not hasattr(coll, "get_offsets"):
                    continue
                try:
                    off = np.asarray(coll.get_offsets())
                    if off.size == 0:
                        continue
                    # offsets are Nx2 in data coords [x, y]
                    xs = off[:, 0].astype(float, copy=False)
                    ys = off[:, 1].astype(float, copy=False)
                    m = (xs >= xlo) & (xs <= xhi) & (ys >= ylo) & (ys <= yhi)
                    if not m.any():
                        continue
                    for xf in xs[m]:
                        dt = mdates.num2date(float(xf))
                        if getattr(dt, "tzinfo", None) is not None:
                            dt = dt.replace(tzinfo=None)
                        picked_ts.append(pd.Timestamp(dt))
                except Exception:
                    continue

        # Nothing in the box → just clear preview
        if not picked_ts:
            self.current_spans.clear()
            self.current_selection = None
            self._commit_spans = []

            # Clear point highlights
            if hasattr(self, '_clear_selected_point_highlights'):
                self._clear_selected_point_highlights()

            self.status_var.set("No points in selection")  # type: ignore[union-attr]
            self._update_strip()
            self.canvas.draw_idle()  # type: ignore[union-attr]
            return

        # Convert timestamps to index positions (nearest) and keep only those inside current data bounds
        idx_full = self.df.index
        pos = []
        for ts in picked_ts:
            j = idx_full.get_indexer([ts], method="nearest")[0]
            if 0 <= j < len(idx_full):
                pos.append(j)

        if not pos:
            self.current_spans.clear()
            self.current_selection = None
            self._commit_spans = []

            # Clear point highlights
            if hasattr(self, '_clear_selected_point_highlights'):
                self._clear_selected_point_highlights()

            self.status_var.set("No points in selection")  # type: ignore[union-attr]
            self._update_strip()
            self.canvas.draw_idle()  # type: ignore[union-attr]
            return

        pos = sorted(set(pos))

        # Split into contiguous runs (diff == 1)
        runs: list[tuple[int, int]] = []
        run_start = pos[0]
        prev = pos[0]
        for j in pos[1:]:
            if j == prev + 1:
                prev = j
                continue
            runs.append((run_start, prev))
            run_start = prev = j
        runs.append((run_start, prev))

        # Turn runs into intervals
        # --- For BOX SELECTIONS: Use exact intervals [first, last] ---
        # This avoids boundary issues where highlighting doesn't match selection
        exact_intervals = self._runs_to_exact_intervals(idx_full, runs)

        # APPLY LOCALIZED PADDING IMMEDIATELY for preview (user's request)
        padded_intervals = self._apply_localized_padding_to_intervals(exact_intervals)

        # Use padded intervals for both preview AND commit
        spans_commit = padded_intervals
        spans_preview = [(s, e) for s, e in padded_intervals]

        # Optional snap (preview only)
        if self.snap_var.get():  # type: ignore[union-attr]
            snapped_prev: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            for s, e in spans_preview:
                ss, ee = self._snap_to_samples(s, e)
                snapped_prev.append((ss, ee))
            spans_preview = snapped_prev

        # Stash both: preview for drawing, commit for "Add Label"
        self.current_selection = None
        self.current_spans = spans_preview        # exact intervals for highlighting
        self._commit_spans = spans_commit         # exact intervals for interval creation

        # UPDATE TIME OVERLAYS for multi-span preview with padded intervals
        self._update_time_overlays_for_multi_spans(spans_preview)

        # UPDATE STRIP PREVIEW with padded intervals
        import matplotlib.dates as mdates
        spans_float = [(mdates.date2num(s), mdates.date2num(e)) for s, e in spans_preview]
        self._draw_strip_preview_spans(spans_float)

        self.status_var.set(
            f"Selected {len(spans_preview)} contiguous block(s) from {len(pos)} point(s)"
        )
        self._update_strip()

        # Show highlight overlays on selected points
        self._show_selected_point_highlights()

        self.canvas.draw_idle()


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
                # Check if this is the already selected interval - if so, deselect it
                if hasattr(self, 'selected_interval') and self.selected_interval is iv:
                    self.selected_interval = None
                    if hasattr(self, '_clear_selected_interval_highlights'):
                        self._clear_selected_interval_highlights()
                    self._update_strip()
                    if hasattr(self, '_update_intervals_list'):
                        self._update_intervals_list()
                    if self.status_var is not None:
                        self.status_var.set("Interval deselected")
                    self.canvas.draw()  # type: ignore[union-attr]
                    return

                # Otherwise, select this interval
                self.selected_interval = iv
                self.status_var.set(  # type: ignore[union-attr]
                    f"Selected: {iv.label} [{iv.start.strftime('%H:%M:%S')} → {iv.end.strftime('%H:%M:%S')}]"
                )
                self._update_strip()
                self._show_selected_interval_highlights()
                self.canvas.draw()  # type: ignore[union-attr]
                break

    def _x_from_anywhere(self, event) -> float | None:
        """
        Get a time-axis x (mdates float) no matter where the cursor is.
        If we're over a time axis, use event.xdata. Otherwise, map screen x
        into the primary time axis.
        """
        import matplotlib as mpl

        if getattr(self, "_primary_time_key", None) is None:
            return None

        primary_ax = self.user_axes.get(self._primary_time_key)
        if primary_ax is None:
            return None

        # If we're already on a time axis (or the strip), event.xdata is correct.
        time_axes = {self.user_axes[k] for k in (self._time_axis_keys or [])}
        if getattr(self, "strip_ax", None) is not None:
            time_axes.add(self.strip_ax)

        if event.inaxes in time_axes and event.xdata is not None:
            return float(event.xdata)

        # Otherwise, map canvas pixel x to primary time-axis data x.
        inv = primary_ax.transData.inverted()
        # pick a y inside the primary axes bbox
        ymid = primary_ax.bbox.y0 + primary_ax.bbox.height * 0.5
        try:
            x_data, _ = inv.transform((event.x, ymid))
            return float(x_data)
        except Exception:
            return None

    def _box_select_on_not_time_axis(self, ax, xlo: float, xhi: float,
                                      ylo: float, yhi: float,
                                      triggered_key: Optional[str] = None) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Given a box on a "not-time" axis (e.g., position plot X-Y), find which points fall inside,
        map them to timestamps via their order in the windowed dataframe, and return time intervals.

        First tries direct dataframe filtering (if x_col/y_col configured),
        then falls back to artist-based extraction for backwards compatibility.

        Returns:
            List of (start, end) timestamp tuples for half-open intervals [start, end)
        """
        # Try direct dataframe filtering first
        # Use the triggered selector key if available, otherwise fall back to search
        ax_key = triggered_key or self._find_axes_key(ax)
        if ax_key:
            intervals = self._try_dataframe_box_filter(ax_key, xlo, xhi, ylo, yhi)
            if intervals is not None:
                return intervals

        # Fallback to artist-based extraction (original method)
        return self._box_select_via_artists(ax, xlo, xhi, ylo, yhi)

    def _try_dataframe_box_filter(self, ax_key: str, xlo: float, xhi: float,
                                   ylo: float, yhi: float) -> Optional[list[tuple[pd.Timestamp, pd.Timestamp]]]:
        """
        Try direct dataframe filtering using configured column mappings.

        Args:
            ax_key: Key for the axes (e.g., "xy_gse", "xy_sse")
            xlo, xhi, ylo, yhi: Box bounds in data coordinates

        Returns:
            List of intervals if successful, None if no column mapping available
        """
        # Get area configuration - check multiple possible locations
        area_config = None

        # Try axes_meta first
        if hasattr(self, 'axes_meta') and self.axes_meta:
            area_config = self.axes_meta.get(ax_key, {})

        # Always also check layout_spec for custom keys (in case they got stripped)
        if hasattr(self, 'layout_spec') and self.layout_spec:
            # Look through the areas in layout_spec
            areas = self.layout_spec.get('areas', [])
            for area in areas:
                if area.get('key') == ax_key:
                    layout_config = area
                    # Merge layout_spec config into area_config (layout_spec takes precedence for custom keys)
                    if area_config is None:
                        area_config = layout_config
                    else:
                        # Merge, with layout_spec taking precedence for x_col/y_col
                        area_config = {**area_config, **{k: v for k, v in layout_config.items()
                                                        if k in ['x_col', 'y_col']}}
                    break

        if not area_config:
            return None

        # Check if column mappings are configured
        x_col = area_config.get('x_col')
        y_col = area_config.get('y_col')

        if not x_col or not y_col:
            return None  # No column mapping - use fallback method

        # Verify columns exist in dataframe
        if x_col not in self.df.columns or y_col not in self.df.columns:
            return None  # Columns don't exist - use fallback method

        try:
            # Get the current windowed dataframe
            windowed_df = self.df.loc[self.t0:self.t1].copy()
            if windowed_df.empty:
                return []

            # Filter by box bounds using dataframe columns directly
            mask = (
                (windowed_df[x_col] >= xlo) & (windowed_df[x_col] <= xhi) &
                (windowed_df[y_col] >= ylo) & (windowed_df[y_col] <= yhi)
            )

            selected_df = windowed_df[mask]

            if selected_df.empty:
                return []

            # Get timestamps of selected points
            selected_timestamps = selected_df.index.tolist()

            # Map to index positions in the FULL dataframe
            idx_full = self.df.index
            pos_in_full = []
            for ts in selected_timestamps:
                j = idx_full.get_indexer([ts], method="nearest")[0]
                if 0 <= j < len(idx_full):
                    pos_in_full.append(j)

            if not pos_in_full:
                return []

            # Find contiguous runs
            runs = self._find_contiguous_runs(pos_in_full)

            # Convert to exact intervals for box selections (avoids boundary issues)
            intervals = self._runs_to_exact_intervals(idx_full, runs)

            return intervals

        except Exception as e:
            # If anything goes wrong, fall back to artist method
            return None

    def _box_select_via_artists(self, ax, xlo: float, xhi: float,
                                ylo: float, yhi: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Box selection using artist-based extraction (original method).

        This is the fallback method when direct dataframe filtering is not available.
        """
        import numpy as np

        # Get the windowed index (cached from last plot)
        windowed_idx = getattr(self, "_last_windowed_index", None)
        if windowed_idx is None or len(windowed_idx) == 0:
            return []

        # Collect indices of points inside the box
        picked_indices = []  # positions in windowed_idx

        # Check Line2D objects
        for artist in ax.lines:
            if not hasattr(artist, "get_xdata"):
                continue
            try:
                xs = np.asarray(artist.get_xdata(orig=False), dtype=float)
                ys = np.asarray(artist.get_ydata(orig=False), dtype=float)
                if xs.size != ys.size or xs.size == 0:
                    continue
                mask = (xs >= xlo) & (xs <= xhi) & (ys >= ylo) & (ys <= yhi)
                # The TRUE indices in mask correspond to indices in windowed_idx
                picked_indices.extend(np.where(mask)[0].tolist())
            except Exception:
                continue

        # Check Scatter collections (PathCollections)
        for artist in ax.collections:
            if not hasattr(artist, "get_offsets"):
                continue
            try:
                offsets = np.asarray(artist.get_offsets())
                if offsets.size == 0:
                    continue
                xs = offsets[:, 0]
                ys = offsets[:, 1]
                mask = (xs >= xlo) & (xs <= xhi) & (ys >= ylo) & (ys <= yhi)
                picked_indices.extend(np.where(mask)[0].tolist())
            except Exception:
                continue

        if not picked_indices:
            return []

        # Deduplicate and sort
        picked_indices = sorted(set(picked_indices))

        # Convert windowed indices to timestamps
        picked_timestamps = [pd.Timestamp(windowed_idx[i]) for i in picked_indices
                            if 0 <= i < len(windowed_idx)]

        # Map to index positions in the FULL dataframe (not windowed)
        idx_full = self.df.index
        pos_in_full = []
        for ts in picked_timestamps:
            j = idx_full.get_indexer([ts], method="nearest")[0]
            if 0 <= j < len(idx_full):
                pos_in_full.append(j)

        if not pos_in_full:
            return []

        # Find contiguous runs
        runs = self._find_contiguous_runs(pos_in_full)

        # Convert to exact intervals for box selections (avoids boundary issues)
        return self._runs_to_exact_intervals(idx_full, runs)

    def _on_right_click_cancel(self, event) -> None:
        """
        Handle right-click to cancel active selections or deselect interval.
        Works on any axis (time axes, position axes, strip).
        """
        if getattr(event, "button", None) != 3:  # Only handle right-click
            return

        # Check if we have any active selection or preview
        has_selection = (
            getattr(self, "_pick_anchor_ts", None) is not None or
            getattr(self, "current_selection", None) is not None or
            bool(getattr(self, "current_spans", None)) or
            bool(getattr(self, "_commit_spans", None)) or
            getattr(self, "_two_click_active", False)
        )

        if has_selection:
            # Cancel the active selection
            self._cancel_active_selection()
            if self.status_var is not None:
                self.status_var.set("Selection canceled (right-click)")
            return

        # If no active selection, check for selected interval to deselect
        if hasattr(self, 'selected_interval') and self.selected_interval is not None:
            self.selected_interval = None
            if hasattr(self, '_clear_selected_interval_highlights'):
                self._clear_selected_interval_highlights()
            self._update_strip()
            if hasattr(self, '_update_intervals_list'):
                self._update_intervals_list()
            if self.canvas is not None:
                self.canvas.draw_idle()
            if self.status_var is not None:
                self.status_var.set("Interval deselected (right-click)")
            return

    # ========== Rectangle Selection Edge Clamping ==========

    def _on_rect_selector_press(self, eclick) -> None:
        """
        Track when rectangle selector drag starts.

        Called automatically when user starts dragging a selection box.
        Records which axes the drag started in so we can clamp to that
        axes if the mouse leaves it during the drag.

        Also disables other rectangle selectors to prevent coordinate interference.

        Args:
            eclick: matplotlib mouse event (button press)
        """
        if eclick.button != 1:  # Only track left mouse button
            return

        # Store the axes where drag started
        self._rect_drag_axes = getattr(eclick, 'inaxes', None)

        # Store the starting point (for detecting actual drag vs click)
        if self._rect_drag_axes is not None:
            self._rect_drag_start = (eclick.xdata, eclick.ydata)
        else:
            self._rect_drag_start = None

        # Disable all other rectangle selectors to prevent interference
        if hasattr(self, 'rect_selectors') and self._rect_drag_axes is not None:
            active_key = None
            for key, ax in self.user_axes.items():
                if ax is self._rect_drag_axes:
                    active_key = key
                    break

            # Disable all selectors except the active one
            for key, selector in self.rect_selectors.items():
                if key != active_key:
                    selector.set_active(False)

    def _on_rect_selector_motion(self, event) -> None:
        """
        Handle mouse motion during rectangle selection.

        When the mouse leaves the axes during an active drag, this method
        clamps the rectangle corner to the axes edges so users can easily
        select all the way to corners without pixel-perfect precision.

        Args:
            event: matplotlib mouse motion event
        """
        # Only act if we have an active drag
        drag_axes = getattr(self, '_rect_drag_axes', None)
        if drag_axes is None:
            return

        # Check if this axes has a rectangle selector
        rect_selector = self.rect_selectors.get(self._find_axes_key(drag_axes))
        if rect_selector is None or not rect_selector.active:
            return

        # If mouse is still inside the original axes, do nothing
        # (let RectangleSelector handle it normally)
        if event.inaxes == drag_axes:
            return

        # Mouse left the axes during drag - clamp to edges
        self._clamp_rectangle_to_axes(event, drag_axes, rect_selector)

    def _clamp_rectangle_to_axes(self, event, axes, rect_selector) -> None:
        """
        Clamp rectangle selection to axes bounds when mouse leaves axes.

        Transforms the mouse position (in figure coordinates) to data
        coordinates, clamps to axes limits, and manually updates the
        rectangle selector extents using fast blitting.

        Args:
            event: matplotlib mouse event
            axes: The axes where the drag started
            rect_selector: The RectangleSelector for this axes
        """
        try:
            # Transform figure coordinates to axes data coordinates
            inv = axes.transData.inverted()
            x_data, y_data = inv.transform((event.x, event.y))

            # Get axes limits (the boundaries we clamp to)
            xmin, xmax = axes.get_xlim()
            ymin, ymax = axes.get_ylim()

            # Clamp to axes bounds
            x_clamped = max(xmin, min(xmax, x_data))
            y_clamped = max(ymin, min(ymax, y_data))

            # Get the starting corner from the selector
            # (the corner that's anchored when user drags)
            if not hasattr(rect_selector, '_eventpress') or rect_selector._eventpress is None:
                return

            x_start = rect_selector._eventpress.xdata
            y_start = rect_selector._eventpress.ydata

            if x_start is None or y_start is None:
                return

            # Calculate rectangle extents
            left = min(x_start, x_clamped)
            right = max(x_start, x_clamped)
            bottom = min(y_start, y_clamped)
            top = max(y_start, y_clamped)

            # Fast blitting approach: update rectangle patch directly
            # This is much faster than canvas.draw_idle() which redraws everything
            try:
                # Access the rectangle patch (RectangleSelector internal)
                rect_patch = rect_selector._selection_artist

                if rect_patch is None:
                    return

                # Update patch geometry directly
                width = right - left
                height = top - bottom
                rect_patch.set_bounds(left, bottom, width, height)

                # Make sure it's visible
                if not rect_patch.get_visible():
                    rect_patch.set_visible(True)

                # Use BlitHelper for fast redraw (same technique as two-click selection)
                if hasattr(self, '_blit') and self._blit is not None:
                    self._blit.draw([rect_patch])
                else:
                    # Fallback to axes-specific blit (still faster than full redraw)
                    axes.draw_artist(rect_patch)
                    self.canvas.blit(axes.bbox)

            except AttributeError:
                # If we can't access internals, fall back to setting extents
                # (slower but still works)
                extents = [left, right, bottom, top]
                rect_selector.extents = extents
                self.canvas.draw_idle()

        except Exception:
            # Silently fail - better to have normal behavior than crash
            # This can happen if coordinate transforms are invalid
            pass

    def _on_rect_selector_release(self, erelease) -> None:
        """
        Clean up rectangle selection tracking on mouse release.

        Re-enables all rectangle selectors that were disabled during drag.

        Called automatically when user releases the mouse button.

        Args:
            erelease: matplotlib mouse event (button release)
        """
        # Re-enable all rectangle selectors
        if hasattr(self, 'rect_selectors'):
            for key, selector in self.rect_selectors.items():
                selector.set_active(True)

        # Clear tracking state
        self._rect_drag_axes = None
        self._rect_drag_start = None

    def _find_axes_key(self, axes) -> Optional[str]:
        """
        Find the key for a given axes object in user_axes dict.

        Args:
            axes: matplotlib Axes object

        Returns:
            The key string if found, None otherwise
        """
        for key, ax in self.user_axes.items():
            if ax is axes:
                return key
        return None

    # ========== Selected Points Highlighting ==========

    def _extract_data_at_indices(self, ax, indices: list) -> Tuple[list, list]:
        """
        Extract (x, y) data from axes at specified indices.

        CRITICAL: Only extracts from main dataframe artists, not boundary markers.
        Uses cached windowed data to ensure we only highlight actual dataframe points.

        Args:
            ax: matplotlib Axes object
            indices: List of integer indices into the windowed dataframe

        Returns:
            Tuple of (x_values, y_values) lists
        """
        import numpy as np

        x_vals = []
        y_vals = []

        # Get the windowed index and dataframe (cached from last plot)
        windowed_idx = getattr(self, '_last_windowed_index', None)
        windowed_df = getattr(self, '_last_windowed_df', None)

        if windowed_idx is None or windowed_df is None or len(windowed_idx) == 0:
            # Fallback: create windowed data on the fly
            try:
                windowed_df = self.df.loc[self.t0:self.t1].copy()
                windowed_idx = windowed_df.index
                if windowed_df.empty:
                    return x_vals, y_vals
            except Exception as e:
                return x_vals, y_vals

        # CRITICAL: Instead of extracting from ALL artists (which includes boundary markers),
        # extract directly from the cached windowed dataframe data

        # Find the axes role to determine what data to extract
        ax_key = self._find_axes_key(ax)
        if not ax_key:
            return x_vals, y_vals

        axis_meta = self.axes_meta.get(ax_key, {})
        role = axis_meta.get("role", "time").lower()

        if role == "time":
            # For time axes: x = time (matplotlib date format), y = data column values
            try:
                # Get time values (convert to matplotlib date format)
                import matplotlib.dates as mdates
                time_vals = [mdates.date2num(windowed_idx[idx]) for idx in indices
                            if 0 <= idx < len(windowed_idx)]

                # For time axes, we need to determine which column is being plotted
                # This is trickier - for now, extract from the first legitimate line artist
                # that has the right number of data points
                for line in ax.lines:
                    try:
                        ys = np.asarray(line.get_ydata(orig=False))
                        if len(ys) == len(windowed_idx):  # Main data artist
                            y_vals = [float(ys[idx]) for idx in indices
                                     if 0 <= idx < len(ys)]
                            x_vals = time_vals[:len(y_vals)]  # Match lengths
                            break
                    except Exception:
                        continue

            except Exception:
                return x_vals, y_vals

        elif role == "not-time":
            # For position plots: extract directly from dataframe using configured columns
            try:
                # Get area configuration
                area_config = self.axes_meta.get(ax_key, {})
                if hasattr(self, 'layout_spec') and self.layout_spec:
                    areas = self.layout_spec.get('areas', [])
                    for area in areas:
                        if area.get('key') == ax_key:
                            area_config.update(area)
                            break

                x_col = area_config.get('x_col')
                y_col = area_config.get('y_col')

                if x_col and y_col and x_col in windowed_df.columns and y_col in windowed_df.columns:
                    # Extract directly from dataframe - guaranteed to be main data only
                    for idx in indices:
                        if 0 <= idx < len(windowed_df):
                            row = windowed_df.iloc[idx]
                            x_vals.append(float(row[x_col]))
                            y_vals.append(float(row[y_col]))
                else:
                    # Fallback: extract from first artist with correct data length
                    for line in ax.lines:
                        try:
                            xs = np.asarray(line.get_xdata(orig=False))
                            ys = np.asarray(line.get_ydata(orig=False))
                            if len(xs) == len(windowed_idx) and len(ys) == len(windowed_idx):
                                for idx in indices:
                                    if 0 <= idx < len(xs) and 0 <= idx < len(ys):
                                        x_vals.append(float(xs[idx]))
                                        y_vals.append(float(ys[idx]))
                                break
                        except Exception:
                            continue

            except Exception:
                return x_vals, y_vals

        return x_vals, y_vals

    def _show_selected_point_highlights(self, redraw: bool = True) -> None:
        """
        Highlight selected points across all axes with overlay markers.

        Creates red scatter markers on top of selected data points to show
        exactly which points are included in the current preview selection.
        Works on both time-series and position/cross plots.

        Args:
            redraw: If True, trigger canvas redraw after adding highlights.
                    Set to False when calling from _update_plot() to avoid
                    redundant redraws (since _update_plot already draws).

        Called automatically when preview selection changes.
        """
        # Clear any existing highlights first
        self._clear_selected_point_highlights()

        # Get selected timestamps from current preview
        selected_timestamps = self._get_preview_timestamps()

        if not selected_timestamps:
            return

        # Convert timestamps to indices in the windowed dataframe
        selected_indices = self._timestamps_to_indices(selected_timestamps)

        if not selected_indices:
            return

        # Downsample if too many points (for performance)
        if len(selected_indices) > 2000:
            # Show every Nth point to keep ~1000 markers per axes
            step = len(selected_indices) // 1000
            selected_indices = selected_indices[::step]

        # Create highlights on all user axes (time and position plots)
        for key, ax in self.user_axes.items():
            x_vals, y_vals = self._extract_data_at_indices(ax, selected_indices)

            if len(x_vals) == 0:
                continue  # No data extracted, skip this axes

            try:
                # Create scatter overlay
                scatter = ax.scatter(
                    x_vals, y_vals,
                    c='red',           # Red color for selected points
                    s=20,              # Marker size
                    alpha=0.6,         # Semi-transparent
                    marker='o',        # Circle markers
                    zorder=100,        # Draw on top
                    edgecolors='darkred',
                    linewidths=0.5
                )

                # Track this highlight for later removal
                if not hasattr(self, '_preview_highlights'):
                    self._preview_highlights = []
                self._preview_highlights.append(scatter)

            except Exception:
                # Silently fail if scatter creation fails
                continue

        # Redraw canvas to show highlights (unless caller will handle it)
        if redraw and hasattr(self, 'canvas') and self.canvas is not None:
            self.canvas.draw_idle()

    def _clear_selected_point_highlights(self) -> None:
        """
        Remove all selected point highlight overlays from axes.

        Called when preview is cleared or new selection is made.
        """
        if not hasattr(self, '_preview_highlights'):
            self._preview_highlights = []
            return

        # Remove each highlight artist from its axes
        for artist in self._preview_highlights:
            try:
                artist.remove()
            except Exception:
                pass  # Already removed or axes destroyed

        self._preview_highlights.clear()

    def _get_preview_timestamps(self) -> list:
        """
        Get list of timestamps from current preview selection.

        Returns:
            List of pd.Timestamp objects representing selected times
        """
        import pandas as pd

        timestamps = []

        # Check for multi-span preview (box selection)
        if hasattr(self, 'current_spans') and self.current_spans:
            for start_ts, end_ts in self.current_spans:
                # Extract all timestamps in this span
                try:
                    mask = (self.df.index >= start_ts) & (self.df.index <= end_ts)
                    span_timestamps = self.df.index[mask].tolist()
                    timestamps.extend(span_timestamps)
                except Exception:
                    pass

        # Check for single-span preview (two-click or full-height selection)
        elif hasattr(self, 'current_selection') and self.current_selection:
            start_ts, end_ts = self.current_selection
            try:
                mask = (self.df.index >= start_ts) & (self.df.index <= end_ts)
                timestamps = self.df.index[mask].tolist()
            except Exception:
                pass

        return timestamps

    def _timestamps_to_indices(self, timestamps: list) -> list:
        """
        Convert timestamps to indices in the windowed dataframe.

        This allows us to extract data from position plots where the
        axes don't use time, but we can use array indices.

        Args:
            timestamps: List of pd.Timestamp objects

        Returns:
            List of integer indices
        """
        # Get the windowed index (cached from last plot)
        windowed_idx = getattr(self, '_last_windowed_index', None)

        if windowed_idx is None or len(windowed_idx) == 0:
            return []

        indices = []

        for ts in timestamps:
            try:
                # Find position of this timestamp in windowed index
                idx = windowed_idx.get_loc(ts)

                # Handle duplicate timestamps (returns slice)
                if isinstance(idx, slice):
                    idx = idx.start

                if idx is not None and 0 <= idx < len(windowed_idx):
                    indices.append(int(idx))

            except Exception:
                # Try nearest neighbor approach as fallback
                try:
                    nearest_idx = windowed_idx.get_indexer([ts], method="nearest")[0]
                    if 0 <= nearest_idx < len(windowed_idx):
                        indices.append(int(nearest_idx))
                except Exception:
                    continue

        return indices

    # ========== Selected Interval Point Highlighting ==========

    def _show_selected_interval_highlights(self) -> None:
        """
        Highlight points for the currently selected interval with blue markers.

        Similar to preview highlighting but uses a different color (blue vs red)
        and works on the selected interval rather than preview selection.
        """
        # Clear any existing interval highlights
        self._clear_selected_interval_highlights()

        # Check if we have a selected interval
        if not hasattr(self, 'selected_interval') or self.selected_interval is None:
            return

        interval = self.selected_interval

        # Get timestamps for this interval
        try:
            mask = (self.df.index >= interval.start) & (self.df.index <= interval.end)
            selected_timestamps = self.df.index[mask].tolist()
        except Exception:
            return

        if not selected_timestamps:
            return

        # Convert timestamps to indices in the windowed dataframe
        selected_indices = self._timestamps_to_indices(selected_timestamps)

        if not selected_indices:
            return

        # Downsample if too many points (for performance)
        if len(selected_indices) > 2000:
            # Show every Nth point to keep ~1000 markers per axes
            step = len(selected_indices) // 1000
            selected_indices = selected_indices[::step]

        # Create highlights on all user axes (time and position plots)
        for key, ax in self.user_axes.items():
            x_vals, y_vals = self._extract_data_at_indices(ax, selected_indices)

            if len(x_vals) == 0:
                continue  # No data extracted, skip this axes

            try:
                # Create scatter overlay with blue color to distinguish from preview (red)
                scatter = ax.scatter(
                    x_vals, y_vals,
                    c='blue',          # Blue color for selected interval points
                    s=15,              # Slightly smaller than preview markers
                    alpha=0.5,         # Semi-transparent
                    marker='s',        # Square markers to distinguish from preview circles
                    zorder=99,         # Draw below preview highlights
                    edgecolors='darkblue',
                    linewidths=0.3
                )

                # Track this highlight for later removal
                if not hasattr(self, '_interval_highlights'):
                    self._interval_highlights = []
                self._interval_highlights.append(scatter)

            except Exception:
                # Silently fail if scatter creation fails
                continue

    def _clear_selected_interval_highlights(self) -> None:
        """
        Remove all selected interval highlight overlays from axes.

        Called when interval selection changes or is cleared.
        """
        if not hasattr(self, '_interval_highlights'):
            self._interval_highlights = []
            return

        # Remove each highlight artist from its axes
        for artist in self._interval_highlights:
            try:
                artist.remove()
            except Exception:
                pass  # Already removed or axes destroyed

        self._interval_highlights.clear()

    # ========== Helper Methods for Selection Processing ==========

    def _clear_two_click_state(self, *, keep_selection: bool = False) -> None:
        """Stop preview + clear anchor; optionally keep the current preview selection."""
        if getattr(self, "_twoclick_motion_cid", None) is not None and self.canvas is not None:
            try:
                self.canvas.mpl_disconnect(self._twoclick_motion_cid)
            except Exception:
                pass
            self._twoclick_motion_cid = None

        self._pick_anchor_ts = None

        if not keep_selection:
            self.current_selection = None
            self._update_strip()
            if self.canvas is not None:
                self.canvas.draw_idle()

        # also cancel overlay two-click state
        self._two_click_active = False
        self._two_click_t0 = None
        self._two_click_last_x = None
        self._hide_time_overlays()

    def _cancel_active_selection(self) -> None:
        """
        Cancel any active selection - SIMPLE BUT THOROUGH approach.
        """
        # Clear all selection state
        self.current_selection = None
        if hasattr(self, 'current_spans'):
            self.current_spans.clear()
        if hasattr(self, '_commit_spans'):
            self._commit_spans.clear()

        # Clear two-click state
        self._two_click_active = False
        self._two_click_t0 = None
        self._two_click_last_x = None

        # Clear point highlights
        if hasattr(self, '_clear_selected_point_highlights'):
            self._clear_selected_point_highlights()

        # Hide overlays using existing method
        self._hide_time_overlays()

        # Clear strip previews
        self._draw_strip_preview_spans([])

        # Update strip
        self._update_strip()

        # Force canvas redraw
        if hasattr(self, 'canvas') and self.canvas is not None:
            self.canvas.draw_idle()

    def _apply_localized_padding_to_intervals(self, intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Apply localized padding to intervals during preview creation.

        For each interval, computes localized median time difference around start and end points,
        then expands the interval bounds as:
        [time_of_first_pt - local_med_time_diff/2, time_of_last_pt + local_med_time_diff/2]

        Args:
            intervals: List of (start, end) timestamp pairs

        Returns:
            List of padded (start, end) timestamp pairs
        """
        if not intervals:
            return intervals

        padded_intervals = []

        for start_time, end_time in intervals:
            if start_time == end_time:
                # Single-point interval - apply symmetric padding
                local_med_diff = self._compute_localized_median_time_diff(start_time)
                half_diff = local_med_diff / 2

                padded_start = start_time - half_diff
                padded_end = end_time + half_diff
            else:
                # Multi-point interval - compute separate medians for start and end
                start_local_med = self._compute_localized_median_time_diff(start_time)
                end_local_med = self._compute_localized_median_time_diff(end_time)

                padded_start = start_time - (start_local_med / 2)
                padded_end = end_time + (end_local_med / 2)

            # Clamp to data bounds
            padded_start = max(padded_start, self.data_start)
            padded_end = min(padded_end, self.data_end)

            # Ensure start <= end after padding
            if padded_start > padded_end:
                padded_start, padded_end = start_time, end_time

            padded_intervals.append((padded_start, padded_end))

        return padded_intervals

    def _compute_localized_median_time_diff(self, target_time: pd.Timestamp, window_points: int = 50) -> pd.Timedelta:
        """
        Compute localized median time difference using ±window_points around target_time.

        Args:
            target_time: The timestamp around which to compute the local median
            window_points: Number of points on each side (total window = 2 * window_points)

        Returns:
            Median time difference around the target point
        """
        try:
            target_idx = self.df.index.get_indexer([target_time], method="nearest")[0]
            if target_idx < 0:
                return pd.Timedelta(seconds=60)

            start_idx = max(0, target_idx - window_points)
            end_idx = min(len(self.df.index), target_idx + window_points + 1)

            local_times = self.df.index[start_idx:end_idx]

            if len(local_times) < 2:
                return pd.Timedelta(seconds=60)

            time_diffs = []
            for i in range(1, len(local_times)):
                diff = local_times[i] - local_times[i-1]
                time_diffs.append(diff)

            if not time_diffs:
                return pd.Timedelta(seconds=60)

            diff_seconds = [diff.total_seconds() for diff in time_diffs]
            diff_seconds.sort()
            n = len(diff_seconds)

            if n % 2 == 0:
                median_seconds = (diff_seconds[n//2 - 1] + diff_seconds[n//2]) / 2.0
            else:
                median_seconds = diff_seconds[n//2]

            return pd.Timedelta(seconds=median_seconds)

        except Exception:
            return pd.Timedelta(seconds=60)

    def _runs_to_exact_intervals(
        self,
        idx: pd.DatetimeIndex,
        runs: list[tuple[int, int]],   # inclusive index ranges [(i0, i1), ...]
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Convert inclusive index runs to exact [start, end] timestamp pairs for box selections:
          start = time of first included sample
          end   = time of last included sample (NOT the sample after)

        This creates intervals that exactly match what the user visually selected,
        avoiding the boundary issues with half-open intervals for box selections.
        """
        out: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        for i0, i1 in runs:
            s = pd.Timestamp(idx[i0])  # First selected point
            e = pd.Timestamp(idx[i1])  # Last selected point

            # Clamp to data bounds
            s = max(s, self.data_start)
            e = min(e, self.data_end)

            if e >= s:
                out.append((s, e))
        return out

    def _find_contiguous_runs(self, indices: list[int]) -> list[tuple[int, int]]:
        """
        Given sorted indices, return [(start, end), ...] for contiguous runs.
        Both start and end are inclusive.
        """
        if not indices:
            return []

        runs = []
        run_start = indices[0]
        prev = indices[0]

        for i in indices[1:]:
            if i == prev + 1:
                prev = i
            else:
                runs.append((run_start, prev))
                run_start = prev = i

        runs.append((run_start, prev))
        return runs

    def _is_time_lane_axis(self, ax) -> bool:
        """
        Return True if `ax` is a time-series axis in column 0 (the 'time lane').
        In legacy/simple mode (no axes_meta), treat all user axes as time axes.
        """
        if ax is None or ax is self.strip_ax:
            return False

        meta = getattr(self, "axes_meta", None)
        if isinstance(meta, dict) and meta:
            for k, a in self.user_axes.items():
                if a is ax:
                    m = meta.get(k, {})
                    return m.get("role") == "time" and int(m.get("col", 0)) == 0
        return False
