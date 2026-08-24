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

import logging

import pandas as pd
import numpy as np
from ...utils.fasttime import naive_timestamps_from_num

from .base import TOOL_GID_PREFIX
from ...utils.fastindex import positions_exact_then_nearest, positions_nearest

logger = logging.getLogger(__name__)


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

        # A coalesced redraw may still be queued (Pack 5 R4d). Everything
        # below reads what it writes -- the drawn artists and
        # _last_windowed_index -- so render it now instead of mapping this
        # gesture onto the previous window.
        self._flush_pending_redraw()

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

            # CRITICAL: Full-height selection doesn't use component filtering - clear stale values
            self._selected_component_labels = None
            if hasattr(self, 'active_pane') and self.active_pane is not None:
                self.active_pane._selected_component_labels = None

            self.status_var.set(  # type: ignore[union-attr]
                f"Selected: {self.current_selection[0].strftime('%H:%M:%S')} → {self.current_selection[1].strftime('%H:%M:%S')}"
            )
            self._update_strip()

            # Show highlight overlays on selected points
            self._show_selected_point_highlights()

            self.canvas.draw_idle()  # type: ignore[union-attr]
            return

        # === BOX-SELECT path (points-in-rect over time lane axes OR not-time axes) ===
        # Draw decimation is a RENDERING optimisation and may never cost
        # selection accuracy (Pack 5 R11: DRAW-ONLY). Everything below
        # reads the drawn artists, so if this frame was decimated, render
        # it once at full resolution first and scan THAT.
        # This is not belt-and-braces. Measured on the 43k window, a box
        # y-band against a decimated trace versus the raw scan:
        #     band 60% of the axis  recall 1.000  precision 0.934
        #     band 10% of the axis  recall 0.994  precision 0.269
        #     band  3% of the axis  recall 0.687  precision 0.125
        # -- a thin band silently mislabels. One extra full-resolution
        # frame (332 ms at 43k) against a gesture that used to cost 16.1 s
        # is not a trade worth thinking about.
        if getattr(self, "_decim_active", False):
            self._decim_suspend = True
            try:
                self._update_plot()
            finally:
                self._decim_suspend = False

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

            # Half-open commit spans first; the preview derives from them
            # (WYSIWYG: what is highlighted is exactly what will be labeled)
            if self.snap_var.get():
                # Sample-aligned boundaries: [t_first, t_after_last)
                spans_commit = self._exact_spans_to_half_open(exact_intervals)
                spans_preview = [(s, e) for s, e in exact_intervals]
            else:
                # Padded midpoint boundaries that visually wrap the samples
                spans_commit = self._apply_localized_padding_to_intervals(exact_intervals)
                spans_preview = [(s, e) for s, e in spans_commit]

            self.current_selection = None
            self.current_spans = spans_preview
            self._commit_spans = spans_commit

            # CRITICAL: Not-time axis selection doesn't use component filtering - clear stale values
            self._selected_component_labels = None
            if hasattr(self, 'active_pane') and self.active_pane is not None:
                self.active_pane._selected_component_labels = None

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
        # Pack 5 R13: parts are DatetimeIndex chunks, joined once below,
        # instead of a Python list appended to per point.
        picked_parts: list = []
        line_contributions: dict = {}  # Track which line contributed which timestamps

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

                    # Track which line these timestamps came from
                    line_label = ln.get_label()  # e.g., "BX" or "_line0" if no label
                    if line_label.startswith('_'):
                        # Matplotlib auto-labels start with underscore - generate better name
                        line_label = f"Line {len(line_contributions) + 1}"

                    # Convert selected xs (float days) to naive timestamps
                    # -- ONE vectorized conversion (Pack 5 R13), bit-exact
                    # with the per-point mdates.num2date loop it replaces
                    # (measured max |delta| 0 ns on three real frames and
                    # five edge branches), 7,741x faster on the reference
                    # gesture's haul.
                    line_timestamps = naive_timestamps_from_num(xs_sel)
                    picked_parts.append(line_timestamps)

                    # Store which timestamps came from this line
                    if len(line_timestamps):
                        if line_label not in line_contributions:
                            line_contributions[line_label] = []
                        line_contributions[line_label].append(line_timestamps)

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
                    # ONE vectorized conversion (Pack 5 R13), as above.
                    picked_parts.append(naive_timestamps_from_num(xs[m]))
                except Exception:
                    continue

        # Nothing in the box → just clear preview
        # ONE concatenation instead of N appends (Pack 5 R13). The parts
        # are DatetimeIndex chunks; a pandas Index is not truth-testable,
        # so the emptiness gates below become explicit length checks.
        if picked_parts:
            picked_ts = (picked_parts[0] if len(picked_parts) == 1
                         else picked_parts[0].append(picked_parts[1:]))
        else:
            picked_ts = pd.DatetimeIndex([])
        for _label, _parts in list(line_contributions.items()):
            line_contributions[_label] = (
                _parts[0] if len(_parts) == 1
                else _parts[0].append(_parts[1:]))

        if len(picked_ts) == 0:
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

        # NEW: Check if multiple lines contributed
        if len(line_contributions) > 1:
            # Show component selection dialog
            self._show_component_selection_dialog(line_contributions, pane)
            return  # Dialog will call back when user selects

        # CRITICAL: Single-component selection - clear any component filtering
        # This ensures ALL components are highlighted on multi-component plots
        self._selected_component_labels = None
        if hasattr(self, 'active_pane') and self.active_pane is not None:
            self.active_pane._selected_component_labels = None

        # Single line or no line info - proceed with finalization
        self._finalize_box_selection(picked_ts)


    def _on_strip_click(self, event, pane) -> None:
        # Only process events on the active pane
        if pane is not self.active_pane:
            return

        # The artist is only a GATE ("was the click on a band at all?") --
        # the interval is re-derived from the mouse x below, so which
        # artist was hit is never used. Pack 5 R14 draws the strip's bands
        # as ONE PolyCollection, so the gate accepts collections too.
        if (event.artist not in pane.strip_ax.patches  # type: ignore[union-attr]
                and event.artist not in pane.strip_ax.collections):  # type: ignore[union-attr]
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

    def _box_select_on_not_time_axis(self, ax, xlo: float, xhi: float,
                                      ylo: float, yhi: float,
                                      triggered_key: Optional[str] = None) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Given a box on a "not-time" axis (e.g., position plot X-Y), find which points fall inside,
        map them to timestamps via their order in the windowed dataframe, and return time intervals.

        First tries direct dataframe filtering (if x_col/y_col configured),
        then falls back to artist-based extraction for backwards compatibility.

        Returns:
            List of CLOSED (first_sample, last_sample) exact pairs; the
            caller converts these to half-open [start, end) at commit.
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

            # Map to index positions in the FULL dataframe (Pack 5 R1: ONE
            # vectorized get_indexer for the whole haul; bit-exact, and it
            # raises the same ValueError on a non-monotonic index -- which
            # the enclosing try still downgrades to the artist scan)
            idx_full = self.df.index
            pos_in_full = positions_nearest(idx_full, selected_timestamps)

            if not pos_in_full:
                return []

            # Find contiguous runs
            runs = self._find_contiguous_runs(pos_in_full)

            # Convert to exact intervals for box selections (avoids boundary issues)
            intervals = self._runs_to_exact_intervals(idx_full, runs)

            return intervals

        except Exception:
            # Fall back to the artist method -- but RECORD the downgrade:
            # the fallback is the phantom-prone path Pack 3 had to guard.
            # A failure in the good algorithm silently switching to the
            # bad one is Pack 4 A7.
            logger.warning(
                "dataframe box-filter failed; falling back to the "
                "artist scan", exc_info=True)
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
            if str(artist.get_gid() or "").startswith(TOOL_GID_PREFIX):
                continue  # tool-owned overlay (ink), never data (T1)
            try:
                xs = np.asarray(artist.get_xdata(orig=False), dtype=float)
                ys = np.asarray(artist.get_ydata(orig=False), dtype=float)
                if xs.size != ys.size or xs.size == 0:
                    continue
                if xs.size != len(windowed_idx):
                    # The ordinal mapping below (artist point i -> windowed
                    # row i) is only valid for one-point-per-row artists.
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
            if str(artist.get_gid() or "").startswith(TOOL_GID_PREFIX):
                continue  # tool-owned overlay (ink), never data (T1)
            try:
                offsets = np.asarray(artist.get_offsets())
                if (
                    offsets.ndim != 2
                    or offsets.shape[1] != 2
                    or offsets.shape[0] != len(windowed_idx)
                ):
                    # Same ordinal-validity condition as the line scan; also
                    # rejects PolyCollection sentinel offsets ([[0, 0]]).
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

        # Map to index positions in the FULL dataframe (not windowed).
        # Pack 5 R1: ONE vectorized get_indexer, bit-exact with the scalar
        # loop it replaces and raising the same ValueError on a
        # non-monotonic index (this site is unguarded today and stays so).
        idx_full = self.df.index
        pos_in_full = positions_nearest(idx_full, picked_timestamps)

        if not pos_in_full:
            return []

        # Find contiguous runs
        runs = self._find_contiguous_runs(pos_in_full)

        # Convert to exact intervals for box selections (avoids boundary issues)
        return self._runs_to_exact_intervals(idx_full, runs)

    def _on_right_click_cancel(self, event, pane) -> None:
        """
        Handle right-click to cancel active selections or deselect interval.
        Works on any axis (time axes, position axes, strip).
        """
        # Only process events on the active pane
        if pane is not self.active_pane:
            return

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

        ENHANCED: Now detects mouse outside axes in ANY direction (horizontal
        or vertical) using screen coordinate bounding box checks.
        FIXED: Properly handles multi-pane and multi-axes setups.

        Args:
            event: matplotlib mouse motion event
        """
        # Only act if we have an active drag
        drag_axes = getattr(self, '_rect_drag_axes', None)
        if drag_axes is None:
            return

        # CRITICAL FIX: Find which pane this drag belongs to
        # Selectors are stored per-pane, not globally
        drag_pane = None
        drag_key = None

        # Check active pane first (most common case)
        if hasattr(self, 'active_pane'):
            pane = self.active_pane
            if hasattr(pane, 'user_axes'):
                for key, ax in pane.user_axes.items():
                    if ax is drag_axes:
                        drag_pane = pane
                        drag_key = key
                        break

        # If not found on active pane, check all panes
        if drag_pane is None and hasattr(self, 'panes'):
            for pane in self.panes:
                if hasattr(pane, 'user_axes'):
                    for key, ax in pane.user_axes.items():
                        if ax is drag_axes:
                            drag_pane = pane
                            drag_key = key
                            break
                    if drag_pane is not None:
                        break

        # If we couldn't find the pane/key, can't proceed
        if drag_pane is None or drag_key is None:
            return

        # Get the rectangle selector for this specific pane and axes
        if not hasattr(drag_pane, 'rect_selectors'):
            return

        rect_selector = drag_pane.rect_selectors.get(drag_key)
        if rect_selector is None:
            return

        # CRITICAL FIX: Ensure the selector is active
        # Sometimes selectors become inactive during drag - reactivate them
        if not rect_selector.active:
            rect_selector.set_active(True)

        # ENHANCED: Check if mouse is outside axes using bounding box
        # This catches vertical movement outside axes that event.inaxes might miss
        mouse_outside_axes = False

        try:
            bbox = drag_axes.bbox  # Axes bounding box in screen coordinates

            # Check if mouse is outside axes horizontally OR vertically
            if (event.x < bbox.x0 or event.x > bbox.x1 or
                event.y < bbox.y0 or event.y > bbox.y1):
                mouse_outside_axes = True
        except Exception:
            # Fallback to old check if bbox access fails
            mouse_outside_axes = (event.inaxes != drag_axes)

        # If mouse is still fully inside axes, let RectangleSelector handle it normally
        if not mouse_outside_axes and event.inaxes == drag_axes:
            return

        # Mouse is outside axes (horizontally or vertically) - apply clamping
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

            # Pack 6 R1, ported from the shadowed copy in overlays.py
            # before it was deleted. The affine clamp above is WRONG on an
            # inverted axis: with ylim = (4.2, -0.2), max(ymin, min(ymax,
            # y)) collapses to the constant 4.2 no matter where the mouse
            # is, so the rectangle snaps to the wrong edge on every
            # out-of-axes motion event. Adjudicated by executing both
            # arithmetics over 400 mouse positions on four axis kinds:
            # identical on linear-y (0/100) and datetime-x (0/100),
            # different on log-y (3/100, where the non-affine inverse
            # transform returns a garbage x) and inverted-y (40/100).
            # ax.invert_yaxis() is routine for altitude, pressure and
            # energy panels, so this is not an exotic case.
            #
            # Screen-space bbox comparison, so it is immune to whatever the
            # data transform does. Wrapped: if bbox access fails the
            # affine values above still stand.
            try:
                bbox = axes.bbox  # axes bounding box in SCREEN coordinates

                # Mouse above the axes -> the top edge, whichever end of
                # get_ylim() that happens to be.
                if event.y > bbox.y1:
                    y_clamped = ymax
                elif event.y < bbox.y0:
                    y_clamped = ymin

                if event.x > bbox.x1:
                    x_clamped = xmax
                elif event.x < bbox.x0:
                    x_clamped = xmin

            except Exception:
                pass

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
                pane = self.active_pane if hasattr(self, 'active_pane') else self
                blit = getattr(pane, '_blit', None)
                canvas = pane.canvas if hasattr(pane, 'canvas') else getattr(self, 'canvas', None)

                if blit is not None:
                    blit.draw([rect_patch])
                elif canvas is not None:
                    # Fallback to axes-specific blit (still faster than full redraw)
                    axes.draw_artist(rect_patch)
                    canvas.blit(axes.bbox)

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

        # Build the windowed view this extraction reads.
        #
        # Pack 6 D11. There used to be a `_last_windowed_df` read above
        # this, and a guard on it. NOTHING in the tree ever assigned that
        # name, so the guard was unconditionally true and the block below
        # was the only path that ever ran -- which is why deleting the
        # GUARD as well as the read is what preserves behaviour.
        #
        # Keeping the guard on `_last_windowed_index` alone would NOT be
        # equivalent, and the suite cannot see the difference: that name IS
        # written (plotting.py:342), so the guard would be FALSE in the
        # normal case, this block would be skipped, `windowed_df` would be
        # unbound, and the NameError at the role == "not-time" branch below
        # would be swallowed by its own `except Exception`. Measured on a
        # real two-axis app: cross-plot highlights drop from 5 points to 0
        # with the suite still green.
        try:
            windowed_df = self.df.loc[self.t0:self.t1].copy()
            windowed_idx = windowed_df.index
            if windowed_df.empty:
                return x_vals, y_vals
        except Exception:
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

                # CRITICAL: Check component filter on correct object
                # Priority: active_pane (if exists) > self
                selected_components = None

                if hasattr(self, 'active_pane') and self.active_pane is not None:
                    selected_components = getattr(self.active_pane, '_selected_component_labels', None)
                else:
                    selected_components = getattr(self, '_selected_component_labels', None)

                # For time axes, we need to determine which column is being plotted
                # Filter by selected components if specified
                for line in ax.lines:
                    try:
                        ys = np.asarray(line.get_ydata(orig=False))
                        if len(ys) != len(windowed_idx):  # Not main data artist
                            continue

                        # If no filter (None), include ALL lines
                        if selected_components is None:
                            # No filtering - extract from this line
                            pass
                        else:
                            # Apply component filtering
                            line_label = line.get_label()

                            # Handle auto-generated labels (matplotlib creates these)
                            if line_label.startswith('_'):
                                # Include auto-labeled lines by default for backward compatibility
                                pass
                            else:
                                # Normalize labels for comparison (strip whitespace, case-insensitive)
                                line_label_normalized = line_label.strip().upper()
                                selected_labels_normalized = [lbl.strip().upper() for lbl in selected_components]

                                # Exact match required
                                if line_label_normalized not in selected_labels_normalized:
                                    # This line is not in the selected components - skip it
                                    continue

                        y_vals_for_line = [float(ys[idx]) for idx in indices
                                          if 0 <= idx < len(ys)]
                        x_vals_for_line = time_vals[:len(y_vals_for_line)]  # Match lengths

                        # Append these values
                        x_vals.extend(x_vals_for_line)
                        y_vals.extend(y_vals_for_line)

                        # FIXED: Process ALL lines when no filter, not just first one
                        # This ensures all components are highlighted on multi-component plots

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
                    # Fallback: extract from the first artist whose length
                    # matches the windowed dataframe.  Cross-plots are drawn
                    # with ax.scatter() (PathCollection on ax.collections)
                    # rather than ax.plot() (Line2D on ax.lines), so iterate
                    # both.  The length filter avoids picking up our own
                    # highlight overlay (which has len == len(selected)).
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
                    else:
                        # No matching Line2D -- try PathCollections (scatter)
                        for artist in ax.collections:
                            try:
                                offsets = np.asarray(artist.get_offsets())
                                if (
                                    offsets.ndim != 2
                                    or offsets.shape[1] != 2
                                    or offsets.shape[0] != len(windowed_idx)
                                ):
                                    continue
                                for idx in indices:
                                    if 0 <= idx < offsets.shape[0]:
                                        x_vals.append(float(offsets[idx, 0]))
                                        y_vals.append(float(offsets[idx, 1]))
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
        # Check if point highlighting is enabled (performance optimization)
        if not getattr(self, 'enable_point_highlighting', True):
            return

        # Clear any existing highlights first
        self._clear_selected_point_highlights()

        # Get selected timestamps from current preview
        selected_timestamps = self._get_preview_timestamps()

        if not selected_timestamps:
            return

        # Downsample if too many points (for performance).
        # Pack 5 R2: the cap now runs BEFORE the mapping. It used to sit
        # one step too late -- the per-probe get_loc loop had already run
        # over the FULL preview set, so the 2000-marker guard protected
        # the scatter call and not the loop that fed it (pack5_g1 S6).
        if len(selected_timestamps) > 2000:
            # Show every Nth point to keep ~1000 markers per axes
            step = len(selected_timestamps) // 1000
            selected_timestamps = selected_timestamps[::step]

        # Convert timestamps to indices in the windowed dataframe
        selected_indices = self._timestamps_to_indices(selected_timestamps)

        if not selected_indices:
            return

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
                # Name-tag as tool ink so artist scans skip it (T1)
                scatter.set_gid(TOOL_GID_PREFIX + "preview-highlight")

                # Track this highlight for later removal
                if not hasattr(self, '_preview_highlights'):
                    self._preview_highlights = []
                self._preview_highlights.append(scatter)

            except Exception:
                # Silently fail if scatter creation fails
                continue

        # Redraw canvas to show highlights (unless caller will handle it)
        if redraw:
            # Use active pane's canvas for multi-pane support
            canvas = self.active_pane.canvas if hasattr(self, 'active_pane') else self.canvas
            if canvas is not None:
                canvas.draw_idle()

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

        WYSIWYG (Pack 3): derived from the COMMIT-equivalent spans with the
        same half-open [start, end) mask the export uses, so the highlighted
        samples are exactly the in-window samples the _add_interval door
        would label. Strip drag is the one exception: _preview_selection
        hands us an interval's ALREADY-exclusive end, so the dots there show
        one sample more than a ResizeIntervalCommand stores (unchanged from
        today; parked with S8).

        Returns:
            List of pd.Timestamp objects representing selected times
        """
        timestamps = []

        commit_spans = getattr(self, '_commit_spans', None) or []
        if not commit_spans and getattr(self, 'current_selection', None):
            # Single-span flows commit current_selection through the same
            # conversion (_add_interval door); mirror it here.
            commit_spans = self._exact_spans_to_half_open([self.current_selection])

        # Clip to the visible window: _timestamps_to_indices maps missing
        # timestamps to the NEAREST windowed row, so out-of-window samples
        # (e.g. full-range rule commits) must not reach the highlighter.
        lo = getattr(self, 't0', None)
        hi = getattr(self, 't1', None)

        # Pack 5 R2: clip to the window ONCE, then mask inside it. The old
        # shape rebuilt three to four FULL-LENGTH boolean masks over
        # self.df.index PER SPAN -- O(spans x N), measured at 1.5e-8 s per
        # (span x row), i.e. ~2.3 s for 100 spans on the 1.46M-row frame
        # (pack5_g1 3e). Same rows, same order: the window clip is exactly
        # the mask term that was ANDed in per span. Boolean masks (not
        # searchsorted) are deliberate -- they stay correct on a
        # NON-MONOTONIC index, where searchsorted silently mislabels.
        base_idx = self.df.index
        if lo is not None and hi is not None:
            try:
                base_idx = base_idx[(base_idx >= lo) & (base_idx <= hi)]
            except Exception:
                base_idx = self.df.index

        for start_ts, end_ts in commit_spans:
            try:
                mask = (base_idx >= start_ts) & (base_idx < end_ts)
                timestamps.extend(base_idx[mask].tolist())
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

        # Pack 5 R2: one exact get_indexer for the whole list, then ONE
        # nearest pass over the probes that missed -- the same two-step
        # ladder the per-probe loop ran, in two calls instead of 2N. A
        # NON-UNIQUE index keeps the scalar ladder inside the helper:
        # get_loc returns a slice there (whose .start this code took) and
        # get_indexer refuses outright, so vectorizing it would turn a
        # working answer into an exception.
        return positions_exact_then_nearest(windowed_idx, timestamps)

    # ========== Selected Interval Point Highlighting ==========

    def _show_selected_interval_highlights(self) -> None:
        """
        Highlight points for the currently selected interval with blue markers.

        Similar to preview highlighting but uses a different color (blue vs red)
        and works on the selected interval rather than preview selection.
        """
        # Check if point highlighting is enabled (performance optimization)
        if not getattr(self, 'enable_point_highlighting', True):
            return

        # Clear any existing interval highlights
        self._clear_selected_interval_highlights()

        # Check if we have a selected interval
        if not hasattr(self, 'selected_interval') or self.selected_interval is None:
            return

        interval = self.selected_interval

        # Get timestamps for this interval (half-open, matches iv.contains)
        try:
            mask = (self.df.index >= interval.start) & (self.df.index < interval.end)
            selected_timestamps = self.df.index[mask].tolist()
        except Exception:
            return

        if not selected_timestamps:
            return

        # Downsample if too many points (for performance) -- BEFORE the
        # mapping, same reorder and same reason as the preview
        # highlighter above (Pack 5 R2, pack5_g1 S6).
        if len(selected_timestamps) > 2000:
            # Show every Nth point to keep ~1000 markers per axes
            step = len(selected_timestamps) // 1000
            selected_timestamps = selected_timestamps[::step]

        # Convert timestamps to indices in the windowed dataframe
        selected_indices = self._timestamps_to_indices(selected_timestamps)

        if not selected_indices:
            return

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
                # Name-tag as tool ink so artist scans skip it (T1)
                scatter.set_gid(TOOL_GID_PREFIX + "interval-highlight")

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
        self._hide_time_overlays()

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

            # Clamp to data bounds. The end cap sits 1ns PAST data_end so a
            # tail interval still labels the final sample under half-open
            # [start, end) semantics (T4).
            padded_start = max(padded_start, self.data_start)
            padded_end = min(padded_end, self._end_after_inclusive(self.data_end))

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
            # This guess pads committed span boundaries; on a 3 s cadence
            # it inflates every padded edge 20x (Pack 4 A8).
            logger.warning(
                "localized median cadence failed; guessing 60 s",
                exc_info=True)
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

        These closed pairs are DISPLAY/geometry values. Commit paths convert
        them to half-open [start, end) via _exact_spans_to_half_open (snap)
        or _apply_localized_padding_to_intervals (padded) -- Pack 3, WYSIWYG.
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

    def _finalize_box_selection(self, picked_ts: list[pd.Timestamp]) -> None:
        """
        Finalize box selection with given timestamps.

        Converts timestamps to intervals, applies padding and snapping,
        updates previews, and shows highlights.

        Args:
            picked_ts: List of pd.Timestamp objects to create intervals from
        """
        # Convert timestamps to index positions (nearest) and keep only those inside current data bounds
        # Pack 5 R1: ONE vectorized get_indexer instead of one call per
        # picked point. This is the site that owned 91-98% of the gesture
        # (measured: 47.7 s of a 48.5 s drag on the 1.46M-row frame), and
        # the one whose probes genuinely need method="nearest" -- they came
        # back through mdates.num2date a median 484 ns off the true sample.
        idx_full = self.df.index
        pos = positions_nearest(idx_full, picked_ts)

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

        # Half-open commit spans first; the preview derives from them
        # (WYSIWYG: what is highlighted is exactly what will be labeled)
        if self.snap_var.get():  # type: ignore[union-attr]
            # Sample-aligned boundaries: [t_first, t_after_last)
            spans_commit = self._exact_spans_to_half_open(exact_intervals)
            spans_preview = [(s, e) for s, e in exact_intervals]
        else:
            # Padded midpoint boundaries that visually wrap the samples
            spans_commit = self._apply_localized_padding_to_intervals(exact_intervals)
            spans_preview = [(s, e) for s, e in spans_commit]

        # Stash both: preview for drawing, commit for "Add Label"
        self.current_selection = None
        self.current_spans = spans_preview        # closed display spans for highlighting
        self._commit_spans = spans_commit         # half-open spans for interval creation

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

        self.canvas.draw_idle()  # type: ignore[union-attr]

    def _show_component_selection_dialog(
        self,
        line_contributions: dict[str, list[pd.Timestamp]],
        pane
    ) -> None:
        """
        Show dialog for user to choose which line/component to select.

        Creates a modal dialog with buttons for each component found in the
        dragbox selection, plus "All" and "Cancel" options. When user selects
        a component, calls _finalize_box_selection with only that component's
        timestamps.

        Args:
            line_contributions: Dict mapping line labels to list of timestamps
            pane: The TabPane where selection occurred
        """
        import tkinter as tk
        from tkinter import ttk

        # Get root window - handle multi-pane structure
        root = getattr(self, 'root', None) or getattr(self, 'master', None)
        if root is None and hasattr(self, 'canvas') and self.canvas is not None:
            root = self.canvas.get_tk_widget().winfo_toplevel() if hasattr(self.canvas, 'get_tk_widget') else None

        if root is None:
            # Fallback: just select all if we can't show dialog
            # Pack 5 R13: the values are DatetimeIndex chunks now.
            parts = list(line_contributions.values())
            all_timestamps = (parts[0] if len(parts) == 1
                              else parts[0].append(parts[1:]))
            self._finalize_box_selection(all_timestamps)
            return

        # Create dialog window
        dialog = tk.Toplevel(root)
        dialog.title("Select Component")
        dialog.geometry("380x350")
        dialog.transient(root)
        dialog.grab_set()  # Modal dialog

        # Center dialog on screen
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        # Header
        header = ttk.Label(
            dialog,
            text="Multiple components in selection:",
            font=('Arial', 11, 'bold')
        )
        header.pack(pady=(15, 10))

        # Info label
        info = ttk.Label(
            dialog,
            text="Select which component to label:",
            font=('Arial', 9)
        )
        info.pack(pady=(0, 10))

        # Component buttons frame
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=10)

        # Sort components by number of points (most first) for better UX
        sorted_components = sorted(
            line_contributions.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        def on_component_select(component_label):
            """User selected a specific component."""
            dialog.destroy()
            # Get timestamps for this component only
            selected_timestamps = line_contributions[component_label]

            # CRITICAL: Store which component was selected for highlighting
            self._selected_component_labels = [component_label]

            # CRITICAL: Also store on active pane for multi-pane compatibility
            if hasattr(self, 'active_pane'):
                self.active_pane._selected_component_labels = [component_label]

            self._finalize_box_selection(selected_timestamps)

        def on_select_all():
            """User wants all components (intersection only)."""
            dialog.destroy()

            # Use INTERSECTION of timestamps (same logic as for dialog count)
            idx_full = self.df.index

            # Convert each component's timestamps to dataframe indices
            # (Pack 5 R1: ONE vectorized get_indexer per COMPONENT instead
            # of one per point. The intersection stays position-based --
            # duplicate timestamps would change the answer otherwise.)
            component_index_sets = []
            for component_label, ts_list in line_contributions.items():
                try:
                    component_indices = set(
                        positions_nearest(idx_full, ts_list))
                except Exception:
                    continue
                if component_indices:
                    component_index_sets.append(component_indices)

            # Compute intersection: only indices present in ALL components
            if len(component_index_sets) > 0:
                common_indices = component_index_sets[0]
                for idx_set in component_index_sets[1:]:
                    common_indices = common_indices & idx_set

                # Convert intersection indices back to timestamps
                intersection_timestamps = [idx_full[i] for i in sorted(common_indices)]
            else:
                intersection_timestamps = []

            # CRITICAL: Store that we want ALL components highlighted
            self._selected_component_labels = list(line_contributions.keys())

            # CRITICAL: Also store on active pane for multi-pane compatibility
            if hasattr(self, 'active_pane'):
                self.active_pane._selected_component_labels = list(line_contributions.keys())

            self._finalize_box_selection(intersection_timestamps)

        def on_cancel():
            """User cancelled - clear selection."""
            dialog.destroy()
            self._cancel_active_selection()

        # Create button for each component
        row = 0
        col = 0
        for component_label, timestamps in sorted_components:
            point_count = len(timestamps)
            btn_text = f"{component_label}\n({point_count} pts)"

            btn = ttk.Button(
                btn_frame,
                text=btn_text,
                width=12,
                command=lambda label=component_label: on_component_select(label)
            )
            btn.grid(row=row, column=col, padx=5, pady=5)

            col += 1
            if col >= 3:  # 3 buttons per row
                col = 0
                row += 1

        # Separator
        separator = ttk.Separator(dialog, orient='horizontal')
        separator.pack(fill='x', padx=20, pady=10)

        # "All" button - count INTERSECTION of timestamps across all components
        # This gives timestamps where ALL components have selected points
        idx_full = self.df.index

        # Convert each component's timestamps to a set of dataframe indices
        # (Pack 5 R1, and this loop is the one that runs UNCONDITIONALLY at
        # dialog BUILD time purely to render the "All Components (N pts)"
        # caption -- measured 27.7 s at 43k points, paid even by a user who
        # only ever intends to click BX. pack5_g1 1d/S5.)
        component_index_sets = []
        for component_label, ts_list in line_contributions.items():
            try:
                component_indices = set(positions_nearest(idx_full, ts_list))
            except Exception:
                continue
            if component_indices:  # Only add non-empty sets
                component_index_sets.append(component_indices)

        # Compute intersection: only indices present in ALL components
        if len(component_index_sets) > 0:
            # Start with first set, intersect with all others
            common_indices = component_index_sets[0]
            for idx_set in component_index_sets[1:]:
                common_indices = common_indices & idx_set  # Set intersection
            unique_count = len(common_indices)
        else:
            unique_count = 0

        all_btn = ttk.Button(
            dialog,
            text=f"All Components ({unique_count} pts)",
            width=30,
            command=on_select_all
        )
        all_btn.pack(pady=5)

        # Cancel button
        cancel_btn = ttk.Button(
            dialog,
            text="Cancel",
            width=30,
            command=on_cancel
        )
        cancel_btn.pack(pady=(5, 15))

        # Bind Escape key to cancel
        dialog.bind('<Escape>', lambda e: on_cancel())

        # Store dialog reference (in case we need to close it programmatically)
        self._component_dialog = dialog
