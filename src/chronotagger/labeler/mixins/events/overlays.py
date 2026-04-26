"""
Overlay and highlighting management for event handling.

Part of the ChronoTagger event handling system.
Handles all visual feedback including preview bands, highlights, and markers.
"""

from __future__ import annotations
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.transforms import blended_transform_factory
from matplotlib.collections import PolyCollection
import pandas as pd
import numpy as np


class OverlaysMixin:
    """
    Handles overlay rendering and visual feedback.

    This mixin is part of the EventsMixin composition and provides
    all overlay-related functionality including preview bands, point
    highlights, and multi-span visualizations.
    """

    def _update_time_overlays(self, x0: float, x1: float, color: str = "tab:orange") -> None:
        """
        Move/resize the animated preview band on each time-lane axes and blit only those.
        x0/x1 are Matplotlib date floats.

        Args:
            x0, x1: Time range in matplotlib date format
            color: Color for the overlay ("yellow" for dragbox, "tab:orange" for two-click)

        Skip overlays only if RectangleSelector is actively being dragged by user.
        """
        # Ensure overlay system exists
        if not getattr(self, "_time_overlays", None):
            self._init_time_overlays()

        if not getattr(self, "_time_overlays", None):
            return  # Still failed to create

        left = min(x0, x1)
        width = max(abs(x1 - x0), 0.0)
        artists = []

        # Only skip overlays if RectangleSelector is actively being dragged
        for ax, r in self._time_overlays.items():
            ax_key = self._find_axes_key(ax)

            # Skip axes only if RectangleSelector is actively being used
            should_skip = False
            if hasattr(self, 'rect_selectors') and hasattr(self, '_drag_active'):
                if ax_key and ax_key in self.rect_selectors:
                    rect_sel = self.rect_selectors[ax_key]
                    is_active = getattr(rect_sel, 'active', False)
                    is_dragging = getattr(self, '_drag_active', False)  # Only skip if actually dragging
                    if is_active and is_dragging:
                        should_skip = True

            if should_skip:
                continue  # Skip this axis to avoid conflicts

            r.set_xy((left, 0))
            r.set_width(width)
            # Set the correct color for this selection type
            r.set_facecolor(color)
            if not r.get_visible():
                r.set_visible(True)
            artists.append(r)

        # Use active pane's BlitHelper for fast rendering
        pane = self.active_pane if hasattr(self, 'active_pane') else self
        blit = getattr(pane, "_blit", None)
        if blit is not None and artists:
            blit.draw(artists)
        else:
            # graceful fallback
            canvas = pane.canvas if hasattr(pane, 'canvas') else getattr(self, 'canvas', None)
            if canvas is not None:
                canvas.draw_idle()

    def _update_time_overlays_for_multi_spans(self, spans: list[tuple[pd.Timestamp, pd.Timestamp]]) -> None:
        """
        Update time overlays to show multiple time spans (for box selections).
        Uses PolyCollection for efficient rendering of many spans.
        Uses YELLOW color to distinguish from orange two-click selections.

        Performance: PolyCollection renders all spans in a single draw call,
        giving 10-30x speedup over individual Rectangle patches.

        Args:
            spans: List of (start, end) timestamp pairs
        """
        if not spans:
            self._hide_time_overlays()
            return

        # Ensure overlay system exists
        if not getattr(self, "_time_overlays", None):
            self._init_time_overlays()

        if not getattr(self, "_time_overlays", None):
            return  # Still failed to create

        # Get list of time axes
        axes = []
        if getattr(self, "_time_axis_keys", None):
            for k in self._time_axis_keys:
                ax = self.user_axes.get(k)
                if ax is not None:
                    axes.append(ax)
        if getattr(self, "strip_ax", None) is not None:
            axes.append(self.strip_ax)

        if not axes:
            return

        # Initialize PolyCollection storage if needed
        if not hasattr(self, '_multi_span_overlay_collections'):
            self._multi_span_overlay_collections = {}

        artists = []

        for ax in axes:
            # Build vertices for all spans on this axis
            vertices = self._build_span_vertices(ax, spans)

            if not vertices:
                continue

            # Get or create PolyCollection for this axis
            if ax not in self._multi_span_overlay_collections:
                poly = PolyCollection(
                    vertices,
                    facecolors='yellow',
                    edgecolors='none',
                    alpha=0.25,
                    zorder=ax.get_zorder() + 10,
                )
                # NOTE: NOT using set_animated(True) - PolyCollections blit unreliably
                # Better to use normal rendering for 100% reliability
                ax.add_collection(poly)
                self._multi_span_overlay_collections[ax] = poly
            else:
                # Update existing PolyCollection with new vertices
                poly = self._multi_span_overlay_collections[ax]
                poly.set_verts(vertices)
                poly.set_visible(True)

            artists.append(poly)

        # Use normal rendering (draw_idle) for reliability
        # Blitting PolyCollections is fast but unreliable - causes intermittent disappearance
        if getattr(self, "canvas", None) is not None:
            self.canvas.draw_idle()

    def _build_span_vertices(self, ax, spans: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list:
        """
        Build rectangle vertices for PolyCollection from time spans.

        Each span becomes a rectangle with:
        - x: span start/end times (data coordinates)
        - y: full axis height (data coordinates)

        Args:
            ax: Matplotlib axes to build rectangles for
            spans: List of (start, end) timestamp pairs

        Returns:
            List of vertex arrays, one per span rectangle.
            Each vertex array is [(x0,y0), (x1,y0), (x1,y1), (x0,y1)].
        """
        # Get current y-axis limits in data coordinates
        try:
            ymin, ymax = ax.get_ylim()
        except Exception:
            return []  # Axes not ready

        vertices = []
        for start_ts, end_ts in spans:
            # Convert timestamps to matplotlib date numbers
            x0 = mdates.date2num(start_ts)
            x1 = mdates.date2num(end_ts)

            # Ensure proper ordering
            left = min(x0, x1)
            right = max(x0, x1)

            # Build rectangle vertices (counter-clockwise from bottom-left)
            rect_verts = [
                (left, ymin),   # Bottom-left
                (right, ymin),  # Bottom-right
                (right, ymax),  # Top-right
                (left, ymax),   # Top-left
            ]
            vertices.append(rect_verts)

        return vertices

    def _hide_multi_span_overlays(self) -> None:
        """
        Hide PolyCollection overlays (called during y-axis zoom).

        This is a lightweight operation that just sets visibility to False.
        Overlays can be restored by calling _update_time_overlays_for_multi_spans()
        with the current spans.
        """
        if not hasattr(self, '_multi_span_overlay_collections'):
            return

        for ax, poly in self._multi_span_overlay_collections.items():
            try:
                if poly.get_visible():
                    poly.set_visible(False)
            except Exception:
                continue  # Collection may have been removed

    def _restore_multi_span_overlays(self) -> None:
        """
        Restore PolyCollection overlays after y-axis zoom completes.

        This rebuilds the vertices with current y-axis limits and makes
        the overlays visible again. Only does work if there are active spans.
        """
        # Only restore if we have active spans
        if not hasattr(self, 'current_spans') or not self.current_spans:
            return

        # Rebuild overlays with current y-limits
        self._update_time_overlays_for_multi_spans(self.current_spans)

    def on_ylim_change(self) -> None:
        """
        PUBLIC API: Call this when y-axis limits change (zoom/pan).

        This hides multi-span overlays during the zoom operation.
        To restore overlays after zoom completes, call on_ylim_change_complete().

        Usage example (if you have zoom/pan event handlers):

            def my_ylim_callback(event):
                self.on_ylim_change()  # Hide overlays during zoom

            def my_zoom_complete():
                self.on_ylim_change_complete()  # Restore overlays

        NOTE: Single-span overlays (two-click, full-height dragbox) use
        blended transforms and don't need updating during y-zoom.
        Only multi-span overlays (box selections) need this handling.
        """
        self._hide_multi_span_overlays()

    def on_ylim_change_complete(self) -> None:
        """
        PUBLIC API: Call this when y-axis zoom/pan completes.

        This restores multi-span overlays with updated y-coordinates.
        Safe to call even if no overlays are active.
        """
        self._restore_multi_span_overlays()

    def _hide_time_overlays(self) -> None:
        """
        Hide all time overlay rectangles and PolyCollections with aggressive cleanup.
        Searches for orphaned overlays across all axes.
        """
        changed = []

        # First, hide all tracked single-span overlays (Rectangle patches)
        if hasattr(self, '_time_overlays') and self._time_overlays:
            for ax, r in self._time_overlays.items():
                try:
                    if r.get_visible():
                        r.set_visible(False)
                        changed.append(r)
                except Exception:
                    continue

        # Hide all multi-span overlays (PolyCollections)
        if hasattr(self, '_multi_span_overlay_collections') and self._multi_span_overlay_collections:
            for ax, poly in self._multi_span_overlay_collections.items():
                try:
                    if poly.get_visible():
                        poly.set_visible(False)
                        changed.append(poly)
                except Exception:
                    continue

        # Aggressive cleanup: Search ALL axes for overlay-like rectangles
        all_axes = []
        if hasattr(self, 'user_axes'):
            all_axes.extend(self.user_axes.values())
        if hasattr(self, 'strip_ax') and self.strip_ax is not None:
            all_axes.append(self.strip_ax)

        for ax in all_axes:
            try:
                # Look for any rectangle that could be an overlay
                patches_to_hide = []
                for patch in ax.patches:
                    try:
                        # Check if this looks like one of our overlay rectangles
                        is_overlay = (
                            hasattr(patch, 'get_animated') and
                            hasattr(patch, 'get_facecolor') and
                            hasattr(patch, 'get_alpha') and
                            patch.get_visible()
                        )

                        if is_overlay:
                            # Additional checks to identify our overlays
                            alpha = patch.get_alpha()
                            is_animated = patch.get_animated()

                            # Our overlays have alpha=0.25 and are animated
                            if alpha == 0.25 and is_animated:
                                patches_to_hide.append(patch)
                    except Exception:
                        continue

                # Hide found overlay patches
                for patch in patches_to_hide:
                    try:
                        patch.set_visible(False)
                        changed.append(patch)
                    except Exception:
                        continue

            except Exception:
                continue

        # If no overlays found, we're done
        if not changed:
            return

        # Use active pane's BlitHelper for fast rendering
        pane = self.active_pane if hasattr(self, 'active_pane') else self
        blit = getattr(pane, "_blit", None)
        if blit is not None:
            try:
                blit.draw(changed)
            except Exception:
                # Fallback to full redraw if blitting fails
                canvas = pane.canvas if hasattr(pane, 'canvas') else getattr(self, 'canvas', None)
                if canvas is not None:
                    canvas.draw_idle()
        else:
            # Graceful fallback
            canvas = pane.canvas if hasattr(pane, 'canvas') else getattr(self, 'canvas', None)
            if canvas is not None:
                canvas.draw_idle()

    def _init_time_overlays(self) -> None:
        """
        Create/refresh translucent preview bands on every time axis plus the strip.
        Mark them animated so we can blit them cheaply.

        CRITICAL: Does NOT automatically restore overlays - only creates fresh hidden ones.
        """
        # Always start completely fresh
        self._time_overlays = {}
        self._multi_span_overlay_collections = {}  # Reset PolyCollection dict
        self._two_click_active = False
        self._two_click_t0 = None
        self._two_click_last_x = None

        axes = []
        # Include ALL time axes (not just time-lane axes in column 0)
        if getattr(self, "_time_axis_keys", None):
            for k in self._time_axis_keys:
                ax = self.user_axes.get(k)
                if ax is not None:
                    axes.append(ax)
        if getattr(self, "strip_ax", None) is not None:
            axes.append(self.strip_ax)

        for ax in axes:
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            r = mpatches.Rectangle(
                (0, 0), 0, 1,
                transform=trans,
                facecolor="yellow",  # Default color
                edgecolor="none",
                alpha=0.25,
                zorder=ax.get_zorder() + 10,
                visible=False,  # Always start hidden - NO automatic restoration
            )
            r.set_animated(True)  # <- critical for blitting
            ax.add_patch(r)
            self._time_overlays[ax] = r

        # also prep a (reusable) pool of strip preview rectangles for multi-span previews
        self._strip_preview_pool = []  # created lazily when needed

        # DO NOT restore overlays automatically - only when explicitly requested

    def _restore_current_selection_overlays(self) -> None:
        """
        Restore overlay display for current selection after overlays are recreated.
        Only restores if there's actually an active selection.
        """
        # Only restore if we have actual active selections
        has_active_selection = (
            (hasattr(self, 'current_spans') and self.current_spans) or
            (hasattr(self, 'current_selection') and self.current_selection is not None) or
            getattr(self, "_two_click_active", False)
        )

        if not has_active_selection:
            return  # No active selection, don't restore anything

        # Check for multi-span selection (dragbox) - use YELLOW
        if hasattr(self, 'current_spans') and self.current_spans:
            self._update_time_overlays_for_multi_spans(self.current_spans)
            return

        # Check for single-span selection
        if hasattr(self, 'current_selection') and self.current_selection:
            start_ts, end_ts = self.current_selection
            x0 = mdates.date2num(start_ts)
            x1 = mdates.date2num(end_ts)

            # Use YELLOW for completed selections, ORANGE only for active two-click motion
            color = "yellow"  # Default to yellow for all completed selections
            if getattr(self, "_two_click_active", False):
                color = "tab:orange"  # Only during active two-click motion

            self._update_time_overlays(x0, x1, color=color)
            return

    def _apply_snap_clamp(self, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Apply snapping (if enabled), clamp to data, and enforce min duration."""
        s, e = (start, end) if start <= end else (end, start)

        # Clamp to dataset extent
        s = max(s, self.data_start)
        e = min(e, self.data_end)

        # Enforce min duration
        if (e - s) < self.min_duration:
            e = s + self.min_duration
            if e > self.data_end:
                # shift left if we fell off the right
                s = max(self.data_start, e - self.min_duration)

        # Optional snapping (use current window to avoid large jumps)
        if self.snap_var is not None and self.snap_var.get():
            # Snap independently; ensure ordering after snap
            ss, ee = self._snap_to_samples(s, e)
            s, e = (ss, ee) if ss <= ee else (ee, ss)

        return s, e

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

    def _clamp_rectangle_to_axes(self, event, axes, rect_selector) -> None:
        """
        Clamp rectangle selection to axes bounds when mouse leaves axes.

        Transforms the mouse position (in figure coordinates) to data
        coordinates, clamps to axes limits, and manually updates the
        rectangle selector extents using fast blitting.

        ENHANCED: Aggressive edge snapping when mouse is outside axes bounds.
        Detects which edge(s) the mouse is beyond and snaps to those edges.

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

            # ENHANCED: Detect which edges mouse is outside and snap aggressively
            # This ensures vertical (Y-axis) clamping works as expected
            try:
                bbox = axes.bbox  # Axes bounding box in screen coordinates

                # If mouse is above axes, force top edge
                if event.y > bbox.y1:
                    y_clamped = ymax

                # If mouse is below axes, force bottom edge
                elif event.y < bbox.y0:
                    y_clamped = ymin

                # If mouse is right of axes, force right edge
                if event.x > bbox.x1:
                    x_clamped = xmax

                # If mouse is left of axes, force left edge
                elif event.x < bbox.x0:
                    x_clamped = xmin

            except Exception:
                # If edge detection fails, use the clamped values from above
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
                pane = self.active_pane if hasattr(self, 'active_pane') else self
                canvas = pane.canvas if hasattr(pane, 'canvas') else getattr(self, 'canvas', None)
                extents = [left, right, bottom, top]
                rect_selector.extents = extents
                if canvas is not None:
                    canvas.draw_idle()

        except Exception:
            # Silently fail - better to have normal behavior than crash
            # This can happen if coordinate transforms are invalid
            pass

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
                    # matches the windowed dataframe.  Cross-plots use
                    # ax.scatter() (PathCollection on ax.collections), not
                    # ax.plot() (Line2D on ax.lines), so iterate both.  The
                    # length filter avoids picking up our own highlight
                    # overlay scatter.
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
