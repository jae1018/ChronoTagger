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
