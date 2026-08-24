"""
Mouse event handlers mixin.

Responsibilities:
- Two-click time selection with blitted preview
- Mouse motion tracking during two-click selection
- Strip drag/resize/move interactions
- Coordinate transformations (_px_to_data_dx, _x_from_anywhere, _ts_from_event)
- Hit testing for strip interval edges
- Cursor management during drag operations

Note: Rectangle selection and right-click cancellation are handled by SelectionMixin.
"""

from __future__ import annotations

from typing import Optional, Tuple
import matplotlib.dates as mdates
import pandas as pd


HANDLE_PX = 8  # hit tolerance in screen pixels for edge resize


class MouseEventsMixin:
    def _px_to_data_dx(self, ax, px=2) -> float:
        """Return the data-domain x-width that corresponds to ~px screen pixels."""
        inv = ax.transData.inverted()
        # use the vertical center of the axis for stable mapping
        x0 = ax.bbox.x0 + ax.bbox.width * 0.5
        ymid = ax.bbox.y0 + ax.bbox.height * 0.5
        (x_data0, _y0) = inv.transform((x0, ymid))
        (x_data1, _y1) = inv.transform((x0 + float(px), ymid))
        return abs(x_data1 - x_data0) or 1e-12


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


    def _on_time_click(self, event, pane) -> None:
        """
        Two-click selection with blitted preview (canvas-wide).
          • Left-click #1 arms at t0 and shows slim band across time-lane panels + strip.
          • Left-click #2 finalizes [t0, t1] and keeps the preview visible (no full redraw).
          • Right-click cancels.
        """
        # Only process events on the active pane
        if pane is not self.active_pane:
            return

        import pandas as pd, matplotlib.dates as mdates

        btn = getattr(event, "button", None)

        # Right-click cancels - use the new comprehensive cancellation system
        if btn == 3:
            # Check if we have any active selection (not just two-click)
            has_selection = (
            getattr(self, "_pick_anchor_ts", None) is not None or
            getattr(self, "current_selection", None) is not None or
            bool(getattr(self, "current_spans", None)) or
            bool(getattr(self, "_commit_spans", None))
            )

            if has_selection:
                self._cancel_active_selection()
                if self.status_var is not None:
                    self.status_var.set("Selection canceled")
            return

        if btn != 1:
            return

        # ---- ignore if this mouse cycle was a drag
        if getattr(self, "_drag_active", False):
            return

        # Check if click is on allowed time axes
        _allowed_axes = {self.user_axes[k] for k in self._time_axis_keys}
        on_time_axes = event.inaxes in _allowed_axes

        # ENHANCED: Allow clicks outside axes ONLY if we're finalizing active two-click
        # This enables completing selection by clicking anywhere in the figure after
        # the first click armed the selection inside a time axis
        if not on_time_axes:
            # First click must be on time axes to arm the selection
            if not getattr(self, "_two_click_active", False):
                return  # Not armed yet - require being over time axes
            # Otherwise continue - we're finalizing an active selection from outside axes

        x_any = self._x_from_anywhere(event)
        if x_any is None:
            return

        # Clamp to primary axis view if present
        primary_ax = self.user_axes.get(self._primary_time_key, None)
        if primary_ax is not None:
            lo, hi = sorted([primary_ax.viewLim.x0, primary_ax.viewLim.x1])
            x_any = min(max(float(x_any), lo), hi)

        # First click: arm
        if not getattr(self, "_two_click_active", False):
            self._two_click_active = True
            self._two_click_t0 = float(x_any)

            # Show a visible sliver
            eps = self._px_to_data_dx(primary_ax or event.inaxes, 2) if (primary_ax or event.inaxes) is not None else 1e-10
            self._update_time_overlays(self._two_click_t0, self._two_click_t0 + eps, color="tab:orange")

            t0 = pd.Timestamp(mdates.num2date(self._two_click_t0)).tz_localize(None)

            # CRITICAL: Clear box-select / rule-preview state so the new
            # two-click selection's highlights are computed from scratch
            # (rather than blending with the previous box's spans or a
            # leftover component filter).
            if hasattr(self, 'current_spans'):
                self.current_spans.clear()
            if hasattr(self, '_commit_spans'):
                self._commit_spans.clear()
            if hasattr(self, "_selected_component_labels"):
                self._selected_component_labels = None
            if hasattr(self, "active_pane") and hasattr(
                self.active_pane, "_selected_component_labels"
            ):
                self.active_pane._selected_component_labels = None

            self.current_selection = (t0, t0)
            # draw strip sliver
            self._draw_strip_preview_spans([(self._two_click_t0, self._two_click_t0 + eps)])

            # NOTE: Don't show highlights on first click - only on completion
            # self._show_selected_point_highlights(redraw=False)  # Disabled for performance
            return

        # Second click: finalize
        t0 = float(getattr(self, "_two_click_t0", x_any))
        t1 = float(x_any)

        self._two_click_active = False
        self._two_click_t0 = None
        # keep preview visible at final span (user can press Enter to add)
        # Use yellow for completed selections to match dragbox behavior
        self._update_time_overlays(t0, t1, color="yellow")

        lo_f, hi_f = sorted([t0, t1])
        s_ts = pd.Timestamp(mdates.num2date(lo_f)).tz_localize(None)
        e_ts = pd.Timestamp(mdates.num2date(hi_f)).tz_localize(None)

        # Snap to nearest if requested
        try:
            if self.snap_var.get():
                i0 = self.df.index.get_indexer([s_ts], method="nearest")[0]
                i1 = self.df.index.get_indexer([e_ts], method="nearest")[0]
                if i0 >= 0: s_ts = pd.Timestamp(self.df.index[i0]).tz_localize(None)
                if i1 >= 0: e_ts = pd.Timestamp(self.df.index[i1]).tz_localize(None)
        except Exception:
            pass

        # Clamp to visible window
        try:
            s_ts = max(s_ts, self.t0); e_ts = min(e_ts, self.t1)
        except Exception:
            pass

        # CRITICAL: Clear box selection state so highlighting works properly
        if hasattr(self, 'current_spans'):
            self.current_spans.clear()
        if hasattr(self, '_commit_spans'):
            self._commit_spans.clear()

        self.current_selection = (s_ts, e_ts)

        x0 = mdates.date2num(s_ts); x1 = mdates.date2num(e_ts)
        self._draw_strip_preview_spans([(x0, x1)])

        # Show point highlights ONLY after final selection is complete
        self._show_selected_point_highlights(redraw=True)



    def _on_time_motion(self, event, pane):
        """
        While first-click is active, keep the multi-panel overlay AND the strip preview
        in sync with the cursor using blitting (no full redraws).
        """
        # Only process events on the active pane
        if pane is not self.active_pane:
            return

        if not getattr(self, "_two_click_active", False):
            return

        # ---- while RectangleSelector drag is in progress, do not run 2-click preview
        if getattr(self, "_drag_active", False):
            return

        import pandas as pd, matplotlib.dates as mdates

        x_any = self._x_from_anywhere(event)
        if x_any is None:
            return

        # Clamp to primary axis view
        primary_ax = self.user_axes.get(self._primary_time_key, None)
        if primary_ax is not None:
            lo, hi = sorted([primary_ax.viewLim.x0, primary_ax.viewLim.x1])
            x_any = min(max(float(x_any), lo), hi)

            # ENHANCED: If mouse left the time axes, clamp more aggressively to edges
            # This ensures selection extends to edges smoothly
            time_axes = {self.user_axes[k] for k in (self._time_axis_keys or [])}
            if event.inaxes not in time_axes:
                # Mouse is outside time axes - user wants to select to edge
                # Check which edge they're near based on screen position
                try:
                    # Get axis bounds in screen coordinates
                    bbox = primary_ax.get_window_extent()
                    mouse_x = event.x  # Screen X coordinate

                    # If mouse is left of axis, snap to left edge
                    if mouse_x < bbox.x0:
                        x_any = lo
                    # If mouse is right of axis, snap to right edge
                    elif mouse_x > bbox.x1:
                        x_any = hi
                    # Otherwise use the clamped value we already calculated
                except Exception:
                    # If screen coord transform fails, keep using clamped value
                    pass

        lo_f, hi_f = sorted([float(self._two_click_t0), float(x_any)])
        s_ts = pd.Timestamp(mdates.num2date(lo_f)).tz_localize(None)
        e_ts = pd.Timestamp(mdates.num2date(hi_f)).tz_localize(None)

        try:
            if self.snap_var.get():
                i0 = self.df.index.get_indexer([s_ts], method="nearest")[0]
                i1 = self.df.index.get_indexer([e_ts], method="nearest")[0]
                if i0 >= 0: s_ts = pd.Timestamp(self.df.index[i0]).tz_localize(None)
                if i1 >= 0: e_ts = pd.Timestamp(self.df.index[i1]).tz_localize(None)
        except Exception:
            pass

        try:
            s_ts = max(s_ts, self.t0)
            e_ts = min(e_ts, self.t1)
        except Exception:
            pass

        # CRITICAL: Clear box selection state so highlighting works properly
        if hasattr(self, 'current_spans'):
            self.current_spans.clear()
        if hasattr(self, '_commit_spans'):
            self._commit_spans.clear()

        # keep the preview state; no full redraws here
        self.current_selection = (s_ts, e_ts)
        x0 = mdates.date2num(s_ts); x1 = mdates.date2num(e_ts)

        # Ensure a visible sliver when endpoints coincide
        if x1 <= x0:
            eps = self._px_to_data_dx(primary_ax or event.inaxes, 2) if (primary_ax or event.inaxes) is not None else 1e-10
            x1 = x0 + eps

        self._update_time_overlays(x0, x1, color="tab:orange")
        self._draw_strip_preview_spans([(x0, x1)])

        # NOTE: No real-time highlighting during motion for performance
        # self._show_selected_point_highlights(redraw=False)  # Disabled for performance


    # === Helpers for strip editing ===

    def _ts_from_event(self, event) -> Optional[pd.Timestamp]:
        """Convert Matplotlib event.xdata to a naive pd.Timestamp (or None)."""
        if event.xdata is None:
            return None
        dt = mdates.num2date(event.xdata)
        if getattr(dt, "tzinfo", None) is not None:
            dt = dt.replace(tzinfo=None)
        return pd.Timestamp(dt)

    def _hit_test_selected(self, event) -> Optional[str]:
        """
        If pointer is near selected interval on the strip, return a mode:
          "resize_left" | "resize_right" | "move" | None
        """
        strip_ax = self.active_pane.strip_ax if hasattr(self, 'active_pane') else self.strip_ax
        if self.selected_interval is None or event.inaxes is not strip_ax:
            return None

        ax = strip_ax
        iv = self.selected_interval
        x0 = mdates.date2num(iv.start)
        x1 = mdates.date2num(iv.end)
        if x0 > x1:
            x0, x1 = x1, x0

        # Convert data x-coords to display (pixel) coords
        x0_px = ax.transData.transform((x0, 0))[0]
        x1_px = ax.transData.transform((x1, 0))[0]
        mx = event.x  # pixel x

        # Edge handles first
        if abs(mx - x0_px) <= HANDLE_PX:
            return "resize_left"
        if abs(mx - x1_px) <= HANDLE_PX:
            return "resize_right"
        # Inside span?
        if x0_px < mx < x1_px:
            return "move"
        return None

    def _set_cursor(self, name: Optional[str]) -> None:
        """Set cursor on the Tk canvas widget."""
        widget = self.canvas.get_tk_widget()  # type: ignore[union-attr]
        if name is None:
            widget.configure(cursor="")
        else:
            # Common cross-platform names: "sb_h_double_arrow" (resize), "fleur" (move)
            widget.configure(cursor=name)


    # === Strip drag/resize/move handlers ===

    def _on_strip_press(self, event, pane) -> None:
        # Only process events on the active pane
        if pane is not self.active_pane:
            return

        # Left-click only, and only on the strip axis
        if event.button != 1 or event.inaxes is not pane.strip_ax:
            return

        # If no selected interval yet, select the one under the cursor (if any)
        click_ts = self._ts_from_event(event)
        if click_ts is None:
            return

        if self.selected_interval is None:
            # Select if inside any interval
            for iv in self.intervals:
                if iv.contains(click_ts):
                    self.selected_interval = iv
                    if self.status_var is not None:
                        self.status_var.set(
                            f"Selected: {iv.label} [{iv.start.strftime('%H:%M:%S')} → {iv.end.strftime('%H:%M:%S')}]"
                        )
                    self._update_strip()
                    self.canvas.draw_idle()  # type: ignore[union-attr]
                    break

        # Determine drag mode against the selected interval
        mode = self._hit_test_selected(event)
        if mode is None or self.selected_interval is None:
            return

        iv = self.selected_interval
        self._drag_mode = mode
        self._drag_iv = iv
        # A drag works in CLOSED geometry; the interval is stored HALF-OPEN.
        # A tail interval's stored end is data_end PLUS one index unit -- the
        # cap _on_strip_release applies below -- and carrying that epsilon
        # into the drag width RATCHETS: move a tail interval to the interior
        # with snapping OFF and it covers one extra sample for the rest of the
        # session (measured 10 -> 11 -> 11, 11, 11 on all four resolutions).
        # Undo the cap HERE, at the one place it leaks, so the cap applies
        # only to the commit that actually reaches the final sample. The min()
        # is byte-for-byte _apply_snap_clamp's own end clamp, so an interval
        # that has never been to the tail is handed through unchanged.
        self._drag_initial = (iv.start, min(iv.end, self.data_end))
        if mode == "move":
            self._drag_offset = click_ts - iv.start
            self._set_cursor("fleur")
        else:
            self._set_cursor("sb_h_double_arrow")

    def _on_strip_motion(self, event, pane) -> None:
        # Only process events on the active pane
        if pane is not self.active_pane:
            return

        # Hover cursor feedback when not dragging
        if self._drag_mode is None:
            if event.inaxes is pane.strip_ax:
                mode = self._hit_test_selected(event)
                if mode == "move":
                    self._set_cursor("fleur")
                elif mode in ("resize_left", "resize_right"):
                    self._set_cursor("sb_h_double_arrow")
                else:
                    self._set_cursor(None)
            else:
                self._set_cursor(None)
            return

        # During drag: compute live preview
        if event.inaxes is not pane.strip_ax:
            return
        ts = self._ts_from_event(event)
        if ts is None or self._drag_iv is None or self._drag_initial is None:
            return

        s0, e0 = self._drag_initial
        if self._drag_mode == "move":
            if self._drag_offset is None:
                return
            width = e0 - s0
            new_start = ts - self._drag_offset
            new_end = new_start + width
        elif self._drag_mode == "resize_left":
            new_start = ts
            new_end = e0
        elif self._drag_mode == "resize_right":
            new_start = s0
            new_end = ts
        else:
            return

        new_start, new_end = self._apply_snap_clamp(new_start, new_end)
        self._drag_preview = (new_start, new_end)
        self._preview_selection(new_start, new_end)

    def _on_strip_release(self, event, pane) -> None:
        # Only process events on the active pane
        if pane is not self.active_pane:
            return

        if self._drag_mode is None:
            return

        # Commit the resize/move via a command (undoable)
        if self._drag_iv is not None and self._drag_preview is not None:
            from chronotagger.core.commands import ResizeIntervalCommand
            s_new, e_new = self._drag_preview
            # _apply_snap_clamp pins a right-edge drag AT data_end, and a
            # half-open [s, e) that ENDS at the final sample does not label
            # it: the sample under the handle exports as -1 (Pack 6 census
            # F10c, measured). Cap the tail exactly as the Pack 3 commit
            # paths do. It lands HERE, after the snap, because
            # _snap_to_samples would pull a capped end back onto data_end;
            # and it is the TAIL ONLY, because routing the whole span
            # through _exact_spans_to_half_open grows a repeatedly moved or
            # left-resized interval by one sample per drag -- measured.
            if e_new == self.data_end:
                e_new = self._end_after_inclusive(e_new)
            cmd = ResizeIntervalCommand(self, self._drag_iv, s_new, e_new)
            self._execute_command(cmd)

            # Reselect the interval covering the new midpoint (if any)
            mid = s_new + (e_new - s_new) / 2
            self.selected_interval = None
            for iv in self.intervals:
                if iv.contains(mid) and iv.label == self._drag_iv.label:
                    self.selected_interval = iv
                    break

            if self.status_var is not None:
                if self.selected_interval is not None:
                    iv = self.selected_interval
                    self.status_var.set(
                        f"Resized: {iv.label} [{iv.start.strftime('%H:%M:%S')} → {iv.end.strftime('%H:%M:%S')}]"
                    )
                else:
                    self.status_var.set("Resized interval")

            # Clear preview & refresh -- coalesced (Pack 5 R4d): a strip
            # drag-resize ends in a burst of release-adjacent redraws.
            self.current_selection = None
            self._request_redraw()
            self._save_autosave()

        # Reset drag state & cursor
        self._drag_mode = None
        self._drag_iv = None
        self._drag_initial = None
        self._drag_offset = None
        self._drag_preview = None
        self._set_cursor(None)
