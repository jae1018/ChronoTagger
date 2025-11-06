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

from typing import Optional, Tuple
import matplotlib.pyplot as plt
from chronotagger.core.commands import ResizeIntervalCommand

import tkinter as tk
from tkinter import messagebox
import matplotlib.dates as mdates

import pandas as pd
import numpy as np


HANDLE_PX = 8  # hit tolerance in screen pixels for edge resize


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
            self._clear_selected_interval_highlights()
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
                self._show_selected_interval_highlights()
                self.canvas.draw()  # type: ignore[union-attr]
        except Exception:
            self.selected_interval = None
            self._clear_selected_interval_highlights()
            
    def _to_timestamp(self, x):
        import pandas as pd, matplotlib.dates as mdates
        # x is a Matplotlib date float or a datetime
        return pd.Timestamp(mdates.num2date(x) if isinstance(x, (int, float)) else x).tz_localize(None)
    
    def _snap_nearest(self, t):
        # snap t to the nearest df.index sample (no “inside” trimming)
        import pandas as pd
        idx = self.df.index.get_indexer([t], method="nearest")[0]
        return pd.Timestamp(self.df.index[idx]).tz_localize(None) if idx >= 0 else t


    def _on_rectangle_select(self, eclick, erelease) -> None:
        """
        Handle both:
          • Full-height time selection (t0..t1)  -> single preview span (legacy)
          • Box selection (t/y bounded)          -> one or more preview spans from points
    
        Rule of thumb:
          If the selection covers ~entire y-range of the axis (>=95%), we treat it as time-only.
          Otherwise, we treat it as a box over data points on time-lane panels.
        """
        # Which selector triggered this?
        triggered_selector_key = None
        for key, selector in getattr(self, 'rect_selectors', {}).items():
            if self.user_axes.get(key) is eclick.inaxes:
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
        
        if drag_ax is self.strip_ax:
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
        for k, a in self.user_axes.items():
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
        role = self.axes_meta.get(meta_key, {}).get("role", "time").lower()
        
        xlo, xhi = float(x_lo), float(x_hi)
        ylo, yhi = float(y_lo), float(y_hi)
        
        # Branch on axis role
        if role == "not-time":
            # === NOT-TIME AXIS path (position plots, phase space, etc.) ===
            # Use the triggered selector key directly
            spans_commit = self._box_select_on_not_time_axis(drag_ax, xlo, xhi, ylo, yhi, triggered_selector_key)
            
            if not spans_commit:
                self.current_spans.clear()
                self.current_selection = None
                self._commit_spans = []
                self.status_var.set("No points in selection")
                self._update_strip()
                self.canvas.draw_idle()
                return
            
            # Convert to preview format (spans_commit are already half-open)
            spans_preview = [(s, e) for s, e in spans_commit]
            
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
    
        # Turn runs into half-open [start, end) spans
        # --- Build both views of the runs ---
        # (a) What we COMMIT: half-open [first, just-after-last]
        spans_commit = self._runs_to_half_open_intervals(idx_full, runs)
        
        # (b) What we PREVIEW on panels/strip: [first, last] (ends AT last included point)
        spans_preview: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for i0, i1 in runs:
            s = pd.Timestamp(idx_full[i0])
            e = pd.Timestamp(idx_full[i1])  # last included sample time
            s = max(s, self.data_start)
            e = min(e, self.data_end)
            if e >= s:
                spans_preview.append((s, e))
        
        # Optional snap (preview only; commit stays half-open)
        if self.snap_var.get():  # type: ignore[union-attr]
            snapped_prev: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            for s, e in spans_preview:
                ss, ee = self._snap_to_samples(s, e)
                snapped_prev.append((ss, ee))
            spans_preview = snapped_prev
        
        # Stash both: preview for drawing, commit for "Add Label"
        self.current_selection = None
        self.current_spans = spans_preview        # was: self.current_spans = spans
        self._commit_spans = spans_commit         # NEW: used by _add_interval
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
                self.selected_interval = iv
                self.status_var.set(  # type: ignore[union-attr]
                    f"Selected: {iv.label} [{iv.start.strftime('%H:%M:%S')} → {iv.end.strftime('%H:%M:%S')}]"
                )
                self._update_strip()
                self._show_selected_interval_highlights()
                self.canvas.draw()  # type: ignore[union-attr]
                break

    def _focused_widget_is_editable(self) -> bool:
        """
        Return True if the current keyboard focus is on an editable widget
        (Entry, Text, or Combobox). In that case we should not handle global
        navigation/shortcut keys, so the widget's native editing behavior wins.
        """
        if getattr(self, "root", None) is None:
            return False
        w = self.root.focus_get()
        if w is None:
            return False
    
        # Prefer Tk class names — reliable across ttk/tk variants.
        try:
            cls = w.winfo_class()
        except Exception:
            return False
    
        # Common editable classes: 'Entry' (tk), 'TEntry' (ttk), 'Text', 'TCombobox'
        return cls in {"Entry", "TEntry", "Text", "TCombobox"}


    def _on_key_press(self, event) -> None:
        key = event.keysym

        # ---- Escape cancels ANY active selection/preview ----
        if key == "Escape":
            # Check if we have any active selection or preview
            has_selection = (
                getattr(self, "_pick_anchor_ts", None) is not None or 
                getattr(self, "current_selection", None) is not None or
                bool(getattr(self, "current_spans", None)) or
                bool(getattr(self, "_commit_spans", None)) or
                getattr(self, "_two_click_active", False)
            )
            
            if has_selection:
                # Clear all selection states
                self._cancel_active_selection()
                if self.status_var is not None:
                    self.status_var.set("Selection canceled (Escape)")
                return

        # ---- Focus-aware early exit -------------------------------------------
        if self._focused_widget_is_editable():
            if not (event.state & 0x4):  # 0x4 => Control modifier
                return
        # -----------------------------------------------------------------------

        # Class selection with digits 1..9
        if key.isdigit() and int(key) > 0:
            idx = int(key) - 1
            if idx < len(self.classes):
                self.current_class_var.set(self.classes[idx])  # type: ignore[union-attr]
                self.status_var.set(f"Selected class: {self.classes[idx]}")  # type: ignore[union-attr]
            return

        # Navigation
        if key in ("n", "N", "Right"):
            self._next_window()
            return
        if key in ("p", "P", "Left"):
            self._prev_window()
            return

        # Actions
        if key in ("a", "A", "Return"):
            self._add_interval()
            return
        if key in ("d", "D", "Delete"):
            self._delete_interval()
            return
        if key in ("u", "U"):
            if "UNKNOWN" in self.classes:
                self.current_class_var.set("UNKNOWN")  # type: ignore[union-attr]
                self.status_var.set("Selected class: UNKNOWN")  # type: ignore[union-attr]
            return

        # Save / export (Ctrl+S / Ctrl+E)
        if key in ("s", "S") and (event.state & 0x4):
            self._save_session()
            return
        if key in ("e", "E") and (event.state & 0x4):
            self._export_intervals()
            return

        # Undo / Redo (Ctrl+Z / Ctrl+Y) + Backspace ergonomics
        if (key == "z" and (event.state & 0x4)) or key == "BackSpace":
            self._undo()
            return
        if (key == "y" and (event.state & 0x4)) or (key == "BackSpace" and (event.state & 0x1)):
            self._redo()
            return
        
    
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
    
    
    def _update_time_overlays(self, x0: float, x1: float) -> None:
        """
        Move/resize the animated preview band on each time-lane axes and blit only those.
        x0/x1 are Matplotlib date floats.
        
        Skip overlays only if RectangleSelector is actively being dragged by user.
        """
        if not getattr(self, "_time_overlays", None):
            return
            
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
            if not r.get_visible():
                r.set_visible(True)
            artists.append(r)
    
        blit = getattr(self, "_blit", None)
        if blit is not None and artists:
            blit.draw(artists)
        else:
            # graceful fallback
            if getattr(self, "canvas", None) is not None:
                self.canvas.draw_idle()
    
    
    def _hide_time_overlays(self) -> None:
        if not getattr(self, "_time_overlays", None):
            return
        changed = []
        for r in self._time_overlays.values():
            if r.get_visible():
                r.set_visible(False)
                changed.append(r)
        if not changed:
            return
        blit = getattr(self, "_blit", None)
        if blit is not None:
            blit.draw(changed)
        else:
            if getattr(self, "canvas", None) is not None:
                self.canvas.draw_idle()
    
    
    def _init_time_overlays(self) -> None:
        """
        Create/refresh translucent preview bands on every time axis plus the strip. 
        Mark them animated so we can blit them cheaply.
        """
        import matplotlib.patches as mpatches
        from matplotlib.transforms import blended_transform_factory
    
        self._time_overlays = {}
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
                facecolor="tab:orange",
                edgecolor="none",
                alpha=0.25,
                zorder=ax.get_zorder() + 10,
                visible=False,
            )
            r.set_animated(True)  # <- critical for blitting
            ax.add_patch(r)
            self._time_overlays[ax] = r
    
        # also prep a (reusable) pool of strip preview rectangles for multi-span previews
        self._strip_preview_pool = []  # created lazily when needed
    
    
    def _on_time_click(self, event) -> None:
        """
        Two-click selection with blitted preview (canvas-wide).
          • Left-click #1 arms at t0 and shows slim band across time-lane panels + strip.
          • Left-click #2 finalizes [t0, t1] and keeps the preview visible (no full redraw).
          • Right-click cancels.
        """
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
        
        # ---- ignore if this mouse cycle was a drag, or not on a time/strip axis
        if getattr(self, "_drag_active", False):
            return
        if event.inaxes is None:
            return
        # Only allow two-click selection on time-series axes (NOT on strip)
        _allowed_axes = {self.user_axes[k] for k in self._time_axis_keys}
        if event.inaxes not in _allowed_axes:
            return
    
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
            self._update_time_overlays(self._two_click_t0, self._two_click_t0 + eps)
    
            t0 = pd.Timestamp(mdates.num2date(self._two_click_t0)).tz_localize(None)
            
            # CRITICAL: Clear box selection state so highlighting works properly
            if hasattr(self, 'current_spans'):
                self.current_spans.clear()
            if hasattr(self, '_commit_spans'):
                self._commit_spans.clear()
            
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
        self._two_click_last_x = None
        # keep preview visible at final span (user can press Enter to add)
        self._update_time_overlays(t0, t1)
    
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
        self._show_selected_point_highlights(redraw=True)  # Force redraw to ensure highlights appear

    
    
    def _on_time_motion(self, event):
        """
        While first-click is active, keep the multi-panel overlay AND the strip preview
        in sync with the cursor using blitting (no full redraws).
        """
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
    
        self._update_time_overlays(x0, x1)
        self._draw_strip_preview_spans([(x0, x1)])
        
        # NOTE: No real-time highlighting during motion for performance
        # self._show_selected_point_highlights(redraw=False)  # Disabled for performance



    def _ensure_strip_preview_pool(self, needed: int) -> list:
        """
        Ensure there are at least `needed` animated preview rectangles on the strip.
        Returns the pool.
        """
        import matplotlib.patches as mpatches
        from matplotlib.transforms import blended_transform_factory
    
        if getattr(self, "strip_ax", None) is None:
            return []
        if not hasattr(self, "_strip_preview_pool"):
            self._strip_preview_pool = []
    
        ax = self.strip_ax
        trans = blended_transform_factory(ax.transData, ax.transAxes)
    
        while len(self._strip_preview_pool) < needed:
            r = mpatches.Rectangle(
                (0, 0), 0, 0.9,
                transform=trans,
                facecolor="yellow",
                edgecolor="orange",
                linewidth=2,
                alpha=0.30,
                linestyle="--",
                visible=False,
            )
            r.set_animated(True)
            ax.add_patch(r)
            self._strip_preview_pool.append(r)
    
        # hide extras for now (cheap to flip visible later)
        for i, r in enumerate(self._strip_preview_pool):
            r.set_visible(i < needed and r.get_visible())
    
        return self._strip_preview_pool
    
    def _draw_strip_preview_spans(self, spans_float: list[tuple[float, float]]) -> None:
        """
        Update the (animated) strip preview rectangles to depict one or more spans.
        spans_float uses Matplotlib date floats [(x0,x1), ...].
        """
        pool = self._ensure_strip_preview_pool(len(spans_float))
        artists = []
        for i, (x0, x1) in enumerate(spans_float):
            r = pool[i]
            left = min(x0, x1); width = max(abs(x1 - x0), 0.0)
            r.set_xy((left, 0.05))
            r.set_width(width)
            if not r.get_visible():
                r.set_visible(True)
            artists.append(r)
    
        # hide any unused previews
        for j in range(len(spans_float), len(pool)):
            if pool[j].get_visible():
                pool[j].set_visible(False)
                artists.append(pool[j])
    
        blit = getattr(self, "_blit", None)
        if blit is not None and artists:
            blit.draw(artists)
        else:
            if getattr(self, "canvas", None) is not None:
                self.canvas.draw_idle()


    
    
    
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
        if self.selected_interval is None or event.inaxes is not self.strip_ax:
            return None
    
        ax = self.strip_ax
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
    
    def _preview_selection(self, start, end) -> None:
        """Show live preview across panels using current_selection (blitted)."""
        import matplotlib.dates as mdates
        
        # CRITICAL: Clear box selection state so highlighting works properly
        if hasattr(self, 'current_spans'):
            self.current_spans.clear()
        if hasattr(self, '_commit_spans'):
            self._commit_spans.clear()
        
        self.current_selection = (start, end)
        x0 = mdates.date2num(start); x1 = mdates.date2num(end)
        self._update_time_overlays(x0, x1)
        self._draw_strip_preview_spans([(x0, x1)])
        
        # Show point highlights during strip editing preview  
        self._show_selected_point_highlights(redraw=True)  # Force redraw to ensure highlights appear
    
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
    
    def _end_after_inclusive(self, last_ts: pd.Timestamp) -> pd.Timestamp:
        """
        Return an end timestamp that is just after `last_ts` so [start, end)
        includes the last selected sample without guessing sampling cadence.
        """
        try:
            return last_ts + pd.Timedelta(nanoseconds=1)
        except Exception:
            # ultra-conservative fallback
            return last_ts + pd.Timedelta(microseconds=1)
    
    
    def _runs_to_half_open_intervals(
        self,
        idx: pd.DatetimeIndex,
        runs: list[tuple[int, int]],   # inclusive index ranges [(i0, i1), ...]
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Convert inclusive index runs to half-open [start, end) timestamp pairs:
          start = time of first included sample
          end   = time of the sample AFTER the last included (if it exists),
                  else a tiny epsilon after the last sample.
        """
        out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        n = len(idx)
    
        for i0, i1 in runs:
            s = pd.Timestamp(idx[i0])
            if i1 + 1 < n:
                e = pd.Timestamp(idx[i1 + 1])
            else:
                e = self._end_after_inclusive(pd.Timestamp(idx[i1]))
    
            # allow at most an epsilon beyond data_end
            cap = self.data_end + pd.Timedelta(nanoseconds=1)
            if e > cap:
                e = cap
    
            out.append((s, e))
        return out
    
    
    
    # === Strip drag/resize/move handlers ===

    def _on_strip_press(self, event) -> None:
        # Left-click only, and only on the strip axis
        if event.button != 1 or event.inaxes is not self.strip_ax:
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
        self._drag_initial = (iv.start, iv.end)
        if mode == "move":
            self._drag_offset = click_ts - iv.start
            self._set_cursor("fleur")
        else:
            self._set_cursor("sb_h_double_arrow")
    
    def _on_strip_motion(self, event) -> None:
        # Hover cursor feedback when not dragging
        if self._drag_mode is None:
            if event.inaxes is self.strip_ax:
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
        if event.inaxes is not self.strip_ax:
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
    
    def _on_strip_release(self, event) -> None:
        if self._drag_mode is None:
            return
    
        # Commit the resize/move via a command (undoable)
        if self._drag_iv is not None and self._drag_preview is not None:
            from chronotagger.core.commands import ResizeIntervalCommand
            s_new, e_new = self._drag_preview
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
    
            # Clear preview & refresh
            self.current_selection = None
            self._update_plot()
            self._maybe_autosave()
    
        # Reset drag state & cursor
        self._drag_mode = None
        self._drag_iv = None
        self._drag_initial = None
        self._drag_offset = None
        self._drag_preview = None
        self._set_cursor(None)
    
    
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
            
            # Convert to half-open intervals
            intervals = self._runs_to_half_open_intervals(idx_full, runs)
            
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
        
        # Convert to half-open intervals
        return self._runs_to_half_open_intervals(idx_full, runs)
    
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
        Cancel any active selection or preview state.
        Clears all selection types: two-click, box selection, and previews.
        """
        # Clear two-click state
        self._clear_two_click_state()
        
        # Clear all selection states
        self.current_selection = None
        if hasattr(self, 'current_spans'):
            self.current_spans.clear()
        if hasattr(self, '_commit_spans'):
            self._commit_spans.clear()
        
        # Clear point highlights
        if hasattr(self, '_clear_selected_point_highlights'):
            self._clear_selected_point_highlights()
        
        # Hide time overlays
        self._hide_time_overlays()
        
        # Clear strip previews
        self._draw_strip_preview_spans([])
        
        # Update strip display
        self._update_strip()
        
        # Refresh display
        if hasattr(self, 'canvas') and self.canvas is not None:
            self.canvas.draw_idle()
    
    def _on_right_click_cancel(self, event) -> None:
        """
        Handle right-click to cancel active selections.
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
            # Prevent event from propagating to other handlers
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

    
