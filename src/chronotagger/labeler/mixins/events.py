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

        # ---- Escape cancels two-click preview / current selection ----
        if key == "Escape":
            if getattr(self, "_pick_anchor_ts", None) is not None or self.current_selection is not None:
                self._clear_two_click_state()
                if self.status_var is not None:
                    self.status_var.set("Selection canceled")
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
        if getattr(self, "_time_axis_keys", None):
            time_axes = {self.user_axes[k] for k in self._time_axis_keys}
        else:
            time_axes = set(self.user_axes.values())
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
        """Show/align the preview band on every time-series panel (and strip)."""
        if not getattr(self, "_time_overlays", None):
            return
        left = min(x0, x1)
        width = max(abs(x1 - x0), 0.0)
        for r in self._time_overlays.values():
            r.set_xy((left, 0))
            r.set_width(width)
            if not r.get_visible():
                r.set_visible(True)
        self._two_click_last_x = x1
        if getattr(self, "canvas", None) is not None:
            self.canvas.draw_idle()
    
    
    def _hide_time_overlays(self) -> None:
        if not getattr(self, "_time_overlays", None):
            return
        for r in self._time_overlays.values():
            if r.get_visible():
                r.set_visible(False)
        if getattr(self, "canvas", None) is not None:
            self.canvas.draw_idle()
    
    
    def _init_time_overlays(self) -> None:
        """
        Create/refresh a translucent band on every time-lane panel (col 0, role='time')
        plus the strip, for two-click preview. Full-height in axes coords.
        """
        import matplotlib.patches as mpatches
        from matplotlib.transforms import blended_transform_factory
    
        self._time_overlays = {}
        self._two_click_active = False
        self._two_click_t0 = None
        self._two_click_last_x = None
    
        axes = []
        # only time-lane axes in column 0
        if getattr(self, "_time_axis_keys", None):
            for k in self._time_axis_keys:
                ax = self.user_axes.get(k)
                if ax is not None and self._is_time_lane_axis(ax):
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
            ax.add_patch(r)
            self._time_overlays[ax] = r
    
    
    def _on_time_click(self, event):
        """
        Two-click behavior:
          1st left-click on any time-lane axis or strip: start preview at t0,
             show a thin visible band immediately on ALL time-lane panels.
          2nd left-click: commit [t0, t1] through the existing rectangle-select path.
          Right-click: cancel.
        """
        # right-click cancels anywhere
        if getattr(event, "button", None) == 3:
            self._two_click_active = False
            self._two_click_t0 = None
            self._two_click_last_x = None
            self._hide_time_overlays()
            # also clear strip preview
            self.current_selection = None
            self._update_strip()
            if self.canvas is not None:
                self.canvas.draw_idle()
            return
    
        # only left-clicks on the time lane (col 0 time axes) or the strip
        if getattr(event, "button", None) != 1:
            return
        ax = getattr(event, "inaxes", None)
        if not (self._is_time_lane_axis(ax) or ax is self.strip_ax):
            return
    
        # map to float-date x
        import matplotlib.dates as mdates
        x = getattr(event, "xdata", None)
        if x is None:
            return
    
        # first click: start & show ~2px sliver immediately
        if not self._two_click_active:
            self._two_click_active = True
            self._two_click_t0 = float(x)
    
            # show sliver on all overlays
            primary_ax = self.user_axes.get(self._primary_time_key, ax)
            eps = self._px_to_data_dx(primary_ax, 2) if primary_ax is not None else 1e-10
            self._update_time_overlays(self._two_click_t0, self._two_click_t0 + eps)
    
            # also seed strip preview so users see it there too
            t0 = mdates.num2date(self._two_click_t0)
            if getattr(t0, "tzinfo", None) is not None:
                t0 = t0.replace(tzinfo=None)
            import pandas as pd
            ts0 = pd.Timestamp(t0)
            self.current_selection = (ts0, ts0)
            self._update_strip()
            if self.canvas is not None:
                self.canvas.draw_idle()
            return
    
        # second click: finalize
        t0, t1 = self._two_click_t0, float(x)
        self._two_click_active = False
        self._two_click_t0 = None
        self._two_click_last_x = None
        self._hide_time_overlays()
    
        # call the existing rectangle-select logic on the primary time axis
        import types
        primary_ax = self.user_axes.get(self._primary_time_key, ax)
        e1 = types.SimpleNamespace(xdata=t0, ydata=0.0, inaxes=primary_ax)
        e2 = types.SimpleNamespace(xdata=t1, ydata=1.0, inaxes=primary_ax)
        self._on_rectangle_select(e1, e2)
    
    
    def _on_time_motion(self, event):
        """
        While first-click is active, keep the multi-panel overlay AND the strip preview
        in sync with the cursor.
        """
        if not self._two_click_active:
            return
        # only track motion over time-lane axes or strip (ignore XY panels)
        ax = getattr(event, "inaxes", None)
        if not (self._is_time_lane_axis(ax) or ax is self.strip_ax):
            return
        x = getattr(event, "xdata", None)
        if x is None:
            return
    
        self._update_time_overlays(self._two_click_t0, float(x))
    
        # keep the strip preview in lockstep
        import matplotlib.dates as mdates, pandas as pd
        s = pd.Timestamp(mdates.num2date(min(self._two_click_t0, float(x))).replace(tzinfo=None))
        e = pd.Timestamp(mdates.num2date(max(self._two_click_t0, float(x))).replace(tzinfo=None))
        s, e = self._apply_snap_clamp(s, e)
        self.current_selection = (s, e)
        self._update_strip()
        if self.canvas is not None:
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
    
    def _preview_selection(self, start: pd.Timestamp, end: pd.Timestamp) -> None:
        """Show live preview across panels using current_selection."""
        self.current_selection = (start, end)
        self._update_strip()
        self.canvas.draw_idle()  # type: ignore[union-attr]
    
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
        # legacy: all user axes are time
        return ax in self.user_axes.values()

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

    
