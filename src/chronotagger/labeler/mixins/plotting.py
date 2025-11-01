# src/chronotagger/labeler/mixins/plotting.py
"""
Plotting & axis formatting mixin.

Responsibilities:
- Redraw user panels and strip
- Keep all x-axes aligned and formatted as dates
"""

from __future__ import annotations

import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

from ..utils.timeaxis import apply_time_axis_format
from ..utils.overlays import draw_interval_bands


class PlottingMixin:
    # Expects on self: fig, canvas, user_axes, strip_ax, class_colors, etc.

    def _apply_time_axis_format(self, ax) -> None:
        apply_time_axis_format(ax)

    def _overlays_enabled(self) -> bool:
        """Safe check for overlay toggle; default True if toggle not built yet."""
        try:
            return bool(self.overlays_var.get())  # created in view_build Options
        except Exception:
            return True

    def _update_plot(self) -> None:
        """Redraw user panels and strip, preserving two-click preview overlays."""
        # Clear data panels (but keep colorbars/inset axes if user created separately)
        for ax in self.user_axes.values():
            ax.clear()
    
        # Choose the dataframe slice for time plots
        try:
            sub_df = self.df.loc[self.t0:self.t1]
        except Exception:
            sub_df = self.df
    
        # User plot function
        try:
            self.plot_fn(self.user_axes, sub_df, self.t0, self.t1)
        except Exception as e:
            for ax in self.user_axes.values():
                ax.text(
                    0.5, 0.5, f"Plot error:\n{e}", transform=ax.transAxes,
                    ha="center", va="center"
                )
    
        # Partition axes into time vs non-time
        if getattr(self, "_time_axis_keys", None):
            time_axes = {k: self.user_axes[k] for k in self._time_axis_keys if k in self.user_axes}
        else:
            # Legacy simple mode: all are time
            time_axes = dict(self.user_axes)
    
        # Align limits + date formatting for time axes (single pass)
        for ax in time_axes.values():
            ax.set_xlim(self.t0, self.t1)
            self._apply_time_axis_format(ax)
            ax.margins(x=0.01)
    
        # Compact x-labels per column (unchanged policy)
        self._apply_time_xlabel_policy()
    
        # Background interval overlays across time panels only
        if self._overlays_enabled() and time_axes:
            draw_interval_bands(
                time_axes,
                self.intervals,
                self.t0, self.t1,
                self.class_colors,
                selected_interval=self.selected_interval,
                preview=self.current_selection,
                alpha=0.15,          # match tests
                alpha_selected=0.16,
                alpha_preview=0.12,
                zorder=0.05,
            )
    
        # Strip + sidebar
        self._update_strip()
        self._update_intervals_list()
    
        # --- Keep two-click preview overlays alive across redraws (NEW) ---
        if getattr(self, "two_click_mode", False):
            # Recreate overlay patches if ax.clear() removed them
            try:
                self._rebuild_time_overlays_if_needed()
                # If user is mid-selection, restore current preview band
                if getattr(self, "_two_click_active", False) and getattr(self, "_two_click_t0", None) is not None:
                    last = getattr(self, "_two_click_last_x", self._two_click_t0)
                    self._update_time_overlays(self._two_click_t0, last)
            except Exception:
                # Fail-safe: do not break drawing if overlays stumble
                pass
    
        self.canvas.draw()
        
    def _apply_time_xlabel_policy(self) -> None:
        """
        Keep Matplotlib defaults everywhere, except:
          • hide x tick labels and xlabel for time-role axes in column 0
            (the strip owns the time axis).
        """
        meta = getattr(self, "axes_meta", None)
        if not isinstance(meta, dict) or not meta:
            return
    
        for key, m in meta.items():
            ax = self.user_axes.get(key)
            if ax is None:
                continue
    
            col = int(m.get("col", 0))
            role = str(m.get("role", "time")).lower()
    
            if col == 0 and role == "time":
                ax.tick_params(axis="x", labelbottom=False)
                ax.set_xlabel("")
                fmt = ax.xaxis.get_major_formatter()
                if hasattr(fmt, "show_offset"):
                    fmt.show_offset = False
                ax.xaxis.get_offset_text().set_visible(False)
            else:
                # Show normal labels and make sure they aren't clipped by their own axes box
                ax.tick_params(axis="x", labelbottom=True)
                for lbl in ax.get_xticklabels():
                    lbl.set_clip_on(False)
                    
    def _rebuild_time_overlays_if_needed(self) -> None:
        """
        Ensure the two-click preview rectangles exist on each time axis + strip
        after a redraw that cleared axes.
        """
        import matplotlib.patches as mpatches
        from matplotlib.transforms import blended_transform_factory
    
        if not getattr(self, "two_click_mode", False):
            return
    
        if not hasattr(self, "_time_overlays"):
            # First-time init from EventsMixin
            if hasattr(self, "_init_time_overlays"):
                self._init_time_overlays()
            return
    
        axes = []
        if getattr(self, "_time_axis_keys", None):
            axes.extend(self.user_axes[k] for k in self._time_axis_keys if k in self.user_axes)
        if getattr(self, "strip_ax", None) is not None:
            axes.append(self.strip_ax)
    
        # Recreate missing patches
        for ax in axes:
            rect = self._time_overlays.get(ax)
            if rect is None or rect not in ax.patches:
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
    
        # Remove stale entries (axes that no longer exist)
        stale = [ax for ax in self._time_overlays.keys() if ax not in axes]
        for ax in stale:
            self._time_overlays.pop(ax, None)
            
    def _build_window_attrs_view(self, j0: int, j1: int) -> dict:
        """
        Build a dict for sub_df.attrs where array-likes that look 'time-like' are
        sliced to the current window. Handles:
          • 1D arrays length N == len(index)          -> v[j0:j1]
          • ND arrays with last axis N == len(index)  -> v[..., j0:j1]
          • tuple/list of such arrays                  -> elementwise sliced
        If N != len(index) but N is plausibly time-like, we apply a
        *proportional* slice: map [j0, j1] from base length to N.
        Everything else is passed through unchanged.
        """
        import numpy as _np
    
        base = getattr(self.df, "attrs", {}) or {}
        base_len = len(self.df.index)
        out: dict = {}
    
        def _proj_bounds(n: int) -> tuple[int, int]:
            """
            Proportionally project [j0, j1] (in base_len units) to [p0, p1] in n units.
            Keeps at least one element when possible.
            """
            if base_len <= 0 or n <= 0:
                return 0, 0
            p0 = int(round(j0 * n / base_len))
            p1 = int(round(j1 * n / base_len))
            if p1 <= p0:
                p1 = min(p0 + 1, n)
            p0 = max(0, min(p0, n))
            p1 = max(0, min(p1, n))
            return p0, p1
    
        def _slice_like(v):
            # numpy / array-ish with ndim
            if hasattr(v, "shape") and getattr(v, "ndim", 0) >= 1:
                n_last = v.shape[-1]
                # exact last-axis alignment
                if base_len and n_last == base_len:
                    return v[..., j0:j1]
                # proportional fallback if length is plausibly time-like (not wildly different)
                ratio = (n_last / base_len) if base_len else 0
                if base_len and n_last >= 8 and 0.1 <= ratio <= 10.0:
                    p0, p1 = _proj_bounds(n_last)
                    return v[..., p0:p1]
                return v
    
            # generic sequences (lists/tuples) — leave slicing to the container case below
            return v
    
        for k, v in base.items():
            try:
                # Handle tuples/lists by element
                if isinstance(v, (list, tuple)):
                    sliced_elems = []
                    changed = False
                    for item in v:
                        new_item = _slice_like(item)
                        if new_item is not item:
                            changed = True
                        sliced_elems.append(new_item)
                    out[k] = type(v)(sliced_elems) if changed else v
                    continue
    
                # 1D pandas Series treated like arrays (len check)
                if hasattr(v, "__len__") and not hasattr(v, "shape"):
                    n = len(v)
                    if base_len and n == base_len:
                        return_v = v[j0:j1]
                        out[k] = return_v
                        continue
                    if base_len and n >= 8 and 0.1 <= (n / base_len) <= 10.0:
                        p0, p1 = _proj_bounds(n)
                        out[k] = v[p0:p1]
                        continue
    
                out[k] = _slice_like(v)
            except Exception:
                out[k] = v
    
        return out

    def _update_strip(self) -> None:
        """Redraw annotation strip (intervals + current selection preview)."""
        ax = self.strip_ax  # type: ignore[assignment]
        ax.clear()
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel("Labels", fontsize=9)

        # Reset limits/formatting because clearing resets formatter
        ax.set_xlim(self.t0, self.t1)
        self._apply_time_axis_format(ax)

        # Labeled intervals in strip
        for iv in self.intervals:
            if iv.end <= self.t0 or iv.start >= self.t1:
                continue
            s = max(iv.start, self.t0)
            e = min(iv.end, self.t1)

            color = self.class_colors.get(iv.label, "#cccccc")
            alpha = 0.8 if iv == self.selected_interval else 0.6
            edgecolor = "red" if iv == self.selected_interval else "black"
            lw = 2 if iv == self.selected_interval else 0.5

            rect = Rectangle(
                (mdates.date2num(s), 0.1),
                mdates.date2num(e) - mdates.date2num(s),
                0.8,
                facecolor=color,
                edgecolor=edgecolor,
                linewidth=lw,
                alpha=alpha,
                picker=True,
            )
            ax.add_patch(rect)

        # Preview rectangle (strip)
        if self.current_selection:
            s, e = self.current_selection
            rect = Rectangle(
                (mdates.date2num(s), 0.05),
                mdates.date2num(e) - mdates.date2num(s),
                0.9,
                facecolor="yellow",
                edgecolor="orange",
                linewidth=2,
                alpha=0.3,
                linestyle="--",
            )
            ax.add_patch(rect)
