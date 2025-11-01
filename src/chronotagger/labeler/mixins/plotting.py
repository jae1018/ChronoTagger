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
        """Redraw user panels and strip (window-aware: df and df.attrs are sliced for you)."""
        # Clear data panels (keep any user-added inset/colorbar axes)
        for ax in self.user_axes.values():
            ax.clear()
    
        # --- window slice ---
        idx = self.df.index
        if len(idx):
            j0 = idx.get_indexer([self.t0], method="nearest")[0]
            j1 = idx.get_indexer([self.t1], method="nearest")[0] + 1
            if j1 <= j0:  # keep at least one sample robustly
                j1 = min(j0 + 1, len(idx))
        else:
            j0 = j1 = 0
    
        # DataFrame time slice (safe fallback)
        try:
            sub_df = self.df.loc[self.t0:self.t1]
        except Exception:
            sub_df = self.df
    
        # Build a window-scoped attrs view so plot_fn doesn't need to slice anything.
        sub_df.attrs = self._build_window_attrs_view(j0, j1)
    
        # --- user plot (defensive) ---
        try:
            self.plot_fn(self.user_axes, sub_df, self.t0, self.t1)
        except Exception as e:
            for ax in self.user_axes.values():
                ax.text(0.5, 0.5, f"Plot error:\n{e}", transform=ax.transAxes,
                        ha="center", va="center")
    
        # Partition axes into time vs non-time
        if getattr(self, "_time_axis_keys", None):
            time_axes = {k: self.user_axes[k] for k in self._time_axis_keys if k in self.user_axes}
        else:
            time_axes = dict(self.user_axes)  # legacy: all are time
    
        # Align limits + date formatting for time axes
        for ax in time_axes.values():
            ax.set_xlim(self.t0, self.t1)
            self._apply_time_axis_format(ax)
            ax.margins(x=0.01)
    
        # Only hide x labels on *time* axes in column 0
        self._apply_xlabel_policy_per_column()
    
        # Overlays on time axes only
        if self._overlays_enabled() and time_axes:
            draw_interval_bands(
                time_axes,
                self.intervals,
                self.t0, self.t1,
                self.class_colors,
                selected_interval=self.selected_interval,
                preview=self.current_selection,
                alpha=0.15,
                alpha_selected=0.16,
                alpha_preview=0.12,
                zorder=0.05,
            )
    
        # Strip + sidebar
        self._update_strip()
        self._update_intervals_list()
        self.canvas.draw()
        
    def _apply_xlabel_policy_per_column(self) -> None:
        """
        Column-aware x-label policy that avoids label collisions:
    
          • Column 0 (time lane): hide x labels on all time-role axes, since the strip owns the time axis.
          • Columns >= 1: show x labels only on the bottom-most axes in each column; hide on others.
    
        Works for both time and XY roles and does not rely on constrained_layout.
        """
        meta = getattr(self, "axes_meta", None)
        if not isinstance(meta, dict) or not meta:
            # legacy/simple mode (1 col): keep existing behavior (strip shows time)
            return
    
        # Group all user axes by column
        by_col: dict[int, list[tuple[str, dict]]] = {}
        for key, m in meta.items():
            # only consider actual user axes we created
            if key in self.user_axes:
                by_col.setdefault(int(m.get("col", 0)), []).append((key, m))
    
        for col, items in by_col.items():
            # Identify the bottom-most axes in this column
            # (max of row + rowspan - 1)
            bottom_key = max(items, key=lambda kv: kv[1].get("row", 0) + kv[1].get("rowspan", 1) - 1)[0]
    
            for key, m in items:
                ax = self.user_axes.get(key)
                if ax is None:
                    continue
    
                role = str(m.get("role", "time")).lower()
                is_bottom = (key == bottom_key)
    
                if col == 0 and role == "time":
                    # Time lane: strip owns the x axis → hide all time-panel x labels
                    ax.tick_params(axis="x", labelbottom=False)
                    ax.set_xlabel("")
                    fmt = ax.xaxis.get_major_formatter()
                    if hasattr(fmt, "show_offset"):
                        fmt.show_offset = False
                    ax.xaxis.get_offset_text().set_visible(False)
                else:
                    # Other columns: only bottom-most keeps x labels
                    if is_bottom:
                        ax.tick_params(axis="x", labelbottom=True, pad=2)
                        # If this is a time axis using ConciseDateFormatter, enable the offset line
                        fmt = ax.xaxis.get_major_formatter()
                        if hasattr(fmt, "show_offset"):
                            fmt.show_offset = True
                    else:
                        ax.tick_params(axis="x", labelbottom=False)
                        ax.set_xlabel("")
                        fmt = ax.xaxis.get_major_formatter()
                        if hasattr(fmt, "show_offset"):
                            fmt.show_offset = False
                        ax.xaxis.get_offset_text().set_visible(False)
            
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
