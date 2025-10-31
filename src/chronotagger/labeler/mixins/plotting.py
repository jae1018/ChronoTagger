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
        """Redraw user panels and strip."""
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

        # Align limits + date formatting for time axes
        for ax in time_axes.values():
            ax.set_xlim(self.t0, self.t1)
            self._apply_time_axis_format(ax)
            ax.margins(x=0.01)

        # Non-time axes: do not touch xlim/formatting (users control them)

        # Background interval overlays across time panels only
        if self._overlays_enabled() and time_axes:
            draw_interval_bands(
                time_axes,
                self.intervals,
                self.t0, self.t1,
                self.class_colors,
                selected_interval=self.selected_interval,
                preview=self.current_selection,
                alpha=0.10,
                alpha_selected=0.16,
                alpha_preview=0.12,
                zorder=0.05,
            )

        # Strip + sidebar
        self._update_strip()
        self._update_intervals_list()

        self.canvas.draw()

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
