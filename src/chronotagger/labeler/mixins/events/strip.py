"""
Strip interaction methods extracted from chronotagger.labeler.mixins

This module contains methods for:
- Drawing animated preview spans on the strip
- Managing the pool of preview rectangles
- Updating the strip display
- Converting matplotlib events to timestamps
"""

from __future__ import annotations

from typing import Optional
from contextlib import contextmanager

import pandas as pd
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory


class StripInteractionMixin:
    """
    Mixin class containing strip-related interaction methods.

    These methods were extracted from:
    - src/chronotagger/labeler/mixins/events.py (_draw_strip_preview_spans, _ensure_strip_preview_pool, _ts_from_event)
    - src/chronotagger/labeler/mixins/plotting.py (_update_strip)
    """

    @contextmanager
    def _squelch_xlim_events(self):
        """Context manager to temporarily suppress xlim change events."""
        prev = getattr(self, "_squelch_xlim", False)
        self._squelch_xlim = True
        try:
            yield
        finally:
            self._squelch_xlim = prev

    def _apply_time_axis_format(self, ax) -> None:
        """Apply time-based axis formatting (implementation depends on external utility)."""
        # This would normally call: apply_time_axis_format(ax)
        # from ..utils.timeaxis import apply_time_axis_format
        pass

    def _ensure_strip_preview_pool(self, needed: int) -> list:
        """
        Ensure there are at least `needed` animated preview rectangles on the strip.
        Returns the pool. Uses active pane's strip axis for multi-pane support.
        """
        import matplotlib.patches as mpatches
        from matplotlib.transforms import blended_transform_factory

        # Use active pane's strip axis for multi-pane support
        pane = self.active_pane if hasattr(self, 'active_pane') else self
        ax = pane.strip_ax if hasattr(pane, 'strip_ax') else getattr(self, "strip_ax", None)

        if ax is None:
            return []

        # Store preview pool per-pane to support multiple panes
        if not hasattr(pane, "_strip_preview_pool"):
            pane._strip_preview_pool = []

        trans = blended_transform_factory(ax.transData, ax.transAxes)

        while len(pane._strip_preview_pool) < needed:
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
            pane._strip_preview_pool.append(r)

        # hide extras for now (cheap to flip visible later)
        for i, r in enumerate(pane._strip_preview_pool):
            r.set_visible(i < needed and r.get_visible())

        return pane._strip_preview_pool

    def _draw_strip_preview_spans(self, spans_float: list[tuple[float, float]]) -> None:
        """
        Update the (animated) strip preview rectangles to depict one or more spans.
        spans_float uses Matplotlib date floats [(x0,x1), ...].
        Uses active pane's canvas for multi-pane support.
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

        # Use active pane's blit helper and canvas for multi-pane support
        pane = self.active_pane if hasattr(self, 'active_pane') else self
        blit = getattr(pane, "_blit", None)

        if blit is not None and artists:
            try:
                blit.draw(artists)
            except Exception:
                # Blit failed (likely no background saved yet)
                # Fallback: temporarily disable animation and force canvas redraw
                for r in artists:
                    if hasattr(r, 'set_animated'):
                        r.set_animated(False)
                canvas = pane.canvas if hasattr(pane, 'canvas') else getattr(self, 'canvas', None)
                if canvas is not None:
                    canvas.draw_idle()
                # Re-enable animation for next time
                for r in artists:
                    if hasattr(r, 'set_animated'):
                        r.set_animated(True)
        else:
            # No blit helper - draw directly
            canvas = pane.canvas if hasattr(pane, 'canvas') else getattr(self, "canvas", None)
            if canvas is not None:
                canvas.draw_idle()





    # === Helpers for strip editing ===

    def _ts_from_event(self, event) -> Optional[pd.Timestamp]:
        """Convert Matplotlib event.xdata to a naive pd.Timestamp (or None)."""
        if event.xdata is None:
            return None
        dt = mdates.num2date(event.xdata)
        if getattr(dt, "tzinfo", None) is not None:
            dt = dt.replace(tzinfo=None)
        return pd.Timestamp(dt)

    def _update_strip(self) -> None:
        """Redraw annotation strip (intervals + current selection preview)."""
        import matplotlib.dates as mdates

        ax = self.strip_ax  # type: ignore[assignment]
        with self._squelch_xlim_events():
            ax.clear()
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_ylabel("Labels", fontsize=9)

            # Reset limits/formatting because clearing resets formatter
            ax.set_xlim(self.t0, self.t1, emit=False)
            self._apply_time_axis_format(ax)

            # zero horizontal padding so the strip matches time panels exactly
            try:
                ax.set_xmargin(0.0)
            except Exception:
                pass
            try:
                ax.margins(x=0.0)
            except Exception:
                pass

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

        # single-span preview
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

        # multi-span preview
        if getattr(self, "current_spans", None):
            for (s, e) in self.current_spans:
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
