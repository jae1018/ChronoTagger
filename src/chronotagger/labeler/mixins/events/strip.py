"""
Strip interaction methods extracted from chronotagger.labeler.mixins

This module contains methods for:
- Drawing animated preview spans on the strip
- Managing the pool of preview rectangles
- Updating the strip display
- Converting matplotlib events to timestamps
"""

from __future__ import annotations


class StripInteractionMixin:
    """
    The strip PREVIEW machinery: the animated rectangles a drag paints on
    the Labels strip, and the pool they are recycled from.

    Both are load-bearing and stay. Pack 6 re-measured the blit path they
    feed at 79-81x (2.34 ms against 187.2 ms per preview frame at 2,000
    intervals, flat in interval count) -- it fires per motion_notify_event
    during a drag, so without it the app would run at roughly 5 fps
    mid-gesture. Pack 5's R14 fixed a DIFFERENT cost (the full-strip
    redraw) and did not subsume this one.

    Everything ELSE this class used to carry was a copy the MRO never
    reached -- PlottingMixin (position 7) and MouseEventsMixin (position
    10) both precede StripInteractionMixin (position 12) -- and Pack 6
    deleted all four:

    - _squelch_xlim_events   live at plotting.py:50, identical body
    - _apply_time_axis_format live at plotting.py:59; the copy here was a
                             docstring and `pass`
    - _ts_from_event         live at events/mouse.py:306, byte-identical
    - _update_strip          live at plotting.py:714; the copy here was the
                             PRE-R14 one-Rectangle-per-interval version,
                             measured 1,419.9 ms against 27.0 ms at 2,000
                             intervals

    The last two were landmines rather than clutter: reordering the bases
    in app.py:48-58 would have silently switched to a no-op time-axis
    formatter and a ~50x slower strip, with no error and no test able to
    see it. The module-level imports went with them; the two matplotlib
    names the survivors need are imported function-locally below, which is
    where they already were.
    """

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
