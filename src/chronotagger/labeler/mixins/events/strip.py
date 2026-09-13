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

        # Pack M2: the pool's rectangles are built on the ACTIVE LANE's
        # band, not on the whole strip. This pool is NOT reachable from the
        # painter -- measured: after the painter had followed the active
        # lane at every K, these rectangles were still `xy=(0, 0.05)
        # height=0.9000`, full-strip -- so it has to ask for the band
        # itself. At K == 1 `active_band` returns (0.05, 0.9), the two
        # constants this file used to carry.
        from chronotagger.core.lanes import (active_band,
                                             strip_preview_gid)
        _band_y, _band_h = active_band(self)

        # Pack M2, and this one is a PRE-EXISTING DEFECT this pack is
        # obliged to fix because it makes the four ops above invisible.
        # `_update_strip` calls `ax.clear()`, which DETACHES every pooled
        # rectangle (`r.axes` becomes None and it leaves `ax.patches`), and
        # nothing ever re-adds it. BlitHelper.draw groups its artists BY
        # `a.axes`, so from the first repaint onwards it draws NOTHING and
        # the drag preview is invisible -- measured on `ded305a` itself:
        # `after a repaint: r.axes is None, in ax.patches: False`, and the
        # "blit" then costs 0.008 ms because it does nothing at all.
        # DROP anything the clear took, and let the loop below build it
        # again against the CURRENT axes. Re-adding the old object would
        # also work today and is NOT what this does: the rectangle carries a
        # blended transform built from the axes it was created on, and a
        # rebuild cannot be wrong about that. (Measured: the blit costs the
        # same either way -- what it buys is that it draws at all.)
        pane._strip_preview_pool = [
            _r for _r in pane._strip_preview_pool
            if getattr(_r, "axes", None) is ax]

        while len(pane._strip_preview_pool) < needed:
            r = mpatches.Rectangle(
                (0, _band_y), 0, _band_h,
                transform=trans,
                facecolor="yellow",
                edgecolor="orange",
                linewidth=2,
                alpha=0.30,
                linestyle="--",
                visible=False,
            )
            r.set_gid(strip_preview_gid())
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
        # Pack M2: the band is computed LIVE on every frame, because the
        # pool is built once and the active lane moves -- and because a
        # cached band is a STALE band after any lane switch that does not
        # repaint. Measured: the live call is 0.0023 ms against a 1.78 ms
        # blit, and everything this pack adds to the drag frame totals
        # 0.013 ms, so there is nothing to cache away.
        from chronotagger.core.lanes import active_band
        band_y, band_h = active_band(self)
        artists = []
        for i, (x0, x1) in enumerate(spans_float):
            r = pool[i]
            left = min(x0, x1); width = max(abs(x1 - x0), 0.0)
            r.set_xy((left, band_y))
            if r.get_height() != band_h:
                # Only when it MOVED. Measured: an unconditional
                # set_height() on every motion event cost +1.2 ms on a
                # 1.68 ms blit -- the height only changes when the active
                # lane does, which is once per lane switch, not once per
                # frame.
                r.set_height(band_h)
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
