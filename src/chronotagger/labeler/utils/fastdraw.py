# src/chronotagger/labeler/utils/fastdraw.py
from __future__ import annotations
from typing import Dict, Iterable, List
import matplotlib.axes

class BlitHelper:
    """
    Minimal per-axes blitter.
    - Cache a background for each axes.
    - Restore only that region, draw a few animated artists, and blit.
    Falls back to draw_idle() automatically if anything goes wrong.
    """
    def __init__(self, fig, canvas):
        self.fig = fig
        self.canvas = canvas
        self.axes: List[matplotlib.axes.Axes] = []
        self._bg: Dict[matplotlib.axes.Axes, object] = {}

    def add_axes(self, axes: Iterable[matplotlib.axes.Axes]) -> None:
        # de-dup while preserving order
        seen, ordered = set(), []
        for ax in axes:
            if ax and ax not in seen:
                seen.add(ax); ordered.append(ax)
        self.axes = ordered
        self.recache()

    def recache(self, _evt=None) -> None:
        # called on draw_event and when layout changes
        self._bg.clear()
        for ax in self.axes:
            try:
                self._bg[ax] = self.canvas.copy_from_bbox(ax.bbox)
            except Exception:
                # benign; we'll just bail to draw_idle on use
                self._bg[ax] = None

    def draw(self, artists: Iterable, include_other_animated: bool = True) -> None:
        """
        Restore cached backgrounds and draw:
          • the requested artists, plus
          • any other visible *animated* artists on that axes (e.g., RectangleSelector).
        This keeps selector boxes/handles visible when we blit our own overlays.
        """
        by_ax: Dict[matplotlib.axes.Axes, list] = {}
        for a in artists:
            ax = getattr(a, "axes", None)
            if ax is not None:
                by_ax.setdefault(ax, []).append(a)
    
        try:
            for ax, group in by_ax.items():
                bg = self._bg.get(ax)
                if bg is None:
                    raise RuntimeError("no background")
    
                # Restore background for this axes
                self.canvas.restore_region(bg)
    
                # Optionally include any other visible animated artists on this axes
                if include_other_animated:
                    seen = {id(g) for g in group}
                    for child in ax.get_children():
                        try:
                            is_anim = getattr(child, "get_animated", None)
                            if is_anim and child.get_visible() and child.get_animated():
                                if id(child) not in seen:
                                    group.append(child)
                                    seen.add(id(child))
                        except Exception:
                            # be permissive; skip odd artists
                            pass
    
                # Draw all in order
                for a in group:
                    a.axes.draw_artist(a)
    
                # Blit just this axes
                self.canvas.blit(ax.bbox)
        except Exception:
            # Safe fallback
            self.canvas.draw_idle()
