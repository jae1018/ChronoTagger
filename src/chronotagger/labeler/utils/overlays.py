# src/chronotagger/labeler/utils/overlays.py
"""
Background interval overlays for data panels.

Draws faint vertical bands for every labeled interval that overlaps the
current window across *all* user panels, plus emphasis for the selected
and the preview (drag) interval. Bands are drawn behind traces and
excluded from legends.
"""

from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import matplotlib.axes


def _clip_interval(
    s: pd.Timestamp, e: pd.Timestamp, t0: pd.Timestamp, t1: pd.Timestamp
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Clip [s, e] to [t0, t1]; return None if no overlap."""
    if e <= t0 or s >= t1:
        return None
    return max(s, t0), min(e, t1)


def draw_interval_bands(
    axs: Dict[str, matplotlib.axes.Axes],
    intervals: Iterable,  # Iterable[Interval]-like; must have .start, .end, .label
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    class_colors: Dict[str, str],
    *,
    selected_interval: Optional[object] = None,
    preview: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    alpha: float = 0.10,
    alpha_selected: float = 0.16,
    alpha_preview: float = 0.12,
    zorder: float = 0.05,
) -> None:
    """
    Draw faint full-height background bands for all visible intervals.

    Parameters
    ----------
    axs : dict[str, Axes]
        User panels (NOT the strip).
    intervals : Iterable
        Items with .start, .end, .label
    t0, t1 : Timestamp
        Current visible window.
    class_colors : dict[str, str]
        Color per label.
    selected_interval : object, optional
        If provided and present in `intervals`, the band uses alpha_selected.
    preview : (Timestamp, Timestamp), optional
        Current drag selection; shown as a faint band across all panels.
    alpha, alpha_selected, alpha_preview : float
        Opacities for normal, selected, and preview bands.
    zorder : float
        Low z-order so bands sit behind data.
    """
    # Draw labeled interval bands
    for iv in intervals:
        clipped = _clip_interval(iv.start, iv.end, t0, t1)
        if clipped is None:
            continue
        s, e = clipped
        col = class_colors.get(getattr(iv, "label", ""), "#cccccc")
        this_alpha = alpha_selected if (selected_interval is not None and iv is selected_interval) else alpha

        for ax in axs.values():
            # Full-height vertical span; not in legend; behind data
            ax.axvspan(
                s, e,
                ymin=0.0, ymax=1.0,
                facecolor=col,
                edgecolor="none",
                alpha=this_alpha,
                zorder=zorder,
                label="_nolegend_",
            )

    # Draw preview band (if any)
    if preview is not None:
        ps, pe = preview
        clipped = _clip_interval(ps, pe, t0, t1)
        if clipped is not None:
            ps, pe = clipped
            for ax in axs.values():
                ax.axvspan(
                    ps, pe,
                    ymin=0.0, ymax=1.0,
                    facecolor="yellow",
                    edgecolor="none",      # keep it subtle; we already show dashed edge in strip
                    alpha=alpha_preview,
                    zorder=zorder + 0.01,  # slightly above normal bands, still under data
                    label="_nolegend_",
                )
