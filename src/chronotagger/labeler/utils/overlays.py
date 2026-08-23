# src/chronotagger/labeler/utils/overlays.py
"""
Background interval overlays for data panels.

Draws faint vertical bands for every labeled interval that overlaps the
current window across *all* user panels, plus emphasis for the selected
and the preview (drag) interval. Bands are drawn behind traces and
excluded from legends.

Pack 5 (R4c): the bands are ONE PolyCollection per axis instead of one
``axvspan`` Rectangle per interval per axis. The old shape built 400
patches for 100 intervals on a 4-panel figure and 8,000 for 2,000, and
interval count -- not point count -- is what actually dominates a frame:
going 43k -> 500k points costs +264 ms, going 0 -> 2,000 intervals at
fixed 43k points costs +8,375 ms (evidence ``pack5_g2_redraw_cost_report``
section 4). MEASURED here, 4 panels on the 43k window, create + draw:

    100 intervals   axvspan  569.9 ms   ->  PolyCollection  305.0 ms
    500 intervals   axvspan 1766.8 ms   ->  PolyCollection  361.2 ms
   2000 intervals   axvspan 6911.6 ms   ->  PolyCollection  611.4 ms

The pattern is not new to the codebase -- ``events/overlays.py`` already
draws the multi-span preview this way; this module simply stopped being
the exception.

Two details are load-bearing rather than cosmetic:
- a BLENDED transform (x in data coords, y in axes coords) keeps the
  bands full-height exactly as ``axvspan`` did, including after a
  Y-axis-only wheel zoom, which never replots;
- every collection carries a ``chronotagger:`` gid, because Pack 3's
  artist scans read ``ax.collections`` and tool ink must never be
  mistaken for data.
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.axes


def _clip_interval(
    s: pd.Timestamp, e: pd.Timestamp, t0: pd.Timestamp, t1: pd.Timestamp
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Clip [s, e] to [t0, t1]; return None if no overlap."""
    if e <= t0 or s >= t1:
        return None
    return max(s, t0), min(e, t1)


def _add_band_collection(ax, spans, facecolors, zorder: float,
                         gid: str) -> None:
    """One PolyCollection carrying every band on this axis."""
    import matplotlib.dates as mdates
    from matplotlib.collections import PolyCollection
    from matplotlib.transforms import blended_transform_factory

    x0 = mdates.date2num(pd.DatetimeIndex([s for s, _ in spans]).to_numpy())
    x1 = mdates.date2num(pd.DatetimeIndex([e for _, e in spans]).to_numpy())
    verts = np.empty((len(spans), 4, 2), dtype=float)
    verts[:, 0, 0] = x0
    verts[:, 1, 0] = x1
    verts[:, 2, 0] = x1
    verts[:, 3, 0] = x0
    verts[:, 0, 1] = 0.0
    verts[:, 1, 1] = 0.0
    verts[:, 2, 1] = 1.0
    verts[:, 3, 1] = 1.0

    poly = PolyCollection(
        verts,
        facecolors=facecolors,
        edgecolors="none",
        linewidths=0.0,
        zorder=zorder,
        label="_nolegend_",
    )
    # y in AXES coordinates: full height now and after any Y zoom, which
    # is what axvspan gave us and what a data-coordinate rectangle would
    # quietly lose.
    poly.set_transform(blended_transform_factory(ax.transData, ax.transAxes))
    # Tool ink, never data (Pack 3 S7): the box-select artist scan walks
    # ax.collections.
    from ..mixins.events.base import TOOL_GID_PREFIX
    poly.set_gid(TOOL_GID_PREFIX + gid)
    # autolim=False: the vertices are half in axes coordinates, so letting
    # them feed the data limits would corrupt the y range.
    ax.add_collection(poly, autolim=False)


def draw_interval_bands(
    axs: Dict[str, matplotlib.axes.Axes],
    intervals: Iterable,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    class_colors: Dict[str, str],
    *,
    selected_interval: Optional[object] = None,
    preview: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    preview_spans: Optional[Iterable[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
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
    from matplotlib.colors import to_rgba

    # Clip once, colour once: the geometry is identical on every panel, so
    # only the artists are per-axis.
    band_spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    band_faces: List[tuple] = []
    for iv in intervals:
        clipped = _clip_interval(iv.start, iv.end, t0, t1)
        if clipped is None:
            continue
        s, e = clipped
        col = class_colors.get(getattr(iv, "label", ""), "#cccccc")
        this_alpha = alpha_selected if (
            selected_interval is not None and iv is selected_interval) else alpha
        band_spans.append((s, e))
        band_faces.append(to_rgba(col, this_alpha))

    # Preview bands, single and multi, share one collection: identical
    # colour, identical z-order, and they never overlap the labelled set
    # in meaning.
    prev_spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    if preview is not None:
        clipped = _clip_interval(preview[0], preview[1], t0, t1)
        if clipped is not None:
            prev_spans.append(clipped)
    if preview_spans:
        for (ps, pe) in preview_spans:
            clipped = _clip_interval(ps, pe, t0, t1)
            if clipped is None:
                continue
            prev_spans.append(clipped)

    if not band_spans and not prev_spans:
        return

    prev_faces = [to_rgba("yellow", alpha_preview)] * len(prev_spans)

    for ax in axs.values():
        if band_spans:
            _add_band_collection(ax, band_spans, band_faces, zorder,
                                 "interval-bands")
        if prev_spans:
            _add_band_collection(ax, prev_spans, prev_faces, zorder + 0.01,
                                 "interval-preview-bands")
