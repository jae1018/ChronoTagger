"""
Envelope-preserving draw decimation (Pack 5, R4b/R11).

DOCTRINE, and it is the whole point: nothing here is averaged, binned or
synthesised. Every row handed to ``plot_fn`` is an ORIGINAL sample with
its true timestamp; decimation only chooses WHICH original rows to draw.

Per screen-pixel bin, per NUMERIC column INDEPENDENTLY, the argmin and
argmax rows are selected, and the drawn frame is the UNION of those row
positions across all numeric columns (plus the window's own first and
last row so the trace still spans the axis). A one-sample spike in ANY
numeric column therefore survives by construction: it is that column's
extremum in its own bin. The plan is recomputed for every redraw window,
so zooming in genuinely reveals raw data, and the early exit below turns
decimation off entirely once there are few enough samples per pixel that
it would buy nothing.

MEASURED on this tree (matplotlib 3.10.9, 1400x800 figure, 1340 px
panels, 4 line panels + strip, real ARTEMIS peif columns):

  frame     N        numeric cols   kept        plan cost   frame: full -> decimated
  win_43k    43,000   9             11,346       4.6 ms      332 ms -> 325 ms
  win_100k  100,000   9             13,881       7.9 ms      515 ms -> 426 ms
  win_500k  500,000   9             16,420      34.0 ms      660 ms -> 355 ms

and stacked with the frozen layout engine: 43k 172 ms, 100k 235 ms,
500k 189 ms -- reproducing the gather's measured stack (161 / 171 / 187).

WIDE-FRAME NOTE (R11 asked for this one explicitly). The real 34-numeric-
column spinres frame costs 24.7 ms at 100k rows, 137.7 ms at 500k and
293.4 ms at 1.03M, and the union keeps 24,866 / 36,664 / 40,617 rows
(24.9% / 7.3% / 3.9%). The union cost grows with COLUMN COUNT even though
only a handful of columns are drawn -- that is the price of guaranteeing
spike survival in a column the user has not plotted yet.

HONEST LIMITS (they belong in the README, and they are there):
- Under markers, a decimated series reads as a different dot density.
- Structure finer than one pixel column is not reconstructable; zoom in.
- ``pcolormesh`` / spectrogram panels get nothing from this: decimating
  the time axis of a 2-D mesh drops whole columns of spectral data, which
  is a scientific change, not a rendering optimisation.
- Escape hatch: ``TimeIntervalLabeler(..., decimate=False)``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Below this many samples per pixel column the plan costs more than the
# draw it would save, so decimation is a no-op (R11's early exit).
MIN_POINTS_PER_PIXEL = 4.0


def _bin_edges(idx: pd.Index, n: int, n_bins: int) -> np.ndarray:
    """
    Row-position edges of ``n_bins`` bins, length n_bins + 1.

    TIME bins when the window index is monotonic (equal wall-clock width
    per screen pixel, which is what the eye expects); equal-row-count bins
    otherwise. The fallback matters: a two-spacecraft frame loaded without
    a re-sort is non-monotonic, and time bins there would be nonsense.
    """
    edges = None
    if getattr(idx, "is_monotonic_increasing", False) and n > 1 and idx[0] < idx[-1]:
        try:
            t_edges = pd.date_range(idx[0], idx[-1], periods=n_bins + 1)
            edges = np.asarray(idx.searchsorted(t_edges, side="left"),
                               dtype=np.int64)
        except Exception:
            edges = None
    if edges is None:
        edges = np.linspace(0, n, n_bins + 1).astype(np.int64)
    edges[0] = 0
    edges[-1] = n
    # searchsorted on a monotonic index is already sorted; the accumulate
    # is cheap insurance against a degenerate date_range.
    np.maximum.accumulate(edges, out=edges)
    return edges


def plan_decimation(
    df: pd.DataFrame,
    n_px: int,
    min_points_per_pixel: float = MIN_POINTS_PER_PIXEL,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return ``(kept, bin_starts, bin_ends)`` or None when decimation is a
    no-op for this window.

    ``kept`` is a sorted array of ROW POSITIONS into ``df``; the caller
    draws ``df.take(kept)``. ``bin_starts`` / ``bin_ends`` are the
    half-open row-position ranges of the bins those rows represent; they
    are returned for diagnostics and for any future caller that needs the
    pixel-column geometry.
    """
    n = len(df.index)
    n_bins = int(n_px)
    if n_bins <= 1 or n < 2:
        return None
    if n < min_points_per_pixel * n_bins:
        return None

    edges = _bin_edges(df.index, n, n_bins)
    starts = edges[:-1]
    ends = edges[1:]
    nonempty = ends > starts
    starts = starts[nonempty]
    ends = ends[nonempty]
    if starts.size == 0:
        return None

    bin_id = np.repeat(np.arange(starts.size, dtype=np.int64), ends - starts)
    picks = [np.array([0, n - 1], dtype=np.int64)]

    for col in df.columns:
        dtype = df[col].dtype
        if not pd.api.types.is_numeric_dtype(dtype):
            continue
        if pd.api.types.is_bool_dtype(dtype):
            continue
        try:
            v = df[col].to_numpy(dtype="float64", copy=False)
        except (TypeError, ValueError):
            continue
        for reduce_op in (np.fmin, np.fmax):
            # fmin/fmax IGNORE NaN, so a gappy column still yields the
            # extremes of its real samples; an all-NaN bin yields NaN,
            # which matches nothing and simply contributes no row.
            extreme = reduce_op.reduceat(v, starts)
            hit = np.flatnonzero(v == extreme[bin_id])
            if hit.size == 0:
                continue
            # hit is ascending, so unique(return_index=True) gives the
            # FIRST row that attains the extreme in each bin -- the same
            # tie-break np.argmin/argmax would make.
            _, first = np.unique(bin_id[hit], return_index=True)
            picks.append(hit[first])

    kept = np.unique(np.concatenate(picks))
    if kept.size >= n:
        return None
    return kept, starts, ends
