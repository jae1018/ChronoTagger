"""
Vectorized index mapping for the selection path (Pack 5, R1/R2).

What this replaces: one ``Index.get_indexer([ts])`` call PER SELECTED
POINT. Measured on the user's real 1,464,070-row working set, a single
scalar call costs 2.34 ms -- and that per-probe cost is itself
super-linear in frame size (0.404 ms at 10k rows, 2.34 ms at 1.46M), so
a 20,000-point drag box paid 45.6 s of frozen UI
(evidence ``pack5_g1_selection_latency_report.md`` 2b/2c/S3).

Contract: BIT-EXACT, including the failure mode.
- Same positions. Measured max |deviation| = 0 against the scalar loop at
  every frame size from 10k to 1.46M rows (2d).
- Same raise. On a NON-MONOTONIC index pandas raises the same
  ``ValueError: index must be monotonic increasing or decreasing`` from
  the same guard whether it is asked for one probe or a million (2e).
- ``np.searchsorted`` is REJECTED even though it measured ~2x faster: it
  does not validate sortedness and returned 500 confidently-wrong
  positions (0/500 correct) on a real two-spacecraft frame (2e/S4). In a
  labelling tool a loud failure is worth more than a fast wrong answer.

``method="nearest"`` is load-bearing at exactly one site. The time-lane
box path rebuilds its timestamps through ``mdates.num2date``, whose
float64 day-number round trip lands a median 484 ns (max 999 ns) off the
true sample, so an exact match would recover 2 probes in 2000 (2a/S12).
The other four sites feed timestamps taken straight off a real index and
could use the cheaper exact form; they keep ``nearest`` anyway so all
five sites share one behaviour and one raise.

Unit note (pack5_g1 S1): a DatetimeIndex built from Python datetimes
lands in ``datetime64[us]`` while one built from int64 nanoseconds lands
in ``datetime64[ns]``; ``.asi8`` on the two differs by 1000x and
``np.searchsorted`` maps every probe to position 0 WITHOUT raising.
``Index.get_indexer`` re-units internally, which is one more reason this
module never reaches for the raw integer views.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd


def _probe_index(timestamps: Sequence) -> pd.Index:
    """Wrap the probe sequence as an Index without copying when possible."""
    if isinstance(timestamps, pd.Index):
        return timestamps
    return pd.DatetimeIndex(timestamps)


def positions_nearest(idx: pd.Index, timestamps: Sequence) -> List[int]:
    """
    ONE ``get_indexer(method="nearest")`` for every probe.

    Replaces::

        pos = []
        for ts in timestamps:
            j = idx.get_indexer([ts], method="nearest")[0]
            if 0 <= j < len(idx):
                pos.append(j)

    The in-range filter is preserved as a MASK. A vectorized
    ``get_indexer`` returns -1 for a probe it cannot place, and -1 kept in
    a position list would silently index the LAST row (invariant V1 in
    the gather's semantics contract). Probe ORDER is preserved -- the
    callers de-duplicate and sort afterwards, and one of them intersects
    position SETS across components (V2/V5).
    """
    if timestamps is None or len(timestamps) == 0:
        return []
    pos = idx.get_indexer(_probe_index(timestamps), method="nearest")
    pos = pos[(pos >= 0) & (pos < len(idx))]
    return [int(j) for j in pos]


def positions_exact_then_nearest(idx: pd.Index,
                                 timestamps: Sequence) -> List[int]:
    """
    Vectorized twin of the ``_timestamps_to_indices`` ladder: exact first,
    NEAREST only for the probes that miss.

    The scalar original tried ``idx.get_loc(ts)`` per probe and fell back
    to ``get_indexer([ts], method="nearest")`` inside the except. Three
    behaviours are carried over exactly:

    1. A probe that is a real member resolves to its own position.
    2. A probe that is not resolves to the nearest row -- unless the index
       is non-monotonic, in which case pandas raises and the CURRENT code
       skips that probe silently. Here the whole nearest pass raises once
       and the same probes are skipped.
    3. A DUPLICATED index makes ``get_loc`` return a slice, and the
       current code takes ``slice.start``. ``get_indexer`` refuses a
       non-unique Index outright (InvalidIndexError), so a non-unique
       index keeps the scalar ladder -- vectorizing it would change a
       working answer into an exception.
    """
    if timestamps is None or len(timestamps) == 0:
        return []
    n = len(idx)
    if n == 0:
        return []
    if not idx.is_unique:
        return _scalar_exact_then_nearest(idx, timestamps, n)

    probes = _probe_index(timestamps)
    pos = np.asarray(idx.get_indexer(probes)).copy()
    miss = pos < 0
    if miss.any():
        try:
            pos[miss] = idx.get_indexer(probes[miss], method="nearest")
        except Exception:
            # Non-monotonic index: the scalar loop's inner except skipped
            # exactly these probes, and so do we (the mask below drops -1).
            pass
    pos = pos[(pos >= 0) & (pos < n)]
    return [int(j) for j in pos]


def _scalar_exact_then_nearest(idx: pd.Index, timestamps: Sequence,
                               n: int) -> List[int]:
    """The original per-probe ladder, kept for non-unique indexes only."""
    out: List[int] = []
    for ts in timestamps:
        try:
            j = idx.get_loc(ts)
            if isinstance(j, slice):
                j = j.start
            if j is not None and 0 <= j < n:
                out.append(int(j))
        except Exception:
            try:
                j = idx.get_indexer([ts], method="nearest")[0]
                if 0 <= j < n:
                    out.append(int(j))
            except Exception:
                continue
    return out
