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
- ``np.searchsorted`` is REJECTED AS A GENERAL SUBSTITUTE even though it
  measured ~2x faster: it does not validate sortedness and returned 500
  confidently-wrong positions (0/500 correct) on a real two-spacecraft
  frame (2e/S4). In a labelling tool a loud failure is worth more than a
  fast wrong answer. Pack 8.5-B B2 uses it at ONE site and only behind a
  sortedness check -- ``positions_nearest`` on an index that is
  duplicated AND monotonic increasing, which is the only shape
  ``get_indexer`` refuses outright and the only shape a searchsorted
  answer is provably right for. Every other index keeps the pandas call
  and therefore keeps the pandas raise.

DUPLICATED TIMESTAMPS ARE REAL DATA. A THEMIS ESA survey day carries
them (1 repeated stamp in 22,282 for thb 2011-08-14, 0.0045 %), and one
of them used to kill every rectangle select in the tool with
``InvalidIndexError`` from ``positions_nearest`` -- five call sites, four
of them unguarded, no dialog, no status change and no forensic log line
(evidence ``pack85b_g1_regressions.md`` section 2).

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
    probes = _probe_index(timestamps)
    if not idx.is_unique and idx.is_monotonic_increasing:
        # Pack 8.5-B B2. get_indexer refuses a non-unique Index outright
        # (InvalidIndexError), and this helper had no guard while its twin
        # at :105 did -- so ONE repeated timestamp in a real survey day
        # killed every rectangle select in the tool. A sorted index makes
        # the answer well defined, so compute it instead of raising.
        pos = _nearest_on_sorted_duplicates(idx, probes)
    else:
        # Unique, or non-monotonic: unchanged, INCLUDING the raise. A
        # non-monotonic index still gets ValueError from pandas, and a
        # non-monotonic DUPLICATED one still gets InvalidIndexError --
        # searchsorted is exactly the tool that answers those confidently
        # and wrongly (module docstring, 2e/S4).
        pos = idx.get_indexer(probes, method="nearest")
    pos = pos[(pos >= 0) & (pos < len(idx))]
    return [int(j) for j in pos]


def _nearest_on_sorted_duplicates(idx: pd.Index, probes: pd.Index):
    """
    ``get_indexer(method="nearest")`` for an index that REPEATS a value.

    Only ever called on a monotonic-increasing index (the caller checks),
    which is what makes ``searchsorted`` safe here and unsafe everywhere
    else in this module.

    Semantics, pinned value-identical to pandas on unique indexes:

    * an exact hit resolves to the FIRST row carrying that value, which
      is the same convention ``_scalar_exact_then_nearest`` takes from
      ``slice.start``;
    * a probe between two rows takes the closer one, and a TIE takes the
      LATER one -- measured, that is what pandas does
      (``np.where(left_dist < right_dist, left, right)`` in
      ``Index._get_nearest_indexer``);
    * a probe outside either end clamps to that end.

    Units, not integers. ``asi8`` on a ``datetime64[us]`` index and on a
    ``datetime64[ns]`` one differ by 1000x, and ``Index.searchsorted``
    refuses the mix rather than converting, so both sides are promoted to
    a common datetime64 unit first and compared as VALUES.
    """
    n = len(idx)
    if n == 0:
        return np.zeros(len(probes), dtype=np.int64) - 1
    try:
        iv = np.asarray(idx)
        pv = np.asarray(probes)
        common = np.promote_types(iv.dtype, pv.dtype)
        iv = iv.astype(common, copy=False)
        pv = pv.astype(common, copy=False)
        right = np.searchsorted(iv, pv, side="left").astype(np.int64)
    except Exception:
        # An index this cannot be computed on (object dtype, mixed types)
        # keeps today's behaviour, which is the pandas refusal.
        return idx.get_indexer(probes, method="nearest")
    left = right - 1
    np.clip(right, 0, n - 1, out=right)
    np.clip(left, 0, n - 1, out=left)
    return np.where(np.abs(pv - iv[left]) < np.abs(pv - iv[right]),
                    left, right)


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
