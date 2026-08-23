"""
Vectorized matplotlib-date -> naive-Timestamp conversion (Pack 5, DRAFT
AMENDMENT R13).

What this replaces: one ``mdates.num2date(float(xf))`` plus one
``pd.Timestamp(dt)`` PER SELECTED POINT, at ``selection.py:266-272`` and
its PathCollection twin at ``:297-301``. After R1 removed the scalar
``get_indexer`` loop, THIS became the largest single term in a box-select
gesture -- measured 996.3 ms of a 1,635 ms reference gesture on the 43k
window (34,503 picked points).

The obvious vectorization does NOT help: ``mdates.num2date`` on an array
is ``np.vectorize(_from_ordinalf, otypes="O")``, i.e. still a Python-level
loop, and the per-element ``pd.Timestamp`` survives it. Measured 1.40x to
2.08x -- nowhere near enough.

What works is replicating ``_from_ordinalf``'s arithmetic in numpy and
building ONE DatetimeIndex, with no Python datetime object anywhere:

    dt64 = epoch + round(x * microseconds_per_day) microseconds
    if |x| > 70*365 days: round the MICROSECOND FIELD to the nearest 20 us

The tz round trip inside ``_from_ordinalf`` (``replace(tzinfo=UTC)`` then
``astimezone(tz)``) is the identity when ``rcParams['timezone']`` is UTC,
which is matplotlib's default and what the caller then strips anyway.

BIT-EXACT, measured against the per-point loop it replaces -- max |delta|
in NANOSECONDS after normalising both sides to ``datetime64[ns]``:

    win_43k   34,503 picked points      0 ns
    win_100k  80,540 picked points      0 ns
    win_500k 397,014 picked points      0 ns
    post-2040 index (the >70*365 20-us fixup branch)   0 ns
    pre-1900 index (negative day numbers)              0 ns
    100-microsecond cadence                            0 ns
    3-millisecond cadence                              0 ns

and 7,741x faster on the reference gesture's haul (996.3 ms -> 0.129 ms).

NOT taken, deliberately: reading the timestamps straight off the drawn
index. It is strictly more accurate -- it skips the float64 round trip
entirely -- but it therefore DEVIATES by up to 780 ns from what the
current code produces (G1's measured mdates round-trip error, median
484 ns / max 999 ns). Identical final positions after the nearest
mapping, but a different ``picked_ts``, which is a behaviour change and
out of scope under R3. Bit-exactness is the contract here, not accuracy.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

_MICROSECONDS_PER_DAY = 24.0 * 3600.0 * 1e6

# matplotlib applies its nearest-20-microsecond round-off fix beyond this
# many days from the epoch (matplotlib.dates._from_ordinalf).
_ROUNDOFF_FIX_DAYS = 70 * 365


def naive_timestamps_from_num(xs: Sequence) -> pd.DatetimeIndex:
    """
    Bit-exact vectorized twin of::

        for xf in xs:
            dt = mdates.num2date(float(xf))
            if getattr(dt, "tzinfo", None) is not None:
                dt = dt.replace(tzinfo=None)
            out.append(pd.Timestamp(dt))

    Returns a tz-naive ``DatetimeIndex`` in microsecond resolution -- the
    same resolution ``pd.Timestamp(datetime)`` produces, so downstream
    ``get_indexer`` behaviour is unchanged (G1's S1 unit trap is avoided
    because nothing here touches ``.asi8`` or a raw integer view).
    """
    x = np.asarray(xs, dtype=np.float64)
    if x.size == 0:
        return pd.DatetimeIndex([], dtype="datetime64[us]")

    epoch = np.datetime64(mdates.get_epoch())
    us = np.rint(x * _MICROSECONDS_PER_DAY).astype(np.int64)
    vals = epoch + us.astype("timedelta64[us]")

    big = np.abs(x) > _ROUNDOFF_FIX_DAYS
    if big.any():
        # matplotlib's round-off fix, replicated: the MICROSECOND FIELD of
        # the datetime is rounded to the nearest 20 us, carrying into the
        # seconds. The field is the value modulo 1e6; numpy's floor-mod
        # keeps that non-negative for pre-epoch dates too, so the same
        # expression covers negative day numbers.
        sub = vals[big].astype("datetime64[us]").astype(np.int64)
        frac = np.mod(sub, 1000000)
        sub = sub - frac + np.rint(frac / 20.0).astype(np.int64) * 20
        vals[big] = sub.astype("datetime64[us]")

    return pd.DatetimeIndex(vals)
