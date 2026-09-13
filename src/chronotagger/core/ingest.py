"""
ingest.py

Turn a per-sample label COLUMN into label INTERVALS on one track.
Pack M1, R5.

WHY THIS EXISTS. The user's own driver already hand-builds a model-label
lane as a one-row `pcolormesh` of `df["rule_label"]` in all four of his
pane layouts, because ChronoTagger has one strip. It costs him a gridspec
row per pane, it cannot be clicked, edited, undone, selected, saved or
exported, and it reaches the labeler only by smuggling a column into the
DataFrame. This module is the run-length decoder that makes that lane a
real track.

THE THREE THINGS THAT ARE EASY TO GET WRONG, all measured:

1. THE END MUST LAND ON THE NEXT SAMPLE, not on the run's last one.
   Measured on the user's real 2016 rule-label column (115,339 rows):
   end-on-the-last-sample gives 2,349 intervals and 2,349 round-trip
   MISMATCHES -- one dropped row per run; end-on-the-next-sample gives
   2,349 intervals and ZERO. This is the same half-open rule
   EventsBaseMixin._runs_to_half_open_intervals already implements, and
   `add_track_from_column` passes that method's own
   `_end_after_inclusive` in, so the shipped rule (and its 20 timestamp
   pins) is what runs.

2. `k x MEDIAN dt` IS THE WRONG GAP-TOLERANCE DEFAULT, and on this user's
   own data it is wrong by a factor of 14.7. The ARTEMIS ESA cadence is
   BIMODAL -- 137.5 s fast survey (80,344 gaps) and 549-550 s slow survey
   (34,127 gaps) -- so `2 x median = 275 s` sits BETWEEN the two modes and
   splits at every survey-mode change: 34,428 intervals instead of 2,349,
   and 34,155 "gaps" that are not data gaps at all. p95 sits ABOVE every
   nominal cadence mode present in the record, so only real gaps exceed
   it: `3 x p95 = 27m31s` gives 2,360 intervals and 11 splits. The
   default is therefore `3 x p95(dt)`, and whatever tolerance is chosen,
   the ingest REFUSES when it would split more than 10 % of the runs.

3. A GAP-SPLIT FRAGMENT MUST NOT END ON THE NEXT SAMPLE, or the adjacency
   fuse silently puts it back together: measured, 70 of 71 splits were
   re-merged by _sort_and_merge_intervals. `_end_after_inclusive` (one
   index unit) survives the fuse but leaves intervals ONE NANOSECOND wide,
   which the strip paints at zero pixels. "Last sample plus the run's own
   local spacing, clamped inside the gap" survives the fuse AND renders
   (measured minimum width 2m16s). All three choices are 0 round-trip
   mismatches, so exactness does not decide it -- honesty does.

GUI-FREE ON PURPOSE. `intervals_from_column` takes an index and a
sequence and returns Intervals; `TimeIntervalLabeler.add_track_from_column`
is the wrapper that creates the track, runs the gesture and autosaves.
That is the `write_label_map_sidecar` split: the free function is the
testable core.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from .models import Interval
from .tracks import DEFAULT_TRACK_ID

# Refuse a tolerance that shreds the column. 10 % of the runs is the line;
# the 2 x median disaster above split 34,155 of 2,349 runs, which is
# 1,454 %, so this guard turns that particular catastrophe into a message.
MAX_SPLIT_FRACTION = 0.10

# ... but the ratio needs an ABSOLUTE FLOOR, or a single real data gap in
# a single-label column trips it: one split of one run is 100 %. Five is
# the floor. The 2 x median disaster is 34,155 splits of 2,349 runs and is
# refused either way.
MIN_SPLITS_BEFORE_REFUSAL = 5


def index_unit_epsilon(index) -> pd.Timedelta:
    """ONE step of `index`'s own datetime64 resolution.

    Quoted from EventsBaseMixin._index_unit_epsilon (Pack 6.5 R65-1) so
    this module needs no labeler: pandas 3.0 hands out MICROSECOND
    DatetimeIndexes by default, and on such an index a hardcoded +1 ns end
    cap is UNREPRESENTABLE -- the Timestamp promotes to ns and
    idx.searchsorted(end) raises ValueError("Cannot losslessly convert
    units"). pd.Timedelta(1, unit="ns") is bit-identical to
    pd.Timedelta(nanoseconds=1), so a nanosecond index -- every cached
    ARTEMIS frame -- keeps the old cap to the bit.
    """
    unit = getattr(index, "unit", None)
    if unit not in ("ns", "us", "ms", "s"):
        unit = "ns"
    return pd.Timedelta(1, unit=unit)


def index_spacings(index):
    """The gaps between consecutive samples, as a Timedelta Series."""
    return pd.Series(pd.DatetimeIndex(index)).diff().dropna()


def default_gap_tolerance(index) -> pd.Timedelta:
    """3 x p95(dt) of the frame's own index. See the module docstring.

    p95 rather than the median because the median is BELOW the slow-survey
    cadence mode on this user's flagship dataset, and a tolerance below a
    real cadence splits at every mode change. p95 sits above every nominal
    mode present in the record, so only genuine data gaps exceed it.
    """
    d = index_spacings(index)
    if len(d) == 0:
        return pd.Timedelta(0)
    return pd.Timedelta(d.quantile(0.95)) * 3


def _is_unlabeled(value, sentinels) -> bool:
    """True when this sample carries no label.

    NaN / NaT / None are always unlabeled: a float NaN hole inside a run
    would otherwise become a class literally named "nan", which is the
    silent-corruption shape.
    """
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    try:
        return value in sentinels
    except TypeError:
        return False


def _local_spacing(idx, a, b, tol) -> pd.Timedelta:
    """The run's own sample spacing: the median gap inside [a, b].

    Falls back to the frame's own median gap (below tolerance) and then to
    one index unit, so a single-sample fragment still gets a width.
    """
    if b > a:
        gaps = [idx[k + 1] - idx[k] for k in range(a, b)]
        gaps = sorted(g for g in gaps if g <= tol)
        if gaps:
            return pd.Timedelta(gaps[len(gaps) // 2])
    d = index_spacings(idx)
    d = d[d <= tol]
    if len(d):
        return pd.Timedelta(d.median())
    return index_unit_epsilon(idx)


def intervals_from_column(
    index,
    values,
    track_id: str = DEFAULT_TRACK_ID,
    gap_tolerance=None,
    source: str = "column",
    class_order=None,
    unlabeled=(None, -1),
    end_after_inclusive=None,
    data_end=None,
    max_split_fraction: float = MAX_SPLIT_FRACTION,
) -> Tuple[List[Interval], Dict[str, Any]]:
    """Run-length decode a per-sample label column into Intervals.

    index        the frame's DatetimeIndex.
    values       one label per row of `index`. A pd.Series or any
                 sequence.
    track_id     the track every returned interval sits on.
    gap_tolerance a pd.Timedelta (or anything pd.Timedelta accepts), or
                 None for `3 x p95(dt)` of `index`.
    source       what goes into each interval's
                 meta["import.source"], e.g. "column:rule_label".
    class_order  the vocabulary, in the order the ids should take.
                 Defaults to the column's distinct values, SORTED.
    unlabeled    sample values that produce NO interval. NaN / NaT / None
                 are always unlabeled on top of these.
    end_after_inclusive
                 the labeler's own _end_after_inclusive, so the
                 end-of-frame boundary is the SHIPPED rule rather than a
                 re-derivation. Defaults to "+ one index unit".
    data_end     the frame's last timestamp, used as the same cap
                 _runs_to_half_open_intervals applies.

    Returns (intervals, info). `info` carries classes, runs, fragments,
    splits, gap_tolerance and unlabeled_rows -- everything a status line
    or a test needs, so the caller never has to recount.

    Raises ValueError when the tolerance would split more than
    `max_split_fraction` of the runs, naming both counts.
    """
    idx = pd.DatetimeIndex(index)
    if hasattr(values, "to_numpy"):
        vals = list(values.to_numpy())
    else:
        vals = list(values)
    n = len(idx)
    if len(vals) != n:
        raise ValueError(
            "the label column has %d rows and the index has %d; an "
            "ingested column must be aligned to the frame's index"
            % (len(vals), n))

    tol = (default_gap_tolerance(idx) if gap_tolerance is None
           else pd.Timedelta(gap_tolerance))
    eps = index_unit_epsilon(idx)
    if end_after_inclusive is None:
        def end_after_inclusive(ts, _step=eps):
            return ts + _step
    cap = (end_after_inclusive(pd.Timestamp(data_end))
           if data_end is not None else None)

    sentinels = set()
    for s in (unlabeled or ()):
        try:
            sentinels.add(s)
        except TypeError:
            pass

    # --- runs of equal label -------------------------------------------
    runs = []
    unlabeled_rows = 0
    i = 0
    while i < n:
        v = vals[i]
        if _is_unlabeled(v, sentinels):
            unlabeled_rows += 1
            i += 1
            continue
        j = i
        while (j + 1 < n and not _is_unlabeled(vals[j + 1], sentinels)
               and vals[j + 1] == v):
            j += 1
        runs.append((i, j, v))
        i = j + 1

    # --- cut every run at a real data gap ------------------------------
    frags = []
    n_splits = 0
    for a, b, v in runs:
        s = a
        for k in range(a, b):
            if (idx[k + 1] - idx[k]) > tol:
                frags.append((s, k, v))
                n_splits += 1
                s = k + 1
        frags.append((s, b, v))

    if (runs and n_splits >= MIN_SPLITS_BEFORE_REFUSAL
            and n_splits > max_split_fraction * len(runs)):
        raise ValueError(
            "the gap tolerance %s would split %d of %d runs (%.1f %%), "
            "which is more than the %.0f %% this ingest accepts. TWO things "
            "look like this and the fix is different for each. (1) The "
            "tolerance is BELOW one of the record's own cadence modes, so "
            "every mode change cuts a run -- check the frame's dt "
            "distribution. (2) The RECORD IS HOLEY: the tolerance is right "
            "and the data really does stop and start, in which case a run "
            "SHOULD be cut and the refusal is the guard being too strict "
            "for this frame. Measured on the user's own C05 frame: 31 real "
            "gaps above 15 min, 27 of them inside a run, and every "
            "tolerance from 5 min to 45 min is refused while 1h is "
            "accepted. Either way the answer is an explicit, LARGER "
            "gap_tolerance."
            % (tol, n_splits, len(runs),
               100.0 * n_splits / len(runs), 100.0 * max_split_fraction))

    # --- boundaries -----------------------------------------------------
    out = []
    tol_iso = tol.isoformat()
    for a, b, v in frags:
        start = pd.Timestamp(idx[a])
        if b + 1 < n:
            if (idx[b + 1] - idx[b]) <= tol:
                # Inside the record: the end IS the next sample. This is
                # the only rule that reproduces the column exactly.
                end = pd.Timestamp(idx[b + 1])
            else:
                # At a real gap: last sample plus the run's own spacing,
                # kept strictly inside the gap so the adjacency fuse
                # cannot put the two halves back together.
                spacing = _local_spacing(idx, a, b, tol)
                ceiling = pd.Timestamp(idx[b + 1]) - eps
                end = pd.Timestamp(idx[b]) + spacing
                if end > ceiling:
                    end = ceiling
        else:
            # End of the record: the shipped _end_after_inclusive rule.
            end = end_after_inclusive(pd.Timestamp(idx[b]))
        if cap is not None and end > cap:
            end = cap
        if end <= start:
            end = end_after_inclusive(start)
        out.append(Interval(
            start, end, str(v), None,
            track=track_id,
            meta={
                "import.source": source,
                "import.gap_tolerance": tol_iso,
                "import.run_rows": int(b - a + 1),
            },
        ))

    seen = []
    for _a, _b, v in frags:
        name = str(v)
        if name not in seen:
            seen.append(name)
    if class_order:
        classes = [str(c) for c in class_order]
    else:
        classes = sorted(seen)

    info = {
        "classes": classes,
        "present": sorted(seen),
        "runs": len(runs),
        "fragments": len(frags),
        "splits": n_splits,
        "gap_tolerance": tol,
        "unlabeled_rows": unlabeled_rows,
        "unlabeled": sorted(str(s) for s in sentinels) + ["NaN", "NaT", "None"],
    }
    return out, info
