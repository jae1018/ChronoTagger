"""
Interval merging operations mixin.

This module provides functionality for sorting and merging adjacent intervals
with the same label.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chronotagger.core.models import Interval


class IntervalMergeMixin:
    """Mixin providing interval merging operations."""
    
    intervals: list[Interval]
    
    def _sort_and_merge_intervals(self) -> None:
        """Sort by start and merge adjacent same-label intervals WITHIN
        one track.

        Pack M1. Two things had to change together here.

        The FUSE condition gains `iv.track == last.track`, because
        without it one track absorbs another: measured, a `storm`
        interval on track `humanA` ending exactly where a `storm`
        interval on track `modelB` begins fuses into ONE interval
        carrying `humanA`, and the ensemble comparison this whole
        program exists for is destroyed on the first
        _sort_and_merge_intervals(). The column ingest meets the same
        trap from the other side: at a 5x-median gap tolerance, 70 of 71
        gap splits were silently re-fused.

        The SCAN gains a per-track `last`, because `merged[-1]` is only
        the previous candidate while one lane exists. The list is
        therefore walked in (track, start) order and RE-SORTED by start
        before it is rebound, because self.intervals is declared sorted
        by start and the strip painter, _repoint_selected_interval and
        _get_first_labeled_rows all read it that way.

        `sorted(...)` rather than `self.intervals.sort(...)`: the old
        list must not be left ordered by (track, start) for anything
        still holding a reference to it. The rebind at the end is
        unchanged -- list identity is explicitly NOT an invariant here
        (core/commands.py).
        """
        if not self.intervals:
            return
        ordered = sorted(self.intervals, key=lambda x: (x.track, x.start))
        merged = []
        last_by_track = {}
        for iv in ordered:
            last = last_by_track.get(iv.track)
            if (last is not None and iv.start == last.end
                    and iv.label == last.label):
                last.end = iv.end
                continue
            merged.append(iv)
            last_by_track[iv.track] = iv
        merged.sort(key=lambda x: x.start)
        self.intervals = merged
