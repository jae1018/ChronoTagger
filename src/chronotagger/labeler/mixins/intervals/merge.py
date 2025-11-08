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
        """Sort by start and merge adjacent intervals with the same label."""
        if not self.intervals:
            return
        self.intervals.sort(key=lambda x: x.start)
        merged = [self.intervals[0]]
        for iv in self.intervals[1:]:
            last = merged[-1]
            if iv.start == last.end and iv.label == last.label:
                last.end = iv.end
            else:
                merged.append(iv)
        self.intervals = merged
