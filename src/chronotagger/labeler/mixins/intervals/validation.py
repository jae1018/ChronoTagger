"""
Interval validation and overlap handling mixin.

This mixin provides methods for:
- Counting overlaps between proposed and existing intervals
- Applying overlap policies (skip/replace) to interval spans
- Previewing interval changes before committing
- Carving/truncating existing intervals to make room for new ones
"""

from __future__ import annotations
from typing import List, Tuple
import pandas as pd

from chronotagger.core.models import Interval
from chronotagger.core.commands import (
    AddIntervalCommand,
    DeleteIntervalCommand,
    ResizeIntervalCommand,
)


class IntervalValidationMixin:
    """Mixin for interval validation and overlap policy enforcement."""

    def _count_overlapping_intervals(
        self, spans: List[Tuple[pd.Timestamp, pd.Timestamp]]
    ) -> int:
        """
        Count how many existing intervals overlap with the proposed spans.

        Args:
            spans: List of (start, end) timestamp pairs to check

        Returns:
            Number of unique intervals that have any overlap with spans
        """
        overlapping = set()
        for s, e in spans:
            for iv in self.intervals:
                # Check if interval overlaps with span
                if not (iv.end <= s or iv.start >= e):
                    overlapping.add(id(iv))
        return len(overlapping)

    def _preview_with_policy(
        self, spans: List[Tuple[pd.Timestamp, pd.Timestamp]], policy: str
    ) -> tuple[int, int]:
        """
        Apply overlap policy to spans and update preview.

        Called by OverlapResolutionDialog when user selects a policy.
        Updates the yellow preview to show what will actually be added.

        Args:
            spans: Original spans before policy application
            policy: "skip" or "replace"

        Returns:
            Tuple of (num_points, num_intervals) after policy application
        """
        # Apply policy
        final_spans = self._apply_overlap_policy_to_spans(spans, policy)

        # Update preview to show actual result
        # Convert to preview format (end at last included sample)
        preview_spans = self._commit_to_preview_spans(final_spans)
        self.current_spans = preview_spans
        self._commit_spans = final_spans

        # Count points in final spans
        total_points = 0
        for s, e in final_spans:
            try:
                sub = self.df.loc[s:e]
                total_points += len(sub)
            except Exception:
                pass

        # Update plot to show new preview
        self._update_plot()

        return total_points, len(final_spans)

    def _apply_overlap_policy_to_spans(
        self, spans: List[Tuple[pd.Timestamp, pd.Timestamp]], policy: str
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Return spans after applying the requested overlap policy against
        current self.intervals.

        policy:
          - "skip"     -> remove any portions that overlap existing intervals
          - "replace"  -> (preview) leave spans as-is; carving happens when we add
          - anything else -> passthrough
        """
        policy = (policy or "").lower()
        if policy == "skip":
            pieces: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
            for s, e in spans:
                pieces.extend(self._subtract_overlaps_from_span(s, e))
            return pieces
        # For preview, "replace" shows what you'll add; carving is done at add-time.
        return spans

    def _subtract_overlaps_from_span(
        self,
        s: pd.Timestamp,
        e: pd.Timestamp,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Given a candidate half-open span [s, e), subtract any currently labeled
        intervals and return a list of non-overlapping subspans inside [s, e).
        """
        out: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        if e <= s:
            return out

        # Collect overlaps with existing intervals
        overlaps = [iv for iv in self.intervals if not (iv.end <= s or iv.start >= e)]
        overlaps.sort(key=lambda iv: iv.start)

        cur = s
        for iv in overlaps:
            if iv.start > cur:
                left_end = min(iv.start, e)
                if left_end > cur:
                    out.append((cur, left_end))
            if iv.end > cur:
                cur = max(cur, iv.end)
            if cur >= e:
                break

        if cur < e:
            out.append((cur, e))

        return out

    def _carve_existing_for_new_span(self, s: pd.Timestamp, e: pd.Timestamp) -> None:
        """
        Modify existing intervals so that [s, e) becomes free space:
          - Fully covered intervals are deleted.
          - Left/right edge overlaps are resized.
          - Middle overlaps (new span cuts an interval in two) are split by
            deleting the original and adding two trimmed intervals.
        All changes run inside the caller's gesture; the gesture
        snapshot provides undo/redo (see intervals/commands.py).
        """
        if e <= s:
            return

        # Iterate over a stable snapshot; commands mutate self.intervals
        existing = list(self.intervals)
        for iv in existing:
            # no overlap
            if iv.end <= s or iv.start >= e:
                continue

            # Case A: fully covered by [s,e) -> delete
            if iv.start >= s and iv.end <= e:
                self._execute_command(DeleteIntervalCommand(self, iv))
                continue

            # Case B: overlap on the right edge (keep left)
            if iv.start < s <= iv.end <= e:
                self._execute_command(ResizeIntervalCommand(self, iv, iv.start, s))
                continue

            # Case C: overlap on the left edge (keep right)
            if s <= iv.start < e < iv.end:
                self._execute_command(ResizeIntervalCommand(self, iv, e, iv.end))
                continue

            # Case D: [s,e) strictly inside iv => split into left + right
            if iv.start < s and iv.end > e:
                # delete original
                self._execute_command(DeleteIntervalCommand(self, iv))
                # add left + right fragments with same label/notes
                left_start, left_end = iv.start, s
                right_start, right_end = e, iv.end
                from chronotagger.core.models import Interval
                self._execute_command(AddIntervalCommand(self, Interval(left_start, left_end, iv.label, iv.notes)))
                self._execute_command(AddIntervalCommand(self, Interval(right_start, right_end, iv.label, iv.notes)))
                continue

    def _remove_overlapping_intervals(
        self, new_interval: Interval
    ) -> Tuple[List[Interval], List[Interval]]:
        """
        LEGACY: Remove/trim intervals that overlap `new_interval`.

        Returns
        -------
        removed : list[Interval]
            Original intervals removed due to overlap.
        trims : list[Interval]
            New trimmed intervals added back (non-overlapping parts).
        """
        removed: List[Interval] = []
        trims: List[Interval] = []

        for iv in self.intervals[:]:
            if not iv.overlaps(new_interval):
                continue

            self.intervals.remove(iv)
            removed.append(iv)

            if iv.start < new_interval.start:
                trims.append(Interval(iv.start, new_interval.start, iv.label, iv.notes))

            if iv.end > new_interval.end:
                trims.append(Interval(new_interval.end, iv.end, iv.label, iv.notes))

        self.intervals.extend(trims)
        return removed, trims

    def _commit_to_preview_spans(
        self, spans: List[Tuple[pd.Timestamp, pd.Timestamp]]
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Convert half-open [s, e) commit spans into preview spans that end AT
        the last included sample (s, e_last_included] so they render correctly
        in the strip / overlays.
        """
        out: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        idx = self.df.index
        for s, e in spans:
            j = idx.searchsorted(e, side="left") - 1
            if j >= 0:
                out.append((s, pd.Timestamp(idx[j])))
        return out
