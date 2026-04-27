"""
Unit tests for overlap detection and resolution functionality.

Tests the core overlap handling logic in IntervalsMixin:
- Overlap counting
- Skip policy (subtracting overlaps from spans)
- Replace policy (carving existing intervals)
- Policy application to multiple spans
"""

import pytest
import pandas as pd
import numpy as np
from typing import List, Tuple
from unittest.mock import MagicMock

from chronotagger.core.models import Interval
from chronotagger.core.commands import AddIntervalCommand, DeleteIntervalCommand, ResizeIntervalCommand


class MockIntervalsMixin:
    """
    Mock version of IntervalsMixin for testing overlap logic.

    Only includes the methods needed for overlap resolution testing,
    avoiding GUI dependencies.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.intervals: List[Interval] = []
        self.undo_stack: List = []
        self.redo_stack: List = []
        self.max_undo = 100
        self.modified = False

        # Import the methods we want to test
        from chronotagger.labeler.mixins.intervals import IntervalsMixin
        self._count_overlapping_intervals = IntervalsMixin._count_overlapping_intervals.__get__(self)
        self._subtract_overlaps_from_span = IntervalsMixin._subtract_overlaps_from_span.__get__(self)
        self._carve_existing_for_new_span = IntervalsMixin._carve_existing_for_new_span.__get__(self)
        self._apply_overlap_policy_to_spans = IntervalsMixin._apply_overlap_policy_to_spans.__get__(self)
        self._execute_command = IntervalsMixin._execute_command.__get__(self)
        self._remove_overlapping_intervals = IntervalsMixin._remove_overlapping_intervals.__get__(self)
        self._sort_and_merge_intervals = IntervalsMixin._sort_and_merge_intervals.__get__(self)


@pytest.fixture
def df_sample():
    """Create sample DataFrame with DatetimeIndex for testing."""
    idx = pd.date_range("2024-01-01 00:00:00", periods=120, freq="1min")
    return pd.DataFrame(
        {"value": np.linspace(0, 100, len(idx))},
        index=idx
    )


@pytest.fixture
def mixin(df_sample):
    """Create a MockIntervalsMixin instance for testing."""
    return MockIntervalsMixin(df_sample)


# ============================================================================
# Test: Overlap Counting (_count_overlapping_intervals)
# ============================================================================

class TestOverlapCounting:
    """Test the overlap counting logic."""

    def test_no_overlaps(self, mixin):
        """No existing intervals should return 0."""
        mixin.intervals = []

        spans = [
            (pd.Timestamp("2024-01-01 00:10:00"), pd.Timestamp("2024-01-01 00:20:00"))
        ]

        count = mixin._count_overlapping_intervals(spans)
        assert count == 0

    def test_one_overlap(self, mixin):
        """One overlapping interval should return 1."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:05:00"),
                pd.Timestamp("2024-01-01 00:15:00"),
                "A"
            )
        ]

        spans = [
            (pd.Timestamp("2024-01-01 00:10:00"), pd.Timestamp("2024-01-01 00:20:00"))
        ]

        count = mixin._count_overlapping_intervals(spans)
        assert count == 1

    def test_multiple_overlaps_same_interval(self, mixin):
        """Multiple spans overlapping same interval should return 1."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 01:00:00"),
                "A"
            )
        ]

        spans = [
            (pd.Timestamp("2024-01-01 00:10:00"), pd.Timestamp("2024-01-01 00:20:00")),
            (pd.Timestamp("2024-01-01 00:30:00"), pd.Timestamp("2024-01-01 00:40:00")),
        ]

        count = mixin._count_overlapping_intervals(spans)
        assert count == 1  # Same interval overlapped by both spans

    def test_multiple_overlaps_different_intervals(self, mixin):
        """Multiple spans overlapping different intervals should count all."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 00:15:00"),
                "A"
            ),
            Interval(
                pd.Timestamp("2024-01-01 00:25:00"),
                pd.Timestamp("2024-01-01 00:35:00"),
                "B"
            ),
            Interval(
                pd.Timestamp("2024-01-01 00:45:00"),
                pd.Timestamp("2024-01-01 00:55:00"),
                "C"
            ),
        ]

        spans = [
            (pd.Timestamp("2024-01-01 00:10:00"), pd.Timestamp("2024-01-01 00:20:00")),  # Overlaps A
            (pd.Timestamp("2024-01-01 00:30:00"), pd.Timestamp("2024-01-01 00:50:00")),  # Overlaps B and C
        ]

        count = mixin._count_overlapping_intervals(spans)
        assert count == 3  # A, B, and C all overlap

    def test_partial_overlap_counts(self, mixin):
        """Partial overlaps should be counted."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:10:00"),
                pd.Timestamp("2024-01-01 00:20:00"),
                "A"
            )
        ]

        # Test left edge overlap
        spans = [
            (pd.Timestamp("2024-01-01 00:05:00"), pd.Timestamp("2024-01-01 00:15:00"))
        ]
        count = mixin._count_overlapping_intervals(spans)
        assert count == 1

        # Test right edge overlap
        spans = [
            (pd.Timestamp("2024-01-01 00:15:00"), pd.Timestamp("2024-01-01 00:25:00"))
        ]
        count = mixin._count_overlapping_intervals(spans)
        assert count == 1

    def test_complete_overlap_counts(self, mixin):
        """Span completely covering interval should count."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:15:00"),
                pd.Timestamp("2024-01-01 00:25:00"),
                "A"
            )
        ]

        spans = [
            (pd.Timestamp("2024-01-01 00:10:00"), pd.Timestamp("2024-01-01 00:30:00"))
        ]

        count = mixin._count_overlapping_intervals(spans)
        assert count == 1

    def test_adjacent_not_overlap(self, mixin):
        """Adjacent but non-overlapping intervals should not count."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 00:10:00"),
                "A"
            )
        ]

        # Span starts exactly where interval ends (half-open semantics)
        spans = [
            (pd.Timestamp("2024-01-01 00:10:00"), pd.Timestamp("2024-01-01 00:20:00"))
        ]

        count = mixin._count_overlapping_intervals(spans)
        assert count == 0


# ============================================================================
# Test: Skip Policy - Subtract Overlaps (_subtract_overlaps_from_span)
# ============================================================================

class TestSkipPolicy:
    """Test the skip policy logic (subtract overlaps from span)."""

    def test_no_existing_intervals(self, mixin):
        """No existing intervals should return the full span."""
        mixin.intervals = []

        s = pd.Timestamp("2024-01-01 00:00:00")
        e = pd.Timestamp("2024-01-01 01:00:00")

        result = mixin._subtract_overlaps_from_span(s, e)

        assert len(result) == 1
        assert result[0] == (s, e)

    def test_left_edge_overlap(self, mixin):
        """Left edge overlap should return right portion only."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 00:30:00"),
                "A"
            )
        ]

        s = pd.Timestamp("2024-01-01 00:20:00")
        e = pd.Timestamp("2024-01-01 01:00:00")

        result = mixin._subtract_overlaps_from_span(s, e)

        assert len(result) == 1
        assert result[0] == (pd.Timestamp("2024-01-01 00:30:00"), e)

    def test_right_edge_overlap(self, mixin):
        """Right edge overlap should return left portion only."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:30:00"),
                pd.Timestamp("2024-01-01 01:00:00"),
                "A"
            )
        ]

        s = pd.Timestamp("2024-01-01 00:00:00")
        e = pd.Timestamp("2024-01-01 00:45:00")

        result = mixin._subtract_overlaps_from_span(s, e)

        assert len(result) == 1
        assert result[0] == (s, pd.Timestamp("2024-01-01 00:30:00"))

    def test_middle_hole(self, mixin):
        """Existing interval inside span should create two gaps."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:20:00"),
                pd.Timestamp("2024-01-01 00:40:00"),
                "A"
            )
        ]

        s = pd.Timestamp("2024-01-01 00:00:00")
        e = pd.Timestamp("2024-01-01 01:00:00")

        result = mixin._subtract_overlaps_from_span(s, e)

        assert len(result) == 2
        assert result[0] == (s, pd.Timestamp("2024-01-01 00:20:00"))
        assert result[1] == (pd.Timestamp("2024-01-01 00:40:00"), e)

    def test_multiple_existing_intervals(self, mixin):
        """Multiple existing intervals should return all gaps between them."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:10:00"),
                pd.Timestamp("2024-01-01 00:20:00"),
                "A"
            ),
            Interval(
                pd.Timestamp("2024-01-01 00:30:00"),
                pd.Timestamp("2024-01-01 00:40:00"),
                "B"
            ),
            Interval(
                pd.Timestamp("2024-01-01 00:50:00"),
                pd.Timestamp("2024-01-01 00:55:00"),
                "C"
            ),
        ]

        s = pd.Timestamp("2024-01-01 00:00:00")
        e = pd.Timestamp("2024-01-01 01:00:00")

        result = mixin._subtract_overlaps_from_span(s, e)

        assert len(result) == 4
        assert result[0] == (s, pd.Timestamp("2024-01-01 00:10:00"))
        assert result[1] == (pd.Timestamp("2024-01-01 00:20:00"), pd.Timestamp("2024-01-01 00:30:00"))
        assert result[2] == (pd.Timestamp("2024-01-01 00:40:00"), pd.Timestamp("2024-01-01 00:50:00"))
        assert result[3] == (pd.Timestamp("2024-01-01 00:55:00"), e)

    def test_fully_covered_span(self, mixin):
        """Span fully covered by existing interval should return empty list."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 02:00:00"),
                "A"
            )
        ]

        s = pd.Timestamp("2024-01-01 00:30:00")
        e = pd.Timestamp("2024-01-01 01:00:00")

        result = mixin._subtract_overlaps_from_span(s, e)

        assert len(result) == 0

    def test_adjacent_not_overlapping(self, mixin):
        """Adjacent but non-overlapping intervals should return full span."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 00:30:00"),
                "A"
            ),
            Interval(
                pd.Timestamp("2024-01-01 01:00:00"),
                pd.Timestamp("2024-01-01 01:30:00"),
                "B"
            ),
        ]

        s = pd.Timestamp("2024-01-01 00:30:00")
        e = pd.Timestamp("2024-01-01 01:00:00")

        result = mixin._subtract_overlaps_from_span(s, e)

        assert len(result) == 1
        assert result[0] == (s, e)

    def test_invalid_span(self, mixin):
        """Span where end <= start should return empty list."""
        mixin.intervals = []

        s = pd.Timestamp("2024-01-01 01:00:00")
        e = pd.Timestamp("2024-01-01 00:00:00")

        result = mixin._subtract_overlaps_from_span(s, e)

        assert len(result) == 0

    def test_overlapping_existing_intervals(self, mixin):
        """Overlapping existing intervals should be merged before subtracting."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:10:00"),
                pd.Timestamp("2024-01-01 00:25:00"),
                "A"
            ),
            Interval(
                pd.Timestamp("2024-01-01 00:20:00"),
                pd.Timestamp("2024-01-01 00:35:00"),
                "B"
            ),
        ]

        s = pd.Timestamp("2024-01-01 00:00:00")
        e = pd.Timestamp("2024-01-01 01:00:00")

        result = mixin._subtract_overlaps_from_span(s, e)

        # Should have two gaps: before and after the merged interval
        assert len(result) == 2
        assert result[0] == (s, pd.Timestamp("2024-01-01 00:10:00"))
        assert result[1] == (pd.Timestamp("2024-01-01 00:35:00"), e)


# ============================================================================
# Test: Replace Policy - Carve Existing (_carve_existing_for_new_span)
# ============================================================================

class TestReplacePolicy:
    """Test the replace policy logic (carve existing intervals)."""

    def test_fully_covered_interval_deleted(self, mixin):
        """Interval fully covered by new span should be deleted."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:20:00"),
                pd.Timestamp("2024-01-01 00:30:00"),
                "A"
            )
        ]

        s = pd.Timestamp("2024-01-01 00:10:00")
        e = pd.Timestamp("2024-01-01 00:40:00")

        mixin._carve_existing_for_new_span(s, e)

        assert len(mixin.intervals) == 0

    def test_left_edge_overlap_truncated(self, mixin):
        """Left edge overlap should truncate interval to keep left part."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 00:30:00"),
                "A"
            )
        ]

        s = pd.Timestamp("2024-01-01 00:20:00")
        e = pd.Timestamp("2024-01-01 00:50:00")

        mixin._carve_existing_for_new_span(s, e)

        assert len(mixin.intervals) == 1
        assert mixin.intervals[0].start == pd.Timestamp("2024-01-01 00:00:00")
        assert mixin.intervals[0].end == pd.Timestamp("2024-01-01 00:20:00")
        assert mixin.intervals[0].label == "A"

    def test_right_edge_overlap_truncated(self, mixin):
        """Right edge overlap should truncate interval to keep right part."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:30:00"),
                pd.Timestamp("2024-01-01 01:00:00"),
                "A"
            )
        ]

        s = pd.Timestamp("2024-01-01 00:00:00")
        e = pd.Timestamp("2024-01-01 00:40:00")

        mixin._carve_existing_for_new_span(s, e)

        assert len(mixin.intervals) == 1
        assert mixin.intervals[0].start == pd.Timestamp("2024-01-01 00:40:00")
        assert mixin.intervals[0].end == pd.Timestamp("2024-01-01 01:00:00")
        assert mixin.intervals[0].label == "A"

    def test_span_cuts_through_middle_splits(self, mixin):
        """Span cutting through middle should split interval into two."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 01:00:00"),
                "A",
                "Note"
            )
        ]

        s = pd.Timestamp("2024-01-01 00:20:00")
        e = pd.Timestamp("2024-01-01 00:40:00")

        mixin._carve_existing_for_new_span(s, e)

        assert len(mixin.intervals) == 2

        # Left part
        left = [iv for iv in mixin.intervals if iv.start < s][0]
        assert left.start == pd.Timestamp("2024-01-01 00:00:00")
        assert left.end == pd.Timestamp("2024-01-01 00:20:00")
        assert left.label == "A"
        assert left.notes == "Note"

        # Right part
        right = [iv for iv in mixin.intervals if iv.start >= e][0]
        assert right.start == pd.Timestamp("2024-01-01 00:40:00")
        assert right.end == pd.Timestamp("2024-01-01 01:00:00")
        assert right.label == "A"
        assert right.notes == "Note"

    def test_no_overlap_unchanged(self, mixin):
        """Interval with no overlap should remain unchanged."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 00:10:00"),
                "A"
            )
        ]

        s = pd.Timestamp("2024-01-01 00:20:00")
        e = pd.Timestamp("2024-01-01 00:30:00")

        mixin._carve_existing_for_new_span(s, e)

        assert len(mixin.intervals) == 1
        assert mixin.intervals[0].start == pd.Timestamp("2024-01-01 00:00:00")
        assert mixin.intervals[0].end == pd.Timestamp("2024-01-01 00:10:00")

    def test_multiple_intervals_affected(self, mixin):
        """Multiple intervals should all be carved correctly."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 00:15:00"),
                "A"
            ),
            Interval(
                pd.Timestamp("2024-01-01 00:20:00"),
                pd.Timestamp("2024-01-01 00:35:00"),
                "B"
            ),
            Interval(
                pd.Timestamp("2024-01-01 00:40:00"),
                pd.Timestamp("2024-01-01 01:00:00"),
                "C"
            ),
        ]

        s = pd.Timestamp("2024-01-01 00:10:00")
        e = pd.Timestamp("2024-01-01 00:50:00")

        mixin._carve_existing_for_new_span(s, e)

        # A should be truncated to keep left part
        a_intervals = [iv for iv in mixin.intervals if iv.label == "A"]
        assert len(a_intervals) == 1
        assert a_intervals[0].end == pd.Timestamp("2024-01-01 00:10:00")

        # B should be deleted (fully covered)
        b_intervals = [iv for iv in mixin.intervals if iv.label == "B"]
        assert len(b_intervals) == 0

        # C should be truncated to keep right part
        c_intervals = [iv for iv in mixin.intervals if iv.label == "C"]
        assert len(c_intervals) == 1
        assert c_intervals[0].start == pd.Timestamp("2024-01-01 00:50:00")

    def test_invalid_span_no_changes(self, mixin):
        """Invalid span (end <= start) should make no changes."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 00:30:00"),
                "A"
            )
        ]

        s = pd.Timestamp("2024-01-01 00:30:00")
        e = pd.Timestamp("2024-01-01 00:10:00")

        mixin._carve_existing_for_new_span(s, e)

        assert len(mixin.intervals) == 1
        assert mixin.intervals[0].start == pd.Timestamp("2024-01-01 00:00:00")
        assert mixin.intervals[0].end == pd.Timestamp("2024-01-01 00:30:00")

    def test_adjacent_intervals_unchanged(self, mixin):
        """Adjacent intervals (touching but not overlapping) should remain unchanged."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 00:20:00"),
                "A"
            ),
            Interval(
                pd.Timestamp("2024-01-01 00:40:00"),
                pd.Timestamp("2024-01-01 01:00:00"),
                "B"
            ),
        ]

        s = pd.Timestamp("2024-01-01 00:20:00")
        e = pd.Timestamp("2024-01-01 00:40:00")

        mixin._carve_existing_for_new_span(s, e)

        assert len(mixin.intervals) == 2
        # Both should be unchanged
        a_iv = [iv for iv in mixin.intervals if iv.label == "A"][0]
        assert a_iv.start == pd.Timestamp("2024-01-01 00:00:00")
        assert a_iv.end == pd.Timestamp("2024-01-01 00:20:00")

        b_iv = [iv for iv in mixin.intervals if iv.label == "B"][0]
        assert b_iv.start == pd.Timestamp("2024-01-01 00:40:00")
        assert b_iv.end == pd.Timestamp("2024-01-01 01:00:00")


# ============================================================================
# Test: Policy Application (_apply_overlap_policy_to_spans)
# ============================================================================

class TestPolicyApplication:
    """Test applying overlap policies to multiple spans."""

    def test_skip_policy_removes_overlaps(self, mixin):
        """Skip policy should remove overlapping portions from spans."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:15:00"),
                pd.Timestamp("2024-01-01 00:25:00"),
                "A"
            )
        ]

        spans = [
            (pd.Timestamp("2024-01-01 00:00:00"), pd.Timestamp("2024-01-01 00:30:00"))
        ]

        result = mixin._apply_overlap_policy_to_spans(spans, "skip")

        assert len(result) == 2
        assert result[0] == (pd.Timestamp("2024-01-01 00:00:00"), pd.Timestamp("2024-01-01 00:15:00"))
        assert result[1] == (pd.Timestamp("2024-01-01 00:25:00"), pd.Timestamp("2024-01-01 00:30:00"))

    def test_replace_policy_unchanged(self, mixin):
        """Replace policy should leave spans unchanged (carving happens at add-time)."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:15:00"),
                pd.Timestamp("2024-01-01 00:25:00"),
                "A"
            )
        ]

        spans = [
            (pd.Timestamp("2024-01-01 00:00:00"), pd.Timestamp("2024-01-01 00:30:00"))
        ]

        result = mixin._apply_overlap_policy_to_spans(spans, "replace")

        assert len(result) == 1
        assert result[0] == spans[0]

    def test_unknown_policy_passthrough(self, mixin):
        """Unknown policy should pass through spans unchanged."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:15:00"),
                pd.Timestamp("2024-01-01 00:25:00"),
                "A"
            )
        ]

        spans = [
            (pd.Timestamp("2024-01-01 00:00:00"), pd.Timestamp("2024-01-01 00:30:00"))
        ]

        result = mixin._apply_overlap_policy_to_spans(spans, "unknown")

        assert len(result) == 1
        assert result[0] == spans[0]

    def test_empty_policy_passthrough(self, mixin):
        """Empty/None policy should pass through spans unchanged."""
        mixin.intervals = []

        spans = [
            (pd.Timestamp("2024-01-01 00:00:00"), pd.Timestamp("2024-01-01 00:30:00"))
        ]

        result = mixin._apply_overlap_policy_to_spans(spans, "")
        assert result == spans

        result = mixin._apply_overlap_policy_to_spans(spans, None)
        assert result == spans

    def test_skip_policy_multiple_spans(self, mixin):
        """Skip policy should handle multiple spans correctly."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:15:00"),
                pd.Timestamp("2024-01-01 00:25:00"),
                "A"
            ),
            Interval(
                pd.Timestamp("2024-01-01 00:35:00"),
                pd.Timestamp("2024-01-01 00:45:00"),
                "B"
            ),
        ]

        spans = [
            (pd.Timestamp("2024-01-01 00:00:00"), pd.Timestamp("2024-01-01 00:30:00")),
            (pd.Timestamp("2024-01-01 00:30:00"), pd.Timestamp("2024-01-01 00:50:00")),
        ]

        result = mixin._apply_overlap_policy_to_spans(spans, "skip")

        # Should have 4 pieces total:
        # From first span: [0:00-0:15], [0:25-0:30]
        # From second span: [0:30-0:35], [0:45-0:50]
        assert len(result) == 4

    def test_skip_policy_fully_covered_spans(self, mixin):
        """Skip policy with fully covered spans should return empty list."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 02:00:00"),
                "A"
            )
        ]

        spans = [
            (pd.Timestamp("2024-01-01 00:30:00"), pd.Timestamp("2024-01-01 01:00:00"))
        ]

        result = mixin._apply_overlap_policy_to_spans(spans, "skip")

        assert len(result) == 0

    def test_case_insensitive_policy(self, mixin):
        """Policy should be case-insensitive."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:15:00"),
                pd.Timestamp("2024-01-01 00:25:00"),
                "A"
            )
        ]

        spans = [
            (pd.Timestamp("2024-01-01 00:00:00"), pd.Timestamp("2024-01-01 00:30:00"))
        ]

        # Test various cases
        for policy in ["SKIP", "Skip", "SkIp"]:
            result = mixin._apply_overlap_policy_to_spans(spans, policy)
            assert len(result) == 2

        for policy in ["REPLACE", "Replace", "RePLaCe"]:
            result = mixin._apply_overlap_policy_to_spans(spans, policy)
            assert len(result) == 1
            assert result[0] == spans[0]

    def test_skip_policy_is_idempotent(self, mixin):
        """
        Applying the skip policy twice returns the same result as applying
        it once.  The by-rule preview fix (commit c8c15a1) carves spans
        through this helper at preview-time and then `_add_intervals_with_policy`
        carves them again at add-time -- so this idempotency is a load-bearing
        invariant: the second carve must be a no-op.
        """
        mixin.intervals = [
            # Two existing intervals carving holes into the candidate spans
            Interval(
                pd.Timestamp("2024-01-01 00:10:00"),
                pd.Timestamp("2024-01-01 00:20:00"),
                "A",
            ),
            Interval(
                pd.Timestamp("2024-01-01 00:35:00"),
                pd.Timestamp("2024-01-01 00:45:00"),
                "B",
            ),
        ]

        candidate_spans = [
            # Spans the rule preview would generate, intersecting both intervals
            (pd.Timestamp("2024-01-01 00:00:00"), pd.Timestamp("2024-01-01 00:25:00")),
            (pd.Timestamp("2024-01-01 00:30:00"), pd.Timestamp("2024-01-01 00:50:00")),
        ]

        once = mixin._apply_overlap_policy_to_spans(candidate_spans, "skip")
        twice = mixin._apply_overlap_policy_to_spans(once, "skip")

        # The second pass must not carve anything further.
        assert twice == once

        # And the carve must have actually happened on the first pass --
        # otherwise this test would be vacuously true.
        assert once != candidate_spans


# ============================================================================
# Integration Tests
# ============================================================================

class TestOverlapIntegration:
    """Integration tests combining multiple overlap operations."""

    def test_skip_then_count(self, mixin):
        """Test skip policy followed by overlap count."""
        mixin.intervals = [
            Interval(
                pd.Timestamp("2024-01-01 00:20:00"),
                pd.Timestamp("2024-01-01 00:40:00"),
                "A"
            )
        ]

        # Original span overlaps
        original_spans = [
            (pd.Timestamp("2024-01-01 00:00:00"), pd.Timestamp("2024-01-01 01:00:00"))
        ]
        count_before = mixin._count_overlapping_intervals(original_spans)
        assert count_before == 1

        # After skip policy, result should not overlap
        result_spans = mixin._apply_overlap_policy_to_spans(original_spans, "skip")
        count_after = mixin._count_overlapping_intervals(result_spans)
        assert count_after == 0

    def test_carve_multiple_scenarios(self, mixin):
        """
        Test carving with various overlap scenarios in one operation.

        Uses non-overlapping initial intervals to test different carving cases
        without interference from automatic overlap resolution.
        """
        mixin.intervals = [
            # Will be deleted (fully covered)
            Interval(
                pd.Timestamp("2024-01-01 00:20:00"),
                pd.Timestamp("2024-01-01 00:30:00"),
                "Delete"
            ),
            # Will be unchanged (no overlap)
            Interval(
                pd.Timestamp("2024-01-01 01:10:00"),
                pd.Timestamp("2024-01-01 01:20:00"),
                "Unchanged"
            ),
        ]

        s = pd.Timestamp("2024-01-01 00:10:00")
        e = pd.Timestamp("2024-01-01 00:50:00")

        initial_count = len(mixin.intervals)
        mixin._carve_existing_for_new_span(s, e)

        # Check results
        labels = {iv.label for iv in mixin.intervals}

        # Delete should be gone (it was fully covered)
        assert "Delete" not in labels

        # Unchanged should still exist and be untouched
        unchanged = [iv for iv in mixin.intervals if iv.label == "Unchanged"][0]
        assert unchanged.start == pd.Timestamp("2024-01-01 01:10:00")
        assert unchanged.end == pd.Timestamp("2024-01-01 01:20:00")

        # Only one interval should remain (Unchanged)
        assert len(mixin.intervals) == 1
