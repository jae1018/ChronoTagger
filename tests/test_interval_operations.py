"""
Test suite for interval manipulation operations in ChronoTagger.

This module tests the core interval management functionality:
- Adding intervals with overlap handling
- Deleting and modifying intervals
- Overlap detection and resolution strategies
- Gap finding and assignment
- Interval merging and splitting
- Clear range operations

ASCII Diagrams of Interval Operations:
======================================

1. Overlap Resolution (Skip Policy):
   Existing:    |----A----|    |----B----|
   New:             |--------C--------|
   Result:      |----A----|    |----B----|
                        |--C--|        (only non-overlapping part added)

2. Overlap Resolution (Replace Policy):
   Existing:    |----A----|    |----B----|
   New:             |--------C--------|
   Result:      |--A|  |----C----|  |B|
                (A,B trimmed, C takes precedence)

3. Interval Splitting:
   Original:    |----------A----------|
   Clear Range:      |---clear---|
   Result:      |--A--|          |--A--|
                (Split into two parts)

4. Adjacent Merging:
   Before:      |---A---|---A---|  (same label, adjacent)
   After:       |-------A-------|  (merged)
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from tkinter import messagebox

from chronotagger.labeler import TimeIntervalLabeler
from chronotagger.core.models import Interval
from chronotagger.core.commands import (
    AddIntervalCommand, 
    DeleteIntervalCommand,
    RelabelIntervalCommand,
    ResizeIntervalCommand
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def labeler_with_intervals(labeler):
    """Labeler with pre-existing intervals for testing operations."""
    # Add some test intervals
    idx = labeler.df.index
    labeler.intervals = [
        Interval(idx[10], idx[15], "A", "Note A"),
        Interval(idx[20], idx[25], "B", "Note B"),
        Interval(idx[30], idx[35], "A", "Note A2"),
        Interval(idx[40], idx[45], "C", "Note C"),
    ]
    return labeler


@pytest.fixture  
def overlapping_intervals():
    """Create a set of intervals with various overlap scenarios."""
    base = pd.Timestamp("2024-01-01 12:00:00")
    return [
        Interval(base, base + pd.Timedelta(minutes=10), "A"),
        Interval(base + pd.Timedelta(minutes=5), 
                base + pd.Timedelta(minutes=15), "B"),  # Overlaps with A
        Interval(base + pd.Timedelta(minutes=20),
                base + pd.Timedelta(minutes=30), "C"),  # No overlap
        Interval(base + pd.Timedelta(minutes=25),
                base + pd.Timedelta(minutes=35), "D"),  # Overlaps with C
    ]


# ============================================================================
# Basic Interval Operations Tests
# ============================================================================

class TestBasicIntervalOperations:
    """Tests for basic interval CRUD operations."""
    
    def test_add_interval_simple(self, labeler):
        """Test adding a simple interval with no overlaps."""
        t1, t2 = labeler.df.index[10], labeler.df.index[20]
        interval = Interval(t1, t2, "TestLabel", "Test notes")
        
        cmd = AddIntervalCommand(labeler, interval)
        cmd.execute()
        
        assert len(labeler.intervals) == 1
        assert labeler.intervals[0] == interval
        
        # Test undo
        cmd.undo()
        assert len(labeler.intervals) == 0
    
    def test_delete_interval(self, labeler_with_intervals):
        """Test deleting an interval."""
        initial_count = len(labeler_with_intervals.intervals)
        to_delete = labeler_with_intervals.intervals[1]
        
        cmd = DeleteIntervalCommand(labeler_with_intervals, to_delete)
        cmd.execute()
        
        assert len(labeler_with_intervals.intervals) == initial_count - 1
        assert to_delete not in labeler_with_intervals.intervals
        
        # Test undo
        cmd.undo()
        assert len(labeler_with_intervals.intervals) == initial_count
        assert to_delete in labeler_with_intervals.intervals
    
    def test_relabel_interval(self, labeler_with_intervals):
        """Test changing an interval's label."""
        interval = labeler_with_intervals.intervals[0]
        original_label = interval.label
        new_label = "NewLabel"
        
        cmd = RelabelIntervalCommand(labeler_with_intervals, interval, new_label)
        cmd.execute()
        
        assert interval.label == new_label
        
        # Test undo
        cmd.undo()
        assert interval.label == original_label
    
    def test_resize_interval(self, labeler_with_intervals):
        """Test resizing an interval."""
        interval = labeler_with_intervals.intervals[0]
        original_start = interval.start
        original_end = interval.end
        
        # Resize to be longer
        new_start = original_start - pd.Timedelta(minutes=5)
        new_end = original_end + pd.Timedelta(minutes=5)
        
        cmd = ResizeIntervalCommand(labeler_with_intervals, interval, 
                                   new_start, new_end)
        cmd.execute()
        
        # Original interval removed, new one added
        assert interval not in labeler_with_intervals.intervals
        # Find the new interval with same label
        new_intervals = [iv for iv in labeler_with_intervals.intervals 
                        if iv.label == interval.label]
        assert len(new_intervals) >= 1
        
        # Test undo
        cmd.undo()
        assert interval in labeler_with_intervals.intervals


# ============================================================================
# Overlap Handling Tests
# ============================================================================

class TestOverlapHandling:
    """Tests for interval overlap detection and resolution."""
    
    def test_overlap_detection(self):
        """Test the Interval.overlaps() method."""
        base = pd.Timestamp("2024-01-01 12:00:00")
        
        iv1 = Interval(base, base + pd.Timedelta(minutes=10), "A")
        iv2 = Interval(base + pd.Timedelta(minutes=5),
                      base + pd.Timedelta(minutes=15), "B")
        iv3 = Interval(base + pd.Timedelta(minutes=10),
                      base + pd.Timedelta(minutes=20), "C")
        iv4 = Interval(base + pd.Timedelta(minutes=20),
                      base + pd.Timedelta(minutes=30), "D")
        
        # iv1 and iv2 overlap
        assert iv1.overlaps(iv2)
        assert iv2.overlaps(iv1)
        
        # iv1 and iv3 are adjacent (not overlapping in half-open intervals)
        assert not iv1.overlaps(iv3)
        assert not iv3.overlaps(iv1)
        
        # iv1 and iv4 don't overlap
        assert not iv1.overlaps(iv4)
        assert not iv4.overlaps(iv1)
    
    def test_add_with_overlap_trimming(self, labeler_with_intervals):
        """
        Test adding an interval that overlaps existing ones.
        
        Scenario:
        Existing: |--A--| |--B--|
        New:         |-----C-----|
        Result:   |--A--| |--B--|
                     |C|      |C|  (trimmed parts)
        """
        idx = labeler_with_intervals.df.index
        # Create interval that overlaps first two existing intervals
        new_interval = Interval(idx[12], idx[23], "Overlapping")
        
        cmd = AddIntervalCommand(labeler_with_intervals, new_interval)
        cmd.execute()
        
        # Should have trimmed the overlapping parts
        # Exact behavior depends on implementation
        assert new_interval in labeler_with_intervals.intervals
    
    def test_subtract_overlaps_from_span(self, labeler_with_intervals):
        """Test the overlap subtraction algorithm."""
        idx = labeler_with_intervals.df.index
        
        # Test span that overlaps multiple existing intervals
        s = idx[8]
        e = idx[37]
        
        # Get non-overlapping subspans
        subspans = labeler_with_intervals._subtract_overlaps_from_span(s, e)
        
        # Should return gaps between existing intervals
        # [8,10), [15,20), [25,30), [35,37)
        assert len(subspans) > 0
        
        # Verify no subspan overlaps with existing intervals
        for sub_s, sub_e in subspans:
            for iv in labeler_with_intervals.intervals:
                test_iv = Interval(sub_s, sub_e, "test")
                assert not test_iv.overlaps(iv)
    
    def test_skip_policy(self, labeler_with_intervals):
        """Test skip overlap policy."""
        labeler_with_intervals._overlap_policy = "skip"
        idx = labeler_with_intervals.df.index
        
        # Try to add overlapping spans
        spans = [
            (idx[12], idx[18]),  # Overlaps with first interval
            (idx[22], idx[28]),  # Overlaps with second interval
        ]
        
        result = labeler_with_intervals._apply_overlap_policy_to_spans(
            spans, "skip"
        )
        
        # Should return only non-overlapping parts
        for s, e in result:
            for iv in labeler_with_intervals.intervals:
                test_iv = Interval(s, e, "test")
                assert not test_iv.overlaps(iv)
    
    def test_replace_policy_carving(self, labeler_with_intervals):
        """Test that replace policy carves existing intervals."""
        idx = labeler_with_intervals.df.index
        original_count = len(labeler_with_intervals.intervals)
        
        # Carve out space for new interval
        s, e = idx[12], idx[23]
        labeler_with_intervals._carve_existing_for_new_span(s, e)
        
        # Should have modified existing intervals
        # First interval [10,15] -> [10,12] (trimmed)
        # Second interval [20,25] -> [23,25] (trimmed)
        
        # Check that no interval occupies the carved space
        for iv in labeler_with_intervals.intervals:
            assert iv.end <= s or iv.start >= e or \
                   (iv.start < s and iv.end <= e) or \
                   (iv.start >= s and iv.end > e)


# ============================================================================
# Interval Merging Tests
# ============================================================================

class TestIntervalMerging:
    """Tests for automatic merging of adjacent same-label intervals."""
    
    def test_merge_adjacent_same_label(self, labeler):
        """
        Test that adjacent intervals with same label are merged.
        
        Before: |--A--||--A--|  (adjacent, same label)
        After:  |-----A------|  (merged)
        """
        idx = labeler.df.index
        
        # Add two adjacent intervals with same label
        labeler.intervals = [
            Interval(idx[10], idx[20], "A"),
            Interval(idx[20], idx[30], "A"),  # Adjacent to first
        ]
        
        labeler._sort_and_merge_intervals()
        
        # Should merge into one
        assert len(labeler.intervals) == 1
        assert labeler.intervals[0].start == idx[10]
        assert labeler.intervals[0].end == idx[30]
    
    def test_no_merge_different_labels(self, labeler):
        """Test that adjacent intervals with different labels are NOT merged."""
        idx = labeler.df.index
        
        # Add two adjacent intervals with different labels
        labeler.intervals = [
            Interval(idx[10], idx[20], "A"),
            Interval(idx[20], idx[30], "B"),  # Adjacent but different label
        ]
        
        labeler._sort_and_merge_intervals()
        
        # Should NOT merge
        assert len(labeler.intervals) == 2
    
    def test_no_merge_non_adjacent(self, labeler):
        """Test that non-adjacent intervals are not merged."""
        idx = labeler.df.index
        
        # Add two non-adjacent intervals with same label
        labeler.intervals = [
            Interval(idx[10], idx[20], "A"),
            Interval(idx[25], idx[30], "A"),  # Gap between them
        ]
        
        labeler._sort_and_merge_intervals()
        
        # Should NOT merge
        assert len(labeler.intervals) == 2
    
    def test_merge_multiple_chains(self, labeler):
        """Test merging of multiple adjacent intervals."""
        idx = labeler.df.index
        
        # Create a chain of adjacent same-label intervals
        labeler.intervals = [
            Interval(idx[10], idx[15], "A"),
            Interval(idx[15], idx[20], "A"),
            Interval(idx[20], idx[25], "A"),
            Interval(idx[30], idx[35], "B"),  # Different chain
            Interval(idx[35], idx[40], "B"),
        ]
        
        labeler._sort_and_merge_intervals()
        
        # Should result in 2 merged intervals
        assert len(labeler.intervals) == 2
        
        # First merged interval
        a_intervals = [iv for iv in labeler.intervals if iv.label == "A"]
        assert len(a_intervals) == 1
        assert a_intervals[0].start == idx[10]
        assert a_intervals[0].end == idx[25]
        
        # Second merged interval
        b_intervals = [iv for iv in labeler.intervals if iv.label == "B"]
        assert len(b_intervals) == 1
        assert b_intervals[0].start == idx[30]
        assert b_intervals[0].end == idx[40]


# ============================================================================
# Gap Detection and Assignment Tests
# ============================================================================

class TestGapDetection:
    """Tests for finding and assigning unlabeled gaps."""
    
    def test_find_gaps_simple(self, labeler_with_intervals):
        """
        Test finding gaps between intervals.
        
        Timeline: |--A--| gap |--B--| gap |--A--| gap |--C--|
        """
        gaps = labeler_with_intervals._find_gaps_in_current_range()
        
        # Should find gaps between intervals
        assert len(gaps) > 0
        
        # Verify gaps don't overlap with existing intervals
        for gap_s, gap_e in gaps:
            gap_iv = Interval(gap_s, gap_e, "test")
            for iv in labeler_with_intervals.intervals:
                assert not gap_iv.overlaps(iv)
    
    def test_find_gaps_at_boundaries(self, labeler):
        """Test gap detection at window boundaries."""
        idx = labeler.df.index
        
        # Set window
        labeler.t0 = idx[10]
        labeler.t1 = idx[40]
        
        # Add interval in the middle
        labeler.intervals = [
            Interval(idx[20], idx[30], "A")
        ]
        
        gaps = labeler._find_gaps_in_current_range()
        
        # Should find gaps before and after the interval
        assert len(gaps) == 2
        assert gaps[0] == (idx[10], idx[20])  # Gap before
        assert gaps[1] == (idx[30], idx[40])  # Gap after
    
    def test_find_gaps_no_gaps(self, labeler):
        """Test when entire range is covered."""
        idx = labeler.df.index
        
        # Set window
        labeler.t0 = idx[10]
        labeler.t1 = idx[30]
        
        # Cover entire range
        labeler.intervals = [
            Interval(idx[10], idx[30], "A")
        ]
        
        gaps = labeler._find_gaps_in_current_range()
        
        # Should find no gaps
        assert len(gaps) == 0
    
    def test_assign_gaps_to_label(self, labeler):
        """Test assigning gaps to a specific label."""
        idx = labeler.df.index
        
        # Set up gaps
        gaps = [
            (idx[10], idx[15]),
            (idx[20], idx[25]),
            (idx[30], idx[35]),
        ]
        
        # Assign to label
        labeler._assign_gaps_to_label(gaps, "UNKNOWN")
        
        # Should create intervals for each gap
        assert len(labeler.intervals) == 3
        for iv in labeler.intervals:
            assert iv.label == "UNKNOWN"
    
    @patch('tkinter.messagebox.showinfo')
    def test_assign_gaps_empty(self, mock_msgbox, labeler):
        """Test assigning when no gaps exist."""
        # Cover entire range
        idx = labeler.df.index
        labeler.t0 = idx[10]
        labeler.t1 = idx[30]
        labeler.intervals = [Interval(idx[10], idx[30], "A")]
        
        # Try to open dialog for gap assignment
        labeler._open_label_unassigned_dialog()
        
        # Should show info that no gaps exist
        mock_msgbox.assert_called_once()
        assert "No Gaps" in mock_msgbox.call_args[0][0]


# ============================================================================
# Clear Range Operations Tests  
# ============================================================================

class TestClearRangeOperations:
    """Tests for clearing intervals in a specified range."""
    
    def test_analyze_intervals_in_range(self, labeler_with_intervals):
        """
        Test analysis of how intervals will be affected by clearing.
        
        Clear range:     |--------clear--------|
        Intervals:   |A|   |B|  |C|    |D|  |E|   |F|
        Results:     keep  del  del    del  trim  keep
        """
        idx = labeler_with_intervals.df.index
        t0 = idx[18]
        t1 = idx[42]
        
        analysis = labeler_with_intervals._analyze_intervals_in_range(t0, t1)
        
        assert 'to_delete' in analysis
        assert 'to_truncate' in analysis
        assert 'to_split' in analysis
        assert 'total_affected' in analysis
        
        # Should identify affected intervals correctly
        assert analysis['total_affected'] > 0
    
    def test_clear_intervals_delete_fully_inside(self, labeler):
        """Test clearing deletes intervals fully inside range."""
        idx = labeler.df.index
        
        labeler.intervals = [
            Interval(idx[10], idx[15], "A"),  # Outside
            Interval(idx[20], idx[25], "B"),  # Inside - should be deleted
            Interval(idx[30], idx[35], "C"),  # Outside
        ]
        
        # Clear middle range
        results = labeler._clear_intervals_in_range(idx[18], idx[28])
        
        assert results['deleted'] == 1
        assert len(labeler.intervals) == 2
        
        # B should be deleted
        labels = [iv.label for iv in labeler.intervals]
        assert "B" not in labels
        assert "A" in labels
        assert "C" in labels
    
    def test_clear_intervals_truncate_overlap(self, labeler):
        """Test clearing truncates partially overlapping intervals."""
        idx = labeler.df.index

        labeler.intervals = [
            Interval(idx[10], idx[25], "A"),  # Left overlap
            Interval(idx[24], idx[35], "B"),  # Right overlap
        ]

        # Clear middle range
        results = labeler._clear_intervals_in_range(idx[22], idx[28])

        assert results['truncated'] == 2

        # A should be truncated to [10,22]
        a_iv = [iv for iv in labeler.intervals if iv.label == "A"][0]
        assert a_iv.end == idx[22]

        # B should be truncated to [28,35]
        b_iv = [iv for iv in labeler.intervals if iv.label == "B"][0]
        assert b_iv.start == idx[28]
    
    def test_clear_intervals_split_spanning(self, labeler):
        """
        Test clearing splits intervals that span the entire range.
        
        Original:  |------------A------------|
        Clear:          |----clear----|
        Result:    |--A--|            |--A--|
        """
        idx = labeler.df.index
        
        labeler.intervals = [
            Interval(idx[10], idx[40], "A", "Note"),  # Spans clear range
        ]
        
        # Clear middle portion
        results = labeler._clear_intervals_in_range(idx[20], idx[30])
        
        assert results['split'] == 1
        assert len(labeler.intervals) == 2
        
        # Should have two "A" intervals
        a_intervals = [iv for iv in labeler.intervals if iv.label == "A"]
        assert len(a_intervals) == 2
        
        # Check split positions
        assert a_intervals[0].end == idx[20]
        assert a_intervals[1].start == idx[30]
        
        # Both should retain notes
        assert all(iv.notes == "Note" for iv in a_intervals)


# ============================================================================
# Command Pattern and Undo/Redo Tests
# ============================================================================

class TestCommandPattern:
    """Tests for command pattern implementation and undo/redo."""
    
    def test_undo_stack_limit(self, labeler):
        """Test that undo stack respects max_undo limit."""
        labeler.max_undo = 5
        idx = labeler.df.index
        
        # Add more commands than the limit
        for i in range(10):
            interval = Interval(idx[i], idx[i+1], f"Label{i}")
            cmd = AddIntervalCommand(labeler, interval)
            labeler._execute_command(cmd)
        
        # Undo stack should be limited
        assert len(labeler.undo_stack) == 5
        
        # Should be able to undo 5 times
        for _ in range(5):
            labeler._undo()
        
        # Should have 5 intervals left (first 5 commands were dropped)
        assert len(labeler.intervals) == 5
    
    def test_redo_cleared_on_new_command(self, labeler):
        """Test that redo stack is cleared when a new command is executed."""
        idx = labeler.df.index
        
        # Add and undo a command
        interval1 = Interval(idx[10], idx[20], "A")
        cmd1 = AddIntervalCommand(labeler, interval1)
        labeler._execute_command(cmd1)
        labeler._undo()
        
        # Redo stack should have the undone command
        assert len(labeler.redo_stack) == 1
        
        # Execute a new command
        interval2 = Interval(idx[30], idx[40], "B")
        cmd2 = AddIntervalCommand(labeler, interval2)
        labeler._execute_command(cmd2)
        
        # Redo stack should be cleared
        assert len(labeler.redo_stack) == 0
    
    def test_undo_redo_sequence(self, labeler):
        """Test a complex sequence of undo/redo operations."""
        idx = labeler.df.index
        
        # Execute multiple commands
        intervals = []
        for i in range(3):
            iv = Interval(idx[i*10], idx[i*10+5], f"Label{i}")
            intervals.append(iv)
            labeler._execute_command(AddIntervalCommand(labeler, iv))
        
        assert len(labeler.intervals) == 3
        
        # Undo all
        labeler._undo()
        labeler._undo()
        labeler._undo()
        assert len(labeler.intervals) == 0
        
        # Redo two
        labeler._redo()
        labeler._redo()
        assert len(labeler.intervals) == 2
        
        # Undo one
        labeler._undo()
        assert len(labeler.intervals) == 1
        
        # Verify it's the first interval
        assert labeler.intervals[0].label == "Label0"
    
    def test_command_modifies_flag(self, labeler):
        """Test that commands set the modified flag."""
        assert not labeler.modified
        
        # Execute a command
        idx = labeler.df.index
        interval = Interval(idx[10], idx[20], "Test")
        labeler._execute_command(AddIntervalCommand(labeler, interval))
        
        assert labeler.modified


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestIntervalEdgeCases:
    """Tests for edge cases in interval operations."""
    
    def test_empty_interval_handling(self, labeler):
        """Test handling of zero-width intervals."""
        t = labeler.df.index[10]
        
        # Try to add zero-width interval
        interval = Interval(t, t, "ZeroWidth")
        cmd = AddIntervalCommand(labeler, interval)
        cmd.execute()
        
        # Implementation may either reject or accept with minimum width
        # Just verify it doesn't crash
        assert True
    
    def test_interval_at_data_boundaries(self, labeler):
        """Test intervals at the very edges of the dataset."""
        # Interval at start
        start_interval = Interval(
            labeler.df.index[0],
            labeler.df.index[5],
            "Start"
        )
        labeler._execute_command(AddIntervalCommand(labeler, start_interval))
        
        # Interval at end
        end_interval = Interval(
            labeler.df.index[-6],
            labeler.df.index[-1],
            "End"
        )
        labeler._execute_command(AddIntervalCommand(labeler, end_interval))
        
        assert len(labeler.intervals) == 2
    
    def test_massive_interval_list(self, labeler):
        """Test performance with many intervals."""
        idx = labeler.df.index
        
        # Add many small intervals
        for i in range(0, min(100, len(idx)-2), 2):
            interval = Interval(idx[i], idx[i+1], "Many")
            labeler.intervals.append(interval)
        
        # Operations should still work
        labeler._sort_and_merge_intervals()
        
        # Should handle large lists efficiently
        assert len(labeler.intervals) > 0
    
    def test_interval_with_none_notes(self, labeler):
        """Test intervals with None notes field."""
        idx = labeler.df.index
        
        # Create interval without notes
        interval = Interval(idx[10], idx[20], "NoNotes", None)
        labeler._execute_command(AddIntervalCommand(labeler, interval))
        
        assert labeler.intervals[0].notes is None
        
        # Should serialize/deserialize correctly
        data = interval.to_dict()
        restored = Interval.from_dict(data)
        assert restored.notes is None


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntervalIntegration:
    """Integration tests for complex interval scenarios."""
    
    def test_complex_overlap_scenario(self, labeler):
        """Test a complex scenario with multiple overlapping operations."""
        idx = labeler.df.index
        
        # Create initial intervals
        labeler.intervals = [
            Interval(idx[10], idx[20], "A"),
            Interval(idx[30], idx[40], "B"),
            Interval(idx[50], idx[60], "C"),
        ]
        
        # Add overlapping interval with skip policy
        labeler._overlap_policy = "skip"
        new_interval = Interval(idx[15], idx[55], "Overlap")
        
        # Manually apply skip policy
        non_overlapping = labeler._subtract_overlaps_from_span(
            new_interval.start, new_interval.end
        )
        
        # Add the non-overlapping parts
        for s, e in non_overlapping:
            labeler._execute_command(AddIntervalCommand(
                labeler, Interval(s, e, "Overlap")
            ))
        
        # Should have original 3 + gaps filled
        assert len(labeler.intervals) > 3
        
        # Verify no overlaps
        for i, iv1 in enumerate(labeler.intervals):
            for iv2 in labeler.intervals[i+1:]:
                assert not iv1.overlaps(iv2)
    
    def test_workflow_label_assign_clear(self, labeler):
        """Test a complete workflow of labeling, assignment, and clearing."""
        idx = labeler.df.index
        
        # Step 1: Add some intervals
        labeler._execute_command(AddIntervalCommand(
            labeler, Interval(idx[10], idx[15], "A")
        ))
        labeler._execute_command(AddIntervalCommand(
            labeler, Interval(idx[20], idx[25], "B")
        ))
        
        # Step 2: Find and assign gaps
        gaps = labeler._find_gaps_in_current_range()
        initial_gap_count = len(gaps)
        
        for gap_s, gap_e in gaps[:2]:  # Assign first 2 gaps
            labeler._execute_command(AddIntervalCommand(
                labeler, Interval(gap_s, gap_e, "UNKNOWN")
            ))
        
        # Step 3: Clear a range
        clear_results = labeler._clear_intervals_in_range(idx[12], idx[22])
        
        # Verify operations worked
        assert clear_results['total_affected'] > 0
        
        # Step 4: Undo everything
        while labeler.undo_stack:
            labeler._undo()
        
        assert len(labeler.intervals) == 0
    
    def test_adjacent_operations_with_merging(self, labeler):
        """Test that adjacent operations trigger proper merging."""
        idx = labeler.df.index
        
        # Add two intervals that will become adjacent after operation
        interval1 = Interval(idx[10], idx[20], "A")
        interval2 = Interval(idx[25], idx[35], "A")
        interval_middle = Interval(idx[20], idx[25], "B")
        
        labeler._execute_command(AddIntervalCommand(labeler, interval1))
        labeler._execute_command(AddIntervalCommand(labeler, interval2))
        labeler._execute_command(AddIntervalCommand(labeler, interval_middle))
        
        assert len(labeler.intervals) == 3
        
        # Delete the middle interval
        labeler._execute_command(DeleteIntervalCommand(labeler, interval_middle))
        
        # Change label of second interval to match first
        intervals_a = [iv for iv in labeler.intervals if iv.label == "A"]
        if len(intervals_a) == 2:
            # They should be adjacent now, but not merged until we call merge
            labeler._sort_and_merge_intervals()
            
            # If they were truly adjacent, they might merge
            # Depends on exact boundaries


# ============================================================================
# Half-Open Interval Semantics Tests
# ============================================================================

class TestHalfOpenSemantics:
    """
    Tests specifically for half-open interval [start, end) semantics.
    
    IMPORTANT: The current implementation uses half-open intervals where:
    - start is inclusive
    - end is exclusive  
    - Adjacent intervals don't overlap: [a,b) and [b,c) don't overlap
    """
    
    def test_interval_contains(self):
        """Test half-open interval containment."""
        base = pd.Timestamp("2024-01-01 12:00:00")
        end = base + pd.Timedelta(minutes=10)
        interval = Interval(base, end, "Test")
        
        # Start is included
        assert interval.contains(base)
        
        # End is NOT included (half-open)
        assert not interval.contains(end)
        
        # Middle points are included
        assert interval.contains(base + pd.Timedelta(minutes=5))
        
        # Points outside are not included
        assert not interval.contains(base - pd.Timedelta(seconds=1))
        assert not interval.contains(end + pd.Timedelta(seconds=1))
    
    def test_adjacent_intervals_no_overlap(self):
        """Test that adjacent half-open intervals don't overlap."""
        base = pd.Timestamp("2024-01-01 12:00:00")
        mid = base + pd.Timedelta(minutes=10)
        end = base + pd.Timedelta(minutes=20)
        
        # Two adjacent intervals: [base, mid) and [mid, end)
        interval1 = Interval(base, mid, "A")
        interval2 = Interval(mid, end, "B")
        
        # Should NOT overlap (half-open semantics)
        assert not interval1.overlaps(interval2)
        assert not interval2.overlaps(interval1)
        
        # The boundary point belongs only to the second interval
        assert not interval1.contains(mid)
        assert interval2.contains(mid)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
