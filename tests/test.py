"""
Test script for Time Interval Labeler

Demonstrates and tests the non-overlapping interval logic.
"""

import pandas as pd
from labeler import Interval, TimeIntervalLabeler


def test_interval_overlap():
    """Test the interval overlap detection."""
    print("Testing interval overlap detection...")
    
    i1 = Interval(
        pd.Timestamp("2015-01-01 10:00:00"),
        pd.Timestamp("2015-01-01 12:00:00"),
        "A"
    )
    
    i2 = Interval(
        pd.Timestamp("2015-01-01 11:00:00"),
        pd.Timestamp("2015-01-01 13:00:00"),
        "B"
    )
    
    i3 = Interval(
        pd.Timestamp("2015-01-01 13:00:00"),
        pd.Timestamp("2015-01-01 14:00:00"),
        "C"
    )
    
    assert i1.overlaps(i2), "i1 and i2 should overlap"
    assert not i2.overlaps(i3), "i2 and i3 should not overlap (adjacent)"
    assert not i1.overlaps(i3), "i1 and i3 should not overlap"
    
    print("✓ Overlap detection works correctly")


def test_interval_contains():
    """Test the timestamp containment check."""
    print("\nTesting timestamp containment...")
    
    interval = Interval(
        pd.Timestamp("2015-01-01 10:00:00"),
        pd.Timestamp("2015-01-01 12:00:00"),
        "A"
    )
    
    assert interval.contains(pd.Timestamp("2015-01-01 10:00:00")), "Should contain start"
    assert interval.contains(pd.Timestamp("2015-01-01 11:00:00")), "Should contain middle"
    assert not interval.contains(pd.Timestamp("2015-01-01 12:00:00")), "Should not contain end (exclusive)"
    assert not interval.contains(pd.Timestamp("2015-01-01 09:00:00")), "Should not contain before"
    assert not interval.contains(pd.Timestamp("2015-01-01 13:00:00")), "Should not contain after"
    
    print("✓ Timestamp containment works correctly")


def test_non_overlapping_logic():
    """Test the non-overlapping interval management."""
    print("\nTesting non-overlapping interval logic...")
    
    # Create a minimal dataframe
    rng = pd.date_range("2015-01-01", "2015-01-02", freq="1H")
    df = pd.DataFrame({"dummy": range(len(rng))}, index=rng)
    
    def dummy_plot(axs, df, t0, t1):
        pass
    
    # Create labeler
    labeler = TimeIntervalLabeler(df, dummy_plot)
    
    # Add first interval [10:00 - 12:00) labeled "A"
    i1 = Interval(
        pd.Timestamp("2015-01-01 10:00:00"),
        pd.Timestamp("2015-01-01 12:00:00"),
        "A"
    )
    labeler.intervals.append(i1)
    
    # Add overlapping interval [11:00 - 13:00) labeled "B"
    # This should split i1 into [10:00 - 11:00) and remove [11:00 - 12:00)
    i2 = Interval(
        pd.Timestamp("2015-01-01 11:00:00"),
        pd.Timestamp("2015-01-01 13:00:00"),
        "B"
    )
    
    removed = labeler._remove_overlapping_intervals(i2)
    labeler.intervals.append(i2)
    labeler._sort_and_merge_intervals()
    
    print(f"  After adding overlapping interval:")
    for iv in labeler.intervals:
        print(f"    [{iv.start} - {iv.end}) : {iv.label}")
    
    # Should have: [10:00-11:00) A, [11:00-13:00) B
    assert len(labeler.intervals) == 2, f"Expected 2 intervals, got {len(labeler.intervals)}"
    assert labeler.intervals[0].label == "A"
    assert labeler.intervals[0].start == pd.Timestamp("2015-01-01 10:00:00")
    assert labeler.intervals[0].end == pd.Timestamp("2015-01-01 11:00:00")
    assert labeler.intervals[1].label == "B"
    
    print("✓ Non-overlapping split works correctly")
    
    # Test merging adjacent same-label intervals
    i3 = Interval(
        pd.Timestamp("2015-01-01 13:00:00"),
        pd.Timestamp("2015-01-01 14:00:00"),
        "B"
    )
    labeler.intervals.append(i3)
    labeler._sort_and_merge_intervals()
    
    print(f"\n  After adding adjacent same-label interval:")
    for iv in labeler.intervals:
        print(f"    [{iv.start} - {iv.end}) : {iv.label}")
    
    # Should have merged: [10:00-11:00) A, [11:00-14:00) B
    assert len(labeler.intervals) == 2, f"Expected 2 intervals after merge, got {len(labeler.intervals)}"
    assert labeler.intervals[1].end == pd.Timestamp("2015-01-01 14:00:00")
    
    print("✓ Adjacent interval merging works correctly")


def test_serialization():
    """Test JSON serialization."""
    print("\nTesting JSON serialization...")
    
    interval = Interval(
        pd.Timestamp("2015-01-01 10:00:00"),
        pd.Timestamp("2015-01-01 12:00:00"),
        "TestLabel",
        notes="Test note"
    )
    
    # Convert to dict
    d = interval.to_dict()
    assert d['label'] == "TestLabel"
    assert d['notes'] == "Test note"
    
    # Convert back
    interval2 = Interval.from_dict(d)
    assert interval2.start == interval.start
    assert interval2.end == interval.end
    assert interval2.label == interval.label
    assert interval2.notes == interval.notes
    
    print("✓ Serialization works correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Time Interval Labeler - Test Suite")
    print("=" * 60)
    
    try:
        test_interval_overlap()
        test_interval_contains()
        test_serialization()
        test_non_overlapping_logic()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
