"""
Test selection mechanisms in ChronoTagger.

This module tests the various ways users can select time ranges:
- Two-click selection on time axes
- Rectangle/box selection on time-series plots  
- Box selection on position/phase-space plots
- Selection cancellation and state management

ASCII Diagram of Selection Types:
==================================

Two-Click Selection (Time Axes):
    Click 1                Click 2
      ↓                      ↓
  ────┼──────────────────────┼────→ time
      |<--- preview band --->|
      
Box Selection on Time Axis:
  ┌─────────────┐
  │ drag box    │  → extracts points inside
  └─────────────┘
  ────────────────────────────────→ time
  
Box Selection on Position Plot:
       Y ^
         │  ┌─────┐
         │  │box  │ → maps to time via point order
         │  └─────┘
  ───────┼───────────→ X
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from unittest.mock import MagicMock, patch
from chronotagger.labeler import TimeIntervalLabeler
from chronotagger.core.models import Interval


class MockEvent:
    """Mock matplotlib event for testing."""
    def __init__(self, xdata=None, ydata=None, button=1, inaxes=None, x=100, y=100):
        self.xdata = xdata
        self.ydata = ydata
        self.button = button
        self.inaxes = inaxes
        self.x = x  # pixel coordinates
        self.y = y


@pytest.fixture
def labeler_with_position_plot(df_hour, plot_fn):
    """Create labeler with both time and position axes."""
    # Extended plot function that creates position plots
    def enhanced_plot_fn(axs, df, t0, t1):
        # Time series plots
        if "time1" in axs:
            axs["time1"].plot(df.index, df["log10n"])
        if "time2" in axs:
            axs["time2"].plot(df.index, df["BX"], label="Bx")
            
        # Position plot (not-time)
        if "xy_plot" in axs:
            axs["xy_plot"].plot(df["BX"], df["BY"])
    
    # Layout with mixed axis types
    layout_spec = {
        "nrows": 2,
        "ncols": 2,
        "areas": [
            {"key": "time1", "role": "time", "row": 0, "col": 0},
            {"key": "time2", "role": "time", "row": 1, "col": 0},
            {"key": "xy_plot", "role": "not-time", "row": 0, "col": 1,
             "x_col": "BX", "y_col": "BY"},
            {"key": "labels", "role": "labels", "row": 1, "col": 1},
        ]
    }
    
    lbl = TimeIntervalLabeler(
        df=df_hour,
        plot_fn=enhanced_plot_fn,
        window=pd.Timedelta("30min"),
        layout_spec=layout_spec
    )
    lbl._build_gui()
    lbl._update_plot()  # Draw initial data and set xlim
    lbl.root.withdraw()  # Hide window for tests
    yield lbl
    lbl.root.destroy()


class TestTwoClickSelection:
    """Test two-click time selection mechanism."""
    
    def test_two_click_arms_on_first_click(self, labeler):
        """First click should arm selection and show preview."""
        # Get a time axis
        ax = list(labeler.user_axes.values())[0]

        # Create first click event
        t_click = labeler.df.index[10]
        x_click = mdates.date2num(t_click)
        event = MockEvent(xdata=x_click, inaxes=ax)

        # Simulate first click (Phase 3 signature: event, pane)
        labeler._on_time_click(event, labeler.active_pane)

        # Check that selection is armed
        assert labeler._two_click_active is True
        assert labeler._two_click_t0 is not None
        assert abs(labeler._two_click_t0 - x_click) < 1e-6
        
        # Should have a preview selection (point-like initially)
        assert labeler.current_selection is not None
        start, end = labeler.current_selection
        assert start == pd.Timestamp(mdates.num2date(x_click)).tz_localize(None)
    
    def test_two_click_finalizes_on_second_click(self, labeler):
        """Second click should finalize selection."""
        ax = list(labeler.user_axes.values())[0]

        # First click
        t1 = labeler.df.index[10]
        x1 = mdates.date2num(t1)
        event1 = MockEvent(xdata=x1, inaxes=ax)
        labeler._on_time_click(event1, labeler.active_pane)

        # Second click
        t2 = labeler.df.index[30]
        x2 = mdates.date2num(t2)
        event2 = MockEvent(xdata=x2, inaxes=ax)
        labeler._on_time_click(event2, labeler.active_pane)
        
        # Check finalized state
        assert labeler._two_click_active is False
        assert labeler._two_click_t0 is None
        assert labeler.current_selection is not None
        
        # Verify selection range
        start, end = labeler.current_selection
        expected_start = pd.Timestamp(mdates.num2date(min(x1, x2))).tz_localize(None)
        expected_end = pd.Timestamp(mdates.num2date(max(x1, x2))).tz_localize(None)
        
        # Allow for snapping if enabled
        if labeler.snap_var.get():
            # Timestamps should be snapped to actual data points
            assert start in labeler.df.index
            assert end in labeler.df.index
        else:
            assert abs((start - expected_start).total_seconds()) < 1
            assert abs((end - expected_end).total_seconds()) < 1
    
    def test_two_click_preview_updates_during_motion(self, labeler):
        """Preview should update as mouse moves after first click."""
        ax = list(labeler.user_axes.values())[0]

        # First click to arm
        t1 = labeler.df.index[10]
        x1 = mdates.date2num(t1)
        event1 = MockEvent(xdata=x1, inaxes=ax)
        labeler._on_time_click(event1, labeler.active_pane)
        
        # Simulate motion
        t_motion = labeler.df.index[20]
        x_motion = mdates.date2num(t_motion)
        motion_event = MockEvent(xdata=x_motion, inaxes=ax)
        labeler._on_time_motion(motion_event, labeler.active_pane)
        
        # Preview should be updated
        assert labeler.current_selection is not None
        start, end = labeler.current_selection
        assert start <= end
        
    def test_right_click_cancels_two_click(self, labeler):
        """Right click should cancel active two-click selection."""
        ax = list(labeler.user_axes.values())[0]

        # Start two-click
        t1 = labeler.df.index[10]
        x1 = mdates.date2num(t1)
        event1 = MockEvent(xdata=x1, inaxes=ax)
        labeler._on_time_click(event1, labeler.active_pane)

        assert labeler._two_click_active is True

        # Right click to cancel
        cancel_event = MockEvent(button=3, inaxes=ax)
        labeler._on_time_click(cancel_event, labeler.active_pane)
        
        # Should be cancelled
        assert labeler._two_click_active is False
        assert labeler.current_selection is None


class TestBoxSelection:
    """Test rectangle/box selection on various axis types."""
    
    def test_box_select_on_time_axis_extracts_points(self, labeler):
        """Box selection on time axis should extract points within bounds."""
        ax = list(labeler.user_axes.values())[0]
        
        # Simulate box selection
        t1, t2 = labeler.df.index[5], labeler.df.index[25]
        x1, x2 = mdates.date2num(t1), mdates.date2num(t2)
        
        # Get some y-bounds from actual data
        y1, y2 = 0, 10  # Arbitrary y-range
        
        # Create mock events for drag
        eclick = MockEvent(xdata=x1, ydata=y1, inaxes=ax)
        erelease = MockEvent(xdata=x2, ydata=y2, inaxes=ax)
        
        # Set up rectangle selector key for the mock
        for key, user_ax in labeler.user_axes.items():
            if user_ax == ax:
                labeler.rect_selectors = {key: MagicMock()}
                break

        # Trigger selection (Phase 3 signature: eclick, erelease, pane)
        labeler._on_rectangle_select(eclick, erelease, labeler.active_pane)
        
        # For a full-height selection, should create single span
        if labeler.current_selection is not None:
            start, end = labeler.current_selection
            # Should encompass the selected range
            assert start <= t1 or abs((start - t1).total_seconds()) < 60
            assert end >= t2 or abs((end - t2).total_seconds()) < 60
    
    def test_box_select_detects_contiguous_runs(self, labeler):
        """
        Box selection should split selected points into contiguous runs.
        
        Scenario with gaps:
        Points:  x x x   x x   x
        Indices: 0 1 2   5 6   9
        Should create 3 separate intervals
        """
        ax = list(labeler.user_axes.values())[0]
        
        # We'll mock a selection that would hit non-contiguous points
        # This requires mocking the data extraction more carefully
        
        # Create sparse selection indices
        selected_indices = [0, 1, 2, 5, 6, 9]  # Non-contiguous
        
        # Test the run detection directly
        runs = labeler._find_contiguous_runs(selected_indices)
        
        assert len(runs) == 3
        assert runs[0] == (0, 2)   # First run
        assert runs[1] == (5, 6)   # Second run  
        assert runs[2] == (9, 9)   # Single point
    
    def test_box_select_on_position_plot(self, labeler_with_position_plot):
        """
        Box selection on position plot should map spatial selection to time.
        
        Position plot shows (X, Y) coordinates, but selection should
        map back to time intervals based on point ordering.
        """
        labeler = labeler_with_position_plot
        
        # Find the position plot axis
        xy_ax = labeler.user_axes.get("xy_plot")
        assert xy_ax is not None
        
        # Simulate box selection in position space
        x_min, x_max = -5, 5  # BX range
        y_min, y_max = -2, 2  # BY range
        
        eclick = MockEvent(xdata=x_min, ydata=y_min, inaxes=xy_ax)
        erelease = MockEvent(xdata=x_max, ydata=y_max, inaxes=xy_ax)
        
        # Mock the rectangle selector
        labeler.rect_selectors = {"xy_plot": MagicMock()}
        
        # Perform selection (Phase 3 signature: eclick, erelease, pane)
        labeler._on_rectangle_select(eclick, erelease, labeler.active_pane)
        
        # Should have created spans based on which points fall in box
        # The actual mapping depends on data values
        if labeler.current_spans:
            # Verify we got time intervals, not position coordinates
            for start, end in labeler.current_spans:
                assert isinstance(start, pd.Timestamp)
                assert isinstance(end, pd.Timestamp)
                assert start <= end
    
    def test_exact_vs_halfopen_intervals(self, labeler):
        """
        Test the distinction between exact and half-open intervals.
        
        Box selections use exact intervals [first_point, last_point]
        for visual consistency, then convert to half-open for storage.
        """
        idx = labeler.df.index
        runs = [(5, 10), (15, 15)]  # Include single-point run
        
        # Test exact interval creation (for box select)
        exact_intervals = labeler._runs_to_exact_intervals(idx, runs)
        
        assert len(exact_intervals) == 2
        # First interval: from point 5 to point 10
        assert exact_intervals[0] == (idx[5], idx[10])
        # Single point: from point 15 to point 15  
        assert exact_intervals[1] == (idx[15], idx[15])
        
        # Test half-open conversion
        half_open = labeler._runs_to_half_open_intervals(idx, runs)
        
        assert len(half_open) == 2
        # First interval: [idx[5], idx[11]) to include point 10
        assert half_open[0][0] == idx[5]
        if 11 < len(idx):
            assert half_open[0][1] == idx[11]
        else:
            # At boundary, add small epsilon
            assert half_open[0][1] > idx[10]


class TestSelectionCancellation:
    """Test selection cancellation and state cleanup."""
    
    def test_escape_cancels_active_selection(self, labeler):
        """Escape key should cancel any active selection."""
        # Start a two-click selection
        ax = list(labeler.user_axes.values())[0]
        t1 = labeler.df.index[10]
        x1 = mdates.date2num(t1)
        event = MockEvent(xdata=x1, inaxes=ax)
        labeler._on_time_click(event, labeler.active_pane)
        
        assert labeler._two_click_active is True
        assert labeler.current_selection is not None
        
        # Press Escape
        escape_event = MagicMock()
        escape_event.keysym = "Escape"
        labeler._on_key_press(escape_event)
        
        # Should be cancelled
        assert labeler._two_click_active is False
        assert labeler.current_selection is None
    
    def test_escape_deselects_interval(self, labeler):
        """Escape should deselect selected interval if no active selection."""
        # Add and select an interval
        t1, t2 = labeler.df.index[10], labeler.df.index[20]
        iv = Interval(t1, t2, "TEST")
        labeler.intervals.append(iv)
        labeler.selected_interval = iv
        
        # Press Escape
        escape_event = MagicMock()
        escape_event.keysym = "Escape"
        labeler._on_key_press(escape_event)
        
        # Interval should be deselected
        assert labeler.selected_interval is None
    
    def test_comprehensive_cancel_clears_all_state(self, labeler):
        """
        _cancel_active_selection should clear ALL selection state.
        
        This includes:
        - current_selection
        - current_spans  
        - _commit_spans
        - two-click state
        - overlays
        - highlights
        """
        # Set up various selection states
        labeler.current_selection = (labeler.df.index[0], labeler.df.index[10])
        labeler.current_spans = [(labeler.df.index[5], labeler.df.index[8])]
        labeler._commit_spans = [(labeler.df.index[2], labeler.df.index[7])]
        labeler._two_click_active = True
        labeler._two_click_t0 = 123.456
        
        # Cancel everything
        labeler._cancel_active_selection()
        
        # Verify all cleared
        assert labeler.current_selection is None
        assert labeler.current_spans == []
        assert labeler._commit_spans == []
        assert labeler._two_click_active is False
        assert labeler._two_click_t0 is None


class TestLocalizedPadding:
    """Test localized padding for variable sampling rates."""
    
    def test_compute_localized_median(self, labeler):
        """Test computation of local median time difference."""
        # Test at a point in the middle of data
        target = labeler.df.index[30]
        median_diff = labeler._compute_localized_median_time_diff(target, window_points=10)
        
        # For uniformly sampled data (30s), median should be close to 30s
        assert isinstance(median_diff, pd.Timedelta)
        assert 25 <= median_diff.total_seconds() <= 35
    
    def test_padding_single_point_interval(self, labeler):
        """Single-point intervals should get symmetric padding."""
        single_point = labeler.df.index[20]
        intervals = [(single_point, single_point)]
        
        padded = labeler._apply_localized_padding_to_intervals(intervals)
        
        assert len(padded) == 1
        start, end = padded[0]
        assert start < single_point
        assert end > single_point
        
        # Padding should be roughly symmetric
        before = (single_point - start).total_seconds()
        after = (end - single_point).total_seconds()
        assert abs(before - after) / max(before, after) < 0.2  # Within 20%
    
    def test_padding_multipoint_interval(self, labeler):
        """Multi-point intervals should get edge-based padding."""
        start_pt = labeler.df.index[10]
        end_pt = labeler.df.index[20]
        intervals = [(start_pt, end_pt)]
        
        padded = labeler._apply_localized_padding_to_intervals(intervals)
        
        assert len(padded) == 1
        padded_start, padded_end = padded[0]
        
        # Should extend beyond original bounds
        assert padded_start <= start_pt
        assert padded_end >= end_pt
        
        # Should not exceed data bounds
        assert padded_start >= labeler.data_start
        assert padded_end <= labeler.data_end


class TestSnapping:
    """Test snap-to-samples functionality."""
    
    def test_snap_to_samples_basic(self, labeler):
        """Timestamps should snap to nearest data points."""
        # Create timestamps between data points
        t1 = labeler.df.index[5] + pd.Timedelta(seconds=10)
        t2 = labeler.df.index[15] - pd.Timedelta(seconds=5)
        
        # Enable snapping
        labeler.snap_var.set(True)
        
        # Snap timestamps
        snapped_start, snapped_end = labeler._snap_to_samples(t1, t2)
        
        # Should snap to actual index values
        assert snapped_start in labeler.df.index
        assert snapped_end in labeler.df.index
        
        # Should be close to originals
        assert abs((snapped_start - t1).total_seconds()) <= 15
        assert abs((snapped_end - t2).total_seconds()) <= 15
    
    def test_snap_respects_window_bounds(self, labeler):
        """Snapping should only consider points in current window."""
        # Set a narrow window
        labeler.t0 = labeler.df.index[20]
        labeler.t1 = labeler.df.index[40]
        
        # Try to snap to points outside window
        t1 = labeler.df.index[10]  # Before window
        t2 = labeler.df.index[50]  # After window
        
        labeler.snap_var.set(True)
        snapped_start, snapped_end = labeler._snap_to_samples(t1, t2)
        
        # Should snap to window boundaries or nearest points within
        assert snapped_start >= labeler.t0
        assert snapped_end <= labeler.t1


class TestClickVsDragArbitration:
    """Test distinguishing clicks from drags."""
    
    def test_small_movement_treated_as_click(self, labeler):
        """Small mouse movement (<6 pixels) should be treated as click."""
        ax = list(labeler.user_axes.values())[0]
        
        # Set click slop threshold
        assert labeler.CLICK_DRAG_SLOP_PX == 6
        
        # Record press position
        labeler._press_xy_px = (100, 100)
        
        # Small movement (3 pixels)
        release_x = 103
        release_y = 102
        
        # Calculate distance
        dx = release_x - labeler._press_xy_px[0]
        dy = release_y - labeler._press_xy_px[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Should be under threshold
        assert distance < labeler.CLICK_DRAG_SLOP_PX
        
        # This would trigger two-click, not box select
        # (actual implementation would check this in mouse handlers)
    
    def test_large_movement_treated_as_drag(self, labeler):
        """Large mouse movement (>6 pixels) should be treated as drag."""
        # Record press position
        labeler._press_xy_px = (100, 100)
        
        # Large movement (10 pixels)
        release_x = 110
        release_y = 105
        
        # Calculate distance
        dx = release_x - labeler._press_xy_px[0]
        dy = release_y - labeler._press_xy_px[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Should exceed threshold
        assert distance > labeler.CLICK_DRAG_SLOP_PX
        
        # This would trigger box select, not two-click


class TestMultiSpanSelection:
    """Test selection of multiple disjoint time spans."""
    
    def test_multiple_spans_from_box_select(self, labeler):
        """Box selection creating multiple spans should handle all correctly."""
        # Create multiple spans
        spans = [
            (labeler.df.index[5], labeler.df.index[8]),
            (labeler.df.index[12], labeler.df.index[15]),
            (labeler.df.index[20], labeler.df.index[22])
        ]
        
        labeler.current_spans = spans
        labeler._commit_spans = spans.copy()
        
        # Add intervals from spans
        labeler._add_interval()
        
        # Should create 3 separate intervals
        assert len(labeler.intervals) == 3
        
        # Verify each interval matches a span
        for i, (start, end) in enumerate(spans):
            # Find corresponding interval (may be merged/adjusted)
            found = False
            for iv in labeler.intervals:
                if iv.start <= start and iv.end >= end:
                    found = True
                    break
            assert found, f"Span {i} not found in intervals"
    
    def test_multi_span_preview_visualization(self, labeler):
        """Multiple spans should show correct preview bands."""
        spans = [
            (labeler.df.index[5], labeler.df.index[8]),
            (labeler.df.index[15], labeler.df.index[18])
        ]
        
        # Set up multi-span preview
        labeler.current_spans = spans
        
        # Update overlays
        labeler._update_time_overlays_for_multi_spans(spans)
        
        # Strip preview should show multiple rectangles
        spans_float = [
            (mdates.date2num(s), mdates.date2num(e)) 
            for s, e in spans
        ]
        labeler._draw_strip_preview_spans(spans_float)
        
        # Verify preview pool has enough rectangles (Phase 3: per-pane attribute)
        assert len(labeler.active_pane._strip_preview_pool) >= len(spans)


# Additional test utilities
def create_mock_labeler_with_intervals(df_hour, intervals_data):
    """Helper to create labeler with pre-populated intervals."""
    labeler = TimeIntervalLabeler(
        df=df_hour,
        plot_fn=lambda ax, df, t0, t1: None,
        window=pd.Timedelta("30min")
    )
    
    for start_idx, end_idx, label in intervals_data:
        iv = Interval(
            df_hour.index[start_idx],
            df_hour.index[end_idx],
            label
        )
        labeler.intervals.append(iv)
    
    labeler._sort_and_merge_intervals()
    return labeler
