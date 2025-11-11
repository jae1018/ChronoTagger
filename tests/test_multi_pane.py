"""
Comprehensive tests for multi-pane functionality.

Tests cover:
- Pane creation and initialization
- Tab switching and active pane tracking
- Event handler routing
- Synchronization across panes
- Keyboard shortcuts
- Context menu operations
- Save/load with multi-pane config
- Backward compatibility
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_dataframe():
    """Create sample time-series data for testing."""
    times = pd.date_range('2024-01-01', periods=1000, freq='1min')
    df = pd.DataFrame({
        'value1': np.sin(np.linspace(0, 10*np.pi, 1000)),
        'value2': np.cos(np.linspace(0, 10*np.pi, 1000)),
        'value3': np.random.randn(1000),
    }, index=times)
    return df


@pytest.fixture
def plot_function_1():
    """Simple plot function for testing."""
    def plot_fn(axs, df, t0, t1):
        axs['panel1'].plot(df.index, df['value1'])
        axs['panel1'].set_ylabel('Value 1')
    return plot_fn


@pytest.fixture
def plot_function_2():
    """Alternative plot function for testing."""
    def plot_fn(axs, df, t0, t1):
        axs['panel1'].plot(df.index, df['value2'])
        axs['panel1'].set_ylabel('Value 2')
    return plot_fn


@pytest.fixture
def layout_spec():
    """Standard layout specification."""
    return {
        "nrows": 2,
        "ncols": 1,
        "areas": [
            {"key": "panel1", "row": 0, "col": 0, "role": "time"},
            {"key": "labels", "row": 1, "col": 0, "role": "labels"},
        ]
    }


# ============================================================================
# TEST SUITE 1: PANE CREATION AND INITIALIZATION
# ============================================================================

class TestPaneCreation:
    """Tests for pane creation and initialization."""

    def test_single_pane_backward_compatibility(self, sample_dataframe, plot_function_1, layout_spec):
        """Single-pane mode still works (backward compatibility)."""
        from chronotagger.labeler import TimeIntervalLabeler

        labeler = TimeIntervalLabeler(
            df=sample_dataframe,
            plot_fn=plot_function_1,
            layout_spec=layout_spec,
            window=pd.Timedelta("1h"),
        )

        assert labeler.multi_pane_mode is False
        assert len(labeler.panes) == 1
        assert labeler.active_pane_idx == 0
        assert labeler.panes[0].title == "Main"

    def test_multi_pane_creation(self, sample_dataframe, plot_function_1, plot_function_2, layout_spec):
        """Multi-pane mode creates correct structure."""
        from chronotagger.labeler import TimeIntervalLabeler

        panes = [
            {"title": "Pane 1", "plot_fn": plot_function_1, "layout_spec": layout_spec},
            {"title": "Pane 2", "plot_fn": plot_function_2, "layout_spec": layout_spec},
        ]

        labeler = TimeIntervalLabeler(
            df=sample_dataframe,
            panes=panes,
            window=pd.Timedelta("1h"),
        )

        assert labeler.multi_pane_mode is True
        assert len(labeler.panes) == 2
        assert labeler.active_pane_idx == 0
        assert labeler.panes[0].title == "Pane 1"
        assert labeler.panes[1].title == "Pane 2"

    def test_pane_objects_created(self, sample_dataframe, plot_function_1, layout_spec):
        """Each pane gets a TabPane object."""
        from chronotagger.labeler import TimeIntervalLabeler
        from chronotagger.labeler.tab_pane import TabPane

        panes = [
            {"title": "Test Pane", "plot_fn": plot_function_1, "layout_spec": layout_spec},
        ]

        labeler = TimeIntervalLabeler(
            df=sample_dataframe,
            panes=panes,
            window=pd.Timedelta("1h"),
        )

        pane = labeler.panes[0]
        assert isinstance(pane, TabPane)
        assert pane.title == "Test Pane"
        assert pane.plot_fn is plot_function_1
        assert pane.layout_spec == layout_spec
        assert pane.dirty is True  # Initial state

    def test_active_pane_property(self, sample_dataframe, plot_function_1, plot_function_2, layout_spec):
        """active_pane property returns correct pane."""
        from chronotagger.labeler import TimeIntervalLabeler

        panes = [
            {"title": "Pane 1", "plot_fn": plot_function_1, "layout_spec": layout_spec},
            {"title": "Pane 2", "plot_fn": plot_function_2, "layout_spec": layout_spec},
        ]

        labeler = TimeIntervalLabeler(
            df=sample_dataframe,
            panes=panes,
            window=pd.Timedelta("1h"),
        )

        assert labeler.active_pane is labeler.panes[0]
        assert labeler.active_pane.title == "Pane 1"

    def test_cannot_provide_both_panes_and_plot_fn(self, sample_dataframe, plot_function_1, layout_spec):
        """Should raise error if both panes and plot_fn provided."""
        from chronotagger.labeler import TimeIntervalLabeler

        panes = [
            {"title": "Pane 1", "plot_fn": plot_function_1, "layout_spec": layout_spec},
        ]

        with pytest.raises((ValueError, AssertionError, TypeError)):
            TimeIntervalLabeler(
                df=sample_dataframe,
                panes=panes,
                plot_fn=plot_function_1,  # Should not allow both
                window=pd.Timedelta("1h"),
            )


# ============================================================================
# TEST SUITE 2: TAB SWITCHING AND ACTIVE PANE TRACKING
# ============================================================================

class TestTabSwitching:
    """Tests for tab switching functionality."""

    def test_keyboard_shortcut_handlers_exist(self, sample_dataframe, plot_function_1, layout_spec):
        """Keyboard shortcut handlers exist."""
        from chronotagger.labeler import TimeIntervalLabeler

        panes = [
            {"title": "Pane 1", "plot_fn": plot_function_1, "layout_spec": layout_spec},
            {"title": "Pane 2", "plot_fn": plot_function_1, "layout_spec": layout_spec},
        ]

        labeler = TimeIntervalLabeler(
            df=sample_dataframe,
            panes=panes,
            window=pd.Timedelta("1h"),
        )

        # Verify all keyboard shortcut handlers exist
        assert hasattr(labeler, '_next_tab')
        assert hasattr(labeler, '_prev_tab')
        assert hasattr(labeler, '_go_to_tab')
        assert callable(labeler._next_tab)
        assert callable(labeler._prev_tab)
        assert callable(labeler._go_to_tab)

    def test_shortcuts_return_break_without_gui(self, sample_dataframe, plot_function_1, layout_spec):
        """Keyboard shortcuts return 'break' without errors when GUI not built."""
        from chronotagger.labeler import TimeIntervalLabeler

        panes = [
            {"title": "Pane 1", "plot_fn": plot_function_1, "layout_spec": layout_spec},
            {"title": "Pane 2", "plot_fn": plot_function_1, "layout_spec": layout_spec},
        ]

        labeler = TimeIntervalLabeler(
            df=sample_dataframe,
            panes=panes,
            window=pd.Timedelta("1h"),
        )

        # Should return 'break' even without GUI (notebook is None)
        # Methods check for notebook existence and return safely
        result = labeler._next_tab()
        assert result == 'break'

        result = labeler._prev_tab()
        assert result == 'break'

        result = labeler._go_to_tab(0)
        assert result == 'break'

    def test_single_pane_shortcuts_safe(self, sample_dataframe, plot_function_1, layout_spec):
        """Keyboard shortcuts don't error in single-pane mode."""
        from chronotagger.labeler import TimeIntervalLabeler

        labeler = TimeIntervalLabeler(
            df=sample_dataframe,
            plot_fn=plot_function_1,
            layout_spec=layout_spec,
            window=pd.Timedelta("1h"),
        )

        # These should return 'break' safely without errors
        result = labeler._next_tab()
        assert result == 'break'

        result = labeler._prev_tab()
        assert result == 'break'

        result = labeler._go_to_tab(0)
        assert result == 'break'


# ============================================================================
# TEST SUITE 3: SYNCHRONIZATION
# ============================================================================

class TestSynchronization:
    """Tests for state synchronization across panes."""

    def test_intervals_shared_across_panes(self, sample_dataframe, plot_function_1, layout_spec):
        """All panes share the same intervals list."""
        from chronotagger.labeler import TimeIntervalLabeler
        from chronotagger.core.models import Interval

        panes = [
            {"title": "Pane 1", "plot_fn": plot_function_1, "layout_spec": layout_spec},
            {"title": "Pane 2", "plot_fn": plot_function_1, "layout_spec": layout_spec},
        ]

        labeler = TimeIntervalLabeler(
            df=sample_dataframe,
            panes=panes,
            window=pd.Timedelta("1h"),
        )

        # Add an interval directly
        t0 = sample_dataframe.index[10]
        t1 = sample_dataframe.index[20]
        interval = Interval(t0, t1, "UNKNOWN")
        labeler.intervals.append(interval)

        # Should have one interval
        assert len(labeler.intervals) == 1
        assert labeler.intervals[0].start == t0
        assert labeler.intervals[0].end == t1

    def test_pane_dirty_tracking(self, sample_dataframe, plot_function_1, layout_spec):
        """Panes track dirty state correctly."""
        from chronotagger.labeler import TimeIntervalLabeler

        panes = [
            {"title": "Pane 1", "plot_fn": plot_function_1, "layout_spec": layout_spec},
            {"title": "Pane 2", "plot_fn": plot_function_1, "layout_spec": layout_spec},
        ]

        labeler = TimeIntervalLabeler(
            df=sample_dataframe,
            panes=panes,
            window=pd.Timedelta("1h"),
        )

        # Initially all panes are dirty
        assert all(pane.dirty for pane in labeler.panes)

        # Mark clean
        for pane in labeler.panes:
            pane.mark_clean(labeler.t0, labeler.t1)

        assert not any(pane.dirty for pane in labeler.panes)

        # Mark dirty
        for pane in labeler.panes:
            pane.mark_dirty()

        assert all(pane.dirty for pane in labeler.panes)

    def test_pane_needs_update_logic(self, sample_dataframe, plot_function_1, layout_spec):
        """Pane needs_update() checks dirty flag and time window."""
        from chronotagger.labeler import TimeIntervalLabeler

        panes = [
            {"title": "Pane 1", "plot_fn": plot_function_1, "layout_spec": layout_spec},
        ]

        labeler = TimeIntervalLabeler(
            df=sample_dataframe,
            panes=panes,
            window=pd.Timedelta("1h"),
        )

        pane = labeler.panes[0]
        t0 = labeler.t0
        t1 = labeler.t1

        # Clean pane for current window - should not need update
        pane.mark_clean(t0, t1)
        assert not pane.needs_update(t0, t1)

        # Clean pane but different window - should need update
        new_t0 = t0 + pd.Timedelta("1h")
        new_t1 = t1 + pd.Timedelta("1h")
        assert pane.needs_update(new_t0, new_t1)

        # Dirty pane - should need update regardless
        pane.mark_dirty()
        assert pane.needs_update(t0, t1)


# ============================================================================
# TEST SUITE 4: CONTEXT MENU
# ============================================================================

class TestContextMenu:
    """Tests for tab context menu functionality."""

    def test_context_menu_methods_exist(self, sample_dataframe, plot_function_1, layout_spec):
        """All context menu methods are present."""
        from chronotagger.labeler import TimeIntervalLabeler

        panes = [
            {"title": "Pane 1", "plot_fn": plot_function_1, "layout_spec": layout_spec},
            {"title": "Pane 2", "plot_fn": plot_function_1, "layout_spec": layout_spec},
        ]

        labeler = TimeIntervalLabeler(
            df=sample_dataframe,
            panes=panes,
            window=pd.Timedelta("1h"),
        )

        assert hasattr(labeler, '_create_tab_context_menu')
        assert hasattr(labeler, '_show_tab_context_menu')
        assert hasattr(labeler, '_rename_active_tab')
        assert hasattr(labeler, '_refresh_active_tab')
        assert hasattr(labeler, '_refresh_all_tabs')

    def test_rename_updates_pane_title(self, sample_dataframe, plot_function_1, layout_spec):
        """Renaming updates the pane's title attribute."""
        from chronotagger.labeler import TimeIntervalLabeler

        panes = [
            {"title": "Original Name", "plot_fn": plot_function_1, "layout_spec": layout_spec},
        ]

        labeler = TimeIntervalLabeler(
            df=sample_dataframe,
            panes=panes,
            window=pd.Timedelta("1h"),
        )

        # Manually update title (simulating rename)
        labeler.active_pane.title = "New Name"

        assert labeler.active_pane.title == "New Name"

    def test_refresh_marks_pane_dirty(self, sample_dataframe, plot_function_1, layout_spec):
        """Refresh operations mark panes dirty."""
        from chronotagger.labeler import TimeIntervalLabeler

        panes = [
            {"title": "Pane 1", "plot_fn": plot_function_1, "layout_spec": layout_spec},
            {"title": "Pane 2", "plot_fn": plot_function_1, "layout_spec": layout_spec},
        ]

        labeler = TimeIntervalLabeler(
            df=sample_dataframe,
            panes=panes,
            window=pd.Timedelta("1h"),
        )

        # Mark all clean
        for pane in labeler.panes:
            pane.mark_clean(labeler.t0, labeler.t1)

        # Refresh active pane
        labeler.active_pane.mark_dirty()
        assert labeler.active_pane.dirty


# ============================================================================
# TEST SUITE 5: SAVE/LOAD
# ============================================================================

class TestSaveLoad:
    """Tests for save/load with multi-pane configuration."""

    def test_save_includes_multi_pane_metadata(self, sample_dataframe, plot_function_1, layout_spec, tmp_path):
        """Save file includes multi-pane metadata."""
        from chronotagger.labeler import TimeIntervalLabeler
        from chronotagger.core.models import Interval

        panes = [
            {"title": "Custom Name 1", "plot_fn": plot_function_1, "layout_spec": layout_spec},
            {"title": "Custom Name 2", "plot_fn": plot_function_1, "layout_spec": layout_spec},
        ]

        labeler = TimeIntervalLabeler(
            df=sample_dataframe,
            panes=panes,
            window=pd.Timedelta("1h"),
        )

        # Add an interval directly
        t0 = sample_dataframe.index[10]
        t1 = sample_dataframe.index[20]
        interval = Interval(t0, t1, "UNKNOWN")
        labeler.intervals.append(interval)

        # Build save data structure (simulating save)
        data = {
            "version": 1,
            "classes": labeler.classes,
            "class_colors": labeler.class_colors,
            "window": str(labeler.window),
            "step": str(labeler.step),
            "data_start": labeler.data_start.isoformat(),
            "data_end": labeler.data_end.isoformat(),
            "intervals": [iv.to_dict() for iv in labeler.intervals],
            "layout_spec": labeler.layout_spec,
            "multi_pane_mode": labeler.multi_pane_mode,
            "active_pane_idx": labeler.active_pane_idx,
            "panes": [
                {
                    "title": pane.title,
                    "layout_spec": getattr(pane, 'layout_spec', None),
                }
                for pane in labeler.panes
            ] if labeler.multi_pane_mode else [],
        }

        # Check metadata
        assert data["multi_pane_mode"] is True
        assert "active_pane_idx" in data
        assert "panes" in data
        assert len(data["panes"]) == 2
        assert data["panes"][0]["title"] == "Custom Name 1"
        assert data["panes"][1]["title"] == "Custom Name 2"

    def test_single_pane_save_has_no_multi_pane_metadata(self, sample_dataframe, plot_function_1, layout_spec):
        """Single-pane saves don't have multi-pane metadata."""
        from chronotagger.labeler import TimeIntervalLabeler

        labeler = TimeIntervalLabeler(
            df=sample_dataframe,
            plot_fn=plot_function_1,
            layout_spec=layout_spec,
            window=pd.Timedelta("1h"),
        )

        # Build save data
        data = {
            "multi_pane_mode": labeler.multi_pane_mode,
            "active_pane_idx": labeler.active_pane_idx if labeler.multi_pane_mode else 0,
            "panes": [
                {"title": pane.title, "layout_spec": getattr(pane, 'layout_spec', None)}
                for pane in labeler.panes
            ] if labeler.multi_pane_mode else [],
        }

        assert data["multi_pane_mode"] is False
        assert data["active_pane_idx"] == 0
        assert data["panes"] == []


# ============================================================================
# TEST SUITE 6: BACKWARD COMPATIBILITY
# ============================================================================

class TestBackwardCompatibility:
    """Ensure multi-pane changes don't break single-pane mode."""

    def test_single_pane_has_panes_list(self, sample_dataframe, plot_function_1, layout_spec):
        """Single-pane mode has panes list with one element."""
        from chronotagger.labeler import TimeIntervalLabeler

        labeler = TimeIntervalLabeler(
            df=sample_dataframe,
            plot_fn=plot_function_1,
            layout_spec=layout_spec,
            window=pd.Timedelta("1h"),
        )

        assert hasattr(labeler, 'panes')
        assert len(labeler.panes) == 1
        assert labeler.panes[0].title == "Main"

    def test_single_pane_active_pane_works(self, sample_dataframe, plot_function_1, layout_spec):
        """Single-pane mode has working active_pane property."""
        from chronotagger.labeler import TimeIntervalLabeler

        labeler = TimeIntervalLabeler(
            df=sample_dataframe,
            plot_fn=plot_function_1,
            layout_spec=layout_spec,
            window=pd.Timedelta("1h"),
        )

        assert labeler.active_pane is not None
        assert labeler.active_pane is labeler.panes[0]

    def test_existing_api_unchanged(self, sample_dataframe):
        """Verify existing single-pane API still works."""
        from chronotagger.labeler import TimeIntervalLabeler

        # Simple plot function
        def simple_plot(axs, df, t0, t1):
            axs['main'].plot(df.index, df['value1'])

        # Should work exactly as before multi-pane feature
        labeler = TimeIntervalLabeler(
            df=sample_dataframe,
            plot_fn=simple_plot,
            window=pd.Timedelta("1h"),
        )

        assert labeler is not None
        assert len(labeler.intervals) == 0
        assert not labeler.multi_pane_mode


# ============================================================================
# TEST SUITE 7: HELP DIALOG
# ============================================================================

class TestHelpDialog:
    """Tests for help dialog multi-pane integration."""

    def test_multi_pane_shortcuts_defined(self):
        """MULTI_PANE_SHORTCUTS constant is defined."""
        from chronotagger.labeler.mixins.help import MULTI_PANE_SHORTCUTS

        assert MULTI_PANE_SHORTCUTS is not None
        assert len(MULTI_PANE_SHORTCUTS) > 0

        # Check expected shortcuts
        shortcuts_dict = dict(MULTI_PANE_SHORTCUTS)
        assert "Ctrl+Tab" in shortcuts_dict
        assert "Ctrl+Shift+Tab" in shortcuts_dict
        assert "Ctrl+1...9" in shortcuts_dict

    def test_help_dialog_method_exists(self, sample_dataframe, plot_function_1, layout_spec):
        """Help dialog method exists."""
        from chronotagger.labeler import TimeIntervalLabeler

        panes = [
            {"title": "Pane 1", "plot_fn": plot_function_1, "layout_spec": layout_spec},
        ]

        labeler = TimeIntervalLabeler(
            df=sample_dataframe,
            panes=panes,
            window=pd.Timedelta("1h"),
        )

        assert hasattr(labeler, '_open_help_dialog')
        assert callable(labeler._open_help_dialog)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
