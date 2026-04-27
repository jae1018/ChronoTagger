"""
Unit tests for quick-start wizard validation helpers.

The TabPlanner validates that any custom_grid tab whose layout was designed
against an earlier column selection still references columns the user has
selected (commit 66b6a58 added the check after a codex review).  The
helper that drives the check is `_columns_referenced_by_plot_config`,
which is a pure function on a plot_config dict and exercised here without
spinning up the Tk dialog.
"""

import pytest

from chronotagger.quickstart.tab_planner import TabPlannerDialog


extract = TabPlannerDialog._columns_referenced_by_plot_config


class TestColumnsReferencedByPlotConfig:
    """Cover the column extractor that drives custom-grid stale detection."""

    def test_empty_config_returns_empty_set(self):
        assert extract({}) == set()

    def test_single_time_panel(self):
        plot_config = {
            "panel0": {"role": "time", "y_column": "feat_0"},
        }
        assert extract(plot_config) == {"feat_0"}

    def test_single_cross_plot_panel(self):
        plot_config = {
            "panel0": {"role": "not-time", "x_column": "feat_0", "y_column": "feat_1"},
        }
        assert extract(plot_config) == {"feat_0", "feat_1"}

    def test_mixed_time_and_cross_plot(self):
        plot_config = {
            "panel0": {"role": "time", "y_column": "feat_0"},
            "panel1": {"role": "time", "y_column": "feat_1"},
            "panel2": {
                "role": "not-time",
                "x_column": "feat_2",
                "y_column": "feat_3",
            },
        }
        assert extract(plot_config) == {"feat_0", "feat_1", "feat_2", "feat_3"}

    def test_overlapping_columns_deduplicated(self):
        # Two panels can legitimately reference the same column (e.g. the
        # user plots feat_0 in a time panel AND uses it as the x-axis of a
        # cross-plot).  The helper returns a set, so duplicates collapse.
        plot_config = {
            "panel0": {"role": "time", "y_column": "feat_0"},
            "panel1": {
                "role": "not-time",
                "x_column": "feat_0",
                "y_column": "feat_1",
            },
        }
        assert extract(plot_config) == {"feat_0", "feat_1"}

    def test_non_dict_panel_value_does_not_crash(self):
        # plot_config values are normally dicts but a corrupt session-load
        # could leave a non-dict in there.  The helper must skip rather
        # than raise, so a single bad entry doesn't take the whole wizard
        # down.
        plot_config = {
            "panel0": {"role": "time", "y_column": "feat_0"},
            "panel1": "not a dict",
            "panel2": None,
        }
        assert extract(plot_config) == {"feat_0"}

    def test_missing_column_keys_skipped(self):
        # A panel dict without x_column/y_column fields (e.g. an annotation
        # or labels panel that snuck into plot_config) should contribute
        # nothing.
        plot_config = {
            "panel0": {"role": "time"},  # no y_column
            "panel1": {"role": "not-time", "x_column": "feat_0"},  # no y_column
        }
        assert extract(plot_config) == {"feat_0"}

    def test_empty_string_column_ignored(self):
        # A blank string from an unfinished UI selection should not be
        # treated as a column reference.
        plot_config = {
            "panel0": {"role": "time", "y_column": ""},
            "panel1": {"role": "not-time", "x_column": "feat_0", "y_column": ""},
        }
        assert extract(plot_config) == {"feat_0"}

    def test_non_string_column_value_ignored(self):
        # A None or numeric value where a column name is expected should
        # not raise.
        plot_config = {
            "panel0": {"role": "time", "y_column": None},
            "panel1": {"role": "not-time", "x_column": 42, "y_column": "feat_0"},
        }
        assert extract(plot_config) == {"feat_0"}


class TestStaleDetectionUsage:
    """
    Confirm the calling pattern in TabPlannerDialog._validate matches the
    helper's behavior: missing = referenced - selected.  Tests the math
    rather than the dialog-driving code (which needs a Tk root).
    """

    def test_no_columns_missing_when_selection_is_superset(self):
        plot_config = {
            "panel0": {"role": "not-time", "x_column": "feat_0", "y_column": "feat_1"},
        }
        selected = {"feat_0", "feat_1", "feat_2", "feat_3"}
        missing = extract(plot_config) - selected
        assert missing == set()

    def test_missing_columns_reported_when_selection_drops_a_referenced_column(self):
        plot_config = {
            "panel0": {"role": "time", "y_column": "feat_0"},
            "panel1": {
                "role": "not-time",
                "x_column": "feat_2",
                "y_column": "feat_3",
            },
        }
        # User unchecked feat_2 after designing the layout.
        selected = {"feat_0", "feat_1", "feat_3"}
        missing = extract(plot_config) - selected
        assert missing == {"feat_2"}

    def test_multiple_missing_columns_reported(self):
        plot_config = {
            "panel0": {"role": "not-time", "x_column": "feat_0", "y_column": "feat_1"},
        }
        selected = {"feat_2"}
        missing = extract(plot_config) - selected
        assert missing == {"feat_0", "feat_1"}
