"""
Unit tests for chronotagger.labeler.utils.plot_generator.generate_plot_fn.

These tests pin down the artist-removal behaviour added in commit 90efa42
("Fix shared-axis state reset in generate_plot_fn").  Earlier the
generated plot_fn called ax.clear() between renders, which also reset
the datetime unit converter and the shared-axis state the labeler had
just set up -- so subsequent set_xlim(t0, t1) calls only constrained
whichever axis matplotlib touched first, leaving the other axis with
its auto-fit (~60-year) range and rendering its line invisibly.

The fix is to remove only data artists (lines, collections, patches,
texts) explicitly.  These tests confirm:

  1. A second render does not accumulate artists on top of the first.
  2. Shared-axis (sharex) state survives a re-render.  This is the
     load-bearing regression check: ax.clear() detaches the datetime
     unit converter as a side effect, which is what broke sharex
     propagation between time axes in the original bug.
"""

import matplotlib

matplotlib.use("Agg")  # no display required

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from chronotagger.labeler.utils.plot_generator import generate_plot_fn


@pytest.fixture
def df_two_features():
    idx = pd.date_range("2025-01-01 00:00:00", periods=120, freq="30s")
    return pd.DataFrame(
        {
            "feat_0": np.linspace(0.5, 2.0, len(idx)),
            "feat_1": np.sin(np.linspace(0, 30, len(idx))) * 10,
        },
        index=idx,
    )


@pytest.fixture
def time_plot_config():
    return {
        "panel0": {"role": "time", "y_column": "feat_0"},
        "panel1": {"role": "time", "y_column": "feat_1"},
    }


def _make_axs(fig, keys, sharex=True):
    """Build a {key: Axes} dict with the labeler's sharex wiring."""
    axs = {}
    first = None
    for key in keys:
        if first is None or not sharex:
            ax = fig.add_subplot(len(keys), 1, len(axs) + 1)
            first = first or ax
        else:
            ax = fig.add_subplot(len(keys), 1, len(axs) + 1, sharex=first)
        axs[key] = ax
    return axs


class TestArtistRemoval:
    """The core regression: re-rendering must not accumulate artists."""

    def test_second_render_does_not_double_lines(self, df_two_features, time_plot_config):
        fig = Figure()
        axs = _make_axs(fig, ["panel0", "panel1"])
        plot_fn = generate_plot_fn(time_plot_config)

        plot_fn(axs, df_two_features, df_two_features.index[0], df_two_features.index[-1])
        first_pass_counts = {k: len(ax.lines) for k, ax in axs.items()}

        plot_fn(axs, df_two_features, df_two_features.index[0], df_two_features.index[-1])
        second_pass_counts = {k: len(ax.lines) for k, ax in axs.items()}

        # Both panels are time-series with style='line' (default).  Each
        # render adds exactly one Line2D and removes the previous one --
        # so counts must be 1 after pass 1 AND after pass 2.
        assert first_pass_counts == {"panel0": 1, "panel1": 1}
        assert second_pass_counts == {"panel0": 1, "panel1": 1}

    def test_repeated_renders_do_not_accumulate(self, df_two_features, time_plot_config):
        fig = Figure()
        axs = _make_axs(fig, ["panel0", "panel1"])
        plot_fn = generate_plot_fn(time_plot_config)

        for _ in range(5):
            plot_fn(axs, df_two_features, df_two_features.index[0], df_two_features.index[-1])

        for ax in axs.values():
            assert len(ax.lines) == 1, (
                "Each ax must hold exactly one Line2D after any number of "
                "renders; ax.clear() would have reset shared-axis state, "
                "while not removing artists at all would produce N>1."
            )

    def test_scatter_panel_replaces_collection(self, df_two_features):
        # Cross-plots use ax.scatter -> PathCollection on ax.collections,
        # not a Line2D.  Verify those are also removed between renders.
        plot_config = {
            "panel0": {
                "role": "not-time",
                "x_column": "feat_0",
                "y_column": "feat_1",
            },
        }
        fig = Figure()
        axs = _make_axs(fig, ["panel0"], sharex=False)
        plot_fn = generate_plot_fn(plot_config)

        plot_fn(axs, df_two_features, df_two_features.index[0], df_two_features.index[-1])
        plot_fn(axs, df_two_features, df_two_features.index[0], df_two_features.index[-1])

        # One scatter -> one PathCollection.  After two renders, exactly
        # one collection should remain.
        assert len(axs["panel0"].collections) == 1
        # And no line artists from carry-over.
        assert len(axs["panel0"].lines) == 0


class TestSharedAxisStateSurvives:
    """
    The original bug: ax.clear() reset shared-axis wiring, so subsequent
    set_xlim on one axis stopped propagating to the other.  Verify the
    sharex relationship survives a re-render.
    """

    def test_sharex_relationship_survives_rerender(self, df_two_features, time_plot_config):
        fig = Figure()
        axs = _make_axs(fig, ["panel0", "panel1"], sharex=True)
        plot_fn = generate_plot_fn(time_plot_config)

        plot_fn(axs, df_two_features, df_two_features.index[0], df_two_features.index[-1])
        plot_fn(axs, df_two_features, df_two_features.index[0], df_two_features.index[-1])

        # Setting xlim on panel0 must propagate to panel1 via the
        # shared-axis link.  If ax.clear() had been called between
        # renders, the link would be broken and panel1's xlim would
        # remain at its auto value.  This is the actual regression
        # check for the original bug -- ax.clear() detached the
        # datetime unit converter as a side effect, which broke
        # sharex propagation for date axes.
        new_xlim = (
            mdates.date2num(pd.Timestamp("2025-01-01 00:10:00")),
            mdates.date2num(pd.Timestamp("2025-01-01 00:20:00")),
        )
        axs["panel0"].set_xlim(new_xlim)
        np.testing.assert_allclose(axs["panel1"].get_xlim(), new_xlim, rtol=1e-6)
