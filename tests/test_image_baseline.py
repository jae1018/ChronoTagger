# tests/test_image_baseline.py
import pytest
import matplotlib
matplotlib.use("Agg", force=True)  # non-interactive backend for tests

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pandas as pd

@pytest.mark.mpl_image_compare(tolerance=20)
def test_baseline_overlay_image(df_hour, plot_fn):
    from chronotagger.labeler import TimeIntervalLabeler

    # Define layout_spec for grid-only mode
    layout_spec = {
        'nrows': 3,
        'ncols': 1,
        'areas': [
            {'key': 'panel1', 'row': 0, 'col': 0, 'rowspan': 1, 'colspan': 1, 'role': 'time'},
            {'key': 'panel2', 'row': 1, 'col': 0, 'rowspan': 1, 'colspan': 1, 'role': 'time'},
            {'key': 'labels', 'row': 2, 'col': 0, 'rowspan': 1, 'colspan': 1, 'role': 'labels'}
        ]
    }

    lbl = TimeIntervalLabeler(df=df_hour, plot_fn=plot_fn, layout_spec=layout_spec)

    # --- Build figure & axes without Tk ---
    fig = Figure(figsize=(10, 6))
    canvas = FigureCanvasAgg(fig)
    lbl.fig = fig
    lbl.canvas = canvas

    gs = fig.add_gridspec(5, 1, height_ratios=[3, 3, 3, 3, 1], hspace=0.3)
    lbl.user_axes = {
        "panel1": fig.add_subplot(gs[0, 0]),
        "panel2": fig.add_subplot(gs[1, 0]),
    }
    lbl.strip_ax = fig.add_subplot(gs[4, 0], sharex=lbl.user_axes["panel1"])

    # Set up time axis tracking (normally done by _build_gui)
    lbl._time_axis_keys = {"panel1", "panel2"}
    lbl._primary_time_key = "panel1"

    # Apply time formatting like the app would
    for ax in list(lbl.user_axes.values()) + [lbl.strip_ax]:
        lbl._apply_time_axis_format(ax)

    # Draw once
    lbl._update_plot()

    # Return figure for mpl_image_compare
    return fig
