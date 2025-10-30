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

    lbl = TimeIntervalLabeler(df=df_hour, plot_fn=plot_fn)

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

    # Apply time formatting like the app would
    for ax in list(lbl.user_axes.values()) + [lbl.strip_ax]:
        lbl._apply_time_axis_format(ax)

    # Draw once
    lbl._update_plot()

    # Return figure for mpl_image_compare
    return fig
