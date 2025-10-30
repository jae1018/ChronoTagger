import matplotlib.dates as mdates


def test_time_axis_uses_concise_formatter(labeler):
    for ax in list(labeler.user_axes.values()) + [labeler.strip_ax]:
        fmt = ax.xaxis.get_major_formatter()
        assert isinstance(fmt, mdates.ConciseDateFormatter)


def test_overlay_rectangles_drawn_on_user_axes(labeler):
    # Add one interval that is inside the window
    from chronotagger.core.models import Interval
    a, b = labeler.df.index[10], labeler.df.index[40]
    labeler.intervals = [Interval(a, b, "PS")]
    labeler._update_plot()

    # On each user axis we expect exactly 1 dim overlay (alpha ~= 0.15)
    for ax in labeler.user_axes.values():
        overlays = [p for p in ax.patches if abs((p.get_alpha() or 1.0) - 0.15) < 1e-6]
        assert len(overlays) == 1
