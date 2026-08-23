import matplotlib.dates as mdates


def test_time_axis_uses_concise_formatter(labeler):
    for ax in list(labeler.user_axes.values()) + [labeler.strip_ax]:
        fmt = ax.xaxis.get_major_formatter()
        assert isinstance(fmt, mdates.ConciseDateFormatter)


def test_overlay_bands_drawn_on_user_axes(labeler):
    """
    Pack 5 R4c: interval bands are ONE PolyCollection per axis, not one
    axvspan Rectangle per interval per axis (measured: 8,000 patches for
    2,000 intervals on a 4-panel figure, 6.9 s of frame time). The
    SEMANTICS this test owns are unchanged -- one dim band per interval,
    on every user axis, at the caller's alpha.
    """
    from chronotagger.core.models import Interval
    a, b = labeler.df.index[10], labeler.df.index[40]
    labeler.intervals = [Interval(a, b, "PS")]
    labeler._update_plot()

    for ax in labeler.user_axes.values():
        bands = [c for c in ax.collections
                 if str(c.get_gid() or "").endswith("interval-bands")]
        assert len(bands) == 1, "exactly one band collection per axis"
        faces = bands[0].get_facecolor()
        assert len(faces) == 1, "one face per visible interval"
        assert abs(float(faces[0][3]) - 0.15) < 1e-6, "alpha carried per face"
        assert len(bands[0].get_paths()) == 1
