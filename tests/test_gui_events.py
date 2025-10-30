import matplotlib.dates as mdates


def test_drag_select_then_add(labeler):
    t1 = labeler.df.index[5]
    t2 = labeler.df.index[20]
    e1 = type("E", (), {"xdata": mdates.date2num(t1)})
    e2 = type("E", (), {"xdata": mdates.date2num(t2)})

    labeler._on_rectangle_select(e1, e2)
    labeler._add_interval()

    assert len(labeler.intervals) == 1
    iv = labeler.intervals[0]
    assert iv.start == t1 and iv.end == t2


def test_mouse_wheel_zoom_in_if_available(labeler):
    """Only assert if the project has the wheel-zoom handler hooked up."""
    handler = getattr(labeler, "_on_scroll", None)
    if handler is None:
        return  # feature not implemented in your tree yet

    mid = labeler.t0 + (labeler.t1 - labeler.t0) / 2
    ax = list(labeler.user_axes.values())[0]

    before = labeler.t1 - labeler.t0
    ev = type("Evt", (), {"inaxes": ax, "xdata": mid, "button": "up", "step": 1})
    handler(ev)
    after = labeler.t1 - labeler.t0
    assert after < before
