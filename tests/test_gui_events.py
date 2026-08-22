import pytest
import matplotlib.dates as mdates


@pytest.fixture(autouse=True)
def _stub_messagebox(monkeypatch):
    """Mutated trees route _add_interval into warning branches the green
    suite never reaches, and an unstubbed messagebox pops a REAL blocking
    dialog on a machine with a live display (observed twice during Pack 3
    verification). Every real-Tk test module stubs it."""
    import tkinter.messagebox as mb
    for kind in ("showinfo", "showwarning", "showerror", "askyesno"):
        monkeypatch.setattr(mb, kind, lambda *a, **kw: True)


def test_drag_select_then_add(labeler):
    t1 = labeler.df.index[5]
    t2 = labeler.df.index[20]
    ax = list(labeler.user_axes.values())[0]
    # No ydata on purpose: NaN routes down the full-height (time-only) branch.
    e1 = type("E", (), {"xdata": mdates.date2num(t1), "inaxes": ax})
    e2 = type("E", (), {"xdata": mdates.date2num(t2), "inaxes": ax})

    # Pass active_pane as third argument (Phase 3 signature change)
    labeler._on_rectangle_select(e1, e2, labeler.active_pane)
    labeler._add_interval()

    assert len(labeler.intervals) == 1
    iv = labeler.intervals[0]
    # WYSIWYG (Pack 3, R5): the drag visibly covers samples 5..20, so the
    # stored half-open interval is [idx[5], idx[21]) and labels sample 20.
    assert iv.start == t1
    assert iv.end == labeler.df.index[21]
    assert iv.contains(t2)


def test_mouse_wheel_zoom_time_range(labeler):
    """Wheel over empty canvas (inaxes=None) zooms the time window in.

    Replaces the permanently dead test that probed a handler named
    _on_scroll (never existed; the real one is _on_scroll_zoom)."""
    before = labeler.t1 - labeler.t0
    ev = type("Evt", (), {"inaxes": None, "xdata": None, "button": "up",
                          "step": 1, "key": None})
    labeler._on_scroll_zoom(ev, labeler.active_pane)
    after = labeler.t1 - labeler.t0
    assert after < before
