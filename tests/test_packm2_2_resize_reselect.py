"""Pack M2.2 -- AFTER A RESIZE, THE SELECTION STAYS ON THE DRAGGED LANE.

The post-resize reselect in events/mouse.py used to scan every interval
in the session for one covering the new midpoint with the same label,
with no lane filter. With two lanes sharing a class name, a right-edge
drag on lane A committed correctly but the selection landed on lane B's
same-label interval when it came first in the list, the status bar
reported lane B's span, and the next Delete removed lane B's interval, a
hidden lane included. Only a lock stopped it. Found by the overnight
data-safety refuter on the implemented Pack M2 tree.

The drag state below is exactly what _on_strip_press leaves behind for a
right-edge resize on the ACTIVE, unlocked lane: the lock gate and the
lane-gated hit test are upstream and both pass.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from chronotagger.core.models import Interval

LAYOUT = {"nrows": 2, "ncols": 1,
          "areas": [{"key": "p1", "row": 0, "col": 0, "role": "time"},
                    {"key": "labels", "row": 1, "col": 0, "role": "labels"}]}
IDX = pd.date_range("2024-01-01 00:00:00", periods=241, freq="30s")


def lanes(model_locked=False, model_visible=True):
    return [
        {"id": "human", "name": "Human", "classes": ["UNKNOWN", "solar_wind"],
         "class_colors": {"UNKNOWN": "#7f7f7f", "solar_wind": "#4e79a7"},
         "order": 0},
        {"id": "model", "name": "Model", "classes": ["UNKNOWN", "solar_wind"],
         "class_colors": {"UNKNOWN": "#999999", "solar_wind": "#ff0000"},
         "locked": model_locked, "visible": model_visible, "order": 1},
    ]


@pytest.fixture(autouse=True)
def _stub_dialogs(monkeypatch):
    """STANDING RULE (Pack 3): stub every dialog on any reachable path."""
    import tkinter.messagebox as mb
    import tkinter.filedialog as fd
    calls = []
    for kind in ("showinfo", "showwarning", "showerror", "askyesno",
                 "askyesnocancel", "askokcancel"):
        monkeypatch.setattr(
            mb, kind, lambda *a, _k=kind, **kw: calls.append(_k) or True)
    for kind in ("asksaveasfilename", "askopenfilename", "askdirectory"):
        monkeypatch.setattr(fd, kind, lambda *a, **kw: "")
    return calls


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def build(tmp_path, tracks):
    from chronotagger.labeler import TimeIntervalLabeler
    df = pd.DataFrame({"v": np.linspace(0.0, 1.0, len(IDX))}, index=IDX)
    app = TimeIntervalLabeler(df=df, plot_fn=lambda axs, sub, a, b: None,
                              tracks=tracks, window=pd.Timedelta("2h"),
                              autosave_folder=str(tmp_path),
                              layout_spec=dict(LAYOUT))
    app._build_gui()
    app.root.withdraw()
    # the MODEL lane's interval is FIRST in the list and spans the record
    model_iv = Interval(IDX[0], IDX[240], "solar_wind", None, "model")
    human_iv = Interval(IDX[40], IDX[80], "solar_wind", None, "human")
    app.intervals = [model_iv, human_iv]
    app._set_active_track("human")
    app._update_plot()
    return app, human_iv


def drag_right_edge(app, human_iv):
    app.selected_interval = human_iv
    app._drag_mode = "resize_right"
    app._drag_iv = human_iv
    app._drag_initial = (human_iv.start, min(human_iv.end, app.data_end))
    app._drag_preview = (IDX[40], IDX[100])

    class Ev(object):
        button = 1
        inaxes = app.active_pane.strip_ax
        x = 0
        y = 0

    app._on_strip_release(Ev(), app.active_pane)


def per_lane(app):
    return {t.id: len([iv for iv in app.intervals if iv.track == t.id])
            for t in app.tracks}


@pytest.mark.parametrize("model_locked,model_visible", [
    (False, True), (False, False), (True, True)])
def test_the_selection_after_a_resize_is_the_dragged_lanes_interval(
        tmp_path, model_locked, model_visible):
    app, human_iv = build(tmp_path, lanes(model_locked, model_visible))
    try:
        drag_right_edge(app, human_iv)
        sel = app.selected_interval
        assert sel is not None
        assert sel.track == "human", sel.track
        assert (sel.start, sel.end) == (IDX[40], IDX[100])
        assert "00:20:00" in app.status_var.get()
        assert per_lane(app) == {"human": 1, "model": 1}
        app._delete_interval()
        assert per_lane(app) == {"human": 0, "model": 1}, per_lane(app)
    finally:
        app.root.destroy()


def test_a_resize_never_moves_the_selection_to_a_hidden_lane(tmp_path):
    app, human_iv = build(tmp_path, lanes(model_visible=False))
    try:
        drag_right_edge(app, human_iv)
        assert app.selected_interval.track == "human"
        app._delete_interval()
        assert per_lane(app)["model"] == 1
    finally:
        app.root.destroy()
