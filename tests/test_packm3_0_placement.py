"""Pack M3.0, item B -- THE THREE EVERY-LAUNCH PAPERCUTS.

1 and 2. WHERE A DIALOG OPENS. Not one dialog in `labeler/dialogs/` set a
position at all, so the window manager cascaded it onto the primary
monitor; the two that DID place themselves -- the recovery box and Select
Component -- centred on the SCREEN, never on the app. One helper,
`dialogs/_placement.py`, now centres every one of them over the MAIN
WINDOW and clamps it to the screen that holds it.

    Measured at base, inside the method: the recovery box calls
    transient(parent) between geometry("640x700") and update_idletasks(),
    and transient on an unmapped Toplevel throws the requested size away.
    winfo_width() answers 1, so the corner landed at (960, 540) of a
    1920x1080 screen -- a 640x700 box whose bottom 160 px are off the
    edge. The same sequence WITHOUT transient() measures 640x700 and
    centres correctly, which is why a reconstruction that left that line
    out saw no defect. The fix stops the measurement: the size is passed
    in, and the centre is the main window's.

3. THE PLOT TOOLBAR. The canvas was packed FIRST and asks for 800 px of
height; the Tk packer serves parcels in pack order out of a shrinking
cavity, so at the default 1600x900 root the toolbar -- packed after it,
needing 50 -- got nothing on every tab unless the window was maximized.
The toolbar is now served first and is kept on the pane instead of in a
throwaway local.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tkinter as tk

DAY = "2015-01-03 "

REGION = {"id": "region", "name": "Region (human)",
          "classes": ["sw", "UNKNOWN"], "order": 0}
WAKE = {"id": "wake", "name": "Wake (umbra)",
        "classes": ["umbra", "UNKNOWN"], "order": 1}

LAYOUT = {
    "nrows": 3, "ncols": 1,
    "areas": [
        {"key": "panel1", "row": 0, "col": 0, "role": "time"},
        {"key": "panel2", "row": 1, "col": 0, "role": "time"},
        {"key": "labels", "row": 2, "col": 0, "role": "labels"},
    ],
}


def frame():
    idx = pd.date_range(DAY + "00:00:00", periods=120, freq="30s")
    return pd.DataFrame({"a": np.linspace(0.0, 1.0, len(idx)),
                         "b": np.linspace(1.0, 2.0, len(idx))}, index=idx)


def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["a"])
    axs["panel2"].plot(df.index, df["b"])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def _no_modal_blocking(monkeypatch):
    """wait_visibility, grab_set and wait_window HANG head-lessly."""
    monkeypatch.setattr(tk.Misc, "wait_visibility", lambda *a, **k: None)
    monkeypatch.setattr(tk.Misc, "grab_set", lambda *a, **k: None)
    monkeypatch.setattr(tk.Misc, "wait_window", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _quiet_toplevel(monkeypatch):
    """The two mixin-built boxes make a REAL Toplevel. Never map one."""
    real = tk.Toplevel

    class Quiet(real):
        def __init__(self, *a, **k):
            real.__init__(self, *a, **k)
            try:
                self.withdraw()
            except Exception:
                pass

    monkeypatch.setattr(tk, "Toplevel", Quiet)
    return Quiet


def build(tracks, tmp_path, panes=None):
    from chronotagger.labeler import TimeIntervalLabeler
    if panes is None:
        kw = {"plot_fn": plot_fn, "layout_spec": dict(LAYOUT)}
    else:
        kw = {"panes": panes}
    lbl = TimeIntervalLabeler(
        df=frame(),
        window=pd.Timedelta("60min"), autosave_folder=str(tmp_path),
        tracks=[dict(t) for t in tracks], **kw)
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    return lbl


@pytest.fixture
def app(tmp_path):
    lbl = build([REGION, WAKE], tmp_path)
    yield lbl
    lbl.root.destroy()


# ============================== 1. the helper, as pure numbers


def test_the_helper_centres_over_the_parent():
    from chronotagger.labeler.dialogs._placement import centered_geometry
    # a 400x300 dialog over a 1000x800 window whose corner is at (100, 50)
    assert centered_geometry(100, 50, 1000, 800, 400, 300,
                             1920, 1080) == (400, 300)


def test_the_helper_clamps_at_every_screen_edge():
    from chronotagger.labeler.dialogs._placement import centered_geometry
    # parent hanging off the LEFT and the TOP
    assert centered_geometry(-100, -60, 400, 300, 400, 300,
                             1920, 1080) == (0, 0)
    # parent hanging off the RIGHT and the BOTTOM
    assert centered_geometry(1800, 1000, 100, 60, 400, 300,
                             1920, 1080) == (1520, 780)


def test_a_dialog_bigger_than_the_screen_pins_to_the_top_left():
    from chronotagger.labeler.dialogs._placement import centered_geometry
    assert centered_geometry(0, 0, 800, 600, 3000, 2000,
                             1920, 1080) == (0, 0)


def test_a_parent_on_another_monitor_is_not_dragged_back():
    """`winfo_screenwidth` describes the PRIMARY display only."""
    from chronotagger.labeler.dialogs._placement import centered_geometry
    # main window at x=2200 -- a second monitor to the right
    assert centered_geometry(2200, 100, 1000, 800, 400, 300,
                             1920, 1080) == (2500, 350)


def test_the_helper_sets_the_geometry_it_computed(app):
    from chronotagger.labeler.dialogs import _placement
    seen = []
    dlg = tk.Toplevel(app.root)
    try:
        dlg.geometry = lambda spec: seen.append(spec)
        got = _placement.center_on_parent(dlg, app.root, 400, 300)
        assert seen == [got]
        assert got.startswith("+")
    finally:
        dlg.destroy()


def test_the_helper_never_raises_on_a_dead_widget(app):
    from chronotagger.labeler.dialogs import _placement
    assert _placement.center_on_parent(None, app.root) is None
    assert _placement.center_on_parent(app.root, None) is None


# ============================== 2. every dialog asks the helper


@pytest.fixture
def spy(monkeypatch):
    from chronotagger.labeler.dialogs import _placement
    seen = []
    real = _placement.center_on_parent

    def _spy(dialog, parent, width=None, height=None):
        seen.append({"dialog": type(dialog).__name__, "parent": parent,
                     "width": width, "height": height})
        return real(dialog, parent, width, height)

    monkeypatch.setattr(_placement, "center_on_parent", _spy)
    return seen


def test_manage_labels_is_centred_on_its_parent(app, spy):
    from chronotagger.labeler.dialogs.label_manager import (
        LabelManagerDialog)
    dlg = LabelManagerDialog(parent=app.root, classes=["a"],
                             class_colors={"a": "#111111"},
                             usage_counts={"a": 0})
    try:
        assert [s["dialog"] for s in spy] == ["LabelManagerDialog"]
        assert spy[0]["parent"] is app.root
    finally:
        dlg.destroy()


def test_the_reassign_box_is_centred_on_manage_labels(app, spy):
    from chronotagger.labeler.dialogs.label_manager import _ReassignDialog
    dlg = _ReassignDialog(app.root, "sw", ["msh", "UNKNOWN"])
    try:
        assert [s["dialog"] for s in spy] == ["_ReassignDialog"]
    finally:
        dlg.destroy()


def test_label_by_rule_is_centred_on_its_parent(app, spy):
    from chronotagger.labeler.dialogs.label_by_rule import (
        LabelByRuleDialog)
    dlg = LabelByRuleDialog(parent=app.root, numeric_columns=["a", "b"],
                            on_preview=lambda r: (0, 0),
                            on_clear_preview=lambda: None)
    try:
        assert [s["dialog"] for s in spy] == ["LabelByRuleDialog"]
    finally:
        dlg.destroy()


def test_the_overlap_box_is_centred_on_its_parent(app, spy):
    from chronotagger.labeler.dialogs.overlap_resolution import (
        OverlapResolutionDialog)
    dlg = OverlapResolutionDialog(app.root, 3, lambda policy: (10, 2))
    try:
        assert [s["dialog"] for s in spy] == ["OverlapResolutionDialog"]
    finally:
        dlg.destroy()


def test_the_recovery_box_is_centred_on_the_main_window(app, spy):
    app._show_recovery_dialog({"intervals": [], "metadata": {},
                               "label_stats": {}})
    assert len(spy) == 1
    assert spy[0]["parent"] is app.root
    assert (spy[0]["width"], spy[0]["height"]) == (640, 700), \
        "its content is built AFTER the call, so the size is passed in"


def test_select_component_is_centred_on_the_main_window(app, spy):
    app._show_component_selection_dialog(
        {"B_x": app.df.index[:5], "B_y": app.df.index[2:8]},
        app.active_pane)
    assert len(spy) == 1
    assert spy[0]["parent"] is app.root
    assert (spy[0]["width"], spy[0]["height"]) == (380, 350)


def test_the_recovery_box_no_longer_measures_an_empty_window(app):
    """SOURCE-SHAPE PIN, chosen over a spy on purpose: what has to be
    true is that the broken arithmetic is GONE from the file, and a spy
    can only ever say that one run did not reach it."""
    import chronotagger.labeler.mixins.io_export as io_mod
    import chronotagger.labeler.mixins.events.selection as sel_mod
    for mod in (io_mod, sel_mod):
        src = open(mod.__file__, "r", encoding="utf-8").read()
        assert "dialog.winfo_width()" not in src, mod.__name__
        assert "dialog.winfo_height()" not in src, mod.__name__
        assert "winfo_screenwidth() // 2" not in src, mod.__name__


# ============================== 3. the plot toolbar


def test_every_pane_keeps_its_toolbar_and_serves_it_first(app):
    for pane in app.panes:
        tb = getattr(pane, "toolbar", None)
        assert tb is not None, "the toolbar must be kept on the pane"
        widget = pane.canvas.get_tk_widget()
        slaves = widget.master.pack_slaves()
        assert tb in slaves
        assert slaves.index(tb) < slaves.index(widget), \
            "the packer must serve the toolbar before the canvas"
        assert tb.winfo_reqheight() > 0


def test_the_toolbar_is_there_on_every_tab_of_a_multi_pane_window(
        tmp_path):
    lbl = build([REGION, WAKE], tmp_path,
                panes=[{"title": "One", "plot_fn": plot_fn,
                        "layout_spec": dict(LAYOUT)},
                       {"title": "Two", "plot_fn": plot_fn,
                        "layout_spec": dict(LAYOUT)},
                       {"title": "Three", "plot_fn": plot_fn,
                        "layout_spec": dict(LAYOUT)}])
    try:
        assert len(lbl.panes) == 3
        for pane in lbl.panes:
            widget = pane.canvas.get_tk_widget()
            slaves = widget.master.pack_slaves()
            assert slaves.index(pane.toolbar) < slaves.index(widget)
    finally:
        lbl.root.destroy()
