"""Pack M2.8 -- A LIST REFILL MUST NOT DROP THE SELECTION.

THE DEFECT (session-5 S5.1, measured on 82acc7a by probe_s5_lowdown Q1a
and Q1b). Select an interval -- by clicking its row in the sidebar list
OR by clicking its band on the strip -- and press Ctrl+Down. The status
bar says

    Active lane: Model -- the selected interval stays on 'Human', which
    this list is not showing

and ONE TURN OF THE EVENT LOOP LATER `selected_interval` is None. The
refill rebuilds the Treeview WITHOUT the selected row, ttk answers the
clear with a <<TreeviewSelect>> carrying an EMPTY selection, and
`_on_interval_tree_select`'s `if not sel:` branch clears the selection
with no word on the bar. Ctrl+Up then says only `Active lane: Human` and
the row comes back unselected. The bar had stated something false one
tick earlier.

The same mechanism fires whenever ANY refill drops the selected row: the
Lanes filter, the Show: window filter after n or p, a tab change.

EVERY PIN HERE PUMPS THE EVENT LOOP. ttk queues <<TreeviewSelect>> with
Tcl_QueueEvent, so it is delivered on a LATER turn and `root.update()`
is what delivers it. Without the pump the defect is invisible and so is
the fix -- which is exactly why the Pack M0 pin that lives next door
recorded it as an "honest limit" instead of holding it.

THE LAST GROUP is the other half of the ruling: every user deselect path
still deselects.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from chronotagger.core.models import Interval

DAY = "2015-01-03 "

HUMAN = {"id": "human", "name": "Human", "classes": ["sw", "msh", "UNKNOWN"],
         "class_colors": {"sw": "#4e79a7", "msh": "#f28e2b",
                          "UNKNOWN": "#7f7f7f"}, "order": 0}
MODEL = {"id": "model", "name": "Model", "classes": ["0", "1"],
         "class_colors": {"0": "#111111", "1": "#222222"}, "order": 1}

LAYOUT = {
    "nrows": 2, "ncols": 1,
    "areas": [
        {"key": "panel1", "row": 0, "col": 0, "role": "time"},
        {"key": "labels", "row": 1, "col": 0, "role": "labels"},
    ],
}

STAYS = ("Active lane: Model -- the selected interval stays on 'Human', "
         "which this list is not showing")


def ts(hhmmss):
    return pd.Timestamp(DAY + hhmmss)


def frame():
    idx = pd.date_range(DAY + "00:00:00", periods=240, freq="30s")
    return pd.DataFrame({"a": np.linspace(0.0, 1.0, len(idx))}, index=idx)


def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["a"])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
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


@pytest.fixture
def app(tmp_path):
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(
        df=frame(), plot_fn=plot_fn, layout_spec=dict(LAYOUT),
        window=pd.Timedelta("60min"), autosave_folder=str(tmp_path),
        tracks=[dict(HUMAN), dict(MODEL)])
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    yield lbl
    lbl.root.destroy()


def pump(app, n=4):
    """Turn the event loop so ttk's queued <<TreeviewSelect>> arrives."""
    for _ in range(n):
        app.root.update()


def one_human(app):
    """One interval on the ACTIVE lane, list scoped to the active lane."""
    app.intervals[:] = [
        Interval(ts("00:05:00"), ts("00:08:00"), "sw", None, track="human")]
    assert app.active_track_id == "human"
    assert app._interval_track_scope() == "active"
    app._update_intervals_list()
    return app.intervals[0]


def iid_of(app, iv):
    return [k for k, v in app._interval_row_map.items() if v is iv][0]


def row_click(app, iv):
    """Select the way a click on the list row does: set the Treeview's
    selection and let the event loop deliver the virtual event."""
    app.intervals_tree.selection_set(iid_of(app, iv))
    pump(app)


# ================================ 1. the lane round trip, both selections


def test_a_row_click_selection_survives_a_lane_round_trip(app):
    want = one_human(app)
    row_click(app, want)
    assert app.selected_interval is want

    app._cycle_active_lane(1)
    assert app.status_var.get() == STAYS
    pump(app)
    assert app.selected_interval is want, \
        "the queued empty <<TreeviewSelect>> dropped the selection"
    assert app.status_var.get() == STAYS, \
        "and the bar still says what it said"

    app._cycle_active_lane(-1)
    pump(app)
    assert app.selected_interval is want
    sel = app.intervals_tree.selection()
    assert len(sel) == 1 and app._interval_row_map[sel[0]] is want, \
        "coming back, DR6's re-assert has a selection left to find"


def test_a_band_click_selection_survives_a_lane_round_trip(app):
    """A band click sets selected_interval and repaints; it never touches
    the Treeview selection itself. Both ways in, one way out."""
    want = one_human(app)
    app.selected_interval = want
    app._update_intervals_list()
    pump(app)
    assert app.selected_interval is want

    app._cycle_active_lane(1)
    pump(app)
    assert app.selected_interval is want

    app._cycle_active_lane(-1)
    pump(app)
    assert app.selected_interval is want
    sel = app.intervals_tree.selection()
    assert len(sel) == 1 and app._interval_row_map[sel[0]] is want


def test_delete_still_works_one_tick_after_the_lane_switch(app):
    """The consequence the user meets: today the next Delete pops 'No
    Selection' because the selection went away on the idle turn."""
    want = one_human(app)
    row_click(app, want)
    app._cycle_active_lane(1)
    pump(app)
    app._set_active_track("human", announce=False, repaint=True)
    pump(app)
    app._delete_interval()
    assert app.intervals == []


# ================================ 2. the window filter and the tab change


def test_the_window_filter_does_not_drop_the_selection(app):
    """Show: window, then move the window past the interval and back --
    which is what n and then p do."""
    want = one_human(app)
    row_click(app, want)
    app.interval_scope_var.set("window")
    app._on_interval_scope_change()
    pump(app)
    assert app.selected_interval is want

    app.t0 = ts("01:00:00")
    app.t1 = ts("02:00:00")
    app._update_intervals_list()
    assert app.intervals_tree.get_children() == ()
    pump(app)
    assert app.selected_interval is want, \
        "the window filter dropped a selection it merely stopped showing"

    app.t0 = ts("00:00:00")
    app.t1 = ts("01:00:00")
    app._update_intervals_list()
    pump(app)
    assert app.selected_interval is want
    sel = app.intervals_tree.selection()
    assert len(sel) == 1 and app._interval_row_map[sel[0]] is want


def test_a_tab_change_does_not_drop_the_selection(tmp_path, monkeypatch):
    """The third refill that drops the row. A tab change replots, and the
    replot refills the list (mixins/plotting.py).

    The loop is drained BEFORE anything is selected: a freshly built
    Notebook fires its own <<NotebookTabChanged>> on the first turn, and
    that first tab change refills the list too. Selecting into an
    undrained queue would be measuring the setup, not the gesture.
    """
    from chronotagger.labeler import TimeIntervalLabeler
    panes = [
        {"title": "One", "plot_fn": plot_fn, "layout_spec": dict(LAYOUT)},
        {"title": "Two", "plot_fn": plot_fn, "layout_spec": dict(LAYOUT)},
    ]
    lbl = TimeIntervalLabeler(
        df=frame(), panes=panes, window=pd.Timedelta("60min"),
        autosave_folder=str(tmp_path),
        tracks=[dict(HUMAN), dict(MODEL)])
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    try:
        assert lbl.multi_pane_mode is True
        for _ in range(4):
            lbl.root.update()
        want = Interval(ts("00:05:00"), ts("00:08:00"), "sw", None,
                        track="human")
        lbl.intervals[:] = [want]
        lbl._update_intervals_list()
        lbl.intervals_tree.selection_set(
            [k for k, v in lbl._interval_row_map.items() if v is want][0])
        for _ in range(4):
            lbl.root.update()
        assert lbl.selected_interval is want

        lbl._set_active_track("model", announce=False, repaint=True)
        for _ in range(4):
            lbl.root.update()
        assert lbl.selected_interval is want

        lbl.notebook.select(1)
        lbl._on_tab_changed(None)
        for _ in range(4):
            lbl.root.update()
        assert lbl.active_pane_idx == 1
        assert lbl.selected_interval is want, \
            "the tab change's replot refilled the list and dropped it"
    finally:
        lbl.root.destroy()


# ================================ 3. every user deselect still deselects


def test_clicking_the_selected_row_again_still_deselects(app):
    want = one_human(app)
    row_click(app, want)
    assert app.selected_interval is want

    # the same row clicked a second time: the handler's toggle branch
    app._on_interval_tree_select(None)
    assert app.selected_interval is None
    assert app.status_var.get() == "Interval deselected"

    # and the selection_remove that branch just did fires its OWN empty
    # event on the next turn, which must land harmlessly
    pump(app)
    assert app.selected_interval is None
    assert app.status_var.get() == "Interval deselected"


def test_escape_still_deselects(app):
    want = one_human(app)
    row_click(app, want)
    evt = type("E", (), {"keysym": "Escape", "state": 0, "char": ""})()
    app._on_key_press(evt)
    pump(app)
    assert app.selected_interval is None
    assert app.status_var.get() == "Interval deselected (Escape)"


def test_right_click_still_deselects(app):
    want = one_human(app)
    row_click(app, want)
    evt = type("E", (), {"button": 3, "xdata": None, "ydata": None,
                         "inaxes": None})()
    app._on_right_click_cancel(evt, app.active_pane)
    pump(app)
    assert app.selected_interval is None
    assert app.status_var.get() == "Interval deselected (right-click)"


def test_a_click_on_the_selected_band_still_deselects(app):
    want = one_human(app)
    app.selected_interval = want
    app._update_plot()
    pump(app)
    assert app.selected_interval is want

    pane = app.active_pane
    bands = [c for c in app.strip_ax.collections
             if str(c.get_gid() or "").endswith("strip-bands")]
    assert bands, "the strip drew no band to click"
    # A real pick carries `ind`, and with more than one lane on screen
    # that index is the ONLY thing _strip_click_candidates will resolve
    # through (mixins/lane_controls.py:403). It indexes the array the
    # painter wrote on this pane, so it is built from that array here.
    painted = list(getattr(pane, "_strip_band_ivs", None) or [])
    ind = [i for i, iv in enumerate(painted) if iv is want]
    assert ind, "the painter did not record the band that was clicked"
    mid = want.start + (want.end - want.start) / 2
    mouse = type("ME", (), {"xdata": mdates.date2num(mid), "ydata": 0.5,
                            "inaxes": app.strip_ax})
    pick = type("PE", (), {"artist": bands[0], "mouseevent": mouse,
                           "ind": ind})
    app._on_strip_click(pick, pane)
    pump(app)
    assert app.selected_interval is None
    assert app.status_var.get() == "Interval deselected"


def test_the_guard_only_covers_a_row_the_list_is_not_showing(app):
    """The narrowness, stated as a pin. With the row PRESENT, an empty
    selection still clears -- that is the Ctrl+click path, and it is the
    only way a user can empty this Treeview by hand."""
    want = one_human(app)
    app.selected_interval = want
    app._update_intervals_list()
    pump(app)
    assert any(iv is want for iv in app._interval_row_map.values())

    app.intervals_tree.selection_remove(*app.intervals_tree.selection())
    app._on_interval_tree_select(None)
    assert app.selected_interval is None, \
        "a row the list IS showing must still deselect the old way"
