"""Pack M3.0, item A -- CLICK A LANE TO MAKE IT ACTIVE.

With more than one lane painted, a plain left click on a lane's NAME, or
on ANY EMPTY PART of its row, makes that lane the active lane. At base
both did nothing at all: the empty-row press found no interval to select
and returned, and the names are y TICK LABELS drawn OUTSIDE the axes, so
`event.inaxes` is None over them and the press handler declined on its
first test (probe_m30.py A.1, A.6).

A LOCKED lane's name or empty row DOES switch to it -- this is only a
lane switch -- and the bar says `lane locked (Ctrl+L to unlock)` so the
user knows why he cannot draw there and what to press.

UNCHANGED, and pinned elsewhere: a click on a BAND of an unlocked lane
selects AND activates; a band on a LOCKED lane selects WITHOUT activating
(tests/test_packm2_lane_scope_lock.py). The gutters, above the top band
and below the bottom one still do nothing. With one lane there are no
names and nothing to switch to.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from chronotagger.core.models import Interval

DAY = "2015-01-03 "

REGION = {"id": "region", "name": "Region (human)",
          "classes": ["sw", "msh", "UNKNOWN"], "order": 0}
WAKE = {"id": "wake", "name": "Wake (umbra)",
        "classes": ["umbra", "UNKNOWN"], "order": 1}
AGENT = {"id": "agent", "name": "Agent (C-MMAE)", "classes": ["0", "1"],
         "locked": True, "order": 2}

LAYOUT = {
    "nrows": 3, "ncols": 1,
    "areas": [
        {"key": "panel1", "row": 0, "col": 0, "role": "time"},
        {"key": "panel2", "row": 1, "col": 0, "role": "time"},
        {"key": "labels", "row": 2, "col": 0, "role": "labels"},
    ],
}


def ts(hhmmss):
    return pd.Timestamp(DAY + hhmmss)


def frame():
    idx = pd.date_range(DAY + "00:00:00", periods=120, freq="30s")
    return pd.DataFrame({"a": np.linspace(0.0, 1.0, len(idx))}, index=idx)


def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["a"])
    axs["panel2"].plot(df.index, df["a"])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def build(tracks, tmp_path):
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(
        df=frame(), plot_fn=plot_fn, layout_spec=dict(LAYOUT),
        window=pd.Timedelta("60min"), autosave_folder=str(tmp_path),
        tracks=[dict(t) for t in tracks])
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    return lbl


@pytest.fixture
def app(tmp_path):
    lbl = build([REGION, WAKE, AGENT], tmp_path)
    yield lbl
    lbl.root.destroy()


def band(app, tid):
    from chronotagger.core.lanes import lane_band, pad_for
    pane = app.active_pane
    k = pane._strip_lane_count
    r = list(pane._strip_lane_ids).index(tid)
    return lane_band(r, k, pad_for(k))


def press(app, xfrac, yfrac):
    """The event a plain left click INSIDE the strip axes produces."""
    from matplotlib.backend_bases import MouseEvent
    ax = app.active_pane.strip_ax
    bb = ax.get_window_extent()
    ev = MouseEvent("button_press_event", ax.figure.canvas,
                    int(round(bb.x0 + xfrac * bb.width)),
                    int(round(bb.y0 + yfrac * bb.height)), button=1)
    ev.inaxes = ax
    return ev


def name_press(app, text):
    """A click at the centre of a lane NAME, which is outside the axes."""
    from matplotlib.backend_bases import MouseEvent
    ax = app.active_pane.strip_ax
    app.active_pane.canvas.draw()
    lab = [t for t in ax.get_yticklabels() if t.get_text() == text][0]
    e = lab.get_window_extent()
    ev = MouseEvent("button_press_event", ax.figure.canvas,
                    int(round(0.5 * (e.x0 + e.x1))),
                    int(round(0.5 * (e.y0 + e.y1))), button=1)
    ev.inaxes = None
    return ev


def empty_click(app, tid, xfrac=0.80):
    lo, hi = band(app, tid)
    app._on_strip_press(press(app, xfrac, 0.5 * (lo + hi)),
                        app.active_pane)


# ====================================== 1. the empty part of a row


def test_an_empty_row_click_makes_that_lane_active(app):
    assert app.active_track_id == "region"
    empty_click(app, "wake")
    assert app.active_track_id == "wake"
    assert app.status_var.get() == "Active lane: Wake (umbra)"
    assert list(app.class_combo["values"]) == ["umbra", "UNKNOWN"], \
        "the class dropdown follows the active lane"


def test_a_locked_lanes_empty_row_switches_and_says_how_to_unlock(app):
    empty_click(app, "agent")
    assert app.active_track_id == "agent"
    assert app.status_var.get() == (
        "Active lane: Agent (C-MMAE) -- lane locked (Ctrl+L to unlock)")


def test_the_already_active_lane_does_nothing_and_says_nothing(app):
    app.status_var.set("SENTINEL")
    empty_click(app, "region")
    assert app.active_track_id == "region"
    assert app.status_var.get() == "SENTINEL"


def test_the_gutter_and_the_edges_still_do_nothing(app):
    lo0, hi0 = band(app, "region")
    lo1, hi1 = band(app, "wake")
    for yf in (0.5 * (lo0 + hi1), 0.999, 0.001):
        app.status_var.set("SENTINEL")
        app._on_strip_press(press(app, 0.80, yf), app.active_pane)
        assert app.active_track_id == "region"
        assert app.status_var.get() == "SENTINEL"


def test_a_click_over_a_band_is_not_an_empty_row_click(app):
    """An interval under the cursor is a selection, never this."""
    app.intervals[:] = [Interval(start=ts("00:10:00"), end=ts("00:20:00"),
                                 label="umbra", track="wake")]
    app._update_plot()
    app.selected_interval = app.intervals[0]
    lo, hi = band(app, "wake")
    xf = float((ts("00:15:00") - app.t0) / (app.t1 - app.t0))
    app.status_var.set("SENTINEL")
    app._on_strip_press(press(app, xf, 0.5 * (lo + hi)), app.active_pane)
    assert app.active_track_id == "region", \
        "the band path decides this click, not the empty-row path"


def test_one_lane_has_nothing_to_switch_to(tmp_path):
    lbl = build([REGION], tmp_path)
    try:
        assert lbl.active_pane._strip_lane_count == 1
        assert lbl.active_pane.strip_ax.get_yticklabels() == []
        lbl.status_var.set("SENTINEL")
        lbl._on_strip_press(press(lbl, 0.8, 0.5), lbl.active_pane)
        assert lbl.active_track_id == "region"
        assert lbl.status_var.get() == "SENTINEL"
    finally:
        lbl.root.destroy()


# ====================================== 2. the lane's NAME


def test_a_click_on_a_lane_name_makes_that_lane_active(app):
    app._on_strip_press(name_press(app, "Wake (umbra)"), app.active_pane)
    assert app.active_track_id == "wake"
    assert app.status_var.get() == "Active lane: Wake (umbra)"


def test_a_click_on_a_locked_lanes_name_switches_and_says_so(app):
    app._on_strip_press(name_press(app, "Agent (C-MMAE)"),
                        app.active_pane)
    assert app.active_track_id == "agent"
    assert "lane locked (Ctrl+L to unlock)" in app.status_var.get()


def test_a_click_beside_a_name_does_nothing(app):
    from matplotlib.backend_bases import MouseEvent
    ax = app.active_pane.strip_ax
    app.active_pane.canvas.draw()
    lab = [t for t in ax.get_yticklabels()
           if t.get_text() == "Wake (umbra)"][0]
    e = lab.get_window_extent()
    ev = MouseEvent("button_press_event", ax.figure.canvas,
                    int(round(0.5 * (e.x0 + e.x1))),
                    int(round(e.y1 + 40)), button=1)
    ev.inaxes = None
    app.status_var.set("SENTINEL")
    app._on_strip_press(ev, app.active_pane)
    assert app.active_track_id == "region"
    assert app.status_var.get() == "SENTINEL"


def test_the_name_hit_test_declines_inside_the_axes(app):
    """Only a click that belongs to no axes can be a name."""
    app.active_pane.canvas.draw()
    ev = press(app, 0.5, 0.5)
    assert app._strip_name_hit(ev, app.active_pane) is None


def test_the_name_hit_test_answers_each_lane(app):
    app.active_pane.canvas.draw()
    for tid, text in (("region", "Region (human)"),
                      ("wake", "Wake (umbra)"),
                      ("agent", "Agent (C-MMAE)")):
        ev = name_press(app, text)
        assert app._strip_name_hit(ev, app.active_pane) == tid


# ====================================== 3. a live selection


def test_a_press_inside_the_selected_interval_still_arms_the_drag(app):
    app.intervals[:] = [Interval(start=ts("00:10:00"), end=ts("00:20:00"),
                                 label="sw", track="region")]
    app._update_plot()
    app.selected_interval = app.intervals[0]
    lo, hi = band(app, "region")
    xf = float((ts("00:15:00") - app.t0) / (app.t1 - app.t0))
    app._on_strip_press(press(app, xf, 0.5 * (lo + hi)), app.active_pane)
    assert app._drag_mode == "move"
    assert app.active_track_id == "region"
    app._drag_mode = None


def test_an_empty_row_click_elsewhere_keeps_the_selection(app):
    app.intervals[:] = [Interval(start=ts("00:10:00"), end=ts("00:20:00"),
                                 label="sw", track="region")]
    app._update_plot()
    want = app.intervals[0]
    app.selected_interval = want
    empty_click(app, "wake", xfrac=0.90)
    assert app.active_track_id == "wake"
    assert app.selected_interval is want, \
        "the very same object is still selected"
    assert app._drag_mode is None


# ====================================== 4. a staged rule preview


def test_a_switch_clears_a_staged_rule_and_says_so(app):
    app._commit_spans = [(ts("00:02:00"), ts("00:05:00"))]
    empty_click(app, "wake")
    assert app.active_track_id == "wake"
    assert list(app._commit_spans) == []
    assert app.status_var.get() == (
        "Active lane: Wake (umbra) -- rule preview cleared (it belonged "
        "to the lane you left)")
