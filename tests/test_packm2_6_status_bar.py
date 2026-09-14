"""Pack M2.6 -- THE STATUS BAR IS A FULL-WIDTH BAR, AND IT WRAPS.

THE DEFECT. `view_build/sidebar.py` packed the status ttk.Label into the
~322 px SIDEBAR column with `fill=X` and no `wraplength`, and the sidebar
overflows the window bottom. A ttk.Label with no wraplength never wraps
-- a line longer than the widget is simply CUT -- so on screen about 52
characters of a 70-character refusal survived, with the bottom half of
the glyphs sliced off as well. Every lock refusal loses exactly its
reason clause: not one of the 130 screenshots from computer-use session
3 shows "-- drag refused", "-- delete refused" or an overlap hint's
count in full.

THE FIX. The bar is built on the ROOT and packed side=BOTTOM BEFORE the
main pane frame, so the packer gives it its strip of the window first and
no sidebar overflow can push it off screen. Its `wraplength` is refreshed
from its own width on every <Configure>, so a long line wraps onto a
second line instead of being cut. `status_var` and every reader of it are
unchanged.

WHAT THIS MODULE CAN AND CANNOT MEASURE. A withdrawn window has no real
geometry, so these pins measure the STRUCTURE (which widget the label is
a child of, which side it is packed on, that Hide Panel does not touch
it) and the ARITHMETIC (the <Configure> handler, driven with a synthetic
event). What a headless pin cannot show is the picture; the drafter took
exactly one mapped-window capture for that and it is cited in the pack.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tkinter as tk
from tkinter import ttk

DAY = "2015-01-03 "

TWO = [
    {"id": "human", "name": "Human", "classes": ["sw", "UNKNOWN"],
     "class_colors": {"sw": "#4e79a7", "UNKNOWN": "#7f7f7f"}, "order": 0},
    {"id": "model", "name": "Model", "classes": ["0", "1"],
     "class_colors": {"0": "#111111", "1": "#222222"}, "order": 1},
]

LAYOUT = {
    "nrows": 2, "ncols": 1,
    "areas": [
        {"key": "panel1", "row": 0, "col": 0, "role": "time"},
        {"key": "labels", "row": 1, "col": 0, "role": "labels"},
    ],
}

# A real refusal, at the length the live window clipped.
LONG = ("track 'Agent (C-MMAE)' is locked (press Ctrl+L to unlock) -- "
        "clear range refused")


class Ev(object):
    """The one field the <Configure> handler reads."""

    def __init__(self, width):
        self.width = width


def frame():
    idx = pd.date_range(DAY + "00:00:00", periods=120, freq="30s")
    return pd.DataFrame({"a": np.linspace(0.0, 1.0, len(idx))}, index=idx)


def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["a"])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def app(tmp_path):
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(
        df=frame(), plot_fn=plot_fn, layout_spec=dict(LAYOUT),
        window=pd.Timedelta("60min"), autosave_folder=str(tmp_path),
        tracks=[dict(t) for t in TWO])
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    yield lbl
    lbl.root.destroy()


# ======================================================== where it lives


def test_the_status_label_is_not_a_child_of_the_sidebar(app):
    lbl = app.status_label
    assert isinstance(lbl, ttk.Label)
    assert lbl.master is not app.sidebar_frame
    assert str(lbl.master) == str(app.root), \
        "the bar spans the WINDOW, so the window is its master"


def test_it_is_packed_on_the_root_at_the_bottom_filling_x(app):
    info = app.status_label.pack_info()
    assert info["side"] == "bottom"
    assert info["fill"] in ("x", "both")
    assert str(info["in"]) == str(app.root)


def test_it_is_packed_BEFORE_the_main_pane_frame(app):
    """Pack order is the whole point: the packer hands out the window in
    the order children were packed, so the bar must claim its strip
    before the frame that holds the panes and the sidebar."""
    kids = [str(w) for w in app.root.pack_slaves()]
    bar = str(app.status_label)
    assert bar in kids
    main = str(app.sidebar_frame.master)
    assert kids.index(bar) < kids.index(main), kids


def test_the_status_variable_and_its_readers_are_unchanged(app):
    assert isinstance(app.status_var, tk.StringVar)
    assert str(app.status_label.cget("textvariable")) == str(app.status_var)
    app.status_var.set(LONG)
    assert app.status_var.get() == LONG
    assert len(LONG) > 70


def test_hide_panel_does_not_hide_the_status_bar(app):
    app.status_var.set(LONG)
    app._toggle_sidebar()
    assert app.status_label in app.root.pack_slaves()
    assert app.status_var.get() == LONG
    app._toggle_sidebar()
    assert app.status_label in app.root.pack_slaves()


def test_the_error_colour_still_finds_the_bar_where_it_now_lives(app):
    """`canvas.py::_status_widgets` walks from the root, so moving the
    label out of the sidebar must not lose it."""
    app._status_widgets_cache = None
    found = [str(w) for w in app._status_widgets()]
    assert str(app.status_label) in found


# ======================================================== the wrapping


def test_a_configure_gives_the_bar_a_real_wraplength(app):
    app._on_status_label_configure(Ev(1600))
    got = int(app.status_label.cget("wraplength"))
    assert got > 0, "a ttk.Label with wraplength 0 CUTS instead of wrapping"
    assert got == 1600 - app.STATUS_WRAP_GUTTER


def test_the_wraplength_covers_the_window_it_is_given(app):
    """Driven with the root's own requested width, the bar wraps at very
    nearly that width -- the only gutter is the sunken relief's."""
    reqw = int(app.root.winfo_reqwidth())
    app._on_status_label_configure(Ev(reqw))
    got = int(app.status_label.cget("wraplength"))
    assert got >= reqw - app.STATUS_WRAP_GUTTER
    assert got >= max(reqw - 16, app.STATUS_WRAP_MIN)


def test_the_wraplength_follows_a_resize(app):
    app._on_status_label_configure(Ev(1600))
    wide = int(app.status_label.cget("wraplength"))
    app._on_status_label_configure(Ev(900))
    narrow = int(app.status_label.cget("wraplength"))
    assert narrow < wide
    assert narrow == 900 - app.STATUS_WRAP_GUTTER


def test_a_silly_narrow_window_still_wraps_rather_than_stops(app):
    app._on_status_label_configure(Ev(40))
    assert int(app.status_label.cget("wraplength")) == app.STATUS_WRAP_MIN


def test_the_bar_answers_its_own_configure(app):
    """The binding is what keeps the wraplength honest through a resize,
    and it is the only thing that arms it at all in the live window."""
    script = app.status_label.bind("<Configure>")
    assert script, "the bar must answer <Configure>"


def test_the_handler_survives_being_called_with_nothing(app):
    app._on_status_label_configure(None)
    assert int(app.status_label.cget("wraplength")) > 0


def test_the_bar_asks_for_two_lines_when_the_line_is_too_long_for_one(app):
    """The wrap, measured rather than asserted: at a wraplength narrower
    than the text, the label ASKS FOR MORE HEIGHT than it does at a
    wraplength wider than the text. That is the difference between
    wrapping and clipping."""
    app.status_var.set(LONG)
    # Tk fires its OWN <Configure> the moment the label's size changes,
    # and the production binding answers it with the bar's REAL width --
    # which is exactly what the app wants and exactly what a measurement
    # cannot have, because it overwrites the width this pin just set.
    # Measured: without the unbind both heights come back 19, because the
    # narrow wraplength is undone on the very next idle pass. That the
    # binding exists at all is pinned separately, just above.
    app.status_label.unbind("<Configure>")
    app._on_status_label_configure(Ev(1600))
    app.root.update_idletasks()
    tall_at_full_width = app.status_label.winfo_reqheight()
    app._on_status_label_configure(Ev(320))
    app.root.update_idletasks()
    tall_at_sidebar_width = app.status_label.winfo_reqheight()
    assert tall_at_sidebar_width > tall_at_full_width, \
        "at the old sidebar width the same sentence needs more rows"
