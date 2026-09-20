"""Pack M2.8 -- THE TWO OPTION BOXES ACT WHEN CLICKED.

THE DEFECT, from the session-4 backlog and seen again in session 5. In
`mixins/view_build/sidebar.py` the Checkbuttons 'Snap to samples' and
'Show interval overlays on panels' were built with NO `command`. The
variable flipped and nothing else happened: unticking the overlays box
left the bands on every panel until something else repainted -- session
5 had to press n and then p -- and the snap box gave no sign at all that
it had taken. 'Highlight Points', built three lines below them, has had
`command=_on_highlight_points_toggle` since Pack 8.5B.

THE FIX gives each box a command written the way that one is written.
The overlays box repaints and says `shown` / `hidden`; the snap box says
`on` / `off` and repaints nothing, because nothing on the draw path
reads snap_var.

THE PINS DRIVE THE WIDGET, not the method: `invoke()` on the real
Checkbutton is what a click does, and it is the only thing that can tell
a wired box from an unwired one.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tkinter as tk
from tkinter import ttk

from chronotagger.core.models import Interval

DAY = "2015-01-03 "

BAND_GID = "chronotagger:interval-bands"

LAYOUT = {
    "nrows": 2, "ncols": 1,
    "areas": [
        {"key": "panel1", "row": 0, "col": 0, "role": "time"},
        {"key": "labels", "row": 1, "col": 0, "role": "labels"},
    ],
}


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
    for kind in ("showinfo", "showwarning", "showerror", "askyesno",
                 "askyesnocancel", "askokcancel"):
        monkeypatch.setattr(mb, kind, lambda *a, **kw: True)


@pytest.fixture
def app(tmp_path):
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(
        df=frame(), plot_fn=plot_fn, layout_spec=dict(LAYOUT),
        window=pd.Timedelta("60min"), autosave_folder=str(tmp_path))
    lbl._build_gui()
    lbl.intervals[:] = [
        Interval(ts("00:05:00"), ts("00:08:00"), "UNKNOWN", None)]
    lbl._update_plot()
    lbl.root.withdraw()
    yield lbl
    lbl.root.destroy()


def box(app, text):
    """The real Checkbutton the user clicks, found by the text he reads."""
    found = []

    def walk(w):
        for child in w.winfo_children():
            if isinstance(child, (ttk.Checkbutton, tk.Checkbutton)):
                try:
                    if str(child.cget("text")) == text:
                        found.append(child)
                except Exception:
                    pass
            walk(child)

    walk(app.root)
    assert len(found) == 1, "%r matched %d check boxes" % (text, len(found))
    return found[0]


def bands(app):
    out = []
    for ax in app.user_axes.values():
        out += [c for c in ax.collections if c.get_gid() == BAND_GID]
    return out


# ==================================== 1. both boxes are wired at all


def test_both_boxes_carry_a_command(app):
    assert str(box(app, "Snap to samples").cget("command"))
    assert str(box(app, "Show interval overlays on panels").cget("command"))


# ==================================== 2. snap: a status line, no repaint


def test_the_snap_box_says_on_and_off(app):
    drawn = []
    app._request_redraw = lambda: drawn.append(1)
    b = box(app, "Snap to samples")

    assert app.snap_var.get() is False
    b.invoke()
    assert app.snap_var.get() is True
    assert app.status_var.get() == "Snap to samples: on"

    b.invoke()
    assert app.snap_var.get() is False
    assert app.status_var.get() == "Snap to samples: off"

    assert drawn == [], "nothing on screen depends on snap until the next " \
        "selection, so the box must not force a repaint"


# ==================================== 3. overlays: a repaint AND a line


def test_the_overlays_box_requests_a_repaint_and_says_hidden(app):
    drawn = []
    app._request_redraw = lambda: drawn.append(1)
    b = box(app, "Show interval overlays on panels")

    assert app.overlays_var.get() is True
    b.invoke()
    assert app.overlays_var.get() is False
    assert drawn == [1], "the overlays box did not ask for a repaint"
    assert app.status_var.get() == "Interval overlays on panels: hidden"

    b.invoke()
    assert app.overlays_var.get() is True
    assert drawn == [1, 1]
    assert app.status_var.get() == "Interval overlays on panels: shown"


def test_the_bands_really_leave_the_panels_and_come_back(app):
    """The consequence on screen, and NOTHING calls _update_plot here.

    The repaint has to come from the box itself, so the loop is pumped
    instead: _request_redraw coalesces onto Tk's idle queue (Pack 5 R4d)
    and root.update() is what runs it. On the unwired box this pin is
    red -- the bands simply stay, which is what session 5 saw.
    """
    assert bands(app), "the fixture drew no interval bands to hide"

    b = box(app, "Show interval overlays on panels")
    b.invoke()
    for _ in range(4):
        app.root.update()
    assert bands(app) == [], "the bands survived 'hidden'"

    b.invoke()
    for _ in range(4):
        app.root.update()
    assert bands(app), "the bands did not come back"
