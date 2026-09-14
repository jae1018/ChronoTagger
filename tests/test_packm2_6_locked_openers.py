"""Pack M2.6 -- A LOCKED LANE REFUSES AT THE DOOR, NOT AFTER THE WORK.

Pack M2 put the lock guard at the COMMIT of nine paths, and that is
correct: a programmatic caller gets the same refusal a pair of hands
does. It is not enough. Three of those nine are reached through an
OPENER that had no guard of its own, so on a locked lane the app
presented a live editor and then threw the work away:

  Clear...        the whole mode picker opens (Current range / Custom /
                  Entire dataset), Next is answered, and one status line
                  refuses.
  Fill Gaps...    with a real coverage hole the full "Label Unassigned
                  Points" dialog opens, lists the classes and promises
                  "Will create 1 interval(s)" in blue. Measured: the
                  locked agent lane of the feel-test driver has 19 holes
                  across the record.
  Manage Labels   opens with Add / Rename / Change color / Move / Delete
                  ALL ENABLED, and OK discards every one of them.

This module pins the door. The commit guards are Pack M2's and are
pinned in tests/test_packm2_lane_scope_lock.py, unchanged.

HOW "NO DIALOG" IS MEASURED, and why it is not just a Toplevel spy.
`LabelManagerDialog` SUBCLASSES `tk.Toplevel` and binds that base class
at class-definition time, so a `tkinter.Toplevel` patched later never
sees it -- measured, in probe_s3_locked_dialogs' own spy control
("counted_by_spy=False"). So the primary measurement here is a CENSUS OF
THE ROOT'S CHILDREN, which catches a subclass exactly as well as a
direct call, and the `tkinter.Toplevel` spy is kept alongside it for the
two openers that really do call `tk.Toplevel(self.root)` directly.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tkinter as tk

from chronotagger.core.models import Interval

DAY = "2015-01-03 "

HUMAN = {"id": "human", "name": "Human", "classes": ["sw", "msh", "UNKNOWN"],
         "class_colors": {"sw": "#4e79a7", "msh": "#f28e2b",
                          "UNKNOWN": "#7f7f7f"}, "order": 0}
AGENT = {"id": "agent", "name": "Agent (C-MMAE)", "classes": ["0", "1"],
         "class_colors": {"0": "#111111", "1": "#222222"},
         "locked": True, "order": 1}

LAYOUT = {
    "nrows": 2, "ncols": 1,
    "areas": [
        {"key": "panel1", "row": 0, "col": 0, "role": "time"},
        {"key": "labels", "row": 1, "col": 0, "role": "labels"},
    ],
}

OPENERS = (
    ("_open_clear_intervals_dialog", "clear range refused"),
    ("_open_label_unassigned_dialog", "fill gaps refused"),
    ("_open_label_manager", "edit the label schema refused"),
)


def ts(hhmmss):
    return pd.Timestamp(DAY + hhmmss)


def frame():
    idx = pd.date_range(DAY + "00:00:00", periods=120, freq="30s")
    return pd.DataFrame({"a": np.linspace(0.0, 1.0, len(idx))}, index=idx)


def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["a"])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def _no_modal_machinery(monkeypatch):
    """A modal built headless must not grab or wait: both HANG against a
    withdrawn parent. Patched for EVERY test here, so the unlocked half
    can let the dialogs really open."""
    monkeypatch.setattr(tk.Misc, "grab_set", lambda self, *a, **k: None)
    monkeypatch.setattr(tk.Misc, "wait_visibility",
                        lambda self, *a, **k: None)
    monkeypatch.setattr(tk.Misc, "wait_window", lambda self, *a, **k: None)


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    """Every messagebox / filedialog entry point, COUNTED."""
    import tkinter.filedialog as fd
    import tkinter.messagebox as mb
    calls = []
    for kind in ("showinfo", "showwarning", "showerror", "askyesno",
                 "askyesnocancel", "askokcancel"):
        monkeypatch.setattr(
            mb, kind, lambda *a, _k=kind, **kw: calls.append(_k) or True)
    for kind in ("asksaveasfilename", "askopenfilename", "askdirectory"):
        monkeypatch.setattr(fd, kind, lambda *a, **kw: calls.append(kind) or "")
    return calls


@pytest.fixture
def app(tmp_path):
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(
        df=frame(), plot_fn=plot_fn, layout_spec=dict(LAYOUT),
        window=pd.Timedelta("60min"), autosave_folder=str(tmp_path),
        tracks=[dict(HUMAN), dict(AGENT)])
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    yield lbl
    lbl.root.destroy()


def tops(app):
    """Every Toplevel standing under the root, subclasses included."""
    return [w for w in app.root.winfo_children() if isinstance(w, tk.Toplevel)]


def close_tops(app):
    for w in tops(app):
        w.destroy()


def seed(app):
    """Something for Clear... to find, and a hole for Fill Gaps... to see."""
    app.intervals[:] = [
        Interval(ts("00:05:00"), ts("00:08:00"), "sw", None, track="human"),
        Interval(ts("00:20:00"), ts("00:25:00"), "0", None, track="agent"),
    ]


# ================================================== the locked lane refuses


@pytest.mark.parametrize("opener,refusal", OPENERS)
def test_the_opener_refuses_on_a_locked_lane(app, _dialogs, monkeypatch,
                                             opener, refusal):
    seed(app)
    app._set_active_track("agent", announce=False, repaint=False)
    built = []
    _real = tk.Toplevel

    class Spy(_real):
        def __init__(self, *a, **k):
            built.append(1)
            _real.__init__(self, *a, **k)

    monkeypatch.setattr(tk, "Toplevel", Spy)
    app.status_var.set("")
    getattr(app, opener)()
    assert tops(app) == [], "no window may appear on a locked lane"
    assert built == []
    assert _dialogs == [], "and no messagebox either"
    assert refusal in app.status_var.get()
    assert "Agent (C-MMAE)" in app.status_var.get()


def test_the_locked_lane_is_byte_identical_after_all_three_openers(app,
                                                                   _dialogs):
    seed(app)
    before = sorted((str(iv.start), str(iv.end), iv.label, iv.track)
                    for iv in app.intervals)
    classes = {t.id: list(t.classes) for t in app.tracks}
    app._set_active_track("agent", announce=False, repaint=False)
    for opener, _ in OPENERS:
        getattr(app, opener)()
    assert sorted((str(iv.start), str(iv.end), iv.label, iv.track)
                  for iv in app.intervals) == before
    assert {t.id: list(t.classes) for t in app.tracks} == classes
    assert _dialogs == []


# ================================================ the unlocked lane opens


@pytest.mark.parametrize("opener,_refusal", OPENERS)
def test_the_opener_still_opens_on_an_unlocked_lane(app, opener, _refusal):
    seed(app)
    assert app.active_track_id == "human"
    assert app.track_by_id("human").locked is False
    getattr(app, opener)()
    assert len(tops(app)) == 1, "the editor must still be reachable"
    close_tops(app)


def test_fill_gaps_really_has_a_hole_to_offer_on_the_unlocked_lane(app):
    seed(app)
    gaps = app._find_gaps_in_current_range()
    assert gaps, "the human lane leaves holes in this window"
    assert len(gaps) >= 2


def test_unlocking_the_lane_reopens_every_door(app, _dialogs):
    seed(app)
    app._set_active_track("agent", announce=False, repaint=False)
    for opener, _ in OPENERS:
        getattr(app, opener)()
    assert tops(app) == []
    app.track_by_id("agent").locked = False
    for opener, _ in OPENERS:
        getattr(app, opener)()
        assert len(tops(app)) == 1, opener
        close_tops(app)
