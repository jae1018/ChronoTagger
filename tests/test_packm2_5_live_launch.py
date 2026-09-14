"""Pack M2.5 -- WHAT THE FEEL-TEST DRIVER LOOKS LIKE THE MOMENT IT OPENS.

These pins drive `test_drivers/drive_multilabel_thb.py` itself, because
three of the defects the live launch turned up are in the launch sequence
and not in a mixin:

  the Start and End boxes read the START OF THE RECORD (2020-09-01
  00:00:38) while the strip showed 2020-09-02 06:00, because the boxes are
  written by the navigation mixin and the driver set t0/t1 by hand;

  the class dropdown opened on UNKNOWN, whose colour is grey, so the
  ACTIVE lane's bold name and focus ring came up PALER than the two
  inactive names -- the one cue that is supposed to say where you are,
  inverted;

  and the driver's own docstring claims ONE Ctrl+Z removes the whole
  256-interval ingest, which nobody had ever tested.

Plus the double build, at the place it actually happened: the driver
builds the window, ingests, and then calls `run()`.

SKIPPED WITHOUT THE CASE FRAME. The driver reads its data from
`edit_pack/campaign/C05_cmmae/c05_frame.parquet`, which is gitignored
battle-test campaign evidence, so a clean checkout -- CI included -- does
not have it and these pins cannot run there. Everything they pin about the
PRODUCT is pinned again without the frame in
tests/test_packm2_5_one_root.py.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
import tkinter as tk

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVERS = os.path.join(REPO, "test_drivers")
C05_FRAME = os.path.join(REPO, "edit_pack", "campaign", "C05_cmmae",
                         "c05_frame.parquet")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(C05_FRAME),
    reason=("the feel-test driver reads edit_pack/campaign/C05_cmmae/"
            "c05_frame.parquet, which is gitignored campaign evidence and "
            "is absent on a clean checkout (CI included)"))


@pytest.fixture(autouse=True)
def _dialog_counter(monkeypatch):
    """STANDING RULE (Pack 3): every dialog stubbed, and counted."""
    import tkinter.messagebox as mb
    import tkinter.filedialog as fd
    import tkinter.simpledialog as sd
    calls = []
    for kind in ("showinfo", "showwarning", "showerror", "askyesno",
                 "askyesnocancel", "askokcancel"):
        monkeypatch.setattr(
            mb, kind, lambda *a, _k=kind, **kw: calls.append(_k) or True)
    for kind in ("asksaveasfilename", "askopenfilename", "askdirectory"):
        monkeypatch.setattr(fd, kind, lambda *a, **kw: calls.append(kind) or "")
    for kind in ("askstring", "askinteger", "askfloat"):
        monkeypatch.setattr(sd, kind, lambda *a, **kw: calls.append(kind))
    return calls


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def driver(monkeypatch, tmp_path):
    """The driver module, writing its autosave into tmp_path."""
    if DRIVERS not in sys.path:
        sys.path.insert(0, DRIVERS)
    import drive_multilabel_thb as D
    monkeypatch.setattr(D, "CASE_DIR", str(tmp_path))
    monkeypatch.setattr(D, "AUTOSAVE_FOLDER",
                        os.path.join(str(tmp_path), "chronotagger_autosave"))
    return D


class Key(object):
    def __init__(self, keysym, state=0):
        self.keysym = keysym
        self.state = state


def per_lane(app):
    out = {}
    for iv in app.intervals:
        out[iv.track] = out.get(iv.track, 0) + 1
    return out


def test_the_time_boxes_open_on_the_window_the_strip_is_showing(driver):
    app = driver.build(run=False)
    try:
        assert app.t0 == driver.OPEN_AT
        assert app.t1 == driver.OPEN_AT + driver.WINDOW
        assert app.start_time_entry.get() == str(app.t0)
        assert app.end_time_entry.get() == str(app.t1)
        assert app.start_time_entry.get().startswith("2020-09-02 06:00:00")
        assert app.status_var.get().startswith("Window: 06:00:00")
    finally:
        app.root.destroy()


def test_the_opening_figure_is_painted_at_the_opening_window(driver):
    """The flush the driver asks for after _sync_entries_and_plot.

    _sync_entries_and_plot requests a COALESCED redraw, which a real Tk root
    defers to its idle queue -- and run=False never starts a mainloop to
    service it, so without the flush every headless consumer of this driver
    would measure the record start instead of 2020-09-02 06:00.
    """
    app = driver.build(run=False)
    try:
        assert [p._strip_lane_count for p in app.panes] == [3, 3, 3]
        for i, pane in enumerate(app.panes):
            assert len(pane._strip_band_ivs) == 33, (i,
                                                     len(pane._strip_band_ivs))
    finally:
        app.root.destroy()


def test_the_class_dropdown_opens_on_a_class_that_has_a_colour(driver):
    app = driver.build(run=False)
    try:
        assert app.current_class_var.get() == "solar_wind"
        # the lane's own vocabulary is UNCHANGED -- UNKNOWN is still there
        assert list(app.class_combo["values"]) == driver.REGION_CLASSES
        assert app.classes[0] == "UNKNOWN"
        colour = app.track_by_id("region").class_colors["solar_wind"]
        assert colour == "#4e79a7"
    finally:
        app.root.destroy()


def test_one_ctrl_z_removes_the_whole_ingest(driver):
    """The driver's docstring has claimed this since Pack M2 and nobody had
    measured it: the 256-interval import is ONE undoable command."""
    app = driver.build(run=False)
    try:
        assert per_lane(app) == {"agent": 256}
        app._on_key_press(Key("z", 0x4))
        assert app.intervals == []
        assert per_lane(app) == {}
    finally:
        app.root.destroy()


def test_the_real_launch_path_builds_the_window_exactly_once(driver,
                                                             monkeypatch):
    """build() builds the GUI and then calls run(), which used to build it
    again -- two tk.Tk() interpreters, nine variables in the hidden one."""
    from chronotagger.labeler import TimeIntervalLabeler
    from chronotagger.labeler.mixins.view_build import window as window_mod
    made = []
    real = window_mod._new_tk_root
    monkeypatch.setattr(window_mod, "_new_tk_root",
                        lambda *a, **k: made.append(1) or real(*a, **k))
    monkeypatch.setattr(TimeIntervalLabeler, "_check_autosave",
                        lambda self: None)
    monkeypatch.setattr(tk.Tk, "mainloop", lambda self, *a, **k: None)
    monkeypatch.setattr(tk, "_default_root", None, raising=False)
    app = driver.main(run=True)
    try:
        assert made == [1], "the launch path built %d Tk roots" % len(made)
        assert tk._default_root is app.root
        wrong = sorted(n for n, v in vars(app).items()
                       if isinstance(v, tk.Variable)
                       and v._tk is not app.root.tk)
        assert wrong == [], wrong
        assert app.lane_var.get() == "Region (human)"
        assert [p._strip_lane_count for p in app.panes] == [3, 3, 3]
    finally:
        app.root.destroy()
