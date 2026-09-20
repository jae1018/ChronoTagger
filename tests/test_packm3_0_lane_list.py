"""Pack M3.0, items E and F -- THE LANE LIST, AND THE TITLE THAT NAMES
ITS LANE.

E. The sidebar's Lane list turned its own entry back into a lane by
   DISPLAY NAME (`_track_id_for_choice`). Display names are free text and
   nothing enforces uniqueness, so two lanes called "Dup" were one lane
   to the list: measured at base, picking the SECOND row activated the
   FIRST (probe_m30.py E.2). It now asks the widget which ROW was picked
   and resolves that against table order. The text lookup stays as the
   fallback for a caller that sets `lane_var` by hand.

F. Manage Labels edits ONE lane's vocabulary and its title named none.
   With more than one lane the title now reads
   `Manage Labels -- <lane>`; with exactly one it is byte-for-byte what
   it is today, and the wizard's call site is not touched.
"""

import inspect
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tkinter as tk

DAY = "2015-01-03 "

ONE = {"id": "one", "name": "Dup", "classes": ["x", "UNKNOWN"], "order": 0}
TWO = {"id": "two", "name": "Dup", "classes": ["y", "UNKNOWN"], "order": 1}
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
    return pd.DataFrame({"a": np.linspace(0.0, 1.0, len(idx))}, index=idx)


def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["a"])
    axs["panel2"].plot(df.index, df["a"])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def _no_modal_blocking(monkeypatch):
    """wait_visibility and grab_set HANG against a withdrawn parent."""
    monkeypatch.setattr(tk.Misc, "wait_visibility", lambda *a, **k: None)
    monkeypatch.setattr(tk.Misc, "grab_set", lambda *a, **k: None)


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
def twins(tmp_path):
    lbl = build([ONE, TWO], tmp_path)
    yield lbl
    lbl.root.destroy()


@pytest.fixture
def pair(tmp_path):
    lbl = build([REGION, WAKE], tmp_path)
    yield lbl
    lbl.root.destroy()


# ================================ E. the list resolves by position


def test_two_lanes_share_one_name_and_the_second_row_is_the_second_lane(
        twins):
    assert twins._lane_choices() == ["Dup", "Dup"]
    assert twins.active_track_id == "one"
    twins.lane_combo.current(1)
    twins.root.update()
    twins._on_lane_combo_change()
    assert twins.active_track_id == "two", \
        "the SECOND row must mean the SECOND lane"


def test_and_the_first_row_still_means_the_first_lane(twins):
    twins._set_active_track("two", announce=False, repaint=False)
    twins.lane_combo.current(0)
    twins.root.update()
    twins._on_lane_combo_change()
    assert twins.active_track_id == "one"


def test_the_index_resolver_is_table_order_and_refuses_the_rest(twins):
    assert twins._track_id_for_index(0) == "one"
    assert twins._track_id_for_index(1) == "two"
    assert twins._track_id_for_index(2) is None
    assert twins._track_id_for_index(-1) is None, \
        "a combobox whose text matches no value reports -1"
    assert twins._track_id_for_index(None) is None
    assert twins._track_id_for_index("nonsense") is None


def test_the_text_fallback_survives_for_callers_that_set_the_var(pair):
    """A head-less caller that only writes `lane_var` still works."""
    pair.lane_var.set("Wake (umbra)")
    pair._on_lane_combo_change()
    assert pair.active_track_id == "wake"
    assert pair._track_id_for_choice("Wake (umbra)") == "wake"
    assert pair._track_id_for_choice("Wake (umbra) (hidden)") == "wake"
    assert pair._track_id_for_choice("no such lane") is None


def test_picking_a_hidden_row_still_unhides_it(pair):
    pair._toggle_active_lane_visible()          # region hidden
    assert pair._lane_choices()[0] == "Region (human) (hidden)"
    n = len(pair.undo_stack)
    pair.lane_combo.current(0)
    pair.root.update()
    pair._on_lane_combo_change()
    assert pair.tracks[0].visible is True
    assert pair.active_track_id == "region"
    assert len(pair.undo_stack) == n + 1, "item C: and it is one undo step"


def test_refresh_keeps_the_selected_index_on_the_active_lane(twins):
    twins._set_active_track("two", announce=False, repaint=False)
    twins._refresh_lane_controls()
    assert twins.lane_combo.current() == 1
    twins._set_active_track("one", announce=False, repaint=False)
    twins._refresh_lane_controls()
    assert twins.lane_combo.current() == 0


# ================================ F. the title names its lane


class _Spy:
    """Stands in for LabelManagerDialog: records, never opens a window."""
    seen = []

    def __init__(self, **kw):
        _Spy.seen.append(kw)
        self.result = None


def _spy_manager(monkeypatch):
    import chronotagger.labeler.mixins.labels as labels_mod
    _Spy.seen = []
    monkeypatch.setattr(labels_mod, "LabelManagerDialog", _Spy)
    monkeypatch.setattr(tk.Misc, "wait_window", lambda *a, **k: None)
    return _Spy.seen


def test_with_two_lanes_the_title_names_the_active_lane(pair, monkeypatch):
    seen = _spy_manager(monkeypatch)
    pair._set_active_track("wake", announce=False, repaint=False)
    pair._open_label_manager()
    assert len(seen) == 1
    assert seen[0]["title"] == "Manage Labels -- Wake (umbra)"


def test_with_one_lane_the_title_is_exactly_manage_labels(tmp_path,
                                                          monkeypatch):
    lbl = build([REGION], tmp_path)
    try:
        seen = _spy_manager(monkeypatch)
        lbl._open_label_manager()
        assert len(seen) == 1
        assert seen[0]["title"] == "Manage Labels"
    finally:
        lbl.root.destroy()


def test_the_dialog_default_title_is_unchanged(pair):
    from chronotagger.labeler.dialogs.label_manager import (
        LabelManagerDialog)
    sig = inspect.signature(LabelManagerDialog.__init__)
    assert sig.parameters["title"].default == "Manage Labels"
    dlg = LabelManagerDialog(parent=pair.root, classes=["a"],
                             class_colors={"a": "#111111"},
                             usage_counts={"a": 0})
    try:
        assert dlg.title() == "Manage Labels"
    finally:
        dlg.destroy()
    dlg = LabelManagerDialog(parent=pair.root, classes=["a"],
                             class_colors={"a": "#111111"},
                             usage_counts={"a": 0},
                             title="Manage Labels -- Wake (umbra)")
    try:
        assert dlg.title() == "Manage Labels -- Wake (umbra)"
    finally:
        dlg.destroy()


def test_the_wizard_call_site_passes_no_title(pair):
    """The wizard's copy stays exactly what the golden tests pin."""
    import chronotagger.quickstart.wizard as wiz
    src = open(os.path.abspath(wiz.__file__), "r",
               encoding="utf-8").read()
    i = src.index("LabelManagerDialog(")
    call = src[i:src.index(")", src.index("reserved=", i))]
    assert "title" not in call, call
