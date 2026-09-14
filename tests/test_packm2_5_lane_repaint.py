"""Pack M2.5 -- PICKING THE HIDDEN ACTIVE LANE REPAINTS THE STRIP.

Pack M2's Lane list is the only way back to a hidden lane: picking it
un-hides AND activates it. That works whenever the hidden lane is not the
active one. In the state Pack M2's SECTION 0e item 10 describes -- an Undo
restores a lane table in which the ACTIVE lane is hidden -- picking it set
`visible = True` and said "Active lane: ...", and the strip kept painting
K-1 lanes until something unrelated redrew, because the one setter repaints
only when the active id actually CHANGED and here it did not. Measured by
the post-implementation interaction refuter: `visible` True while
`[pane._strip_lane_count for pane in app.panes]` stayed [2, 2, 2].

The fix repaints when the VISIBILITY flag changed as well, and only then:
picking the lane that is already active and already visible still asks for
no repaint at all, which is what the second pin holds.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

DAY = "2015-01-03 "

TWO = [
    {"id": "human", "name": "Human", "classes": ["UNKNOWN", "sw"],
     "class_colors": {"UNKNOWN": "#7f7f7f", "sw": "#4e79a7"}, "order": 0},
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


@pytest.fixture(autouse=True)
def _dialog_counter(monkeypatch):
    """STANDING RULE (Pack 3): every dialog stubbed, and counted."""
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


def frame():
    idx = pd.date_range(DAY + "00:00:00", periods=60, freq="30s")
    return pd.DataFrame({"a": np.linspace(0, 1, len(idx))}, index=idx)


def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["a"])


@pytest.fixture
def two(tmp_path):
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(
        df=frame(), plot_fn=plot_fn, layout_spec=dict(LAYOUT),
        window=pd.Timedelta("15min"), autosave_folder=str(tmp_path),
        tracks=[dict(t) for t in TWO])
    lbl._build_gui()
    lbl.root.withdraw()
    lbl._update_plot()
    yield lbl
    lbl.root.destroy()


def lane_counts(app):
    return [p._strip_lane_count for p in app.panes]


def pick(app, text):
    app.lane_var.set(text)
    app._on_lane_combo_change()


def test_picking_the_hidden_ACTIVE_lane_brings_it_back_on_screen(
        two, _dialog_counter):
    """The undo edge: the ACTIVE lane is the hidden one."""
    assert two.active_track_id == "human"
    two.track_by_id("human").visible = False
    two._update_plot()
    assert lane_counts(two) == [1]
    assert two._lane_choices() == ["Human (hidden)", "Model"]

    pick(two, "Human (hidden)")

    assert two.track_by_id("human").visible is True
    assert two.active_track_id == "human"
    assert lane_counts(two) == [2], "the strip must not wait for the next " \
                                    "unrelated repaint"
    assert _dialog_counter == []


def test_no_repaint_is_asked_for_when_nothing_about_the_lane_changed(
        two, monkeypatch):
    calls = []
    real = two._update_plot
    monkeypatch.setattr(two, "_update_plot",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    # the lane that is already active and already visible
    pick(two, "Human")
    assert calls == []
    # a different lane: the setter's own single repaint
    pick(two, "Model")
    assert len(calls) == 1
    assert two.active_track_id == "model"
    # the hidden ACTIVE lane: exactly one repaint, this pack's
    del calls[:]
    two.track_by_id("model").visible = False
    pick(two, "Model (hidden)")
    assert len(calls) == 1, calls
    assert lane_counts(two) == [2]


def test_picking_a_hidden_NON_active_lane_still_works(two, _dialog_counter):
    """Unchanged by this pack, pinned so the fix cannot break it."""
    two.track_by_id("model").visible = False
    two._update_plot()
    assert lane_counts(two) == [1]
    pick(two, "Model (hidden)")
    assert two.track_by_id("model").visible is True
    assert two.active_track_id == "model"
    assert lane_counts(two) == [2]
    assert _dialog_counter == []
