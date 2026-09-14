"""Pack M2.1 -- THE LANE NAMES AND THE LEGEND STAY INSIDE THE FIGURE.

Pack M2 paints the lane names as the strip's y tick labels and ONE legend
outside the strip for the active lane. Both are artists the constrained-
layout solver had never seen when the figure's geometry was frozen after
its first draw (Pack 5 R4a), so on a layout with a narrow margin they
fall off the figure: measured on the C05 feel-test driver at 1400 x 800
px, 43-62 px of every lane name and 78 % of the legend were invisible on
the Context tab, and 34-55 px of every name on the Orbit + Fields tab.
M2.1 asks for ONE more solve whenever the thing the solver must make room
for changes -- the lane count, the lane set, or the active lane -- and
only above one lane, so a single-lane figure keeps the byte-identical
geometry Pack M2 pinned.

Every test builds a REAL labeler with a REAL Tk root and a REAL figure
(Agg), because what is being pinned is where an artist lands in display
pixels, not what a function returns.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

DAY = "2015-01-03 "

ONE_LANE = [{"id": "human", "name": "Human", "classes": ["sw", "msh"],
             "class_colors": {"sw": "#4e79a7", "msh": "#f28e2b"}}]
THREE_LANES = [
    {"id": "region", "name": "Region (human)",
     "classes": ["UNKNOWN", "solar_wind", "magnetosheath", "plasma_sheet",
                 "lobe", "mantle"], "order": 0},
    {"id": "wake", "name": "Wake (umbra)", "classes": ["umbra"],
     "order": 1},
    {"id": "agent", "name": "Agent (C-MMAE)", "classes": ["0", "1", "2", "3"],
     "order": 2},
]

LAYOUT = {
    "nrows": 3, "ncols": 1,
    "areas": [
        {"key": "panel1", "row": 0, "col": 0, "role": "time"},
        {"key": "panel2", "row": 1, "col": 0, "role": "time"},
        {"key": "labels", "row": 2, "col": 0, "role": "labels"},
    ],
}


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


def frame():
    idx = pd.date_range(DAY + "00:00:00", periods=120, freq="30s")
    return pd.DataFrame({"a": np.linspace(0, 1, len(idx)),
                         "b": np.linspace(1, 0, len(idx))}, index=idx)


def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["a"])
    axs["panel2"].plot(df.index, df["b"])


def build(tracks, tmp_path):
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(df=frame(), window=pd.Timedelta("30min"),
                              autosave_folder=str(tmp_path),
                              tracks=[dict(t) for t in tracks],
                              plot_fn=plot_fn, layout_spec=dict(LAYOUT))
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    return lbl


@pytest.fixture
def one(tmp_path):
    lbl = build(ONE_LANE, tmp_path)
    yield lbl
    lbl.root.destroy()


@pytest.fixture
def three(tmp_path):
    lbl = build(THREE_LANES, tmp_path)
    yield lbl
    lbl.root.destroy()


def ctrl(lbl, keysym):
    class E(object):
        pass
    e = E()
    e.keysym = keysym
    e.state = 0x4
    lbl._on_key_press(e)


def overflow(lbl):
    """(pixels of the widest lane name outside the LEFT figure edge,
    pixels of the legend outside the RIGHT figure edge), after a draw."""
    pane = lbl.active_pane
    fig = pane.fig
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax = pane.strip_ax
    names = max([0.0] + [-t.get_window_extent(r).x0
                         for t in ax.get_yticklabels()])
    leg = ax.get_legend()
    legend = 0.0
    if leg is not None:
        legend = max(0.0, leg.get_window_extent(r).x1 - fig.bbox.width)
    return names, legend


def frozen(lbl):
    pane = lbl.active_pane
    return (bool(getattr(pane, "_layout_frozen", False)),
            type(pane.fig.get_layout_engine()).__name__)


# ---------------------------------------------------------------------
# three lanes: everything the solver must make room for is inside
# ---------------------------------------------------------------------

def test_three_lane_names_and_the_legend_sit_inside_the_figure(three):
    names, legend = overflow(three)
    assert names < 0.5, "%.1f px of a lane name are off the left edge" % names
    assert legend < 0.5, "%.1f px of the legend are off the right edge" % legend
    assert frozen(three) == (True, "PlaceHolderLayoutEngine")


def test_a_lane_switch_keeps_the_legend_inside_and_refreezes(three):
    for _ in range(2):
        ctrl(three, "Down")
        three._update_plot()
        names, legend = overflow(three)
        assert names < 0.5 and legend < 0.5, (three.active_track_id,
                                              names, legend)
        assert frozen(three) == (True, "PlaceHolderLayoutEngine")
    assert three.active_track_id == "agent"


def test_hiding_a_lane_keeps_the_names_inside(three):
    ctrl(three, "h")
    three._update_plot()
    assert three.active_pane._strip_lane_count == 2
    names, legend = overflow(three)
    assert names < 0.5 and legend < 0.5, (names, legend)
    assert frozen(three) == (True, "PlaceHolderLayoutEngine")


def test_the_re_solve_is_asked_once_per_change_not_per_repaint(three):
    pane = three.active_pane
    sig = pane._strip_layout_sig
    assert sig[0] == 3 and sig[2] == "region"
    pos = tuple(round(v, 6) for v in pane.strip_ax.get_position().bounds)
    for _ in range(3):
        three._update_plot()
        assert pane._strip_layout_sig == sig
        assert frozen(three) == (True, "PlaceHolderLayoutEngine")
        now = tuple(round(v, 6) for v in pane.strip_ax.get_position().bounds)
        assert now == pos


# ---------------------------------------------------------------------
# one lane: the byte-identity floor -- never a re-solve
# ---------------------------------------------------------------------

def test_a_single_lane_figure_never_asks_for_a_re_solve(one):
    pane = one.active_pane
    assert pane._strip_layout_sig == (1, ("human",), "human")
    pos = tuple(round(v, 6) for v in pane.strip_ax.get_position().bounds)
    for _ in range(3):
        one._update_plot()
        assert frozen(one) == (True, "PlaceHolderLayoutEngine")
        now = tuple(round(v, 6) for v in pane.strip_ax.get_position().bounds)
        assert now == pos
    assert pane.strip_ax.get_legend() is None
    assert [t.get_text() for t in pane.strip_ax.get_yticklabels()] == []
