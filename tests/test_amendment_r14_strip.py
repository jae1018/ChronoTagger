"""
DRAFT AMENDMENT R14 (Pack 5 v2) -- the Labels strip's per-interval
Rectangles become ONE PolyCollection, closing acceptance gate 3.

Severable: if J.E. declines R14, EDITs 162-163 and this file are struck
together and nothing else in the pack changes.

The two things that had to be proved before this could be drafted, and
that these tests pin:
  1. the strip-click contract survives -- `event.artist` is only a GATE,
     the interval is re-derived from `event.mouseevent.xdata`;
  2. the bands still cover the intervals they are supposed to cover.
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _stub_messagebox(monkeypatch):
    """STANDING RULE (Pack 3): any dialog-reachable path, real-Tk or not."""
    import tkinter.messagebox as mb
    calls = []
    for kind in ("showinfo", "showwarning", "showerror", "askyesno",
                 "askyesnocancel"):
        monkeypatch.setattr(
            mb, kind, lambda *a, _k=kind, **kw: calls.append(_k) or True)
    return calls


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _bands(labeler):
    return [c for c in labeler.strip_ax.collections
            if str(c.get_gid() or "").endswith("strip-bands")]


def _make_intervals(labeler, n=4):
    """n non-overlapping intervals, all INSIDE the fixture's 30-minute
    window (the strip clips to [t0, t1), so an interval past t1 would
    correctly contribute no band and make the counts below lie)."""
    from chronotagger.core.models import Interval
    idx = labeler.df.index
    labeler.classes = ["UNKNOWN", "PS", "LOBE"]
    labeler.class_colors = {"UNKNOWN": "#cccccc", "PS": "#4e79a7",
                            "LOBE": "#f28e2b"}
    ivs = [Interval(idx[5 + i * 12], idx[11 + i * 12],
                    "PS" if i % 2 == 0 else "LOBE") for i in range(n)]
    labeler.intervals = ivs
    assert all(iv.end <= labeler.t1 for iv in ivs), "fixture window too small"
    return ivs


def test_strip_draws_one_collection_not_one_patch_per_interval(labeler):
    """R14's whole point: at 2,000 intervals the old loop built 2,000
    pickable patches every frame (measured 1,419.9 ms of a 2,251.4 ms
    redraw); this builds one collection (27.0 ms)."""
    ivs = _make_intervals(labeler, 4)
    labeler._update_plot()

    bands = _bands(labeler)
    assert len(bands) == 1, "exactly one band collection on the strip"
    assert len(bands[0].get_paths()) == len(ivs), "one face per interval"
    # no per-interval Rectangles left behind (the one patch that remains
    # is the permanently-invisible two-click overlay, which predates this
    # pack and is not an interval band)
    interval_patches = [p for p in labeler.strip_ax.patches
                        if p.get_visible()]
    assert interval_patches == []


def test_strip_bands_cover_the_intervals_they_represent(labeler):
    """Coverage pin: each face spans exactly its interval in x, clipped to
    the window, in matplotlib date numbers."""
    ivs = _make_intervals(labeler, 4)
    labeler._update_plot()

    paths = _bands(labeler)[0].get_paths()
    assert len(paths) == len(ivs)
    for iv, path in zip(ivs, paths):
        xs = path.vertices[:, 0]
        assert np.isclose(xs.min(), mdates.date2num(iv.start))
        assert np.isclose(xs.max(), mdates.date2num(iv.end))
        ys = path.vertices[:, 1]
        assert np.isclose(ys.min(), 0.1) and np.isclose(ys.max(), 0.9)


def test_the_selected_interval_keeps_its_emphasis(labeler):
    """The Rectangles carried per-interval alpha / edgecolor / linewidth;
    the collection carries them per FACE. Losing that would make the
    selected interval indistinguishable."""
    ivs = _make_intervals(labeler, 4)
    labeler.selected_interval = ivs[2]
    labeler._update_plot()

    band = _bands(labeler)[0]
    faces = band.get_facecolor()
    edges = band.get_edgecolor()
    widths = band.get_linewidth()
    assert abs(float(faces[2][3]) - 0.8) < 1e-6, "selected face alpha"
    assert abs(float(faces[0][3]) - 0.6) < 1e-6, "unselected face alpha"
    assert float(edges[2][0]) > 0.9 and float(edges[2][1]) < 0.1, "red edge"
    assert abs(float(widths[2]) - 2.0) < 1e-6
    assert abs(float(widths[0]) - 0.5) < 1e-6


def test_strip_click_still_selects_the_interval_under_the_cursor(labeler):
    """THE contract R14 had to survive. `event.artist` is only a gate; the
    interval comes from `event.mouseevent.xdata`. Four different targets,
    all through the one collection."""
    ivs = _make_intervals(labeler, 4)
    labeler._update_plot()

    for target in ivs:
        # Re-fetch: a successful select redraws the strip and replaces the
        # collection, exactly as it replaced the Rectangles before.
        band = _bands(labeler)[0]
        mid = target.start + (target.end - target.start) / 2
        mouse = type("ME", (), {"xdata": mdates.date2num(mid), "ydata": 0.5,
                                "inaxes": labeler.strip_ax})
        pick = type("PE", (), {"artist": band, "mouseevent": mouse})
        labeler.selected_interval = None
        labeler._on_strip_click(pick, labeler.active_pane)
        assert labeler.selected_interval is target


def test_strip_click_gate_still_rejects_a_foreign_artist(labeler):
    """The widening must add the strip's own collections and nothing else:
    a pick on an artist that is not on the strip is still ignored."""
    _make_intervals(labeler, 4)
    labeler._update_plot()

    foreign = labeler.user_axes["panel1"].lines[0]
    mouse = type("ME", (), {"xdata": mdates.date2num(labeler.df.index[15]),
                            "ydata": 0.5, "inaxes": labeler.strip_ax})
    pick = type("PE", (), {"artist": foreign, "mouseevent": mouse})
    labeler.selected_interval = None
    labeler._on_strip_click(pick, labeler.active_pane)
    assert labeler.selected_interval is None


def test_strip_click_toggles_the_selection_off(labeler):
    """Clicking the already-selected interval deselects it -- the branch
    below the gate, unchanged by R14 and worth keeping honest."""
    ivs = _make_intervals(labeler, 4)
    labeler._update_plot()
    target = ivs[1]
    mid = target.start + (target.end - target.start) / 2

    def click():
        # Re-fetch the collection each time: selecting redraws the strip,
        # which replaces the artist -- exactly as it replaced the
        # Rectangles before. matplotlib always dispatches a pick against
        # the CURRENT artist, so this mirrors the real event source.
        band = _bands(labeler)[0]
        mouse = type("ME", (), {"xdata": mdates.date2num(mid), "ydata": 0.5,
                                "inaxes": labeler.strip_ax})
        labeler._on_strip_click(
            type("PE", (), {"artist": band, "mouseevent": mouse}),
            labeler.active_pane)

    click()
    assert labeler.selected_interval is target
    click()
    assert labeler.selected_interval is None


def test_preview_spans_stay_rectangles(labeler):
    """R14 converts the LABELLED bands only. The previews are a handful of
    artists, carry no picker, and are left alone."""
    _make_intervals(labeler, 2)
    idx = labeler.df.index
    labeler.current_selection = (idx[50], idx[58])
    labeler._update_plot()

    assert len(_bands(labeler)) == 1
    # alpha 0.3 identifies the preview; the strip also carries one
    # permanently-invisible two-click overlay rectangle (alpha 0.25) that
    # predates this pack.
    previews = [p for p in labeler.strip_ax.patches
                if abs((p.get_alpha() or 1.0) - 0.3) < 1e-6]
    assert len(previews) == 1, "the preview is still a patch"


def test_an_empty_interval_list_draws_no_band_collection(labeler):
    labeler.intervals = []
    labeler._update_plot()
    assert _bands(labeler) == []
