"""
Pack 8.5-C (scatter highlights + the fast-plot_fn guide) regression tests.

GC1 pins run in the regime the suite never covered: a `role='time'`
panel whose data artist is a PathCollection rather than a Line2D.
Before this pack such a panel selected correctly and showed ZERO red
preview marks and ZERO blue interval marks under every gesture, at both
decimate settings, silently.

What each group owns:
  GC1 highlights -- a scatter-drawn time panel marks exactly what the
                    line control marks, the marks land on the drawn
                    offsets, the tool never reads its own ink, and the
                    two gates (ordinal length, component label) still
                    reject what they always rejected
  GC2 the guide  -- docs/fast-plot-fn.md exists, is ASCII, is linked
                    from the README, does not re-print the README's own
                    performance figures, and states rule 2 the way the
                    measurement supports
  GC4 the demo   -- examples/dual_pane_demo.py draws both of its
                    role='time' panels against the time axis, and both
                    of them can be highlighted
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import chronotagger

REPO = Path(chronotagger.__file__).resolve().parents[2]
PREVIEW_GID = "chronotagger:preview-highlight"
INTERVAL_GID = "chronotagger:interval-highlight"


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


# --------------------------------------------------------------- helpers

def marks(lbl, key, gid):
    """(artist count, total marker count) on ONE panel."""
    arts = [c for c in lbl.user_axes[key].collections
            if (c.get_gid() or "") == gid]
    return len(arts), sum(int(a.get_offsets().shape[0]) for a in arts)


class NoY:
    """No ydata attribute at all -> the full-height (time-only) branch."""

    def __init__(self, xdata, inaxes, button=1):
        self.xdata = xdata
        self.inaxes = inaxes
        self.button = button


def full_height_drag(lbl, key, lo=0.25, hi=0.75):
    ax = lbl.user_axes[key]
    x0 = mdates.date2num(lbl.t0 + (lbl.t1 - lbl.t0) * lo)
    x1 = mdates.date2num(lbl.t0 + (lbl.t1 - lbl.t0) * hi)
    lbl._on_rectangle_select(NoY(x0, ax), NoY(x1, ax), lbl.active_pane)


def build(df, areas, plot_fn, tmp_path, decimate=True, window=None):
    from chronotagger.labeler import TimeIntervalLabeler
    nrow = max(a["row"] for a in areas) + 1
    layout = {"nrows": nrow, "ncols": 1, "areas": areas}
    lbl = TimeIntervalLabeler(
        df=df, plot_fn=plot_fn, layout_spec=layout, decimate=decimate,
        window=window if window is not None else df.index[-1] - df.index[0],
        autosave_folder=str(tmp_path))
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    return lbl


def time_areas(keys):
    areas = [{"key": k, "row": i, "col": 0, "role": "time"}
             for i, k in enumerate(keys)]
    areas.append({"key": "labels", "row": len(keys), "col": 0,
                  "role": "labels"})
    return areas


# ------------------------------------------------- GC1: the scatter panel

@pytest.fixture
def df_small():
    idx = pd.date_range("2015-01-03", periods=2000, freq="1s")
    t = np.linspace(0, 40, 2000)
    return pd.DataFrame({"BX": np.sin(t) * 10, "BY": np.cos(t) * 5},
                        index=idx)


@pytest.fixture
def pair_labeler(df_small, tmp_path):
    """The same data drawn two ways on two role='time' panels."""
    def plot_fn(axs, df, t0, t1):
        y = df["BX"].to_numpy(dtype=float)
        axs["line"].plot(df.index, y)
        axs["scat"].scatter(df.index, y, s=2)

    lbl = build(df_small, time_areas(["line", "scat"]), plot_fn, tmp_path,
                decimate=False)
    yield lbl
    lbl.root.destroy()


def test_the_pair_fixture_really_is_a_collection_against_a_line(pair_labeler):
    """Every pin below is vacuous if the scatter panel grew a Line2D."""
    lbl = pair_labeler
    assert len(lbl.user_axes["line"].lines) == 1
    assert len(lbl.user_axes["scat"].lines) == 0
    data = [c for c in lbl.user_axes["scat"].collections
            if not (c.get_gid() or "").startswith("chronotagger:")]
    assert len(data) == 1
    assert np.asarray(data[0].get_offsets()).shape[0] == len(
        lbl._last_windowed_index)


def test_a_scatter_time_panel_gets_preview_marks(pair_labeler):
    """GC1's headline. Measured on the shipped tree at 575fa28, three
    gesture families x two decimate settings: 0 red marks on the scatter
    panel, against 1,000 on the line control every time."""
    lbl = pair_labeler
    full_height_drag(lbl, "scat")
    line_art, line_pts = marks(lbl, "line", PREVIEW_GID)
    scat_art, scat_pts = marks(lbl, "scat", PREVIEW_GID)
    assert line_pts > 0
    assert (scat_art, scat_pts) == (line_art, line_pts)


def test_a_scatter_time_panel_gets_interval_marks(pair_labeler):
    """The blue family shares the extractor, so it shared the defect."""
    from chronotagger.core.models import Interval
    lbl = pair_labeler
    idx = lbl.df.index
    lbl.selected_interval = Interval(start=idx[100], end=idx[600],
                                     label="UNKNOWN")
    lbl._show_selected_interval_highlights()
    assert marks(lbl, "line", INTERVAL_GID)[1] > 0
    assert marks(lbl, "scat", INTERVAL_GID) == marks(lbl, "line",
                                                     INTERVAL_GID)


def test_the_extractor_agrees_point_for_point_with_the_line(pair_labeler):
    """Not just the same COUNT: the same numbers. The scatter's offsets
    are the line's vertices."""
    lbl = pair_labeler
    want = lbl._extract_data_at_indices(lbl.user_axes["line"],
                                        list(range(100, 140)))
    got = lbl._extract_data_at_indices(lbl.user_axes["scat"],
                                       list(range(100, 140)))
    assert len(want[0]) == 40
    assert np.allclose(np.asarray(got[0]), np.asarray(want[0]))
    assert np.allclose(np.asarray(got[1]), np.asarray(want[1]))


def test_the_marks_land_on_the_scatter_offsets(pair_labeler):
    """WYSIWYG: a mark is a vertex the user can see, which for a
    PathCollection means a row of get_offsets()."""
    lbl = pair_labeler
    full_height_drag(lbl, "scat")
    data = [c for c in lbl.user_axes["scat"].collections
            if not (c.get_gid() or "").startswith("chronotagger:")][0]
    drawn = set(map(tuple, np.round(np.asarray(data.get_offsets()), 9)))
    art = [c for c in lbl.user_axes["scat"].collections
           if (c.get_gid() or "") == PREVIEW_GID][0]
    got = set(map(tuple, np.round(np.asarray(art.get_offsets()), 9)))
    assert got and got <= drawn


def test_the_extractor_never_reads_the_tools_own_ink(pair_labeler):
    """The red and blue overlays ARE PathCollections on this axes, and a
    selection covering the whole window gives them EXACTLY the length the
    ordinal gate accepts -- measured, 2,000 offsets against a 2,000-row
    window.

    The preview path clears its own artists before it extracts, so the
    hazard that survives is the BLUE family, which it does not clear: a
    drag taken while an interval is highlighted returns 4,000 marks
    instead of 2,000 if the gid guard is removed (measured), and it does
    so on the line panel too."""
    from chronotagger.core.models import Interval
    lbl = pair_labeler
    n_drawn = len(lbl._last_windowed_index)
    full_height_drag(lbl, "scat", 0.0, 1.0)
    clean = marks(lbl, "scat", PREVIEW_GID)
    assert clean == (1, n_drawn), clean

    lbl._clear_selected_point_highlights()
    idx = lbl.df.index
    lbl.selected_interval = Interval(start=idx[0],
                                     end=idx[-1] + pd.Timedelta("1s"),
                                     label="UNKNOWN")
    lbl._show_selected_interval_highlights()
    assert marks(lbl, "scat", INTERVAL_GID) == (1, n_drawn)

    full_height_drag(lbl, "scat", 0.0, 1.0)
    assert marks(lbl, "scat", PREVIEW_GID) == clean, "read its own ink"
    assert marks(lbl, "line", PREVIEW_GID) == clean, "read its own ink"


def test_a_scatter_panel_marks_while_decimation_is_active(tmp_path):
    """The two regimes are independent: the drawn frame is shorter AND
    the artist is a collection."""
    idx = pd.date_range("2015-01-03", periods=20000, freq="1s")
    t = np.linspace(0, 200, 20000)
    df = pd.DataFrame({"BX": np.sin(t) * 10}, index=idx)

    def plot_fn(axs, df_, t0, t1):
        axs["scat"].scatter(df_.index, df_["BX"].to_numpy(dtype=float), s=2)

    lbl = build(df, time_areas(["scat"]), plot_fn, tmp_path, decimate=True)
    try:
        assert lbl._decim_active is True
        drawn = np.asarray(
            [c for c in lbl.user_axes["scat"].collections
             if not (c.get_gid() or "").startswith("chronotagger:")
             ][0].get_offsets())
        assert drawn.shape[0] < len(lbl.df.loc[lbl.t0:lbl.t1])
        full_height_drag(lbl, "scat")
        n_art, n_pts = marks(lbl, "scat", PREVIEW_GID)
        assert n_art == 1 and n_pts > 0
        art = [c for c in lbl.user_axes["scat"].collections
               if (c.get_gid() or "") == PREVIEW_GID][0]
        xs = set(np.round(np.asarray(art.get_offsets())[:, 0], 9))
        assert xs <= set(np.round(drawn[:, 0], 9)), \
            "a mark landed on a sample that was never drawn"
    finally:
        lbl.root.destroy()


def test_a_short_scatter_and_a_fill_between_are_not_read_as_data(df_small,
                                                                 tmp_path):
    """The ordinal gate (artist point i -> windowed row i) is what makes
    the mapping legal, and it rejects an annotation scatter of a
    different length as well as the [[0, 0]] sentinel offsets a
    fill_between PolyCollection carries.

    The discriminating probe is the LOW one: the annotation scatter holds
    10 points and the sentinel holds 1, so ordinals 100..139 fall outside
    both whatever the gate does. At ordinals 0..9 the gate is the only
    thing between the answer and 10 + 10 + 1 = 21 (measured, with the
    gate removed)."""
    def plot_fn(axs, df, t0, t1):
        ax = axs["mixed"]
        y = df["BX"].to_numpy(dtype=float)
        ax.plot(df.index, y)
        ax.fill_between(df.index, y - 1.0, y + 1.0, alpha=0.2)
        ax.scatter(df.index[::200], y[::200], s=30, marker="x", c="k")

    lbl = build(df_small, time_areas(["mixed"]), plot_fn, tmp_path,
                decimate=False)
    try:
        colls = [c for c in lbl.user_axes["mixed"].collections
                 if not (c.get_gid() or "").startswith("chronotagger:")]
        assert len(colls) == 2, [type(c).__name__ for c in colls]
        for probe, want in ((list(range(0, 10)), 10),
                            (list(range(100, 140)), 40)):
            xs, ys = lbl._extract_data_at_indices(lbl.user_axes["mixed"],
                                                  probe)
            assert len(xs) == want, (
                "the line alone should answer ordinals %d..%d; got %d"
                % (probe[0], probe[-1], len(xs)))
    finally:
        lbl.root.destroy()


def test_a_line_and_a_full_length_scatter_are_both_read(df_small, tmp_path):
    """Deliberate: the collections pass is NOT conditional on the line
    scan coming back empty, so a panel that draws its data twice is
    marked twice -- the same way the box scan reads both artist
    families."""
    def plot_fn(axs, df, t0, t1):
        y = df["BX"].to_numpy(dtype=float)
        axs["both"].plot(df.index, y)
        axs["both"].scatter(df.index, y, s=2)

    lbl = build(df_small, time_areas(["both"]), plot_fn, tmp_path,
                decimate=False)
    try:
        xs, ys = lbl._extract_data_at_indices(lbl.user_axes["both"],
                                              list(range(100, 140)))
        assert len(xs) == 80
    finally:
        lbl.root.destroy()


def test_the_component_filter_applies_to_scatters_too(df_small, tmp_path):
    """A labelled scatter is filtered exactly the way a labelled line is;
    otherwise the meaning of the component filter would depend on which
    artist class the plot_fn author happened to pick."""
    def plot_fn(axs, df, t0, t1):
        ax = axs["comps"]
        ax.scatter(df.index, df["BX"].to_numpy(dtype=float), s=2,
                   label="B_x")
        ax.scatter(df.index, df["BY"].to_numpy(dtype=float), s=2,
                   label="B_y")

    lbl = build(df_small, time_areas(["comps"]), plot_fn, tmp_path,
                decimate=False)
    try:
        idx = list(range(100, 140))
        assert len(lbl._extract_data_at_indices(
            lbl.user_axes["comps"], idx)[0]) == 80
        lbl._selected_component_labels = ["B_x"]
        lbl.active_pane._selected_component_labels = ["B_x"]
        assert len(lbl._extract_data_at_indices(
            lbl.user_axes["comps"], idx)[0]) == 40
    finally:
        lbl.root.destroy()


def test_an_auto_labelled_scatter_survives_a_component_filter(df_small,
                                                              tmp_path):
    """matplotlib auto-labels start with '_'; the line loop includes
    those unconditionally and so does this one, which is what keeps the
    wizard's own unlabelled ax.scatter marked."""
    def plot_fn(axs, df, t0, t1):
        axs["auto"].scatter(df.index, df["BX"].to_numpy(dtype=float), s=2)

    lbl = build(df_small, time_areas(["auto"]), plot_fn, tmp_path,
                decimate=False)
    try:
        lbl._selected_component_labels = ["SOMETHING ELSE"]
        lbl.active_pane._selected_component_labels = ["SOMETHING ELSE"]
        assert len(lbl._extract_data_at_indices(
            lbl.user_axes["auto"], list(range(100, 140)))[0]) == 40
    finally:
        lbl.root.destroy()


def test_the_wizards_own_scatter_style_time_panel_gets_marks(df_small,
                                                             tmp_path):
    """GC4's second half, pinned rather than only recorded.
    `plot_generator.py:164-165` puts a PathCollection on a role='time'
    panel whenever `style='scatter'`, and `driver_export.py:746-748`
    writes the same call into user-owned source. That configuration is
    reachable from the wizard's own preset, and before this pack it
    produced zero marks on every panel -- which is why the gather called
    it the worst violation in the tree. It needs no generator change; it
    needs this."""
    from chronotagger.labeler.utils.plot_generator import (
        generate_plot_fn, vertical_stack_config)
    layout_spec, plot_config = vertical_stack_config(["BX", "BY"])
    for cfg in plot_config.values():
        cfg["style"] = "scatter"

    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(
        df=df_small, plot_fn=generate_plot_fn(plot_config),
        layout_spec=layout_spec, decimate=False,
        window=df_small.index[-1] - df_small.index[0],
        autosave_folder=str(tmp_path))
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    try:
        for key in ("panel_1", "panel_2"):
            ax = lbl.user_axes[key]
            assert len(ax.lines) == 0
            assert [type(c).__name__ for c in ax.collections] == [
                "PathCollection"], key
        full_height_drag(lbl, "panel_1")
        for key in ("panel_1", "panel_2"):
            assert marks(lbl, key, PREVIEW_GID)[1] > 0, key
    finally:
        lbl.root.destroy()


# ------------------------------------------------------- GC2: the guide

def _repo_layout():
    """Skip only when the package was installed WITHOUT the repo beside
    it. Inside the repo the files below are not optional, so a missing
    one is RED, not a skip -- deleting the page must not quietly pass."""
    if not (REPO / "src" / "chronotagger").is_dir():
        pytest.skip("not a repo layout (installed without the source tree)")


def guide_path():
    _repo_layout()
    p = REPO / "docs" / "fast-plot-fn.md"
    assert p.is_file(), "docs/fast-plot-fn.md is missing"
    return p


def readme_path():
    _repo_layout()
    p = REPO / "README.md"
    assert p.is_file(), "README.md is missing"
    return p


def test_the_guide_exists_and_is_ascii_and_is_one_page():
    p = guide_path()
    raw = p.read_bytes()
    assert not any(c > 127 for c in bytearray(raw)), "the guide is not ASCII"
    n = len(raw.decode("ascii").splitlines())
    assert 80 <= n <= 220, "one page's worth, not a manual; got %d lines" % n


def test_the_guide_carries_all_six_rules():
    text = guide_path().read_text(encoding="ascii")
    for h in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6."):
        assert h in text, h


def test_the_readme_links_the_guide_exactly_once():
    """The README link IS the wiring: docs/*.md have no build and are
    reached only by relative link."""
    text = readme_path().read_text(encoding="utf-8")
    assert text.count("(docs/fast-plot-fn.md)") == 1
    assert "[`docs/fast-plot-fn.md`](docs/fast-plot-fn.md)" in text
    guide_path()   # the target of that link exists


def test_the_link_sits_in_the_plot_function_contract_section():
    """The contract section is where a plot_fn author already is, and it
    says nothing about cost."""
    text = readme_path().read_text(encoding="utf-8").replace("\r\n", "\n")
    contract = text.index("### Plot Function Contract")
    panels = text.index("### Panel Count Resolution")
    link = text.index("(docs/fast-plot-fn.md)")
    assert contract < link < panels


def test_the_guide_does_not_reprint_the_readmes_own_figures():
    """Two numbers in the README were measured on a different figure
    geometry and a different frame. Duplicating either would create a
    third copy that drifts."""
    text = guide_path().read_text(encoding="ascii")
    for forbidden in ("1.3 s", "3.3 s", "43,000", "100,000", "10-30"):
        assert forbidden not in text, forbidden


def test_rule_2_is_stated_the_way_the_measurement_supports():
    """The charter's own rule-2 example measured NO penalty, so the old
    'compute outside, draw inside' wording cannot ship as advice."""
    text = guide_path().read_text(encoding="ascii").lower()
    assert "compute outside" not in text
    assert "draw inside" not in text
    assert "vectorized" in text
    assert "582.2" in text and "51.2" in text


def test_rule_5_blames_artist_count_and_not_alpha():
    text = guide_path().read_text(encoding="ascii")
    assert "Alpha is free" in text
    assert "697.7" in text and "13.5x" in text


# -------------------------------------------------------- GC4: the demo

def demo_path():
    _repo_layout()
    p = REPO / "examples" / "dual_pane_demo.py"
    assert p.is_file(), "examples/dual_pane_demo.py is missing"
    return p


def test_the_dual_pane_demo_has_no_value_axis_histogram():
    """`.hist()` on a role='time' panel is three rule violations and a
    correctness bug: plotting.py puts that axis on a date scale and pins
    set_xlim(t0, t1) before plot_fn runs, so the bars land off-axis --
    measured, 20 patches at VALUE coordinates against an x limit of
    [19723.00, 19723.08]."""
    text = demo_path().read_text(encoding="utf-8")
    assert ".hist(" not in text
    assert "rolling(" in text
    assert "docs/fast-plot-fn.md" in text


def test_the_dual_pane_demo_draws_both_panels_on_the_time_axis():
    """Both areas are role='time'; both panels must therefore draw
    against the window's own x limits, and both must be highlightable."""
    import importlib.util
    demo = demo_path()
    from chronotagger.labeler.app import TimeIntervalLabeler as _App

    built = {}

    def fake_run(self):
        self._build_gui()
        self.root.withdraw()
        built["lab"] = self

    real_run = _App.run
    _App.run = fake_run
    try:
        spec = importlib.util.spec_from_file_location("dual_pane_demo_pin",
                                                      demo)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        _App.run = real_run

    lbl = built["lab"]
    try:
        for i in range(len(lbl.panes)):
            if i:
                lbl.notebook.select(i)
                lbl._on_tab_changed(None)
            lbl._update_plot()
            full_height_drag(lbl, "top")
            for key in ("top", "bottom"):
                ax = lbl.user_axes[key]
                xlo, xhi = ax.get_xlim()
                wide = [p for p in ax.patches if float(p.get_width()) != 0.0]
                assert wide == [], \
                    "pane %d %s drew %d patches" % (i, key, len(wide))
                for ln in ax.lines:
                    xs = np.asarray(ln.get_xdata(orig=False), dtype=float)
                    assert xs.min() >= xlo - 1e-9 and xs.max() <= xhi + 1e-9
                assert marks(lbl, key, PREVIEW_GID)[1] > 0, \
                    "pane %d %s got no preview marks" % (i, key)
    finally:
        lbl.root.destroy()
