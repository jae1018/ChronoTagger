"""Pack 7 -- one generator, and the x_col/y_col the designer never emitted.

Four groups.

1. THE MERGE (W1).  `vertical_stack_config` is the wizard's Vertical
   Stack radio expressed over the designer's vocabulary, and the wizard
   now reaches ONE generator whichever radio the user picked.  The
   deleted module and the deleted code-string ramp are absence-checked
   with the Pack 6 detector, which sees names inside string literals.
2. THE W2 A/B (the pack's headline behaviour change).  A layout built by
   the real `_generate_layout_spec` is driven through the labeler's own
   box filter.  Before the fix that call returned None -- "no column
   mapping" -- and the labeler fell back to scanning drawn artists.
   After it, the same call returns an exact dataframe-filtered interval.
3. LAYOUT_BUILDER EMISSION.  The first tests this package has for
   `_generate_layout_spec` / `_generate_plot_config` (the gather counted
   0 of 29 functions covered).  Not the GUI -- just what those two
   emit, driven from a bare host with no Tk.
4. THE RUNTIME half of W7: `generate_plot_fn` honours a `yscale` key on
   both roles, so the live figure and the emitted driver can express the
   same panel.

No timing assertion appears anywhere in this file.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import chronotagger
from chronotagger.labeler import TimeIntervalLabeler
from chronotagger.labeler.utils import (
    generate_plot_fn,
    normalize_time_columns,
    validate_plot_inputs,
    vertical_stack_config,
)
from chronotagger.labeler.utils.layout_builder.models import PanelConfig
from chronotagger.labeler.utils.layout_builder.preview import PreviewMixin

SRC_ROOT = Path(chronotagger.__file__).resolve().parent


def _all_src_files():
    return [p for p in SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts]


def _strip_docstrings(node):
    body = getattr(node, "body", None)
    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                         ast.AsyncFunctionDef)) and body:
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    for child in ast.iter_child_nodes(node):
        _strip_docstrings(child)


def _code_and_literals(path: Path) -> str:
    """Pack 6's detector, reused verbatim.

    Code INCLUDING string literals, with comments and docstrings gone.
    It must SEE a name inside a string (this pack leaves several deleted
    names in prose that a plain substring scan would match) and must NOT
    see one in a comment.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    _strip_docstrings(tree)
    return ast.unparse(tree)


def test_the_absence_detector_sees_strings_but_not_prose(tmp_path):
    """The same guard Pack 6 carries: if a future refactor swaps this
    for a code-tokens-only scan, THIS fails instead of the absence
    checks below silently going vacuous."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        '"""A docstring mentioning DOCSTRING_NAME."""\n'
        "# a comment mentioning COMMENT_NAME\n"
        "def f(self):\n"
        "    return getattr(self, 'STRING_NAME', None)\n"
        "CODE_NAME = 1\n",
        encoding="utf-8")
    seen = _code_and_literals(sample)
    assert "STRING_NAME" in seen
    assert "CODE_NAME" in seen
    assert "COMMENT_NAME" not in seen
    assert "DOCSTRING_NAME" not in seen


# =====================================================================
# 1 -- THE MERGE
# =====================================================================

def test_the_second_generator_module_is_gone():
    assert importlib.util.find_spec("chronotagger.quickstart.plot_builder") \
        is None


@pytest.mark.parametrize("name", [
    "build_plot_function", "build_layout_spec",
    "generate_plot_code", "print_plot_code",
])
def test_the_deleted_generator_names_appear_nowhere_in_src(name):
    hits = [str(p.relative_to(SRC_ROOT))
            for p in _all_src_files()
            if name in _code_and_literals(p)]
    assert hits == []


def test_the_utils_export_surface_lost_the_code_ramp():
    from chronotagger.labeler import utils

    assert "generate_plot_code" not in utils.__all__
    assert "print_plot_code" not in utils.__all__
    assert not hasattr(utils, "generate_plot_code")
    assert not hasattr(utils, "print_plot_code")
    for name in ("build_layout", "generate_plot_fn", "vertical_stack_config",
                 "normalize_time_columns", "validate_plot_inputs"):
        assert name in utils.__all__
        assert hasattr(utils, name)


def test_quickstart_star_import_still_works():
    import chronotagger.quickstart as qs

    assert "plot_builder" not in qs.__all__
    ns = {}
    exec("from chronotagger.quickstart import *", ns)   # must not raise


def test_vertical_stack_preset_has_the_designers_shape():
    """Four measured disagreements between the deleted Generator A and
    the designer, all resolved the designer's way (gather, area 3)."""
    layout, cfg = vertical_stack_config(["BX", "n_linear", "X"])

    assert layout["nrows"] == 4 and layout["ncols"] == 1
    # (a) hspace / wspace present -- A emitted neither
    assert layout["hspace"] == 0.15
    assert layout["wspace"] == 0.12

    keys = [a["key"] for a in layout["areas"]]
    # (b) panel_1..panel_N, not panel0..panel_{N-1}
    assert keys == ["panel_1", "panel_2", "panel_3", "labels"]

    for area in layout["areas"]:
        # (c) rowspan / colspan only when they exceed 1 -- A always wrote
        # them, even as 1
        assert "rowspan" not in area
        assert "colspan" not in area

    labels = layout["areas"][-1]
    assert labels == {"key": "labels", "row": 3, "col": 0, "role": "labels"}

    # (d) a plot_config comes back too, which is what lets the same
    # answer drive a generated driver file and not only a live closure
    assert cfg == {
        "panel_1": {"role": "time", "y_column": "BX"},
        "panel_2": {"role": "time", "y_column": "n_linear"},
        "panel_3": {"role": "time", "y_column": "X"},
    }
    # ...and NO labels entry: the labeler holds that strip outside `axs`.
    assert "labels" not in cfg


def test_vertical_stack_preset_refuses_an_empty_column_list():
    with pytest.raises(ValueError):
        vertical_stack_config([])


def test_the_preset_renders_through_the_one_generator_without_accumulating():
    """The A-era bug, pinned gone.

    Measured on the deleted generator: three renders onto the same axes
    left THREE lines, because it relied on the labeler having called
    ax.clear() first and did no cleaning of its own.  The surviving
    generator removes its own data artists, so the same three renders
    leave ONE.
    """
    layout, cfg = vertical_stack_config(["BX", "n_linear"])
    plot_fn = generate_plot_fn(cfg)

    idx = pd.date_range("2021-03-01", periods=50, freq="10s")
    df = pd.DataFrame({"BX": np.arange(50.0),
                       "n_linear": np.arange(50.0) + 1.0}, index=idx)

    fig = Figure()
    axs = {"panel_1": fig.add_subplot(2, 1, 1),
           "panel_2": fig.add_subplot(2, 1, 2)}

    for _ in range(3):
        plot_fn(axs, df, idx[0], idx[-1])

    assert len(axs["panel_1"].lines) == 1
    assert len(axs["panel_2"].lines) == 1
    # A labels key in the layout must not become a KeyError: the
    # generator skips panels it was not given axes for.
    assert "labels" not in axs


def test_the_absorbed_helpers_still_work_where_they_landed():
    """normalize_time_columns and validate_plot_inputs moved out of the
    deleted module byte-for-byte; they are layout helpers, not wizard
    screens."""
    spec = {"nrows": 3, "ncols": 3, "areas": [
        {"key": "panel_1", "row": 0, "col": 1, "colspan": 2, "role": "time"},
        {"key": "panel_2", "row": 1, "col": 0, "role": "time"},
        {"key": "labels", "row": 2, "col": 0, "role": "labels"},
        {"key": "xp", "row": 0, "col": 0, "role": "not-time"},
    ]}
    normalize_time_columns(spec)
    by_key = {a["key"]: a for a in spec["areas"]}
    assert (by_key["panel_2"]["col"], by_key["panel_2"]["colspan"]) == (1, 2)
    assert (by_key["labels"]["col"], by_key["labels"]["colspan"]) == (1, 2)
    assert by_key["xp"]["col"] == 0 and "colspan" not in by_key["xp"]

    df = pd.DataFrame({"a": [1], "b": [2]})
    validate_plot_inputs(df, ["a"])
    with pytest.raises(ValueError, match="nope"):
        validate_plot_inputs(df, ["a", "nope"])


def test_the_wizard_reaches_one_generator_on_both_radios():
    """_build_tab_plot, driven without Tk.  Both layout types must come
    back with a plot_fn AND the plot_config the driver emitter needs --
    a closure cannot be read back out of a plot_fn."""
    from chronotagger.quickstart.wizard import QuickStartWizard

    class _Host:
        _build_tab_plot = QuickStartWizard._build_tab_plot

        def __init__(self, df):
            self.df = df

    idx = pd.date_range("2021-03-01", periods=20, freq="1min")
    df = pd.DataFrame({"BX": np.arange(20.0), "X": np.arange(20.0),
                       "Y": np.arange(20.0)}, index=idx)
    host = _Host(df)

    plot_fn, layout, cfg = host._build_tab_plot(
        {"title": "T", "columns": ["BX"], "layout_type": "vertical_stack"})
    assert callable(plot_fn)
    assert [a["key"] for a in layout["areas"]] == ["panel_1", "labels"]
    assert cfg == {"panel_1": {"role": "time", "y_column": "BX"}}
    assert plot_fn.__module__ == "chronotagger.labeler.utils.plot_generator"

    designed_layout = {"nrows": 2, "ncols": 2, "areas": [
        {"key": "panel_1", "row": 0, "col": 1, "role": "time"},
        {"key": "labels", "row": 1, "col": 0, "role": "labels"},
        {"key": "xp", "row": 0, "col": 0, "role": "not-time",
         "x_col": "X", "y_col": "Y"}]}
    designed_cfg = {
        "panel_1": {"role": "time", "y_column": "BX"},
        "xp": {"role": "not-time", "x_column": "X", "y_column": "Y"}}
    plot_fn2, layout2, cfg2 = host._build_tab_plot({
        "title": "T", "columns": ["BX", "X", "Y"],
        "layout_type": "custom_grid",
        "layout_spec": designed_layout, "plot_config": designed_cfg})
    assert plot_fn2.__module__ == "chronotagger.labeler.utils.plot_generator"
    assert cfg2 is designed_cfg
    # the designed layout was normalized in place, as before
    by_key = {a["key"]: a for a in layout2["areas"]}
    assert by_key["labels"]["col"] == 1
    # ...and x_col/y_col survived normalization untouched
    assert by_key["xp"]["x_col"] == "X" and by_key["xp"]["y_col"] == "Y"

    with pytest.raises(ValueError, match="Unknown layout type"):
        host._build_tab_plot({"title": "T", "columns": ["BX"],
                              "layout_type": "spiral"})


# =====================================================================
# 2 -- LAYOUT_BUILDER EMISSION (first tests for these two methods)
# =====================================================================

class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _DesignerHost(PreviewMixin):
    """Bare host carrying only the emission half of the designer.

    No Tk, no canvas, no widgets -- `_generate_layout_spec` and
    `_generate_plot_config` read exactly three attributes.
    """

    def __init__(self, panels, nrows, ncols):
        self.panels = panels
        self.nrows_var = _Var(nrows)
        self.ncols_var = _Var(ncols)


def _designed_panels():
    """What the designer holds after: two time panels stacked in col 0,
    the auto-created labels strip beneath them, and a cross-plot spanning
    two rows in col 1."""
    return [
        PanelConfig(key="labels", row=2, col=0, role="labels", locked=True),
        PanelConfig(key="panel_1", row=0, col=0, role="time", y_column="BX"),
        PanelConfig(key="panel_2", row=1, col=0, role="time",
                    y_column="n_linear"),
        PanelConfig(key="panel_3", row=0, col=1, rowspan=2, role="not-time",
                    x_column="X", y_column_2="Y"),
    ]


def test_generate_layout_spec_shape():
    spec = _DesignerHost(_designed_panels(), 3, 2)._generate_layout_spec()

    assert spec["nrows"] == 3 and spec["ncols"] == 2
    assert spec["hspace"] == 0.15 and spec["wspace"] == 0.12
    assert [a["key"] for a in spec["areas"]] == [
        "labels", "panel_1", "panel_2", "panel_3"]

    by_key = {a["key"]: a for a in spec["areas"]}
    assert by_key["labels"]["role"] == "labels"
    # rowspan/colspan only when they exceed 1
    assert "rowspan" not in by_key["panel_1"]
    assert "colspan" not in by_key["panel_1"]
    assert by_key["panel_3"]["rowspan"] == 2
    assert "colspan" not in by_key["panel_3"]


def test_generate_layout_spec_emits_x_col_and_y_col_on_cross_plots():
    """W2.  The designer stores the pair as x_column / y_column_2; the
    labeler's box filter reads x_col / y_col off the layout_spec.  The
    two vocabularies used never to meet."""
    spec = _DesignerHost(_designed_panels(), 3, 2)._generate_layout_spec()
    by_key = {a["key"]: a for a in spec["areas"]}

    assert by_key["panel_3"]["x_col"] == "X"
    assert by_key["panel_3"]["y_col"] == "Y"
    # time and labels areas get neither -- they select on time
    assert "x_col" not in by_key["panel_1"]
    assert "y_col" not in by_key["panel_1"]
    assert "x_col" not in by_key["labels"]


def test_generate_layout_spec_omits_the_pair_when_the_designer_has_no_columns():
    """A cross-plot the user dropped on the grid but never filled in must
    not emit x_col=None: `if not x_col` in the box filter would treat it
    as absent anyway, and a None in the layout reads like a bug."""
    panels = [PanelConfig(key="panel_1", row=0, col=0, role="time",
                          y_column="BX"),
              PanelConfig(key="panel_2", row=0, col=1, role="not-time")]
    spec = _DesignerHost(panels, 2, 2)._generate_layout_spec()
    by_key = {a["key"]: a for a in spec["areas"]}
    assert "x_col" not in by_key["panel_2"]
    assert "y_col" not in by_key["panel_2"]


def test_generate_plot_config_shape():
    cfg = _DesignerHost(_designed_panels(), 3, 2)._generate_plot_config()

    assert cfg["panel_1"] == {"role": "time", "y_column": "BX"}
    assert cfg["panel_3"] == {"role": "not-time", "x_column": "X",
                              "y_column": "Y"}
    # The designer does put a labels entry in plot_config.  It is inert
    # at runtime (generate_plot_fn skips keys with no axes) and the
    # driver emitter skips it by role -- pinned here so a reader knows
    # it is expected, not an oversight.
    assert cfg["labels"]["role"] == "labels"


# =====================================================================
# 3 -- THE W2 A/B: the same box drag, on a designer-built layout
# =====================================================================

@pytest.fixture
def crossplot_labeler(tmp_path):
    """A real labeler whose layout came from the real designer emitter.

    The constructor builds no GUI (that is `run()`), so this needs no Tk.
    """
    idx = pd.date_range("2021-03-01", periods=120, freq="30s")
    df = pd.DataFrame({
        "BX": np.linspace(0.0, 10.0, 120),
        "n_linear": np.linspace(1.0, 2.0, 120),
        "X": np.linspace(-60.0, -20.0, 120),
        "Y": np.linspace(-9.0, 9.0, 120),
    }, index=idx)

    spec = _DesignerHost(_designed_panels(), 3, 2)._generate_layout_spec()
    cfg = _DesignerHost(_designed_panels(), 3, 2)._generate_plot_config()

    app = TimeIntervalLabeler(
        df=df,
        plot_fn=generate_plot_fn(cfg),
        layout_spec=spec,
        autosave_folder=str(tmp_path),
        window=pd.Timedelta("1h"),
    )
    app.t0 = df.index[0]
    app.t1 = df.index[-1]
    return app


def test_a_designer_built_crossplot_selects_by_dataframe_filter(
        crossplot_labeler):
    """THE PACK'S HEADLINE A/B.

    `_try_dataframe_box_filter` returns None to mean "no column mapping,
    use the artist scan".  Measured before this pack, on a layout built
    by this very method: None.  On a hand-written driver layout carrying
    x_col/y_col, the same call over the same data returned one exact
    interval.  Wizard-designed cross-plots were silently on the worse
    path.  Now they are not.
    """
    app = crossplot_labeler

    # A box over the first quarter of the X/Y track.
    x_lo, x_hi = -60.0, -50.0
    y_lo, y_hi = -9.0, 9.0
    got = app._try_dataframe_box_filter("panel_3", x_lo, x_hi, y_lo, y_hi)

    assert got is not None, (
        "None means the labeler fell back to scanning drawn artists -- "
        "which is what happened before x_col/y_col were emitted")
    assert len(got) == 1
    start, end = got[0]

    # The exact filter's answer, computed independently from the frame.
    mask = ((app.df["X"] >= x_lo) & (app.df["X"] <= x_hi)
            & (app.df["Y"] >= y_lo) & (app.df["Y"] <= y_hi))
    expected = app.df.index[mask]
    assert start == expected[0]
    assert end == expected[-1]
    assert len(expected) < len(app.df)      # a real subset, not everything


def test_a_time_panel_is_unaffected_by_the_new_keys(crossplot_labeler):
    """Time areas carry no x_col/y_col, so the box filter still declines
    them and the time-axis path is untouched."""
    app = crossplot_labeler
    assert app._try_dataframe_box_filter("panel_1", 0, 1, 0, 1) is None


# =====================================================================
# 4 -- W7's runtime half: yscale
# =====================================================================

def test_generate_plot_fn_sets_a_log_y_scale_on_a_time_panel():
    plot_fn = generate_plot_fn(
        {"p": {"role": "time", "y_column": "n", "yscale": "log"}})
    idx = pd.date_range("2021-03-01", periods=10, freq="1min")
    df = pd.DataFrame({"n": np.linspace(1.0, 100.0, 10)}, index=idx)

    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    plot_fn({"p": ax}, df, idx[0], idx[-1])
    assert ax.get_yscale() == "log"


def test_generate_plot_fn_sets_a_log_y_scale_on_a_cross_plot():
    plot_fn = generate_plot_fn(
        {"p": {"role": "not-time", "x_column": "x", "y_column": "y",
               "yscale": "log"}})
    idx = pd.date_range("2021-03-01", periods=10, freq="1min")
    df = pd.DataFrame({"x": np.linspace(1.0, 10.0, 10),
                       "y": np.linspace(1.0, 100.0, 10)}, index=idx)

    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    plot_fn({"p": ax}, df, idx[0], idx[-1])
    assert ax.get_yscale() == "log"


def test_an_absent_yscale_leaves_the_axis_alone():
    """Every plot_config written before this pack must render exactly as
    it did."""
    plot_fn = generate_plot_fn({"p": {"role": "time", "y_column": "n"}})
    idx = pd.date_range("2021-03-01", periods=10, freq="1min")
    df = pd.DataFrame({"n": np.linspace(1.0, 100.0, 10)}, index=idx)

    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    plot_fn({"p": ax}, df, idx[0], idx[-1])
    assert ax.get_yscale() == "linear"
