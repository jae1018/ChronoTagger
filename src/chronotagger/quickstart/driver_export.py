"""Driver-file emitter for the ChronoTagger quick-start wizard.

The wizard's product used to be a live `TimeIntervalLabeler` in RAM and
nothing else: closing it meant re-walking every screen by hand.  This
module turns the same collected state into a **driver file** -- a small,
self-contained Python script that loads the data, defines the layout and
the plot function, and launches the labeler.  The file, not the RAM
object, is the thing the user keeps.

Two functions, no dialogs and no Tk:

    text = generate_driver(config, data_path=..., ...)   # -> str
    write_driver(text, path)                             # -> Path

`generate_driver` is pure: same inputs, same bytes.  Every file-choosing
and file-writing affordance belongs to the caller (the wizard), so this
module can be exercised head-lessly and pinned with golden files.

WHAT IT CAN EXPRESS (and deliberately no more)
----------------------------------------------
Exactly what `chronotagger.labeler.utils.plot_generator.generate_plot_fn`
can express at runtime: one y-column per time panel, one (x, y) pair per
cross-plot, line-or-scatter style, colour, axis labels, title, grid, a
log/linear y scale, and `equal_aspect` on a cross-plot.  The emitted
`plot_fn` is the source-code twin of the runtime one -- same panel keys,
same drawing calls, same defaults, and the SAME GUARDS: a missing column
or an empty window costs one panel and a red notice, never the figure.

Richer figures -- multi-trace panels, spectrograms, twin axes, reference
lines, overlays -- are NOT emitted.  They are what the `[YOURS]` blocks
in the generated file are for: the driver is a starting point you own,
not a round-tripped configuration.

ONE PANE.  A driver file describes a single pane.  A multi-tab wizard
session is refused BY NAME rather than half-emitted; emitting a pane list
is a Pack 8 decision (one driver per tab, or one driver carrying a
`PANES` list) that has not been made.  Pass one pane's
`(layout_spec, plot_config)` pair -- e.g. `wizard.pane_specs[0]`.

FORWARD-ONLY
------------
Nothing here parses a driver file, and nothing merges into one.  A
regeneration writes a complete fresh file; the `[GEN]` / `[YOURS]`
markers are comments telling a human which half survives a rewrite, not
machine-readable regions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

__all__ = ["generate_driver", "write_driver"]


# The labeler's own defaults, spelled out because a driver file states
# its whole configuration rather than inheriting it (app.py:199-207).
DEFAULT_CLASSES: List[str] = ["UNKNOWN", "label_1", "label_2"]
DEFAULT_COLORS: List[str] = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc949", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ac",
]

SUPPORTED_FORMATS = ("csv", "parquet")

# `pd.to_datetime(int_column)` reads NANOSECONDS.  A driver whose epoch
# is in any other unit and does not say so silently relocates the whole
# dataset -- measured: a microsecond epoch lands in 1970 and validates
# clean.  So an epoch column must name its unit, and these are the units
# `pd.to_datetime` accepts for one.
EPOCH_UNITS = ("s", "ms", "us", "ns")

# `autosave_folder="."` is the constructor default and it drops a
# fingerprinted JSON plus a log file into whatever directory the user
# happened to launch from.  A generated driver always names a folder.
# These are the spellings `os.path.normpath` cannot collapse for us.
_BARE_CWD_TOKENS = {"$PWD", "%CD%", "$(pwd)", "`pwd`"}

# The roles the runtime generator actually draws.  Anything else it
# ignores in silence, so the emitter emits nothing for it -- same
# figure, same non-behaviour.
_DRAWN_ROLES = ("time", "not-time")


# ---------------------------------------------------------------------
# literal helpers -- every emitted literal goes through one of these, so
# the output is ASCII by construction even when a column name is not,
# and a value that cannot be spelled as a literal is refused rather than
# written out as an unbound name
# ---------------------------------------------------------------------

_LITERAL_TYPES = (str, bool, int, float, type(None))


def _plain(value: Any) -> Any:
    """Coerce a value to something whose `ascii()` is a bare literal.

    `ascii()` is `repr()`, and under numpy 2 `repr(np.int64(3))` is
    `'np.int64(3)'` -- syntactically valid Python naming a module the
    generated driver never imports, so the file compiles and dies on
    import with `NameError: name 'np' is not defined`.  `pd.Timestamp`
    and `pd.Timedelta` repr the same way.

    Nothing in the wizard produces such a value today (`tk.IntVar.get()`
    and `PanelConfig.row` are plain ints), but `generate_driver` is
    advertised as accepting the shapes this package hands around, and a
    layout built from `np.arange` or `DataFrame.shape` is the normal
    idiom for this user base.  Coerce what has an obvious plain form;
    `_lit` refuses the rest.
    """
    if isinstance(value, Mapping):
        return {_plain(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    # numpy scalars carry .item(); plain int/float/str/bool do not.  The
    # check must come FIRST, because np.float64 IS a float subclass and
    # np.str_ IS a str subclass -- an isinstance shortcut would let them
    # straight through to their np-prefixed repr.
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes)):
        try:
            return _plain(item())
        except Exception:
            return value
    return value


def _lit(value: Any) -> str:
    """A Python literal for `value`, guaranteed ASCII.

    `ascii()` rather than `repr()`: repr renders a non-ASCII character
    literally, and this package's source rule is ASCII-only.  A column
    named with a Greek letter therefore lands as an escape sequence that
    still compares equal to the real name.

    Plain strings get double quotes, because the generated file is meant
    to read like something a person wrote and the rest of the template
    uses them.  Anything that would need escaping falls back to
    `ascii()`, whose single-quoted form is always correct.
    """
    value = _plain(value)
    if isinstance(value, (list, dict)):
        return ascii(value)
    if not isinstance(value, _LITERAL_TYPES):
        raise ValueError(
            "cannot emit %r as a Python literal: a %s has no plain "
            "literal form, and writing its repr() would put an unbound "
            "name in the generated file. Convert it first."
            % (value, type(value).__name__))
    if (isinstance(value, str)
            and value.isascii()
            and value.isprintable()
            and '"' not in value
            and "'" not in value
            and "\\" not in value):
        return '"%s"' % value
    return ascii(value)


def _comment(value: Any) -> str:
    """Text safe to drop into a generated COMMENT.

    Column names reach comments too ("# [GEN] panel_1: <column>"), and a
    comment is source like any other line: one Greek letter in a CSV
    header would otherwise put a non-ASCII byte in the file.  Strip the
    quotes off `ascii()` and what is left is the escaped, readable form.
    """
    return ascii(str(_plain(value)))[1:-1]


def _path_lit(path: Any) -> str:
    """A path literal: a raw string when that is unambiguous, else an
    ordinary escaped literal.

    Windows paths read far better as `r"C:\\data\\x.parquet"` than as
    `'C:\\\\data\\\\x.parquet'`, but a raw string cannot end in a
    backslash and cannot contain the quote character, and it must not
    carry non-ASCII bytes.  When any of that bites, fall back.
    """
    text = str(path)
    if (text
            and text.isascii()
            and '"' not in text
            and "\n" not in text
            and "\r" not in text
            and not text.endswith("\\")):
        return 'r"%s"' % text
    return _lit(text)


def _timedelta_lit(value: Union[str, pd.Timedelta]) -> str:
    """`pd.Timedelta("...")` from a string or a Timedelta.

    A caller-supplied string is emitted verbatim, so `"30min"` stays
    `"30min"` and the generated file reads the way the user typed it.  A
    Timedelta is emitted as `str(td)` (`'0 days 00:30:00'`), which
    pandas parses back to exactly the same value.

    The string goes through `_lit`, not into a hand-built `"%s"`: pandas
    accepts unit spellings that are not ASCII (`"1\\xb5s"` is a legal
    microsecond), and the ASCII guarantee has to hold for the RETURNED
    TEXT, not only for what `write_driver` will accept.
    """
    if isinstance(value, str):
        text = value
        # Validate now, in the emitter, rather than at the user's first
        # run of the generated file.
        pd.Timedelta(text)
    else:
        text = str(pd.Timedelta(value))
    return "pd.Timedelta(%s)" % _lit(text)


def _dict_lit(mapping: Mapping[str, str], indent: str = "    ") -> str:
    """A multi-line dict literal, one key per line, stable order."""
    if not mapping:
        return "{}"
    rows = ["{"]
    for key, value in mapping.items():
        rows.append("%s%s: %s," % (indent, _lit(key), _lit(value)))
    rows.append("}")
    return "\n".join(rows)


def _layout_lit(layout_spec: Mapping[str, Any]) -> str:
    """A readable multi-line literal for the layout_spec.

    Hand-written drivers put one area per line; a single `repr()` of the
    whole dict is one 400-character line nobody can edit.  Scalar keys
    come first, then `areas`.
    """
    rows = ["{"]
    for key, value in layout_spec.items():
        if key == "areas":
            continue
        rows.append("    %s: %s," % (_lit(key), _lit(value)))
    rows.append("    %s: [" % _lit("areas"))
    for area in layout_spec.get("areas", []):
        rows.append("        %s," % _area_lit(area))
    rows.append("    ],")
    rows.append("}")
    return "\n".join(rows)


def _area_lit(area: Mapping[str, Any]) -> str:
    """One area dict on one line, keys in a fixed, readable order."""
    order = ["key", "row", "col", "rowspan", "colspan", "role",
             "x_col", "y_col"]
    seen = set()
    parts = []
    for name in order:
        if name in area:
            seen.add(name)
            parts.append("%s: %s" % (_lit(name), _lit(area[name])))
    for name, value in area.items():
        if name not in seen:
            parts.append("%s: %s" % (_lit(name), _lit(value)))
    return "{%s}" % ", ".join(parts)


# ---------------------------------------------------------------------
# input normalisation
# ---------------------------------------------------------------------

def _is_pane_list(config: Any) -> bool:
    """Is this a LIST OF PANES rather than one pane's config pair?

    `wizard.pane_specs` is a list of `{'title', 'layout_spec',
    'plot_config'}` dicts, one per tab.  Handed straight to
    `generate_driver` it used to unpack as if it were a
    `(layout_spec, plot_config)` pair and die complaining that the first
    pane had no 'areas' -- a misleading error at exactly the seam where a
    Pack 8 implementer meets it.
    """
    if isinstance(config, (Mapping, str, bytes)):
        return False
    if not isinstance(config, Sequence):
        return False
    return bool(config) and all(
        isinstance(entry, Mapping)
        and ("layout_spec" in entry or "plot_config" in entry)
        for entry in config)


def _split_config(config: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Accept either shape the package already hands around.

    `build_layout()` (the grid designer) and `vertical_stack_config()`
    (the preset) both return a `(layout_spec, plot_config)` tuple, and
    the tab planner stores the same pair under those two names.  Both
    are accepted so no caller has to repackage.
    """
    if _is_pane_list(config):
        raise ValueError(
            "multi-pane sessions cannot be emitted yet (planned: Pack 8); "
            "got %d panes. Pass ONE pane's (layout_spec, plot_config) "
            "pair -- e.g. wizard.pane_specs[0] -- to emit a driver for "
            "that pane." % len(config))

    if isinstance(config, Mapping):
        if "layout_spec" not in config or "plot_config" not in config:
            raise ValueError(
                "config mapping must carry both 'layout_spec' and "
                "'plot_config' keys; got %s" % sorted(config))
        layout_spec = config["layout_spec"]
        plot_config = config["plot_config"]
    elif isinstance(config, Sequence) and len(config) == 2:
        layout_spec, plot_config = config
    else:
        raise TypeError(
            "config must be a (layout_spec, plot_config) pair or a "
            "mapping carrying those two keys; got %r" % type(config))

    if not isinstance(layout_spec, Mapping):
        raise TypeError("layout_spec must be a mapping")
    if not isinstance(plot_config, Mapping):
        raise TypeError("plot_config must be a mapping")
    if not layout_spec.get("areas"):
        raise ValueError("layout_spec has no 'areas'")
    return dict(layout_spec), dict(plot_config)


def _drawable_areas(layout_spec: Mapping[str, Any],
                    plot_config: Mapping[str, Any]) -> List[Tuple[str, Dict]]:
    """The (key, panel_config) pairs the plot function must draw.

    The `labels` area is skipped BY ROLE.  The grid designer's
    `_generate_plot_config` puts a `labels` entry in plot_config, and the
    labeler holds the labels strip separately (`strip_ax`) rather than in
    `axs` -- so emitting a block for it produced `axs['labels']` ->
    KeyError, measured on the old `generate_plot_code`.  Driving the loop
    from the LAYOUT and filtering on role removes that class of bug.

    Any role the runtime generator does not draw is skipped for the same
    reason: `generate_plot_fn` dispatches `time` / `not-time` and does
    nothing at all otherwise, so emitting a scatter for `role="spectro"`
    would be the emitter inventing a figure B never renders.

    A drawable panel whose column names are missing is REFUSED here.
    `df[None]` compiles and dies on the first render; refusing at
    emission is what this module does everywhere else.
    """
    out: List[Tuple[str, Dict]] = []
    for area in layout_spec.get("areas", []):
        key = area.get("key")
        if not key:
            continue
        cfg = dict(plot_config.get(key) or {})
        role = str(cfg.get("role") or area.get("role") or "time")
        if role not in _DRAWN_ROLES:
            continue
        cfg["role"] = role
        if role == "time":
            needed = {"y_column": cfg.get("y_column")}
        else:
            needed = {"x_column": cfg.get("x_column"),
                      "y_column": cfg.get("y_column")}
        blank = sorted(name for name, col in needed.items()
                       if not isinstance(col, str) or not col)
        if blank:
            raise ValueError(
                "panel %r has role %r but no %s in plot_config, so there "
                "is no column to draw. Fill it in, drop the area, or give "
                "the area a role the generator does not draw."
                % (key, role, " or ".join(blank)))
        out.append((key, cfg))
    if not out:
        raise ValueError(
            "layout_spec contains no drawable panels (only a labels "
            "area?); nothing to plot")
    return out


# ---------------------------------------------------------------------
# section emitters
# ---------------------------------------------------------------------

_HEADER = '''\
# ============================================================
# [GEN] ChronoTagger driver -- generated by the Quick-Start Wizard.
# [GEN]
# [GEN] Run it with:   python %(basename)s
# [GEN]
# [GEN] Every line below is either [GEN] -- written from what the
# [GEN] wizard collected -- or [YOURS] -- scaffolding written ONCE for
# [GEN] you to fill in.  The markers are comments for a human reader.
# [GEN] There is NO merge-on-regenerate: re-running the wizard writes a
# [GEN] complete fresh file over this one.  Once you have edited a
# [GEN] [YOURS] block, this file is the product -- keep editing it here
# [GEN] rather than regenerating.
# [GEN]
# [GEN] HAND-EDIT WARNING -- autosave identity.  ChronoTagger names its
# [GEN] autosave after a fingerprint of the frame it was given: the
# [GEN] sorted column names, the first and last index values, and the
# [GEN] row count.  Change what load_dataframe() returns -- drop a
# [GEN] column, filter rows, subset a time range -- and that
# [GEN] fingerprint MOVES, so an autosave written before the edit is no
# [GEN] longer offered for recovery.  Nothing is lost from disk, but it
# [GEN] will not be found automatically.  SOURCE_NAME below is the
# [GEN] stable human-readable identity; the fingerprint is not.
# ============================================================

import pandas as pd                                        # [GEN]

from chronotagger.labeler import TimeIntervalLabeler       # [GEN]
'''


def _data_section(data_path: Any, fmt: str, time_column: Optional[str],
                  time_is_epoch: bool, time_unit: Optional[str]) -> str:
    reader = "pd.read_csv" if fmt == "csv" else "pd.read_parquet"
    rows = [
        "# ---------------- DATA ------------------------------------- [GEN]",
        "# The driver OWNS data loading, so `python this_file.py` is",
        "# self-contained.  ALL columns are read -- including the non-numeric",
        "# ones the wizard cannot plot -- because your own code below may",
        "# well want them.",
        "DATA_PATH = %s" % _path_lit(data_path),
    ]
    if time_column is not None:
        rows.append("TIME_COLUMN = %s" % _lit(time_column))
    if time_is_epoch:
        rows.append("TIME_UNIT = %s" % _lit(time_unit))
    rows.append("")
    rows.append("")
    rows.append("def load_dataframe():")
    rows.append("    df = %s(DATA_PATH)" % reader)
    if time_column is None:
        rows.append("    if not isinstance(df.index, pd.DatetimeIndex):")
        rows.append("        raise TypeError(")
        rows.append('            "%s carries no DatetimeIndex. Set a time '
                    'column here "')
        rows.append('            "and convert it, e.g. df.index = '
                    'pd.to_datetime(df[col])."')
        rows.append("            % DATA_PATH)")
        rows.append('    df.index.name = "time"')
        rows.append("    return df.sort_index()")
    else:
        if time_is_epoch:
            rows.append("    # TIME_COLUMN counts from the epoch, so the unit is"
                        " stated.")
            rows.append("    # A BARE pd.to_datetime() reads every integer as"
                        " NANOSECONDS,")
            rows.append("    # which moves a microsecond epoch to 1970 without"
                        " complaining")
            rows.append("    # -- the index is still a DatetimeIndex, just the"
                        " wrong one.")
            rows.append("    # Edit TIME_UNIT above, not this call.")
            rows.append("    df.index = pd.to_datetime(df[TIME_COLUMN], "
                        "unit=TIME_UNIT)")
        else:
            rows.append("    # TIME_COLUMN holds dates/timestamps, so no unit is"
                        " involved.")
            rows.append("    # If yours is really an INTEGER epoch, regenerate it"
                        " as one")
            rows.append("    # (or add unit=\"s\"/\"ms\"/\"us\"/\"ns\" below):"
                        " a bare call reads")
            rows.append("    # integers as NANOSECONDS and relocates the data to"
                        " 1970.")
            rows.append("    df.index = pd.to_datetime(df[TIME_COLUMN])")
        rows.append('    df.index.name = "time"')
        rows.append("    df = df.drop(columns=[TIME_COLUMN]).sort_index()")
        rows.append("    return df")
    return "\n".join(rows)


def _schema_section(classes: Sequence[str],
                    colors: Mapping[str, str]) -> str:
    rows = [
        "# ---------------- LABEL SCHEMA ----------------------------- [GEN]",
        "# Rename, add or recolour freely -- this list IS the schema the",
        "# labeler opens with.  Intervals already saved under a class you",
        "# delete keep their old name in the session file.",
        "CLASSES = [%s]" % ", ".join(_lit(c) for c in classes),
        "COLORS = %s" % _dict_lit(colors),
    ]
    return "\n".join(rows)


def _layout_section(layout_spec: Mapping[str, Any]) -> str:
    rows = [
        "# ---------------- LAYOUT ----------------------------------- [GEN]",
        "# x_col / y_col on a cross-plot area are what let a box drag on",
        "# that panel select rows by DATAFRAME FILTER instead of by",
        "# scanning drawn artists.  Keep them in step with plot_fn below.",
        "LAYOUT = %s" % _layout_lit(layout_spec),
    ]
    return "\n".join(rows)


_PLOT_HELPERS = '''\
# ---------------- PLOT FUNCTION ---------------------------- [GEN]
def _clear_panel(ax):                                      # [GEN]
    """Drop the data artists WITHOUT calling ax.clear().

    The labeler has already run ax.clear(), xaxis_date() and
    set_xlim(t0, t1) before it calls plot_fn.  Calling clear() again
    here throws away the shared-axis links and the datetime unit
    converter it just installed, and the labeler's set_xlim then reaches
    only one axis -- the others keep a ~50-year auto-fit range and draw
    their line invisibly.  Removing artists is the safe half.
    """
    for artist in (list(ax.lines) + list(ax.collections)
                   + list(ax.patches) + list(ax.texts)):
        artist.remove()


def _have(ax, df, *columns):                               # [GEN]
    """The wizard generator's own guards, in source form.

    chronotagger.labeler.utils.plot_generator.generate_plot_fn draws a
    red "Column not found" notice on a panel it cannot draw and moves on
    to the next one, rather than raising.  This driver does the same, so
    editing load_dataframe() or the [YOURS] hook below -- dropping a
    column, renaming one, filtering every row away -- costs you ONE
    panel and a visible message, not the whole figure.
    """
    if len(df.index) == 0:
        ax.text(0.5, 0.5, "No data in window", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return False
    for column in columns:
        if column not in df.columns:
            ax.text(0.5, 0.5, "Error: Column '%s' not found" % column,
                    ha="center", va="center", transform=ax.transAxes,
                    color="red")
            return False
    return True
'''


def _panel_block(key: str, cfg: Mapping[str, Any]) -> List[str]:
    role = cfg.get("role", "time")
    rows: List[str] = []
    if role == "time":
        y_col = cfg.get("y_column")
        rows.append("    # [GEN] %s: %s" % (_comment(key), _comment(y_col)))
        rows.append("    ax = axs[%s]" % _lit(key))
        rows.append("    _clear_panel(ax)")
        rows.append("    if _have(ax, df, %s):" % _lit(y_col))
        style = cfg.get("style", "line")
        color = cfg.get("color", "#1f77b4")
        if style == "scatter":
            rows.append("        ax.scatter(df.index, df[%s], s=3, c=%s, "
                        "alpha=0.7)" % (_lit(y_col), _lit(color)))
        else:
            rows.append("        ax.plot(df.index, df[%s], color=%s, "
                        "linewidth=1.0)" % (_lit(y_col), _lit(color)))
        yscale = cfg.get("yscale")
        if yscale:
            rows.append("        ax.set_yscale(%s)" % _lit(yscale))
        rows.append("        ax.set_ylabel(%s, fontsize=9)"
                    % _lit(cfg.get("ylabel", y_col)))
        if cfg.get("grid", True):
            rows.append("        ax.grid(alpha=0.3, linewidth=0.5)")
        if cfg.get("title"):
            rows.append("        ax.set_title(%s, fontsize=10)"
                        % _lit(cfg["title"]))
    else:
        x_col = cfg.get("x_column")
        y_col = cfg.get("y_column")
        rows.append("    # [GEN] %s: %s vs %s -- drawn in df.index order, so"
                    % (_comment(key), _comment(x_col), _comment(y_col)))
        rows.append("    # [GEN] a box drag here maps points back to time")
        rows.append("    ax = axs[%s]" % _lit(key))
        rows.append("    _clear_panel(ax)")
        rows.append("    if _have(ax, df, %s, %s):"
                    % (_lit(x_col), _lit(y_col)))
        style = cfg.get("style", "scatter")
        color = cfg.get("color", "#2ca02c")
        if style == "line":
            rows.append("        ax.plot(df[%s], df[%s], color=%s, "
                        "linewidth=1.0)"
                        % (_lit(x_col), _lit(y_col), _lit(color)))
        else:
            rows.append("        ax.scatter(df[%s], df[%s], s=5, c=%s, "
                        "alpha=0.6)"
                        % (_lit(x_col), _lit(y_col), _lit(color)))
        yscale = cfg.get("yscale")
        if yscale:
            rows.append("        ax.set_yscale(%s)" % _lit(yscale))
        rows.append("        ax.set_xlabel(%s, fontsize=9)"
                    % _lit(cfg.get("xlabel", x_col)))
        rows.append("        ax.set_ylabel(%s, fontsize=9)"
                    % _lit(cfg.get("ylabel", y_col)))
        if cfg.get("grid", True):
            rows.append("        ax.grid(alpha=0.3, linewidth=0.5)")
        if cfg.get("title"):
            rows.append("        ax.set_title(%s, fontsize=10)"
                        % _lit(cfg["title"]))
        if cfg.get("equal_aspect", False):
            rows.append('        ax.set_aspect("equal", adjustable="box")')
    return rows


_YOURS_PLOT = '''\
    # ------------- YOUR ADDITIONS ------------------------- [YOURS]
    # Anything you draw here is a VISUAL AID: it renders, but box
    # selection never picks it up, because selection reads the dataframe
    # (or the data artists the blocks above drew), not your annotations.
    # Reference lines, model boundaries, extra traces and shaded bands
    # all belong here.  Example:
    #
    #     from geospacefronts import shue_mp
    #     x_mp, y_mp = shue_mp(...)
    #     axs[%s].plot(x_mp, y_mp, "k--", lw=1)
    # ------------------------------------------------------------'''


def _plot_section(panels: Sequence[Tuple[str, Dict[str, Any]]]) -> str:
    rows = [_PLOT_HELPERS.rstrip("\n"), "", "",
            "def plot_fn(axs, df, t0, t1):"]
    for i, (key, cfg) in enumerate(panels):
        if i:
            rows.append("")
        rows.extend(_panel_block(key, cfg))
    # Point the [YOURS] example at a panel this layout really has -- a
    # copy-pasteable example beats a plausible-looking one.
    example_key = panels[0][0]
    for key, cfg in panels:
        if cfg.get("role") != "time":
            example_key = key
            break
    rows.append("")
    rows.append(_YOURS_PLOT % _lit(example_key))
    return "\n".join(rows)


def _launch_section(window: Any, step: Any, autosave_folder: Any,
                    source_name: str, decimate: bool) -> str:
    rows = [
        "# ---------------- LAUNCH ----------------------------------- [GEN]",
        "AUTOSAVE_FOLDER = %s" % _path_lit(autosave_folder),
        "SOURCE_NAME = %s" % _lit(source_name),
        "",
        "",
        'if __name__ == "__main__":',
        "    df = load_dataframe()",
        "",
        "    # ------------- YOUR PRE-LAUNCH HOOK ------------------- [YOURS]",
        "    # Filter to one spacecraft, subset a time range, derive",
        "    # preloaded intervals from a label column -- anything that",
        "    # shapes the frame the labeler sees.  Read the autosave",
        "    # warning at the top of this file before you change which",
        "    # rows or columns survive.  Example:",
        "    #",
        '    #     df = df[df["probe"] == 0].drop(columns=["probe"])',
        "    # ------------------------------------------------------------",
        "",
        "    app = TimeIntervalLabeler(",
        "        df=df,",
        "        plot_fn=plot_fn,",
        "        layout_spec=LAYOUT,",
        "        classes=CLASSES,",
        "        class_colors=COLORS,",
        "        window=%s," % _timedelta_lit(window),
        "        step=%s," % _timedelta_lit(step),
        "        autosave_folder=AUTOSAVE_FOLDER,",
        "        source_name=SOURCE_NAME,",
        "        decimate=%s," % ("True" if decimate else "False"),
        "    )",
        "    app.run()",
    ]
    return "\n".join(rows)


# ---------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------

def generate_driver(
    config: Any,
    *,
    data_path: Any,
    fmt: str = "csv",
    time_column: Optional[str] = None,
    time_is_epoch: bool = False,
    time_unit: Optional[str] = None,
    classes: Optional[Sequence[str]] = None,
    colors: Optional[Mapping[str, str]] = None,
    window: Union[str, pd.Timedelta] = "30min",
    step: Union[str, pd.Timedelta] = "15min",
    autosave_folder: Any = "./chronotagger_autosave",
    source_name: Optional[str] = None,
    decimate: bool = True,
    file_name: str = "chronotagger_driver.py",
) -> str:
    """Return the complete source of a runnable ChronoTagger driver.

    Args:
        config: `(layout_spec, plot_config)`, or a mapping carrying both
            under those names.  Exactly what `build_layout()` and
            `vertical_stack_config()` return.  A LIST of pane specs is
            refused by name -- one driver describes one pane.
        data_path: Path to the data file the driver will load.
        fmt: "csv" or "parquet".
        time_column: Column to convert into the DatetimeIndex.  `None`
            means the file already carries one (parquet only).
        time_is_epoch: True when `time_column` holds INTEGERS counting
            from the epoch.  `pd.to_datetime` reads a bare integer
            column as nanoseconds, so an epoch column must name its
            unit; saying so here is what makes `time_unit` required
            instead of guessed.
        time_unit: "s", "ms", "us" or "ns".  REQUIRED when
            `time_is_epoch`; rejected otherwise, because a unit means
            nothing for a column of date strings.
        classes / colors: The label schema.  Defaults mirror the
            labeler's own constructor defaults.
        window / step: `pd.Timedelta`s, or strings pandas can parse.
        autosave_folder: Where autosaves and the forensic log go.  A
            bare CWD is refused -- a generated driver must not scatter
            state into whatever directory it happens to be run from.
        source_name: Stable human identity for the dataset.  Defaults to
            the data file's stem.
        decimate: The constructor's draw-only decimation flag.
        file_name: Name used in the generated `python <name>` line only.

    Returns:
        ASCII source text with "\\n" line endings.  `write_driver`
        normalises to the platform/repo convention on the way out.
    """
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            "fmt must be one of %s; got %r" % (SUPPORTED_FORMATS, fmt))
    if fmt == "csv" and time_column is None:
        raise ValueError(
            "a csv driver needs time_column: pd.read_csv produces a "
            "RangeIndex, so there is nothing to fall back on")

    if time_is_epoch:
        if time_column is None:
            raise ValueError(
                "time_is_epoch needs a time_column to convert")
        if time_unit is None:
            raise ValueError(
                "time_is_epoch requires time_unit (one of %s). "
                "pd.to_datetime reads a bare integer column as "
                "NANOSECONDS, so an unlabelled microsecond epoch "
                "silently relocates the whole dataset to 1970 and still "
                "validates clean. Say which unit it is."
                % (EPOCH_UNITS,))
        if time_unit not in EPOCH_UNITS:
            raise ValueError(
                "time_unit must be one of %s; got %r"
                % (EPOCH_UNITS, time_unit))
    elif time_unit is not None:
        raise ValueError(
            "time_unit is only meaningful for an integer epoch column; "
            "pass time_is_epoch=True with it, or drop it if "
            "%r holds dates rather than integers" % (time_column,))

    folder = str(autosave_folder).strip()
    if folder in _BARE_CWD_TOKENS or os.path.normpath(folder or ".") == ".":
        raise ValueError(
            "autosave_folder must name a folder, not the process CWD "
            "(got %r, which resolves to the CWD).  A driver that "
            "autosaves into whatever directory it was launched from "
            "loses its own recovery file." % (autosave_folder,))

    layout_spec, plot_config = _split_config(config)
    panels = _drawable_areas(layout_spec, plot_config)

    classes = list(classes) if classes else list(DEFAULT_CLASSES)
    if colors:
        colors = dict(colors)
    else:
        colors = {
            name: DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            for i, name in enumerate(classes)
        }
    if source_name is None:
        source_name = Path(str(data_path)).stem

    sections = [
        _HEADER.rstrip("\n") % {"basename": _comment(file_name)},
        _data_section(data_path, fmt, time_column, time_is_epoch, time_unit),
        _schema_section(classes, colors),
        _layout_section(layout_spec),
        _plot_section(panels),
        _launch_section(window, step, autosave_folder, source_name,
                        decimate),
    ]
    return "\n\n\n".join(sections) + "\n"


def write_driver(text: str, path: Any, *, newline: str = "\r\n") -> Path:
    """Write `text` to `path`, normalising line endings.  No dialogs.

    The text is normalised to "\\n" first and then re-emitted with
    `newline`, so a driver generated on one platform and written on
    another carries one consistent convention rather than a mixture.
    `encoding="ascii"` is the enforcement point for the package's
    ASCII-only rule: a non-ASCII byte raises here instead of crashing a
    cp1252 console later.  `generate_driver` already guarantees ASCII, so
    this catches hand-assembled text, not its own output.
    """
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="ascii", newline="") as fh:
        fh.write(normalised.replace("\n", newline))
    return target
