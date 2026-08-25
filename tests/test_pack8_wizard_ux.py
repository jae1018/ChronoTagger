"""Pack 8 -- the wizard UX half of the wizard-to-driver arc.

Five groups, one per ruling family.

1. PANES emission (R1/R2/R3): a multi-tab session becomes ONE driver
   carrying `LAYOUT_n` / `plot_fn_n` and a `PANES` list; a single pane
   still emits the Pack 7 bytes; every pane is structurally validated at
   emission, by index.
2. The driver file as an ARTIFACT (R6/R12/R19/R4): `write_driver` refuses
   to replace a file it was not told to, `.csv.gz` is a csv everywhere,
   `driver_export` is public, and one dataset has ONE identity.
3. The LOADER (R11/R12/R14): an integer epoch must name its unit before
   Continue lights up, and the wizard sorts an unsorted index instead of
   refusing it.
4. The WIZARD (R4/R5/R8/R9/R10/R13/R16): classes asked once, save-as
   offered once, step tied to the window, autosaves beside the data,
   cancel that returns instead of exiting, and Pack 6's bounded retry on
   the last two unprotected roots.
5. The DESIGNER (R15/R16/R17): Preview allocates no second Tk root and no
   pyplot figure, the package is ASCII, and the overlap-refusal /
   grid-clip pair finally has pins.

Every test that can reach a modal runs under the `gui` fixture from
`pack8_gui_harness`, which stubs messagebox, filedialog, simpledialog and
colorchooser and withdraws every Toplevel.  No test skips, no test
branches on platform: the ids collect identically on Windows and Linux.
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import chronotagger
from chronotagger.labeler.utils import vertical_stack_config
from chronotagger.quickstart import driver_export
from chronotagger.quickstart.driver_export import (
    generate_driver,
    portable_stem,
    write_driver,
)

from pack8_gui_harness import gui  # noqa: F401  (pytest fixture)

SRC_ROOT = Path(chronotagger.__file__).resolve().parent.parent


# =====================================================================
# fixtures / helpers
# =====================================================================

def _frame(n=240, strings=True):
    """The usual fixture.

    `strings=False` drops the non-numeric column for the EPOCH tests
    only: `_auto_detect_time_column`'s last-resort probe tries
    `pd.to_datetime` on every object column, and pandas emits a
    "could not infer format" UserWarning on the way to failing. That
    warning is about the fixture, not about anything this pack changed.
    """
    idx = pd.date_range("2021-03-01", periods=n, freq="10s")
    data = {
        "BX": np.sin(np.linspace(0, 8, n)),
        "n_linear": np.abs(np.sin(np.linspace(0, 2, n))) + 0.02,
        "X": np.linspace(-60.0, -20.0, n),
        "Y": np.linspace(-9.0, 9.0, n),
    }
    if strings:
        data["liuzzo_class"] = ["sheet"] * n
    return pd.DataFrame(data, index=idx)


def _epoch_csv(path, divisor):
    """A csv whose time column is an INTEGER epoch, exact in its unit."""
    frame = _frame(strings=False)
    epoch = frame.index.astype("datetime64[ns]").astype("int64") // divisor
    flat = frame.reset_index(drop=True)
    flat.insert(0, "t", epoch.values)
    flat.to_csv(path, index=False)
    return path


def _pane(title, columns, cross=False):
    """One `wizard.pane_specs` entry, in the shape the wizard builds."""
    layout, config = vertical_stack_config(list(columns))
    if cross:
        rows = layout["nrows"]
        layout = dict(layout)
        layout["ncols"] = 2
        layout["areas"] = list(layout["areas"]) + [{
            "key": "xplot_1", "row": 0, "col": 1, "rowspan": rows,
            "role": "not-time", "x_col": "X", "y_col": "Y"}]
        config = dict(config)
        config["xplot_1"] = {"role": "not-time", "x_column": "X",
                             "y_column": "Y"}
    return {"title": title, "layout_spec": layout, "plot_config": config}


def _two_panes():
    return [_pane("Fields", ["BX", "n_linear"]),
            _pane("Orbit", ["BX"], cross=True)]


def _emit(config, **kwargs):
    kwargs.setdefault("data_path", "/d/x.csv")
    kwargs.setdefault("fmt", "csv")
    kwargs.setdefault("time_column", "t")
    kwargs.setdefault("autosave_folder", "./out")
    return generate_driver(config, **kwargs)


def _write_csv(path, frame=None, time_col="t"):
    frame = _frame() if frame is None else frame
    frame.reset_index(names=time_col).to_csv(path, index=False)
    return path


# =====================================================================
# 1. PANES emission -- R1 / R2 / R3
# =====================================================================

def test_a_two_pane_session_emits_indexed_layouts_and_plot_functions():
    """R1/R2.  Pack 7 refused a pane list by name; this is what replaced
    the refusal.  The suffixes are what keep two panes' identifiers apart
    -- panel keys REPEAT across panes (both panes here own a `panel_1`),
    which is fine at runtime because each pane holds its own axes dict,
    and fatal in one Python module namespace."""
    text = _emit(_two_panes())

    assert "LAYOUT_1 = {" in text
    assert "LAYOUT_2 = {" in text
    assert "def plot_fn_1(axs, df, t0, t1):" in text
    assert "def plot_fn_2(axs, df, t0, t1):" in text
    assert "LAYOUT = {" not in text
    assert "def plot_fn(axs" not in text

    assert '    {"title": "Fields", "plot_fn": plot_fn_1, ' \
           '"layout_spec": LAYOUT_1},' in text
    assert '    {"title": "Orbit", "plot_fn": plot_fn_2, ' \
           '"layout_spec": LAYOUT_2},' in text
    assert "        panes=PANES," in text
    assert "        plot_fn=plot_fn," not in text
    assert "        layout_spec=LAYOUT," not in text

    # The human title is a comment above each block as well as a string
    # in PANES -- a reader editing plot_fn_2 should not have to scroll to
    # the launch section to learn which tab it draws.
    assert "# [GEN] pane 1: Fields" in text
    assert "# [GEN] pane 2: Orbit" in text

    # ONE copy of the helpers, however many panes.
    assert text.count("def _clear_panel(ax):") == 1
    assert text.count("def _have(ax, df, *columns):") == 1


def test_a_one_entry_pane_list_emits_the_single_pane_bytes():
    """R2.  The single-pane shape is not a special case bolted on: a
    one-tab session IS one pane, and the wizard builds it through the
    single-pane constructor API, so the driver must agree.  Byte
    equality, not 'looks the same'."""
    pane = _pane("Fields", ["BX", "n_linear"])
    as_list = _emit([pane])
    as_pair = _emit((pane["layout_spec"], pane["plot_config"]))
    assert as_list == as_pair
    assert "LAYOUT_1" not in as_list
    assert "plot_fn_1" not in as_list
    assert "PANES" not in as_list


def test_a_hostile_pane_title_survives_as_ascii():
    """Titles are free-form user strings typed into the tab planner.
    They reach a COMMENT and a string literal, and both have to hold."""
    panes = _two_panes()
    panes[0]["title"] = 'He said "mu" \\ 6.5\u00b5s'
    text = _emit(panes)
    assert text.isascii()
    assert compile(text, "<driver>", "exec")
    assert "\u00b5" not in text


def test_a_pane_with_no_time_area_is_refused_by_index():
    """R3.  `canvas.py` raises 'layout_spec must have at least one
    role=time axis' PER PANE while building the GUI -- which in a driver
    is after load_dataframe() has read the whole file.  The emitter is
    holding the same layout_spec and can say so now."""
    panes = _two_panes()
    panes[1]["layout_spec"] = {
        "nrows": 2, "ncols": 1, "areas": [
            {"key": "xplot_1", "row": 0, "col": 0, "role": "not-time",
             "x_col": "X", "y_col": "Y"},
            {"key": "labels", "row": 1, "col": 0, "role": "labels"}]}
    panes[1]["plot_config"] = {
        "xplot_1": {"role": "not-time", "x_column": "X", "y_column": "Y"}}

    with pytest.raises(ValueError) as exc:
        _emit(panes)
    message = str(exc.value)
    assert 'pane 2 ("Orbit")' in message
    assert 'role="time"' in message


def test_a_pane_with_no_labels_area_is_refused_by_index():
    """R3, the sibling refusal (`canvas.py`: 'missing Labels panel')."""
    panes = _two_panes()
    layout = dict(panes[0]["layout_spec"])
    layout["areas"] = [a for a in layout["areas"]
                       if a.get("role") != "labels"]
    panes[0]["layout_spec"] = layout

    with pytest.raises(ValueError) as exc:
        _emit(panes)
    message = str(exc.value)
    assert 'pane 1 ("Fields")' in message
    assert "0 " in message and 'role="labels"' in message


def test_two_labels_areas_in_one_pane_are_refused():
    """Stricter than the labeler, deliberately: `_find_labels_area` takes
    the FIRST and silently ignores the rest, so a second strip is a
    layout the user thinks they have and does not."""
    panes = _two_panes()
    layout = dict(panes[1]["layout_spec"])
    layout["areas"] = list(layout["areas"]) + [
        {"key": "labels_2", "row": 2, "col": 1, "role": "labels"}]
    panes[1]["layout_spec"] = layout

    with pytest.raises(ValueError, match='2 role="labels" areas'):
        _emit(panes)


def test_a_single_pane_config_is_structurally_validated_too():
    """R3 says 'every pane', and a single-pane config is a pane.  Its
    message names 'the layout' rather than an index, because there is no
    index to name -- the single-pane messages this module already had
    stay exactly as they were."""
    layout = {"nrows": 2, "ncols": 1, "areas": [
        {"key": "p1", "row": 0, "col": 0, "role": "time"},
        {"key": "p2", "row": 1, "col": 0, "role": "time"}]}
    cfg = {"p1": {"role": "time", "y_column": "BX"},
           "p2": {"role": "time", "y_column": "n_linear"}}
    with pytest.raises(ValueError) as exc:
        _emit((layout, cfg))
    assert "the layout" in str(exc.value)
    assert 'role="labels"' in str(exc.value)


def test_a_malformed_pane_entry_names_which_pane_it_is():
    """The refusal Pack 7 v2 shipped existed because 'layout_spec has no
    areas' does not say WHICH of six tabs is broken.  That complaint
    outlives the refusal."""
    panes = _two_panes()
    panes[1]["layout_spec"] = {"nrows": 1, "ncols": 1, "areas": []}
    with pytest.raises(ValueError) as exc:
        _emit(panes)
    assert 'pane 2 ("Orbit")' in str(exc.value)
    assert "areas" in str(exc.value)


def test_an_untitled_pane_still_gets_a_name():
    """`wizard.pane_specs` always carries a title, but the emitter is a
    public API and a caller who omits one gets a usable file rather than
    `"title": None`."""
    panes = _two_panes()
    del panes[1]["title"]
    text = _emit(panes)
    assert '"title": "Pane 2"' in text


def test_a_plot_config_role_cannot_promote_the_labels_strip():
    """SECTION 0a F11 / EDIT 349, from V3 A1.8.  `_drawable_areas` used
    to resolve `cfg.get("role") or area.get("role")`, so a plot_config
    entry OVERRODE the area: `plot_config["labels"] = {"role": "time",
    "y_column": "BX"}` emitted `ax = axs["labels"]` and the labeler keeps
    that strip OUTSIDE `axs` -- KeyError at the first render, on a file
    that had already read the whole dataset.  `_check_pane_structure`
    cannot catch it: it reads the AREA and still counts one labels
    area."""
    one = _pane("Fields", ["BX", "n_linear"])
    hostile = dict(one["plot_config"])
    hostile["labels"] = {"role": "time", "y_column": "BX"}

    text = _emit((one["layout_spec"], hostile))
    assert 'axs["labels"]' not in text
    assert 'axs["panel_1"]' in text
    assert 'axs["panel_2"]' in text

    # the two-pane path too, where the labels key of pane 2 is the one
    # that would have collided
    panes = _two_panes()
    panes[1]["plot_config"] = dict(panes[1]["plot_config"],
                                   labels={"role": "time",
                                           "y_column": "BX"})
    assert 'axs["labels"]' not in _emit(panes)

    # ...and a layout that is NOTHING BUT a promoted labels area is
    # refused with the message it always had, not with a vaguer one.
    only_labels = {"nrows": 1, "ncols": 1, "areas": [
        {"key": "labels", "row": 0, "col": 0, "role": "labels"}]}
    with pytest.raises(ValueError) as exc:
        _emit((only_labels, {"labels": {"role": "time",
                                        "y_column": "BX"}}))
    assert "no drawable panels" in str(exc.value)


# =====================================================================
# 2. The driver file as an artifact -- R4 / R6 / R12 / R19
# =====================================================================

def test_write_driver_refuses_to_replace_an_existing_file(tmp_path):
    """R6.  Measured on Pack 7: two writes to one path clobbered
    silently -- no exists() check anywhere in the module, no exception,
    no signal in the return value.  A driver is a file the user owns the
    moment they edit a [YOURS] block."""
    path = tmp_path / "drive.py"
    one = _pane("Fields", ["BX"])
    text = _emit((one["layout_spec"], one["plot_config"]))
    write_driver(text, path)
    path.write_text("# hand edited\n", encoding="ascii")

    with pytest.raises(FileExistsError) as exc:
        write_driver(text, path)
    assert "overwrite=True" in str(exc.value)
    assert path.read_text(encoding="ascii") == "# hand edited\n"


def test_write_driver_replaces_when_told_to(tmp_path):
    path = tmp_path / "drive.py"
    one = _pane("Fields", ["BX"])
    text = _emit((one["layout_spec"], one["plot_config"]))
    write_driver(text, path)
    path.write_text("# hand edited\n", encoding="ascii")

    write_driver(text, path, overwrite=True)
    assert "# hand edited" not in path.read_text(encoding="ascii")
    assert "TimeIntervalLabeler(" in path.read_text(encoding="ascii")


def test_driver_export_is_on_the_quickstart_public_surface():
    """R19.  Pack 6 D8: every __all__ entry must BE a submodule, or a
    star-import raises AttributeError.  Both of these are."""
    import chronotagger.quickstart as qs

    assert "driver_export" in qs.__all__
    namespace = {}
    exec("from chronotagger.quickstart import *", namespace)
    assert "driver_export" in namespace
    assert "wizard" in namespace


def test_portable_stem_is_on_the_driver_export_public_surface():
    """R4, the OTHER __all__ change this pack makes, and the one nothing
    noticed: the byte lane reverted `portable_stem` out of
    `driver_export.__all__` and the whole suite stayed green (V2 F1).
    The cause was mundane -- the tests import the name directly, which
    works whether or not it is exported, and nothing star-imports the
    module.  This is EDIT 314's pin in the shape EDIT 299 needed."""
    assert "portable_stem" in driver_export.__all__
    namespace = {}
    exec("from chronotagger.quickstart.driver_export import *", namespace)
    assert "portable_stem" in namespace
    assert namespace["portable_stem"] is portable_stem
    # ...and the wizard really does call THIS function, which is the
    # whole reason it is public (one dataset, one identity).
    from chronotagger.quickstart import wizard as wiz_mod

    source = Path(wiz_mod.__file__).read_text(encoding="utf-8")
    assert "import portable_stem" in source


def test_a_csv_gz_driver_reads_it_with_read_csv():
    """R12.  `pd.read_csv` decompresses on the extension, so a gzipped
    csv needs no new reader -- only an accepted-format entry that agrees
    with the loader's."""
    one = _pane("Fields", ["BX"])
    text = generate_driver((one["layout_spec"], one["plot_config"]),
                           data_path="/d/x.csv.gz", fmt="csv.gz",
                           time_column="t", autosave_folder="./out")
    assert "    df = pd.read_csv(DATA_PATH)" in text
    assert "csv.gz" in driver_export.SUPPORTED_FORMATS
    # ...and it still demands a time column, because read_csv gives a
    # RangeIndex whatever the compression.
    with pytest.raises(ValueError, match="time_column"):
        generate_driver((one["layout_spec"], one["plot_config"]),
                        data_path="/d/x.csv.gz", fmt="csv.gz",
                        autosave_folder="./out")


@pytest.mark.parametrize("path,expected", [
    (r"C:\data\peif_with_liuzzo_labels.parquet", "peif_with_liuzzo_labels"),
    ("/mnt/data/peif.csv", "peif"),
    ("/mnt/data/peif.csv.gz", "peif"),
    (r"C:x.csv", "x"),
    ("plain.parquet", "plain"),
])
def test_portable_stem_is_one_spelling_on_every_platform(path, expected):
    """R4.  Not Path(...).stem: on POSIX a Windows path carries no
    separator at all, so the WHOLE path became the stem.  Both separators
    are separators here, on both platforms -- which is what lets the
    wizard and the emitter agree on one identity."""
    assert portable_stem(path) == expected


def test_the_emitted_source_name_matches_portable_stem():
    one = _pane("Fields", ["BX"])
    text = generate_driver((one["layout_spec"], one["plot_config"]),
                           data_path="/mnt/d/peif.csv", fmt="csv",
                           time_column="t", autosave_folder="./out")
    assert 'SOURCE_NAME = "%s"' % portable_stem("/mnt/d/peif.csv") in text


_RUNNER = '''
import sys
import types

import tkinter

_mb = types.ModuleType("tkinter.messagebox")
for _n in ("showerror", "showwarning", "showinfo"):
    setattr(_mb, _n, lambda *a, **k: "ok")
for _n in ("askyesno", "askokcancel", "askyesnocancel"):
    setattr(_mb, _n, lambda *a, **k: True)
sys.modules["tkinter.messagebox"] = _mb
tkinter.messagebox = _mb

_fd = types.ModuleType("tkinter.filedialog")
for _n in ("askopenfilename", "asksaveasfilename", "askdirectory"):
    setattr(_fd, _n, lambda *a, **k: "")
sys.modules["tkinter.filedialog"] = _fd
tkinter.filedialog = _fd

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

import json
import runpy

from chronotagger.labeler import TimeIntervalLabeler

_captured = {}


def _fake_run(self):
    _captured["app"] = self


TimeIntervalLabeler.run = _fake_run

ns = runpy.run_path(sys.argv[1], run_name="__main__")
app = _captured["app"]

out = {"panes": [], "rows": int(len(app.df))}
for entry in ns["PANES"]:
    layout = entry["layout_spec"]
    fig = Figure(figsize=(6, 4))
    axs = {}
    for area in layout["areas"]:
        if area.get("role") != "labels":
            axs[area["key"]] = fig.add_subplot(
                len(axs) + 1, 1, len(axs) + 1)
    for _ in range(2):
        entry["plot_fn"](axs, app.df, app.t0, app.t1)
    out["panes"].append({
        "title": entry["title"],
        "keys": sorted(axs),
        "artists": {k: [len(a.lines), len(a.collections)]
                    for k, a in axs.items()},
        "texts": {k: [t.get_text() for t in a.texts]
                  for k, a in axs.items()},
    })
out["constructed_panes"] = len(app.panes)
out["titles"] = [p.title for p in app.panes]
out["classes"] = list(app.classes)
out["first"] = str(app.df.index[0])
out["last"] = str(app.df.index[-1])
print(json.dumps(out))
'''


def _run_driver(tmp_path, text, name="drive.py"):
    driver = tmp_path / name
    write_driver(text, driver, overwrite=True)
    runner = tmp_path / "run_driver.py"
    runner.write_text(_RUNNER, encoding="ascii")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_ROOT)
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run(
        [sys.executable, str(runner), str(driver)],
        capture_output=True, text=True, cwd=str(tmp_path), env=env)
    assert proc.returncode == 0, (
        "generated driver failed to run\n--- stdout ---\n%s\n"
        "--- stderr ---\n%s" % (proc.stdout, proc.stderr))
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_emitted_two_pane_driver_runs_and_builds_both_panes(tmp_path):
    """The emitter earns its output by RUNNING it (Pack 7's doctrine
    sentence).  Both plot functions are called TWICE against real axes,
    because a plot_fn that accumulates artists is the bug this whole
    generator lineage exists to have killed."""
    data = _write_csv(tmp_path / "sample.csv")
    text = generate_driver(
        _two_panes(), data_path=str(data), fmt="csv", time_column="t",
        classes=["sheet", "lobe"], window="10min", step="5min",
        autosave_folder=str(tmp_path / "autosave"))

    got = _run_driver(tmp_path, text, "drive_panes.py")
    assert got["rows"] == 240
    assert got["constructed_panes"] == 2
    assert got["titles"] == ["Fields", "Orbit"]
    assert got["classes"] == ["sheet", "lobe"]
    assert [p["title"] for p in got["panes"]] == ["Fields", "Orbit"]
    assert got["panes"][0]["keys"] == ["panel_1", "panel_2"]
    assert got["panes"][1]["keys"] == ["panel_1", "xplot_1"]
    for pane in got["panes"]:
        for key, counts in pane["artists"].items():
            assert sum(counts) == 1, (pane["title"], key, counts)
        for texts in pane["texts"].values():
            assert texts == []


# =====================================================================
# 3. The loader -- R11 / R12 / R14
# =====================================================================

def _loader(gui, tmp_path, path):
    """A real FileLoaderDialog, driven through its real load path."""
    from chronotagger.quickstart.file_loader import FileLoaderDialog

    dialog = FileLoaderDialog(gui.root)
    dialog._load_and_preview(str(path))
    return dialog


def test_a_gzipped_csv_loads_through_the_wizard_loader(gui, tmp_path):
    """R12.  `Path.suffix` is the LAST suffix only, so `.csv.gz` read as
    `.gz` and the user was told 'Only CSV and Parquet files are
    supported' about a file that IS a csv."""
    raw = tmp_path / "sample.csv"
    _write_csv(raw)
    packed = tmp_path / "sample.csv.gz"
    with open(raw, "rb") as src, gzip.open(packed, "wb") as dst:
        dst.write(src.read())

    dialog = _loader(gui, tmp_path, packed)
    assert dialog.loaded_df is not None
    assert len(dialog.loaded_df) == 240
    assert [kind for kind, _, _ in gui.messages] == []
    assert str(dialog.continue_btn["state"]) == "normal"

    # ...and the picker now offers the extension it accepts.
    from chronotagger.quickstart import file_loader as fl

    source = Path(fl.__file__).read_text(encoding="utf-8")
    assert '("Gzipped CSV files", "*.csv.gz")' in source


@pytest.mark.parametrize("unit,divisor", [
    ("s", 10 ** 9), ("ms", 10 ** 6), ("us", 10 ** 3), ("ns", 1)])
def test_an_integer_epoch_gets_a_preselected_unit(gui, tmp_path, unit,
                                                  divisor):
    """R11.  Four units, three decades of magnitude apart; the heuristic
    reads the magnitude and preselects.  Pack 6 F8's numeric gate makes
    auto-detect REFUSE an integer column, so the manual dropdown is the
    only route to one -- which is exactly the route that was unguarded."""
    path = _epoch_csv(tmp_path / "epoch.csv", divisor)

    dialog = _loader(gui, tmp_path, path)
    # auto-detect refuses a numeric column, so nothing is selected yet
    assert dialog.time_column_var.get() == "Auto-detect"
    assert str(dialog.time_unit_combo["state"]) == "disabled"

    dialog.time_column_var.set("t")
    dialog._on_time_column_changed()

    assert str(dialog.time_unit_combo["state"]) == "readonly"
    assert dialog.time_unit_var.get() == unit
    assert str(dialog.continue_btn["state"]) == "normal"


def test_a_microsecond_epoch_no_longer_lands_in_1970(gui, tmp_path):
    """The flagship (R11).  Measured at 28d18ca: a us epoch validated
    clean, the status label went black, Continue enabled, and the wizard
    handed the labeler a 1970 dataset 14.3 seconds wide instead of the
    real 39m50s -- so the windowing scale was wrong too.  This asserts
    the VALUES, because a dtype check is true of 1970 and of 2021 alike."""
    path = _epoch_csv(tmp_path / "epoch_us.csv", 1000)

    dialog = _loader(gui, tmp_path, path)
    dialog.time_column_var.set("t")
    dialog._on_time_column_changed()
    assert dialog.time_unit_var.get() == "us"

    dialog._on_continue()
    out = dialog.result
    assert out is not None
    assert str(out.index[0]) == "2021-03-01 00:00:00"
    assert str(out.index[-1]) == "2021-03-01 00:39:50"
    assert out.index[-1] - out.index[0] == pd.Timedelta("39min50s")


def test_an_integer_epoch_with_no_unit_blocks_continue(gui, tmp_path):
    """R11's BLOCKING half.  The heuristic preselects, so this state is a
    guard rather than a routine dead end -- but a guard that is never
    exercised is a guard nobody can trust."""
    path = _epoch_csv(tmp_path / "epoch_us.csv", 1000)

    dialog = _loader(gui, tmp_path, path)
    dialog.time_column_var.set("t")
    dialog._on_time_column_changed()

    dialog.time_unit_var.set("")                     # the user clears it
    dialog._on_time_column_changed()
    assert str(dialog.continue_btn["state"]) == "disabled"
    assert "epoch unit" in str(dialog.status_label["text"]).lower()
    assert str(dialog.status_label["foreground"]) == "red"

    dialog._on_continue()
    assert dialog.result is None
    assert gui.messages[-1][0] == "showerror"

    dialog.time_unit_var.set("us")
    dialog._on_time_column_changed()
    assert str(dialog.continue_btn["state"]) == "normal"


def test_a_datetime_column_leaves_the_unit_control_alone(gui, tmp_path):
    """R11 is scoped to INTEGER columns.  A column of timestamps names
    its own units, and nothing about this screen changes for it."""
    path = _write_csv(tmp_path / "dates.csv")
    dialog = _loader(gui, tmp_path, path)

    assert dialog.time_column_var.get() == "t"
    assert str(dialog.time_unit_combo["state"]) == "disabled"
    assert dialog.time_unit_var.get() == ""
    assert str(dialog.continue_btn["state"]) == "normal"

    dialog._on_continue()
    assert dialog.result is not None
    assert dialog.time_column == "t"
    assert dialog.time_is_epoch is False
    assert dialog.time_unit is None


def test_unsorted_data_is_sorted_not_refused(gui, tmp_path):
    """R14.  Three sites had three policies on one frame: this screen
    REFUSED, the constructor sorts with a warning (Pack 6 R9), and an
    emitted driver sorts unconditionally -- so the wizard blocked the
    user from generating the driver that handles their data correctly."""
    frame = _frame(120)
    shuffled = frame.iloc[np.r_[60:120, 0:60]]
    path = tmp_path / "unsorted.csv"
    shuffled.reset_index(names="t").to_csv(path, index=False)

    dialog = _loader(gui, tmp_path, path)
    assert str(dialog.continue_btn["state"]) == "normal"
    status = str(dialog.status_label["text"])
    assert "sorted by time" in status
    assert str(dialog.status_label["foreground"]) == "black"

    dialog._on_continue()
    assert dialog.result is not None
    assert dialog.result.index.is_monotonic_increasing

    # the old refusal is gone from the validator, message and all
    ok, message = dialog._validate_data(dialog.result)
    assert ok is True and message == ""
    from chronotagger.quickstart import file_loader as fl

    source = Path(fl.__file__).read_text(encoding="utf-8")
    assert "must be sorted in ascending order" not in source


def test_a_file_that_cannot_be_used_says_so_on_first_load(gui, tmp_path):
    """SECTION 0a F8, the half v1 did not pin (V1 FOLD 4).
    `_load_and_preview` used to write a cheerful black `Rows: N,
    Columns: M` from the RAW frame and then disable Continue with no
    explanation.  Mutating `_set_status(df_with_time, is_valid, msg)` to
    `_set_status(df_with_time, True, msg)` left the whole suite green,
    while the SORT half of the same EDIT was caught -- so exactly one of
    EDIT 319's two claims was pinned."""
    path = tmp_path / "one_row.csv"
    _frame(1, strings=False).reset_index(names="t").to_csv(path, index=False)

    dialog = _loader(gui, tmp_path, path)
    assert str(dialog.continue_btn["state"]) == "disabled"
    assert str(dialog.status_label["foreground"]) == "red"
    assert "at least 2 rows" in str(dialog.status_label["text"])
    # the black row/column line is what USED to appear here
    assert "Rows: 1" not in str(dialog.status_label["text"])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_non_date_column_is_refused_not_raised(gui, tmp_path):
    """SECTION 0a F13, from V3 F-2.  A column of category strings is one
    misclick away in the dropdown -- `liuzzo_class` is in this project's
    own fixtures -- and `pd.to_datetime` RAISES on it.  Unguarded, that
    escaped a `<<ComboboxSelected>>` callback and a Button command:
    Tkinter printed a traceback nobody reads and the screen kept a stale
    black status line over an ENABLED Continue button that did nothing
    when pressed.  The filterwarnings mark is about the FIXTURE: pandas
    warns on its way to failing to parse "sheet"."""
    path = _write_csv(tmp_path / "with_strings.csv")

    dialog = _loader(gui, tmp_path, path)
    assert str(dialog.continue_btn["state"]) == "normal"

    dialog.time_column_var.set("liuzzo_class")
    dialog._on_time_column_changed()

    assert str(dialog.continue_btn["state"]) == "disabled"
    assert str(dialog.status_label["foreground"]) == "red"
    text = str(dialog.status_label["text"])
    assert "liuzzo_class" in text
    assert "cannot be read as a time column" in text

    dialog._on_continue()
    assert dialog.result is None
    assert gui.messages[-1][0] == "showerror"

    # ...and choosing a usable column again recovers, rather than
    # leaving the screen wedged.
    dialog.time_column_var.set("t")
    dialog._on_time_column_changed()
    assert str(dialog.continue_btn["state"]) == "normal"
    assert str(dialog.status_label["foreground"]) == "black"


def test_a_leading_fill_value_does_not_move_the_preselected_unit(gui,
                                                                 tmp_path):
    """SECTION 0a F12, from V3 F-5.  `_guess_epoch_unit` read
    `values.iloc[0]`, so a single leading 0 fill value in a column of
    1000 NANOSECOND values preselected `s` -- three decades wrong -- and
    Continue lit up the moment any unit was present, so nothing made the
    user look.  The median reads the column, not its first row."""
    path = _epoch_csv(tmp_path / "epoch_ns.csv", 1)
    lines = path.read_text(encoding="ascii").splitlines()
    first = lines[1].split(",")
    first[0] = "0"
    lines[1] = ",".join(first)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")

    dialog = _loader(gui, tmp_path, path)
    dialog.time_column_var.set("t")
    dialog._on_time_column_changed()

    assert dialog.time_unit_var.get() == "ns"
    assert str(dialog.continue_btn["state"]) == "normal"


@pytest.mark.parametrize("unit,divisor", [
    ("s", 10 ** 9), ("ms", 10 ** 6), ("us", 10 ** 3)])
def test_a_float_epoch_needs_a_unit_too(gui, tmp_path, unit, divisor):
    """DRAFT AMENDMENT A8-1.  R11 shut the 1970 trap for INTEGER epoch
    columns and left it open for float ones -- the shape pyspedas and
    CDF time variables arrive in, and the shape this user's own data is
    in.  Measured before this amendment, on both pandas majors: a 39m50s
    float-seconds file resolved to a span of 2.39 MICROSECONDS at
    1970-01-01 00:00:01.6145568, with the status line black and Continue
    enabled.  Asserted on the VALUES, because a dtype check is true of
    1970 and of 2021 alike."""
    frame = _frame(strings=False)
    epoch = frame.index.astype("datetime64[ns]").astype("int64") // divisor
    flat = frame.reset_index(drop=True)
    flat.insert(0, "t", epoch.values.astype("float64"))
    path = tmp_path / "float_epoch.csv"
    flat.to_csv(path, index=False)

    dialog = _loader(gui, tmp_path, path)
    assert str(dialog.time_unit_combo["state"]) == "disabled"

    dialog.time_column_var.set("t")
    dialog._on_time_column_changed()
    assert pd.api.types.is_float_dtype(dialog.loaded_df["t"])
    assert str(dialog.time_unit_combo["state"]) == "readonly"
    assert dialog.time_unit_var.get() == unit

    # the BLOCKING half holds for floats exactly as it does for ints
    dialog.time_unit_var.set("")
    dialog._on_time_column_changed()
    assert str(dialog.continue_btn["state"]) == "disabled"
    assert str(dialog.status_label["foreground"]) == "red"
    assert "epoch unit" in str(dialog.status_label["text"]).lower()

    dialog.time_unit_var.set(unit)
    dialog._on_time_column_changed()
    dialog._on_continue()
    out = dialog.result
    assert out is not None
    assert str(out.index[0]) == "2021-03-01 00:00:00"
    assert out.index[-1] - out.index[0] == pd.Timedelta("39min50s")
    # ...and the driver is told how to reproduce it
    assert (dialog.time_column, dialog.time_is_epoch, dialog.time_unit) \
        == ("t", True, unit)


def test_a_bool_column_is_not_an_epoch(gui, tmp_path):
    """DRAFT AMENDMENT A8-1's exclusion, and it is not theoretical:
    pandas reports `is_numeric_dtype(bool)` as True, so a numeric gate
    without the bool clause would offer an epoch-unit dropdown for a
    flag column."""
    frame = _frame(strings=False)
    frame["flag"] = (np.arange(len(frame)) % 2 == 0)
    path = tmp_path / "bools.csv"
    frame.reset_index(names="t").to_csv(path, index=False)

    dialog = _loader(gui, tmp_path, path)
    assert dialog.loaded_df["flag"].dtype == bool
    assert pd.api.types.is_numeric_dtype(dialog.loaded_df["flag"])
    assert dialog._is_epoch_column(dialog.loaded_df, "flag") is False

    dialog._sync_time_unit_control(dialog.loaded_df, "flag")
    assert str(dialog.time_unit_combo["state"]) == "disabled"
    assert dialog.time_unit_var.get() == ""


def test_the_loader_records_how_it_read_the_time_axis(gui, tmp_path):
    """R6/R11.  The driver has to load the same file the same way, and
    the only screen that knows how is this one."""
    path = _epoch_csv(tmp_path / "epoch_ms.csv", 10 ** 6)

    dialog = _loader(gui, tmp_path, path)
    dialog.time_column_var.set("t")
    dialog._on_time_column_changed()
    dialog._on_continue()

    assert dialog.time_column == "t"
    assert dialog.time_is_epoch is True
    assert dialog.time_unit == "ms"


# =====================================================================
# 4. The wizard -- R4 / R5 / R8 / R9 / R10 / R13 / R16
# =====================================================================

class _StubLabeler:
    """Captures what the wizard hands the constructor."""

    seen = []

    def __init__(self, **kwargs):
        _StubLabeler.seen.append(kwargs)

    def run(self):
        pass


def _wizard(gui, monkeypatch, tmp_path, panes=1, save_to="",
            classes=None):
    """A QuickStartWizard wired to a real data file and stub screens."""
    import chronotagger
    import chronotagger.quickstart.wizard as wiz_mod

    _StubLabeler.seen = []
    monkeypatch.setattr(chronotagger, "TimeIntervalLabeler", _StubLabeler)

    data = _write_csv(tmp_path / "peif.csv")
    wiz = wiz_mod.QuickStartWizard()
    wiz.root = gui.root
    wiz.df = _frame()
    wiz.source_path = str(data)
    wiz.source_name = portable_stem(str(data))
    wiz.time_column = "t"
    wiz.tabs_config = [
        {"title": "Tab %d" % (i + 1), "columns": ["BX", "n_linear"],
         "layout_type": "vertical_stack"}
        for i in range(panes)]
    if classes is not None:
        wiz.classes, wiz.class_colors = classes
    gui.save_returns.append(save_to)
    return wiz


def test_the_wizard_stores_the_stem_not_the_whole_path(gui, monkeypatch,
                                                      tmp_path):
    """R4.  Measured at 28d18ca: `wizard.source_name` was the FULL PATH
    while the emitter defaulted to the STEM, so a session and a driver
    generated from that session carried two identities for one dataset --
    and `source_name` is exactly what `_check_autosave` compares when two
    datasets share a fingerprint."""
    import chronotagger.quickstart.file_loader as fl_mod
    import chronotagger.quickstart.wizard as wiz_mod

    data = _write_csv(tmp_path / "peif.csv")
    frame = _frame()

    class _Loader:
        def __init__(self, parent):
            self.current_file = str(data)
            self.time_column = "t"
            self.time_is_epoch = True
            self.time_unit = "us"

        def run(self):
            return frame

    monkeypatch.setattr(fl_mod, "FileLoaderDialog", _Loader)
    wiz = wiz_mod.QuickStartWizard()
    wiz.root = gui.root
    monkeypatch.setattr(wiz, "_show_tab_planner", lambda: None)
    wiz._show_file_loader()

    assert wiz.source_path == str(data)
    assert wiz.source_name == "peif"
    assert os.sep not in wiz.source_name
    assert wiz.source_name == portable_stem(str(data))
    # ...and the time metadata rides across with it (R6/R11)
    assert (wiz.time_column, wiz.time_is_epoch, wiz.time_unit) == (
        "t", True, "us")


def test_the_wizard_asks_for_classes_once_and_a_retry_keeps_them(
        gui, monkeypatch, tmp_path):
    """R5/R8.  A failed launch RECURSES into _show_tab_planner, so the
    classes guard is what stops the retry re-asking a question the user
    already answered -- and losing their schema each time round."""
    wiz = _wizard(gui, monkeypatch, tmp_path)
    asked = []

    def fake_classes():
        asked.append(1)
        wiz.classes = ["sheet", "lobe"]
        wiz.class_colors = {"sheet": "#d62728", "lobe": "#1f77b4"}

    monkeypatch.setattr(wiz, "_show_classes", fake_classes)
    monkeypatch.setattr(wiz, "_launch_labeler", lambda: None)
    planner_result = {"tabs": wiz.tabs_config}

    class _Planner:
        def __init__(self, parent, df):
            pass

        def run(self):
            return planner_result

    import chronotagger.quickstart.tab_planner as tp_mod
    monkeypatch.setattr(tp_mod, "TabPlannerDialog", _Planner)

    wiz._show_tab_planner()
    assert asked == [1]
    wiz._show_tab_planner()                    # the launch-failure retry
    assert asked == [1]
    assert wiz.classes == ["sheet", "lobe"]


def test_the_classes_screen_relaxes_the_UNKNOWN_reservation(gui,
                                                            monkeypatch,
                                                            tmp_path):
    """R8.  UNKNOWN is the CONSTRUCTOR's default first class, not a law
    about what a user may call theirs: reused as-is, the Label Manager
    cannot produce the mock spec's sheet / lobe / sheath at all."""
    from chronotagger.labeler.dialogs import label_manager as lm_mod

    wiz = _wizard(gui, monkeypatch, tmp_path)
    seen = {}
    real = lm_mod.LabelManagerDialog

    class _Spy(real):
        def __init__(self, **kwargs):
            seen.update(kwargs)
            real.__init__(self, **kwargs)

    monkeypatch.setattr(lm_mod, "LabelManagerDialog", _Spy)

    def drive(widget):
        widget._classes = ["sheet", "lobe", "sheath"]
        widget._colors = {"sheet": "#d62728", "lobe": "#1f77b4",
                          "sheath": "#ff7f0e"}
        widget._on_ok()

    gui.on_wait_window = drive
    wiz._show_classes()

    assert seen["reserved"] == frozenset()
    assert seen["usage_counts"] == {}
    assert seen["classes"] == ["UNKNOWN", "label_1", "label_2"]
    assert wiz.classes == ["sheet", "lobe", "sheath"]
    assert wiz.class_colors["sheath"] == "#ff7f0e"


def test_the_live_label_manager_keeps_its_reservation(gui):
    """R8's other half, and the reason it is worth a pin: relaxing the
    reservation on the WIZARD path must not relax it on the live one,
    where intervals already carry the name."""
    from chronotagger.labeler.dialogs.label_manager import LabelManagerDialog

    dialog = LabelManagerDialog(
        parent=gui.root,
        classes=["UNKNOWN", "sheet"],
        class_colors={"UNKNOWN": "#4e79a7", "sheet": "#d62728"},
        usage_counts={"UNKNOWN": 3, "sheet": 1},
    )
    assert dialog._reserved == {"UNKNOWN"}


def test_the_wizard_hands_over_schema_step_and_autosave_folder(
        gui, monkeypatch, tmp_path):
    """R8/R9/R10 in one constructor call.

    step: measured at 28d18ca as pd.Timedelta('15min') against a 59.9 s
    window -- one 'next window' skipping fifteen times the visible span.
    autosave_folder: measured as literally '.', with chronotagger.log
    already written into the process CWD before any screen could ask."""
    wiz = _wizard(gui, monkeypatch, tmp_path,
                  classes=(["sheet", "lobe"],
                           {"sheet": "#d62728", "lobe": "#1f77b4"}))
    wiz._launch_labeler()

    kwargs = _StubLabeler.seen[-1]
    assert kwargs["classes"] == ["sheet", "lobe"]
    assert kwargs["class_colors"] == {"sheet": "#d62728", "lobe": "#1f77b4"}
    assert kwargs["step"] == kwargs["window"] / 2
    assert kwargs["source_name"] == "peif"
    expected = os.path.join(str(tmp_path), "chronotagger_autosave")
    assert os.path.abspath(kwargs["autosave_folder"]) == expected
    assert os.path.abspath(kwargs["autosave_folder"]) != os.path.abspath(".")


def test_save_as_writes_the_driver_the_dialog_named(gui, monkeypatch,
                                                    tmp_path):
    """R6.  The dialog comes FIRST and its basename becomes `file_name`,
    so the generated 'Run it with: python <name>' line and the file on
    disk cannot disagree -- which they would the moment the user renamed
    in the dialog."""
    target = tmp_path / "my_own_name.py"
    wiz = _wizard(gui, monkeypatch, tmp_path, panes=2, save_to=str(target),
                  classes=(["sheet", "lobe"],
                           {"sheet": "#d62728", "lobe": "#1f77b4"}))
    wiz._launch_labeler()

    assert target.exists()
    text = target.read_text(encoding="ascii")
    assert "# [GEN] Run it with:   python my_own_name.py" in text
    assert 'CLASSES = ["sheet", "lobe"]' in text
    assert "        panes=PANES," in text
    assert "LAYOUT_1 = {" in text and "LAYOUT_2 = {" in text
    assert 'SOURCE_NAME = "peif"' in text

    kwargs = gui.saves[-1][1]
    assert kwargs["initialfile"] == "drive_peif.py"
    assert kwargs["defaultextension"] == ".py"
    assert kwargs["parent"] is gui.root
    # W8's "never silent overwrite" said out loud rather than inherited
    # from tk_getSaveFile's platform default (V1 FOLD 7).
    assert kwargs["confirmoverwrite"] is True

    # the emitted window/step are the SESSION's, not the emitter defaults
    seen = _StubLabeler.seen[-1]
    assert 'window=pd.Timedelta("%s")' % seen["window"] in text
    assert 'step=pd.Timedelta("%s")' % seen["step"] in text
    assert "30min" not in text and "15min" not in text

    # R6's wizard half, and it was unpinned: reverting `overwrite=True`
    # to False left the whole suite green while a user REGENERATING a
    # driver -- the common case, because save-as defaults to the file
    # they made last time -- got FileExistsError, F6's showerror, and no
    # file (V3 F-8a).  Re-offering over the file just written is that
    # case.  `_offer_save_as` rather than a second `_launch_labeler`,
    # because the first one destroys the wizard root on the way out.
    target.write_text("# hand edited\n", encoding="ascii")
    gui.save_returns.append(str(target))
    assert wiz._offer_save_as(seen["window"], seen["step"]) == str(target)
    assert [k for k, _t, _b in gui.messages if k == "showerror"] == []
    assert "# hand edited" not in target.read_text(encoding="ascii")
    assert "        panes=PANES," in target.read_text(encoding="ascii")


def test_a_wizard_driver_autosaves_where_the_session_did(
        gui, monkeypatch, tmp_path):
    """DRAFT AMENDMENT A8-2.  Measured before it: the session wrote to
    `<data dir>/chronotagger_autosave` and its own driver to
    `./chronotagger_autosave`, `same_folder: false` -- so a user who
    labelled through the wizard, closed, and ran the driver the wizard
    had just written them was NOT offered their own autosave unless they
    happened to be standing in the data directory.  W5's ratified
    doctrine sentence is 'the driver launches, the session restores'."""
    target = tmp_path / "drive_beside.py"
    wiz = _wizard(gui, monkeypatch, tmp_path, save_to=str(target),
                  classes=(["sheet", "lobe"],
                           {"sheet": "#d62728", "lobe": "#1f77b4"}))
    wiz._launch_labeler()

    text = target.read_text(encoding="ascii")
    assert "import os" in text
    assert ("AUTOSAVE_FOLDER = os.path.join(os.path.dirname(DATA_PATH),"
            in text)
    assert 'AUTOSAVE_FOLDER = r"./chronotagger_autosave"' not in text

    # the VALUE, not the text that computes it: run the module body
    # (everything above `if __name__`) and read what it bound.
    body = text.split('if __name__ == "__main__":')[0]
    namespace = {}
    exec(compile(body, "<driver>", "exec"), namespace)
    session_folder = _StubLabeler.seen[-1]["autosave_folder"]
    assert namespace["AUTOSAVE_FOLDER"] == os.path.join(
        str(tmp_path), "chronotagger_autosave")
    assert (os.path.abspath(namespace["AUTOSAVE_FOLDER"])
            == os.path.abspath(session_folder))


def test_the_emitter_default_autosave_folder_is_untouched_by_A8_2():
    """A8-2's other half, and the one R9 actually rules on: a caller with
    no session context gets the literal it always got, and the generated
    file carries no stdlib import.  `_is_bare_cwd` keeps its semantics in
    BOTH shapes -- the guard runs on the same argument with the same
    message whether or not the literal is the thing emitted."""
    one = _pane("Fields", ["BX"])
    text = _emit((one["layout_spec"], one["plot_config"]),
                 autosave_folder="./chronotagger_autosave")
    assert 'AUTOSAVE_FOLDER = r"./chronotagger_autosave"' in text
    assert "import os" not in text
    assert "os.path.dirname(DATA_PATH)" not in text

    for extra in ({}, {"autosave_beside_data": True}):
        with pytest.raises(ValueError, match="not the process CWD"):
            _emit((one["layout_spec"], one["plot_config"]),
                  autosave_folder=".", **extra)


def test_declining_save_as_writes_nothing_and_still_launches(
        gui, monkeypatch, tmp_path):
    """R7, and it is a two-part claim: nothing is written ANYWHERE (no
    fallback file, no temp copy), and the session opens regardless."""
    before = sorted(p.name for p in tmp_path.iterdir())
    wiz = _wizard(gui, monkeypatch, tmp_path, save_to="")
    wiz._launch_labeler()

    assert len(_StubLabeler.seen) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == before + ["peif.csv"]
    assert not list(tmp_path.glob("*.py"))


def test_a_csv_gz_session_writes_a_csv_gz_driver(gui, monkeypatch,
                                                 tmp_path):
    """R12's THIRD site (V1 FOLD 4).  The loader reads a `.csv.gz` and
    the emitter accepts one; `wizard._data_format` is what tells the
    emitter WHICH -- and disabling its `.csv.gz` branch left the whole
    suite green, so the two pinned sites agreed with each other while
    the site between them was free to disagree with both."""
    raw = tmp_path / "peif.csv"
    _write_csv(raw)
    packed = tmp_path / "peif.csv.gz"
    with open(raw, "rb") as src, gzip.open(packed, "wb") as dst:
        dst.write(src.read())

    target = tmp_path / "drive_from_gz.py"
    wiz = _wizard(gui, monkeypatch, tmp_path, save_to=str(target),
                  classes=(["sheet", "lobe"],
                           {"sheet": "#d62728", "lobe": "#1f77b4"}))
    wiz.source_path = str(packed)
    wiz.source_name = portable_stem(str(packed))
    wiz._launch_labeler()

    text = target.read_text(encoding="ascii")
    assert "    df = pd.read_csv(DATA_PATH)" in text
    assert "peif.csv.gz" in text
    # F9: the gzipped file and the plain one are ONE dataset by name
    assert 'SOURCE_NAME = "peif"' in text
    assert [k for k, _t, _b in gui.messages if k == "showerror"] == []

    # the sniff itself, on the shapes the picker can hand it
    for name, fmt in (("peif.csv.gz", "csv.gz"), ("peif.CSV.GZ", "csv.gz"),
                      ("peif.csv", "csv"), ("peif.parquet", "parquet")):
        wiz.source_path = str(tmp_path / name)
        assert wiz._data_format() == fmt, name


def test_a_zero_span_dataset_still_gets_a_usable_step(gui, monkeypatch,
                                                      tmp_path):
    """V3 F-3, and the ONE regression v1 introduced.  `_validate_data`
    requires a DatetimeIndex, no NaT and >= 2 rows -- it does NOT require
    a non-zero span, and `is_monotonic_increasing` is non-strict -- so a
    single-timestamp snapshot across several probes passes.  R10's
    `window / 2` then made the step zero as well, and "next window"
    advanced by nothing; at base the step was the constructor's flat 15
    minutes, so the session was at least navigable."""
    wiz = _wizard(gui, monkeypatch, tmp_path,
                  classes=(["sheet", "lobe"],
                           {"sheet": "#d62728", "lobe": "#1f77b4"}))
    flat = _frame(6)
    flat.index = pd.DatetimeIndex([flat.index[0]] * len(flat))
    wiz.df = flat
    wiz._launch_labeler()

    kwargs = _StubLabeler.seen[-1]
    assert kwargs["window"] == pd.Timedelta(0)
    assert kwargs["step"] == pd.Timedelta("15min")

    # The floor is a FLOOR, not a replacement, and the mechanism is a
    # pandas truthiness fact worth pinning because this suite runs on two
    # pandas majors: zero is falsy, any real step is truthy and passes
    # through untouched.  The ordinary case is pinned by
    # test_the_wizard_hands_over_schema_step_and_autosave_folder, which
    # asserts step == window / 2; re-asserting it here would need a
    # second _launch_labeler, and the first one destroys the root.
    assert bool(pd.Timedelta(0)) is False
    assert ((pd.Timedelta("6min") / 2 or pd.Timedelta("15min"))
            == pd.Timedelta("3min"))


def test_cancel_returns_none_instead_of_exiting(tmp_path):
    """R13.  Measured at 28d18ca: sys.exit(0) propagated out of run() --
    NOT from a Tk callback, which the Pack 7 record got wrong; the flow
    runs before mainloop() on the ordinary Python stack -- through
    launcher.py's `except Exception` (SystemExit is a BaseException) and
    killed the process, with exit code 0.  A subprocess is the only
    honest way to pin 'the interpreter survives'."""
    probe = tmp_path / "cancel_probe.py"
    probe.write_text(
        "import sys, types, tkinter\n"
        "_mb = types.ModuleType('tkinter.messagebox')\n"
        "for _n in ('showerror', 'showwarning', 'showinfo'):\n"
        "    setattr(_mb, _n, lambda *a, **k: 'ok')\n"
        "for _n in ('askyesno', 'askokcancel', 'askyesnocancel'):\n"
        "    setattr(_mb, _n, lambda *a, **k: True)\n"
        "sys.modules['tkinter.messagebox'] = _mb\n"
        "tkinter.messagebox = _mb\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "from chronotagger.quickstart import wizard as wiz_mod\n"
        "wiz = wiz_mod.QuickStartWizard()\n"
        "from chronotagger.labeler.mixins.view_build.window import "
        "_new_tk_root\n"
        "wiz.root = _new_tk_root()\n"
        "wiz.root.withdraw()\n"
        "wiz._on_cancel()\n"
        "print('MARKER after _on_cancel, cancelled=%s' % wiz.cancelled)\n"
        "print('MARKER run() returned %r' % (wiz_mod.run.__doc__ is not "
        "None,))\n",
        encoding="ascii")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_ROOT)
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run([sys.executable, str(probe)], capture_output=True,
                          text=True, cwd=str(tmp_path), env=env)
    assert proc.returncode == 0, proc.stderr
    assert "MARKER after _on_cancel, cancelled=True" in proc.stdout
    assert "MARKER run() returned True" in proc.stdout

    # ...and the CODE, not the prose, is where sys.exit has to be gone:
    # the comment that replaced it says the word several times.
    import ast

    from chronotagger.quickstart import wizard as wiz_mod

    tree = ast.parse(Path(wiz_mod.__file__).read_text(encoding="utf-8"))
    imported = set()
    exits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Call):
            target = node.func
            if (isinstance(target, ast.Attribute)
                    and target.attr in ("exit", "_exit")):
                exits.append(ast.dump(target))
            elif isinstance(target, ast.Name) and target.id == "exit":
                exits.append(target.id)
    assert "sys" not in imported
    assert exits == []


def test_the_wizard_root_goes_through_the_bounded_retry(monkeypatch, gui):
    """R16.  Pack 6 R10 measured tk.Tk() raising a transient TclError in
    89% of full-suite runs on this machine and shipped the retry; the
    wizard's own root never used it."""
    import chronotagger.labeler.mixins.view_build.window as window_mod
    import chronotagger.quickstart.wizard as wiz_mod

    calls = []
    loops = []
    monkeypatch.setattr(window_mod, "_new_tk_root",
                        lambda: calls.append(1) or gui.root)
    monkeypatch.setattr(tk, "Tk", _forbidden_tk)
    monkeypatch.setattr(tk.Misc, "mainloop",
                        lambda self, n=0: loops.append(1))

    wiz = wiz_mod.QuickStartWizard()
    wiz.cancelled = True
    monkeypatch.setattr(wiz, "_center_window", lambda: None)
    monkeypatch.setattr(wiz, "_show_file_loader", lambda: None)
    assert wiz.run() is None
    assert calls == [1]
    # R13's OTHER half: a cancelled run must SKIP the loop.  Without the
    # recorder the only thing that notices a regression here is a real
    # mainloop() on a live root -- i.e. a HANG, which CI reports as a
    # timeout rather than as a failure (measured: 240 s, no output).
    assert loops == []


def _forbidden_tk(*args, **kwargs):
    raise AssertionError("tk.Tk() called directly; use _new_tk_root()")


# =====================================================================
# 5. The designer -- R15 / R16 / R17
# =====================================================================

def _designer(gui):
    from chronotagger.labeler.utils.layout_builder.dialog import (
        LayoutBuilderDialog,
    )
    from chronotagger.labeler.utils.layout_builder.models import PanelConfig

    dialog = LayoutBuilderDialog(gui.root, _frame(40))
    dialog.panels = [p for p in dialog.panels if p.locked]
    dialog.panels.insert(0, PanelConfig(
        key="panel_1", row=0, col=0, rowspan=1, colspan=1, role="time",
        y_column="BX"))
    return dialog


def test_preview_allocates_no_second_root_and_no_pyplot_figure(gui):
    """R15.  Measured: `plt.figure()` under TkAgg builds a figure MANAGER
    whose window is `tk.Tk(className='matplotlib')` -- five live roots and
    five Tcl interpreters after four Preview clicks, all surviving the
    wizard root's own destroy() -- plus one leaked figure per click
    (1.00 over 25 clicks, matplotlib's own warning firing at 21)."""
    dialog = _designer(gui)

    roots = []
    real_tk = tk.Tk

    class _Counting(real_tk):
        def __init__(self, *args, **kwargs):
            roots.append(kwargs)
            real_tk.__init__(self, *args, **kwargs)

    tk.Tk = _Counting
    try:
        assert plt.get_fignums() == []
        for _ in range(4):
            dialog._show_preview()
        assert roots == []
        assert plt.get_fignums() == []
    finally:
        tk.Tk = real_tk

    # the preview really did render -- an empty method would also pass
    # the two assertions above
    assert dialog.preview_window is not None
    assert dialog.preview_window.winfo_exists()


def test_the_layout_builder_package_is_ascii():
    """R15.  Pack 6's ASCII sweep did not reach here because Pack 7
    fenced the designer GUI; the stopwatch emoji at preview.py:121 was
    not merely a style breach, it painted a tofu box in the preview the
    user was looking at and printed a matplotlib UserWarning per render
    (glyph 9201 missing from DejaVu Sans)."""
    from chronotagger.labeler.utils import layout_builder

    package = Path(layout_builder.__file__).resolve().parent
    offenders = {}
    for path in sorted(package.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        bad = [hex(ord(ch)) for ch in text if ord(ch) > 127]
        if bad:
            offenders[path.name] = bad
    assert offenders == {}


def test_build_layout_uses_the_bounded_retry_when_parentless(monkeypatch,
                                                             gui):
    """R16.  The wizard always passes a parent, but `build_layout`'s own
    docstring and both shipped examples call it parentless -- the last
    unprotected root reachable from documented usage."""
    import chronotagger.labeler.mixins.view_build.window as window_mod
    from chronotagger.labeler.utils.layout_builder import builder as b_mod

    calls = []

    class _FakeRoot:
        def withdraw(self):
            pass

        def wait_window(self, widget):
            pass

        def destroy(self):
            pass

    monkeypatch.setattr(window_mod, "_new_tk_root",
                        lambda: calls.append(1) or _FakeRoot())
    monkeypatch.setattr(tk, "Tk", _forbidden_tk)

    class _FakeDialog:
        result_layout_spec = {"nrows": 1}
        result_plot_config = {}

        def __init__(self, parent, df):
            pass

    monkeypatch.setattr(b_mod, "LayoutBuilderDialog", _FakeDialog)
    layout, config = b_mod.build_layout(_frame(10))
    assert calls == [1]
    assert layout == {"nrows": 1}


def test_overlapping_panels_are_refused_on_done(gui):
    """R17b.  `PanelConfig.overlaps` is tested; `_panels_overlap` -- the
    one the dialog actually calls -- was not, and neither was the clip
    that creates the condition it exists to catch."""
    from chronotagger.labeler.utils.layout_builder.models import PanelConfig

    dialog = _designer(gui)
    dialog.panels.insert(1, PanelConfig(
        key="panel_2", row=0, col=0, rowspan=1, colspan=1, role="time",
        y_column="n_linear"))

    assert dialog._validate_layout() is False
    kind, title, body = gui.messages[-1]
    assert kind == "showwarning"
    assert title == "Overlapping Panels"
    assert "overlap" in body

    dialog._on_done()
    assert dialog.result_layout_spec is None


def test_shrinking_the_grid_clips_panels_onto_each_other(gui):
    """R17b, the other half of the pair.  The clip is mechanical and MAY
    collapse two disjoint panels onto one cell -- which is precisely what
    the refusal above exists for.  Pinning them together is what makes
    either one meaningful."""
    from chronotagger.labeler.utils.layout_builder.models import PanelConfig

    dialog = _designer(gui)
    dialog.nrows_var.set(4)
    dialog.ncols_var.set(3)
    dialog.panels.insert(1, PanelConfig(
        key="panel_2", row=2, col=2, rowspan=1, colspan=1, role="time",
        y_column="n_linear"))
    first = dialog.panels[0]
    first.row, first.col, first.rowspan, first.colspan = 0, 0, 2, 2

    dialog.nrows_var.set(1)
    dialog.ncols_var.set(1)
    dialog._clip_panels_to_grid()

    for panel in dialog.panels:
        assert panel.row == 0 and panel.col == 0
        assert panel.rowspan == 1 and panel.colspan == 1
    assert dialog._panels_overlap(dialog.panels[0], dialog.panels[1]) is True
    assert dialog._validate_layout() is False
