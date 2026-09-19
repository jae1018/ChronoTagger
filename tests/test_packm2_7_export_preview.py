"""Pack M2.7 -- THE EXPORT SCOPE IS RENAMED AND ITS PREVIEW TELLS THE
TRUTH.

THE NAME. "Selected intervals only" never meant the intervals you
selected -- `app.selected_interval` is not consulted anywhere on the
export path. It means the rows the ACTIVE lane has labeled. The radio,
the docstring and the preview's own "Scope:" line now say that. The wire
VALUE stays "selected", because seven assertions in the suite pass that
literal string into the writer and the preview.

THE PREVIEW was wrong four ways under that scope (probe_s4_export Q2, Q5):

    its rows came from EVERY lane      Agent rows a day before any Wake
                                       interval, every one of them -1 in
                                       a column the file can never hold
                                       as -1
    its limit counted CHUNKS not rows  a limit of 10 returned 37
    its total was a cross-lane sum     ~9,633 against a 9,602-row frame,
                                       and 25 rows actually written
    its size estimate followed         "~0.2 MB" for a 1,004-byte file
                                       and "~0.2 MB" for a 374,502-byte
                                       one

All four come from one question, and the answer is the writer's own:
`mask = (the ACTIVE lane's label id series) != -1`. The preview's total
is now the file's row count by construction. The preview's label column
is `label_id_column(active lane)` -- the name the FILE uses -- and the
wide preview's column chooser follows the same helper.

This module uses a three-lane fixture whose lanes OVERLAP in time, which
is what made the cross-lane walk and the per-interval total visible.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from chronotagger.core.models import Interval

DAY = "2015-01-03 "

REGION = {"id": "region", "name": "Region (human)",
          "classes": ["plasma_sheet", "lobe", "UNKNOWN"],
          "class_colors": {"plasma_sheet": "#4e79a7", "lobe": "#f28e2b",
                           "UNKNOWN": "#7f7f7f"}, "order": 0}
WAKE = {"id": "wake", "name": "Wake", "classes": ["umbra", "UNKNOWN"],
        "class_colors": {"umbra": "#333333", "UNKNOWN": "#7f7f7f"},
        "order": 1}
AGENT = {"id": "agent", "name": "Agent (C-MMAE)", "classes": ["0", "1"],
         "class_colors": {"0": "#111111", "1": "#222222"}, "order": 2}

LAYOUT = {
    "nrows": 2, "ncols": 1,
    "areas": [
        {"key": "panel1", "row": 0, "col": 0, "role": "time"},
        {"key": "labels", "row": 1, "col": 0, "role": "labels"},
    ],
}

COLS = ("a", "b", "c", "d", "e", "f", "g")


class _Var(object):
    def __init__(self):
        self._v = ""

    def set(self, v):
        self._v = str(v)

    def get(self):
        return self._v


def ts(hhmmss):
    return pd.Timestamp(DAY + hhmmss)


def frame():
    idx = pd.date_range(DAY + "00:00:00", periods=120, freq="30s")
    data = {}
    for k, name in enumerate(COLS):
        data[name] = np.linspace(float(k), float(k) + 1.0, len(idx))
    return pd.DataFrame(data, index=idx)


def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["a"])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    import tkinter.messagebox as mb
    calls = []
    for kind in ("showinfo", "showwarning", "showerror", "askyesno",
                 "askyesnocancel", "askokcancel"):
        monkeypatch.setattr(
            mb, kind, lambda *a, _k=kind, **kw: calls.append(_k) or True)
    return calls


@pytest.fixture
def host(tmp_path):
    """A GUI-free labeler: the export path needs no window. The ACTIVE
    lane is Wake, and the Agent lane covers nearly the whole record --
    which is what the old preview walked instead."""
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(
        df=frame(), plot_fn=plot_fn, layout_spec=dict(LAYOUT),
        window=pd.Timedelta("60min"), autosave_folder=str(tmp_path),
        tracks=[dict(REGION), dict(WAKE), dict(AGENT)])
    lbl.status_var = _Var()
    lbl._update_plot = lambda: None
    lbl.intervals[:] = [
        Interval(ts("00:00:00"), ts("00:40:00"), "0", None, track="agent"),
        Interval(ts("00:40:00"), ts("01:00:00"), "1", None, track="agent"),
        Interval(ts("00:05:00"), ts("00:12:00"), "plasma_sheet", None,
                 track="region"),
        Interval(ts("00:10:00"), ts("00:15:00"), "umbra", None,
                 track="wake"),
    ]
    lbl._active_track_id = "wake"
    return lbl


# ================================================ 1. the words


def test_the_radio_says_the_ruled_words():
    import inspect
    from chronotagger.labeler.mixins.io_export import IOExportMixin
    src = inspect.getsource(IOExportMixin._export_labels_dialog)
    assert 'text="Labeled rows only (active lane)"' in src
    assert "Selected intervals only" not in src
    assert 'value="selected"' in src, "the wire VALUE does not change"


def test_the_docstring_says_the_same_words():
    from chronotagger.labeler.mixins.io_export import IOExportMixin
    doc = IOExportMixin._export_labels_dialog.__doc__
    assert "Scope: Full dataset  |  Labeled rows only (active lane)" in doc


def test_the_README_bullet_says_the_same_words():
    """The one live doc anchor. README.md:494 is what a reader meets
    before the dialog, and it is beside a screenshot the pack cannot
    edit -- so the bullet at least has to be true."""
    import os
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(repo, "README.md"), "rb").read().decode(
        "utf-8")
    assert "full dataset or labeled rows only (active lane)" in text
    assert "selected intervals only" not in text.lower()


def test_the_scope_line_names_the_ACTIVE_LANE(host):
    pv, total, info = host._generate_export_preview(
        "selected", "index_labels_csv")
    text = host._format_dataframe_preview(pv, total, info)
    assert "  Scope: Labeled rows only (active lane: Wake)" in text


def test_the_full_scope_line_is_unchanged(host):
    pv, total, info = host._generate_export_preview(
        "full", "index_labels_csv")
    text = host._format_dataframe_preview(pv, total, info)
    assert "  Scope: Full dataset" in text


# ================================= 2. the preview shows the ACTIVE lane


def test_every_previewed_row_is_a_labeled_row_of_the_ACTIVE_lane(host):
    pv, total, info = host._generate_export_preview(
        "selected", "index_labels_csv")
    col = info["label_column"]
    assert col == "label_id__wake"
    assert list(pv.columns) == [col]
    assert len(pv) > 0
    assert (pv[col].values != -1).all(), \
        "a scope whose file can never hold -1 must not preview -1"
    wake = [iv for iv in host.intervals if iv.track == "wake"][0]
    assert pv.index.min() >= wake.start
    assert pv.index.max() < wake.end


def test_the_limit_counts_ROWS_not_interval_chunks(host):
    pv, total, info = host._generate_export_preview(
        "selected", "index_labels_csv", limit=3)
    assert len(pv) == 3


def test_the_total_is_the_row_count_the_writer_writes(host, tmp_path):
    pv, total, info = host._generate_export_preview(
        "selected", "index_labels_csv")
    out = tmp_path / "sel.csv"
    assert host._export_labels_do(str(out), "selected",
                                  "index_labels_csv") is True
    written = len(pd.read_csv(out))
    assert total == written
    assert total <= len(host.df.index), \
        "a cross-lane sum used to exceed the whole frame"


def test_the_previews_label_column_is_the_FILES_column(host, tmp_path):
    pv, total, info = host._generate_export_preview(
        "selected", "index_labels_csv")
    out = tmp_path / "sel.csv"
    host._export_labels_do(str(out), "selected", "index_labels_csv")
    cols = list(pd.read_csv(out).columns)
    assert info["label_column"] in cols
    assert "label_id" not in cols, "no lane here is called 'default'"


def test_the_wide_preview_keeps_the_label_column(host):
    pv, total, info = host._generate_export_preview(
        "selected", "full_df_labels_csv")
    assert len(pv.columns) > 6, "wide enough to trigger the column chooser"
    text = host._format_dataframe_preview(pv, total, info)
    assert "Showing 6 of %d columns" % (len(pv.columns),) in text
    assert info["label_column"] in text


def test_the_full_scope_preview_is_untouched(host):
    pv, total, info = host._generate_export_preview(
        "full", "index_labels_csv")
    assert total == len(host.df.index)
    assert len(pv) == 10
    assert list(pv.columns) == ["label_id__wake"]
    assert (pv["label_id__wake"].values == -1).all(), \
        "the full scope really does carry unlabeled rows"


# ======================================= 3. the size estimate is honest


@pytest.mark.parametrize("content", ["index_labels_csv",
                                     "full_df_labels_csv"])
@pytest.mark.parametrize("scope", ["full", "selected"])
def test_the_size_estimate_is_within_a_factor_of_two(host, tmp_path,
                                                     scope, content):
    pv, total, info = host._generate_export_preview(scope, content)
    out = tmp_path / ("e_%s_%s.csv" % (scope, content))
    assert host._export_labels_do(str(out), scope, content) is True
    actual = out.stat().st_size
    est = total * info["est_bytes_per_row"]
    assert 0.5 <= (float(est) / float(actual)) <= 2.0, (est, actual)


def test_the_estimate_counts_every_lanes_label_column(host):
    pv, total, info = host._generate_export_preview(
        "selected", "index_labels_csv")
    # 30 for the ISO stamp, 3 for each of the three lanes' label ids
    assert info["est_bytes_per_row"] == 30 + 3 * 3
    pv, total, info = host._generate_export_preview(
        "selected", "full_df_labels_csv")
    assert info["est_bytes_per_row"] == 30 + 3 * 3 + 20 * len(COLS)
