"""
Pack M1 pins -- EXPORT AND INGEST: the per-track columns and sidecars, the
ACCEPTANCE FLOOR (a single-track export is byte-for-byte the old one), the
orphan gates on the paths a driver script calls, the one per-sample
writer, and the column -> locked track ingest.
"""

import json

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from chronotagger.core.ingest import (
    MAX_SPLIT_FRACTION,
    default_gap_tolerance,
    index_spacings,
    index_unit_epsilon,
    intervals_from_column,
)
from chronotagger.core.models import Interval
from chronotagger.core.tracks import (
    DEFAULT_TRACK_ID,
    Track,
    default_table,
)
from chronotagger.labeler.mixins.io_export import (
    label_column_collision,
    label_id_column,
    label_id_series_for_track,
    label_map_stem_suffix,
    orphan_labels_by_track,
    refuse_export_orphans,
)


# --------------------------------------------------------------- harness

class _Var:
    def __init__(self, v=""):
        self._v = v

    def set(self, v):
        self._v = v

    def get(self):
        return self._v


def _plot_fn(axs, sub, t0, t1):
    pass


def _labeler(df, **kw):
    """A real labeler with NO GUI built: enough for every export and
    ingest path, and _update_plot is stubbed because _undo calls it."""
    from chronotagger.labeler import TimeIntervalLabeler

    kw.setdefault("autosave_folder", ".")
    lbl = TimeIntervalLabeler(df=df, plot_fn=_plot_fn, **kw)
    lbl.status_var = _Var()
    lbl._update_plot = lambda: None
    return lbl


def _frame(periods=240, freq="1min", extra=None):
    idx = pd.date_range("2024-01-01 00:00:00", periods=periods, freq=freq)
    data = {"a": np.linspace(0.0, 1.0, periods)}
    if extra:
        data.update(extra)
    return pd.DataFrame(data, index=idx)


def T(hhmm):
    return pd.Timestamp("2024-01-01 %s:00" % hhmm)


def _pre_m1_label_id_series(host):
    """THE PRE-PACK-M1 WRITER, quoted verbatim from io_export.py at
    b5431d5 + M0: flat over every interval, ids from self.classes,
    last-writer-wins, smallest int dtype, searchsorted on a monotonic
    index. The acceptance floor is measured against THIS.
    """
    label_to_id = {label: i for i, label in enumerate(host.classes)}
    unknown_id = -1
    n = len(label_to_id)
    if n <= np.iinfo(np.int8).max:
        dtype = np.int8
    elif n <= np.iinfo(np.int16).max:
        dtype = np.int16
    else:
        dtype = np.int32
    idx = host.df.index
    ids = np.full(len(idx), fill_value=unknown_id, dtype=dtype)
    monotonic = bool(getattr(idx, "is_monotonic_increasing", False))
    for iv in host.intervals:
        code = label_to_id.get(iv.label, unknown_id)
        if monotonic:
            s = idx.searchsorted(iv.start, side="left")
            e = idx.searchsorted(iv.end, side="left")
            if s < e:
                ids[s:e] = code
        else:
            mask = (idx >= iv.start) & (idx < iv.end)
            if mask.any():
                ids[mask] = code
    return pd.Series(ids, index=idx, name="label_id")


# ======================================== THE ACCEPTANCE FLOOR (R2 / F2)

def test_a_single_track_per_sample_export_is_the_old_writer_exactly(
        tmp_path):
    lbl = _labeler(_frame(), classes=["UNKNOWN", "PS", "LOBE"])
    idx = lbl.df.index
    lbl.intervals = [
        Interval(idx[10], idx[40], "PS"),
        Interval(idx[100], idx[160], "LOBE"),
        Interval(idx[200], idx[210], "UNKNOWN"),
    ]
    old = _pre_m1_label_id_series(lbl)
    new = label_id_series_for_track(lbl.intervals, lbl.df.index,
                                    lbl.tracks[0])
    assert new.name == "label_id" == old.name
    assert new.dtype == old.dtype
    assert list(new.values) == list(old.values)
    assert new.index.equals(old.index)


def test_a_single_track_export_writes_the_same_bytes_as_the_old_frame(
        tmp_path):
    """The FILE, not just the series -- parquet, CSV and the sidecar.

    The reference file is written from the pre-M1 frame through the same
    atomic path, so any difference is this pack's and not pandas'. The
    same-frame-twice determinism control in the same test is what makes
    "byte-identical" a meaningful assertion rather than an accident: on
    the real 115,339-row ARTEMIS frame the measured shas are
    c14add15... (parquet) and 0ce94cae... (csv).
    """
    from chronotagger.labeler.utils.atomic_io import atomic_write_path

    lbl = _labeler(_frame(), classes=["UNKNOWN", "PS", "LOBE"])
    idx = lbl.df.index
    lbl.intervals = [Interval(idx[10], idx[40], "PS"),
                     Interval(idx[100], idx[160], "LOBE")]

    old_frame = pd.DataFrame({"label_id": _pre_m1_label_id_series(lbl)},
                             index=lbl.df.index)
    for fmt, writer in (("parquet", lambda p: old_frame.to_parquet(p)),
                        ("csv", lambda p: old_frame.to_csv(p))):
        ref = tmp_path / ("ref." + fmt)
        atomic_write_path(str(ref), writer, sync_dir=True)
        ref2 = tmp_path / ("ref2." + fmt)
        atomic_write_path(str(ref2), writer, sync_dir=True)
        # determinism control: the writer is reproducible here
        assert ref.read_bytes() == ref2.read_bytes()

        out = tmp_path / ("out." + fmt)
        lbl.export_per_sample(str(out), fmt=fmt)
        assert out.read_bytes() == ref.read_bytes()

    sidecar = tmp_path / "out_label_map.json"
    assert sidecar.exists()
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {
        "UNKNOWN": 0, "PS": 1, "LOBE": 2}
    assert not list(tmp_path.glob("*label_map__*.json"))


def test_the_column_and_sidecar_names_are_the_default_tracks_only(tmp_path):
    assert label_id_column(DEFAULT_TRACK_ID) == "label_id"
    assert label_id_column("quality") == "label_id__quality"
    assert label_map_stem_suffix(DEFAULT_TRACK_ID) == ""
    assert label_map_stem_suffix("quality") == "__quality"


# ============================================ two tracks, two columns

def test_two_tracks_write_one_column_and_one_sidecar_each(tmp_path):
    lbl = _labeler(_frame(), tracks=[
        {"id": DEFAULT_TRACK_ID, "classes": ["UNKNOWN", "sw", "msh"]},
        {"id": "quality", "classes": ["UNKNOWN", "bad"]},
    ])
    idx = lbl.df.index
    lbl.intervals = [
        Interval(idx[10], idx[40], "sw"),
        Interval(idx[20], idx[60], "bad", track="quality"),
    ]
    out = tmp_path / "two.csv"
    lbl.export_per_sample(str(out), fmt="csv")
    back = pd.read_csv(out, index_col=0)
    assert list(back.columns) == ["label_id", "label_id__quality"]
    assert (tmp_path / "two_label_map.json").exists()
    assert (tmp_path / "two_label_map__quality.json").exists()
    assert json.loads((tmp_path / "two_label_map__quality.json").read_text(
        encoding="utf-8")) == {"UNKNOWN": 0, "bad": 1}


def test_each_track_column_equals_that_tracks_intervals_alone(tmp_path):
    """Candidate D is not an approximation of the old contract -- it IS
    the old contract, once per column."""
    lbl = _labeler(_frame(), tracks=[
        {"id": DEFAULT_TRACK_ID, "classes": ["UNKNOWN", "sw"]},
        {"id": "quality", "classes": ["UNKNOWN", "bad"]},
    ])
    idx = lbl.df.index
    lbl.intervals = [
        Interval(idx[10], idx[40], "sw"),
        Interval(idx[20], idx[60], "bad", track="quality"),
    ]
    for trk in lbl.tracks:
        mine = label_id_series_for_track(lbl.intervals,
                                        lbl.df.index, trk)
        solo = _labeler(lbl.df, classes=list(trk.classes))
        solo.intervals = [
            Interval(iv.start, iv.end, iv.label)
            for iv in lbl.intervals if iv.track == trk.id]
        assert list(mine.values) == list(
            _pre_m1_label_id_series(solo).values)


def test_two_tracks_can_carry_different_dtypes(tmp_path):
    """The dtype follows each track's own class count. Today's one-column
    contract never had to say this."""
    lbl = _labeler(_frame(), tracks=[
        {"id": DEFAULT_TRACK_ID, "classes": ["UNKNOWN", "sw"]},
        {"id": "many", "classes": ["c%d" % i for i in range(200)]},
    ])
    idx = lbl.df.index
    lbl.intervals = [Interval(idx[10], idx[40], "sw"),
                     Interval(idx[10], idx[40], "c7", track="many")]
    assert label_id_series_for_track(
        lbl.intervals, lbl.df.index, lbl.tracks[0]).dtype == np.int8
    assert label_id_series_for_track(
        lbl.intervals, lbl.df.index, lbl.tracks[1]).dtype == np.int16


# =========================================== export_intervals + track

def test_export_intervals_carries_a_track_column_in_both_builders(tmp_path):
    lbl = _labeler(_frame(), tracks=[
        {"id": DEFAULT_TRACK_ID, "classes": ["UNKNOWN", "sw"]},
        {"id": "quality", "classes": ["bad"]},
    ])
    idx = lbl.df.index
    lbl.intervals = [Interval(idx[10], idx[40], "sw"),
                     Interval(idx[50], idx[60], "bad", track="quality")]
    out = tmp_path / "iv.csv"
    lbl.export_intervals(str(out), fmt="csv")
    header = out.read_text().splitlines()[0]
    assert header.startswith("start,end,track,label,notes")
    back = pd.read_csv(out)
    assert list(back["track"]) == [DEFAULT_TRACK_ID, "quality"]

    # the GUI twin is a SECOND, independent row builder: same contract
    gui_out = tmp_path / "gui.csv"
    with patch("tkinter.filedialog.asksaveasfilename",
               return_value=str(gui_out)), \
            patch("tkinter.messagebox.showinfo"):
        lbl._export_intervals()
    assert gui_out.read_text().splitlines()[0].startswith(
        "start,end,track,label,notes")
    assert pd.read_csv(gui_out)["track"].tolist() == [DEFAULT_TRACK_ID,
                                                     "quality"]


# ================================================== the orphan gates

def test_the_programmatic_exports_refuse_an_orphan_label(tmp_path):
    """These two paths had NO gate at all: export_per_sample wrote 85 rows
    as -1 with zero dialogs and the orphan absent from the sidecar."""
    lbl = _labeler(_frame(), classes=["UNKNOWN", "PS"])
    idx = lbl.df.index
    lbl.intervals = [Interval(idx[10], idx[20], "NOT_A_CLASS")]
    with pytest.raises(ValueError) as e1:
        lbl.export_per_sample(str(tmp_path / "x.csv"), fmt="csv")
    assert "NOT_A_CLASS" in str(e1.value)
    assert DEFAULT_TRACK_ID in str(e1.value)
    with pytest.raises(ValueError):
        lbl.export_intervals(str(tmp_path / "y.csv"), fmt="csv")
    assert not (tmp_path / "x.csv").exists()
    assert not (tmp_path / "y.csv").exists()


def test_the_programmatic_exports_refuse_an_orphan_TRACK(tmp_path):
    """A separate refusal with its own sentence: an interval on a track
    that does not exist has no class set to check its label against, so
    the label gate cannot even run on it."""
    lbl = _labeler(_frame(), classes=["UNKNOWN", "PS"])
    idx = lbl.df.index
    lbl.intervals = [Interval(idx[10], idx[20], "PS", track="ghost")]
    with pytest.raises(ValueError) as e:
        lbl.export_per_sample(str(tmp_path / "x.csv"), fmt="csv")
    assert "ghost" in str(e.value)
    assert "not in the track table" in str(e.value)


def test_a_label_that_is_legal_in_its_own_track_does_not_block_anything(
        tmp_path):
    """The FLAT gate did not merely under-report: it BLOCKED EVERY EXPORT
    the moment two tracks had different vocabularies."""
    lbl = _labeler(_frame(), tracks=[
        {"id": DEFAULT_TRACK_ID, "classes": ["UNKNOWN", "sw"]},
        {"id": "geom", "classes": ["umbra"]},
    ])
    idx = lbl.df.index
    lbl.intervals = [Interval(idx[10], idx[40], "sw"),
                     Interval(idx[10], idx[40], "umbra", track="geom")]
    by_track, orphan_tracks = orphan_labels_by_track(lbl.intervals,
                                                     lbl.tracks)
    assert by_track == {} and orphan_tracks == []
    lbl.export_per_sample(str(tmp_path / "ok.csv"), fmt="csv")
    assert (tmp_path / "ok.csv").exists()
    refuse_export_orphans(lbl.intervals, lbl.tracks, "probe")   # no raise


def test_a_source_column_named_label_id_refuses_the_full_frame_export(
        tmp_path):
    """Today it is SILENTLY overwritten. The ingest makes the collision
    likely rather than exotic: ingest the label_id column from a model's
    predictions, then export the full frame plus the hand labels."""
    df = _frame(extra={"label_id": np.arange(240) % 3})
    lbl = _labeler(df, classes=["UNKNOWN", "PS"])
    idx = lbl.df.index
    lbl.intervals = [Interval(idx[10], idx[20], "PS")]
    assert label_column_collision(lbl.df, lbl.tracks) == ["label_id"]
    out = tmp_path / "full.csv"
    with patch("tkinter.messagebox.showerror") as err:
        ok = lbl._export_labels_do(str(out), "full", "full_df_labels_csv")
    assert ok is False
    assert err.call_count == 1
    assert "label_id" in err.call_args[0][1]
    assert not out.exists()
    # index + labels only is still allowed: nothing is overwritten there
    ok2 = lbl._export_labels_do(str(tmp_path / "idx.csv"), "full",
                                "index_labels_csv")
    assert ok2 is True


# ============================================== one writer, one preview

def test_the_preview_is_a_slice_of_the_real_writer(tmp_path):
    """The preview used to be a SECOND implementation with a DIFFERENT
    arbitration rule -- FIRST-match-wins against LAST-writer-wins -- and
    on an overlapping pair the two disagreed on 20 of 60 previewed rows."""
    lbl = _labeler(_frame(), classes=["UNKNOWN", "PS", "LOBE"])
    idx = lbl.df.index
    lbl.intervals = [Interval(idx[10], idx[40], "PS"),
                     Interval(idx[100], idx[160], "LOBE")]
    writer = label_id_series_for_track(lbl.intervals, lbl.df.index,
                                       lbl.tracks[0])
    for scope in ("full", "selected"):
        preview_df, _total, _info = lbl._generate_export_preview(
            scope, "index_labels_csv", limit=10)
        for ts, value in preview_df["label_id"].items():
            assert value == writer.loc[ts]


def test_the_preview_survives_a_DUPLICATED_index(tmp_path):
    """_get_first_labeled_rows concatenates rows interval by interval, so
    once cross-track overlap is legal its index can carry DUPLICATES --
    and a duplicated index makes searchsorted slices ambiguous. This is a
    real edge the preview has never had to face."""
    lbl = _labeler(_frame(), tracks=[
        {"id": DEFAULT_TRACK_ID, "classes": ["UNKNOWN", "sw"]},
        {"id": "rules", "classes": ["umbra"]},
    ])
    idx = lbl.df.index
    # two intervals covering the SAME rows on DIFFERENT lanes: the preview
    # builder walks both and hands the same timestamps over twice
    lbl.intervals = [Interval(idx[10], idx[40], "sw"),
                     Interval(idx[10], idx[40], "umbra", track="rules")]
    raw, _total = lbl._get_first_labeled_rows(10)
    assert not raw.index.is_unique          # the edge is real, not theory
    preview_df, _t, _i = lbl._generate_export_preview(
        "selected", "index_labels_csv", limit=10)
    assert preview_df.index.is_unique
    writer = label_id_series_for_track(lbl.intervals, lbl.df.index,
                                       lbl.tracks[0])
    for ts, value in preview_df["label_id"].items():
        assert value == writer.loc[ts]


def test_the_ingest_refuses_a_column_that_is_not_aligned():
    idx = pd.date_range("2016-01-01", periods=30, freq="60s")
    with pytest.raises(ValueError) as e:
        intervals_from_column(idx, ["sw"] * 10, "rules")
    assert "aligned" in str(e.value)


def test_the_end_cap_follows_the_indexs_own_resolution():
    """pandas 3.0 hands out MICROSECOND DatetimeIndexes by default, and on
    such an index a hardcoded +1 ns cap is UNREPRESENTABLE -- the Timestamp
    promotes to ns and searchsorted raises. Pack 6.5 R65-1, quoted into
    the ingest so it needs no labeler."""
    ns = pd.date_range("2016-01-01", periods=5, freq="60s").as_unit("ns")
    us = pd.date_range("2016-01-01", periods=5, freq="60s").as_unit("us")
    assert index_unit_epsilon(ns) == pd.Timedelta(1, unit="ns")
    assert index_unit_epsilon(us) == pd.Timedelta(1, unit="us")
    for idx in (ns, us):
        ivs, _info = intervals_from_column(idx, ["sw"] * 5, "rules")
        assert len(ivs) == 1
        assert ivs[0].end == idx[-1] + index_unit_epsilon(idx)
        # and the cap is representable: searchsorted does not raise
        assert idx.searchsorted(ivs[0].end, side="left") == 5


def test_a_gap_split_fragment_is_a_run_wide_plus_its_own_spacing():
    """Not one index unit (which paints zero pixels) and not the next
    sample (which the fuse re-merges): the run's OWN local spacing."""
    a = pd.date_range("2016-01-01 00:00:00", periods=10, freq="138s")
    b = pd.date_range(a[-1] + pd.Timedelta(hours=2), periods=10, freq="138s")
    idx = a.append(b)
    ivs, info = intervals_from_column(idx, ["sw"] * 20, "rules",
                                      gap_tolerance=pd.Timedelta("10min"))
    assert info["splits"] == 1
    assert len(ivs) == 2
    first = sorted(ivs, key=lambda iv: iv.start)[0]
    assert first.end == a[-1] + pd.Timedelta(seconds=138)
    assert first.end < b[0]                     # strictly inside the gap


def test_the_second_per_sample_implementation_is_gone():
    lbl = _labeler(_frame(), classes=["UNKNOWN", "PS"])
    assert not hasattr(lbl, "_compute_label_id_series_for_subset")
    import chronotagger.labeler.mixins.io_export as io_mod
    src = open(io_mod.__file__, "r", encoding="utf-8").read()
    assert "def _compute_label_id_series_for_subset" not in src


def test_compute_label_id_series_is_the_active_tracks_answer():
    """The flat version was measured to write 1,200 of 2,400
    human-labeled rows as -1, because an imported track's interval
    overwrote them with a class the active track does not contain."""
    lbl = _labeler(_frame(), tracks=[
        {"id": DEFAULT_TRACK_ID, "classes": ["UNKNOWN", "sw"]},
        {"id": "rules", "classes": ["umbra"]},
    ])
    idx = lbl.df.index
    lbl.intervals = [Interval(idx[10], idx[40], "sw"),
                     Interval(idx[0], idx[239], "umbra", track="rules")]
    ids = lbl._compute_label_id_series()
    assert ids.name == "label_id"
    assert set(np.unique(ids.values)) == {-1, 1}
    assert int((ids.values == 1).sum()) == 30


# ==================================================== THE COLUMN INGEST

def _bimodal_index(fast=200, slow=100, fast2=200):
    """The shape of the user's ARTEMIS ESA cadence: a long fast-survey
    stretch at 138 s, a long slow-survey stretch at 550 s, then fast
    again. The median sits BELOW the slow mode; p95 sits above both."""
    t = pd.Timestamp("2016-01-01 00:00:00")
    out = [t]
    for _ in range(fast - 1):
        out.append(out[-1] + pd.Timedelta(seconds=138))
    for _ in range(slow):
        out.append(out[-1] + pd.Timedelta(seconds=550))
    for _ in range(fast2):
        out.append(out[-1] + pd.Timedelta(seconds=138))
    return pd.DatetimeIndex(out)


def test_the_default_gap_tolerance_is_3x_p95_and_not_k_x_median():
    """THE measurement that decided the default: `2 x median` sits BETWEEN
    the two cadence modes and splits at every survey-mode change --
    34,428 intervals instead of 2,349 on the real column, and 34,155
    "gaps" that are not data gaps at all."""
    idx = _bimodal_index()
    d = index_spacings(idx)
    assert pd.Timedelta(d.median()) == pd.Timedelta(seconds=138)
    assert pd.Timedelta(d.quantile(0.95)) == pd.Timedelta(seconds=550)
    tol = default_gap_tolerance(idx)
    assert tol == pd.Timedelta(seconds=550) * 3
    assert tol > pd.Timedelta(seconds=550)          # above BOTH modes
    assert pd.Timedelta(d.median()) * 2 < pd.Timedelta(seconds=550)


def test_the_median_default_would_shred_the_column_and_is_refused():
    idx = _bimodal_index()
    values = ["sw"] * len(idx)
    # the default sees ONE run and no split
    ivs, info = intervals_from_column(idx, values, "rules")
    assert info["splits"] == 0
    assert len(ivs) == 1
    # 2 x median splits at every slow-survey gap, and the guard refuses
    with pytest.raises(ValueError) as e:
        intervals_from_column(idx, values, "rules",
                              gap_tolerance=pd.Timedelta(seconds=276))
    assert "would split" in str(e.value)
    assert "100 of 1 runs" in str(e.value)
    assert MAX_SPLIT_FRACTION == 0.10


def test_the_ingest_reproduces_the_column_exactly():
    """end-on-the-LAST-sample gives one dropped row per run: 2,349
    intervals and 2,349 mismatches. end-on-the-NEXT-sample gives zero."""
    idx = pd.date_range("2016-01-01", periods=300, freq="138s")
    codes = ([-1] * 20 + ["sw"] * 80 + ["msh"] * 50 + [-1] * 10
             + ["sw"] * 90 + ["umbra"] * 50)
    assert len(codes) == 300
    lbl = _labeler(pd.DataFrame({"rule": codes}, index=idx),
                   classes=["UNKNOWN"])
    row = lbl.add_track_from_column("rule", "rules")
    series = label_id_series_for_track(lbl.intervals,
                                       lbl.df.index, row)
    expected = []
    for v in codes:
        expected.append(-1 if v == -1 else row.classes.index(str(v)))
    assert list(series.values) == expected


def test_a_gap_split_fragment_survives_the_adjacency_fuse():
    """With the end on the next sample the fuse re-merged 70 of 71
    splits. Last sample plus the run's own local spacing survives it AND
    renders -- _end_after_inclusive survives it and paints zero pixels."""
    a = pd.date_range("2016-01-01 00:00:00", periods=100, freq="138s")
    b = pd.date_range(a[-1] + pd.Timedelta(hours=2), periods=100,
                      freq="138s")
    idx = a.append(b)
    lbl = _labeler(pd.DataFrame({"rule": ["sw"] * 200}, index=idx),
                   classes=["UNKNOWN"])
    row = lbl.add_track_from_column("rule", "rules")
    on_track = [iv for iv in lbl.intervals if iv.track == "rules"]
    assert len(on_track) == 2
    lbl._sort_and_merge_intervals()
    assert len([iv for iv in lbl.intervals if iv.track == "rules"]) == 2
    # ... and neither fragment is a one-nanosecond sliver
    eps = index_unit_epsilon(idx)
    for iv in on_track:
        assert (iv.end - iv.start) > eps * 2
    # the column still round-trips exactly
    series = label_id_series_for_track(lbl.intervals,
                                       lbl.df.index, row)
    assert set(np.unique(series.values)) == {0}


def test_the_ingest_skips_unlabeled_samples_and_nan():
    idx = pd.date_range("2016-01-01", periods=10, freq="60s")
    values = [np.nan, np.nan, "sw", "sw", -1, None, "msh", "msh", "msh",
              np.nan]
    ivs, info = intervals_from_column(idx, values, "rules")
    assert [iv.label for iv in ivs] == ["sw", "msh"]
    assert info["unlabeled_rows"] == 5
    assert info["classes"] == ["msh", "sw"]
    # NaN never becomes a class literally named "nan"
    assert "nan" not in info["classes"]


def test_the_ingest_stamps_provenance_into_meta():
    idx = pd.date_range("2016-01-01", periods=6, freq="60s")
    ivs, info = intervals_from_column(
        idx, ["sw"] * 3 + ["msh"] * 3, "rules", source="column:rule_label")
    for iv in ivs:
        assert iv.meta["import.source"] == "column:rule_label"
        assert iv.meta["import.gap_tolerance"].startswith("P")
        assert iv.meta["import.run_rows"] == 3
        assert iv.track == "rules"


def test_the_ingested_track_is_locked_and_rides_in_the_autosave(tmp_path):
    idx = pd.date_range("2016-01-01", periods=30, freq="60s")
    lbl = _labeler(pd.DataFrame({"rule": ["sw"] * 15 + ["msh"] * 15},
                                index=idx),
                   classes=["UNKNOWN"])
    row = lbl.add_track_from_column("rule", "rules", name="Rule labels")
    assert row.locked is True
    assert row.name == "Rule labels"
    assert [t.id for t in lbl.tracks] == [DEFAULT_TRACK_ID, "rules"]
    assert lbl.autosave_file.exists()
    data = json.loads(lbl.autosave_file.read_text(encoding="utf-8"))
    assert [t["id"] for t in data["tracks"]] == [DEFAULT_TRACK_ID, "rules"]
    assert data["tracks"][1]["locked"] is True
    assert {d["track"] for d in data["intervals"]} == {"rules"}
    assert data["intervals"][0]["meta"]["import.source"] == "column:rule"


def test_the_ingest_is_one_undoable_gesture():
    idx = pd.date_range("2016-01-01", periods=30, freq="60s")
    lbl = _labeler(pd.DataFrame({"rule": ["sw"] * 15 + ["msh"] * 15},
                                index=idx), classes=["UNKNOWN"])
    lbl.add_track_from_column("rule", "rules")
    assert len(lbl.undo_stack) == 1
    lbl._undo()
    assert [t.id for t in lbl.tracks] == [DEFAULT_TRACK_ID]
    assert lbl.intervals == []
    lbl._redo()
    assert [t.id for t in lbl.tracks] == [DEFAULT_TRACK_ID, "rules"]
    assert len(lbl.intervals) == 2


def test_ingesting_into_an_existing_track_refuses_an_unknown_label():
    idx = pd.date_range("2016-01-01", periods=30, freq="60s")
    lbl = _labeler(pd.DataFrame({"rule": ["sw"] * 15 + ["msh"] * 15},
                                index=idx),
                   tracks=[{"id": DEFAULT_TRACK_ID, "classes": ["UNKNOWN"]},
                           {"id": "rules", "classes": ["sw"]}])
    with pytest.raises(ValueError) as e:
        lbl.add_track_from_column("rule", "rules")
    assert "msh" in str(e.value)
    assert lbl.intervals == []
    assert [t.classes for t in lbl.tracks] == [["UNKNOWN"], ["sw"]]


def test_the_ingest_never_touches_the_frame_or_the_fingerprint():
    """Adding one column moves the fingerprint, which changes the autosave
    FILENAME, which makes every prior autosave unreachable."""
    idx = pd.date_range("2016-01-01", periods=30, freq="60s")
    lbl = _labeler(pd.DataFrame({"rule": ["sw"] * 30}, index=idx),
                   classes=["UNKNOWN"])
    before_cols = list(lbl.df.columns)
    before_fp = lbl._dataset_fingerprint()
    before_name = lbl.autosave_file.name
    lbl.add_track_from_column("rule", "rules")
    assert list(lbl.df.columns) == before_cols
    assert lbl._dataset_fingerprint() == before_fp
    assert lbl.autosave_file.name == before_name


def test_a_bad_track_id_is_refused_before_anything_changes():
    idx = pd.date_range("2016-01-01", periods=30, freq="60s")
    lbl = _labeler(pd.DataFrame({"rule": ["sw"] * 30}, index=idx),
                   classes=["UNKNOWN"])
    for bad in ["a/b", "a.b", "has space", "x" * 33, ""]:
        with pytest.raises(ValueError):
            lbl.add_track_from_column("rule", bad)
    assert [t.id for t in lbl.tracks] == [DEFAULT_TRACK_ID]
    assert lbl.intervals == []
    with pytest.raises(ValueError):
        lbl.add_track_from_column("no_such_column", "rules")


def test_the_ingest_accepts_a_series_and_a_class_order():
    idx = pd.date_range("2016-01-01", periods=30, freq="60s")
    lbl = _labeler(pd.DataFrame({"a": np.arange(30)}, index=idx),
                   classes=["UNKNOWN"])
    col = pd.Series(["msh"] * 10 + ["sw"] * 20, index=idx, name="model")
    row = lbl.add_track_from_column(col, "model",
                                    class_order=["sw", "msh"])
    assert row.classes == ["sw", "msh"]
    assert row.class_colors["sw"].startswith("#")
    assert {iv.meta["import.source"] for iv in lbl.intervals} == {
        "column:model"}
    with pytest.raises(ValueError):
        lbl.add_track_from_column(pd.Series(["sw"] * 5), "short")

# ===================================== LANE-2 FOLD PINS (verifier addition)
# Seven pins for the two GUI export copies, the preview's frame de-dup and
# the fuse's pre-sort -- the surfaces EDITs 453, 454, 455 and 425 change and
# which no pin in v1 of this pack reached: measured, nine single-line
# reverts of those four edits left all 110 v1 pins GREEN.

class _StopAtDialog(Exception):
    """Raised in place of tk.Toplevel so the DOOR can be tested head-less."""


def test_the_export_DIALOG_DOOR_is_per_track_not_flat(tmp_path):
    """EDIT 453. The flat door did not merely under-report: measured, it
    BLOCKED EVERY EXPORT the moment two tracks had different
    vocabularies. tk.Toplevel is replaced so the gate is all that runs."""
    lbl = _labeler(_frame(), tracks=[
        {"id": DEFAULT_TRACK_ID, "classes": ["UNKNOWN", "sw"]},
        {"id": "geom", "classes": ["umbra"]},
    ])
    idx = lbl.df.index
    # (a) a label legal in its OWN lane must not raise the door's dialog
    lbl.intervals = [Interval(idx[10], idx[40], "sw"),
                     Interval(idx[10], idx[40], "umbra", track="geom")]
    with patch("tkinter.messagebox.showerror") as err, \
            patch("tkinter.Toplevel", side_effect=_StopAtDialog) as top:
        with pytest.raises(_StopAtDialog):
            lbl._export_labels_dialog()
    assert err.call_count == 0
    assert top.call_count == 1
    # (b) a label legal in NO lane must be refused AT THE DOOR, once,
    #     naming the lane, and the dialog must never open
    lbl.intervals = [Interval(idx[10], idx[40], "NOT_A_CLASS")]
    with patch("tkinter.messagebox.showerror") as err2, \
            patch("tkinter.Toplevel", side_effect=_StopAtDialog) as top2:
        lbl._export_labels_dialog()
    assert err2.call_count == 1
    assert top2.call_count == 0
    body = err2.call_args[0][1]
    assert "NOT_A_CLASS" in body and DEFAULT_TRACK_ID in body


def test_the_labels_CSV_gate_is_per_track_too(tmp_path):
    """EDIT 454's orphan gate, the GUI twin of the programmatic one."""
    lbl = _labeler(_frame(), tracks=[
        {"id": DEFAULT_TRACK_ID, "classes": ["UNKNOWN", "sw"]},
        {"id": "geom", "classes": ["umbra"]},
    ])
    idx = lbl.df.index
    lbl.intervals = [Interval(idx[10], idx[40], "sw"),
                     Interval(idx[10], idx[40], "umbra", track="geom")]
    out = tmp_path / "ok.csv"
    with patch("tkinter.messagebox.showerror") as err:
        assert lbl._export_labels_do(str(out), "full",
                                     "index_labels_csv") is True
    assert err.call_count == 0
    assert out.exists()
    lbl.intervals = [Interval(idx[10], idx[40], "umbra")]   # wrong lane
    bad = tmp_path / "bad.csv"
    with patch("tkinter.messagebox.showerror") as err2:
        assert lbl._export_labels_do(str(bad), "full",
                                     "index_labels_csv") is False
    assert err2.call_count == 1
    assert not bad.exists()


def test_the_labels_CSV_writes_one_column_and_one_sidecar_per_track(
        tmp_path):
    """EDIT 454's per-track columns and per-track sidecars. With one
    default track this is byte-for-byte the old file; with two it is two
    columns and two sidecars, and the default track's keeps both names."""
    lbl = _labeler(_frame(), tracks=[
        {"id": DEFAULT_TRACK_ID, "classes": ["UNKNOWN", "sw"]},
        {"id": "geom", "classes": ["UNKNOWN", "umbra"]},
    ])
    idx = lbl.df.index
    lbl.intervals = [Interval(idx[10], idx[40], "sw"),
                     Interval(idx[20], idx[60], "umbra", track="geom")]
    out = tmp_path / "lab.csv"
    assert lbl._export_labels_do(str(out), "full",
                                 "full_df_labels_csv") is True
    back = pd.read_csv(out, index_col=0)
    assert "label_id" in back.columns
    assert "label_id__geom" in back.columns
    assert (tmp_path / "lab_label_map.json").exists()
    assert (tmp_path / "lab_label_map__geom.json").exists()
    assert json.loads((tmp_path / "lab_label_map__geom.json").read_text(
        encoding="utf-8")) == {"UNKNOWN": 0, "umbra": 1}
    # every column is that lane's own answer, not the first lane's
    for trk in lbl.tracks:
        want = label_id_series_for_track(lbl.intervals, lbl.df.index, trk)
        assert list(back[label_id_column(trk.id)]) == list(want.values)


def test_the_selected_scope_is_the_ACTIVE_lanes_rows_and_slices_them_all(
        tmp_path):
    """EDIT 454. With K lanes "is this row labeled" has K answers; the
    ACTIVE lane decides, and every other lane's column is sliced to the
    SAME rows or the frame is ragged."""
    lbl = _labeler(_frame(), tracks=[
        {"id": DEFAULT_TRACK_ID, "classes": ["UNKNOWN", "sw"]},
        {"id": "geom", "classes": ["UNKNOWN", "umbra"]},
    ])
    idx = lbl.df.index
    lbl.intervals = [Interval(idx[10], idx[20], "sw"),
                     Interval(idx[100], idx[200], "umbra", track="geom")]
    out = tmp_path / "sel.csv"
    assert lbl._export_labels_do(str(out), "selected",
                                 "index_labels_csv") is True
    back = pd.read_csv(out, index_col=0)
    assert len(back) == 10                       # the ACTIVE lane's rows
    assert list(back.columns) == ["label_id", "label_id__geom"]
    assert set(back["label_id"]) == {1}
    assert set(back["label_id__geom"]) == {-1}   # sliced to the same rows


def test_the_full_df_preview_drops_duplicated_rows_too(tmp_path):
    """The de-dup in _generate_export_preview must de-duplicate the FRAME,
    not just the index: .loc[a de-duplicated index] on a frame whose index
    HAS duplicates returns every matching row again -- measured, 19 rows
    for 14 unique stamps, in pandas 2.3.3 and 3.0.2 alike."""
    lbl = _labeler(_frame(), tracks=[
        {"id": DEFAULT_TRACK_ID, "classes": ["UNKNOWN", "sw"]},
        {"id": "rules", "classes": ["umbra"]},
    ])
    idx = lbl.df.index
    lbl.intervals = [Interval(idx[10], idx[20], "sw"),
                     Interval(idx[15], idx[40], "umbra", track="rules")]
    raw, _total = lbl._get_first_labeled_rows(10)
    assert not raw.index.is_unique              # the edge is real
    idx_only, _t, _i = lbl._generate_export_preview(
        "selected", "index_labels_csv", limit=10)
    full_df, _t2, _i2 = lbl._generate_export_preview(
        "selected", "full_df_labels_csv", limit=10)
    assert full_df.index.is_unique
    assert len(full_df) == len(idx_only)


def test_the_selected_mask_follows_a_NON_FIRST_active_lane(tmp_path):
    """EDIT 454's `active = active_track_of(self)`. M1 ships no control
    that moves the active lane, so with the active lane always being row 0
    `tracks[0]` and `active_track_of(self)` are the same object -- which is
    exactly why this has to be pinned before M2 adds the control."""
    lbl = _labeler(_frame(), tracks=[
        {"id": DEFAULT_TRACK_ID, "classes": ["UNKNOWN", "sw"]},
        {"id": "geom", "classes": ["UNKNOWN", "umbra"]},
    ])
    idx = lbl.df.index
    lbl.intervals = [Interval(idx[10], idx[20], "sw"),
                     Interval(idx[100], idx[200], "umbra", track="geom")]
    lbl._active_track_id = "geom"
    out = tmp_path / "sel_geom.csv"
    assert lbl._export_labels_do(str(out), "selected",
                                 "index_labels_csv") is True
    back = pd.read_csv(out, index_col=0)
    assert len(back) == 100                      # the GEOM lane's rows
    assert set(back["label_id__geom"]) == {1}
    assert set(back["label_id"]) == {-1}


def test_the_fuse_still_fuses_when_the_list_is_NOT_in_start_order():
    """EDIT 425's `(track, start)` pre-sort, on the one path that hands
    this method an UNSORTED list: add_track_from_column does
    self.intervals.extend(ivs) and then calls the fuse, so ingested
    intervals that begin BEFORE the existing ones arrive out of order. A
    per-track `last` that walks the raw list order never sees the pair as
    neighbours, and two abutting same-label intervals stay split."""
    lbl = _labeler(_frame(), tracks=[
        {"id": DEFAULT_TRACK_ID, "classes": ["UNKNOWN", "sw"]},
        {"id": "geom", "classes": ["umbra"]},
    ])
    idx = lbl.df.index
    lbl.intervals = [
        Interval(idx[10], idx[20], "sw"),          # LATER one first
        Interval(idx[30], idx[40], "umbra", track="geom"),
        Interval(idx[0], idx[10], "sw"),           # abuts the first
    ]
    lbl._sort_and_merge_intervals()
    on_default = [iv for iv in lbl.intervals
                  if iv.track == DEFAULT_TRACK_ID]
    assert len(on_default) == 1
    assert on_default[0].start == idx[0]
    assert on_default[0].end == idx[20]
    assert [iv.start for iv in lbl.intervals] == sorted(
        iv.start for iv in lbl.intervals)
