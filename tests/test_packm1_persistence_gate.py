"""
Pack M1 pins -- PERSISTENCE: version 2, the gate that refuses a newer
file, the one-way v1 reader, build-validate-publish on both load paths,
and the per-track autosave accounting.

GUI-free except for the one pin that has to prove the refusal is the
FIRST dialog, which needs a real layout_spec to have a layout question to
come before.
"""

import json

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from chronotagger.core.commands import IntervalInvariantError
from chronotagger.core.models import Interval
from chronotagger.core.tracks import (
    DEFAULT_TRACK_ID,
    Track,
    default_table,
)
from chronotagger.labeler.mixins.intervals import IntervalsMixin
from chronotagger.labeler.mixins.io_export import (
    SESSION_VERSION,
    SessionVersionError,
    IOExportMixin,
    flatten_label_stats,
    label_stats_by_track,
    payload_version,
)


# --------------------------------------------------------------- harness

class _Var:
    def __init__(self, v=""):
        self._v = v

    def set(self, v):
        self._v = v

    def get(self):
        return self._v


class Host:
    """GUI-free host for the persistence mixin, with a track table."""

    _BOUND = [
        "_dataset_fingerprint", "_save_session", "_load_session",
        "_save_autosave", "_check_autosave", "_apply_recovered_autosave",
        "_compute_label_id_series",
    ]

    def __init__(self, df, folder, tracks=None, root=None):
        self.df = df
        self.intervals = []
        self.tracks = tracks if tracks is not None else default_table(
            ["UNKNOWN", "PS", "LOBE"],
            {"UNKNOWN": "#cccccc", "PS": "#ff0000", "LOBE": "#0000ff"})
        self._active_track_id = self.tracks[0].id
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo = 50
        self.modified = False
        self.selected_interval = None
        self.data_start = df.index[0]
        self.data_end = df.index[-1]
        self.window = pd.Timedelta("30min")
        self.step = pd.Timedelta("15min")
        self.t0 = df.index[0]
        self.t1 = df.index[-1]
        self.layout_spec = None
        self.source_name = None
        self.status_var = _Var()
        self.class_combo = None
        self.current_class_var = None
        self.start_time_entry = None
        self.end_time_entry = None
        self.step_entry = None
        self.root = root
        for name in self._BOUND:
            setattr(self, name, getattr(IOExportMixin, name).__get__(self))
        self._check_interval_invariants = \
            IntervalsMixin._check_interval_invariants.__get__(self)
        self._check_interval_invariants_on = \
            IntervalsMixin._check_interval_invariants_on.__get__(self)
        self.autosave_folder = folder
        self.autosave_file = folder / (
            "chronotagger_autosave_%s.json" % self._dataset_fingerprint())

    def _update_plot(self):
        pass

    @property
    def classes(self):
        from chronotagger.core.tracks import active_track_of
        return active_track_of(self).classes

    @classes.setter
    def classes(self, value):
        from chronotagger.core.tracks import active_track_of
        active_track_of(self).classes = [str(c) for c in value]

    @property
    def class_colors(self):
        from chronotagger.core.tracks import active_track_of
        return active_track_of(self).class_colors

    @class_colors.setter
    def class_colors(self, value):
        from chronotagger.core.tracks import active_track_of
        active_track_of(self).class_colors = dict(value)


def _grid(periods=100):
    idx = pd.date_range("2024-01-01 00:00:00", periods=periods, freq="1min")
    return pd.DataFrame({"a": np.linspace(0.0, 1.0, periods)}, index=idx)


def T(hhmm):
    return pd.Timestamp("2024-01-01 %s:00" % hhmm)


@pytest.fixture
def host(tmp_path):
    return Host(_grid(), tmp_path)


@pytest.fixture
def gui_host(tmp_path):
    """Same host with a non-None root, so the refusal takes the DIALOG
    channel instead of the head-less raise."""
    return Host(_grid(), tmp_path, root=object())


def _v1_session(path, classes=None, colors=None, intervals=None,
                version=1, drop_version=False):
    """A v1 session file in exactly the shape all 31 curated sessions on
    disk have: twelve top-level keys, four-key intervals, version 1."""
    classes = classes or ["UNKNOWN", "solar_wind", "umbra"]
    colors = colors or {c: "#cccccc" for c in classes}
    data = {
        "version": version,
        "classes": list(classes),
        "class_colors": dict(colors),
        "window": "0 days 00:30:00",
        "step": "0 days 00:15:00",
        "data_start": "2024-01-01T00:00:00",
        "data_end": "2024-01-01T01:39:00",
        "intervals": intervals if intervals is not None else [
            {"start": "2024-01-01T00:10:00", "end": "2024-01-01T00:20:00",
             "label": "solar_wind", "notes": "kept"},
            {"start": "2024-01-01T00:30:00", "end": "2024-01-01T00:40:00",
             "label": "umbra", "notes": None},
        ],
        "layout_spec": None,
        "multi_pane_mode": False,
        "active_pane_idx": 0,
        "panes": [],
    }
    if drop_version:
        data.pop("version")
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


# ================================================== version 2, both ways

def test_the_session_payload_is_version_2_with_a_track_table(host, tmp_path):
    host.intervals = [Interval(T("00:10"), T("00:20"), "PS", "n",
                               DEFAULT_TRACK_ID, {"k": [1]})]
    target = tmp_path / "s.json"
    assert host._save_session(str(target)) is True
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["version"] == SESSION_VERSION == 2
    assert [t["id"] for t in data["tracks"]] == [DEFAULT_TRACK_ID]
    assert data["tracks"][0]["classes"] == ["UNKNOWN", "PS", "LOBE"]
    assert data["active_track"] == DEFAULT_TRACK_ID
    # The mirror is GONE: the track table is the only schema authority.
    assert "classes" not in data and "class_colors" not in data
    assert data["intervals"][0]["track"] == DEFAULT_TRACK_ID
    assert data["intervals"][0]["meta"] == {"k": [1]}


def test_track_and_meta_survive_a_session_round_trip(host, tmp_path):
    meta = {"zzz_nonsense_key": [1, {"deep": "value"}], "pi": 3.5}
    host.tracks.append(Track(id="rules", classes=["umbra"], locked=True))
    host.intervals = [
        Interval(T("00:10"), T("00:20"), "PS", "note", DEFAULT_TRACK_ID,
                 meta),
        Interval(T("00:15"), T("00:25"), "umbra", None, "rules"),
    ]
    target = tmp_path / "s.json"
    host._save_session(str(target))
    host.intervals = []
    host.tracks[:] = default_table(["UNKNOWN"])
    host._load_session(str(target))
    assert [iv.track for iv in host.intervals] == [DEFAULT_TRACK_ID, "rules"]
    assert host.intervals[0].meta == meta
    assert [t.id for t in host.tracks] == [DEFAULT_TRACK_ID, "rules"]
    assert host.track_by_id("rules").locked is True if hasattr(
        host, "track_by_id") else host.tracks[1].locked is True


def test_track_and_meta_survive_the_autosave_round_trip(host):
    meta = {"import.source": "column:rule_label", "n": [1, 2]}
    host.tracks.append(Track(id="rules", classes=["umbra"], locked=True))
    host.intervals = [
        Interval(T("00:10"), T("00:20"), "PS", None, DEFAULT_TRACK_ID, meta),
        Interval(T("00:15"), T("00:25"), "umbra", None, "rules"),
    ]
    host._save_autosave()
    payload = host._check_autosave()
    assert payload is not None
    host.intervals = []
    host.tracks[:] = default_table(["UNKNOWN"])
    host._apply_recovered_autosave(payload)
    assert [iv.track for iv in host.intervals] == [DEFAULT_TRACK_ID, "rules"]
    assert host.intervals[0].meta == meta
    assert [t.id for t in host.tracks] == [DEFAULT_TRACK_ID, "rules"]
    assert host.tracks[1].locked is True


def test_a_point_track_round_trips_through_session_and_autosave(host,
                                                                tmp_path):
    host.tracks.append(Track(id="xings", classes=["shock"], kind="point"))
    host.intervals = [Interval(T("00:10"), T("00:10"), "shock", None,
                               "xings")]
    target = tmp_path / "s.json"
    host._save_session(str(target))
    host.tracks[:] = default_table(["UNKNOWN"])
    host.intervals = []
    host._load_session(str(target))
    assert host.tracks[1].kind == "point"
    assert host.intervals[0].start == host.intervals[0].end
    host._save_autosave()
    payload = host._check_autosave()
    host.tracks[:] = default_table(["UNKNOWN"])
    host._apply_recovered_autosave(payload)
    assert [t.kind for t in host.tracks] == ["interval", "point"]


# ============================================================== the reader

def test_a_v1_session_loads_as_one_default_track_and_says_so(host, tmp_path):
    p = tmp_path / "v1.json"
    _v1_session(p)
    host._load_session(str(p))
    assert [t.id for t in host.tracks] == [DEFAULT_TRACK_ID]
    assert host.tracks[0].classes == ["UNKNOWN", "solar_wind", "umbra"]
    assert [iv.track for iv in host.intervals] == [DEFAULT_TRACK_ID] * 2
    assert [iv.meta for iv in host.intervals] == [{}, {}]
    assert [iv.notes for iv in host.intervals] == ["kept", None]
    assert "as a single track" in host.status_var.get()
    assert DEFAULT_TRACK_ID in host.status_var.get()


def test_a_session_with_no_version_key_reads_as_v1(host, tmp_path):
    p = tmp_path / "nov.json"
    _v1_session(p, drop_version=True)
    assert payload_version({}) == (1, True)
    host._load_session(str(p))
    assert [t.id for t in host.tracks] == [DEFAULT_TRACK_ID]
    assert len(host.intervals) == 2


def test_the_v1_reader_does_not_rewrite_the_file(host, tmp_path):
    p = tmp_path / "v1.json"
    _v1_session(p)
    before = p.read_bytes()
    host._load_session(str(p))
    assert p.read_bytes() == before
    # ... and the next explicit Save writes v2.
    host._save_session(str(p))
    assert json.loads(p.read_text(encoding="utf-8"))["version"] == 2


# ================================================================ the gate

def test_a_future_session_is_refused_headlessly_and_changes_nothing(
        host, tmp_path):
    host.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    live = list(host.intervals)
    live_tracks = [t.id for t in host.tracks]
    p = tmp_path / "v3.json"
    _v1_session(p, version=3)
    with pytest.raises(SessionVersionError) as e:
        host._load_session(str(p))
    assert "NEWER ChronoTagger" in str(e.value)
    assert host.intervals == live
    assert [t.id for t in host.tracks] == live_tracks
    assert host.classes == ["UNKNOWN", "PS", "LOBE"]
    assert "Refused" in host.status_var.get()


def test_a_future_session_is_refused_with_one_dialog_in_a_gui_session(
        gui_host, tmp_path):
    p = tmp_path / "v3.json"
    _v1_session(p, version=3)
    with patch("tkinter.messagebox.showerror") as err:
        gui_host._load_session(str(p))
    assert err.call_count == 1
    assert "Newer File Format" in err.call_args[0][0]
    assert gui_host.intervals == []


def test_a_non_integer_version_is_refused_like_a_future_one(host, tmp_path):
    for bad in ["2.1-beta", 2.5, True, None]:
        p = tmp_path / "bad.json"
        _v1_session(p, version=bad)
        with pytest.raises(SessionVersionError):
            host._load_session(str(p))
    assert payload_version({"version": 3})[1] is False
    assert payload_version({"version": True})[1] is False
    assert payload_version({"version": 2})[1] is True
    # v2 fold F12: DR5's LOWER edge. A version below 1 is malformed, not
    # readable, and nothing in v1 of this pack pinned it.
    assert payload_version({"version": 0}) == (0, False)
    assert payload_version({"version": -1}) == (-1, False)


def test_the_refusal_is_the_first_dialog_before_the_layout_question(
        labeler, tmp_path):
    """With the gate below the layout-compatibility question, a refused
    file first asks the user to accept a layout RISK -- and if they answer
    No, the file is refused for the WRONG REASON and they never learn the
    real one."""
    p = tmp_path / "v3.json"
    data = {
        "version": 3,
        "tracks": [Track(id=DEFAULT_TRACK_ID, classes=["UNKNOWN"]).to_dict()],
        "active_track": DEFAULT_TRACK_ID,
        "window": "0 days 00:30:00",
        "step": "0 days 00:15:00",
        "data_start": "2015-01-03T00:00:00",
        "data_end": "2015-01-03T00:59:30",
        "intervals": [],
        # deliberately INCOMPATIBLE with the fixture's 3-row layout
        "layout_spec": {"nrows": 9, "ncols": 3, "areas": []},
        "multi_pane_mode": False,
        "active_pane_idx": 0,
        "panes": [],
    }
    p.write_text(json.dumps(data), encoding="utf-8")
    with patch("tkinter.messagebox.showerror") as err, \
            patch("tkinter.messagebox.askyesno") as ask:
        labeler._load_session(str(p))
    assert err.call_count == 1
    assert ask.call_count == 0


def test_a_future_autosave_is_refused_and_the_live_set_is_untouched(host):
    host.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    live = list(host.intervals)
    payload = {"version": 11, "tracks": [], "intervals": [],
               "metadata": {}, "label_stats": {}}
    host.autosave_file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SessionVersionError):
        host._check_autosave()
    assert host.intervals == live


def test_a_versionless_autosave_migrates_to_one_default_track(host):
    """All 33 live autosaves on disk are exactly this shape: five keys and
    NO version key at all."""
    payload = {
        "classes": ["UNKNOWN", "solar_wind", "umbra"],
        "class_colors": {"UNKNOWN": "#cccccc"},
        "intervals": [{"start": "2024-01-01T00:10:00",
                       "end": "2024-01-01T00:20:00",
                       "label": "solar_wind", "notes": None}],
        "label_stats": {},
        "metadata": {},
    }
    host.autosave_file.write_text(json.dumps(payload), encoding="utf-8")
    data = host._check_autosave()
    assert data is not None
    assert data["_migrated_from_v1"] is True
    host._apply_recovered_autosave(data)
    assert [t.id for t in host.tracks] == [DEFAULT_TRACK_ID]
    assert host.classes == ["UNKNOWN", "solar_wind", "umbra"]
    assert [iv.track for iv in host.intervals] == [DEFAULT_TRACK_ID]


# ================================================ build, validate, publish

def test_a_within_track_overlap_leaves_the_previous_session_intact(
        host, tmp_path):
    host.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    live = list(host.intervals)
    p = tmp_path / "bad.json"
    _v1_session(p, intervals=[
        {"start": "2024-01-01T00:00:00", "end": "2024-01-01T00:30:00",
         "label": "solar_wind", "notes": None},
        {"start": "2024-01-01T00:15:00", "end": "2024-01-01T00:45:00",
         "label": "umbra", "notes": None},
    ])
    with pytest.raises(IntervalInvariantError):
        host._load_session(str(p))
    assert host.intervals == live
    assert host.classes == ["UNKNOWN", "PS", "LOBE"]


def test_a_session_on_an_unknown_track_leaves_the_previous_set_intact(
        host, tmp_path):
    host.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    live = list(host.intervals)
    p = tmp_path / "ghost.json"
    data = {
        "version": 2,
        "tracks": [Track(id=DEFAULT_TRACK_ID, classes=["UNKNOWN"]).to_dict()],
        "active_track": DEFAULT_TRACK_ID,
        "window": "0 days 00:30:00", "step": "0 days 00:15:00",
        "data_start": "2024-01-01T00:00:00",
        "data_end": "2024-01-01T01:39:00",
        "intervals": [{"start": "2024-01-01T00:10:00",
                       "end": "2024-01-01T00:20:00",
                       "label": "UNKNOWN", "notes": None,
                       "track": "ghost"}],
        "layout_spec": None, "multi_pane_mode": False,
        "active_pane_idx": 0, "panes": [],
    }
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError) as e:
        host._load_session(str(p))
    assert "not in its track table" in str(e.value)
    assert host.intervals == live


def test_recovery_installs_interval_objects_never_dicts(host):
    """io_export.py's recovery path had NO from_dict: handed raw dicts it
    installed dicts, with no exception and no warning, and every
    downstream iv.label then raised somewhere else."""
    host._apply_recovered_autosave({
        "intervals": [{"start": "2024-01-01T00:10:00",
                       "end": "2024-01-01T00:20:00",
                       "label": "PS", "notes": None}],
    })
    assert [type(iv).__name__ for iv in host.intervals] == ["Interval"]
    assert host.intervals[0].track == DEFAULT_TRACK_ID


def test_recovery_on_an_unknown_track_raises_and_keeps_the_set(host):
    host.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    live = list(host.intervals)
    with pytest.raises(ValueError) as e:
        host._apply_recovered_autosave({
            "intervals": [Interval(T("01:00"), T("01:10"), "PS",
                                   track="ghost")],
            "tracks": [Track(id=DEFAULT_TRACK_ID,
                             classes=["PS"]).to_dict()],
        })
    assert "not in its track table" in str(e.value)
    assert host.intervals == live


def test_the_v1_classes_fallback_in_recovery_is_kept(host):
    """This is a NAMED ENTRY POINT the suite calls with hand-built
    payloads, and a future importer will hand it one it built itself."""
    host._apply_recovered_autosave({
        "intervals": [],
        "classes": ["UNKNOWN", "CUSTOM"],
        "class_colors": {"UNKNOWN": "#cccccc", "CUSTOM": "#00ff00"},
    })
    assert host.classes == ["UNKNOWN", "CUSTOM"]
    assert host.class_colors["CUSTOM"] == "#00ff00"
    # and a payload with neither keeps the live schema
    host._apply_recovered_autosave({"intervals": []})
    assert host.classes == ["UNKNOWN", "CUSTOM"]


# ============================================ per-track stats and coverage

def test_label_stats_are_per_track_and_coverage_is_the_union(host):
    host.tracks.append(Track(id="rules", classes=["PS"]))
    # a third lane with NO intervals: every track in the table gets a key,
    # so a reader can tell "no intervals" from "no such track"
    host.tracks.append(Track(id="quiet", classes=["PS"]))
    host.intervals = [
        Interval(T("00:00"), T("00:30"), "PS"),
        Interval(T("00:10"), T("00:40"), "PS", track="rules"),
    ]
    host._save_autosave()
    data = json.loads(host.autosave_file.read_text(encoding="utf-8"))
    stats = data["label_stats"]
    assert set(stats) == {DEFAULT_TRACK_ID, "rules", "quiet"}
    assert stats["quiet"] == {}
    assert stats[DEFAULT_TRACK_ID]["PS"]["count"] == 1
    assert stats["rules"]["PS"]["count"] == 1
    # the union is 40 minutes of a 99-minute record, not 60 of 99
    cov = data["metadata"]["coverage_percent"]
    assert cov == pytest.approx(40.0 / 99.0 * 100, abs=0.2)
    by_track = data["metadata"]["coverage_percent_by_track"]
    assert set(by_track) == {DEFAULT_TRACK_ID, "rules", "quiet"}
    assert by_track["quiet"] == 0
    assert cov <= 100.0


def test_label_stats_by_track_gives_every_track_a_key():
    tracks = [Track(id="a", classes=["X"]), Track(id="b", classes=["X"])]
    ivs = [Interval(T("00:00"), T("01:00"), "X", track="a")]
    out = label_stats_by_track(ivs, tracks)
    assert set(out) == {"a", "b"}
    assert out["b"] == {}
    assert out["a"]["X"]["duration_hours"] == pytest.approx(1.0)


def test_the_recovery_dialog_reads_both_label_stats_shapes():
    """The dialog can be handed either shape: v1's flat one from an
    autosave already on disk, and v2's per-track one."""
    v1 = {"PS": {"count": 2, "duration_hours": 1.0}}
    v2 = {"default": {"PS": {"count": 2, "duration_hours": 1.0}},
          "rules": {"PS": {"count": 1, "duration_hours": 0.5}}}
    assert flatten_label_stats(v1) == v1
    flat = flatten_label_stats(v2)
    assert flat["PS"]["count"] == 3
    assert flat["PS"]["duration_hours"] == pytest.approx(1.5)
    assert flatten_label_stats({}) == {}
    assert flatten_label_stats(None) == {}


def test_the_autosave_skips_an_unchanged_write(host):
    host.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    host._save_autosave()
    first = host.autosave_file.read_bytes()
    bak = host.autosave_file.with_name(host.autosave_file.name + ".bak")
    assert not bak.exists()

    host._save_autosave()          # nothing changed -> no write, no .bak
    assert not bak.exists()
    assert host.autosave_file.read_bytes() == first

    host.tracks.append(Track(id="rules", classes=["PS"]))
    host._save_autosave()          # the TABLE changed -> a write
    assert bak.exists()
    assert host.autosave_file.read_bytes() != first

    host.autosave_file.unlink()    # the file's existence is part of it
    host._save_autosave()
    assert host.autosave_file.exists()


def test_a_failed_autosave_write_does_not_record_the_signature(host):
    """Only a write that SUCCEEDED may suppress the next one. A failed
    one leaves the signature alone, so the very next gesture retries
    exactly as it did before this pack."""
    from chronotagger.labeler.utils import atomic_io

    host.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    with patch.object(atomic_io.json, "dump", side_effect=OSError("boom")):
        host._save_autosave()
    assert not host.autosave_file.exists()
    assert getattr(host, "_autosave_signature", None) is None
    host._save_autosave()                 # same data: must NOT be skipped
    assert host.autosave_file.exists()


def test_the_active_track_round_trips_through_both_payloads(host, tmp_path):
    """It is view state, but it PERSISTS -- one top-level key rather than
    a field on a row."""
    host.tracks.append(Track(id="rules", classes=["umbra"]))
    host._active_track_id = "rules"
    target = tmp_path / "s.json"
    host._save_session(str(target))
    assert json.loads(target.read_text(encoding="utf-8"))["active_track"]         == "rules"
    host._active_track_id = DEFAULT_TRACK_ID
    host._load_session(str(target))
    assert host._active_track_id == "rules"
    assert host.classes == ["umbra"]

    host._save_autosave()
    payload = host._check_autosave()
    assert payload["active_track"] == "rules"
    host._active_track_id = DEFAULT_TRACK_ID
    host._apply_recovered_autosave(payload)
    assert host._active_track_id == "rules"
    # a STALE active id falls back to the FIRST row rather than raising:
    # a stale id is a view-state bug and must not take the window down
    host._load_session(str(target))
    host.tracks[:] = [t for t in host.tracks if t.id != "rules"]
    assert host._active_track_id == "rules"
    assert host.classes == ["UNKNOWN", "PS", "LOBE"]


def test_a_recovered_within_track_overlap_leaves_the_set_installed(host):
    """The strict invariant on the recovery path, validated against the
    table being INSTALLED and BEFORE anything is published."""
    host.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    live = list(host.intervals)
    with pytest.raises(IntervalInvariantError) as e:
        host._apply_recovered_autosave({
            "intervals": [
                Interval(T("01:00"), T("01:30"), "PS"),
                Interval(T("01:10"), T("01:40"), "PS"),
            ],
            "tracks": [Track(id=DEFAULT_TRACK_ID,
                             classes=["PS"]).to_dict()],
        })
    assert "track 'default'" in str(e.value)
    assert host.intervals == live


def test_the_fingerprint_and_the_autosave_name_do_not_move_with_a_track(
        host):
    before_fp = host._dataset_fingerprint()
    before_name = host.autosave_file.name
    host.tracks.append(Track(id="rules", classes=["PS"]))
    host.intervals = [Interval(T("00:10"), T("00:20"), "PS", track="rules")]
    assert host._dataset_fingerprint() == before_fp
    assert host.autosave_file.name == before_name


# ================================================ the migration, measured

def test_a_real_shaped_v1_session_round_trips_losslessly(host, tmp_path):
    """The corpus pin. The 31 curated sessions on disk are not shippable
    in tests/, so the PROPERTY is pinned on payloads of exactly their
    shape -- his eight-class schema, sub-second boundaries carrying the
    nanosecond parts his frame's cadence snapped them to, and notes.

    Lossless means: the same count, the same labels, the same notes, the
    same nanosecond-exact bounds, and RE-SERIALIZED ISO TEXT BYTE-
    IDENTICAL to the source text.
    """
    classes = ["UNKNOWN", "solar_wind", "magnetosheath", "lobe",
               "plasma_sheet", "umbra", "mantle", "AMBIGUOUS"]
    src = [
        {"start": "2024-01-01T00:10:29.725916",
         "end": "2024-01-01T00:20:36.047051431",
         "label": "solar_wind", "notes": None},
        {"start": "2024-01-01T00:20:36.047051431",
         "end": "2024-01-01T00:30:00.000000001",
         "label": "umbra", "notes": "lunar shadow"},
        {"start": "2024-01-01T00:30:00.000000001",
         "end": "2024-01-01T00:44:59.999999999",
         "label": "solar_wind", "notes": None},
    ]
    p = tmp_path / "thb_like.json"
    _v1_session(p, classes=classes, intervals=src)

    host._load_session(str(p))
    assert len(host.intervals) == 3
    assert [iv.label for iv in host.intervals] == [
        "solar_wind", "umbra", "solar_wind"]
    assert [iv.notes for iv in host.intervals] == [
        None, "lunar shadow", None]
    assert host.tracks[0].classes == classes

    # nanosecond-exact, and the ISO text is the same characters
    for iv, d in zip(host.intervals, src):
        assert iv.start == pd.Timestamp(d["start"])
        assert iv.end == pd.Timestamp(d["end"])
        assert iv.start.isoformat() == d["start"]
        assert iv.end.isoformat() == d["end"]

    # re-save as v2 and read it back: still the same characters
    out = tmp_path / "v2.json"
    host._save_session(str(out))
    v2 = json.loads(out.read_text(encoding="utf-8"))
    assert v2["version"] == 2
    assert [d["start"] for d in v2["intervals"]] == [d["start"] for d in src]
    assert [d["end"] for d in v2["intervals"]] == [d["end"] for d in src]
    assert [d["track"] for d in v2["intervals"]] == [DEFAULT_TRACK_ID] * 3
    assert v2["tracks"][0]["classes"] == classes

    host.intervals = []
    host.tracks[:] = default_table(["UNKNOWN"])
    host._load_session(str(out))
    assert [iv.start.isoformat() for iv in host.intervals] == [
        d["start"] for d in src]
    assert host.tracks[0].classes == classes
    # v2 does NOT say "as a single track" -- that line is the migration's
    assert "as a single track" not in host.status_var.get()


# ============================================ THE v2 FOLD PINS
# Five pins the verifiers asked for: the two table rules on both LOAD
# paths (F2), the autosave signature's completeness including the ACTIVE
# TRACK (F3 / F12), and the diagnosis a version below 1 gets (F7).

def _v2_session(path, tracks, intervals=None, active=None):
    """A v2 session file built from RAW ROWS, so a table this build
    cannot PRODUCE can still be handed to the loader -- which is the
    whole point of validating on the way in."""
    data = {
        "version": 2,
        "tracks": list(tracks),
        "active_track": active or (tracks[0]["id"] if tracks else None),
        "window": "0 days 00:30:00",
        "step": "0 days 00:15:00",
        "data_start": "2024-01-01T00:00:00",
        "data_end": "2024-01-01T01:39:00",
        "intervals": intervals if intervals is not None else [],
        "layout_spec": None,
        "multi_pane_mode": False,
        "active_pane_idx": 0,
        "panes": [],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def test_a_duplicate_track_id_in_a_SESSION_is_refused(host, tmp_path):
    """v2 fold F2. _build_track_table refuses a duplicate id, and DR13
    leans on that refusal for the whole of "the id default is reserved"
    -- but a table arriving OFF DISK comes through a different door.
    Measured on v1 of this pack: a session with two rows sharing one id
    LOADED, the two lanes then collapsed into ONE export column and ONE
    sidecar, and the orphan gate reported the FIRST row's perfectly
    legal label as an orphan because it keys a dict by id."""
    host.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    live = list(host.intervals)
    live_tracks = [t.id for t in host.tracks]
    p = tmp_path / "dup.json"
    _v2_session(p, [
        {"id": "dup", "name": "human", "classes": ["A"]},
        {"id": "dup", "name": "model", "classes": ["B"]},
    ])
    with pytest.raises(ValueError) as e:
        host._load_session(str(p))
    assert "two tracks with the id 'dup'" in str(e.value)
    assert host.intervals == live
    assert [t.id for t in host.tracks] == live_tracks


def test_a_classless_track_in_a_SESSION_is_refused(host, tmp_path):
    """v2 fold F2, the second table rule. Measured on v1 of this pack: a
    one-row table with classes: [] loaded, and a real labeler then died
    on a bare IndexError out of
    self.current_class_var.set(self.classes[0])."""
    p = tmp_path / "empty_vocab.json"
    _v2_session(p, [{"id": DEFAULT_TRACK_ID, "classes": []}])
    with pytest.raises(ValueError) as e:
        host._load_session(str(p))
    assert "no classes" in str(e.value)


def test_a_duplicate_track_id_in_an_AUTOSAVE_is_refused(host):
    """v2 fold F2, the same door on the path a user meets without
    choosing to -- the recovery offer on startup."""
    host.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    live = list(host.intervals)
    with pytest.raises(ValueError) as e:
        host._apply_recovered_autosave({
            "intervals": [],
            "tracks": [{"id": "dup", "classes": ["A"]},
                       {"id": "dup", "classes": ["A"]}],
        })
    assert "two tracks with the id 'dup'" in str(e.value)
    assert host.intervals == live


def test_the_autosave_signature_names_every_field_in_the_payload(host):
    """v2 folds F3 and F12. The skip decides whether a gesture reaches
    disk, so every field the PAYLOAD carries has to be in the SIGNATURE
    or a change to it is swallowed. Measured on v1 of this pack:
    dropping notes, meta, locked or kind from the signature reddened no
    pin, and the ACTIVE TRACK was not in the signature at all -- so a
    lane switch left the file naming the old lane.

    The last two writes are deliberately separate: the new row is added
    and written FIRST, so the only thing that changes on the final write
    is the active id."""
    host.intervals = [Interval(T("00:10"), T("00:20"), "PS", "first",
                               DEFAULT_TRACK_ID, {"k": 1})]
    host._save_autosave()
    last = {"bytes": host.autosave_file.read_bytes()}

    def rewrote():
        now = host.autosave_file.read_bytes()
        changed = now != last["bytes"]
        last["bytes"] = now
        return changed

    host.intervals[0].notes = "second"
    host._save_autosave()
    assert rewrote(), "a notes-only change was swallowed by the skip"
    host.intervals[0].meta["k"] = 2
    host._save_autosave()
    assert rewrote(), "a meta-only change was swallowed by the skip"
    host.tracks[0].locked = True
    host._save_autosave()
    assert rewrote(), "a lock change was swallowed by the skip"
    host.tracks[0].kind = "point"
    host._save_autosave()
    assert rewrote(), "a kind change was swallowed by the skip"
    host.tracks[0].name = "Renamed"
    host._save_autosave()
    assert rewrote(), "a rename was swallowed by the skip"
    host.tracks.append(Track(id="second", classes=["X"]))
    host._save_autosave()
    assert rewrote(), "a new table row was swallowed by the skip"
    host._active_track_id = "second"
    host._save_autosave()
    assert rewrote(), "an active_track change was swallowed by the skip"
    assert json.loads(host.autosave_file.read_text(
        encoding="utf-8"))["active_track"] == "second"


def test_a_version_below_1_is_refused_as_UNRECOGNISED_not_as_NEWER(
        host, tmp_path):
    """v2 fold F7. DR5 refuses version 0 and below, and v1 of this pack
    refused them with the words "written by a NEWER ChronoTagger" -- the
    right refusal with the wrong diagnosis."""
    for bad in (0, -1):
        p = tmp_path / ("v%s.json" % bad)
        _v1_session(p, version=bad)
        with pytest.raises(SessionVersionError) as e:
            host._load_session(str(p))
        assert "NEWER ChronoTagger" not in str(e.value)
        assert "does not recognise" in str(e.value)
    # ... and a version ABOVE this build's still says NEWER
    p = tmp_path / "v9.json"
    _v1_session(p, version=9)
    with pytest.raises(SessionVersionError) as e:
        host._load_session(str(p))
    assert "NEWER ChronoTagger" in str(e.value)


def test_the_adjacent_v1_intervals_do_not_fuse_on_load(host, tmp_path):
    """A load does not merge: the three abutting intervals above stay
    three, because two of them carry different labels and the pair that
    shares one is not adjacent."""
    p = tmp_path / "v1.json"
    _v1_session(p, intervals=[
        {"start": "2024-01-01T00:00:00", "end": "2024-01-01T00:10:00",
         "label": "solar_wind", "notes": None},
        {"start": "2024-01-01T00:10:00", "end": "2024-01-01T00:20:00",
         "label": "solar_wind", "notes": None},
    ])
    host._load_session(str(p))
    assert len(host.intervals) == 2
