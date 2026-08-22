"""
Pack 2 regression tests: atomic persistence, fingerprinted autosaves,
schema-aware recovery, honest exports.

GUI-free: MockPersistHost binds the real IOExportMixin methods onto a
plain object (same pattern as test_undo_integrity.MockUndoMixin). Any
test that can reach a tkinter.messagebox call stubs it -- a modal
dialog hangs headless CI forever (Pack 1 recheck lesson).
"""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from chronotagger.core.models import Interval
from chronotagger.labeler.mixins.io_export import IOExportMixin, dataset_fingerprint
from chronotagger.labeler.utils import atomic_io
from chronotagger.labeler.utils.atomic_io import atomic_write_json, atomic_write_path


class _Var:
    def __init__(self):
        self._v = ""

    def set(self, v):
        self._v = v

    def get(self):
        return self._v


class MockPersistHost:
    """GUI-free host for the persistence mixin methods."""

    _BOUND = [
        "_dataset_fingerprint", "_save_autosave", "_check_autosave",
        "_apply_recovered_autosave", "_save_session",
        "_export_labels_do", "_compute_label_id_series",
    ]

    def __init__(self, df, folder):
        self.df = df
        self.intervals = []
        self.classes = ["UNKNOWN", "PS", "LOBE"]
        self.class_colors = {"UNKNOWN": "#cccccc", "PS": "#ff0000", "LOBE": "#0000ff"}
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo = 50
        self.modified = False
        self.selected_interval = None
        self.data_start = df.index[0]
        self.data_end = df.index[-1]
        self.window = pd.Timedelta("30min")
        self.step = pd.Timedelta("15min")
        self.layout_spec = None
        self.source_name = None
        self.status_var = _Var()
        self.class_combo = None
        self.current_class_var = None
        # Non-None: _save_session takes the GUI branch (its dialogs are
        # patched in the tests that reach them); None would route every
        # mock call down the headless re-raise path (recheck M3).
        self.root = object()

        for name in self._BOUND:
            setattr(self, name, getattr(IOExportMixin, name).__get__(self))
        from chronotagger.labeler.mixins.intervals import IntervalsMixin
        self._check_interval_invariants = \
            IntervalsMixin._check_interval_invariants.__get__(self)

        self.autosave_folder = folder
        self.autosave_file = folder / f"chronotagger_autosave_{self._dataset_fingerprint()}.json"


def _grid(cols=("a", "b", "c"), periods=100):
    idx = pd.date_range("2024-01-01 00:00:00", periods=periods, freq="1min")
    return pd.DataFrame({c: np.linspace(0.0, 1.0, periods) for c in cols}, index=idx)


def T(hhmm):
    return pd.Timestamp(f"2024-01-01 {hhmm}:00")


@pytest.fixture
def host(tmp_path):
    return MockPersistHost(_grid(), tmp_path)


# ---- fingerprint (grill Q1, recipe R1) ----

def test_fingerprint_stability_and_sensitivity():
    df = _grid()
    fp = dataset_fingerprint(df)
    assert len(fp) == 12 and all(c in "0123456789abcdef" for c in fp)

    # Stable: column reorder and value edits do not change identity
    assert dataset_fingerprint(df[["c", "a", "b"]]) == fp
    edited = df.copy()
    edited["a"] = edited["a"] * 2
    assert dataset_fingerprint(edited) == fp

    # Sensitive: added column, renamed column, trimmed range all differ
    extra = df.copy()
    extra["d"] = 0.0
    assert dataset_fingerprint(extra) != fp
    assert dataset_fingerprint(df.rename(columns={"a": "a2"})) != fp
    assert dataset_fingerprint(df.iloc[:50]) != fp


# ---- atomic writes (grill Q5) ----

def _tmp_of(target):
    return target.with_name(target.stem + ".tmp" + target.suffix)


def _write_text(text):
    def writer(p):
        with open(p, "w") as f:
            f.write(text)
    return writer


def test_atomic_write_json_crash_preserves_old_file(tmp_path):
    target = tmp_path / "f.json"
    atomic_write_json(target, {"v": 1})
    before = target.read_bytes()

    def bad_dump(obj, f, **kw):
        f.write('{"v": 2, "partial')
        raise OSError("disk died mid-write")

    with patch.object(atomic_io.json, "dump", side_effect=bad_dump):
        with pytest.raises(OSError):
            atomic_write_json(target, {"v": 2})

    assert target.read_bytes() == before
    assert not _tmp_of(target).exists()


def test_atomic_write_path_crash_preserves_old_file(tmp_path):
    target = tmp_path / "f.csv"
    atomic_write_path(target, _write_text("good\n"))
    before = target.read_bytes()

    def bad_writer(p):
        with open(p, "w") as f:
            f.write("half")
        raise OSError("killed")

    with pytest.raises(OSError):
        atomic_write_path(target, bad_writer)

    assert target.read_bytes() == before
    assert not _tmp_of(target).exists()


def test_atomic_write_path_keeps_compression_suffix(tmp_path):
    """Fold V1/V2/V3 (.gz): the tmp name preserves the final suffix, so
    pandas' extension-inferred compression survives the atomic detour.
    A gzip target must round-trip through pd.read_csv."""
    target = tmp_path / "labels.csv.gz"
    df = pd.DataFrame({"a": [1, 2, 3]})
    atomic_write_path(target, lambda p: df.to_csv(p, index=False))

    assert target.read_bytes()[:2] == b"\x1f\x8b"  # gzip magic
    back = pd.read_csv(target)
    assert list(back["a"]) == [1, 2, 3]
    assert not _tmp_of(target).exists()


# ---- autosave (grill Q1/Q4/Q5) ----

def test_autosave_writes_fingerprinted_file_with_schema(host):
    host.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    host._save_autosave()

    assert host.autosave_file.exists()
    assert host.autosave_file.name.startswith("chronotagger_autosave_")
    data = json.loads(host.autosave_file.read_text(encoding="utf-8"))
    assert data["classes"] == ["UNKNOWN", "PS", "LOBE"]
    assert data["class_colors"]["PS"] == "#ff0000"
    assert data["metadata"]["fingerprint"] == host._dataset_fingerprint()
    assert data["metadata"]["n_rows"] == 100


def test_autosave_keeps_bak_generation(host):
    host.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    host._save_autosave()
    first = host.autosave_file.read_bytes()

    host.intervals.append(Interval(T("00:30"), T("00:40"), "LOBE"))
    host._save_autosave()

    bak = host.autosave_file.with_name(host.autosave_file.name + ".bak")
    assert bak.exists()
    assert bak.read_bytes() == first
    assert host.autosave_file.read_bytes() != first


def test_crash_mid_autosave_preserves_previous_and_surfaces(host):
    host.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    host._save_autosave()
    before = host.autosave_file.read_bytes()

    with patch.object(atomic_io.json, "dump", side_effect=OSError("boom")):
        host._save_autosave()  # must not raise; must not destroy

    assert host.autosave_file.read_bytes() == before
    assert "Autosave failed" in host.status_var.get()


# ---- recovery lookup (grill Q2 clean break, Q3) ----

def test_check_autosave_roundtrip_with_identity(host):
    host.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    host._save_autosave()

    data = host._check_autosave()
    assert data is not None
    assert data["_identity"]["mismatch"] is False
    assert data["_loaded_path"] == str(host.autosave_file)
    assert isinstance(data["intervals"][0], Interval)


def test_check_autosave_falls_back_to_bak_on_corrupt_main(host):
    host.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    host._save_autosave()
    host.intervals.append(Interval(T("00:30"), T("00:40"), "LOBE"))
    host._save_autosave()  # creates .bak of the first save

    host.autosave_file.write_text('{"metadata": {"truncated', encoding="utf-8")

    data = host._check_autosave()
    assert data is not None
    assert data["_loaded_path"].endswith(".bak")
    assert any(".bak" in line for line in data["_identity"]["lines"])
    assert len(data["intervals"]) == 1  # the first save's content


def test_check_autosave_ignores_legacy_name_but_says_so(host, tmp_path):
    # Clean break (Q2): a pre-fingerprint autosave is never read -- but
    # the user is TOLD it is being ignored (fold V3: a silent clean
    # break is a data-loss-shaped surprise).
    legacy = tmp_path / "chronotagger_autosave.json"
    legacy.write_text('{"intervals": [], "metadata": {}}', encoding="utf-8")
    assert host._check_autosave() is None
    assert legacy.exists()  # and never deleted
    assert "no longer read" in host.status_var.get()


def test_bak_not_consulted_when_main_deleted(host):
    """Fold V3: deleting the named autosave file is the user's cleanup
    route; the .bak must not resurrect it."""
    host.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    host._save_autosave()
    host._save_autosave()  # second save creates the .bak
    host.autosave_file.unlink()

    assert host._check_autosave() is None
    assert host.autosave_file.with_name(host.autosave_file.name + ".bak").exists()


def test_check_autosave_warns_on_source_name_mismatch(host):
    """Fold V1: identical-schema same-window datasets share a
    fingerprint (THEMIS-A vs THEMIS-D); the source name is the
    tiebreaker."""
    host.source_name = "tha_2024-03-01.parquet"
    host.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    host._save_autosave()

    host.source_name = "thd_2024-03-01.parquet"
    data = host._check_autosave()
    assert data is not None
    assert data["_identity"]["mismatch"] is True
    joined = "\n".join(data["_identity"]["lines"])
    assert "different source file" in joined
    assert "tha_2024-03-01.parquet" in joined


def test_tz_localized_twin_matches_cleanly(tmp_path):
    """Fold V1/V2/V3 (tz): a naive frame and its UTC-localized twin
    share a fingerprint AND pass the identity check -- the check must
    never contradict the fingerprint about timezones."""
    naive = _grid()
    aware = naive.tz_localize("UTC")
    assert dataset_fingerprint(naive) == dataset_fingerprint(aware)

    host_naive = MockPersistHost(naive, tmp_path)
    host_naive.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    host_naive._save_autosave()

    host_aware = MockPersistHost(aware, tmp_path)
    data = host_aware._check_autosave()
    assert data is not None
    assert data["_identity"]["mismatch"] is False


def test_check_autosave_two_datasets_no_cross_talk(tmp_path):
    host_a = MockPersistHost(_grid(cols=("a", "b", "c")), tmp_path)
    host_b = MockPersistHost(_grid(cols=("x", "y")), tmp_path)

    host_a.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    host_a._save_autosave()

    assert host_b._check_autosave() is None       # B never sees A's labels
    assert host_a._check_autosave() is not None   # A still sees its own


def test_check_autosave_tolerates_missing_keys(host):
    host.autosave_file.write_text('{"intervals": []}', encoding="utf-8")
    data = host._check_autosave()
    assert data is not None
    assert data["intervals"] == []
    assert data["_identity"]["mismatch"] is False


def test_check_autosave_flags_column_mismatch(host, tmp_path):
    other = MockPersistHost(_grid(cols=("a", "b", "z")), tmp_path)
    other.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    other._save_autosave()

    # Simulate a hand-moved file: other's autosave under host's name
    host.autosave_file.write_bytes(other.autosave_file.read_bytes())
    data = host._check_autosave()
    assert data is not None
    assert data["_identity"]["mismatch"] is True
    joined = "\n".join(data["_identity"]["lines"])
    assert "z" in joined and "c" in joined


# ---- recovery apply (grill Q4, D1) ----

def test_recovery_applies_schema_and_invalidates_history(host):
    payload = {
        "intervals": [Interval(T("00:10"), T("00:20"), "CUSTOM")],
        "classes": ["UNKNOWN", "CUSTOM"],
        "class_colors": {"UNKNOWN": "#cccccc", "CUSTOM": "#00ff00"},
    }
    host.undo_stack.append(object())
    host.selected_interval = object()

    host._apply_recovered_autosave(payload)

    assert host.classes == ["UNKNOWN", "CUSTOM"]
    assert host.class_colors["CUSTOM"] == "#00ff00"
    assert host.intervals[0].label == "CUSTOM"
    assert host.undo_stack == [] and host.redo_stack == []
    assert host.selected_interval is None
    assert host.modified is True

    # The restored schema makes the export honest: CUSTOM is mapped,
    # not silently collapsed to -1.
    ids = host._compute_label_id_series()
    assert set(ids.unique()) == {-1, 1}


def test_recovery_without_classes_keeps_current_schema(host):
    host._apply_recovered_autosave({"intervals": []})
    assert host.classes == ["UNKNOWN", "PS", "LOBE"]
    assert host.modified is True


# ---- exports (grill Q4/Q7) ----

def test_export_blocked_on_orphan_labels(host, tmp_path):
    host.intervals = [
        Interval(T("00:10"), T("00:20"), "PS"),
        Interval(T("00:30"), T("00:40"), "GHOST"),
    ]
    out = tmp_path / "labels.csv"
    with patch("tkinter.messagebox.showerror") as err:
        ok = host._export_labels_do(str(out), "full", "index_labels_csv")

    assert ok is False
    assert not out.exists()
    assert "GHOST" in err.call_args.args[1]


def test_export_selected_empty_returns_false(host, tmp_path):
    out = tmp_path / "labels.csv"
    with patch("tkinter.messagebox.showwarning"):
        ok = host._export_labels_do(str(out), "selected", "index_labels_csv")
    assert ok is False
    assert not out.exists()


def test_export_success_writes_csv_and_sidecar(host, tmp_path):
    host.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    out = tmp_path / "labels.csv"

    ok = host._export_labels_do(str(out), "full", "index_labels_csv")

    assert ok is True
    assert out.exists()
    sidecar = tmp_path / "labels_label_map.json"
    mapping = json.loads(sidecar.read_text(encoding="utf-8"))
    assert mapping == {"UNKNOWN": 0, "PS": 1, "LOBE": 2}


# ---- session save (grill Q5/Q7, D2) ----

def test_session_save_returns_true_and_is_atomic(host, tmp_path):
    host.intervals = [Interval(T("00:10"), T("00:20"), "PS")]
    target = tmp_path / "s.json"
    assert host._save_session(str(target)) is True
    before = target.read_bytes()

    with patch.object(atomic_io.json, "dump", side_effect=OSError("boom")), \
         patch("tkinter.messagebox.showerror") as err:
        host.modified = True
        assert host._save_session(str(target)) is False

    assert target.read_bytes() == before      # old session survives
    assert host.modified is True              # failure does not clear it
    assert err.called


def test_public_export_intervals_raises_on_empty(host, tmp_path):
    with pytest.raises(ValueError):
        IOExportMixin.export_intervals.__get__(host)(str(tmp_path / "iv.csv"), fmt="csv")
