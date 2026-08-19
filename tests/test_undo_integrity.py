"""
Pack 1 regression tests: gesture-snapshot undo integrity.

Every test reproduces a corruption mode verified by execution in the
2026-08-19 deep review (vault findings ledger sections 5.1/5.2/6), or
pins the gesture behavior that fixes it. GUI-free: MockUndoMixin binds
the real mixin methods onto a plain object (same pattern as
test_overlap_resolution.MockIntervalsMixin).
"""

import pandas as pd
import numpy as np
import pytest

from chronotagger.core.models import Interval
from chronotagger.core.commands import (
    AddIntervalCommand,
    DeleteIntervalCommand,
    RelabelIntervalCommand,
    ResizeIntervalCommand,
    IntervalInvariantError,
)


class _Var:
    def __init__(self):
        self._v = ""

    def set(self, v):
        self._v = v

    def get(self):
        return self._v


class _Sync:
    def sync_intervals_changed(self):
        pass


class MockUndoMixin:
    """GUI-free host binding the real interval/undo mixin methods."""

    _BOUND = [
        "_execute_command", "_gesture", "_check_interval_invariants",
        "_repoint_selected_interval", "_undo", "_redo",
        "_remove_overlapping_intervals", "_sort_and_merge_intervals",
        "_carve_existing_for_new_span", "_apply_overlap_policy_to_spans",
        "_subtract_overlaps_from_span",
        "_clear_intervals_in_range", "_assign_gaps_to_label",
        "_add_intervals_with_policy",
    ]

    def __init__(self, df):
        self.df = df
        self.intervals = []
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo = 100
        self.modified = False
        self.selected_interval = None
        self.classes = ["UNKNOWN", "A", "B"]
        self.t0 = df.index[0]
        self.t1 = df.index[-1]
        self.status_var = _Var()
        self.sync_manager = _Sync()

        from chronotagger.labeler.mixins.intervals import IntervalsMixin
        for name in self._BOUND:
            setattr(self, name, getattr(IntervalsMixin, name).__get__(self))

    def _update_plot(self):
        pass

    def _save_autosave(self):
        pass


def T(hhmm):
    return pd.Timestamp(f"2024-01-01 {hhmm}:00")


def state(host):
    return [(iv.start, iv.end, iv.label) for iv in host.intervals]


@pytest.fixture
def host():
    idx = pd.date_range("2024-01-01 00:00:00", periods=200, freq="1min")
    df = pd.DataFrame({"value": np.linspace(0.0, 1.0, len(idx))}, index=idx)
    return MockUndoMixin(df)


# ---- the four reproduced corruption modes (ledger 5.1) ----

def test_add_adjacent_after_existing_then_undo(host):
    """Mode 1: undo of an adjacent-after same-label add was a silent no-op."""
    host.intervals = [Interval(T("00:00"), T("01:00"), "A")]
    host._execute_command(AddIntervalCommand(host, Interval(T("01:00"), T("02:00"), "A")))
    assert state(host) == [(T("00:00"), T("02:00"), "A")]

    host._undo()
    assert state(host) == [(T("00:00"), T("01:00"), "A")]


def test_add_adjacent_before_existing_then_undo(host):
    """Mode 2: undo of an adjacent-before add deleted pre-existing data."""
    host.intervals = [Interval(T("01:00"), T("02:00"), "A")]
    host._execute_command(AddIntervalCommand(host, Interval(T("00:00"), T("01:00"), "A")))
    assert state(host) == [(T("00:00"), T("02:00"), "A")]

    host._undo()
    assert state(host) == [(T("01:00"), T("02:00"), "A")]


def test_relabel_into_merge_then_undo(host):
    """Mode 3: undo of a relabel-into-merge destroyed the old label."""
    host.intervals = [
        Interval(T("00:00"), T("00:10"), "A"),
        Interval(T("00:10"), T("00:20"), "B"),
    ]
    host._execute_command(
        RelabelIntervalCommand(host, host.intervals[1], "A"))
    assert state(host) == [(T("00:00"), T("00:20"), "A")]

    host._undo()
    assert state(host) == [
        (T("00:00"), T("00:10"), "A"),
        (T("00:10"), T("00:20"), "B"),
    ]


def test_delete_of_merged_away_value_pushes_nothing(host):
    """Mode 4: deleting a merged-away value used to no-op on execute but
    resurrect an overlapping phantom on undo. Now: a no-change gesture
    pushes nothing, so there is nothing to mis-undo."""
    host.intervals = [
        Interval(T("00:00"), T("00:10"), "A"),
        Interval(T("00:10"), T("00:20"), "A"),
    ]
    host._sort_and_merge_intervals()
    assert state(host) == [(T("00:00"), T("00:20"), "A")]

    ghost = Interval(T("00:10"), T("00:20"), "A")
    host._execute_command(DeleteIntervalCommand(host, ghost))

    assert len(host.undo_stack) == 0          # no-change gesture
    assert state(host) == [(T("00:00"), T("00:20"), "A")]
    host._undo()                              # must be a clean no-op
    assert state(host) == [(T("00:00"), T("00:20"), "A")]


def test_resize_into_adjacency_then_undo(host):
    """Resize whose result abuts a same-label neighbor: undo restores
    the exact prior state, no overlapping pair."""
    host.intervals = [
        Interval(T("00:00"), T("00:10"), "A"),
        Interval(T("00:20"), T("00:30"), "A"),
    ]
    host._execute_command(
        ResizeIntervalCommand(host, host.intervals[1], T("00:10"), T("00:30")))
    assert state(host) == [(T("00:00"), T("00:30"), "A")]

    host._undo()
    assert state(host) == [
        (T("00:00"), T("00:10"), "A"),
        (T("00:20"), T("00:30"), "A"),
    ]


def test_gap_fill_adjacent_same_label_undo(host):
    """Gap fills are exactly adjacent by construction (gaps.py) -- the
    highest-probability merge trigger. Each fill must undo cleanly."""
    host.intervals = [Interval(T("00:00"), T("00:10"), "UNKNOWN")]

    host._assign_gaps_to_label([(T("00:10"), T("00:20"))], "UNKNOWN")
    host._assign_gaps_to_label([(T("00:20"), T("00:30"))], "UNKNOWN")
    assert state(host) == [(T("00:00"), T("00:30"), "UNKNOWN")]
    assert len(host.undo_stack) == 2

    host._undo()
    assert state(host) == [(T("00:00"), T("00:20"), "UNKNOWN")]
    host._undo()
    assert state(host) == [(T("00:00"), T("00:10"), "UNKNOWN")]


# ---- macro boundaries (ledger 6) ----

def test_clear_range_split_is_one_undo(host):
    """A clear-range that splits an interval is ONE undo entry, and one
    undo restores the original exactly (incl. notes)."""
    host.intervals = [Interval(T("00:00"), T("01:00"), "A", "note")]

    results = host._clear_intervals_in_range(T("00:20"), T("00:30"))
    assert results["split"] == 1
    assert len(host.undo_stack) == 1

    host._undo()
    assert len(host.intervals) == 1
    iv = host.intervals[0]
    assert (iv.start, iv.end, iv.label, iv.notes) == (T("00:00"), T("01:00"), "A", "note")


def test_carve_and_add_replace_policy_is_one_undo(host):
    """Replace-policy add (carve + add) is one gesture."""
    host.intervals = [Interval(T("00:00"), T("01:00"), "A")]

    with host._gesture("replace add"):
        host._carve_existing_for_new_span(T("00:20"), T("00:40"))
        host._execute_command(AddIntervalCommand(host, Interval(T("00:20"), T("00:40"), "B")))

    assert len(host.undo_stack) == 1
    assert len(host.intervals) == 3

    host._undo()
    assert state(host) == [(T("00:00"), T("01:00"), "A")]


def test_max_undo_evicts_whole_gestures(host):
    """FIFO eviction drops whole gestures, never fragments, and honors a
    max_undo set after construction."""
    host.max_undo = 3
    for i in range(4):
        base = T(f"0{i}:00")
        host.intervals.append(Interval(base, base + pd.Timedelta(minutes=50), "A"))
    for i in range(4):
        base = T(f"0{i}:00")
        host._clear_intervals_in_range(
            base + pd.Timedelta(minutes=20), base + pd.Timedelta(minutes=30))

    assert len(host.undo_stack) == 3
    # Every reachable state is a whole number of gestures back: each undo
    # restores one interval to a whole [xx:00, xx:50) block.
    for expected_whole in (1, 2, 3):
        host._undo()
        whole = [iv for iv in host.intervals
                 if (iv.end - iv.start) == pd.Timedelta(minutes=50)]
        assert len(whole) == expected_whole
    # Stack exhausted: the 4th gesture was evicted, its state unreachable
    host._undo()
    assert host.status_var.get() == "Nothing to undo"


# ---- strict-mode invariant (ledger 5.1 + 7.1) ----

def test_no_overlap_invariant_fires_under_strict(host, monkeypatch):
    monkeypatch.setenv("CHRONOTAGGER_STRICT", "1")
    host.intervals = [
        Interval(T("00:00"), T("00:20"), "A"),
        Interval(T("00:10"), T("00:20"), "A"),
    ]
    with pytest.raises(IntervalInvariantError):
        host._check_interval_invariants()


def test_no_overlap_invariant_silent_when_off(host, monkeypatch):
    monkeypatch.setenv("CHRONOTAGGER_STRICT", "0")
    host.intervals = [
        Interval(T("00:00"), T("00:20"), "A"),
        Interval(T("00:10"), T("00:20"), "A"),
    ]
    host._check_interval_invariants()  # must not raise


def test_no_overlap_invariant_allows_adjacency(host, monkeypatch):
    monkeypatch.setenv("CHRONOTAGGER_STRICT", "1")
    host.intervals = [
        Interval(T("00:00"), T("00:10"), "A"),
        Interval(T("00:10"), T("00:20"), "B"),
    ]
    host._check_interval_invariants()  # adjacency is legal


# ---- redo correctness (previously untested repo-wide) ----

def test_redo_reproduces_post_execute_state(host):
    host.intervals = [Interval(T("00:00"), T("01:00"), "A")]
    host._execute_command(AddIntervalCommand(host, Interval(T("01:00"), T("02:00"), "A")))
    after = state(host)

    host._undo()
    host._redo()
    assert state(host) == after


def test_undo_redo_undo_is_idempotent(host):
    """Catches snapshot strategies that mutate their own stored copies."""
    host.intervals = [
        Interval(T("00:00"), T("00:10"), "A"),
        Interval(T("00:10"), T("00:20"), "B"),
    ]
    host._execute_command(RelabelIntervalCommand(host, host.intervals[1], "A"))

    host._undo()
    state1 = state(host)
    host._redo()
    host._undo()
    assert state(host) == state1


def test_macro_leaves_redo_stack_consistent(host):
    host.intervals = [Interval(T("00:00"), T("01:00"), "A")]
    host._clear_intervals_in_range(T("00:20"), T("00:30"))
    host._undo()
    assert len(host.redo_stack) == 1

    host._execute_command(AddIntervalCommand(host, Interval(T("02:00"), T("02:10"), "B")))
    assert len(host.redo_stack) == 0


def test_no_change_gesture_preserves_redo(host):
    """A gesture that changes nothing pushes no entry and does NOT
    clear the redo stack (behavior change D3, deliberate)."""
    host.intervals = [Interval(T("00:00"), T("01:00"), "A")]
    host._execute_command(AddIntervalCommand(host, Interval(T("02:00"), T("02:10"), "B")))
    host._undo()
    assert len(host.redo_stack) == 1

    ghost = Interval(T("03:00"), T("03:10"), "C")
    host._execute_command(DeleteIntervalCommand(host, ghost))  # no-change
    assert len(host.redo_stack) == 1
    assert len(host.undo_stack) == 0


def test_undo_and_redo_set_modified(host):
    host.intervals = []
    host._execute_command(AddIntervalCommand(host, Interval(T("00:00"), T("00:10"), "A")))
    assert host.modified is True

    host.modified = False
    host._undo()
    assert host.modified is True

    host.modified = False
    host._redo()
    assert host.modified is True


# ---- transactional gesture + selection upkeep (folds V2-M3/V3-M1/V3-M2) ----

def test_invariant_violation_rolls_back_gesture(host, monkeypatch):
    """A strict-mode violation must be rolled back, never recorded."""
    monkeypatch.setenv("CHRONOTAGGER_STRICT", "1")
    host.intervals = [Interval(T("00:00"), T("01:00"), "A")]

    with pytest.raises(IntervalInvariantError):
        with host._gesture("bad"):
            # Bypass the commands to inject a raw overlap
            host.intervals.append(Interval(T("00:30"), T("00:40"), "B"))

    assert state(host) == [(T("00:00"), T("01:00"), "A")]
    assert len(host.undo_stack) == 0
    assert host.modified is False


def test_selection_repoints_after_undo(host):
    host.intervals = [Interval(T("00:00"), T("00:10"), "A")]
    host._execute_command(AddIntervalCommand(host, Interval(T("01:00"), T("01:10"), "B")))
    host.selected_interval = host.intervals[0]

    host._undo()
    assert host.selected_interval is not None
    assert any(iv is host.selected_interval for iv in host.intervals)
    assert (host.selected_interval.start, host.selected_interval.end) == (T("00:00"), T("00:10"))


def test_forward_merge_clears_consumed_selection(host):
    """Adjacent-BEFORE add consumes the selected object in the merge;
    the gesture close must clear the selection instead of leaving a
    ghost that a follow-up relabel would silently mutate off-list."""
    host.intervals = [Interval(T("00:10"), T("00:20"), "A")]
    host.selected_interval = host.intervals[0]

    host._execute_command(AddIntervalCommand(host, Interval(T("00:00"), T("00:10"), "A")))
    assert state(host) == [(T("00:00"), T("00:20"), "A")]
    assert host.selected_interval is None


# ---- label-schema edits vs undo history (Q2 rider, folds V1-M3/V2-M2/V3-M5) ----

def _bind_label_manager(host):
    from chronotagger.labeler.mixins.labels import LabelsMixin
    host.class_combo = None
    host.current_class_var = None
    host.class_colors = {"UNKNOWN": "#cccccc", "A": "#ff0000", "B": "#0000ff"}
    host._apply_label_manager_result = LabelsMixin._apply_label_manager_result.__get__(host)


def test_label_manager_rename_invalidates_history(host):
    from chronotagger.labeler.dialogs.label_manager import LabelManagerResult
    _bind_label_manager(host)
    host._execute_command(AddIntervalCommand(host, Interval(T("00:00"), T("00:10"), "A")))
    assert len(host.undo_stack) == 1

    host.modified = False
    host._apply_label_manager_result(LabelManagerResult(
        classes=["UNKNOWN", "A2", "B"],
        class_colors={"UNKNOWN": "#cccccc", "A2": "#ff0000", "B": "#0000ff"},
        rename_map={"A": "A2"},
        reassign_map={},
    ))

    assert host.intervals[0].label == "A2"
    assert len(host.undo_stack) == 0
    assert len(host.redo_stack) == 0
    assert host.modified is True


def test_label_manager_recolor_preserves_history(host):
    from chronotagger.labeler.dialogs.label_manager import LabelManagerResult
    _bind_label_manager(host)
    host._execute_command(AddIntervalCommand(host, Interval(T("00:00"), T("00:10"), "A")))

    host._apply_label_manager_result(LabelManagerResult(
        classes=["UNKNOWN", "A", "B"],
        class_colors={"UNKNOWN": "#cccccc", "A": "#00ff00", "B": "#0000ff"},
        rename_map={},
        reassign_map={},
    ))

    assert len(host.undo_stack) == 1
    assert host.modified is True


def test_add_at_full_stack_still_counts_as_change(host):
    """Regression (recheck M2): at max_undo the push+trim leaves the
    stack LENGTH unchanged; no-op detection must use identity of the
    stack top, or a real add at a full stack gets reported as
    'Nothing Added' and skips sync/update/autosave."""
    host.max_undo = 2
    host._execute_command(AddIntervalCommand(host, Interval(T("00:00"), T("00:10"), "A")))
    host._execute_command(AddIntervalCommand(host, Interval(T("01:00"), T("01:10"), "A")))
    assert len(host.undo_stack) == 2  # full

    host._add_intervals_with_policy([(T("02:00"), T("02:10"))], "B", "skip")

    assert state(host)[-1] == (T("02:00"), T("02:10"), "B")
    assert host.status_var.get() == "Added 1 B interval(s)"
    assert len(host.undo_stack) == 2  # oldest evicted, length constant
