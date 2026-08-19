"""
Pack 1 GUI-backed tests: session load invalidates undo history; a
multi-span add is one gesture. See test_undo_integrity.py for the
GUI-free coverage.
"""

from chronotagger.core.models import Interval
from chronotagger.core.commands import AddIntervalCommand


def test_load_session_clears_undo_history(labeler, tmp_path):
    """Ledger 5.2: Ctrl+Z after Load Session used to replay commands
    from the previous session into the loaded one."""
    idx = labeler.df.index

    labeler._execute_command(AddIntervalCommand(labeler, Interval(idx[10], idx[20], "A")))
    p1 = tmp_path / "s1.json"
    labeler.save(str(p1))

    labeler._execute_command(AddIntervalCommand(labeler, Interval(idx[30], idx[40], "B")))

    labeler._load_session(str(p1))
    assert len(labeler.undo_stack) == 0
    assert len(labeler.redo_stack) == 0
    assert labeler.selected_interval is None
    assert labeler.modified is False

    before = [(iv.start, iv.end, iv.label) for iv in labeler.intervals]
    labeler._undo()  # must be a clean no-op, not stale-command injection
    assert [(iv.start, iv.end, iv.label) for iv in labeler.intervals] == before


def test_multi_span_add_is_one_gesture(labeler):
    """Ledger 6: a multi-span add (box-select / rule apply) is ONE undo
    entry, matching the README's 'full undo support' promise."""
    idx = labeler.df.index
    spans = [(idx[5], idx[10]), (idx[20], idx[25]), (idx[40], idx[45])]

    labeler._add_intervals_with_policy(spans, "A", "skip")

    assert len(labeler.intervals) == 3
    assert len(labeler.undo_stack) == 1

    labeler._undo()
    assert len(labeler.intervals) == 0
