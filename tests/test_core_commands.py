from chronotagger.core.models import Interval
from chronotagger.core.commands import AddIntervalCommand, DeleteIntervalCommand, RelabelIntervalCommand


def test_add_merge_and_undo(labeler):
    t0, t1 = labeler.df.index[10], labeler.df.index[30]
    t2, t3 = labeler.df.index[30], labeler.df.index[50]

    # Add first, then an adjacent same-label interval -> should merge
    labeler._execute_command(AddIntervalCommand(labeler, Interval(t0, t1, "A")))
    labeler._execute_command(AddIntervalCommand(labeler, Interval(t2, t3, "A")))

    assert len(labeler.intervals) == 1
    assert labeler.intervals[0].start == t0
    assert labeler.intervals[0].end == t3

    # Undo the second add: the merge must unwind, restoring the first
    # add exactly (regression: this used to be a silent no-op).
    labeler._undo()
    assert len(labeler.intervals) == 1
    assert labeler.intervals[0].start == t0
    assert labeler.intervals[0].end == t1

    # Delete the survivor, then undo the delete
    labeler._execute_command(DeleteIntervalCommand(labeler, labeler.intervals[0]))
    assert len(labeler.intervals) == 0
    labeler._undo()
    assert len(labeler.intervals) == 1
    assert labeler.intervals[0].start == t0
    assert labeler.intervals[0].end == t1


def test_relabel_and_undo(labeler):
    a, b = labeler.df.index[5], labeler.df.index[20]
    labeler._execute_command(AddIntervalCommand(labeler, Interval(a, b, "X")))

    labeler._execute_command(RelabelIntervalCommand(labeler, labeler.intervals[0], "Y"))
    assert labeler.intervals[0].label == "Y"
    labeler._undo()
    assert labeler.intervals[0].label == "X"
