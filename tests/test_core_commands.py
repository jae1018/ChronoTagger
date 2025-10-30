import pandas as pd
from chronotagger.core.models import Interval
from chronotagger.core.commands import AddIntervalCommand, DeleteIntervalCommand, RelabelIntervalCommand


def test_add_merge_and_undo(labeler):
    t0, t1 = labeler.df.index[10], labeler.df.index[30]
    t2, t3 = labeler.df.index[30], labeler.df.index[50]

    # Add first, then an adjacent same-label interval -> should merge
    AddIntervalCommand(labeler, Interval(t0, t1, "A")).execute()
    AddIntervalCommand(labeler, Interval(t2, t3, "A")).execute()

    assert len(labeler.intervals) == 1
    assert labeler.intervals[0].start == t0
    assert labeler.intervals[0].end == t3

    # Now delete and undo
    iv = labeler.intervals[0]
    dc = DeleteIntervalCommand(labeler, iv)
    dc.execute()
    assert len(labeler.intervals) == 0
    dc.undo()
    assert len(labeler.intervals) == 1


def test_relabel_and_undo(labeler):
    a, b = labeler.df.index[5], labeler.df.index[20]
    iv = Interval(a, b, "X")
    AddIntervalCommand(labeler, iv).execute()

    cmd = RelabelIntervalCommand(labeler, iv, "Y")
    cmd.execute()
    assert labeler.intervals[0].label == "Y"
    cmd.undo()
    assert labeler.intervals[0].label == "X"
