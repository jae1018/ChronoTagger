"""
commands.py

Command objects used for undo/redo in ChronoTagger.

Undo model (gesture snapshots): the undo/redo stacks hold
GestureCommand objects only. A GestureCommand stores value-copies of
the whole interval list from before and after one user gesture;
undo/redo restore those copies wholesale. The operation classes below
(Add/Delete/Relabel/Resize) implement execute() only -- reversal is
the gesture snapshot's job, so they carry no undo bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import pandas as pd

from .models import Interval


class IntervalInvariantError(RuntimeError):
    """Raised in strict mode when the interval list violates an invariant."""


def copy_intervals(intervals: List[Interval]) -> List[Interval]:
    """Value-copy a list of intervals (new objects, same field values)."""
    return [Interval(iv.start, iv.end, iv.label, iv.notes) for iv in intervals]


class Command:
    """Base class for undoable commands."""

    def execute(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def undo(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class GestureCommand(Command):
    """
    One user gesture = one undo entry.

    Holds value-copies of the interval list captured before and after
    the gesture ran. undo() restores `before`; execute() (used by redo)
    restores `after`. Restores use slice assignment; note that list
    identity is NOT an invariant in this codebase (the merge rebinds
    self.intervals), so nothing may rely on it either way.
    """
    labeler: "TimeIntervalLabeler" = field(repr=False)
    name: str = ""
    before: List[Interval] = field(default_factory=list)
    after: List[Interval] = field(default_factory=list)

    def execute(self) -> None:
        self.labeler.intervals[:] = copy_intervals(self.after)

    def undo(self) -> None:
        self.labeler.intervals[:] = copy_intervals(self.before)


@dataclass
class AddIntervalCommand(Command):
    """
    Add an interval, trimming/removing overlaps.

    Reversal is the enclosing gesture snapshot's job (GestureCommand);
    no bookkeeping is recorded here.
    """
    labeler: "TimeIntervalLabeler"
    interval: Interval

    def execute(self) -> None:
        self.labeler._remove_overlapping_intervals(self.interval)
        self.labeler.intervals.append(self.interval)
        self.labeler._sort_and_merge_intervals()


@dataclass
class DeleteIntervalCommand(Command):
    """Delete a specific interval (matched by value)."""
    labeler: "TimeIntervalLabeler"
    interval: Interval

    def execute(self) -> None:
        if self.interval in self.labeler.intervals:
            self.labeler.intervals.remove(self.interval)


@dataclass
class RelabelIntervalCommand(Command):
    """Change an interval's label."""
    labeler: "TimeIntervalLabeler"
    interval: Interval
    new_label: str

    def execute(self) -> None:
        self.interval.label = self.new_label
        self.labeler._sort_and_merge_intervals()


@dataclass
class ResizeIntervalCommand(Command):
    """
    Resize or move an existing interval to [new_start, new_end].

    Same overlap pipeline as AddInterval: remove the original, add a
    resized copy (same label/notes), resolve neighbor overlaps, merge
    adjacents. Reversal is the gesture snapshot's job.
    """
    labeler: "TimeIntervalLabeler"
    interval: Interval           # original interval object (by value equality)
    new_start: pd.Timestamp
    new_end: pd.Timestamp

    def execute(self) -> None:
        if self.interval in self.labeler.intervals:
            self.labeler.intervals.remove(self.interval)
        new_iv = Interval(
            start=min(self.new_start, self.new_end),
            end=max(self.new_start, self.new_end),
            label=self.interval.label,
            notes=self.interval.notes,
        )
        self.labeler._remove_overlapping_intervals(new_iv)
        self.labeler.intervals.append(new_iv)
        self.labeler._sort_and_merge_intervals()
