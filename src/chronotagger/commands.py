"""
commands.py

Command objects used for undo/redo in ChronoTagger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .models import Interval


class Command:
    """Base class for undoable commands."""

    def execute(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def undo(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class AddIntervalCommand(Command):
    """
    Add an interval, trimming/removing overlaps. Supports clean undo.

    We record:
      - intervals that were removed (full originals),
      - intervals that were auto-added as trimmed remainders,
      - the new interval itself.

    On undo:
      - remove the new interval,
      - remove the trimmed remainders we added,
      - restore the removed originals.
    """
    labeler: "TimeIntervalLabeler"
    interval: Interval
    removed_intervals: List[Interval] = field(default_factory=list)
    added_trims: List[Interval] = field(default_factory=list)

    def execute(self) -> None:
        self.removed_intervals, self.added_trims = \
            self.labeler._remove_overlapping_intervals(self.interval)
        self.labeler.intervals.append(self.interval)
        self.labeler._sort_and_merge_intervals()

    def undo(self) -> None:
        # Remove the interval we added
        if self.interval in self.labeler.intervals:
            self.labeler.intervals.remove(self.interval)
        # Remove any trimmed fragments we added
        for iv in self.added_trims:
            if iv in self.labeler.intervals:
                self.labeler.intervals.remove(iv)
        # Restore the originals that were removed
        self.labeler.intervals.extend(self.removed_intervals)
        self.labeler._sort_and_merge_intervals()


@dataclass
class DeleteIntervalCommand(Command):
    """Delete a specific interval."""
    labeler: "TimeIntervalLabeler"
    interval: Interval

    def execute(self) -> None:
        if self.interval in self.labeler.intervals:
            self.labeler.intervals.remove(self.interval)

    def undo(self) -> None:
        self.labeler.intervals.append(self.interval)
        self.labeler._sort_and_merge_intervals()


@dataclass
class RelabelIntervalCommand(Command):
    """Change an interval's label."""
    labeler: "TimeIntervalLabeler"
    interval: Interval
    new_label: str
    old_label: str = ""

    def execute(self) -> None:
        self.old_label = self.interval.label
        self.interval.label = self.new_label
        self.labeler._sort_and_merge_intervals()

    def undo(self) -> None:
        self.interval.label = self.old_label
        self.labeler._sort_and_merge_intervals()
