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
from typing import List, Optional
import os
import pandas as pd

from .models import Interval
from .tracks import (Track, copy_meta, copy_tracks, stray_tracks,
                     track_ids)


class IntervalInvariantError(RuntimeError):
    """Raised in strict mode when the interval list violates an invariant."""


def copy_intervals(intervals: List[Interval]) -> List[Interval]:
    """Value-copy a list of intervals (new objects, same field values).

    Pack M1: an EXPLICIT six-field keyword constructor, with a LAZY deep
    copy of `meta`. Measured at 2,000 intervals with empty meta -- which
    is 2,255 of the 2,255 intervals on disk -- against the alternatives:
    this 1.62 ms, the old four-field positional line 1.32 ms (+23 %),
    dataclasses.replace with a lazy deepcopy 3.81 ms (+188 %), replace
    with an unconditional deepcopy 5.76 ms (+336 %). _gesture copies the
    list twice, so the choice is +0.6 ms per gesture against +4.9 ms.

    A shallow dict(iv.meta) is not on the table: measured, it SHARES
    nested lists between snapshots, which lets an undo edit the future.

    The price of naming the fields is that this list can go stale. That
    is bought off with a pin rather than with a slower copy:
    len(dataclasses.fields(Interval)) == 6.
    """
    return [
        Interval(
            start=iv.start,
            end=iv.end,
            label=iv.label,
            notes=iv.notes,
            track=iv.track,
            meta=copy_meta(iv.meta),
        )
        for iv in intervals
    ]


def check_interval_invariants(intervals, table=None) -> None:
    """Strict-mode invariants on an interval set. Pack M1.

    Two clauses, in this order:

      MEMBERSHIP -- every interval sits on a track the table knows.
        This is what makes a missed Interval(...) construction site LOUD
        instead of silent. Measured on the shipped tree: one ordinary
        human add over a range covered by a locked imported track carves
        that track's interval in two and puts BOTH fragments on a track
        called 'default' that is not in the table, with no dialog and no
        error anywhere. With this clause the same add raises and the
        gesture's own rollback leaves the locked track's two intervals
        untouched.

      PER-TRACK NON-OVERLAP -- within one track, no two intervals
        overlap. Half-open semantics: exact adjacency is legal and must
        not trip this. This is STRICTLY STRONGER than the flat check it
        replaces -- it still catches every within-track overlap, and it
        stops calling a legal cross-track pair a violation. The message
        names the track, because with K lanes an unnamed one is
        unactionable.

    `table` is the track table those intervals are about to live under,
    and it must be the table being INSTALLED rather than the live one:
    the first prototype validated a loaded interval set against the OLD
    table and a perfectly legal v1 migration failed with "intervals on
    tracks that are not in the track table: default (table has geom,
    human)". When `table` is None the membership clause is SKIPPED -- a
    host with no table has nothing to be a member of -- and the
    per-track scan still runs.

    Module-level, not a method, for the same reason
    write_label_map_sidecar is (io_export.py): the GUI-free hosts in
    tests/ bind a NAMED LIST of mixin methods onto a plain object, and a
    free function needs no new entry in that list. It also lets a LOAD
    path validate a set it has not published yet.
    """
    if os.environ.get("CHRONOTAGGER_STRICT") != "1":
        return
    if table is not None:
        stray = stray_tracks(intervals, table)
        if stray:
            raise IntervalInvariantError(
                "intervals on tracks that are not in the track table: "
                + ", ".join(stray)
                + " (table has " + ", ".join(sorted(track_ids(table))) + ")")
    by = {}
    for iv in intervals:
        by.setdefault(iv.track, []).append(iv)
    for tid in sorted(by):
        ivs = sorted(by[tid], key=lambda iv: (iv.start, iv.end))
        for a, b in zip(ivs, ivs[1:]):
            if a.end > b.start:
                raise IntervalInvariantError(
                    f"overlapping intervals on track {tid!r}: "
                    f"[{a.start}, {a.end}) {a.label!r} / "
                    f"[{b.start}, {b.end}) {b.label!r}"
                )


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
    # Pack M1: the TRACK TABLE rides in the SAME snapshot as the
    # intervals. It has to, because the column ingest CREATES a track
    # INSIDE a gesture, and a table kept outside the snapshot would
    # leave an empty lane behind after the undo. A second undo stack for
    # the table is the alternative and it is rejected: two stacks can
    # desync -- undo the delete-track and get the table back without its
    # intervals, or the other way round -- and the whole-list snapshot
    # exists precisely to make that impossible. Priced at 0.034 ms /
    # 3,276 bytes for four tracks against 1.068 ms / 934 KB for one
    # 2,000-interval interval snapshot: three orders of magnitude below
    # the noise.
    #
    # None, not [], is the "this host has no table" value. A host built
    # before tracks existed must not have an empty table written onto it
    # by a restore.
    tracks_before: Optional[List[Track]] = None
    tracks_after: Optional[List[Track]] = None

    def execute(self) -> None:
        self.labeler.intervals[:] = copy_intervals(self.after)
        if self.tracks_after is not None:
            self.labeler.tracks[:] = copy_tracks(self.tracks_after)

    def undo(self) -> None:
        self.labeler.intervals[:] = copy_intervals(self.before)
        if self.tracks_before is not None:
            self.labeler.tracks[:] = copy_tracks(self.tracks_before)


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
            # Pack M1, construction site 2 of 11: a resize must not move
            # an interval onto another lane and must not drop its
            # provenance.
            track=self.interval.track,
            meta=copy_meta(self.interval.meta),
        )
        self.labeler._remove_overlapping_intervals(new_iv)
        self.labeler.intervals.append(new_iv)
        self.labeler._sort_and_merge_intervals()
