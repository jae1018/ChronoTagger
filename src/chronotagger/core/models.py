"""
models.py

Dataclasses and core data structures used by ChronoTagger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pandas as pd

from .tracks import DEFAULT_TRACK_ID, copy_meta


@dataclass
class Interval:
    """
    Represents a single labeled half-open time interval [start, end).

    Attributes
    ----------
    start : pd.Timestamp
        Start timestamp (inclusive).
    end : pd.Timestamp
        End timestamp (exclusive).
    label : str
        Class label for the interval.
    notes : Optional[str]
        Freeform notes.
    track : str
        The id of the TRACK this interval sits on (Pack M1). Opaque and
        immutable; see core/tracks.py. The dataclass default is
        DEFAULT_TRACK_ID and that default is LOAD-BEARING: three of the
        eleven Interval(...) construction sites in src/ used to pass
        only three positional arguments, and every bare
        Interval(a, b, "A") in the suite still does.
    meta : Dict[str, Any]
        Free-form, UNINTERPRETED provenance. It round-trips verbatim
        through the session, the autosave, the recovery and undo/redo,
        and nothing in this package reads a key of it. Namespace your
        keys ("import.source", "catalog.fractions") so two writers
        cannot collide.

    ANYONE ADDING A FIELD, READ THIS: copy_intervals (core/commands.py)
    names all six fields explicitly, and a seventh field not added
    there would be erased in silence by every undo. The guard is a
    test, not a convention -- len(dataclasses.fields(Interval)) == 6 is
    pinned in tests/test_packm1_tracks_model.py.
    """
    start: pd.Timestamp
    end: pd.Timestamp
    label: str
    notes: Optional[str] = None
    track: str = DEFAULT_TRACK_ID
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize so that start <= end."""
        if self.start > self.end:
            self.start, self.end = self.end, self.start

    def overlaps(self, other: "Interval") -> bool:
        """
        Return True if this interval overlaps `other`.

        We treat intervals as half-open [start, end); adjacency is not overlap.
        """
        return not (self.end <= other.start or self.start >= other.end)

    def contains(self, timestamp: pd.Timestamp) -> bool:
        """Return True if `timestamp` ∈ [start, end)."""
        return self.start <= timestamp < self.end

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict.

        Pack M1: THIS IS THE SERIALIZER CHOKE POINT. `track` is emitted
        always; `meta` only when it is non-empty, which costs the 2,255
        intervals already on disk nothing at all (not one of them
        carries meta) and keeps a session file readable by eye. Eight of
        the sixteen serialization sites in this package need no edit of
        their own because they route through here.
        """
        d = {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "label": self.label,
            "notes": self.notes,
            "track": self.track,
        }
        if self.meta:
            # v2 fold F8: a COPY, not the live dict. These two methods
            # are the choke point every serialization site routes
            # through, and a caller that post-processes the payload
            # must not be able to edit the MODEL through it. Measured
            # before this was closed: d["meta"] is iv.meta.
            d["meta"] = copy_meta(self.meta)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Interval":
        """Deserialize from a dict produced by `to_dict`.

        A v1 interval -- four keys, no `track`, no `meta` -- reads as an
        interval on the default track with empty meta. That is a default
        for an absent field, in exactly the shape `notes` has always
        had, and not a compatibility layer.
        """
        return cls(
            start=pd.Timestamp(d["start"]),
            end=pd.Timestamp(d["end"]),
            label=d["label"],
            notes=d.get("notes"),
            track=d.get("track") or DEFAULT_TRACK_ID,
            # v2 fold F8: copy_meta, not dict() -- a shallow dict keeps
            # the PAYLOAD's nested values, so mutating the parsed JSON
            # afterwards reached into the interval it produced.
            meta=copy_meta(d.get("meta") or {}),
        )
