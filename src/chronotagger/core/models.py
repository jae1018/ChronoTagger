"""
models.py

Dataclasses and core data structures used by ChronoTagger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd


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
    """
    start: pd.Timestamp
    end: pd.Timestamp
    label: str
    notes: Optional[str] = None

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
        """Serialize to a JSON-friendly dict."""
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "label": self.label,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Interval":
        """Deserialize from a dict produced by `to_dict`."""
        return cls(
            start=pd.Timestamp(d["start"]),
            end=pd.Timestamp(d["end"]),
            label=d["label"],
            notes=d.get("notes"),
        )
