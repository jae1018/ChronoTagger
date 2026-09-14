"""
tracks.py

The TRACK TABLE: what a label lane IS, independent of the intervals
that sit on it.

Pack M1. One `Interval` used to carry the whole story -- a span and one
class name out of one vocabulary. A Track owns the vocabulary, its
colours, a kind, and two view flags; an Interval names its track and
nothing more.

THE ID IS OPAQUE AND IMMUTABLE; THE NAME IS FREE TEXT.
`Interval.track` holds `Track.id`, never `Track.name`, because the id
becomes an export COLUMN NAME (`label_id__<id>`) and a sidecar FILE
STEM (`<data>_label_map__<id>.json`). A display name is text somebody
will put a space or a slash in -- measured: an id containing `/` makes
the sidecar path raise `Invalid name`, and an id containing `.` poisons
`df.query` on the exported parquet. So the id is validated to
`[A-Za-z0-9_-]{1,32}` at construction and never changes; renaming a
track touches one field of one table row and never touches an interval,
which is what makes rename history-preserving.

`default` is the reserved id of the session's own track. Nothing else
enforces the reservation and nothing else has to: the default table
always contains that row, and every id-creating path refuses a
duplicate id.

DEPENDENCY DIRECTION. This module imports `copy`, `dataclasses`,
`typing` and `pandas` and nothing from chronotagger. `core/models.py`
imports DEFAULT_TRACK_ID and copy_meta from here, so the edge
tracks.py -> models.py is one-way and must stay that way. `copy_meta`
lives HERE rather than in core/commands.py for exactly that reason:
models.py's to_dict/from_dict are the serializer choke point and they
need it, and models.py may not import commands.py.

THE `_of(host)` HELPERS. `table_of`, `active_track_of` and
`active_id_of` take a HOST (a labeler, or one of the GUI-free mixin
hosts the suite builds) rather than being methods. That is the
`write_label_map_sidecar` precedent from io_export.py: a free function
needs no entry in a mock host's bound-method list. They are also the
reason a host built before tracks existed still behaves as exactly one
default track instead of raising AttributeError.
"""

from __future__ import annotations

import copy as _copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

DEFAULT_TRACK_ID = "default"
DEFAULT_TRACK_NAME = "Labels"

# The one-lane default palette, lifted here so the GUI-free ingest path
# can colour a new track without a labeler. TimeIntervalLabeler keeps its
# own DEFAULT_COLORS class attribute, unchanged, and passes it in as
# `palette` -- so at runtime there is exactly one authority and this list
# is only the fallback.
DEFAULT_COLORS: List[str] = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc949", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ac",
]

TRACK_KINDS = ("interval", "point")

_ID_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")
_ID_MAX = 32


def validate_track_id(track_id) -> str:
    """Return `track_id` as a str, or raise ValueError saying why not.

    The charset is deliberately narrower than a filename's: the id is
    interpolated into a parquet column name AND into a sidecar
    filename, and the intersection of what both accept safely is
    letters, digits, underscore and hyphen.
    """
    if not isinstance(track_id, str):
        raise ValueError(
            "track id must be a string, not %s" % type(track_id).__name__)
    if not track_id:
        raise ValueError("track id must not be empty")
    if len(track_id) > _ID_MAX:
        raise ValueError(
            "track id %r is %d characters; the limit is %d"
            % (track_id, len(track_id), _ID_MAX))
    bad = sorted({c for c in track_id if c not in _ID_CHARS})
    if bad:
        raise ValueError(
            "track id %r contains characters that are not allowed "
            "(%s); use letters, digits, underscore or hyphen only -- "
            "the id becomes an export column name and a sidecar filename"
            % (track_id, ", ".join(repr(c) for c in bad)))
    return track_id


@dataclass
class Track:
    """One label lane.

    id            opaque, immutable, filename-and-column safe
    name          what the user sees; free text; renameable
    classes       this lane's vocabulary, in id order (index == label id)
    class_colors  {class name: "#rrggbb"} for this lane
    kind          "interval" or "point"  -- MODEL state
    locked        the lane refuses edits                -- MODEL state
    visible       the lane is painted                   -- VIEW state
    order         the lane's position, top to bottom     -- VIEW state

    All four flags persist. `locked` and `kind` are MODEL state because
    they change what would be exported or what an edit is allowed to do;
    `visible` and `order` are VIEW state and change only what you see.
    M1 ships no control that toggles any of them -- the ingest sets
    `locked` at construction -- so the distinction costs nothing yet and
    is written down here so M2 does not have to guess.
    """

    id: str
    name: str = ""
    classes: List[str] = field(default_factory=list)
    class_colors: Dict[str, str] = field(default_factory=dict)
    kind: str = "interval"
    locked: bool = False
    visible: bool = True
    order: int = 0

    def __post_init__(self) -> None:
        self.id = validate_track_id(self.id)
        self.name = str(self.name) if self.name else self.id
        self.classes = [str(c) for c in self.classes]
        self.class_colors = {str(k): str(v)
                             for k, v in dict(self.class_colors).items()}
        if self.kind not in TRACK_KINDS:
            raise ValueError(
                "track kind %r is not one of %s"
                % (self.kind, ", ".join(repr(k) for k in TRACK_KINDS)))
        self.locked = bool(self.locked)
        self.visible = bool(self.visible)
        self.order = int(self.order)

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict. Every field, always."""
        return {
            "id": self.id,
            "name": self.name,
            "classes": list(self.classes),
            "class_colors": dict(self.class_colors),
            "kind": self.kind,
            "locked": bool(self.locked),
            "visible": bool(self.visible),
            "order": int(self.order),
        }

    @classmethod
    def from_dict(cls, d) -> "Track":
        """Deserialize a dict produced by `to_dict`.

        Tolerant of absent optional keys, in the same shape
        Interval.from_dict is tolerant of an absent `notes`, so a
        hand-written table in a driver or a test does not have to spell
        all eight fields.
        """
        if isinstance(d, Track):
            return copy_track(d)
        if not isinstance(d, dict):
            raise ValueError(
                "a track must be a dict or a Track, not %s"
                % type(d).__name__)
        if "id" not in d:
            raise ValueError("a track needs an id")
        return cls(
            id=d["id"],
            name=d.get("name") or "",
            classes=list(d.get("classes") or []),
            class_colors=dict(d.get("class_colors")
                              or d.get("colors") or {}),
            kind=d.get("kind") or "interval",
            locked=bool(d.get("locked", False)),
            visible=bool(d.get("visible", True)),
            order=int(d.get("order", 0) or 0),
        )


def copy_track(t: Track) -> Track:
    """Value-copy one row: new lists and dicts, same values."""
    return Track(
        id=t.id,
        name=t.name,
        classes=list(t.classes),
        class_colors=dict(t.class_colors),
        kind=t.kind,
        locked=t.locked,
        visible=t.visible,
        order=t.order,
    )


def copy_tracks(tracks) -> List[Track]:
    """Value-copy a whole table.

    The gesture snapshot carries one of these beside its interval
    snapshot (core/commands.py). Measured: 0.034 ms / 3,276 bytes at
    four tracks, against 1.068 ms / 934 KB for one 2,000-interval
    interval snapshot -- three orders of magnitude below the noise,
    which is why the table rides in the SAME snapshot rather than in a
    second stack that could desync from it.
    """
    return [copy_track(t) for t in (tracks or [])]


def copy_meta(meta):
    """A LAZY, FLAT-FAST copy of one interval's `meta`. Pack M1.

    Three paths, in the order the cases actually occur:

      EMPTY -> a fresh {}. This is the common case by a mile: not one of
        the 2,255 intervals on disk carries a meta, and deep-copying an
        empty dict 2,000 times per gesture is milliseconds of nothing.

      FLAT (every value is None, str, int, float or bool) -> dict(meta).
        This is what the COLUMN INGEST stamps on every interval it makes
        -- three scalars -- so it is the shape a session with an imported
        rule track copies 2,349 times on every gesture. Measured at 2,000
        such intervals: deepcopy 6.58 ms, dict() 3.13 ms. A shallow copy
        of a flat dict shares nothing that can be mutated, so it is not
        the shortcut that was rejected; it is the same guarantee at half
        the price.

      ANYTHING ELSE -> _copy.deepcopy. A shallow dict(meta) over a NESTED
        value SHARES the nested object between the original and the copy
        -- measured -- which lets an undo edit the future. `type(v) in
        (...)` rather than isinstance on purpose: a subclass takes the
        deep path, which is slower and never wrong.

    Used by copy_intervals, by every split-fragment builder, and by
    Interval.to_dict / from_dict -- a fragment of the user's interval
    must carry the same provenance, not a reference to it, and neither
    must a payload.

    IT LIVES HERE, not in core/commands.py, because models.py needs it
    and models.py may not import commands.py. core/commands.py imports
    it from this module, so there is exactly one implementation and
    chronotagger.core.commands.copy_meta still resolves.
    """
    if not meta:
        return {}
    for v in meta.values():
        if v is None or type(v) in (str, int, float, bool):
            continue
        return _copy.deepcopy(meta)
    return dict(meta)


def snapshot_tracks(host) -> Optional[List[Track]]:
    """A value-copy of a host's track table, or None when it has none.

    None -- not [] -- is the "this host has no table" value, and it
    travels all the way into GestureCommand, whose restore is guarded on
    it. A GUI-free host built before tracks existed therefore behaves
    exactly as it did before Pack M1 instead of having an empty table
    written onto it by a restore.

    A free function, not a method, so a mock host that binds a NAMED LIST
    of mixin methods needs no new entry for it.
    """
    table = getattr(host, "tracks", None)
    return copy_tracks(table) if table is not None else None


def colors_for(classes, palette=None) -> Dict[str, str]:
    """The default colour map for a vocabulary: palette[i % len]."""
    pal = list(palette) if palette else list(DEFAULT_COLORS)
    if not pal:
        pal = list(DEFAULT_COLORS)
    return {str(c): pal[i % len(pal)] for i, c in enumerate(classes)}


def default_table(classes, class_colors=None, palette=None,
                  track_id=DEFAULT_TRACK_ID,
                  name=DEFAULT_TRACK_NAME) -> List[Track]:
    """The one-row table a single-track session runs on.

    This is what `classes=` / `class_colors=` build, and what a v1
    session or autosave migrates into.
    """
    classes = [str(c) for c in (classes or [])]
    colors = (dict(class_colors) if class_colors
              else colors_for(classes, palette))
    return [Track(id=track_id, name=name, classes=classes,
                  class_colors=colors, order=0)]


def find_track(tracks, track_id) -> Optional[Track]:
    """The row with this id, or None."""
    for t in (tracks or []):
        if t.id == track_id:
            return t
    return None


def track_ids(tracks) -> List[str]:
    """Every id in the table, in table order."""
    return [t.id for t in (tracks or [])]


def table_of(host) -> List[Track]:
    """The host's track table, or a synthesized one-row default table.

    Every mixin-side reader goes through this. A host that predates
    tracks -- one of the GUI-free hosts the suite binds mixin methods
    onto, or a caller's own object -- therefore reads as exactly ONE
    default track whose vocabulary is that host's own `classes`, which
    is the same thing the real class's one-row table holds. The
    synthesized table is a fresh list: writing to it changes nothing,
    which is correct, because a host with no table has no table to
    change.
    """
    tbl = getattr(host, "tracks", None)
    if tbl:
        return tbl
    return default_table(list(getattr(host, "classes", None) or []),
                         dict(getattr(host, "class_colors", None) or {}))


def visible_rows(tracks) -> List[Track]:
    """The rows the strip PAINTS, top to bottom. ONE authority.

    Sorted by `order` with the table position as the tie-break, so a
    table whose orders are all 0 -- a hand-written driver table -- paints
    in the order the driver wrote it. A row with `visible=False` is not
    in this list.

    `core.lanes.visible_lanes` is a one-line host-side wrapper over this
    and nothing else re-implements the sort, because Pack M2.6's
    resolution rule below has to agree with the PAINTER exactly, and two
    copies of one sort is how they stop agreeing.
    """
    rows = [t for t in (tracks or []) if getattr(t, "visible", True)]
    return sorted(rows, key=lambda t: (int(getattr(t, "order", 0) or 0),))


def resolve_active_row(host) -> Track:
    """THE ONE RESOLUTION RULE for "which lane is active". Pack M2.6.

    The row `_active_track_id` names -- and, when it names NO ROW OF THE
    TABLE, the FIRST VISIBLE lane, which is exactly the lane
    `lane_layout` already falls back to for painting. With no visible
    lane at all it is the first row, because a table always has one and
    an edit has to land somewhere nameable.

    WHY IT EXISTS. `add_track_from_column` creates a lane INSIDE ONE
    GESTURE, so a single Ctrl+Z removes the row while `_active_track_id`
    still names it: the id is view state and is not part of the gesture
    snapshot. Before this rule the guard and the writers disagreed about
    that state -- `is_locked` looked the id up, found no row and answered
    False, while every writer stamped the STALE ID onto the interval it
    built. Measured end to end on the feel-test driver
    (`probe_s3_refute_reach`): Fill Gaps was NOT refused and wrote an
    interval onto a lane no table holds (256 -> 257, painted nowhere and
    counted nowhere); Manage Labels rewrote the FIRST row's vocabulary,
    reported "Labels updated" and wiped the undo stack; Clear Range
    deleted; and the autosave then persisted an interval whose recovery
    raises ValueError at launch.

    A HIDDEN active lane is NOT stale and is NOT resolved away: the row
    exists, it simply is not painted, and `lane_layout` keeps its
    `active_row` None on purpose.
    """
    tbl = table_of(host)
    tid = getattr(host, "_active_track_id", None)
    if tid:
        row = find_track(tbl, tid)
        if row is not None:
            return row
    vis = visible_rows(tbl)
    return vis[0] if vis else tbl[0]


def active_id_of(host) -> str:
    """The id of the track a gesture writes to and the screen reads.

    Pack M2.6: this is `resolve_active_row(host).id`, so a STALE
    `_active_track_id` -- one naming no row of the table -- reads as the
    lane the user can actually SEE rather than as a ghost. Every writer
    that stamps this id therefore stamps a real row, and `is_locked` /
    `refuse_if_locked`, which resolve through the same call, judge the
    same row.

    It went through `getattr(host, "tracks")` before and it goes through
    `table_of(host)` now, so a host built before tracks existed still
    reads as exactly one default track: `DEFAULT_TRACK_ID` is that
    table's only id, which is the answer it always gave.
    """
    return resolve_active_row(host).id


def active_track_of(host) -> Track:
    """The active row. Pack M2.6: resolved by `resolve_active_row`."""
    return resolve_active_row(host)


def intervals_on(intervals, track_id) -> List[Any]:
    """Just the intervals that sit on one track, in the given order."""
    return [iv for iv in intervals if iv.track == track_id]


def stray_tracks(intervals, tracks) -> List[str]:
    """Track ids the intervals name that the table does not contain.

    Sorted, deduplicated. This is the MEMBERSHIP clause's evidence and
    the export's orphan-track refusal, in one place.
    """
    known = set(track_ids(tracks))
    return sorted({iv.track for iv in intervals} - known)


def union_covered(intervals) -> "pd.Timedelta":
    """The time covered by AT LEAST ONE interval, on any track.

    Summing durations was the right answer while one non-overlapping
    lane existed and is the wrong answer the moment two lanes cover the
    same minute: measured on one hand track plus one ingested rule
    track, the sum reports 100.8 % of the record, and the recovery
    dialog renders that number verbatim. The union is the only figure
    bounded by 100 %.

    Computed in Timedelta space, not seconds, so a nanosecond-resolution
    record's coverage string does not move: with ONE track the merge
    finds nothing to merge and the result is bit-identical to the old
    sum.
    """
    spans = sorted((iv.start, iv.end) for iv in intervals)
    total = pd.Timedelta(0)
    cur_s = None
    cur_e = None
    for s, e in spans:
        if cur_e is None:
            cur_s, cur_e = s, e
        elif s > cur_e:
            total = total + (cur_e - cur_s)
            cur_s, cur_e = s, e
        elif e > cur_e:
            cur_e = e
    if cur_e is not None:
        total = total + (cur_e - cur_s)
    return total
