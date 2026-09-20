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

    Pack M3.0: it is also what Load Session and Recover ask of the
    FILE'S OWN table, BEFORE the merge below runs. A file that names a
    lane its own table lacks is internally inconsistent and is refused
    whatever the live table happens to hold.
    """
    known = set(track_ids(tracks))
    return sorted({iv.track for iv in intervals} - known)


def check_track_table(rows, what=None) -> List[Track]:
    """THE THREE TABLE RULES, IN ONE PLACE. Pack M3.1.

    The table must not be empty, no two rows may share an id, and every
    row must declare at least one class. Those three were written TWICE
    -- `TimeIntervalLabeler._build_track_table` for a caller's `tracks=`
    argument and `io_export.validate_track_table` for a table read off
    disk -- and Pack M3.1 adds a third door (the Manage Lanes box), so
    they are factored here rather than copied again.

    THE TWO VOICES ARE KEPT BYTE FOR BYTE. `what` names the FILE a
    table arrived in; with it the messages read as a refusal to LOAD,
    and without it as a refusal of a caller's argument. The per-row
    order of the checks is the one both callers had, so a table that
    breaks two rules still names the same one first.

    Display-name uniqueness is NOT checked here: it is a rule about
    what the Manage Lanes box may WRITE, not about what a file or a
    driver may hold. See `duplicate_lane_name`.
    """
    if not rows:
        if what is None:
            raise ValueError(
                "'tracks' must contain at least one track; pass "
                "classes=[...] for the single-track default instead.")
        raise ValueError("%s has an empty track table and cannot be "
                         "loaded." % (what,))
    seen = set()
    for row in rows:
        if row.id in seen:
            if what is None:
                raise ValueError(
                    "duplicate track id %r: a track id names an export "
                    "column and a label-map sidecar, so it must be "
                    "unique" % (row.id,))
            raise ValueError(
                "%s has two tracks with the id %r: a track id names an "
                "export column and a label-map sidecar, so it must be "
                "unique." % (what, row.id))
        seen.add(row.id)
        if not row.classes:
            if what is None:
                raise ValueError(
                    "track %r has no classes; every track needs its own "
                    "vocabulary" % (row.id,))
            raise ValueError(
                "%s has a track (%r) with no classes; every track needs "
                "its own vocabulary." % (what, row.id))
    return rows


def lane_display_key(name) -> str:
    """One display name, as the uniqueness rule compares them.

    Whitespace-trimmed and collapsed, lower-cased. `Wake (umbra)` and
    ` wake  (UMBRA) ` are the same name to a pair of eyes and are the
    same name here.
    """
    return " ".join(str(name or "").split()).lower()


def duplicate_lane_name(rows, only_ids=None):
    """The first lane whose DISPLAY NAME collides, or None. Pack M3.1.

    Returns `(lane_id, name)` -- the OFFENDING row, the second one seen.

    `only_ids` limits the answer to the lanes the Manage Lanes box
    actually touched. THAT IS THE WHOLE POINT: display names have never
    been checked for uniqueness, two of the user's own lanes may
    legitimately share one, and a pack that started refusing existing
    files or driver tables would break sessions that open today. The
    rule is "the box may not WRITE a duplicate", not "a duplicate may
    not exist".
    """
    seen = {}
    for row in (rows or []):
        key = lane_display_key(getattr(row, "name", "") or row.id)
        if key in seen:
            if only_ids is None or row.id in set(only_ids):
                return row.id, (getattr(row, "name", "") or row.id)
            if seen[key] in set(only_ids):
                return seen[key], (getattr(row, "name", "") or row.id)
        else:
            seen[key] = row.id
    return None


def lane_id_from_name(name, existing=()) -> str:
    """The id a NEW lane gets from its display name. Pack M3.1.

    Lower-cased; every run of characters outside `[a-z0-9]` becomes ONE
    underscore; leading and trailing underscores trimmed; cut to 32
    characters and trimmed again; empty answers `lane`; a collision with
    `existing` takes `_2`, `_3`, and so on.

    The id is shown and is EDITABLE in the Add Lane box, and is FROZEN
    the moment the lane exists -- it names an export column and a
    sidecar filename, so changing it would strand every interval on it
    (see the module docstring).
    """
    out = []
    prev_us = False
    for ch in str(name or "").lower():
        if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            out.append(ch)
            prev_us = False
        elif not prev_us:
            out.append("_")
            prev_us = True
    base = "".join(out).strip("_")[:_ID_MAX].strip("_")
    if not base:
        base = "lane"
    taken = {str(e) for e in (existing or ())}
    if base not in taken:
        return base
    n = 2
    while True:
        suffix = "_%d" % n
        cand = (base[:_ID_MAX - len(suffix)]).strip("_") + suffix
        if cand not in taken:
            return cand
        n += 1


def renumber_lane_rows(rows) -> List[Track]:
    """`order` becomes the list position, 0..K-1. Pack M3.1.

    The table list is kept in PAINTED order, which is what Pack M3.0's
    DR3 established for the merge: `visible_rows` sorts on `order` with
    the table position only as a tie-break, so a list whose `order`
    values disagree with its positions paints in an order nobody wrote.
    Every edit through the Manage Lanes box ends here.
    """
    for i, t in enumerate(rows or []):
        t.order = i
    return rows


def lane_delete_refusal(rows, lane_id) -> str:
    """Why this lane must not be deleted, in plain words, or "". M3.1.

    Two rules, both ratified: the LAST lane can never be deleted (a
    session with no lane cannot label), and a LOCKED lane must be
    unlocked first (locked says "this came from a file and you must not
    edit it", and deleting it with its intervals is the largest edit
    there is).
    """
    row = find_track(rows, lane_id)
    if row is None:
        return "there is no lane %r" % (lane_id,)
    if len(rows) <= 1:
        return ("'%s' is the only lane -- a session needs at least one, "
                "so it cannot be deleted" % (row.name or row.id,))
    if getattr(row, "locked", False):
        return ("'%s' is locked -- unlock it first"
                % (row.name or row.id,))
    return ""


def lane_hide_refusal(rows, lane_id) -> str:
    """Why this lane must not be hidden, in plain words, or "". M3.1.

    The existing rule, in the box's voice: the LAST VISIBLE lane refuses
    to hide, because a strip with no lane is not a state worth being
    able to reach (`_toggle_active_lane_visible`).
    """
    row = find_track(rows, lane_id)
    if row is None:
        return "there is no lane %r" % (lane_id,)
    if not getattr(row, "visible", True):
        return ""
    others = [t for t in rows
              if t.id != row.id and getattr(t, "visible", True)]
    if not others:
        return ("'%s' is the only lane you can see -- hiding it is "
                "refused" % (row.name or row.id,))
    return ""


def lane_rows_add(rows, name, classes, palette=None, lane_id=None,
                  locked=False, visible=True) -> Track:
    """Append ONE new lane to `rows` and return it. Pack M3.1.

    The id is `lane_id` when the caller gives one -- the Add Lane box
    shows it and lets the user edit it -- and is otherwise made from the
    display name by `lane_id_from_name`. Colours come from `palette`,
    which is the constructor's own `DEFAULT_COLORS`, so a lane made in
    the box is coloured exactly as a lane declared in a driver.

    `order` is the new last position. The caller validates the result;
    nothing is refused here.
    """
    classes = [str(c) for c in (classes or [])]
    tid = lane_id or lane_id_from_name(name, track_ids(rows))
    row = Track(
        id=validate_track_id(tid),
        name=str(name or "") or tid,
        classes=classes,
        class_colors=colors_for(classes, palette),
        locked=bool(locked),
        visible=bool(visible),
        order=len(rows or []),
    )
    rows.append(row)
    return row


def lane_rows_move(rows, lane_id, delta) -> bool:
    """Move one lane `delta` places up (-1) or down (+1). Pack M3.1.

    Returns False at the ends rather than wrapping: the two buttons are
    Move up and Move down, and a lane that jumped from the top to the
    bottom would be a surprise. `order` is renumbered.
    """
    idx = next((i for i, t in enumerate(rows or [])
                if t.id == lane_id), -1)
    if idx < 0:
        return False
    j = idx + int(delta)
    if j < 0 or j >= len(rows):
        return False
    rows[idx], rows[j] = rows[j], rows[idx]
    renumber_lane_rows(rows)
    return True


def lane_rows_delete(rows, intervals, lane_id) -> int:
    """Remove one lane AND every interval on it. Pack M3.1.

    Returns how many intervals went. NEVER leaves strays: an interval
    whose `track` names no row of the table is the orphan state that
    blocks every export, refuses the session on reload and paints
    nowhere (see `stray_tracks`), so the two always move together.

    Both lists are edited IN PLACE, because the live ones are
    `labeler.tracks` and `labeler.intervals` and the gesture snapshot
    restores by slice assignment onto those same objects.
    """
    idx = next((i for i, t in enumerate(rows or [])
                if t.id == lane_id), -1)
    if idx < 0:
        return 0
    del rows[idx]
    renumber_lane_rows(rows)
    keep = [iv for iv in intervals if iv.track != lane_id]
    n = len(intervals) - len(keep)
    intervals[:] = keep
    return n


def merge_track_tables(file_rows, live_rows):
    """The file's lanes, then the lanes only the driver knows. Pack M3.0.

    Load Session and Recover REPLACED the lane table with the file's, so
    a lane the driver declared and the file had never heard of simply
    VANISHED, with its name, its classes and its colours. Measured on a
    three-lane driver: a two-lane file loaded and left two lanes; a v1
    file left ONE, called `default`.

    The rule: the FILE WINS for every lane id it holds, in every field.
    Every live lane whose id the file lacks is KEPT -- empty of the
    file's intervals, because the interval list is the file's -- and
    appended AFTER the file's lanes, in the order the live table holds
    them. `order` is renumbered across the whole result, so the painted
    order really is "the file's lanes, then the kept ones".

    The file's own rows are renumbered in the order they would have been
    PAINTED (by `order`, stable), not in the order they happen to sit in
    the file, so a file whose `order` values are scrambled still paints
    exactly as it does today.

    Returns (merged_rows, n_from_file, n_kept). Nothing here refuses
    anything: the caller validates the result before publishing it.
    """
    out = list(file_rows or [])
    out.sort(key=lambda t: (int(getattr(t, "order", 0) or 0),))
    known = {t.id for t in out}
    kept = [Track.from_dict(t.to_dict()) for t in (live_rows or [])
            if t.id not in known]
    out.extend(kept)
    for i, t in enumerate(out):
        t.order = i
    return out, len(out) - len(kept), len(kept)


def lane_merge_note(n_file, n_kept) -> str:
    """The one sentence a merge that KEPT something owes the user.

    Empty when it kept nothing, which is the point: a session saved from
    the same driver keeps the status line it has today, byte for byte.
    """
    if not n_kept:
        return ""
    return (" -- loaded %d lane%s from the file; kept %d lane%s from the "
            "driver" % (n_file, "" if n_file == 1 else "s",
                        n_kept, "" if n_kept == 1 else "s"))

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
