"""
lanes.py

LANE GEOMETRY AND THE LANE GUARD: where each track is painted on the
Labels strip, which track a pixel belongs to, and whether a track will
accept an edit at all.

Pack M2. Pack M1 put the tracks in the MODEL and deliberately left the
screen alone: with one lane the strip, the sidebar and every export were
byte-for-byte what they were. This module is the arithmetic that lets
the strip show K lanes at once, and it is the ONE authority both the
painter and both hit tests read, so a click can never land on a lane
the painter did not draw.

THE TWO FUNCTIONS THAT MATTER ARE EXACT INVERSES.
`lane_band(row, k, pad_frac)` says where a row is painted;
`lane_from_frac(frac, k)` says which row a fraction of the strip's
height falls in. A unit test on that pair at k = 1..8 over 1,001
fractions is worth more than any GUI test in this pack, because every
wrong-lane bug the gather found is a disagreement between two copies of
this arithmetic.

`lane_strict` is the third form and it is the reason the two hit paths
agree. `lane_from_frac` names a row for EVERY fraction, including the
gutter between two bands and the empty space above the top band and
below the bottom one. The painter's collection, with
`set_pickradius(0)`, declines to be picked there -- measured -- so a
loose press-path lane would select an interval the pick path says is not
under the cursor. `lane_strict` returns None in exactly the places the
collection declines. The PRESS path only asks it AT TWO LANES AND ABOVE:
at one lane it scans the single lane at any height, which is what
`ded305a` did and what the compat floor keeps.

ROW 0 IS THE TOP. `Track.order`'s own docstring says the table is
ordered top to bottom, so visual row 0 is the first visible row of the
table and it occupies the TOP of the strip. One sign flip in
`lane_band` and `lane_from_frac` would put row 0 at the bottom; they are
the only two places that would have to change.

UNIFORM HEIGHT, ON PURPOSE (M2 ruling). Every lane is 1/K of the strip.
Per-lane height -- a tall human lane above a thin model lane -- is a DAW
idiom worth having, and it is deliberately deferred: with the
arithmetic behind these two functions it is later a change to two
bodies and nothing else.

THE PAD IS 0.1 AT K == 1 AND 0.06 ABOVE IT, and that is not a taste
decision: `lane_band(0, 1, 0.1)` is exactly 0.1 .. 0.9, which are the
constants the pre-M2 painter wrote, so a single-lane strip renders
byte-for-byte as it did before this pack. Above one lane 0.1 would
spend 8.5 px of a 42.4 px lane on gutter (measured at K=4 on a 169.6 px
strip); 0.06 spends 5.1 px.

WHY THE GUARD LIVES HERE TOO. `refuse_if_locked` is the lock half of
the same subject: a lane the user cannot edit is a lane whose geometry
the gestures still have to know about. It is a FREE FUNCTION taking a
HOST, exactly like `table_of` in core/tracks.py and for the same
reason -- a GUI-free mock host that binds a named list of mixin methods
needs no new entry for it -- and it never raises and never opens a
dialog: every refusal goes to the status bar, because a modal on a
gesture is the one thing the campaign's acceptance floor forbids.

DEPENDENCY DIRECTION. This module imports `core.tracks` and nothing
else from chronotagger. `core.tracks` does NOT import this module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .tracks import active_id_of, find_track, table_of

# The gutter, as a fraction of ONE lane's height, taken off the top and
# off the bottom of every band.
PAD_FRAC_SINGLE = 0.1
PAD_FRAC_MULTI = 0.06

# The strip's share of a figure row, per lane count (R5). At K == 1 this
# is 1.0, which is what every layout in the tree emits today, so a
# single-lane session's figure is unchanged.
LABELS_HR_PER_LANE = 0.75


# The gid the strip's band collection carries, and the ONE string both hit
# paths gate on. Spelled here rather than imported from
# labeler.mixins.events.base because core must not import labeler -- and
# pinned equal to TOOL_GID_PREFIX + "strip-bands" by a test, so the two
# copies cannot drift.
STRIP_BAND_GID = "chronotagger:strip-bands"


# The strip's other two tool artists. The FOCUS RING marks the active
# lane's full width; the PREVIEW rectangles are the dashed band a
# selection or a drag paints. Both carry a gid for the same reason the
# band collection does: an artist census must be able to name them
# without guessing at a transform (a Patch's get_transform() is a
# COMPOSITE, so comparing it to ax.transAxes is False even when the patch
# was built with it -- measured).
STRIP_FOCUS_GID = "chronotagger:strip-focus"
STRIP_PREVIEW_GID = "chronotagger:strip-preview"


def strip_focus_gid() -> str:
    """The gid of the active lane's focus ring."""
    return STRIP_FOCUS_GID


def strip_preview_gid() -> str:
    """The gid of every dashed preview rectangle on the strip."""
    return STRIP_PREVIEW_GID


def strip_band_gid() -> str:
    """The gid of the strip's band collection. Gate on this, not identity.

    A handler that gated on artist IDENTITY was measured rejecting every
    pick on the strip: `_on_strip_click` calls `_update_strip()` inside its
    own body, which calls `ax.clear()`, so the collection an earlier
    listener captured is not the collection the next pick carries.
    """
    return STRIP_BAND_GID


def pad_for(k: int) -> float:
    """The gutter fraction to use at `k` lanes.

    0.1 at one lane is LOAD-BEARING: it reproduces the pre-M2 band
    exactly (0.1 .. 0.9) and is what makes the single-lane render
    byte-identical. 0.06 above one lane buys 3.4 px of paint per lane at
    K=4 on a 169.6 px strip.
    """
    return PAD_FRAC_SINGLE if int(k) <= 1 else PAD_FRAC_MULTI


def labels_row_height(k: int) -> float:
    """The Labels row's height ratio for `k` lanes: max(1.0, 0.75 * k).

    Measured on the user's four real pane layouts: at K=4 this clears
    44.5-57.3 px per lane, against 19.1-28.7 px without it, and costs
    22-33 % of every data panel's height. At K=1 it is exactly 1.0 --
    a no-op -- so a single-lane figure is the one he ships today.

    ONE AUTHORITY, TWO CALLERS: the canvas builder uses it when a
    layout_spec carries no explicit height_ratios, and a driver that
    writes its own height_ratios calls it for the Labels row. An
    explicit height_ratios in a layout_spec always wins.
    """
    k = int(k)
    return max(1.0, LABELS_HR_PER_LANE * k) if k > 0 else 1.0


def lane_band(row: int, k: int,
              pad_frac: Optional[float] = None) -> Tuple[float, float]:
    """(lo, hi) of visual row `row` of `k`, in AXES fraction. Row 0 = top.

    The strip's ylim is pinned to (0, 1) by the painter, so these are
    also data coordinates -- which is why the `ydata` shortcut in the
    old hit test worked at all. Nothing in this module relies on that;
    the hit tests use the transAxes inverse.
    """
    k = int(k)
    if k < 1:
        raise ValueError("lane_band needs at least one lane, got k=%r" % (k,))
    row = int(row)
    if row < 0 or row >= k:
        raise ValueError("lane row %d is outside 0..%d" % (row, k - 1))
    if pad_frac is None:
        pad_frac = pad_for(k)
    h = 1.0 / k
    top = 1.0 - row * h
    return (top - h + pad_frac * h, top - pad_frac * h)


def lane_from_frac(frac: float, k: int) -> int:
    """The visual row a height fraction falls in -- LOOSE, never None.

    Every fraction names a row, gutters included. Use this only where
    "nearest lane" is the right answer; the hit tests use
    `lane_strict`.
    """
    k = int(k)
    if k < 1:
        raise ValueError("lane_from_frac needs at least one lane, got k=%r"
                         % (k,))
    row = int((1.0 - float(frac)) * k)
    if row < 0:
        row = 0
    if row > k - 1:
        row = k - 1
    return row


def lane_strict(frac: float, k: int,
                pad_frac: Optional[float] = None) -> Optional[int]:
    """The visual row a height fraction is PAINTED in, or None.

    None means "the cursor is not on a band": the gutter between two
    lanes, or above the top band, or below the bottom one. The band
    collection is built with `set_pickradius(0)`, which declines a pick
    in exactly those places, so this is the form that makes the pick
    path and the press path agree everywhere.
    """
    k = int(k)
    if k < 1:
        return None
    if pad_frac is None:
        pad_frac = pad_for(k)
    row = lane_from_frac(frac, k)
    lo, hi = lane_band(row, k, pad_frac)
    f = float(frac)
    if f < lo or f > hi:
        return None
    return row


def preview_band(row: int, k: int,
                 pad_frac: Optional[float] = None) -> Tuple[float, float]:
    """(y, height) of the dashed preview rectangle over row `row`.

    The preview BRACKETS the band by half the gutter rather than
    equalling it -- shipped band 0.1 .. 0.9, shipped preview
    0.05 .. 0.95 -- and that is the third of the three rules the
    single-lane byte identity needs. It was also the one condition that
    failed before the gather found it, so it is stated as arithmetic
    here and pinned in the suite: at k == 1 this returns exactly
    (0.05, 0.9).
    """
    k = int(k)
    if pad_frac is None:
        pad_frac = pad_for(k)
    lo, hi = lane_band(row, k, pad_frac)
    half = pad_frac * (1.0 / k) / 2.0
    lo2 = lo - half
    hi2 = hi + half
    return (lo2, hi2 - lo2)


def frac_from_event(ax, event) -> Optional[float]:
    """The event's height as a fraction of `ax`, via the transAxes INVERSE.

    `event.ydata` would do while the strip's ylim is (0, 1), and the
    painter pins it there -- but the inverse is the same one line and
    cannot be broken by a future ylim change, and a strip axis is the
    one axis in this tool whose y means "which lane", not "a value".
    Returns None when the event carries no pixel position.
    """
    if ax is None or event is None:
        return None
    x = getattr(event, "x", None)
    y = getattr(event, "y", None)
    if x is None or y is None:
        return None
    try:
        return float(ax.transAxes.inverted().transform((float(x),
                                                        float(y)))[1])
    except Exception:
        return None


def visible_lanes(host) -> List[Any]:
    """The tracks the strip paints, top to bottom.

    Sorted by `order` with the table position as the tie-break, so a
    table whose orders are all 0 (a hand-written driver table) paints in
    the order the driver wrote it. A track with `visible=False` is not
    in this list: it is not painted, and because it has no face on the
    collection it is unreachable by BOTH hit tests -- gesture targeting
    falls out of the painter for free rather than needing a second
    filter.
    """
    rows = [t for t in table_of(host) if getattr(t, "visible", True)]
    return sorted(rows, key=lambda t: (int(getattr(t, "order", 0) or 0),))


def lane_layout(host) -> Dict[str, Any]:
    """Everything the painter, the hit tests and the previews agree on.

    Keys:
      lanes       the visible Track rows, top to bottom
      row_of      {track id: visual row}
      k           len(lanes) -- MAY BE 0 when every lane is hidden
      pad         the gutter fraction for this k
      active_id   the active track's id (whether or not it is visible)
      active_row  the active track's visual row, or None when the active
                  lane is hidden (or there are no lanes at all)

    ONE call, five readers. The gather measured what happens when two
    readers compute this separately: paint pane 2 with a lane hidden
    while pane 0 still shows three, and a hit test that kept its own
    lane count mis-lanes every click on the other pane.
    """
    lanes = visible_lanes(host)
    row_of = {t.id: i for i, t in enumerate(lanes)}
    k = len(lanes)
    active_id = active_id_of(host)
    active_row = row_of.get(active_id)
    if (active_row is None and lanes
            and find_track(table_of(host), active_id) is None):
        # The active id names no row of the table at all -- which happens
        # after an undo removes the lane an ingest created, because
        # `_active_track_id` is not part of the table snapshot. Fall back to
        # the FIRST VISIBLE lane, which is exactly what active_track_of
        # already does for `self.classes`, so the screen and the vocabulary
        # cannot disagree. A HIDDEN active lane is a different case and
        # keeps active_row None on purpose: the lane still exists, nothing
        # is painted for it, and the caller decides what that means.
        active_id = lanes[0].id
        active_row = 0
    return {
        "lanes": lanes,
        "row_of": row_of,
        "k": k,
        "pad": pad_for(k if k else 1),
        "active_id": active_id,
        "active_row": active_row,
    }


def active_band(host) -> Tuple[float, float]:
    """(y, height) of the dashed preview rectangle on the ACTIVE lane.

    Falls back to the whole strip -- (0.05, 0.9), the pre-M2 constants
    -- when there is no visible lane or the active lane is hidden.
    Callers use this instead of reaching for the layout, because the
    preview pool in events/strip.py lives on the PANE and is built
    before any paint has happened.
    """
    lay = lane_layout(host)
    k = lay["k"]
    row = lay["active_row"]
    if not k or row is None:
        return (0.05, 0.9)
    return preview_band(row, k, lay["pad"])


def set_status(host, message: str) -> bool:
    """Write one line to the status bar, if the host has one.

    Returns True when the message landed. Every refusal in this module
    goes through here: a labeler built without a sidebar (the suite's
    GUI-free hosts) must refuse just as hard and just as silently.
    """
    var = getattr(host, "status_var", None)
    if var is None:
        return False
    try:
        var.set(str(message))
        return True
    except Exception:
        return False


def track_display_name(host, track_id) -> str:
    """The lane's display NAME for a message, or its id as a fallback."""
    row = find_track(table_of(host), track_id)
    if row is None:
        return str(track_id)
    return row.name or row.id


def is_locked(host, track_id=None) -> bool:
    """True when the named track (default: the ACTIVE one) refuses edits."""
    if track_id is None:
        track_id = active_id_of(host)
    row = find_track(table_of(host), track_id)
    return bool(row is not None and getattr(row, "locked", False))


def refuse_if_locked(host, track_id=None, what: str = "edit") -> bool:
    """True (and the status bar says why) when this edit must not happen.

    `what` completes the sentence "<lane> is locked (press Ctrl+L to
    unlock) -- <what> refused", so pass a noun phrase: "clear range",
    "delete", "relabel", "drag", "add", "fill gaps", "edit the label
    schema".

    NO MODAL, EVER. `crud.py` already carries five `showwarning` calls
    on the label hot path and the campaign's acceptance floor is zero
    unexpected modals, so a refusal that is CORRECT must not be the
    thing that stops the user's hands. Measured across all nine guarded
    paths: 7 distinct status lines, 0 dialogs.

    The ingest (`add_track_from_column`) is deliberately NOT guarded: it
    is the writer that CREATES a locked lane, not a user gesture on
    one, and a lock that blocked it would make the flag impossible to
    use for what it is for.
    """
    if track_id is None:
        track_id = active_id_of(host)
    if not is_locked(host, track_id):
        return False
    set_status(host, "track '%s' is locked (press Ctrl+L to unlock) -- %s "
                     "refused" % (track_display_name(host, track_id), what))
    return True
