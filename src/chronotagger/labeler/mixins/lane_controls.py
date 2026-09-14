"""
Lane controls: the ONE setter that moves the active lane, and the four
controls that call it.

Pack M2. Pack M1 gave the model a track table and left `_active_track_id`
as a plain attribute nothing on screen could move. Everything that writes
it now goes through `_set_active_track`, for one measured reason: the class
dropdown does NOT follow the active track by itself. `self.classes` is a
property over the active row and updates the instant the id changes, but
`class_combo["values"]` and `current_class_var` are written once at build
time and rewritten only by Manage Labels -- so a lane switch that does not
re-point both leaves the Add button building an interval whose label the
new lane does not declare, which is exactly the state that makes the whole
session refuse to export.

Four ways in, one setter:

  Ctrl+Down / Ctrl+Up      cycle to the next / previous VISIBLE lane
  the sidebar's Lane list  pick any lane by name (a hidden one unhides)
  a click on a band        activate the lane you clicked (unlocked only)
  the code                 _set_active_track(track_id)

Ctrl+Up / Ctrl+Down were chosen because they are the only bindings that
are free in all five modifier states AND survive the class combobox's
focus trap: `keyboard.py` returns early for any UNMODIFIED key while an
editable widget has focus, and `class_combo` is a `TCombobox`, so after
the user has clicked the class dropdown once, `l`, `h` and `v` are
swallowed and a Control-modified key is not. They are also ELAN's own
lane bindings.
"""

from __future__ import annotations

from typing import List, Optional

from chronotagger.core.lanes import (lane_layout, refuse_if_locked,
                                     set_status, visible_lanes)
from chronotagger.core.tracks import active_id_of, find_track, table_of

# What the sidebar's Lane list appends to a lane that is not painted.
HIDDEN_SUFFIX = " (hidden)"


class LaneControlMixin:
    """The active lane, and the controls that move it."""

    # ---- the setter -----------------------------------------------------

    def _set_active_track(self, track_id, announce: bool = True,
                          repaint: bool = True) -> bool:
        """Make `track_id` the lane every gesture writes to. The ONE writer.

        Returns True when the active lane actually moved. Re-points the
        class dropdown, refreshes the lane controls, repaints and says so
        on the status bar. A track_id the table does not hold is refused
        on the status bar rather than raising: a stale id is a view-state
        bug and must not take the window down.
        """
        table = table_of(self)
        row = find_track(table, track_id)
        if row is None:
            set_status(self, "no lane called %r" % (track_id,))
            return False
        was = active_id_of(self)
        self._active_track_id = row.id
        # The class dropdown and the current class BOTH have to follow, or
        # the next Add builds an off-vocabulary interval on the new lane.
        self._repoint_class_controls()
        self._refresh_lane_controls()
        if announce:
            set_status(self, "Active lane: %s%s"
                       % (row.name or row.id,
                          "  (locked)" if row.locked else ""))
        if repaint and was != row.id:
            self._update_plot()
        return was != row.id

    def _repoint_class_controls(self) -> None:
        """Point `class_combo` and `current_class_var` at the active lane.

        `self.classes` is already the active row's list (a property), so
        this is purely the two WIDGET writes M1 left behind. The selected
        class is kept when the new lane declares it -- switching between
        two lanes that share a vocabulary does not lose the user's place
        -- and otherwise falls back to UNKNOWN if the lane has it, else
        the lane's first class.
        """
        classes = list(self.classes)
        combo = getattr(self, "class_combo", None)
        var = getattr(self, "current_class_var", None)
        if combo is not None:
            try:
                combo["values"] = classes
            except Exception:
                pass
        if var is not None:
            try:
                cur = var.get()
            except Exception:
                cur = None
            if cur not in classes:
                if "UNKNOWN" in classes:
                    cur = "UNKNOWN"
                elif classes:
                    cur = classes[0]
                else:
                    cur = ""
                try:
                    var.set(cur)
                except Exception:
                    pass

    # ---- cycling --------------------------------------------------------

    def _cycle_active_lane(self, delta: int) -> None:
        """Move the active lane `delta` rows down the table (Ctrl+Down = +1).

        Cycles over the VISIBLE lanes only -- an invisible lane is not a
        place a gesture may land -- and wraps. With one visible lane it
        says so instead of pretending to switch.
        """
        lanes = visible_lanes(self)
        if len(lanes) < 2:
            set_status(self,
                       "only one lane is visible -- nothing to switch "
                       "to")
            return
        ids = [t.id for t in lanes]
        cur = active_id_of(self)
        try:
            i = ids.index(cur)
        except ValueError:
            i = 0
            delta = 0
        self._set_active_track(ids[(i + int(delta)) % len(ids)])

    # ---- the sidebar's Lane list ---------------------------------------

    def _lane_choices(self) -> List[str]:
        """The Lane list's values: EVERY lane, in table order.

        Hidden lanes are listed with a suffix rather than dropped,
        because the list is the only way back to one: picking a hidden
        lane unhides it and activates it.
        """
        out = []
        for t in table_of(self):
            out.append((t.name or t.id)
                       + ("" if getattr(t, "visible", True)
                          else HIDDEN_SUFFIX))
        return out

    def _track_id_for_choice(self, text) -> Optional[str]:
        """The lane id behind one entry of `_lane_choices`."""
        text = str(text)
        if text.endswith(HIDDEN_SUFFIX):
            text = text[: -len(HIDDEN_SUFFIX)]
        for t in table_of(self):
            if (t.name or t.id) == text:
                return t.id
        return None

    def _on_lane_combo_change(self, event=None) -> None:
        """The sidebar's Lane list changed -- switch, unhiding if needed."""
        var = getattr(self, "lane_var", None)
        if var is None:
            return
        try:
            text = var.get()
        except Exception:
            return
        tid = self._track_id_for_choice(text)
        if tid is None:
            return
        row = find_track(table_of(self), tid)
        unhid = False
        if row is not None and not getattr(row, "visible", True):
            row.visible = True
            unhid = True
            set_status(self, "lane '%s' is visible again"
                       % (row.name or row.id,))
        moved = self._set_active_track(tid)
        # Pack M2.5: REPAINT ON THE VISIBILITY CHANGE TOO. The setter
        # repaints only when the active id actually moved, so picking the
        # lane that is ALREADY active while it is HIDDEN -- the state Pack
        # M2's SECTION 0e item 10 says an Undo can restore -- flipped the
        # model flag, said "Active lane: ...", and left every pane painting
        # K-1 lanes until something unrelated redrew. Measured by the
        # post-implementation interaction refuter: visible True, K stuck at
        # [2, 2, 2] on all three panes.
        if unhid and not moved:
            self._update_plot()

    def _refresh_lane_controls(self) -> None:
        """Make the sidebar's lane widgets agree with the table.

        Called by the setter, by both toggles and by the strip painter, so
        an ingest, a session load or an undo that changes the table is
        reflected without anyone remembering to refresh. Every widget is
        optional: a labeler built without a sidebar keeps working.
        """
        combo = getattr(self, "lane_combo", None)
        var = getattr(self, "lane_var", None)
        table = table_of(self)
        row = find_track(table, active_id_of(self)) or (table[0] if table
                                                       else None)
        if combo is not None:
            try:
                combo["values"] = self._lane_choices()
            except Exception:
                pass
        if var is not None and row is not None:
            want = (row.name or row.id) + ("" if getattr(row, "visible", True)
                                           else HIDDEN_SUFFIX)
            try:
                if var.get() != want:
                    var.set(want)
            except Exception:
                pass
        vv = getattr(self, "lane_visible_var", None)
        if vv is not None and row is not None:
            try:
                vv.set(bool(getattr(row, "visible", True)))
            except Exception:
                pass
        lv = getattr(self, "lane_locked_var", None)
        if lv is not None and row is not None:
            try:
                lv.set(bool(getattr(row, "locked", False)))
            except Exception:
                pass

    # ---- the two toggles ------------------------------------------------

    def _toggle_active_lane_locked(self, event=None) -> str:
        """Lock / unlock the ACTIVE lane (Ctrl+L, or the sidebar checkbox).

        `locked` is MODEL state -- it says "this lane came from a file and
        you must not edit it", which is a fact about provenance, not about
        the current window -- so this writes the table and persists with
        it. It is NOT a gesture: it pushes no undo entry, exactly like the
        snap and overlay controls beside it.
        """
        row = find_track(table_of(self), active_id_of(self))
        if row is None:
            return "break"
        row.locked = not bool(row.locked)
        self._refresh_lane_controls()
        set_status(self, "lane '%s' is now %s"
                   % (row.name or row.id,
                      "LOCKED" if row.locked else "unlocked"))
        return "break"

    def _toggle_active_lane_visible(self, event=None) -> str:
        """Hide / show the ACTIVE lane (Ctrl+H, or the sidebar checkbox).

        Hiding the active lane MOVES the active lane to the next visible
        one and the status bar says which -- the alternative is an active
        lane that is not painted, which is the exact shape of the bug this
        pack exists to remove (gestures writing to a lane the user cannot
        see). The LAST visible lane refuses to hide: a strip with no lane
        is not a state worth being able to reach.
        """
        table = table_of(self)
        row = find_track(table, active_id_of(self))
        if row is None:
            return "break"
        if getattr(row, "visible", True):
            others = [t for t in visible_lanes(self) if t.id != row.id]
            if not others:
                set_status(self, "lane '%s' is the only visible lane -- "
                                 "hiding it is refused"
                           % (row.name or row.id,))
                self._refresh_lane_controls()
                return "break"
            row.visible = False
            nxt = others[0]
            set_status(self, "lane '%s' hidden -- active lane is now '%s'"
                       % (row.name or row.id, nxt.name or nxt.id))
            self._set_active_track(nxt.id, announce=False, repaint=False)
            self._update_plot()
        else:
            row.visible = True
            set_status(self, "lane '%s' is visible again"
                       % (row.name or row.id,))
            self._refresh_lane_controls()
            self._update_plot()
        return "break"

    # ---- click-to-activate ----------------------------------------------

    def _activate_lane_from_click(self, track_id) -> bool:
        """A click on a band: activate that lane unless it is LOCKED.

        Ratified exactly this way: clicking an interval on an UNLOCKED
        lane selects it AND activates that lane, so the next Add lands
        where the user is looking; clicking one on a LOCKED lane selects
        it (you may read it, and the sidebar shows it) but does NOT move
        the active lane, because a locked lane cannot be drawn on and
        moving there would make the next Add fail for a reason the user
        did not ask for.
        """
        if track_id is None or track_id == active_id_of(self):
            return False
        row = find_track(table_of(self), track_id)
        if row is None:
            return False
        if getattr(row, "locked", False):
            return False
        return self._set_active_track(track_id, announce=False, repaint=False)

    # ---- the two hit paths, resolved through the PAINT -------------------

    def _strip_click_candidates(self, event, pane, click_ts) -> List:
        """The intervals a PICK on the strip may have meant, best first.

        Resolved through `event.ind` against the arrays the painter wrote
        on this pane, which is the only resolution that cannot name a lane
        the painter did not draw: a hidden lane has no face, so `.ind`
        cannot reach it. `.ind` is read HERE, before the caller's own
        `_update_strip()` invalidates the collection it indexes.

        When the pick carries no usable index and there is ONE lane, it
        falls back to scanning that lane -- NOT the whole list. The shipped
        scan had no track filter either, which post-M1 is already a bug
        with one lane on screen: measured, a click selected an interval on
        the `agent` lane while the only band painted at that x belonged to
        `region`.
        """
        from chronotagger.core.tracks import intervals_on
        ivs = list(getattr(pane, "_strip_band_ivs", None) or [])
        ind = list(getattr(event, "ind", None) or [])
        out = []
        for i in ind:
            if 0 <= int(i) < len(ivs):
                out.append(ivs[int(i)])
        if out:
            # Click-to-activate, on the lane the user actually hit.
            self._activate_lane_from_click(out[0].track)
            return out
        k = int(getattr(pane, "_strip_lane_count", 1) or 1)
        if k > 1:
            # Lanes are on screen and the pick named no face: the click was
            # in a gutter or outside every band. Select NOTHING -- the band
            # collection's pickradius(0) says the same thing.
            return []
        ids = list(getattr(pane, "_strip_lane_ids", None) or [])
        tid = ids[0] if ids else active_id_of(self)
        return intervals_on(self.intervals, tid)

    def _strip_press_candidates(self, event, pane, click_ts) -> List:
        """The intervals a PRESS on the strip may have meant.

        The press path gets no `.ind` -- it is a plain button_press_event
        -- so it resolves the lane from the cursor's height through
        `lane_strict`, which returns None in the gutter and above/below
        every band, and then scans THAT LANE only. With one lane it is
        today's scan plus the track filter the shipped code never had.
        """
        from chronotagger.core.lanes import frac_from_event, lane_strict
        from chronotagger.core.tracks import intervals_on
        k = int(getattr(pane, "_strip_lane_count", 1) or 1)
        ids = list(getattr(pane, "_strip_lane_ids", None) or [])
        if k <= 1:
            tid = ids[0] if ids else active_id_of(self)
            return intervals_on(self.intervals, tid)
        ax = getattr(pane, "strip_ax", None)
        frac = frac_from_event(ax, event)
        if frac is None:
            return []
        row = lane_strict(frac, k, getattr(pane, "_strip_pad_frac", None))
        if row is None or row >= len(ids):
            return []
        cands = intervals_on(self.intervals, ids[row])
        if cands:
            for iv in cands:
                if iv.contains(click_ts):
                    self._activate_lane_from_click(iv.track)
                    break
        return cands

    # ---- the guard, as a method for the gesture sites -------------------

    def _refuse_if_locked(self, track_id=None, what: str = "edit") -> bool:
        """True when this edit must not happen. See core/lanes.py.

        A thin method over the free function so the nine gesture sites
        read as one line, and so a subclass can widen the rule.
        """
        return refuse_if_locked(self, track_id=track_id, what=what)

    # ---- what the layout builder asks ----------------------------------

    def _visible_lane_count(self) -> int:
        """How many lanes the strip would paint right now (>= 1)."""
        return max(1, int(lane_layout(self)["k"]))
