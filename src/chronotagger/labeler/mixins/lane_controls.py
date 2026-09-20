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

from contextlib import contextmanager
from typing import List, Optional

from chronotagger.core.lanes import (lane_layout, refuse_if_locked,
                                     set_status, visible_lanes)
from chronotagger.core.tracks import active_id_of, find_track, table_of

# What the sidebar's Lane list appends to a lane that is not painted.
HIDDEN_SUFFIX = " (hidden)"


class LaneControlMixin:
    """The active lane, and the controls that move it."""

    # ---- one lane-table write = one undo step ---------------------------

    @contextmanager
    def _lane_gesture(self, name: str):
        """Make ONE write to the lane table ONE undo entry. Pack M3.0.

        Ctrl+L, Ctrl+H and un-hiding from the sidebar's Lane list wrote
        `row.locked` / `row.visible` BARE, while every GestureCommand
        restores the WHOLE table (core/commands.py). Measured on a
        three-lane driver: add an interval, lock a lane, press Ctrl+Z
        ONCE -- the interval went AND the lock silently flipped back,
        because the undo restored the table as it stood before the
        interval was added. One press, two things, one of them never
        announced.

        `_gesture` already carries the table in its snapshot, so the fix
        is the wrapper and nothing else: no new command class, no second
        stack, no new state.

        PACK M3.1 AMENDS PACK M3.0's DR2: A LANE-LIST WRITE NOW MARKS
        THE SESSION MODIFIED. M3.0 read `modified` before the block and
        put it back after, because a bare write had never marked it and
        no one had ruled that it should. The consequence was real and is
        now closed: lock a lane, close the window, and nothing asked --
        the lock was simply lost. A lane's lock, its visibility, its
        name and its very existence all persist in the session file, so
        an unsaved lane edit is unsaved work. `_gesture` marks the flag
        at the end of every block and the flag is now left where it
        leaves it.

        A host without the gesture machinery -- the GUI-free hosts in
        tests/ bind a NAMED LIST of mixin methods -- writes bare, as it
        did before.
        """
        _g = getattr(self, "_gesture", None)
        if not callable(_g):
            yield
            return
        with _g(name):
            yield

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
        # Pack M2.6: A LANE SWITCH DROPS A STAGED RULE. The rule preview
        # stages its spans in `_commit_spans` and paints them yellow in
        # `current_spans`, and neither was bound to a lane -- so staging a
        # rule on Region, switching to Wake and pressing Add put
        # Region-carved geometry on Wake, with the class auto-substituted
        # and no prompt (measured, probe_s3_refute A3-A4). Those spans
        # were computed against the lane you were ON, including its
        # overlaps; one lane down they do not mean the same thing. It
        # says so only when something really was staged, and it says it
        # in the SAME sentence as the lane change, because the announce
        # below would otherwise write straight over it.
        dropped = False
        if was != row.id and (getattr(self, "_commit_spans", None)
                              or getattr(self, "current_spans", None)):
            self._clear_preview_state()
            dropped = True
        # The class dropdown and the current class BOTH have to follow, or
        # the next Add builds an off-vocabulary interval on the new lane.
        self._repoint_class_controls()
        self._refresh_lane_controls()
        # Pack M2.7: A LANE SWITCH THAT PUSHES THE SELECTION OUT OF THE
        # LIST SAYS BOTH THINGS IN ONE LINE. The switch changes which lane
        # the list shows, so the selected interval can leave it -- a fact
        # the user has to be told, and one the repaint below would
        # otherwise announce as its own line straight over "Active lane:
        # ...". That is the two-writes-one-line defect Pack M2.6 removed
        # one door back, arriving through a different door. The claim
        # marks the fact as said AND hands back the clause.
        _hidden = ""
        _claim = getattr(self, "_claim_hidden_selection_line", None)
        if callable(_claim):
            _hidden = _claim()
        if announce:
            # Pack M3.0: THE LOCKED CLAUSE SAYS HOW TO UNLOCK. A click on
            # a locked lane's NAME or empty row now switches to it, and
            # "  (locked)" told the user he had landed somewhere he
            # cannot draw on without telling him what to do about it. One
            # sentence, one writer: every door into a locked lane --
            # Ctrl+Up / Ctrl+Down, the sidebar's Lane list, a click on the
            # strip -- says the same thing.
            set_status(self, "Active lane: %s%s%s%s"
                       % (row.name or row.id,
                          " -- lane locked (Ctrl+L to unlock)"
                          if row.locked else "",
                          " -- rule preview cleared (it belonged to the "
                          "lane you left)" if dropped else "",
                          _hidden))
        elif dropped:
            set_status(self, "rule preview cleared -- it belonged to the "
                             "lane you left")
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

    # ---- the reconcile ---------------------------------------------------

    def _reconcile_active_track(self) -> bool:
        """Re-point a STALE active lane at the row the user can see.

        Pack M2.6, the second half of the stale-id fix. EDIT 539 makes
        every READER agree about which row a stale id means; this makes
        the MODEL agree too, so the id stops being stale at the moment it
        becomes stale instead of being resolved again on every call.

        `_active_track_id` is VIEW state and does not ride in the gesture
        snapshot, so the undo of an ingest -- which removes the lane the
        ingest created, inside one gesture -- leaves the id naming a row
        the table no longer holds. Called after every undo and redo. The
        other two table-snapshot restores, `_load_session` and
        `_apply_recovered_autosave`, already re-point `_active_track_id`
        themselves before they publish, so there is nothing stale there
        for this to find.

        Returns True when it re-pointed, and then the status bar carries
        the sentence -- the user pressed one key and the lane he was
        working on went away, which he has to be told.
        """
        from chronotagger.core.tracks import resolve_active_row
        old = getattr(self, "_active_track_id", None)
        if not old:
            self._active_track_id = resolve_active_row(self).id
            return False
        if find_track(table_of(self), old) is not None:
            return False
        row = resolve_active_row(self)
        self._active_track_id = row.id
        self._repoint_class_controls()
        self._refresh_lane_controls()
        set_status(self, "lane '%s' no longer exists -- active lane is "
                         "now '%s'" % (old, row.name or row.id))
        return True

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
        """The lane id behind one entry of `_lane_choices`, BY TEXT.

        Pack M3.0: this is the FALLBACK, not the resolution. Display
        names are free text and nothing forbids two lanes from sharing
        one, and then this can only ever answer the FIRST of them --
        measured, picking the second row activated the first. It is kept
        for the callers that set `lane_var` by hand and never touch the
        widget.
        """
        text = str(text)
        if text.endswith(HIDDEN_SUFFIX):
            text = text[: -len(HIDDEN_SUFFIX)]
        for t in table_of(self):
            if (t.name or t.id) == text:
                return t.id
        return None

    def _track_id_for_index(self, index) -> Optional[str]:
        """The lane id at POSITION `index` of `_lane_choices`. Pack M3.0.

        `_lane_choices` walks `table_of(self)` in table order and emits
        exactly one entry per row, so entry i IS row i. Out of range --
        and the -1 a combobox reports when its text matches none of its
        values -- is None, and the caller falls back to the text.
        """
        try:
            i = int(index)
        except (TypeError, ValueError):
            return None
        table = table_of(self)
        if 0 <= i < len(table):
            return table[i].id
        return None

    def _lane_id_for_pick(self, text) -> Optional[str]:
        """The lane the sidebar's Lane list is pointing at. Pack M3.0.

        BY POSITION FIRST. The widget knows which ROW was picked, and
        that is the only answer that survives two lanes sharing one
        display name. The text lookup is the fallback, for a caller that
        wrote `lane_var` and never touched the widget.
        """
        combo = getattr(self, "lane_combo", None)
        if combo is not None:
            try:
                tid = self._track_id_for_index(combo.current())
            except Exception:
                tid = None
            if tid is not None:
                return tid
        return self._track_id_for_choice(text)

    def _on_lane_combo_change(self, event=None) -> None:
        """The sidebar's Lane list changed -- switch, unhiding if needed."""
        var = getattr(self, "lane_var", None)
        if var is None:
            return
        try:
            text = var.get()
        except Exception:
            return
        # Pack M3.0: BY POSITION, NOT BY NAME -- see `_lane_id_for_pick`.
        tid = self._lane_id_for_pick(text)
        if tid is None:
            return
        row = find_track(table_of(self), tid)
        unhid = False
        if row is not None and not getattr(row, "visible", True):
            # Pack M3.0: ONE UNDO STEP. See `_lane_gesture` -- this write
            # rode in every later snapshot and was silently reverted by
            # the undo of an unrelated edit.
            with self._lane_gesture("show lane %s"
                                    % (row.name or row.id,)):
                row.visible = True
            unhid = True
        # Pack M2.6: ONE SENTENCE, NOT TWO. Unhiding a lane from this
        # list wrote "lane 'Wake (umbra)' is visible again" and then the
        # setter wrote "Active lane: Wake (umbra)" straight over it, so
        # the thing the user had just done never reached the bar. The
        # setter is told not to announce, and the one combined line is
        # written after it.
        # Whether a staged rule is about to be dropped has to be read
        # BEFORE the setter runs, because the setter is the thing that
        # drops it -- and when `unhid` silenced the setter's own
        # announcement, the setter's note about the drop is the line this
        # combined sentence would otherwise write over.
        _had_staging = bool(getattr(self, "_commit_spans", None)
                            or getattr(self, "current_spans", None))
        moved = self._set_active_track(tid, announce=not unhid)
        if unhid and row is not None:
            set_status(self, "lane '%s' is visible again -- active lane%s"
                       % (row.name or row.id,
                          " -- rule preview cleared (it belonged to the "
                          "lane you left)"
                          if (_had_staging and moved) else ""))
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
        # Pack M3.0: AND THE SELECTED INDEX FOLLOWS THE ACTIVE LANE. The
        # widget's own index is what `_lane_id_for_pick` reads back, and
        # writing the TEXT cannot move it when two lanes share a display
        # name: `var.set` is skipped as a no-op and the index stays on
        # whichever row it was already on, so the next pick answered the
        # wrong lane.
        if combo is not None and row is not None:
            _i = next((i for i, t in enumerate(table) if t is row), -1)
            if _i >= 0:
                try:
                    combo.current(_i)
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
        it.

        Pack M3.0: and the write is ONE UNDO STEP. It used to be a bare
        write, which the whole-table snapshot in every later gesture then
        reverted behind the user's back: lock a lane, undo an EARLIER
        edit, and the lock flipped off with nothing said. The entry is
        named for the act and the lane, so the bar reads
        `Undo: lock lane Agent (C-MMAE)`. The session's `modified` flag
        is left exactly as it was found, which is what a bare write did.
        """
        row = find_track(table_of(self), active_id_of(self))
        if row is None:
            return "break"
        _want = not bool(row.locked)
        with self._lane_gesture("%s lane %s"
                                % ("lock" if _want else "unlock",
                                   row.name or row.id)):
            row.locked = _want
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

        Pack M3.0: the WRITE is ONE UNDO STEP (`_lane_gesture`); the
        REFUSAL writes nothing and therefore pushes nothing. Undoing a
        hide brings the lane back VISIBLE and does NOT move the active
        lane back -- the active lane is view state and rides in no
        snapshot -- so the bar says `Undo: hide lane Region (human)` and
        means exactly that and nothing more.
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
            with self._lane_gesture("hide lane %s"
                                    % (row.name or row.id,)):
                row.visible = False
            nxt = others[0]
            set_status(self, "lane '%s' hidden -- active lane is now '%s'"
                       % (row.name or row.id, nxt.name or nxt.id))
            self._set_active_track(nxt.id, announce=False, repaint=False)
            self._update_plot()
        else:
            with self._lane_gesture("show lane %s"
                                    % (row.name or row.id,)):
                row.visible = True
            set_status(self, "lane '%s' is visible again"
                       % (row.name or row.id,))
            self._refresh_lane_controls()
            self._update_plot()
        return "break"

    # ---- the lane-list edit functions (Pack M3.1) ------------------------
    #
    # There were NONE before this pack: the table could be locked, hidden
    # and re-pointed, and that was all. A lane could only be DECLARED in a
    # driver. These six are the model half of the Manage Lanes box -- each
    # one write, each ONE undo step -- and every rule they enforce is a
    # free function in core/tracks.py, so the box's STAGED copies and this
    # LIVE table can never disagree about what is allowed.

    def _lane_add(self, name, classes, lane_id=None, locked=False,
                  visible=True):
        """Add a lane at the END of the table. Returns the row, or None.

        Colours come from the constructor's own palette, so a lane made
        here is coloured exactly as a lane a driver declares. The id is
        made from the NAME unless the caller gives one, and is frozen
        from this moment: it names an export column and a sidecar file.
        """
        from chronotagger.core.tracks import (check_track_table,
                                              lane_rows_add)
        rows = table_of(self)
        if rows is not getattr(self, "tracks", None):
            return None
        try:
            with self._lane_gesture("add lane %s" % (name or lane_id,)):
                row = lane_rows_add(
                    rows, name, classes,
                    palette=getattr(self, "DEFAULT_COLORS", None),
                    lane_id=lane_id, locked=locked, visible=visible)
                check_track_table(rows)
        except ValueError as exc:
            set_status(self, "cannot add that lane: %s" % (exc,))
            return None
        self._refresh_lane_controls()
        return row

    def _lane_rename(self, track_id, name) -> bool:
        """Change one lane's DISPLAY NAME. Never its id.

        The id is what every interval on the lane holds and what the
        export column is called, so a rename is history-preserving by
        construction: it touches one field of one row.
        """
        row = find_track(table_of(self), track_id)
        if row is None:
            return False
        new = " ".join(str(name or "").split())
        if not new or new == row.name:
            return False
        with self._lane_gesture("rename lane %s" % (row.name or row.id,)):
            row.name = new
        self._refresh_lane_controls()
        return True

    def _lane_move(self, track_id, delta) -> bool:
        """Move one lane up (-1) or down (+1); `order` is renumbered."""
        from chronotagger.core.tracks import lane_rows_move
        rows = table_of(self)
        if rows is not getattr(self, "tracks", None):
            return False
        row = find_track(rows, track_id)
        if row is None:
            return False
        moved = False
        with self._lane_gesture("move lane %s" % (row.name or row.id,)):
            moved = lane_rows_move(rows, track_id, delta)
        if moved:
            self._refresh_lane_controls()
        return moved

    def _lane_set_locked(self, track_id, locked) -> bool:
        """Lock or unlock ANY lane, not only the active one."""
        row = find_track(table_of(self), track_id)
        if row is None or bool(row.locked) == bool(locked):
            return False
        with self._lane_gesture("%s lane %s"
                                % ("lock" if locked else "unlock",
                                   row.name or row.id)):
            row.locked = bool(locked)
        self._refresh_lane_controls()
        return True

    def _lane_set_visible(self, track_id, visible) -> bool:
        """Show or hide ANY lane. The LAST VISIBLE one refuses to hide."""
        from chronotagger.core.tracks import lane_hide_refusal
        rows = table_of(self)
        row = find_track(rows, track_id)
        if row is None or bool(row.visible) == bool(visible):
            return False
        if not visible:
            why = lane_hide_refusal(rows, track_id)
            if why:
                set_status(self, "lane %s" % (why,))
                return False
        with self._lane_gesture("%s lane %s"
                                % ("show" if visible else "hide",
                                   row.name or row.id)):
            row.visible = bool(visible)
        self._refresh_lane_controls()
        return True

    def _lane_delete(self, track_id) -> int:
        """Delete a lane AND every interval on it. Returns how many went.

        -1 means REFUSED, and the bar says why: the last lane can never
        be deleted and a locked lane must be unlocked first.
        """
        from chronotagger.core.tracks import (lane_delete_refusal,
                                              lane_rows_delete)
        rows = table_of(self)
        if rows is not getattr(self, "tracks", None):
            return -1
        why = lane_delete_refusal(rows, track_id)
        if why:
            set_status(self, "cannot delete that lane: %s" % (why,))
            return -1
        n = 0
        with self._lane_gesture("delete lane %s"
                                % (find_track(rows, track_id).name
                                   or track_id,)):
            n = lane_rows_delete(rows, self.intervals, track_id)
        self._reconcile_active_track()
        self._refresh_lane_controls()
        return n

    # ---- click-to-activate: the BAND path --------------------------------

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

    # ---- click a lane to make it active (Pack M3.0) ----------------------

    def _activate_lane_from_strip(self, track_id) -> bool:
        """A click on a lane's NAME, or on an EMPTY part of its row.

        Unlike `_activate_lane_from_click` -- the BAND path, which keeps
        refusing locked lanes because a click on a band is first of all a
        SELECTION, and moving the active lane there would make the next
        Add fail for a reason the user did not ask for -- this door DOES
        switch to a locked lane. It is a lane switch and nothing else,
        the bar says `lane locked (Ctrl+L to unlock)`, and the
        alternative is a lane name you can click that does nothing.

        A click on the lane that is ALREADY active does nothing at all
        and writes no status line.
        """
        if track_id is None or track_id == active_id_of(self):
            return False
        if find_track(table_of(self), track_id) is None:
            return False
        return self._set_active_track(track_id)

    def _strip_name_hit(self, event, pane) -> Optional[str]:
        """The lane whose NAME this canvas click landed on, or None.

        The names are the strip's y TICK LABELS (plotting.py), drawn
        OUTSIDE the axes: `event.inaxes` is None over them, so the press
        handler's axes test declines before anything else can look. There
        is no artist to pick either -- the band collection is the only
        picker in the tree -- so the test is the click's PIXEL against
        each label's window extent, asked for on demand because
        `_update_strip` calls `ax.clear()` and throws the artists away on
        every paint.

        `get_window_extent` needs a renderer, which exists only after a
        draw; before the first one it raises and this answers None. The
        labels belong to ONE pane's axes and the caller has already gated
        on the active pane, so a click on another pane's canvas cannot
        reach a lane here; neither can a click on another axes' tick
        labels, which sit at other heights, nor one inside any axes at
        all. At one lane there are no labels.
        """
        ax = getattr(pane, "strip_ax", None)
        if ax is None or getattr(event, "inaxes", None) is not None:
            return None
        x = getattr(event, "x", None)
        y = getattr(event, "y", None)
        if x is None or y is None:
            return None
        if int(getattr(pane, "_strip_lane_count", 1) or 1) <= 1:
            return None
        ids = list(getattr(pane, "_strip_lane_ids", None) or [])
        if not ids:
            return None
        try:
            labels = list(ax.get_yticklabels())
        except Exception:
            return None
        # A few pixels of slack: the extent is tight around a 7-point
        # string and a name is a target for a mouse, not for a compiler.
        pad = 3.0
        for i, lab in enumerate(labels):
            if i >= len(ids):
                break
            try:
                bb = lab.get_window_extent()
            except Exception:
                return None
            if (bb.x0 - pad <= x <= bb.x1 + pad
                    and bb.y0 - pad <= y <= bb.y1 + pad):
                return ids[i]
        return None

    def _activate_lane_from_empty_row(self, event, pane, click_ts) -> bool:
        """A click on an EMPTY part of a lane's row makes it active.

        EMPTY means inside the lane's band with no interval OF THAT LANE
        under the cursor. The gutters between lanes, above the top band
        and below the bottom one are not a lane and do nothing, which is
        what `lane_strict` already answers for the press path's own
        candidate scan.

        The interval test is what keeps this off every other gesture: a
        press over a band is a selection or a drag and is answered before
        this is reached, and a press inside the SELECTED interval never
        reaches here at all, because the drag hit test speaks first.
        """
        k = int(getattr(pane, "_strip_lane_count", 1) or 1)
        if k <= 1:
            return False
        ids = list(getattr(pane, "_strip_lane_ids", None) or [])
        if not ids:
            return False
        from chronotagger.core.lanes import frac_from_event, lane_strict
        from chronotagger.core.tracks import intervals_on
        ax = getattr(pane, "strip_ax", None)
        frac = frac_from_event(ax, event)
        if frac is None:
            return False
        row = lane_strict(frac, k, getattr(pane, "_strip_pad_frac", None))
        if row is None or row >= len(ids):
            return False
        tid = ids[row]
        if click_ts is not None:
            for iv in intervals_on(self.intervals, tid):
                if iv.contains(click_ts):
                    return False
        return self._activate_lane_from_strip(tid)

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

    # ---- the Manage Lanes box (Pack M3.1) -------------------------------

    def _open_manage_lanes(self, event=None) -> str:
        """Open the Manage Lanes box, wait for it, apply its answer.

        NO LOCK GUARD, and that is deliberate: Manage Labels refuses to
        open on a locked lane because everything in it would be
        discarded, and this box is where you UNLOCK. A box you cannot
        open because the thing it exists to change is set would be a
        door locked from the inside.

        Nothing happens to the app until OK. `dlg.result is None` is
        Cancel, and Cancel discards staged deletes and staged imports
        with everything else.
        """
        from chronotagger.core.tracks import intervals_on
        root = getattr(self, "root", None)
        if root is None:
            return "break"
        from ..dialogs.manage_lanes import ManageLanesDialog
        rows = table_of(self)
        counts = {t.id: len(intervals_on(self.intervals, t.id))
                  for t in rows}
        # Pack M3.1, PART E: the column feature reaches the box as two
        # CALLABLES and never as the app object. With PART E cut they
        # are both None and the 'From column...' button is not built.
        dlg = ManageLanesDialog(
            parent=root,
            lanes=rows,
            interval_counts=counts,
            columns_fn=getattr(self, "_lane_column_choices", None),
            preview_fn=getattr(self, "_lane_column_preview", None),
        )
        root.wait_window(dlg)
        if dlg.result is None:
            return "break"
        self._apply_manage_lanes_result(dlg.result)
        return "break"

    def _apply_manage_lanes_result(self, result) -> bool:
        """Publish the box's staged end state, in ONE undo step.

        THE WHOLE OK IS ONE GESTURE -- adds, renames, reorders, lock and
        visibility changes, deletes WITH their intervals and column
        imports WITH theirs -- so ONE Ctrl+Z puts every one of them
        back, a deleted lane with its intervals included, and one Ctrl+Y
        re-applies the lot. `GestureCommand` already snapshots the whole
        lane table beside the interval list, so this needs no new
        command class and no second stack.

        ALL OR NOTHING. The end state is validated BEFORE anything is
        published; on a refusal nothing changes at all and the bar says
        why. An OK that changes nothing pushes nothing and says nothing
        new.
        """
        from chronotagger.core.tracks import (Track, check_track_table,
                                              colors_for,
                                              duplicate_lane_name,
                                              renumber_lane_rows)
        if getattr(self, "tracks", None) is None:
            return False
        try:
            rows = [Track.from_dict(d) for d in (result.lanes or [])]
        except (ValueError, TypeError) as exc:
            set_status(self, "Manage Lanes changed nothing: %s" % (exc,))
            return False
        renumber_lane_rows(rows)
        # THE COLOURS. The box builds a bare `Track` for a lane it adds
        # and for a staged column import, and `Track.from_dict` copies
        # exactly what it is given -- so a lane made in the box arrived
        # here with an EMPTY class_colors and every band on it painted
        # the #cccccc fallback. `_build_track_table` fills the same gap
        # for a caller's `tracks=` argument and `add_track_from_column`
        # fills it for a driver's import; this is that one rule at this
        # door, so a lane made in the box is coloured exactly as a lane
        # a driver declares.
        for _row in rows:
            if not _row.class_colors:
                _row.class_colors = colors_for(
                    _row.classes, getattr(self, "DEFAULT_COLORS", None))
        try:
            check_track_table(rows)
        except ValueError as exc:
            set_status(self, "Manage Lanes changed nothing: %s" % (exc,))
            return False
        # The display-name rule applies to what the BOX WROTE, never to
        # what a file or a driver already held: two lanes have always
        # been allowed to share a name and sessions that do must keep
        # opening.
        _touched = (set(result.added or ())
                    | set((result.renamed or {}).keys())
                    | {d.get("id") for d in (result.imports or [])})
        _dup = duplicate_lane_name(rows, _touched)
        if _dup is not None:
            set_status(self, "Manage Lanes changed nothing: two lanes "
                             "are called '%s'" % (_dup[1],))
            return False
        # THE LAST VISIBLE LANE, CHECKED ON THE END STATE. Show/Hide
        # refuses to hide the last visible lane ONE PRESS AT A TIME, but
        # a hide and a DELETE are two presses the box allows separately
        # that together leave every lane hidden: measured -- hide one of
        # two lanes, delete the other, press OK, and the session is left
        # with one invisible lane, a blank strip and a gesture writing
        # to a lane the user cannot see.
        if rows and not any(getattr(t, "visible", True) for t in rows):
            set_status(self, "Manage Lanes changed nothing: that would "
                             "leave no lane visible")
            return False
        if result.is_empty():
            return False

        _gone = set(result.deleted or ())
        _was_active = active_id_of(self)
        _n_deleted = 0
        _n_imported = 0
        # `_gesture`, NOT `_lane_gesture`: this OK is a structural change
        # to the session and marks it modified the way every other
        # gesture does. `_lane_gesture` exists for the THREE bare writes
        # Pack M3.0's DR2 was about -- Ctrl+L, Ctrl+H and the un-hide --
        # and whichever way that ruling finally goes must not decide
        # whether adding and deleting lanes counts as unsaved work.
        # ALL OR NOTHING, INCLUDING THE STAGED IMPORTS. `_gesture` is
        # transactional and rolls the model back on its own, but the
        # exception used to travel on out of a Tk button callback: a
        # traceback on stderr, nothing on the bar, and the user left
        # looking at the line that was there before.
        try:
            with self._gesture("manage lanes"):
                if _gone:
                    _keep = [iv for iv in self.intervals
                             if iv.track not in _gone]
                    _n_deleted = len(self.intervals) - len(_keep)
                    self.intervals[:] = _keep
                # The table FIRST: an import puts intervals on a lane,
                # and strict mode checks at the end of the gesture that
                # every interval names a row the table holds.
                self.tracks[:] = rows
                for _imp in (result.imports or []):
                    _n_imported += self._ingest_staged_column(_imp)
        except (ValueError, TypeError) as exc:
            set_status(self, "Manage Lanes changed nothing: %s" % (exc,))
            return False

        self._reconcile_active_track()
        # A gesture may not write to a lane the user cannot see, so an
        # active lane the box just hid moves to the first visible one --
        # which is what Ctrl+H has always done.
        _active = find_track(table_of(self), active_id_of(self))
        if _active is not None and not getattr(_active, "visible", True):
            _vis = visible_lanes(self)
            if _vis:
                self._set_active_track(_vis[0].id, announce=False,
                                       repaint=False)
        # Pack M2.6's doctrine: ONE SENTENCE, NOT TWO. Deleting or hiding
        # the lane you were working on MOVES the active lane, and
        # `_reconcile_active_track` writes its own line about it --
        # which the line below would then write straight over, so the
        # user would never learn that the next Add lands somewhere else.
        # The fact is composed into the one sentence instead.
        _now = find_track(table_of(self), active_id_of(self))
        _moved = ""
        if _now is not None and _now.id != _was_active:
            _moved = _now.name or _now.id
        self._repoint_class_controls()
        self._refresh_lane_controls()
        set_status(self, self._manage_lanes_note(result, _n_deleted,
                                                 _n_imported, _moved))
        self._update_plot()
        if getattr(self, "_save_autosave", None) is not None:
            self._save_autosave()
        return True

    @staticmethod
    def _manage_lanes_note(result, n_deleted, n_imported,
                           moved_to="") -> str:
        """One line naming what the OK did. Short, and true.

        `Lanes updated: 1 added, 1 deleted (41 intervals)`, plus
        ` -- active lane is now 'Wake (umbra)'` when the OK moved it.
        """
        def _n(n, word):
            return "%d %s%s" % (n, word, "" if n == 1 else "s")

        parts = []
        if result.added:
            parts.append("%d added" % len(result.added))
        if result.imports:
            parts.append("%d imported (%s)"
                         % (len(result.imports), _n(n_imported,
                                                    "interval")))
        if result.renamed:
            parts.append("%d renamed" % len(result.renamed))
        if result.deleted:
            parts.append("%d deleted (%s)"
                         % (len(result.deleted), _n(n_deleted,
                                                    "interval")))
        if result.reordered:
            parts.append("reordered")
        if result.flagged:
            parts.append("lock/visibility changed")
        return "Lanes updated: %s%s" % (
            ", ".join(parts) or "no change",
            (" -- active lane is now '%s'" % (moved_to,))
            if moved_to else "")

    # ---- the lock guard, as a method for the gesture sites --------------

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
