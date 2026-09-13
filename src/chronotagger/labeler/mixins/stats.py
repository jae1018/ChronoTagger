"""
Sidebar: intervals list and statistics mixin.
"""

from __future__ import annotations
from typing import Dict

import tkinter as tk


class StatsMixin:
    def _update_intervals_list(self) -> None:
        """Refresh the sidebar list and statistics."""
        tree = self.intervals_tree  # type: ignore[assignment]

        # Pack M2 DR6: remember WHICH INTERVAL is selected before the rows
        # go, so the row that carries it can be re-selected after the
        # refill. The clear below drops the Treeview's selection, and
        # without this a refill triggered by anything at all (a lane
        # switch, a filter change, a repaint) silently deselects the
        # interval the user is working on -- and the Delete button then
        # complains there is no selection.
        _want = getattr(self, "selected_interval", None)

        # Clear
        for item in tree.get_children():
            tree.delete(item)

        # Pack M0 R2. Decide WHICH intervals this refill shows.  The
        # default is every interval the session holds, which is what the
        # list always showed; "window" scopes it to [t0, t1] with the
        # SAME test the Labels strip uses (plotting.py:813), so the list
        # and the strip can never disagree about what is on screen.
        # Pack M2 adds the second scope: WHICH LANES. It defaults to the
        # ACTIVE lane, which is what keeps this list the size it has
        # always been (60.6 ms at 8,000 held across four lanes, against
        # 230.8 ms unfiltered), and with ONE lane the two settings are the
        # same list.
        from chronotagger.core.lanes import track_display_name
        from chronotagger.core.tracks import active_id_of
        active_track = active_id_of(self)
        scope = self._interval_list_scope()
        track_scope = self._interval_track_scope()
        rows = []
        for i, iv in enumerate(self.intervals):
            if scope == "window" and self._interval_is_outside_window(iv):
                continue
            if track_scope == "active" and iv.track != active_track:
                continue
            rows.append((i, iv))

        # Pack M0 R3. tag_configure used to be called once per ROW, from
        # inside the insert loop below.  It configures a TAG, not a row,
        # so once per distinct label in this refill is enough.  Measured
        # on this widget at 8,000 rows: 8,000 calls cost 20.5 ms and 4
        # calls cost 0.0 ms, which is 9.6 % of the 214.6 ms the clear
        # and refill take.  The label set is built from the rows actually
        # shown, NOT from self.classes, so an interval whose class was
        # deleted from the schema still gets the "#cccccc" default it
        # got before.
        # Pack M2 closes M1's DR14. The tag name is now PER LANE
        # ("<track>|<label>") and its colour comes from THAT LANE's own
        # map, so two lanes that share a class name no longer share a
        # colour. M1 said this out loud as a known defect: measured on two
        # lanes sharing `solar_wind`, one row was painted in the wrong
        # lane's colour and two got no colour at all, 3 of 6 wrong; with
        # the per-lane tag it is 0 of 6. The cost is nil (-0.8 to -9 ms,
        # inside noise) because the number of DISTINCT tags is what
        # matters and it barely moves.
        from chronotagger.core.tracks import table_of
        _table = table_of(self)
        _colors = {}
        _names = {}
        for _t in _table:
            _colors[_t.id] = _t.class_colors
            _names[_t.id] = _t.name or _t.id
        shown_tags = {}
        for _, iv in rows:
            shown_tags["%s|%s" % (iv.track, iv.label)] = (iv.track, iv.label)
        for tag, (tid, label) in shown_tags.items():
            tree.tag_configure(
                tag, background=_colors.get(tid, {}).get(label, "#cccccc"))

        # Refill.  Pack M0 R1: the row's identity is its Treeview iid,
        # built from the interval's index into the FULL self.intervals
        # list and recorded here in _interval_row_map.  The "#" column is
        # DISPLAY ONLY -- it numbers the rows you can see, 1..n, and
        # nothing may ever index self.intervals with it again.
        # Pack M2: that is exactly what makes the lane filter safe. With
        # the filter on, row 3 of 5 is intervals[11]; the iid says so and
        # the ordinal would have named a different interval on a different
        # lane -- measured, 5 of 5 right through the iid and 0 of 5 right
        # through the ordinal.
        self._interval_row_map = {}
        _reselect = None
        for n, (i, iv) in enumerate(rows):
            dur = iv.end - iv.start
            start_str = iv.start.strftime("%H:%M:%S")
            end_str = iv.end.strftime("%H:%M:%S")
            dur_str = str(dur).split(".")[0]
            iid = "iv:%d" % i
            self._interval_row_map[iid] = iv
            tree.insert(
                "", "end", iid=iid, text=str(n + 1),
                values=(start_str, end_str, iv.label, dur_str,
                        _names.get(iv.track, iv.track)),
                tags=("%s|%s" % (iv.track, iv.label),),
            )
            if _want is not None and iv is _want:
                _reselect = iid

        # Pack M2 DR6: re-assert the selection on the row whose iid still
        # carries the selected interval. Done by IDENTITY (`is`), not by
        # index, because the index moves whenever the filter or the scope
        # changes -- which is the whole point of the row map.
        if _reselect is not None:
            # The flag stops _on_interval_tree_select from treating this as
            # the user clicking the already-selected row, which is its
            # DESELECT gesture (events/base.py). Without it a refill would
            # toggle the selection off.
            self._suppress_tree_select = True
            try:
                tree.selection_set(_reselect)
            except Exception:
                pass
            # ttk sends <<TreeviewSelect>> through Tcl_QueueEvent, so it is
            # delivered on a LATER turn of the event loop, NOT inside
            # selection_set. Clearing the flag here left it False by the
            # time the handler ran -- measured on a real Tk loop: the
            # refill re-selected the row, the queued event walked into the
            # deselect branch, selected_interval became None, the status
            # bar said "Interval deselected" and the next Delete popped
            # "No Selection". An idle callback runs only once the event
            # queue is EMPTY, which is after that event; a host with no
            # `root` (the suite's GUI-free mocks) has no queue, so it
            # drops the flag at once.
            _root = getattr(self, "root", None)
            if _root is None:
                self._suppress_tree_select = False
            else:
                try:
                    _root.after_idle(self._release_tree_select_suppression)
                except Exception:
                    self._suppress_tree_select = False
        elif (_want is not None and track_scope == "active"
                and _want.track != active_track):
            # The selected interval is on a lane this list is filtering
            # out. Say so rather than letting the user wonder where the
            # highlight went; the interval is still selected and Delete
            # still works on it.
            if getattr(self, "status_var", None) is not None:
                try:
                    self.status_var.set(
                        "the selected interval sits on lane '%s', which this "
                        "list is not showing"
                        % (track_display_name(self, _want.track),))
                except Exception:
                    pass

        self._update_statistics()
        
        # Update sidebar scroll region (for scrollable right panel)
        if hasattr(self, '_update_sidebar_scroll_region'):
            self._update_sidebar_scroll_region()

    def _release_tree_select_suppression(self) -> None:
        """Drop DR6's suppression once Tk has delivered the queued event.

        Named rather than a lambda so a test can call it, and so an
        `after_idle` that outlives the widget cannot hold a closure over
        the refill's locals.
        """
        self._suppress_tree_select = False

    # ---- Pack M2: interval-list LANE scope --------------------------------

    def _interval_track_scope(self) -> str:
        """Return "active" or "all" -- WHICH LANES the interval list shows.

        "active" is the default, and it is the setting that keeps this
        list the size it has always been: measured on the real widget at
        8,000 intervals held across four lanes, the unfiltered refill is
        230.8 ms and the active-lane refill is 60.6 ms -- the cost of
        today's one-lane list. Anything that is not exactly "all" reads as
        "active", so a labeler built without the sidebar control behaves
        as though the filter were on, which with ONE lane is the same list
        either way.
        """
        var = getattr(self, "interval_track_scope_var", None)
        if var is None:
            return "active"
        try:
            value = str(var.get())
        except Exception:
            return "active"
        return "all" if value == "all" else "active"

    def _on_interval_track_scope_change(self) -> None:
        """Sidebar Lanes: active / all changed -> refill the list."""
        if not hasattr(self, "intervals_tree"):
            return
        self._update_intervals_list()
        if getattr(self, "status_var", None) is None:
            return
        held = len(self.intervals)
        shown = len(self.intervals_tree.get_children())
        if self._interval_track_scope() == "active":
            from chronotagger.core.lanes import track_display_name
            from chronotagger.core.tracks import active_id_of
            self.status_var.set(
                "Interval list scoped to lane '%s': %d of %d shown"
                % (track_display_name(self, active_id_of(self)), shown, held))
        else:
            self.status_var.set(
                "Interval list showing every lane: %d of %d shown"
                % (shown, held))

    # ---- Pack M0: interval-list scope ------------------------------------

    def _interval_list_scope(self) -> str:
        """Return "all" or "window" -- what the interval list shows.

        "all" is the default and is what the list always did.  The
        sidebar's Show radio buttons write interval_scope_var; anything
        that is not exactly "window" reads as "all", so a labeler built
        without a sidebar (or an older session) behaves exactly as before.
        """
        var = getattr(self, "interval_scope_var", None)
        if var is None:
            return "all"
        try:
            value = str(var.get())
        except Exception:
            return "all"
        return "window" if value == "window" else "all"

    def _interval_is_outside_window(self, iv) -> bool:
        """True when `iv` paints nothing in [t0, t1].

        This is the Labels strip's own test (plotting.py:813), quoted so
        the list and the strip agree by construction.  Half-open: an
        interval that ends exactly at t0, or starts exactly at t1, is
        outside.
        """
        return iv.end <= self.t0 or iv.start >= self.t1

    def _on_interval_scope_change(self) -> None:
        """Sidebar Show: all / window changed -> refill the list.

        The status line says how many rows were kept, because a list that
        silently got short is the thing this control is most likely to be
        blamed for.  Statistics are NOT scoped: coverage always describes
        the whole session (Pack M0 SECTION 0e item 3).
        """
        if not hasattr(self, "intervals_tree"):
            return
        self._update_intervals_list()
        if getattr(self, "status_var", None) is None:
            return
        held = len(self.intervals)
        if self._interval_list_scope() == "window":
            shown = len(self.intervals_tree.get_children())
            self.status_var.set(
                "Interval list scoped to this window: %d of %d shown"
                % (shown, held))
        else:
            self.status_var.set(
                "Interval list showing all %d intervals" % held)

    def _update_statistics(self) -> None:
        txt = self.stats_text  # type: ignore[assignment]
        txt.config(state="normal")
        txt.delete(1.0, tk.END)

        if not self.intervals:
            txt.insert(tk.END, "No intervals labeled yet.")
            txt.config(state="disabled")
            return

        total = self.data_end - self.data_start
        # Pack M1: UNION coverage, not the sum of durations. Measured on a
        # four-interval two-track set through this very box: the sum reads
        # "Labeled: 0 days 01:10:00 / 0 days 00:59:59" and
        # "Coverage: 116.7%" -- more than all of time. With ONE track the
        # union IS the sum, to the nanosecond, because a track's intervals
        # cannot overlap. This is the third copy of that arithmetic; the
        # other two are in io_export._save_autosave and now call the same
        # helper, so the number can no longer differ between the sidebar
        # and the recovery dialog.
        from chronotagger.core.tracks import union_covered
        labeled = union_covered(self.intervals)
        pct = (labeled / total * 100) if total.total_seconds() > 0 else 0.0

        counts: Dict[str, int] = {}
        durations = {}
        for iv in self.intervals:
            counts[iv.label] = counts.get(iv.label, 0) + 1
            durations[iv.label] = durations.get(iv.label, total - total) + (iv.end - iv.start)

        txt.insert(tk.END, f"Total Intervals: {len(self.intervals)}\n")
        txt.insert(tk.END, f"Labeled: {labeled} / {total}\n")
        txt.insert(tk.END, f"Coverage: {pct:.1f}%\n\n")
        txt.insert(tk.END, "By Label:\n")
        for label in sorted(counts):
            lpct = (durations[label] / total * 100) if total.total_seconds() > 0 else 0.0
            txt.insert(tk.END, f"  {label}: {counts[label]} intervals, {lpct:.1f}%\n")

        txt.config(state="disabled")
