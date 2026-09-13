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

        # Clear
        for item in tree.get_children():
            tree.delete(item)

        # Pack M0 R2. Decide WHICH intervals this refill shows.  The
        # default is every interval the session holds, which is what the
        # list always showed; "window" scopes it to [t0, t1] with the
        # SAME test the Labels strip uses (plotting.py:813), so the list
        # and the strip can never disagree about what is on screen.
        scope = self._interval_list_scope()
        rows = []
        for i, iv in enumerate(self.intervals):
            if scope == "window" and self._interval_is_outside_window(iv):
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
        shown_labels = {}
        for _, iv in rows:
            shown_labels[iv.label] = None
        for name in shown_labels:
            tree.tag_configure(name, background=self.class_colors.get(name, "#cccccc"))

        # Refill.  Pack M0 R1: the row's identity is its Treeview iid,
        # built from the interval's index into the FULL self.intervals
        # list and recorded here in _interval_row_map.  The "#" column is
        # DISPLAY ONLY -- it numbers the rows you can see, 1..n, and
        # nothing may ever index self.intervals with it again.
        self._interval_row_map = {}
        for n, (i, iv) in enumerate(rows):
            dur = iv.end - iv.start
            start_str = iv.start.strftime("%H:%M:%S")
            end_str = iv.end.strftime("%H:%M:%S")
            dur_str = str(dur).split(".")[0]
            iid = "iv:%d" % i
            self._interval_row_map[iid] = iv
            tree.insert(
                "", "end", iid=iid, text=str(n + 1),
                values=(start_str, end_str, iv.label, dur_str),
                tags=(iv.label,),
            )

        self._update_statistics()
        
        # Update sidebar scroll region (for scrollable right panel)
        if hasattr(self, '_update_sidebar_scroll_region'):
            self._update_sidebar_scroll_region()

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
        labeled = sum((iv.end - iv.start for iv in self.intervals), total - total)  # zero Timedelta
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
