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

        # Refill
        for i, iv in enumerate(self.intervals):
            dur = iv.end - iv.start
            start_str = iv.start.strftime("%H:%M:%S")
            end_str = iv.end.strftime("%H:%M:%S")
            dur_str = str(dur).split(".")[0]
            tree.insert(
                "", "end", text=str(i + 1),
                values=(start_str, end_str, iv.label, dur_str),
                tags=(iv.label,),
            )
            tree.tag_configure(iv.label, background=self.class_colors.get(iv.label, "#cccccc"))

        self._update_statistics()
        
        # Update sidebar scroll region (for scrollable right panel)
        if hasattr(self, '_update_sidebar_scroll_region'):
            self._update_sidebar_scroll_region()

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
