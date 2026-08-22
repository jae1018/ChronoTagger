"""
Base utility methods mixin for event handling.

This module contains utility methods extracted from EventsMixin that provide
core functionality for time conversion, snapping, interval operations, and UI updates.
"""

from __future__ import annotations

from typing import Tuple
import tkinter as tk
from tkinter import messagebox

import pandas as pd
import matplotlib.dates as mdates

# Name-tag prefix for every artist the tool draws on user axes. Artist
# scans skip anything carrying this prefix: ink is not data (Pack 3, T1).
TOOL_GID_PREFIX = "chronotagger:"


class EventsBaseMixin:
    """Base mixin providing utility methods for event handling and time operations."""

    def _update_time_window(self) -> None:
        try:
            new_t0 = pd.to_datetime(self.start_time_entry.get())  # type: ignore[union-attr]
            new_t1 = pd.to_datetime(self.end_time_entry.get())    # type: ignore[union-attr]
            if new_t0 >= new_t1:
                messagebox.showerror("Invalid Range", "Start time must be before end time.")
                return
            self.t0 = max(new_t0, self.data_start)
            self.t1 = min(new_t1, self.data_end)

            self.start_time_entry.delete(0, tk.END)  # type: ignore[union-attr]
            self.start_time_entry.insert(0, str(self.t0))  # type: ignore[union-attr]
            self.end_time_entry.delete(0, tk.END)  # type: ignore[union-attr]
            self.end_time_entry.insert(0, str(self.t1))  # type: ignore[union-attr]

            self._update_plot()
            self.status_var.set(  # type: ignore[union-attr]
                f"Window updated: {self.t0.strftime('%H:%M:%S')} → {self.t1.strftime('%H:%M:%S')}"
            )
        except Exception as e:
            messagebox.showerror("Invalid Time Format", f"Could not parse time: {e}")

    def _on_interval_tree_select(self, _event) -> None:
        sel = self.intervals_tree.selection()  # type: ignore[union-attr]
        if not sel:
            self.selected_interval = None
            self._clear_selected_interval_highlights()
            return
        item = sel[0]
        try:
            idx = int(self.intervals_tree.item(item)["text"]) - 1  # type: ignore[union-attr]
            if 0 <= idx < len(self.intervals):
                candidate_interval = self.intervals[idx]

                # Check if this is the already selected interval - if so, deselect it
                if hasattr(self, 'selected_interval') and self.selected_interval is candidate_interval:
                    # Deselect by clearing tree selection (this will trigger this method again with no selection)
                    self.intervals_tree.selection_remove(item)  # type: ignore[union-attr]
                    self.selected_interval = None
                    if hasattr(self, '_clear_selected_interval_highlights'):
                        self._clear_selected_interval_highlights()
                    self._update_strip()
                    if self.status_var is not None:
                        self.status_var.set("Interval deselected")
                    self.canvas.draw()  # type: ignore[union-attr]
                    return

                # Otherwise, select this interval
                self.selected_interval = candidate_interval
                iv = self.selected_interval
                self.status_var.set(  # type: ignore[union-attr]
                    f"Selected: {iv.label} [{iv.start.strftime('%H:%M:%S')} → {iv.end.strftime('%H:%M:%S')}]"
                )
                self._update_strip()
                self._show_selected_interval_highlights()
                self.canvas.draw()  # type: ignore[union-attr]
        except Exception:
            self.selected_interval = None
            self._clear_selected_interval_highlights()

    def _to_timestamp(self, x):
        import pandas as pd, matplotlib.dates as mdates
        # x is a Matplotlib date float or a datetime
        return pd.Timestamp(mdates.num2date(x) if isinstance(x, (int, float)) else x).tz_localize(None)

    def _preview_selection(self, start, end) -> None:
        """Show live preview across panels using current_selection (blitted)."""
        import matplotlib.dates as mdates

        # CRITICAL: Clear box selection state so highlighting works properly
        if hasattr(self, 'current_spans'):
            self.current_spans.clear()
        if hasattr(self, '_commit_spans'):
            self._commit_spans.clear()

        self.current_selection = (start, end)
        x0 = mdates.date2num(start); x1 = mdates.date2num(end)
        # Use yellow for strip editing (consistent with other completed selections)
        self._update_time_overlays(x0, x1, color="yellow")
        self._draw_strip_preview_spans([(x0, x1)])

        # Show point highlights during strip editing preview
        self._show_selected_point_highlights(redraw=True)  # Force redraw to ensure highlights appear

    def _end_after_inclusive(self, last_ts: pd.Timestamp) -> pd.Timestamp:
        """
        Return an end timestamp that is just after `last_ts` so [start, end)
        includes the last selected sample without guessing sampling cadence.
        """
        try:
            return last_ts + pd.Timedelta(nanoseconds=1)
        except Exception:
            # ultra-conservative fallback
            return last_ts + pd.Timedelta(microseconds=1)

    def _runs_to_half_open_intervals(
        self,
        idx: pd.DatetimeIndex,
        runs: list[tuple[int, int]],   # inclusive index ranges [(i0, i1), ...]
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Convert inclusive index runs to half-open [start, end) timestamp pairs:
          start = time of first included sample
          end   = time of the sample AFTER the last included (if it exists),
                  else a tiny epsilon after the last sample.

        Used for two-click selections and other cases where precise boundaries are needed.
        """
        out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        n = len(idx)

        for i0, i1 in runs:
            s = pd.Timestamp(idx[i0])
            if i1 + 1 < n:
                e = pd.Timestamp(idx[i1 + 1])
            else:
                e = self._end_after_inclusive(pd.Timestamp(idx[i1]))

            # allow at most an epsilon beyond data_end
            cap = self.data_end + pd.Timedelta(nanoseconds=1)
            if e > cap:
                e = cap

            out.append((s, e))
        return out

    def _exact_spans_to_half_open(
        self,
        spans: list[tuple[pd.Timestamp, pd.Timestamp]],
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Convert closed spans whose end sits ON a sample into half-open:
          [s, e_on_sample]  ->  [s, next_sample)   (or e + 1ns past data_end)
        Ends that fall BETWEEN samples are already half-open-correct and pass
        through unchanged. Span-level twin of _runs_to_half_open_intervals;
        successor of the retired crud._normalize_preview_spans_to_half_open
        (Pack 3, WYSIWYG doctrine).
        """
        out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        idx = self.df.index
        n = len(idx)

        for s, e in spans:
            try:
                loc = idx.get_loc(e)
                # handle duplicate timestamps (slice) or scalar int
                if isinstance(loc, slice):
                    j = loc.stop - 1
                else:
                    j = int(loc)
                if j + 1 < n:
                    e2 = pd.Timestamp(idx[j + 1])
                else:
                    e2 = self._end_after_inclusive(pd.Timestamp(idx[j]))
            except KeyError:
                e2 = e  # not on a sample -> already exclusive-correct
            except Exception:
                e2 = e

            # allow at most an epsilon beyond data_end (same cap as
            # _runs_to_half_open_intervals). Guarded: the retired crud helper
            # could never raise, and this one runs on the redraw path
            # (plotting.py:181 -> _show_selected_point_highlights). max(cap, s)
            # keeps the output a valid (possibly empty) span even for input
            # entirely past data_end.
            try:
                cap = self.data_end + pd.Timedelta(nanoseconds=1)
                if e2 > cap:
                    e2 = max(cap, s)
            except Exception:
                pass

            out.append((s, e2))
        return out
