"""
Synchronization manager for multi-pane mode.

Handles marking panes dirty when shared state changes.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import TimeIntervalLabeler
    import pandas as pd


class PaneSyncManager:
    """
    Manages synchronization of shared state across panes.

    Shared state that triggers sync:
    - Time window (t0, t1)
    - Intervals (add, delete, modify)
    - Labels (add, delete, rename, recolor)
    - Selection (optional - could be per-pane or shared)

    Per-pane state (no sync needed):
    - Figure/canvas/axes
    - Zoom levels
    - Dirty flag
    """

    def __init__(self, labeler: TimeIntervalLabeler):
        """
        Parameters
        ----------
        labeler : TimeIntervalLabeler
            The parent labeler instance
        """
        self.labeler = labeler

    def sync_time_window(self, t0: pd.Timestamp, t1: pd.Timestamp) -> None:
        """
        Time window changed - mark all panes dirty.

        Called when user navigates, zooms, or manually changes time range.
        """
        self.labeler.t0 = t0
        self.labeler.t1 = t1

        # Mark all panes as needing update
        for pane in self.labeler.panes:
            pane.mark_dirty()

    def sync_intervals_changed(self) -> None:
        """
        Intervals modified - mark all panes dirty.

        Called after: add, delete, modify, relabel, merge, split, etc.
        """
        for pane in self.labeler.panes:
            pane.mark_dirty()

    def sync_labels_changed(self) -> None:
        """
        Label schema changed - mark all panes dirty.

        Called after: add label, delete label, rename, recolor, reorder.
        """
        for pane in self.labeler.panes:
            pane.mark_dirty()

    def sync_selection_changed(self) -> None:
        """
        Selection changed - update active pane only.

        Selection preview only needs to show on active pane since
        it's a transient state before adding an interval.

        Alternative design: Could mark all panes dirty to show selection
        on all tabs, but that's probably overkill.
        """
        self.labeler.active_pane.mark_dirty()

    def mark_all_dirty(self) -> None:
        """
        Mark all panes dirty (for any other global state change).
        """
        for pane in self.labeler.panes:
            pane.mark_dirty()
