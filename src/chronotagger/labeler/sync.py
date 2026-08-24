"""
Synchronization manager for multi-pane mode.

Pack 6 R8 emptied this module: it used to mark panes "dirty" when shared
state changed, and the dirty flag it wrote is gone -- TabPane.mark_clean()
had zero production callers, so nothing ever became clean, `dirty` was
permanently True, and the single consumer (view_build/window.py) always
took its redraw branch regardless.

Pack 6 PART C then deleted the four hooks that had no caller at all --
sync_time_window, sync_labels_changed, sync_selection_changed and
mark_all_dirty. A census of sync_manager attribute uses over src/ and
tests/ returns 8 hits and every one of them is sync_intervals_changed, so
those four deletions cost zero call-site edits and touch no other file.

sync_intervals_changed SURVIVES, as a declared no-op, because those 8 live
call sites (app.py:384, intervals/commands.py x2, intervals/crud.py x4,
labels.py) would all have to change with it -- and where multi-pane
invalidation should live is a design question, not a cleanup. It is Pack 7
item 5.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import TimeIntervalLabeler


class PaneSyncManager:
    """
    Notification hook for interval changes in multi-pane mode.

    Nothing here carries state any more. The redraw that the callers of
    sync_intervals_changed rely on comes from their own _update_plot().
    """

    def __init__(self, labeler: TimeIntervalLabeler):
        """
        Parameters
        ----------
        labeler : TimeIntervalLabeler
            The parent labeler instance
        """
        self.labeler = labeler

    def sync_intervals_changed(self) -> None:
        """
        Intervals modified (add, delete, modify, relabel, merge, split).

        Pack 6 R8: a declared no-op. EIGHT live call sites keep calling
        it (app.py:384, intervals/commands.py x2, intervals/crud.py x4,
        labels.py), so the hook stays; what it used to do -- mark every
        pane dirty -- was a write to a flag that was already True and that
        nothing ever cleared. The redraw those callers rely on comes from
        their own _update_plot(), not from here.
        """
