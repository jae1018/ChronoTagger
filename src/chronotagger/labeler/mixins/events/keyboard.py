"""
Keyboard event handling mixin.

This file contains all keyboard event handling methods extracted from
src/chronotagger/labeler/mixins/events.py.

Responsibilities:
- Keyboard shortcuts (digits, navigation, actions)
- Key press event handling
- Special key handling (ESC, Enter, Delete, BackSpace)
- Focus-aware keyboard input
- Modifier key detection (Ctrl+S, Ctrl+Z, etc.)
"""

from __future__ import annotations

from typing import Optional


class KeyboardEventsMixin:
    """
    Mixin class containing all keyboard event handling methods.

    Handles:
    - Key press events via _on_key_press
    - Special keys: Escape, Enter, Delete, BackSpace
    - Keyboard shortcuts: digits (1-9), n/p (navigation), a/d (actions)
    - Modifier combinations: Ctrl+S, Ctrl+E, Ctrl+Z, Ctrl+Y
    - Focus-aware input handling
    """

    def _focused_widget_is_editable(self) -> bool:
        """
        Return True if the current keyboard focus is on an editable widget
        (Entry, Text, or Combobox). In that case we should not handle global
        navigation/shortcut keys, so the widget's native editing behavior wins.
        """
        if getattr(self, "root", None) is None:
            return False
        w = self.root.focus_get()
        if w is None:
            return False

        # Prefer Tk class names — reliable across ttk/tk variants.
        try:
            cls = w.winfo_class()
        except Exception:
            return False

        # Common editable classes: 'Entry' (tk), 'TEntry' (ttk), 'Text', 'TCombobox'
        return cls in {"Entry", "TEntry", "Text", "TCombobox"}


    def _on_key_press(self, event) -> None:
        key = event.keysym

        # ---- Escape cancels ANY active selection/preview OR deselects interval ----
        if key == "Escape":
            # Check if we have any active selection or preview
            has_selection = (
                getattr(self, "_pick_anchor_ts", None) is not None or
                getattr(self, "current_selection", None) is not None or
                bool(getattr(self, "current_spans", None)) or
                bool(getattr(self, "_commit_spans", None)) or
                getattr(self, "_two_click_active", False)
            )

            if has_selection:
                # Clear all selection states
                self._cancel_active_selection()
                if self.status_var is not None:
                    self.status_var.set("Selection canceled (Escape)")
                return

            # If no active selection, check for selected interval to deselect
            if hasattr(self, 'selected_interval') and self.selected_interval is not None:
                self.selected_interval = None
                if hasattr(self, '_clear_selected_interval_highlights'):
                    self._clear_selected_interval_highlights()
                self._update_strip()
                if hasattr(self, '_update_intervals_list'):
                    self._update_intervals_list()
                if self.canvas is not None:
                    self.canvas.draw_idle()
                if self.status_var is not None:
                    self.status_var.set("Interval deselected (Escape)")
                return

        # ---- Focus-aware early exit -------------------------------------------
        if self._focused_widget_is_editable():
            if not (event.state & 0x4):  # 0x4 => Control modifier
                return
        # -----------------------------------------------------------------------

        # Class selection with digits 1..9
        if key.isdigit() and int(key) > 0:
            idx = int(key) - 1
            if idx < len(self.classes):
                self.current_class_var.set(self.classes[idx])  # type: ignore[union-attr]
                self.status_var.set(f"Selected class: {self.classes[idx]}")  # type: ignore[union-attr]
            return

        # Navigation
        if key in ("n", "N", "Right"):
            self._next_window()
            return
        if key in ("p", "P", "Left"):
            self._prev_window()
            return

        # Actions
        if key in ("a", "A", "Return"):
            self._add_interval()
            return
        if key in ("d", "D", "Delete"):
            self._delete_interval()
            return
        if key in ("u", "U"):
            if "UNKNOWN" in self.classes:
                self.current_class_var.set("UNKNOWN")  # type: ignore[union-attr]
                self.status_var.set("Selected class: UNKNOWN")  # type: ignore[union-attr]
            return

        # Save / export (Ctrl+S / Ctrl+E)
        if key in ("s", "S") and (event.state & 0x4):
            self._save_session()
            return
        if key in ("e", "E") and (event.state & 0x4):
            self._export_intervals()
            return

        # Undo / Redo (Ctrl+Z / Ctrl+Y) + Backspace ergonomics
        if (key == "z" and (event.state & 0x4)) or key == "BackSpace":
            self._undo()
            return
        if (key == "y" and (event.state & 0x4)) or (key == "BackSpace" and (event.state & 0x1)):
            self._redo()
            return


    def _cancel_active_selection(self) -> None:
        """
        Cancel any active selection - SIMPLE BUT THOROUGH approach.
        """
        # Clear all selection state
        self.current_selection = None
        if hasattr(self, 'current_spans'):
            self.current_spans.clear()
        if hasattr(self, '_commit_spans'):
            self._commit_spans.clear()

        # Clear two-click state
        self._two_click_active = False
        self._two_click_t0 = None
        self._two_click_last_x = None

        # Clear point highlights
        if hasattr(self, '_clear_selected_point_highlights'):
            self._clear_selected_point_highlights()

        # Hide overlays using existing method
        self._hide_time_overlays()

        # Clear strip previews
        self._draw_strip_preview_spans([])

        # Update strip
        self._update_strip()

        # Force canvas redraw
        if hasattr(self, 'canvas') and self.canvas is not None:
            self.canvas.draw_idle()

    # ---- Multi-pane tab navigation methods ----

    def _next_tab(self, event=None) -> str:
        """Switch to next tab (cycle forward)."""
        if not getattr(self, 'multi_pane_mode', False) or not getattr(self, 'notebook', None):
            return 'break'

        n_tabs = len(self.panes)
        next_idx = (self.active_pane_idx + 1) % n_tabs
        self.notebook.select(next_idx)
        return 'break'  # Prevent default Tab behavior

    def _prev_tab(self, event=None) -> str:
        """Switch to previous tab (cycle backward)."""
        if not getattr(self, 'multi_pane_mode', False) or not getattr(self, 'notebook', None):
            return 'break'

        n_tabs = len(self.panes)
        prev_idx = (self.active_pane_idx - 1) % n_tabs
        self.notebook.select(prev_idx)
        return 'break'

    def _go_to_tab(self, idx: int) -> str:
        """Jump to specific tab by index (0-based)."""
        if not getattr(self, 'multi_pane_mode', False) or not getattr(self, 'notebook', None):
            return 'break'

        if 0 <= idx < len(self.panes):
            self.notebook.select(idx)
        return 'break'
