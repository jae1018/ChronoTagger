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

        # ---- Pack M2: the lane keys ------------------------------------------
        # Ctrl+Down / Ctrl+Up cycle the ACTIVE lane; Ctrl+L locks or unlocks
        # it; Ctrl+H hides or shows it. All four are Control-modified ON
        # PURPOSE: the focus-aware exit six lines above returns for any
        # UNMODIFIED key while an editable widget has focus, and the class
        # dropdown IS one (a TCombobox), so `l`, `h` and `v` are swallowed
        # the moment the user has clicked it once. Measured free in all
        # five modifier states. Ctrl+Up/Ctrl+Down are also ELAN's own lane
        # bindings.
        #
        # PACK M2.6 CORRECTS WHAT THIS COMMENT USED TO CLAIM. It said "no
        # Tk-level bind on root collides (Ctrl+digit and Ctrl+Tab are the
        # pane tabs)", and that was the wrong question: it checked ROOT
        # binds, and the collision is a CLASS bind. The ttk Entry class
        # binds <Control-Key-h> to ttk::entry::Backspace, <Control-Key-d>
        # to ttk::entry::Delete and <Control-Key-k> to "delete to end of
        # line", none of them ending in `break` -- so Ctrl+H typed in the
        # Start box deleted a character AND hid the lane. Those three are
        # now overridden app-wide at build time, by
        # `_neutralize_entry_class_keys` below, which is why Ctrl+H can
        # go on being the hide key everywhere.
        if key in ("Down", "Up") and (event.state & 0x4):
            self._cycle_active_lane(1 if key == "Down" else -1)
            return
        if key in ("l", "L") and (event.state & 0x4):
            self._toggle_active_lane_locked()
            return
        if key in ("h", "H") and (event.state & 0x4):
            self._toggle_active_lane_visible()
            return
        # -----------------------------------------------------------------------

        # Pack M2.5: EVERY PLAIN SHORTCUT BELOW IS DEAD WHILE CONTROL IS
        # HELD. The focus-aware exit above returns only for an UNMODIFIED
        # key, so Ctrl+A in the Start box selected the text (Tk's own
        # <<SelectAll>> on Windows) AND fired Add, which popped a "No
        # Selection" modal over the user's hands -- found in the live window,
        # twice. The same collision was live for Ctrl+D (delete the selected
        # interval), Ctrl+N / Ctrl+P and Ctrl+Left / Ctrl+Right (the window
        # navigation), Ctrl+U (set UNKNOWN) and Ctrl+1..9, which ALSO switch
        # pane tabs at the Tk level, so one keystroke changed the tab and the
        # class. The Control-DEFINED shortcuts are untouched: Ctrl+Up,
        # Ctrl+Down, Ctrl+L, Ctrl+H, Ctrl+S, Ctrl+E, Ctrl+Z, Ctrl+Y.
        plain = not (event.state & 0x4)   # 0x4 => Control modifier

        # Class selection with digits 1..9
        if key.isdigit() and int(key) > 0 and plain:
            idx = int(key) - 1
            if idx < len(self.classes):
                self.current_class_var.set(self.classes[idx])  # type: ignore[union-attr]
                self.status_var.set(f"Selected class: {self.classes[idx]}")  # type: ignore[union-attr]
            return

        # Navigation
        if key in ("n", "N", "Right") and plain:
            self._next_window()
            return
        if key in ("p", "P", "Left") and plain:
            self._prev_window()
            return

        # Actions
        if key in ("a", "A", "Return") and plain:
            self._add_interval()
            return
        if key in ("d", "D", "Delete") and plain:
            self._delete_interval()
            return
        if key in ("u", "U") and plain:
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

        # Undo / Redo (Ctrl+Z / Ctrl+Y) + Backspace ergonomics.
        # Pack M2.5: plain BackSpace still undoes; Ctrl+BackSpace does not.
        # Ctrl+BackSpace is "delete the previous word" in every text box on
        # this platform, and in the Start / End boxes it used to undo the
        # user's last data edit instead. THE REDO BRANCH NEEDS THE SAME
        # GUARD: BackSpace reaches it whenever Shift is held, so guarding
        # only the branch above moved Ctrl+Shift+BackSpace from UNDO to
        # REDO instead of silencing it -- measured, and still a data
        # command fired from inside a text box. Shift+BackSpace on its own
        # still reaches Undo, as it has since long before Pack M2 (DR14).
        if (key == "z" and (event.state & 0x4)) or (key == "BackSpace" and plain):
            self._undo()
            return
        if (key == "y" and (event.state & 0x4)) or (key == "BackSpace" and plain and (event.state & 0x1)):
            self._redo()
            return


    # Tk's OWN emacs keys on a ttk Entry, and the three sequences this
    # app has to take back. Measured on this machine
    # (probe_s3_ctrl_d_entry.txt): the bindtags on the Start / End boxes
    # are (the widget, 'TEntry', '.', 'all'), and the TEntry CLASS binds
    #
    #   <Control-Key-d>   ttk::entry::Delete %W      delete the character
    #   <Control-Key-h>   ttk::entry::Backspace %W   delete the one before
    #   <Control-Key-k>   %W delete insert end       delete to end of line
    #
    # NONE of those three scripts ends in `break`, so after each of them
    # edits the text the event carries on down the bindtags to the root's
    # own <Key> handler as well.
    ENTRY_CLASS_KEYS = ("<Control-Key-d>", "<Control-Key-h>",
                        "<Control-Key-k>")

    def _neutralize_entry_class_keys(self) -> None:
        """Stop Tk's Ctrl+D / Ctrl+H / Ctrl+K editing the time boxes.

        Pack M2.6. Ctrl+H in the Start box did TWO things: Tk's own class
        binding deleted the character before the cursor, and then the
        root handler hid the active lane. Measured end to end:
        "2020-09-02 07:00:00" became "2020-09-02 7:00:00" AND the Region
        lane went, with no undo entry for the hide. Ctrl+D deleted a
        character (that is what computer-use session 3 reported as "the
        Start box reverted to 00:00:00"), and Ctrl+K deleted to the end
        of the line.

        Each of the three is rebound APP-WIDE, on the TEntry CLASS, to a
        Python no-op that returns None -- NOT "break". Returning None is
        the point: Tk carries on down the bindtags to the root's <Key>
        handler, which is the half that must keep working, because Ctrl+H
        is the app's hide-lane key and the lane keys are Control-defined
        precisely so they fire while a text box has focus. After this,
        Ctrl+H hides the lane exactly ONCE and edits nothing, and Ctrl+D
        and Ctrl+K edit nothing and do nothing at all -- Pack M2.5's
        `plain` guard already made those two inert while Control is held.

        Ctrl+A is deliberately NOT touched: it reaches the entry through
        the virtual event <<SelectAll>> (` %W selection range 0 end `),
        not through a Control-Key binding, and selecting all the text in
        the box you are typing in is what the user wants. Pack M2.5
        already stopped it from also firing Add.

        Called once, from `_build_gui`. bind_class is per Tcl
        interpreter, and since Pack M2.5 there is exactly one.

        TCombobox and TSpinbox carry the SAME three scripts and are NOT
        overridden. Measured: every editable box in this window and in
        the By-Rule dialog is a TEntry, and both comboboxes are
        state=readonly, which the ttk scripts decline to edit. Make one
        editable and this method has to grow two more class names.
        """
        root = getattr(self, "root", None)
        if root is None:
            return
        for seq in self.ENTRY_CLASS_KEYS:
            try:
                root.bind_class("TEntry", seq, self._entry_class_key_noop)
            except Exception:
                pass

    def _entry_class_key_noop(self, event=None):
        """The override itself. Does nothing, and RETURNS None.

        Returning None rather than "break" is the entire contract, and it
        is a named method rather than a lambda so a test can assert it:
        None lets Tk keep walking the bindtags to the root's <Key>
        handler, so Ctrl+H typed in a text box still hides the lane.
        "break" would edit nothing AND silence the app's own key, which
        would trade one defect for another.
        """
        return None

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
