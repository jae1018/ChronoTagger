"""
Main window construction mixin.

Responsibilities:
- Build the main tkinter window
- Orchestrate construction of top controls, plot, and sidebar
- Set up global key bindings

This mixin provides the top-level _build_gui method that creates the entire UI.
"""

from __future__ import annotations

import logging
import time
import tkinter as tk
from tkinter import ttk

logger = logging.getLogger(__name__)

# Pack 6 R10 -- bounded retry around tk.Tk().
#
# MEASURED on this Windows box (conda chronotagger_test, Tk/Tcl 8.6.15):
# tk.Tk() raises TclError in the first seconds of a process at a rate of
# 16 of 18 full-suite runs (89%) and 14 of 20 single-module runs (70%).
# It is ALWAYS a setup-time error, never a test failure, and it is NOT a
# multi-root problem: 106 roots are created per suite run, peak
# simultaneous live roots is 1, all 106 are destroyed, and the failing
# root ordinals cluster in the first six and never appear past the 37th.
# The four error strings ("Can't find a usable init.tcl", "Can't find a
# usable tk.tcl", 'invalid command name "tcl_findLibrary"', and -- the
# tell -- "couldn't read file .../ttk/sizegrip.tcl: no such file or
# directory" on a file that demonstrably exists) are one transient
# file-read fault while Tcl sources an already-located library.  It
# self-clears within 250 ms.
#
# Measured alternatives: exporting TCL_LIBRARY/TK_LIBRARY -> 9 of 20,
# still broken; consolidating to a single session root -> does not fix it
# and converts a one-test flake into a whole-suite flake.  The bounded
# retry measured 0 of 20 single-module runs and 0 of 5 full-suite runs,
# with every recovery inside two attempts.  It lives in src rather than
# in tests/conftest.py because the SHIPPED app hits this too on this
# machine, not only the suite.
_TK_ROOT_ATTEMPTS = 5
_TK_ROOT_BACKOFF_S = (0.05, 0.10, 0.15, 0.25)

# Deliberately NOT warn-once.  Every recovery is signal: a rising count
# is the only thing that distinguishes "the retry is catching a live
# fault" from "the machine went quiet and the fix is unproven".
tk_root_retry_recoveries = 0


def _new_tk_root() -> tk.Tk:
    """tk.Tk(), retried on the transient startup TclError (Pack 6 R10).

    At most _TK_ROOT_ATTEMPTS attempts with a short backoff between them.
    ONLY tkinter.TclError is retried -- anything else propagates on the
    first raise -- and the LAST TclError propagates unchanged when every
    attempt fails, so a genuinely broken Tk install still fails loudly
    instead of being masked by the workaround.
    """
    global tk_root_retry_recoveries
    last_exc = None
    for attempt in range(_TK_ROOT_ATTEMPTS):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            last_exc = exc
            if attempt + 1 < _TK_ROOT_ATTEMPTS:
                time.sleep(_TK_ROOT_BACKOFF_S[attempt])
            continue
        if attempt:
            tk_root_retry_recoveries += 1
            logger.warning(
                "tk.Tk() raised TclError on %d attempt(s) before "
                "succeeding; recovery #%d this session. Last error: %s",
                attempt, tk_root_retry_recoveries, last_exc)
        return root
    raise last_exc


class WindowMixin:
    """
    Mixin providing main window construction.

    This mixin expects the following attributes/methods to be available on self:
    - _on_closing() - Callback for window close event
    - _build_top_controls(parent) - Method to build top controls
    - _build_sidebar(parent) - Method to build sidebar
    - _build_plot(parent) - Method to build plot area
    - _on_key_press(event) - Callback for key press events
    - _open_help_dialog(event) - Callback for opening help dialog
    - _toggle_sidebar() - Callback for toggling sidebar visibility

    Attributes created:
    - root: tk.Tk - The main tkinter root window
    - sidebar_frame: ttk.Frame - The sidebar frame
    - plot_frame: ttk.Frame - The plot frame

    Methods provided:
    - _build_gui() - Build the complete GUI (window, controls, plot, sidebar)
    """

    def _build_gui(self) -> None:
        # Mount as Toplevel under an existing Tk root when one is provided
        # (e.g. when launched from the quick-start wizard).  This keeps the
        # process to a single tk.Tk root, which is required for tk.StringVar
        # / IntVar / BooleanVar bindings to resolve to the same Tcl
        # interpreter as the widgets they're bound to.  Otherwise a
        # nested tk.Tk() would create a second interpreter and silently
        # break textvariable links.
        parent = getattr(self, "_parent", None)
        if parent is not None:
            self.root = tk.Toplevel(parent)
        else:
            # Pack 6 R10: bounded retry -- see _new_tk_root above for the
            # measurement (89% of full-suite runs errored here before it).
            self.root = _new_tk_root()
        self.root.title("ChronoTagger - Time Interval Labeler")
        self.root.geometry("1600x900")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self._build_top_controls(top)

        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        # need to create sidebar BEFORE creating full plot, otherwise it
        # doesn't show up
        self.sidebar_frame = ttk.Frame(main, width=320)
        self.sidebar_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        self.sidebar_frame.pack_propagate(False)
        self._build_sidebar(self.sidebar_frame)

        # === Multi-pane vs single-pane logic ===
        if self.multi_pane_mode:
            # Create notebook with tabs
            self.notebook = ttk.Notebook(main)
            self.notebook.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Create a tab for each pane
            for idx, pane in enumerate(self.panes):
                tab_frame = ttk.Frame(self.notebook)
                self.notebook.add(tab_frame, text=pane.title)

                # Build figure/canvas for this pane
                self._build_pane_canvas(pane, tab_frame)

            # Create and bind tab context menu
            self._create_tab_context_menu()
            self.notebook.bind('<Button-3>', self._show_tab_context_menu)

            # Bind tab change event
            self.notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)
        else:
            # Single-pane mode (backward compatible)
            self.plot_frame = ttk.Frame(main)
            self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Build figure/canvas for the single pane
            self._build_plot(self.plot_frame)

        self.root.bind("<Key>", self._on_key_press)
        # F1 opens Help
        self.root.bind("<F1>", self._open_help_dialog)
        # F9 toggles sidebar visibility
        self.root.bind("<F9>", lambda e: self._toggle_sidebar())

        # Multi-pane tab navigation shortcuts
        if self.multi_pane_mode:
            # Ctrl+Tab / Ctrl+Shift+Tab for next/prev tab
            self.root.bind('<Control-Tab>', self._next_tab)
            self.root.bind('<Control-Shift-Tab>', self._prev_tab)

            # Ctrl+1 through Ctrl+9 for direct tab access
            for i in range(1, 10):
                self.root.bind(f'<Control-Key-{i}>',
                              lambda e, idx=i-1: self._go_to_tab(idx))

            # Ctrl+0 for tab 10
            self.root.bind('<Control-Key-0>', lambda e: self._go_to_tab(9))

    def _on_tab_changed(self, event) -> None:
        """Handle notebook tab change event."""
        if not self.multi_pane_mode or self.notebook is None:
            return

        # Update active pane index
        self.active_pane_idx = self.notebook.index(self.notebook.select())

        # Sync pane metadata to main class for backward compatibility
        pane = self.active_pane
        self.axes_meta = pane.axes_meta
        self._time_axis_keys = pane.time_axis_keys
        self._primary_time_key = pane.primary_time_key

        # Redraw the newly active pane.
        #
        # Pack 6 R8: this used to be guarded by
        # `self.active_pane.needs_update(self.t0, self.t1)`, which could
        # never return False. mark_clean() had ZERO production callers, so
        # `dirty` was never cleared and `last_window` was never set, and
        # needs_update() returned True on its first branch every time.
        # Measured on a live two-pane app after building, plotting and tab
        # churn: dirty = [True, True], last_window = [None, None].
        self._update_plot()

        # Update sidebar to show intervals in current window
        if hasattr(self, 'update_intervals_list'):
            self.update_intervals_list()
        if hasattr(self, '_update_stats'):
            self._update_stats()

    # ---- Tab context menu methods ----

    def _create_tab_context_menu(self) -> None:
        """Create right-click context menu for tabs."""
        self.tab_menu = tk.Menu(self.root, tearoff=0)
        self.tab_menu.add_command(
            label="Rename Tab",
            command=self._rename_active_tab
        )
        self.tab_menu.add_command(
            label="Refresh Tab",
            command=self._refresh_active_tab
        )
        self.tab_menu.add_separator()
        self.tab_menu.add_command(
            label="Refresh All Tabs",
            command=self._refresh_all_tabs
        )

    def _show_tab_context_menu(self, event) -> None:
        """Show context menu on right-click."""
        if not self.multi_pane_mode:
            return

        # Determine which tab was clicked
        try:
            clicked_tab = self.notebook.tk.call(
                self.notebook._w, "identify", "tab", event.x, event.y
            )
            if clicked_tab != '':
                # Switch to clicked tab first
                self.notebook.select(clicked_tab)
                # Show menu at cursor
                self.tab_menu.post(event.x_root, event.y_root)
        except Exception:
            pass

    def _rename_active_tab(self) -> None:
        """Rename the currently active tab."""
        if not self.multi_pane_mode:
            return

        pane = self.active_pane

        # Show dialog to get new name
        from tkinter import simpledialog
        new_name = simpledialog.askstring(
            "Rename Tab",
            "Enter new tab name:",
            initialvalue=pane.title,
            parent=self.root
        )

        if new_name and new_name.strip():
            pane.title = new_name.strip()
            idx = self.active_pane_idx
            self.notebook.tab(idx, text=pane.title)

    def _refresh_active_tab(self) -> None:
        """Force refresh of the active tab."""
        if not self.multi_pane_mode:
            return

        self._update_plot()

    def _refresh_all_tabs(self) -> None:
        """Force refresh of all tabs."""
        if not self.multi_pane_mode:
            return

        # Update active pane immediately
        self._update_plot()

        # Others will update when switched to
