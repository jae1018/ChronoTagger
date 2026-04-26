"""
Main application class composed from focused mixins.

This keeps a single public entry point (TimeIntervalLabeler) while letting
implementation details live in small, testable files.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Any

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector

# ABSOLUTE imports from core (moved here)
from chronotagger.core.models import Interval
from chronotagger.core.commands import (
    Command,
    AddIntervalCommand,
    DeleteIntervalCommand,
    RelabelIntervalCommand,
)

# Relative imports to labeler submodules (unchanged)
from .tab_pane import TabPane
from .sync import PaneSyncManager
from .mixins.view_build import ViewBuildMixin
from .mixins.plotting import PlottingMixin
from .mixins.events import EventsMixin
from .mixins.rules import RulesMixin
from .mixins.navigation import NavigationMixin
from .mixins.help import HelpMixin
from .mixins.zoom import ZoomMixin
from .mixins.intervals import IntervalsMixin
from .mixins.stats import StatsMixin
from .mixins.io_export import IOExportMixin
from .mixins.labels import LabelsMixin



class TimeIntervalLabeler(
    ViewBuildMixin,
    PlottingMixin,
    EventsMixin,
    RulesMixin,
    NavigationMixin,
    HelpMixin,
    ZoomMixin,
    IntervalsMixin,
    LabelsMixin,
    StatsMixin,
    IOExportMixin,
):
    
    # Default label colors (Tableau-like)
    DEFAULT_COLORS = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc949", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ac",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        plot_fn: Optional[Callable] = None,
        classes: Optional[List[str]] = None,
        class_colors: Optional[Dict[str, str]] = None,
        window: pd.Timedelta = pd.Timedelta("30min"),
        step: pd.Timedelta = pd.Timedelta("15min"),
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        autosave_folder: str = ".",
        *,
        layout_spec: Optional[Dict[str, Any]] = None,
        panes: Optional[List[Dict[str, Any]]] = None,
        parent: Optional[tk.Misc] = None,
    ) -> None:
        # --- Validate inputs ---
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame must have a DatetimeIndex.")

        # Validate panes vs plot_fn usage
        if panes is not None and plot_fn is not None:
            raise ValueError(
                "Cannot specify both 'panes' and 'plot_fn'. "
                "Use 'panes' for multi-pane mode or 'plot_fn' for single-pane mode."
            )
        if panes is None and plot_fn is None:
            raise ValueError(
                "Must specify either 'panes' (for multi-pane mode) or 'plot_fn' (for single-pane mode)."
            )

        # Determine mode and convert to unified internal representation
        if panes is not None:
            # Multi-pane mode
            self.multi_pane_mode = True
            self._panes_config: List[Dict[str, Any]] = panes
            # Validate that each pane has a plot_fn
            for i, pane in enumerate(self._panes_config):
                if "plot_fn" not in pane or not callable(pane["plot_fn"]):
                    raise ValueError(f"Pane at index {i} must have a callable 'plot_fn'.")
        else:
            # Single-pane mode (backward compatibility)
            if not callable(plot_fn):
                raise TypeError("plot_fn must be callable.")
            self.multi_pane_mode = False
            self._panes_config: List[Dict[str, Any]] = [
                {
                    "title": "Main",
                    "plot_fn": plot_fn,
                    "layout_spec": layout_spec,
                }
            ]

        # Convert config dicts to TabPane objects
        self.panes: List[TabPane] = [
            TabPane(
                title=config["title"],
                plot_fn=config["plot_fn"],
                layout_spec=config.get("layout_spec"),
            )
            for config in self._panes_config
        ]

        # Track active pane
        self.active_pane_idx: int = 0

        # Create sync manager for multi-pane coordination
        self.sync_manager = PaneSyncManager(self)

        # Core data / plotting contract
        self.df = df
        # Note: plot_fn and layout_spec are now properties that delegate to active_pane

        # Label classes & colors
        if classes is None:
            classes = ["UNKNOWN", "label_1", "label_2"]
        self.classes: List[str] = list(classes)

        if class_colors is None:
            class_colors = {
                cls: self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)]
                for i, cls in enumerate(self.classes)
            }
        self.class_colors: Dict[str, str] = dict(class_colors)

        # Time bounds & window
        self.data_start: pd.Timestamp = df.index[0]
        self.data_end: pd.Timestamp = df.index[-1]
        self.window: pd.Timedelta = window
        self.step: pd.Timedelta = step

        if start is None:
            start = self.data_start
        self.t0: pd.Timestamp = max(start, self.data_start)
        self.t1: pd.Timestamp = min(self.t0 + window, self.data_end)

        # Wheel zoom/pan config
        self.zoom_sensitivity: float = 0.20
        self.pan_sensitivity: float = 0.20
        self.min_window: pd.Timedelta = pd.Timedelta("5s")
        self.max_window: pd.Timedelta = self.data_end - self.data_start

        # Intervals & selection
        self.intervals: List[Interval] = []
        self.selected_interval: Optional[Interval] = None
        self.current_selection: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None

        # Undo/redo
        self.undo_stack: List[Command] = []
        self.redo_stack: List[Command] = []
        self.max_undo: int = 20

        # Persistence - Autosave configuration
        self.autosave_folder = Path(autosave_folder)
        self.autosave_folder.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist
        self.autosave_file = self.autosave_folder / "chronotagger_autosave.json"
        self.modified: bool = False

        # GUI state.  When `parent` is provided (e.g. the quick-start wizard),
        # the labeler will mount itself as a tk.Toplevel under that parent so
        # only one tk.Tk root exists per process.  When parent is None, the
        # labeler creates its own tk.Tk root (standalone use).
        self._parent: Optional[tk.Misc] = parent
        self.root: Optional[tk.Misc] = None
        self.notebook: Optional[ttk.Notebook] = None  # Only created if multi_pane_mode

        # === MOVED TO TabPane (will remove after refactoring complete) ===
        # These attributes are now stored per-pane and accessed via properties:
        # self.fig: Optional[plt.Figure] = None
        # self.canvas = None
        # self.user_axes: Dict[str, plt.Axes] = {}
        # self.strip_ax: Optional[plt.Axes] = None
        # self.rect_selectors: Dict[str, RectangleSelector] = {}
        # self._auto_xlims: Dict[plt.Axes, Tuple[float, float]] = {}
        # self._auto_ylims: Dict[plt.Axes, Tuple[float, float]] = {}
        # self._manual_zooms: Dict[plt.Axes, Set[str]] = {}

        # Layout & axes metadata
        # layout_spec supports "role" field for each axis area:
        #   - "time": Time-series data with time on x-axis (box-select uses time coords)
        #   - "not-time": Non-time plots like position/phase space (box-select maps point order to time)
        # Box selection on "time" axes extracts timestamps directly from x-coordinates.
        # Box selection on "not-time" axes uses point order to map back to timestamps.
        # NOTE: layout_spec is now set above from active_pane.layout_spec for backward compatibility
        # NOTE: For now, keeping these on main class until mixins are refactored.
        #       Eventually these will be delegated to active_pane like fig/canvas/user_axes.
        self.axes_meta: Dict[str, Dict[str, Any]] = {}            # key -> {role, row, col, ...}
        self._time_axis_keys: List[str] = []                       # which keys are "time"
        self._primary_time_key: Optional[str] = None               # first time axis in col 0

        # Matplotlib connections
        self.rect_selectors: Dict[str, RectangleSelector] = {}
        self.pick_cid: Optional[int] = None
        self._scroll_cid: Optional[int] = None

        # Drag/resize/move state (unchanged) ...
        self._drag_mode: Optional[str] = None
        self._drag_iv: Optional[Interval] = None
        self._drag_initial: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
        self._drag_offset: Optional[pd.Timedelta] = None
        self._drag_preview: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
        self._press_cid: Optional[int] = None
        self._motion_cid: Optional[int] = None
        self._release_cid: Optional[int] = None
        
        # === Two-click time selection (new) ===
        self.two_click_mode: bool = True         # default on; disable to use drag-selector
        self.two_click_auto_add: bool = False    # if True, auto-creates interval on 2nd click

        # === Point highlighting (performance optimization) ===
        self.enable_point_highlighting: bool = True  # disable for large datasets

        # Sync highlighting state to all panes (for multi-pane mode)
        for pane in self.panes:
            pane.enable_point_highlighting = self.enable_point_highlighting

        self._pick_anchor_ts: Optional[pd.Timestamp] = None  # first click time
        self._twoclick_motion_cid: Optional[int] = None      # (legacy) preview wire-up
        self._time_click_cid: Optional[int] = None           # mpl connection for clicks
        self._time_motion_cid: Optional[int] = None          # mpl connection for motion

        # Minimal interval duration inference (unchanged) ...
        try:
            diffs = self.df.index.to_series().diff().dropna()
            med = diffs.median()
            if not isinstance(med, pd.Timedelta) or med <= pd.Timedelta(0):
                med = pd.Timedelta(seconds=1)
            self.min_duration: pd.Timedelta = med
        except Exception:
            self.min_duration = pd.Timedelta(seconds=1)
            
         # Selection state
        self.current_selection: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
        # NEW: multiple preview spans from box-select (each becomes an interval)
        self.current_spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        self._commit_spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

        # --- Two-click time selection (existing) ---
        self.two_click_mode: bool = True

        # click-vs-drag arbitration (pixel slop)
        self.CLICK_DRAG_SLOP_PX: int = 6
        self._press_xy_px: Optional[Tuple[int, int]] = None
        self._dragging_box: bool = False
        
        # label-rule policy
        self._overlap_policy: str = "skip"   # v1 default for rule-based adds

        # === Axis zoom state (for Y-zoom and cross-plot X/Y zoom) ===
        # NOTE: These are stored per-pane in TabPane, but kept here temporarily
        # until mixins are refactored in Phase 2
        self._auto_xlims: Dict[plt.Axes, Tuple[float, float]] = {}  # Auto X limits for cross-plots
        self._auto_ylims: Dict[plt.Axes, Tuple[float, float]] = {}  # Auto Y limits for all plots
        self._manual_zooms: Dict[plt.Axes, Set[str]] = {}           # Track manual zoom: {ax: {'x', 'y'}}
        self._time_range_dirty: bool = False                        # Flag: time range changed

    # -------- Pane configuration properties --------

    @property
    def active_pane_config(self) -> Dict[str, Any]:
        """Return the configuration dictionary for the currently active pane."""
        return self._panes_config[self.active_pane_idx]

    @property
    def active_pane(self) -> TabPane:
        """Get the currently active TabPane object."""
        return self.panes[self.active_pane_idx]

    # -------- Backward compatibility delegation properties --------
    # These delegate to the active pane so existing code continues to work

    @property
    def fig(self) -> Optional[plt.Figure]:
        """Delegate to active pane for backward compatibility."""
        return self.active_pane.fig

    @fig.setter
    def fig(self, value: Optional[plt.Figure]) -> None:
        """Set fig on active pane."""
        self.active_pane.fig = value

    @property
    def canvas(self) -> Optional[FigureCanvasTkAgg]:
        """Delegate to active pane for backward compatibility."""
        return self.active_pane.canvas

    @canvas.setter
    def canvas(self, value: Optional[FigureCanvasTkAgg]) -> None:
        """Set canvas on active pane."""
        self.active_pane.canvas = value

    @property
    def user_axes(self) -> Dict[str, plt.Axes]:
        """Delegate to active pane for backward compatibility."""
        return self.active_pane.user_axes

    @user_axes.setter
    def user_axes(self, value: Dict[str, plt.Axes]) -> None:
        """Set user_axes on active pane."""
        self.active_pane.user_axes = value

    @property
    def strip_ax(self) -> Optional[plt.Axes]:
        """Delegate to active pane for backward compatibility."""
        return self.active_pane.strip_ax

    @strip_ax.setter
    def strip_ax(self, value: Optional[plt.Axes]) -> None:
        """Set strip_ax on active pane."""
        self.active_pane.strip_ax = value

    @property
    def plot_fn(self) -> Callable:
        """Delegate to active pane for backward compatibility."""
        return self.active_pane.plot_fn

    @property
    def layout_spec(self) -> Optional[Dict[str, Any]]:
        """Delegate to active pane for backward compatibility."""
        return self.active_pane.layout_spec

    # Note: axes_meta and rect_selectors are kept as regular attributes for now
    # until mixins are refactored in Phase 2

    # -------- Public entrypoint --------

    def run(self) -> None:
        """Start the Tkinter main loop with autosave recovery."""
        # Build GUI first
        self._build_gui()

        # Check for autosave BEFORE starting mainloop
        autosave_data = self._check_autosave()

        if autosave_data is not None:
            choice = self._show_recovery_dialog(autosave_data)

            if choice == 'recover':
                # Load intervals from autosave
                self.intervals = autosave_data['intervals']
                # Sync intervals across all panes
                self.sync_manager.sync_intervals_changed()
                # Refresh UI to show loaded intervals
                self._update_plot()
                if hasattr(self, '_update_intervals_list'):
                    self._update_intervals_list()
                self.status_var.set(f"Recovered {len(self.intervals)} intervals from autosave")

            elif choice == 'start_fresh':
                # Don't load autosave, keep empty intervals
                # Autosave file remains for potential future recovery
                self.status_var.set("Starting fresh session (autosave not loaded)")

            elif choice == 'save_backup':
                # Already handled in dialog callback
                self.status_var.set("Starting fresh session (backup saved)")

            elif choice == 'cancel':
                # User wants to exit
                self.root.destroy()
                return  # Exit without starting mainloop

        # Update plot and start GUI event loop.
        # When the labeler is a child Toplevel (launched from the wizard),
        # mainloop() is already running on the parent's Tk.  We block on the
        # Toplevel via wait_window() instead so labeler.run() returns when
        # the user closes the labeler, but the parent's mainloop keeps going.
        self._update_plot()
        if isinstance(self.root, tk.Tk):
            self.root.mainloop()  # type: ignore[union-attr]
        else:
            self.root.wait_window()  # type: ignore[union-attr]
