"""
Main application class composed from focused mixins.

This keeps a single public entry point (TimeIntervalLabeler) while letting
implementation details live in small, testable files.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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

logger = logging.getLogger(__name__)


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
        source_name: Optional[str] = None,
        *,
        layout_spec: Optional[Dict[str, Any]] = None,
        panes: Optional[List[Dict[str, Any]]] = None,
        parent: Optional[tk.Misc] = None,
        decimate: bool = True,
    ) -> None:
        # --- Validate inputs ---
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame must have a DatetimeIndex.")

        # Pack 6 R9. An unsorted index used to be accepted in silence and
        # produce nonsense: data_start / data_end are read blind from
        # index[0] / index[-1], so data_end lands BEFORE data_start, t1 <
        # t0, and df.loc[t0:t1] falls back to a boolean mask that returns
        # the WHOLE dataset -- every "window" contains everything. Measured
        # on a fully shuffled 200-row frame: constructor accepted,
        # _build_gui, _update_plot, _next_window and the redraw after it
        # all succeeded, with no exception anywhere.
        #
        # Sorting rather than raising: the quick-start wizard already
        # refuses such a frame (file_loader.py:457), but the EXPORT path
        # handles non-monotonic frames correctly and deliberately
        # (io_export.py:303-324, Pack 5 R7), so raising here would reject
        # data one half of the package gets right. Sorting is what a caller
        # means. Loud, because silently reordering someone's frame is its
        # own surprise -- and once per construction, not per frame, so a
        # plain warning is the right level (Pack 4 doctrine).
        #
        # IDENTITY CONSEQUENCE, and the reason the warning below carries a
        # second sentence. dataset_fingerprint() (io_export.py:38) hashes
        # index[0] and index[-1]; on an unsorted frame those are arbitrary
        # rows, and after the sort they are the true bounds. So the Pack-2
        # autosave FILENAME moves -- measured A/B on one shuffled 200-row
        # frame, chronotagger_autosave_b1179b6c4918.json before and
        # chronotagger_autosave_fb09764d0822.json after -- and Pack 2 made
        # the fingerprinted name the only name _check_autosave consults.
        # An autosave written from the unsorted frame is therefore not
        # offered for recovery after this change. Correct on the merits --
        # the old fingerprint encoded nonsense bounds and those sessions
        # were already windowing over the whole dataset -- but the user has
        # to be TOLD, not left to discover it.
        #
        # Scope: monotonicity only. Duplicate and NaT index entries are NOT
        # in this pack.
        if not df.index.is_monotonic_increasing:
            logger.warning(
                "DataFrame index is not monotonically increasing (%d rows); "
                "sorting by index. Windowing, navigation and per-sample "
                "export all assume ascending time. This also changes the "
                "dataset fingerprint, so an autosave written from the "
                "unsorted frame will not be offered for recovery.",
                len(df.index))
            df = df.sort_index()

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

        # Draw-only decimation (Pack 5 R4b/R11). ON by default: a window
        # holding more samples than the panel has pixels is drawn from an
        # envelope of ORIGINAL rows -- per pixel column, per numeric
        # column, the argmin and argmax rows -- so single-sample spikes
        # survive and nothing is averaged or synthesised. Selection,
        # rules, labeling and export always read the full-resolution
        # frame. decimate=False draws every sample; see the README's
        # known-limitations note for what that buys and costs.
        self.decimate: bool = bool(decimate)

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
        self.max_undo: int = 50  # gesture-level entries; each holds two list snapshots

        # Persistence - Autosave configuration. The filename carries a
        # dataset fingerprint (sorted columns + time bounds + row
        # count), so differently-shaped datasets sharing one folder or
        # CWD do not collide at recovery. Identical-schema datasets over
        # the same window (two spacecraft via one loader) DO share a
        # name -- the source_name check in _check_autosave is the guard
        # for that case. Clean break (Pack 2 grill Q2): pre-fingerprint
        # autosave files are ignored. The fingerprint is fixed at
        # construction; do not add/rename df columns on a live labeler.
        self.source_name: Optional[str] = source_name
        self.autosave_folder = Path(autosave_folder)
        self.autosave_folder.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist
        # Forensic log lives BESIDE the autosave: it inherits the writable-
        # dir guarantee, the suite's CWD isolation, and the examples'
        # gitignored autosave dirs for free (Pack 4 R2/R3). Idempotent.
        from .._logging import configure_file_logging
        self._log_path = configure_file_logging(self.autosave_folder)
        self.autosave_file = (
            self.autosave_folder
            / f"chronotagger_autosave_{self._dataset_fingerprint()}.json"
        )
        self.modified: bool = False

        # GUI state.  When `parent` is provided (e.g. the quick-start wizard),
        # the labeler will mount itself as a tk.Toplevel under that parent so
        # only one tk.Tk root exists per process.  When parent is None, the
        # labeler creates its own tk.Tk root (standalone use).
        self._parent: Optional[tk.Misc] = parent
        self.root: Optional[tk.Misc] = None
        self.notebook: Optional[ttk.Notebook] = None  # Only created if multi_pane_mode

        # Layout & axes metadata.  layout_spec supports a "role" field for each
        # axis area:
        #   - "time":     Time-series data with time on x-axis (box-select
        #                 uses time coords).
        #   - "not-time": Non-time plots like position/phase space (box-select
        #                 maps point order to time).
        # axes_meta mirrors active_pane.axes_meta for legacy access; figures
        # / canvases / per-axis zoom state live on the TabPane and are
        # accessed via the @property delegates below.
        self.axes_meta: Dict[str, Dict[str, Any]] = {}            # key -> {role, row, col, ...}
        self._time_axis_keys: List[str] = []                       # which keys are "time"
        self._primary_time_key: Optional[str] = None               # first time axis in col 0

        # Matplotlib connections
        self.rect_selectors: Dict[str, RectangleSelector] = {}
        self.pick_cid: Optional[int] = None
        self._scroll_cid: Optional[int] = None

        # Drag/resize/move state
        self._drag_mode: Optional[str] = None
        self._drag_iv: Optional[Interval] = None
        self._drag_initial: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
        self._drag_offset: Optional[pd.Timedelta] = None
        self._drag_preview: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
        self._press_cid: Optional[int] = None
        self._motion_cid: Optional[int] = None
        self._release_cid: Optional[int] = None

        # Two-click time selection
        self.two_click_mode: bool = True         # default on; disable to use drag-selector
        self.two_click_auto_add: bool = False    # if True, auto-creates interval on 2nd click

        # Point highlighting (performance optimization)
        self.enable_point_highlighting: bool = True  # disable for large datasets
        for pane in self.panes:
            pane.enable_point_highlighting = self.enable_point_highlighting

        self._pick_anchor_ts: Optional[pd.Timestamp] = None  # first click time
        self._twoclick_motion_cid: Optional[int] = None      # (legacy) preview wire-up
        self._time_click_cid: Optional[int] = None           # mpl connection for clicks
        self._time_motion_cid: Optional[int] = None          # mpl connection for motion

        # Minimal interval duration inference
        try:
            diffs = self.df.index.to_series().diff().dropna()
            med = diffs.median()
            if not isinstance(med, pd.Timedelta) or med <= pd.Timedelta(0):
                med = pd.Timedelta(seconds=1)
            self.min_duration: pd.Timedelta = med
        except Exception:
            # Gates interval validation; a silent 1 s changes the
            # minimum-length rule's meaning (Pack 4 A9).
            # (Pack 6: same logger, same name -- EDIT 185 gave the module
            # one, so the function-local import is redundant.)
            logger.warning(
                "min_duration inference failed; defaulting to 1 s",
                exc_info=True)
            self.min_duration = pd.Timedelta(seconds=1)

        # Multi-span preview state from box-select / rule preview
        self.current_spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        self._commit_spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

        # Click-vs-drag arbitration (pixel slop)
        self.CLICK_DRAG_SLOP_PX: int = 6
        self._press_xy_px: Optional[Tuple[int, int]] = None
        self._dragging_box: bool = False

        # Label-rule policy
        self._overlap_policy: str = "skip"

        # Per-axis zoom state for Y-zoom and cross-plot X/Y zoom.
        # Keyed by matplotlib Axes so they survive plot rebuilds.
        # (Pack 6 R4: a `self._manual_zooms` lived here too, recording
        # which axes the user had zoomed and in which direction across 13
        # sites. Every read in the tree was `if ax not in self._manual_
        # zooms:` immediately before a write; nothing iterated it, tested
        # membership to make a decision, or read the set contents. The two
        # reset paths that look like consumers -- plotting.py:496 and
        # zoom.py:226 -- iterate _auto_ylims / _auto_xlims and reset
        # EVERYTHING, then clear a record they never consulted. Selective
        # reset would be a feature; this was not scaffolding for one.)
        self._auto_xlims: Dict[plt.Axes, Tuple[float, float]] = {}  # Auto X limits for cross-plots
        self._auto_ylims: Dict[plt.Axes, Tuple[float, float]] = {}  # Auto Y limits for all plots
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
                # Restore intervals + the label schema they were made
                # with; invalidate undo history/selection; validate in
                # strict mode; mark modified. The testable core lives in
                # IOExportMixin._apply_recovered_autosave (Pack 2 D1).
                self._apply_recovered_autosave(autosave_data)
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
