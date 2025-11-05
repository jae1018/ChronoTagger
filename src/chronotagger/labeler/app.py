"""
Main application class composed from focused mixins.

This keeps a single public entry point (TimeIntervalLabeler) while letting
implementation details live in small, testable files.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any, Set

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
        plot_fn: Callable,
        classes: Optional[List[str]] = None,
        class_colors: Optional[Dict[str, str]] = None,
        window: pd.Timedelta = pd.Timedelta("30min"),
        step: pd.Timedelta = pd.Timedelta("15min"),
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        autosave_path: Optional[str] = None,
        *,
        layout_spec: Optional[Dict[str, Any]] = None,
    ) -> None:
        # --- Validate inputs ---
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame must have a DatetimeIndex.")
        if not callable(plot_fn):
            raise TypeError("plot_fn must be callable.")

        # Core data / plotting contract
        self.df = df
        self.plot_fn = plot_fn

        # Label classes & colors
        if classes is None:
            classes = ["PlasmaSheet", "Lobe", "Magnetosheath", "SolarWind", "UNKNOWN"]
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

        # Persistence
        self.autosave_path = Path(autosave_path) if autosave_path else None
        self.modified: bool = False

        # GUI state
        self.root: Optional[tk.Tk] = None
        self.fig: Optional[plt.Figure] = None
        self.canvas = None
        self.user_axes: Dict[str, plt.Axes] = {}
        self.strip_ax: Optional[plt.Axes] = None

        # Layout & axes metadata
        # layout_spec supports "role" field for each axis area:
        #   - "time": Time-series data with time on x-axis (box-select uses time coords)
        #   - "not-time": Non-time plots like position/phase space (box-select maps point order to time)
        # Box selection on "time" axes extracts timestamps directly from x-coordinates.
        # Box selection on "not-time" axes uses point order to map back to timestamps.
        self.layout_spec: Optional[Dict[str, Any]] = layout_spec
        self.axes_meta: Dict[str, Dict[str, Any]] = {}            # key -> {role, row, col, ...}
        self._time_axis_keys: Set[str] = set()                     # which keys are "time"
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
        self._auto_xlims: Dict[plt.Axes, Tuple[float, float]] = {}  # Auto X limits for cross-plots
        self._auto_ylims: Dict[plt.Axes, Tuple[float, float]] = {}  # Auto Y limits for all plots
        self._manual_zooms: Dict[plt.Axes, Set[str]] = {}           # Track manual zoom: {ax: {'x', 'y'}}
        self._time_range_dirty: bool = False                        # Flag: time range changed

    # -------- Public entrypoint --------

    def run(self) -> None:
        """Start the Tkinter main loop."""
        self._build_gui()
        self._update_plot()
        self.root.mainloop()  # type: ignore[union-attr]
