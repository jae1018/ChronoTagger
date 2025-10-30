"""
Main application class composed from focused mixins.

This keeps a single public entry point (TimeIntervalLabeler) while letting
implementation details live in small, testable files.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
from .mixins.navigation import NavigationMixin
from .mixins.intervals import IntervalsMixin
from .mixins.stats import StatsMixin
from .mixins.io_export import IOExportMixin



class TimeIntervalLabeler(
    ViewBuildMixin,
    PlottingMixin,
    EventsMixin,
    NavigationMixin,
    IntervalsMixin,
    StatsMixin,
    IOExportMixin,
):
    """
    Tkinter + Matplotlib app for labeling time intervals in time-series.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with a DatetimeIndex covering the entire dataset.
    plot_fn : Callable[[dict, pd.DataFrame, pd.Timestamp, pd.Timestamp], None]
        User-supplied function that draws panels into provided axes dict.
        Signature: `plot_fn(axs, df, t0, t1)`. The df is already sliced [t0, t1].
    classes : list[str], optional
        Label class names.
    class_colors : dict[str, str], optional
        Mapping class->hex color. If None, a default palette is used.
    window : pd.Timedelta, optional
        Initial visible window duration (default: 30 minutes).
    step : pd.Timedelta, optional
        Navigation step (default: 15 minutes).
    start : pd.Timestamp, optional
        Initial window start (defaults to df.index[0]).
    end : pd.Timestamp, optional
        Data end boundary (defaults to df.index[-1]).
    autosave_path : str | Path, optional
        If provided, autosaves after each modification.
    """

    # Tableau 10 palette
    DEFAULT_COLORS = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        plot_fn: Callable,
        n_panels: Optional[int] = None,
        classes: Optional[List[str]] = None,
        class_colors: Optional[Dict[str, str]] = None,
        window: pd.Timedelta = pd.Timedelta("30min"),
        step: pd.Timedelta = pd.Timedelta("15min"),
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        autosave_path: Optional[str] = None,
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

        # GUI state (set by view_build)
        self.root: Optional[tk.Tk] = None
        self.fig: Optional[plt.Figure] = None
        self.canvas = None  # FigureCanvasTkAgg
        self.user_axes: Dict[str, plt.Axes] = {}
        self.strip_ax: Optional[plt.Axes] = None
        self.rect_selector = None  # RectangleSelector
        self.pick_cid: Optional[int] = None
        # Panels (None means "resolve automatically")
        self.n_panels = n_panels

        # Widgets we update later (set by view_build)
        self.start_time_entry = None
        self.end_time_entry = None
        self.step_entry = None
        self.current_class_var = None
        self.class_combo = None
        self.intervals_tree = None
        self.stats_text = None
        self.snap_var = None
        self.status_var = None

    # -------- Public entrypoint --------

    def run(self) -> None:
        """Start the Tkinter main loop."""
        self._build_gui()
        self._update_plot()
        self.root.mainloop()  # type: ignore[union-attr]
