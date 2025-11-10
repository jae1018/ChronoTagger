"""
TabPane data structure for multi-pane interface.

Each TabPane represents a single tab/pane with its own figure and plotting function,
while sharing interval/label state with other panes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, Set, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


@dataclass
class TabPane:
    """
    Represents a single tab/pane in the multi-pane interface.

    Each pane has its own Figure, Canvas, and plot function,
    but shares interval/label state with other panes.
    """

    # User-provided configuration (required at init)
    title: str
    plot_fn: Callable
    layout_spec: Optional[Dict[str, Any]] = None

    # Matplotlib objects (created during GUI build, initially None)
    fig: Optional[plt.Figure] = None
    canvas: Optional[FigureCanvasTkAgg] = None
    user_axes: Dict[str, plt.Axes] = field(default_factory=dict)
    strip_ax: Optional[plt.Axes] = None

    # Axes metadata (populated during layout building)
    axes_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    time_axis_keys: Set[str] = field(default_factory=set)
    primary_time_key: Optional[str] = None

    # Rendering state (for lazy updates)
    dirty: bool = True  # Needs redraw?
    last_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None

    # Zoom state (per-pane, allows independent zoom on each tab)
    auto_xlims: Dict[plt.Axes, Tuple[float, float]] = field(default_factory=dict)
    auto_ylims: Dict[plt.Axes, Tuple[float, float]] = field(default_factory=dict)
    manual_zooms: Dict[plt.Axes, Set[str]] = field(default_factory=dict)

    # Event connections (matplotlib cids for this pane)
    rect_selectors: Dict[str, Any] = field(default_factory=dict)

    def needs_update(self, t0: pd.Timestamp, t1: pd.Timestamp) -> bool:
        """
        Check if this pane needs replotting for the given time window.

        Returns True if:
        - Pane is marked dirty, OR
        - Time window changed since last render
        """
        if self.dirty:
            return True
        if self.last_window is None:
            return True
        return self.last_window != (t0, t1)

    def mark_clean(self, t0: pd.Timestamp, t1: pd.Timestamp) -> None:
        """Mark pane as up-to-date for the given time window."""
        self.dirty = False
        self.last_window = (t0, t1)

    def mark_dirty(self) -> None:
        """Mark pane as needing redraw (e.g., intervals changed)."""
        self.dirty = True
