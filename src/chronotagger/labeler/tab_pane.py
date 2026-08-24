"""
TabPane data structure for multi-pane interface.

Each TabPane represents a single tab/pane with its own figure and plotting function,
while sharing interval/label state with other panes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional, Tuple

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
    _blit: Optional[Any] = None  # BlitHelper for fast rendering (initialized in canvas.py)

    # Axes metadata (populated during layout building)
    axes_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # List (not set) so iteration order matches the order time areas
    # appear in layout_spec.areas. See canvas.py for context.
    time_axis_keys: List[str] = field(default_factory=list)
    primary_time_key: Optional[str] = None

    # Zoom state (per-pane, allows independent zoom on each tab)
    # (Pack 6 R4: manual_zooms was declared here and referenced exactly
    # once in the whole repo -- this declaration. Pack 6 R8: `dirty` and
    # `last_window` drove a lazy-update system whose mark_clean() had zero
    # production callers, so `dirty` was never cleared and needs_update()
    # could never return False. The state that DOES drive reset behaviour
    # is auto_xlims / auto_ylims below, read at plotting.py:501-506 and
    # zoom.py:232-237.)
    auto_xlims: Dict[plt.Axes, Tuple[float, float]] = field(default_factory=dict)
    auto_ylims: Dict[plt.Axes, Tuple[float, float]] = field(default_factory=dict)

    # Event connections (matplotlib cids for this pane)
    # NOTE: this one IS populated (canvas.py:317/336) -- unlike the
    # same-named dict on the labeler. See canvas.py.
    rect_selectors: Dict[str, Any] = field(default_factory=dict)
