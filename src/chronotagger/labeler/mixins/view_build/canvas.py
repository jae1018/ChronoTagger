"""
Matplotlib canvas construction and interaction mixin.

Responsibilities:
- Build the matplotlib figure and axes based on layout specification
- Configure canvas event handlers (clicks, motion, selections)
- Set up rectangle selectors with edge clamping
- Initialize blitting for fast rendering

This mixin provides the core plotting infrastructure for the labeler.
"""

from __future__ import annotations

from typing import Dict
import tkinter as tk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector


class CanvasMixin:
    """
    Mixin providing matplotlib canvas construction and interaction.

    This mixin expects the following attributes/methods to be available on self:
    - layout_spec: dict - Layout specification defining grid and areas
    - root: tk.Tk - The main tkinter root window
    - _apply_time_axis_format(ax) - Method to format time axes
    - _hook_time_xlim() - Method to sync xlim with t0/t1
    - _on_scroll_zoom(event) - Callback for scroll zoom
    - _on_time_click(event) - Callback for time axis clicks
    - _on_time_motion(event) - Callback for time axis motion
    - _init_time_overlays() - Method to initialize time overlays (if available)
    - _on_rectangle_select(eclick, erelease) - Callback for rectangle selection
    - _on_strip_click(event) - Callback for strip clicks (pick events)
    - _on_strip_press(event) - Callback for strip press events
    - _on_strip_motion(event) - Callback for strip motion events
    - _on_strip_release(event) - Callback for strip release events
    - _on_right_click_cancel(event) - Callback for right-click cancellation
    - _on_rect_selector_press(event) - Callback for rectangle selector press
    - _on_rect_selector_release(event) - Callback for rectangle selector release
    - _on_rect_selector_motion(event) - Callback for rectangle selector motion

    Attributes created:
    - fig: plt.Figure - The matplotlib figure
    - user_axes: dict[str, plt.Axes] - Dictionary of user-defined axes
    - axes_meta: dict[str, dict] - Metadata for each axis (role, row, col, etc.)
    - _time_axis_keys: set[str] - Set of keys for time axes
    - _primary_time_key: str - Key of the primary time axis
    - strip_ax: plt.Axes - The labels strip axis
    - canvas: FigureCanvasTkAgg - The tkinter canvas widget
    - _blit: BlitHelper - Blitting helper for fast rendering
    - _scroll_cid: int - Connection ID for scroll events
    - _time_click_cid: int - Connection ID for time click events
    - _time_motion_cid: int - Connection ID for time motion events
    - rect_selectors: dict[str, RectangleSelector] - Rectangle selectors for each axis
    - pick_cid: int - Connection ID for pick events
    - _press_cid: int - Connection ID for press events
    - _motion_cid: int - Connection ID for motion events
    - _release_cid: int - Connection ID for release events
    - _right_click_cid: int - Connection ID for right-click events
    - _gate_press_cid: int - Connection ID for gate press events
    - _gate_release_cid: int - Connection ID for gate release events
    - _rect_clamp_motion_cid: int - Connection ID for rectangle clamp motion
    - _rect_clamp_press_cid: int - Connection ID for rectangle clamp press
    - _rect_clamp_release_cid: int - Connection ID for rectangle clamp release
    - _drag_active: bool - Whether a drag is currently active
    - _press_event: event - The press event that started a potential drag
    - _rect_drag_axes: plt.Axes or None - The axes where rectangle drag is active

    Methods provided:
    - _build_plot(parent) - Build the matplotlib figure and axes
    - _is_time_axes(ax) - Check if an axis is a time axis
    - _find_labels_area(areas) - Find the Labels panel definition from areas
    - _gate_press(event) - Remember where a potential drag started
    - _gate_release(event) - Mark end of drag/click cycle
    - _setup_rectangle_edge_clamping() - Wire up edge-clamping for rectangles
    - _on_tk_canvas_motion(tk_event) - Handle tkinter motion events
    """

    def _build_plot(self, parent) -> None:
        """
        Build the Matplotlib figure and axes.

        Grid mode only: user-defined grid with Labels panel included in layout_spec.
        The Labels strip position is read from layout_spec (role='labels').
        """
        import matplotlib.pyplot as plt

        # ========== GRID MODE ==========
        if not isinstance(self.layout_spec, dict):
            raise ValueError("grid-only mode requires layout_spec")

        if isinstance(self.layout_spec, dict):
            spec = self.layout_spec
            nrows: int = int(spec.get("nrows", 1))  # No +1 - Labels is already included
            ncols: int = int(spec.get("ncols", 1))
            areas = list(spec.get("areas", []))
            width_ratios = spec.get("width_ratios", None)
            height_ratios = spec.get("height_ratios", None)
            hspace = float(spec.get("hspace", 0.12))
            wspace = float(spec.get("wspace", 0.04))

            if not areas:
                raise ValueError("layout_spec.areas must be a non-empty list.")

            # Validate: at least one time axis exists
            if not any(str(a.get("role", "time")).lower() == "time" for a in areas):
                raise ValueError("layout_spec must have at least one role='time' axis.")

            # Find Labels panel
            labels_area = self._find_labels_area(areas)
            if labels_area is None:
                raise ValueError("layout_spec missing Labels panel (role='labels'). "
                               "Ensure layout was created with Layout Wizard.")

            # height ratios
            if height_ratios is None:
                hrs = [1.0] * nrows
            else:
                if len(height_ratios) != nrows:
                    raise ValueError("layout_spec.height_ratios must have length == nrows")
                hrs = list(map(float, height_ratios))

            # Lane gutter behavior unchanged (kept off unless user provides it)
            use_lane_gutter = isinstance(spec.get("time_lane_cbar_gutter", None), dict)
            use_constrained = not use_lane_gutter

            self.fig = plt.Figure(figsize=(14, 8), constrained_layout=use_constrained)
            gs = self.fig.add_gridspec(
                nrows, ncols,  # Use nrows directly - no +1
                width_ratios=width_ratios, height_ratios=hrs,
                hspace=hspace, wspace=wspace,
            )

            # Build data axes from user-specified areas (skip Labels - handled separately)
            self.user_axes = {}
            self.axes_meta = {}
            self._time_axis_keys = set()
            self._primary_time_key = None

            for a in areas:
                key = str(a["key"])
                role = str(a.get("role", "time")).lower()

                # Skip Labels panel - will be created separately as strip_ax
                if role == "labels":
                    continue

                row = int(a.get("row", 0))
                col = int(a.get("col", 0))
                rowspan = int(a.get("rowspan", 1))
                colspan = int(a.get("colspan", 1))

                if row < 0 or row >= nrows or col < 0 or col >= ncols:
                    raise ValueError(f"Area {key} has out-of-bounds row/col.")

                ax = self.fig.add_subplot(gs[row:row+rowspan, col:col+colspan])
                self.user_axes[key] = ax
                self.axes_meta[key] = {"role": role, "row": row, "col": col,
                                       "rowspan": rowspan, "colspan": colspan}
                if role == "time":
                    self._time_axis_keys.add(key)
                    if self._primary_time_key is None:
                        self._primary_time_key = key

            # Ensure we have a primary time axis
            if self._primary_time_key is None and self._time_axis_keys:
                self._primary_time_key = next(iter(self._time_axis_keys))

            # Share x among time axes
            if self._primary_time_key is not None:
                primary_ax = self.user_axes[self._primary_time_key]
                for k in self._time_axis_keys:
                    if k != self._primary_time_key:
                        self.user_axes[k].sharex(primary_ax)

            # Create Labels strip at position specified in layout_spec
            labels_row = int(labels_area.get("row", nrows - 1))
            labels_col = int(labels_area.get("col", 0))
            labels_colspan = int(labels_area.get("colspan", 1))

            self.strip_ax = self.fig.add_subplot(gs[
                labels_row,
                labels_col:labels_col + labels_colspan
            ])
            self.strip_ax.set_ylabel("Labels", fontsize=9)
            self.strip_ax.set_ylim(0, 1)
            self.strip_ax.set_yticks([])
            self._apply_time_axis_format(self.strip_ax)

            # --- Make the strip share x with the primary time axis (hard lock) ---
            if self._primary_time_key is not None:
                primary_ax = self.user_axes[self._primary_time_key]
                # sharex makes them part of the same shared group
                self.strip_ax.sharex(primary_ax)

            # Time axis formatting for all time panels
            for k in self._time_axis_keys:
                self._apply_time_axis_format(self.user_axes[k])

            # Keep toolbar zoom/pan in sync with t0/t1 and the rest of the panels
            self._hook_time_xlim()

            # Embed in Tk
            self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            toolbar = NavigationToolbar2Tk(self.canvas, parent)  # noqa: F841
            toolbar.update()

            # ── Blitting: cache per-axes backgrounds and keep them fresh ────────────────
            from ...utils.fastdraw import BlitHelper
            self._blit = BlitHelper(self.fig, self.canvas)
            _axes_for_blit = [self.user_axes[k] for k in (self._time_axis_keys or []) if k in self.user_axes]
            if self.strip_ax is not None:
                _axes_for_blit.append(self.strip_ax)
            self._blit.add_axes(_axes_for_blit)
            self.canvas.mpl_connect("draw_event", self._blit.recache)
            # ───────────────────────────────────────────────────────────────────────────

            self.root.after(0, self.canvas.draw_idle)


            # Wheel zoom/pan
            if getattr(self, "_scroll_cid", None) is None:
                self._scroll_cid = self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)

            # Two-click selection wiring (coexists with drag-rectangle)
            if self._time_click_cid is None:
                self._time_click_cid = self.canvas.mpl_connect(
                    "button_release_event", self._on_time_click
                )
            if self._time_motion_cid is None:
                self._time_motion_cid = self.canvas.mpl_connect(
                    "motion_notify_event", self._on_time_motion
                )
            if hasattr(self, "_init_time_overlays"):
                self._init_time_overlays()

            # Rectangle selectors on **ALL** user axes (time and not-time)
            # This allows box selection on both time-series and position plots
            self.rect_selectors = {}
            for k in sorted(self.user_axes.keys()):
                ax = self.user_axes[k]
                rs = RectangleSelector(
                    ax,
                    onselect=self._on_rectangle_select,
                    useblit=True,
                    button=[1],
                    minspanx=5, minspany=5,
                    spancoords="pixels",
                    interactive=False,
                    props=dict(
                        facecolor="yellow",
                        edgecolor="orange",
                        alpha=0.3,
                        linestyle="--",
                        linewidth=2,
                    ),
                )
                self.rect_selectors[k] = rs

            # Wire up edge-clamping for rectangle selectors
            self._setup_rectangle_edge_clamping()

            # Strip interactions
            if self.pick_cid is None:
                self.pick_cid = self.canvas.mpl_connect("pick_event", self._on_strip_click)
            if self._press_cid is None:
                self._press_cid = self.canvas.mpl_connect("button_press_event", self._on_strip_press)
            if self._motion_cid is None:
                self._motion_cid = self.canvas.mpl_connect("motion_notify_event", self._on_strip_motion)
            if self._release_cid is None:
                self._release_cid = self.canvas.mpl_connect("button_release_event", self._on_strip_release)

            # Right-click cancellation (works on any axis)
            if not hasattr(self, '_right_click_cid') or self._right_click_cid is None:
                self._right_click_cid = self.canvas.mpl_connect("button_press_event", self._on_right_click_cancel)

            # --- Drag gate: discriminate drag vs click so click1-click2 doesn't steal events
            if getattr(self, "_gate_press_cid", None) is None:
                self._gate_press_cid = self.canvas.mpl_connect("button_press_event", self._gate_press)
            if getattr(self, "_gate_release_cid", None) is None:
                self._gate_release_cid = self.canvas.mpl_connect("button_release_event", self._gate_release)

            return

    def _is_time_axes(self, ax) -> bool:
        return any(ax is self.user_axes[k] for k in self._time_axis_keys)

    def _find_labels_area(self, areas: list) -> dict | None:
        """
        Find and return the Labels panel definition from layout areas.

        Args:
            areas: List of area dictionaries from layout_spec

        Returns:
            The Labels area dict if found, None otherwise
        """
        for area in areas:
            if str(area.get("role", "")).lower() == "labels":
                return area
        return None

    def _gate_press(self, event):
        # Remember where a potential drag started (time panels only; LMB)
        self._drag_active = False
        if event.button == 1 and event.inaxes is not None and self._is_time_axes(event.inaxes):
            self._press_event = event
        else:
            self._press_event = None

    def _gate_release(self, event):
        # Drag/click cycle ended
        self._drag_active = False
        self._press_event = None

    # ========== Rectangle Selector Edge Clamping Setup ==========

    def _setup_rectangle_edge_clamping(self) -> None:
        """
        Wire up edge-clamping behavior for all rectangle selectors.

        This connects press/motion/release callbacks to enable the rectangle
        to extend to axes edges when the mouse leaves the axes during a drag.
        Called once during plot setup after all rectangle selectors are created.
        """
        # Connect press/release callbacks to track drag state
        # These work across all axes since we store which axes was pressed
        for key, rs in self.rect_selectors.items():
            # Use the selector's internal callback mechanism
            # Store original callbacks if they exist
            original_press = getattr(rs, '_on_press_callback', None)
            original_release = getattr(rs, '_on_release_callback', None)

            # Wrap to call both our tracking and any existing callbacks
            def make_press_wrapper(orig_cb):
                def wrapper(event):
                    self._on_rect_selector_press(event)
                    if orig_cb is not None:
                        orig_cb(event)
                return wrapper

            def make_release_wrapper(orig_cb):
                def wrapper(event):
                    self._on_rect_selector_release(event)
                    if orig_cb is not None:
                        orig_cb(event)
                return wrapper

            # Note: RectangleSelector doesn't expose press/release callbacks directly,
            # so we connect to the figure-level events and track state manually

        # Connect figure-level motion handler for edge clamping
        # This runs on ALL motion events, but only acts during active drags
        if not hasattr(self, '_rect_clamp_motion_cid') or self._rect_clamp_motion_cid is None:
            self._rect_clamp_motion_cid = self.canvas.mpl_connect(
                'motion_notify_event',
                self._on_rect_selector_motion
            )

        # Connect figure-level press/release for state tracking
        if not hasattr(self, '_rect_clamp_press_cid') or self._rect_clamp_press_cid is None:
            self._rect_clamp_press_cid = self.canvas.mpl_connect(
                'button_press_event',
                self._on_rect_selector_press
            )

        if not hasattr(self, '_rect_clamp_release_cid') or self._rect_clamp_release_cid is None:
            self._rect_clamp_release_cid = self.canvas.mpl_connect(
                'button_release_event',
                self._on_rect_selector_release
            )

        # CRITICAL: Add tkinter-level motion binding
        # This captures motion EVERYWHERE on canvas (including figure background)
        # matplotlib events only fire when mouse is over axes, missing the gray areas
        tk_widget = self.canvas.get_tk_widget()
        tk_widget.bind('<Motion>', self._on_tk_canvas_motion, add='+')

    def _on_tk_canvas_motion(self, tk_event) -> None:
        """
        Handle tkinter motion events (fires everywhere on canvas).

        This handler captures mouse motion over the entire canvas, including
        the figure background (gray areas between plots) where matplotlib's
        motion_notify_event doesn't fire. Essential for smooth edge-clamping
        when dragging rectangles outside axes bounds.

        Args:
            tk_event: tkinter motion event
        """
        # Only process during active rectangle drag
        if not hasattr(self, '_rect_drag_axes') or self._rect_drag_axes is None:
            return

        try:
            # Convert tkinter coordinates to matplotlib figure coordinates
            # Tkinter: origin at top-left, y increases downward
            # Matplotlib: origin at bottom-left, y increases upward

            fig_x = tk_event.x  # X is the same
            fig_y = self.fig.bbox.height - tk_event.y  # Flip Y axis

            # Create a pseudo matplotlib event
            # We only need x, y (figure coords) and inaxes (None since we're outside)
            class PseudoMplEvent:
                """Minimal event object compatible with matplotlib event API."""
                def __init__(self, x, y, inaxes=None):
                    self.x = x
                    self.y = y
                    self.inaxes = inaxes

            pseudo_event = PseudoMplEvent(fig_x, fig_y, inaxes=None)

            # Call our existing matplotlib motion handler
            # It will handle the edge-clamping logic
            self._on_rect_selector_motion(pseudo_event)

        except Exception:
            # Silently fail - better to have no update than crash
            # This can happen if figure geometry is not yet initialized
            pass
