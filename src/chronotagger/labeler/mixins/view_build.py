"""
View construction mixin.

Responsibilities:
- Build the Tk window and main frames
- Build the top controls, plot area, sidebar

Note that the module assumes matplotlib 3.8+ is used.
"""

from __future__ import annotations

from typing import Dict
import tkinter as tk
from tkinter import ttk

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector


class ViewBuildMixin:
    # Expects on self:
    # - attributes initialized in app.__init__
    # - _apply_time_axis_format (from PlottingMixin)
    # - _build_top_controls, _build_plot, _build_sidebar (here)

    def _build_gui(self) -> None:
        self.root = tk.Tk()
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
        
        self.plot_frame = ttk.Frame(main)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_plot(self.plot_frame)

        self.root.bind("<Key>", self._on_key_press)
        # F1 opens Help
        self.root.bind("<F1>", self._open_help_dialog)
        # F9 toggles sidebar visibility
        self.root.bind("<F9>", lambda e: self._toggle_sidebar())

    def _build_top_controls(self, parent: ttk.Frame) -> None:
        from tkinter import filedialog  # ensure Tk is initialized on Windows
        _ = filedialog
    
        # ── Time Range (two-row layout with ×2/÷2 buttons) ──────────────────────────
        rng = ttk.LabelFrame(parent, text="Time Range", padding=5)
        rng.pack(side=tk.LEFT, padx=5)
    
        # Make the entry column stretchy
        rng.grid_columnconfigure(1, weight=1)
    
        # Row 0: Start + Update Window button
        ttk.Label(rng, text="Start:").grid(row=0, column=0, padx=(2, 6), pady=2, sticky="w")
        self.start_time_entry = ttk.Entry(rng, width=22)
        self.start_time_entry.insert(0, str(self.t0))
        self.start_time_entry.grid(row=0, column=1, padx=(0, 6), pady=2, sticky="ew")
        ttk.Button(rng, text="Update Window", command=self._update_time_window)\
            .grid(row=0, column=2, padx=(8, 2), pady=2)
    
        # Row 1: End + ×2/÷2 buttons
        ttk.Label(rng, text="End:").grid(row=1, column=0, padx=(2, 6), pady=2, sticky="w")
        self.end_time_entry = ttk.Entry(rng, width=22)
        self.end_time_entry.insert(0, str(self.t1))
        self.end_time_entry.grid(row=1, column=1, padx=(0, 6), pady=2, sticky="ew")
        
        # ×2/÷2 buttons in a small frame
        win_btn_frame = ttk.Frame(rng)
        win_btn_frame.grid(row=1, column=2, padx=(8, 2), pady=2)
        ttk.Button(win_btn_frame, text="×2", width=4, command=self._double_time_window).pack(side=tk.LEFT, padx=2)
        ttk.Button(win_btn_frame, text="÷2", width=4, command=self._halve_time_window).pack(side=tk.LEFT, padx=2)
    
        # ── Navigation (centered) ─────────────────────────────────────────────────
        nav = ttk.LabelFrame(parent, text="Navigation", padding=5)
        nav.pack(side=tk.LEFT, padx=8)
    
        # Create grid frame for consistent alignment with other sections
        nav_grid = ttk.Frame(nav)
        nav_grid.pack()
        
        # Configure grid for center alignment
        nav_grid.grid_columnconfigure(0, weight=1)
        nav_grid.grid_columnconfigure(1, weight=0)
        nav_grid.grid_columnconfigure(2, weight=1)
    
        # Row 0: Previous and Next buttons (aligned with other sections' top rows)
        prev_next_frame = ttk.Frame(nav_grid)
        prev_next_frame.grid(row=0, column=1, pady=2)
        ttk.Button(prev_next_frame, text="<- Prev", command=self._prev_window).pack(side=tk.LEFT, padx=4)
        ttk.Button(prev_next_frame, text="Next ->", command=self._next_window).pack(side=tk.LEFT, padx=4)
    
        # Row 1: Step controls (aligned with other sections' bottom rows)
        step_frame = ttk.Frame(nav_grid)
        step_frame.grid(row=1, column=1, pady=2)
        ttk.Label(step_frame, text="Step:").pack(side=tk.LEFT, padx=(0, 6))
        self.step_entry = ttk.Entry(step_frame, width=16)
        self.step_entry.insert(0, str(self.step))
        self.step_entry.pack(side=tk.LEFT)
        self.step_entry.bind("<Return>", lambda _: self._apply_step_entry())
        ttk.Button(step_frame, text="x2", width=4, command=self._double_step).pack(side=tk.LEFT, padx=4)
        ttk.Button(step_frame, text="/2", width=4, command=self._halve_step).pack(side=tk.LEFT, padx=2)
    
        # ── Label Actions (2x5 grid) ──────────────────────────────────────────────
        label_actions = ttk.LabelFrame(parent, text="Label Actions", padding=5)
        label_actions.pack(side=tk.LEFT, padx=8)
        
        # Create grid frame for 2x5 button layout
        grid_frame = ttk.Frame(label_actions)
        grid_frame.pack()
        
        # Configure grid columns to have equal weight
        for i in range(5):
            grid_frame.columnconfigure(i, weight=1)
        
        # Row 1: [Re-label, Add, Undo, Manage..., Fill Gaps...]
        ttk.Button(
            grid_frame, text="Re-label", command=self._relabel_interval
        ).grid(row=0, column=0, sticky="ew", padx=1, pady=2)
        
        ttk.Button(
            grid_frame, text="Add", command=self._add_interval
        ).grid(row=0, column=1, sticky="ew", padx=1, pady=2)
        
        ttk.Button(
            grid_frame, text="Undo", command=self._undo
        ).grid(row=0, column=2, sticky="ew", padx=1, pady=2)
        
        ttk.Button(
            grid_frame, text="Manage...", command=self._open_label_manager
        ).grid(row=0, column=3, sticky="ew", padx=1, pady=2)
        
        ttk.Button(
            grid_frame, text="Fill Gaps...", command=self._open_label_unassigned_dialog
        ).grid(row=0, column=4, sticky="ew", padx=1, pady=2)
        
        # Row 2: [Dropdown list, Delete, Redo, By-Rule..., Clear...]
        self.current_class_var = tk.StringVar(value=self.classes[0])
        self.class_combo = ttk.Combobox(
            grid_frame,
            textvariable=self.current_class_var,
            values=self.classes,
            state="readonly",
            width=12,
        )
        self.class_combo.grid(row=1, column=0, sticky="ew", padx=1, pady=2)
        
        ttk.Button(
            grid_frame, text="Delete", command=self._delete_interval
        ).grid(row=1, column=1, sticky="ew", padx=1, pady=2)
        
        ttk.Button(
            grid_frame, text="Redo", command=self._redo
        ).grid(row=1, column=2, sticky="ew", padx=1, pady=2)
        
        ttk.Button(
            grid_frame, text="By-Rule...", command=self._open_label_by_rule_dialog
        ).grid(row=1, column=3, sticky="ew", padx=1, pady=2)
        
        ttk.Button(
            grid_frame, text="Clear...", command=self._open_clear_intervals_dialog
        ).grid(row=1, column=4, sticky="ew", padx=1, pady=2)
        
        # Add some spacing after the grid
        ttk.Frame(label_actions, width=10).pack(side=tk.LEFT)
        
        # ── I/O (2x2 grid) ────────────────────────────────────────────────────────
        io_section = ttk.LabelFrame(parent, text="I/O", padding=5)
        io_section.pack(side=tk.LEFT, padx=8)
        
        # Create grid frame for 2x2 button layout
        io_grid = ttk.Frame(io_section)
        io_grid.pack()
        
        # Configure grid columns to have equal weight
        io_grid.columnconfigure(0, weight=1)
        io_grid.columnconfigure(1, weight=1)
        
        # Row 0: Save Session | Load Session
        ttk.Button(
            io_grid, text="Save Session", command=self._save_session
        ).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        
        ttk.Button(
            io_grid, text="Load Session", command=self._load_session
        ).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        
        # Row 1: Export Labels... | (empty)
        ttk.Button(
            io_grid, text="Export Labels...", command=self._export_labels_dialog
        ).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        
        # Column 1, Row 1 is intentionally left empty
    
        # --- Help and Reset Scale (aligned with grid rows) ---
        help_section = ttk.LabelFrame(parent, text="Help", padding=5)
        help_section.pack(side=tk.LEFT, padx=8)
        
        # Create grid frame to align with other sections
        help_grid = ttk.Frame(help_section)
        help_grid.pack()
        
        # Row 0: Help button (aligned with top row of other grids)
        ttk.Button(
            help_grid, text="Help (F1)", command=self._open_help_dialog
        ).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        
        # Row 1: Reset Scale button (aligned with bottom row of other grids)
        ttk.Button(
            help_grid, text="Reset Scale", command=self._reset_all_yscales
        ).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        
        # make sure F1 still opens help (safe to call even if previously bound)
        self.root.bind("<F1>", lambda e: self._open_help_dialog())
        
        # --- Sidebar Toggle (always visible) ---
        sidebar_toggle_section = ttk.LabelFrame(parent, text="View", padding=5)
        sidebar_toggle_section.pack(side=tk.LEFT, padx=8)
        
        # Create grid frame to align with other sections
        toggle_grid = ttk.Frame(sidebar_toggle_section)
        toggle_grid.pack()
        
        # Row 0: Sidebar toggle button (aligned with top row)
        self.sidebar_toggle_btn = ttk.Button(
            toggle_grid, 
            text="Hide Panel ▶", 
            command=self._toggle_sidebar,
            width=12
        )
        self.sidebar_toggle_btn.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        
        # Add tooltip functionality to the button
        self._create_tooltip(self.sidebar_toggle_btn, "Hide sidebar (F9)")
        
        # Row 1: Empty (or could add another view control later)
        # This keeps alignment with other sections
        # Note: Sidebar toggle completely hides/shows the entire sidebar frame


    def _build_plot(self, parent: ttk.Frame) -> None:
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
            from ..utils.fastdraw import BlitHelper
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


    def _build_sidebar(self, parent: ttk.Frame) -> None:
        """
        Build the right sidebar with scrollable content and collapse functionality.
        
        Uses Canvas + Scrollbar pattern for smooth scrolling when content
        exceeds available vertical space. All sidebar sections are placed
        inside a scrollable interior frame.
        """
        # Initialize collapse state
        self.sidebar_collapsed = False
        self.sidebar_expanded_width = 320
        # No collapsed width needed since we completely hide/show
        
        # Create canvas for scrolling
        self.sidebar_canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0)
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.sidebar_canvas.yview)
        
        # Create interior frame (holds all content)
        self.sidebar_interior = ttk.Frame(self.sidebar_canvas)
        
        # Add interior frame to canvas
        self.sidebar_canvas_window = self.sidebar_canvas.create_window(
            (0, 0), window=self.sidebar_interior, anchor="nw"
        )
        
        # Configure canvas scrolling
        self.sidebar_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Build all sections INSIDE the interior frame
        self._build_sidebar_sections(self.sidebar_interior)
        
        # Bind events for scroll region updates
        self.sidebar_interior.bind("<Configure>", self._on_sidebar_configure)
        
        # Bind mouse wheel for smooth scrolling
        self._bind_sidebar_mousewheel()
    
    def _build_sidebar_sections(self, parent: ttk.Frame) -> None:
        """
        Build all sidebar sections (intervals, stats, actions, etc.).
        
        This is separated from _build_sidebar() to keep the scrolling
        setup clean and modular.
        
        Args:
            parent: The interior frame inside the canvas
        """
        # Intervals list (no toggle button here anymore - moved to top bar)
        frame = ttk.LabelFrame(parent, text="Labeled Intervals", padding=5)
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Treeview for intervals
        columns = ("Start", "End", "Label", "Duration")
        self.intervals_tree = ttk.Treeview(
            frame, columns=columns, show="tree headings", height=15
        )
        self.intervals_tree.heading("#0", text="#")
        for col in columns:
            self.intervals_tree.heading(col, text=col)
        self.intervals_tree.column("#0", width=30)
        self.intervals_tree.column("Start", width=80)
        self.intervals_tree.column("End", width=80)
        self.intervals_tree.column("Label", width=90)
        self.intervals_tree.column("Duration", width=70)

        sb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.intervals_tree.yview)
        self.intervals_tree.configure(yscrollcommand=sb.set)
        self.intervals_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.intervals_tree.bind("<<TreeviewSelect>>", self._on_interval_tree_select)

        # Stats
        stats = ttk.LabelFrame(parent, text="Statistics", padding=5)
        stats.pack(fill=tk.X, pady=5)
        self.stats_text = tk.Text(stats, height=8, width=30, state="disabled")
        self.stats_text.pack(fill=tk.BOTH, expand=True)

        # Options
        opts = ttk.LabelFrame(parent, text="Options", padding=5)
        opts.pack(fill=tk.X, pady=5)
        self.snap_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Snap to samples", variable=self.snap_var).pack(anchor=tk.W)
        
        # Overlay toggle
        self.overlays_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opts, text="Show interval overlays on panels", variable=self.overlays_var
        ).pack(anchor=tk.W)

        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.BOTTOM, fill=tk.X
        )
    def _build_axes_from_layout(self, parent: ttk.Frame, layout: dict) -> None:
        """
        Build axes from a declarative layout:
    
        layout = {
          "grid": {
            "nrows": int, "ncols": int,
            "hspace": float, "wspace": float,             # optional
            "height_ratios": [..], "width_ratios": [..],  # optional
          },
          "areas": [
            {"key": "n_log",     "row": 0, "col": 0, "rowspan": 1, "colspan": 2, "role": "time", "sharex_with": "n_log"},
            {"key": "spec_main", "row": 1, "col": 0,                 "role": "time", "sharex_with": "n_log"},
            {"key": "spec_cbar", "row": 1, "col": 1,                 "role": "colorbar", "attach_to": "spec_main"},
            {"key": "scpot",     "row": 2, "col": 0, "colspan": 2,   "role": "time", "sharex_with": "n_log"},
            {"key": "p_log",     "row": 3, "col": 0, "colspan": 2,   "role": "time", "sharex_with": "n_log"},
            {"key": "b_comps",   "row": 4, "col": 0, "colspan": 2,   "role": "time", "sharex_with": "n_log"},
            {"key": "strip",     "row": 5, "col": 0, "colspan": 2,   "role": "strip"},
          ]
        }
        """
        g = layout.get("grid", {})
        nrows = int(g.get("nrows", 2))
        ncols = int(g.get("ncols", 1))
        hspace = g.get("hspace", 0.25)
        wspace = g.get("wspace", 0.05)
        height_ratios = g.get("height_ratios", None)
        width_ratios = g.get("width_ratios", None)
    
        gs = self.fig.add_gridspec(
            nrows, ncols,
            hspace=hspace, wspace=wspace,
            height_ratios=height_ratios, width_ratios=width_ratios,
        )
    
        self.user_axes = {}
        self.axes_roles = {}
        self.strip_ax = None
        self.primary_time_key = None
    
        # First pass: create all axes
        created: Dict[str, plt.Axes] = {}
        for area in layout.get("areas", []):
            key = str(area["key"])
            row = int(area.get("row", 0))
            col = int(area.get("col", 0))
            rowspan = int(area.get("rowspan", 1))
            colspan = int(area.get("colspan", 1))
            role = str(area.get("role", "time")).lower()
    
            ax = self.fig.add_subplot(gs[row:row+rowspan, col:col+colspan])
            created[key] = ax
            self.axes_roles[key] = role
    
            if role == "strip":
                self.strip_ax = ax
            else:
                self.user_axes[key] = ax
                if role == "time" and self.primary_time_key is None:
                    self.primary_time_key = key
    
        if self.strip_ax is None:
            raise ValueError("Layout must include exactly one area with role='strip'.")
    
        # Second pass: sharex relationships for time panels
        for area in layout.get("areas", []):
            key = str(area["key"])
            role = self.axes_roles.get(key)
            if role != "time":
                continue
            share_key = area.get("sharex_with", self.primary_time_key or key)
            if share_key and share_key in created and share_key != key:
                created[key].sharex(created[share_key])
    
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
    
    # ========== Sidebar Scrolling Helper Methods ==========
    
    def _on_sidebar_configure(self, event=None) -> None:
        """
        Update scroll region when sidebar interior frame size changes.
        
        This is called automatically whenever the interior frame is resized
        (e.g., when intervals are added/deleted, window is resized, etc.).
        """
        self._update_sidebar_scroll_region()
    
    def _update_sidebar_scroll_region(self) -> None:
        """
        Update the canvas scroll region to match interior frame size.
        
        Call this manually if you programmatically change sidebar content
        and need to update scrolling immediately.
        """
        if hasattr(self, 'sidebar_canvas') and hasattr(self, 'sidebar_interior'):
            # Update scroll region to encompass all interior content
            self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))
            
            # Update canvas window width to match canvas width (prevents horizontal scrolling)
            canvas_width = self.sidebar_canvas.winfo_width()
            if canvas_width > 1:  # Only update if canvas has been rendered
                self.sidebar_canvas.itemconfig(self.sidebar_canvas_window, width=canvas_width)
    
    def _bind_sidebar_mousewheel(self) -> None:
        """
        Bind mouse wheel to scroll the sidebar canvas.
        
        Supports Windows, Mac, and Linux scroll events.
        Only scrolls when mouse is over the sidebar.
        """
        def on_mouse_wheel(event):
            """Handle mouse wheel scroll events."""
            # Determine scroll direction (cross-platform)
            if event.num == 5 or event.delta < 0:  # Scroll down
                self.sidebar_canvas.yview_scroll(1, "units")
            elif event.num == 4 or event.delta > 0:  # Scroll up
                self.sidebar_canvas.yview_scroll(-1, "units")
        
        def on_enter(event):
            """Enable scrolling when mouse enters sidebar."""
            # Bind mouse wheel events (cross-platform)
            self.sidebar_canvas.bind_all("<MouseWheel>", on_mouse_wheel)  # Windows/Mac
            self.sidebar_canvas.bind_all("<Button-4>", on_mouse_wheel)    # Linux scroll up
            self.sidebar_canvas.bind_all("<Button-5>", on_mouse_wheel)    # Linux scroll down
        
        def on_leave(event):
            """Disable scrolling when mouse leaves sidebar."""
            # Unbind mouse wheel events to avoid interfering with plot zooming
            self.sidebar_canvas.unbind_all("<MouseWheel>")
            self.sidebar_canvas.unbind_all("<Button-4>")
            self.sidebar_canvas.unbind_all("<Button-5>")
        
        # Bind enter/leave events to canvas and interior frame
        self.sidebar_canvas.bind("<Enter>", on_enter)
        self.sidebar_canvas.bind("<Leave>", on_leave)
        self.sidebar_interior.bind("<Enter>", on_enter)
        self.sidebar_interior.bind("<Leave>", on_leave)
    
    def _toggle_sidebar(self) -> None:
        """
        Toggle the sidebar between expanded and collapsed states.
        
        The toggle button is now in the top bar and always visible.
        When collapsed, the entire sidebar is hidden.
        """
        if self.sidebar_collapsed:
            # Expand sidebar
            self.sidebar_collapsed = False
            
            # Update button appearance
            self.sidebar_toggle_btn.configure(text="Hide Panel ▶")
            
            # To ensure proper layout, temporarily unpack plot frame
            # then repack sidebar and plot frame in correct order
            self.plot_frame.pack_forget()
            
            # Pack sidebar first (right side)
            self.sidebar_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
            self.sidebar_frame.pack_propagate(False)
            
            # Then pack plot frame (left side, fills remaining space)
            self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Restore normal width
            self.sidebar_frame.configure(width=self.sidebar_expanded_width)
            
            # Force immediate geometry update
            self.sidebar_frame.update_idletasks()
            self.plot_frame.update_idletasks()
            
            # Force layout updates and refresh
            self.root.after_idle(self._refresh_sidebar_layout)
            
            # Update tooltip
            self._update_tooltip_text("Hide sidebar (F9)")
            
        else:
            # Collapse sidebar
            self.sidebar_collapsed = True
            
            # Update button appearance
            self.sidebar_toggle_btn.configure(text="◀ Show Panel")
            
            # Hide the entire sidebar frame
            self.sidebar_frame.pack_forget()
            
            # Update tooltip
            self._update_tooltip_text("Show sidebar (F9)")
    
    def _create_tooltip(self, widget, text):
        """
        Create a simple tooltip for a widget.
        
        Args:
            widget: The widget to add tooltip to
            text: The tooltip text to display
        """
        # Store initial tooltip text
        widget.tooltip_text = text
        
        def on_enter(event):
            # Use stored text (which may have been updated)
            current_text = getattr(widget, 'tooltip_text', text)
            
            # Create tooltip window
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            # Add tooltip text
            label = tk.Label(
                tooltip, 
                text=current_text, 
                background="lightyellow", 
                relief="solid", 
                borderwidth=1,
                font=("TkDefaultFont", 8)
            )
            label.pack()
            
            # Store reference to tooltip
            widget.tooltip = tooltip
        
        def on_leave(event):
            # Destroy tooltip
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
        
        # Bind events
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)
    
    def _update_tooltip_text(self, new_text):
        """
        Update the tooltip text for the sidebar toggle button.
        
        Args:
            new_text: The new tooltip text to display
        """
        # Store the new text so future tooltip displays will use it
        self.sidebar_toggle_btn.tooltip_text = new_text
    
    def _refresh_sidebar_layout(self):
        """
        Force a complete refresh of the sidebar layout.
        
        This is called after expanding the sidebar to ensure all widgets
        properly recalculate their sizes and positions.
        """
        # Only refresh if sidebar is expanded and exists
        if not self.sidebar_collapsed and hasattr(self, 'sidebar_canvas') and hasattr(self, 'sidebar_interior'):
            # Force geometry updates
            self.sidebar_frame.update_idletasks()
            self.sidebar_interior.update_idletasks()
            self.sidebar_canvas.update_idletasks()
            
            # Reconfigure canvas window width
            canvas_width = self.sidebar_canvas.winfo_width()
            if canvas_width > 1:
                self.sidebar_canvas.itemconfig(self.sidebar_canvas_window, width=canvas_width)
                
            # Update scroll region
            self._update_sidebar_scroll_region()
            
            # Final geometry update
            self.sidebar_canvas.update_idletasks()
            self.sidebar_interior.update_idletasks()
