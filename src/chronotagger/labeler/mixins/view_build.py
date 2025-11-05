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
        sidebar = ttk.Frame(main, width=320)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)
        
        plot_frame = ttk.Frame(main)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_plot(plot_frame)

        self.root.bind("<Key>", self._on_key_press)
        # F1 opens Help
        self.root.bind("<F1>", self._open_help_dialog)

    def _build_top_controls(self, parent: ttk.Frame) -> None:
        from tkinter import filedialog  # ensure Tk is initialized on Windows
        _ = filedialog
    
        # ── Time Range (two-row layout with ×2/÷2 buttons) ──────────────────────────
        rng = ttk.LabelFrame(parent, text="Time Range", padding=6)
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
    
        nav.grid_columnconfigure(0, weight=1)
        nav.grid_columnconfigure(1, weight=0)
        nav.grid_columnconfigure(2, weight=1)
    
        row1 = ttk.Frame(nav)
        row1.grid(row=0, column=1, pady=(0, 2))
        ttk.Button(row1, text="<- Prev", command=self._prev_window).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Next ->", command=self._next_window).pack(side=tk.LEFT, padx=4)
    
        row2 = ttk.Frame(nav)
        row2.grid(row=1, column=1, pady=(2, 0))
        ttk.Label(row2, text="Step:").pack(side=tk.LEFT, padx=(0, 6))
        self.step_entry = ttk.Entry(row2, width=16)
        self.step_entry.insert(0, str(self.step))
        self.step_entry.pack(side=tk.LEFT)
        self.step_entry.bind("<Return>", lambda _: self._apply_step_entry())
        ttk.Button(row2, text="x2", width=4, command=self._double_step).pack(side=tk.LEFT, padx=4)
        ttk.Button(row2, text="/2", width=4, command=self._halve_step).pack(side=tk.LEFT, padx=2)
    
        # ── Current label ─────────────────────────────────────────────────────────
        cls_frame = ttk.LabelFrame(parent, text="Current Label", padding=5)
        cls_frame.pack(side=tk.LEFT, padx=8)
        self.current_class_var = tk.StringVar(value=self.classes[0])
        self.class_combo = ttk.Combobox(
            cls_frame,
            textvariable=self.current_class_var,
            values=self.classes,
            state="readonly",
            width=18,
        )
        self.class_combo.pack(side=tk.LEFT, padx=2)
        
        # Manage labels button
        ttk.Button(cls_frame, text="Manage Labels…", command=self._open_label_manager)\
            .pack(side=tk.LEFT, padx=(6, 0))
    
        # ── Quick actions ─────────────────────────────────────────────────────────
        act = ttk.Frame(parent)
        act.pack(side=tk.LEFT, padx=10)
        ttk.Button(act, text="Add Label", command=self._add_interval).pack(side=tk.LEFT, padx=2)
        ttk.Button(act, text="Label by Rule...", command=self._open_label_by_rule_dialog).pack(side=tk.LEFT, padx=6)
        ttk.Button(act, text="Delete", command=self._delete_interval).pack(side=tk.LEFT, padx=2)
        ttk.Button(act, text="Undo", command=self._undo).pack(side=tk.LEFT, padx=2)
        ttk.Button(act, text="Redo", command=self._redo).pack(side=tk.LEFT, padx=2)
        
        # NEW: always-visible export button
        ttk.Button(act, text="Export Labels…", command=self._export_labels_dialog).pack(side=tk.LEFT, padx=8)
        
        # NEW: Reset Y-Scale button
        ttk.Button(act, text="Reset Y-Scale", command=self._reset_all_yscales).pack(side=tk.LEFT, padx=2)
    
        # --- Help button to show controls ---
        help_box = ttk.Frame(parent)
        help_box.pack(side=tk.LEFT, padx=6)
        ttk.Button(help_box, text="Help (F1)", command=self._open_help_dialog).pack()
        # make sure F1 still opens help (safe to call even if previously bound)
        self.root.bind("<F1>", lambda e: self._open_help_dialog())


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
        # Intervals list
        frame = ttk.LabelFrame(parent, text="Labeled Intervals", padding=5)
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

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

        # Actions
        actions = ttk.LabelFrame(parent, text="Actions", padding=5)
        actions.pack(fill=tk.X, pady=5)
        ttk.Button(actions, text="Relabel Selected", command=self._relabel_interval).pack(fill=tk.X, pady=2)
        ttk.Button(actions, text="Delete Selected", command=self._delete_interval).pack(fill=tk.X, pady=2)
        ttk.Button(actions, text="Assign Remainder → UNKNOWN",
                   command=self._assign_remainder).pack(fill=tk.X, pady=2)
        ttk.Separator(actions, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        ttk.Button(actions, text="Clear All Intervals",
                   command=self._clear_all_intervals).pack(fill=tk.X, pady=2)

        # Files
        files = ttk.LabelFrame(parent, text="File Operations", padding=5)
        files.pack(fill=tk.X, pady=5)
        ttk.Button(files, text="Save Session", command=self._save_session).pack(fill=tk.X, pady=2)
        ttk.Button(files, text="Load Session", command=self._load_session).pack(fill=tk.X, pady=2)
        ttk.Button(files, text="Export Intervals", command=self._export_intervals).pack(fill=tk.X, pady=2)
        ttk.Button(files, text="Export Per-Sample", command=self._export_per_sample).pack(fill=tk.X, pady=2)
        
        # ML-friendly per-sample export (CSV + sidecar)
        ttk.Button(files, text="Export Labels…", command=self._export_labels_dialog).pack(fill=tk.X, pady=2)

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