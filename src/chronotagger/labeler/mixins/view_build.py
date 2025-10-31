"""
View construction mixin.

Responsibilities:
- Build the Tk window and main frames
- Build the top controls, plot area, sidebar
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

        plot_frame = ttk.Frame(main)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_plot(plot_frame)

        sidebar = ttk.Frame(main, width=320)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        self.root.bind("<Key>", self._on_key_press)
        # F1 opens Help
        self.root.bind("<F1>", self._open_help_dialog)

    def _build_top_controls(self, parent: ttk.Frame) -> None:
        from tkinter import filedialog  # ensure Tk is initialized on Windows
        _ = filedialog
    
        # ── Time Range (two-row layout, button centered on the right) ──────────────
        rng = ttk.LabelFrame(parent, text="Time Range", padding=6)
        rng.pack(side=tk.LEFT, padx=5)
    
        # Make the entry column stretchy so entries can grow
        rng.grid_columnconfigure(1, weight=1)
    
        # Row 0: Start
        ttk.Label(rng, text="Start:").grid(row=0, column=0, padx=(2, 6), pady=2, sticky="w")
        self.start_time_entry = ttk.Entry(rng, width=22)
        self.start_time_entry.insert(0, str(self.t0))
        self.start_time_entry.grid(row=0, column=1, padx=(0, 6), pady=2, sticky="ew")
    
        # Row 1: End
        ttk.Label(rng, text="End:").grid(row=1, column=0, padx=(2, 6), pady=2, sticky="w")
        self.end_time_entry = ttk.Entry(rng, width=22)
        self.end_time_entry.insert(0, str(self.t1))
        self.end_time_entry.grid(row=1, column=1, padx=(0, 6), pady=2, sticky="ew")
    
        # Right column: a tiny container so the button sits vertically centered
        right_col = ttk.Frame(rng)
        right_col.grid(row=0, column=2, rowspan=2, padx=(8, 2), sticky="ns")
        ttk.Button(right_col, text="Update Window", command=self._update_time_window)\
            .pack(expand=True)  # expand centers it between Start/End
    
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
        ttk.Button(act, text="Delete", command=self._delete_interval).pack(side=tk.LEFT, padx=2)
        ttk.Button(act, text="Undo", command=self._undo).pack(side=tk.LEFT, padx=2)
        ttk.Button(act, text="Redo", command=self._redo).pack(side=tk.LEFT, padx=2)


    # src/chronotagger/labeler/mixins/view_build.py
    def _build_plot(self, parent: ttk.Frame) -> None:
        # Resolve how many data panels to build
        n = self._resolve_n_panels()
    
        # Figure with constrained layout to manage whitespace cleanly
        self.fig = plt.Figure(figsize=(14, 8), constrained_layout=True)
        gs = self.fig.add_gridspec(n + 1, 1, height_ratios=[3] * n + [1], hspace=0.25)
    
        # User axes (share x with the first)
        self.user_axes: Dict[str, plt.Axes] = {}
        for i in range(n):
            ax = self.fig.add_subplot(gs[i, 0])
            key = f"panel{i+1}"
            self.user_axes[key] = ax
            if i > 0:
                ax.sharex(self.user_axes["panel1"])
    
        # Annotation strip
        self.strip_ax = self.fig.add_subplot(gs[n, 0], sharex=self.user_axes["panel1"])
        self.strip_ax.set_ylabel("Labels", fontsize=9)
        self.strip_ax.set_ylim(0, 1)
        self.strip_ax.set_yticks([])
    
        # Apply date formatting to all axes
        for ax in list(self.user_axes.values()) + [self.strip_ax]:
            self._apply_time_axis_format(ax)
    
        # Embed
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
        toolbar = NavigationToolbar2Tk(self.canvas, parent)  # noqa: F841
        toolbar.update()
    
        # Defer first draw so Tk has correct geometry (prevents clipped labels)
        self.root.after(0, self.canvas.draw_idle)
    
        # Mouse wheel zoom/pan (already present)
        if getattr(self, "_scroll_cid", None) is None:
            self._scroll_cid = self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
    
        # Rectangle selector on the first user panel
        first_ax = self.user_axes["panel1"]
        self.rect_selector = RectangleSelector(
            first_ax,
            onselect=self._on_rectangle_select,
            useblit=True,
            button=[1],  # left mouse
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
            props=dict(
                facecolor="yellow", edgecolor="orange",
                alpha=0.3, linestyle="--", linewidth=2
            ),
        )
    
        # One-time pick handler for the strip (select interval)
        if self.pick_cid is None:
            self.pick_cid = self.canvas.mpl_connect("pick_event", self._on_strip_click)
    
        # NEW: mouse-based resize/move on strip axis
        if self._press_cid is None:
            self._press_cid = self.canvas.mpl_connect("button_press_event", self._on_strip_press)
        if self._motion_cid is None:
            self._motion_cid = self.canvas.mpl_connect("motion_notify_event", self._on_strip_motion)
        if self._release_cid is None:
            self._release_cid = self.canvas.mpl_connect("button_release_event", self._on_strip_release)


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
        
    def _resolve_n_panels(self) -> int:
        """
        Determine how many data panels to build.
    
        Priority:
          1) Explicit self.n_panels if provided (>=1)
          2) plot_fn.n_panels attribute if present (>=1)
          3) Probe the plot function on a throwaway figure
          4) Fallback to 2
        """
        # 1) Explicit param
        if isinstance(self.n_panels, int) and self.n_panels >= 1:
            return self.n_panels
    
        # 2) Function attribute
        advertised = getattr(self.plot_fn, "n_panels", None)
        if isinstance(advertised, int) and advertised >= 1:
            return advertised
    
        # 3) Probe
        probed = self._probe_plot_fn_for_panels()
        if probed >= 1:
            return probed
    
        # 4) Sensible default
        return 2
    
    
    def _probe_plot_fn_for_panels(self, max_panels: int = 6) -> int:
        """
        Call the user's plot_fn with a temporary off-screen Figure containing
        up to `max_panels` Axes, then count which Axes received any artists.
    
        Returns a number in [1, max_panels] or 0 on failure.
        """
        try:
            from matplotlib.figure import Figure
            import pandas as pd
        except Exception:
            return 0
    
        # Build a tiny time slice that's guaranteed to exist
        t0 = self.df.index[0]
        # take ~30 minutes or to data_end
        t1 = min(t0 + pd.Timedelta("30min"), self.df.index[-1])
        sub = self.df.loc[t0:t1]
    
        # Throwaway figure; no Tk/pyplot involved
        fig = Figure()
        axs = {f"panel{i+1}": fig.add_subplot(max_panels, 1, i + 1) for i in range(max_panels)}
    
        try:
            self.plot_fn(axs, sub, t0, t1)
        except Exception:
            # User plot may error on probe—don't hard fail the app
            return 0
        finally:
            # Close figure to free memory (no canvas was created)
            try:
                fig.clf()
            except Exception:
                pass
    
        # Count Axes that look "used"
        used = 0
        for ax in axs.values():
            if ax.lines or ax.patches or ax.collections or ax.images:
                used += 1
    
        return max(1, min(max_panels, used)) if used else 0
