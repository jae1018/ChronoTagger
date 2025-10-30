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

    def _build_top_controls(self, parent: ttk.Frame) -> None:
        from tkinter import filedialog  # just to ensure Tk is initialized on Windows
        _ = filedialog  # silence linter

        # Time range box
        rng = ttk.LabelFrame(parent, text="Time Range", padding=5)
        rng.pack(side=tk.LEFT, padx=5)

        ttk.Label(rng, text="Start:").grid(row=0, column=0, padx=2)
        self.start_time_entry = ttk.Entry(rng, width=20)
        self.start_time_entry.insert(0, str(self.t0))
        self.start_time_entry.grid(row=0, column=1, padx=2)

        ttk.Label(rng, text="End:").grid(row=0, column=2, padx=2)
        self.end_time_entry = ttk.Entry(rng, width=20)
        self.end_time_entry.insert(0, str(self.t1))
        self.end_time_entry.grid(row=0, column=3, padx=2)

        ttk.Button(rng, text="Update Window", command=self._update_time_window).grid(
            row=0, column=4, padx=5
        )

        # Navigation
        nav = ttk.Frame(parent)
        nav.pack(side=tk.LEFT, padx=5)

        ttk.Button(nav, text="◄◄ Prev", command=self._prev_window).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text="Next ►►", command=self._next_window).pack(side=tk.LEFT, padx=2)

        ttk.Label(nav, text="Step:").pack(side=tk.LEFT, padx=(10, 2))
        self.step_entry = ttk.Entry(nav, width=10)
        self.step_entry.insert(0, str(self.step))
        self.step_entry.pack(side=tk.LEFT, padx=2)

        # Current label
        cls_frame = ttk.LabelFrame(parent, text="Current Label", padding=5)
        cls_frame.pack(side=tk.LEFT, padx=5)

        self.current_class_var = tk.StringVar(value=self.classes[0])
        self.class_combo = ttk.Combobox(
            cls_frame,
            textvariable=self.current_class_var,
            values=self.classes,
            state="readonly",
            width=18,
        )
        self.class_combo.pack(side=tk.LEFT, padx=2)

        # Quick actions
        act = ttk.Frame(parent)
        act.pack(side=tk.LEFT, padx=10)
        ttk.Button(act, text="Add Label", command=self._add_interval).pack(side=tk.LEFT, padx=2)
        ttk.Button(act, text="Delete", command=self._delete_interval).pack(side=tk.LEFT, padx=2)
        ttk.Button(act, text="Undo", command=self._undo).pack(side=tk.LEFT, padx=2)
        ttk.Button(act, text="Redo", command=self._redo).pack(side=tk.LEFT, padx=2)

    def _build_plot(self, parent: ttk.Frame) -> None:
        # 2 user panels + 1 strip at bottom (grid spec leaves room to grow)
        self.fig = plt.Figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(5, 1, height_ratios=[3, 3, 3, 3, 1], hspace=0.3)

        # User axes (share x)
        self.user_axes: Dict[str, plt.Axes] = {
            "panel1": self.fig.add_subplot(gs[0, 0]),
            "panel2": self.fig.add_subplot(gs[1, 0]),
        }
        for ax in list(self.user_axes.values())[1:]:
            ax.sharex(self.user_axes["panel1"])

        # Annotation strip
        self.strip_ax = self.fig.add_subplot(gs[4, 0], sharex=self.user_axes["panel1"])
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

        # Rectangle selector on the first user panel
        self.rect_selector = RectangleSelector(
            self.user_axes["panel1"],
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

        # One-time pick handler for the strip
        if self.pick_cid is None:
            self.pick_cid = self.canvas.mpl_connect("pick_event", self._on_strip_click)

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

        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.BOTTOM, fill=tk.X
        )
