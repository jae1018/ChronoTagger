"""
labeler.py

Interactive GUI for labeling time intervals on multi-panel time-series plots.

Key features
------------
- Tkinter + Matplotlib embedded UI
- Rectangle drag selection to add intervals
- Non-overlapping interval management with undo/redo
- Stats and intervals list with color-coding
- Save/Load sessions (JSON)
- Export intervals and per-sample labels (CSV/Parquet)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle

from .models import Interval
from .commands import (
    Command,
    AddIntervalCommand,
    DeleteIntervalCommand,
    RelabelIntervalCommand,
)


class TimeIntervalLabeler:
    """
    Tkinter + Matplotlib app for labeling time intervals in time-series.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with a DatetimeIndex covering the entire dataset.
    plot_fn : Callable[[dict, pd.DataFrame, pd.Timestamp, pd.Timestamp], None]
        User-supplied function that draws panels into provided axes dict.
        Signature: `plot_fn(axs: dict, df: pd.DataFrame, t0: Timestamp, t1: Timestamp)`.
        The `df` passed in is already sliced to [t0, t1].
    classes : list[str], optional
        List of label class names. Default provides typical magnetosphere regions.
    class_colors : dict[str, str], optional
        Mapping of class -> color (hex). If None, a default palette is used.
    window : pd.Timedelta, optional
        Initial visible window duration (default: 30 minutes).
    step : pd.Timedelta, optional
        Navigation step for next/previous (default: 15 minutes).
    start : pd.Timestamp, optional
        Initial window start (defaults to df.index[0]).
    end : pd.Timestamp, optional
        Data end boundary (defaults to df.index[-1]).
    autosave_path : str | Path, optional
        If provided, autosaves after each modification.

    Public API (non-GUI)
    --------------------
    - go_to_window(t0)
    - save(path)
    - load(path)
    - export_intervals(path, fmt="parquet")
    - export_per_sample(path, fmt="parquet", label_on_uncovered="UNKNOWN")
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

        self.df = df
        self.plot_fn = plot_fn

        # Classes & colors
        if classes is None:
            classes = ["PlasmaSheet", "Lobe", "Magnetosheath", "SolarWind", "UNKNOWN"]
        self.classes = list(classes)

        if class_colors is None:
            class_colors = {
                cls: self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)]
                for i, cls in enumerate(self.classes)
            }
        self.class_colors = dict(class_colors)

        # Time bounds
        self.data_start = df.index[0]
        self.data_end = df.index[-1]
        self.window = window
        self.step = step

        if start is None:
            start = self.data_start
        self.t0 = max(start, self.data_start)
        self.t1 = min(self.t0 + window, self.data_end)

        # Intervals & selection
        self.intervals: List[Interval] = []
        self.selected_interval: Optional[Interval] = None
        self.current_selection: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None

        # Undo/redo
        self.undo_stack: List[Command] = []
        self.redo_stack: List[Command] = []
        self.max_undo = 20

        # Persistence
        self.autosave_path = Path(autosave_path) if autosave_path else None
        self.modified = False

        # GUI state
        self.root: Optional[tk.Tk] = None
        self.fig = None
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.user_axes: Dict[str, plt.Axes] = {}
        self.strip_ax: Optional[plt.Axes] = None
        self.rect_selector: Optional[RectangleSelector] = None
        self.pick_cid: Optional[int] = None  # one-time pick handler id

        # Widgets we need to update later
        self.start_time_entry: Optional[ttk.Entry] = None
        self.end_time_entry: Optional[ttk.Entry] = None
        self.step_entry: Optional[ttk.Entry] = None
        self.current_class_var: Optional[tk.StringVar] = None
        self.class_combo: Optional[ttk.Combobox] = None
        self.intervals_tree: Optional[ttk.Treeview] = None
        self.stats_text: Optional[tk.Text] = None
        self.snap_var: Optional[tk.BooleanVar] = None
        self.status_var: Optional[tk.StringVar] = None

    # --------------------- Public ---------------------

    def run(self) -> None:
        """Start the Tkinter main loop."""
        self._build_gui()
        self._update_plot()
        self.root.mainloop()  # type: ignore[union-attr]

    def go_to_window(self, t0: pd.Timestamp) -> None:
        """Programmatically jump to a specific start time."""
        self.t0 = max(t0, self.data_start)
        self.t1 = min(self.t0 + self.window, self.data_end)
        if self.start_time_entry and self.end_time_entry:
            self.start_time_entry.delete(0, tk.END)
            self.start_time_entry.insert(0, str(self.t0))
            self.end_time_entry.delete(0, tk.END)
            self.end_time_entry.insert(0, str(self.t1))
        self._update_plot()

    def save(self, path: Optional[str] = None) -> None:
        """Save the current session to JSON."""
        self._save_session(path)

    def load(self, path: str) -> None:
        """Load a session from JSON."""
        self._load_session(path)

    def export_intervals(self, path: str, fmt: str = "parquet") -> None:
        """Export all intervals to CSV or Parquet."""
        if not self.intervals:
            print("No intervals to export.")
            return
        rows = [
            {"start": iv.start, "end": iv.end, "label": iv.label, "notes": iv.notes}
            for iv in self.intervals
        ]
        df_export = pd.DataFrame(rows)
        if fmt.lower() == "parquet":
            df_export.to_parquet(path, index=False)
        else:
            df_export.to_csv(path, index=False)
        print(f"Exported intervals to {path}")

    def export_per_sample(
        self, path: str, fmt: str = "parquet", label_on_uncovered: Optional[str] = "UNKNOWN"
    ) -> None:
        """
        Export a per-sample label series aligned to the dataframe index.

        Parameters
        ----------
        path : str
            Output path.
        fmt : {'parquet','csv'}
            File format.
        label_on_uncovered : Optional[str]
            Label for samples not covered by any interval (None to keep nulls).
        """
        labels: List[Optional[str]] = []
        for ts in self.df.index:
            lbl = None
            for iv in self.intervals:
                if iv.contains(ts):
                    lbl = iv.label
                    break
            labels.append(lbl if lbl is not None else label_on_uncovered)

        df_export = pd.DataFrame({"label": labels}, index=self.df.index)
        if fmt.lower() == "parquet":
            df_export.to_parquet(path)
        else:
            df_export.to_csv(path)
        print(f"Exported per-sample labels to {path}")

    # --------------------- GUI construction ---------------------

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
        # Time range
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

        # Class selection
        cls_frame = ttk.LabelFrame(parent, text="Current Label", padding=5)
        cls_frame.pack(side=tk.LEFT, padx=5)

        self.current_class_var = tk.StringVar(value=self.classes[0])
        self.class_combo = ttk.Combobox(
            cls_frame, textvariable=self.current_class_var,
            values=self.classes, state="readonly", width=18
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
        # Figure layout: 2 user panels + 1 strip
        self.fig = plt.Figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(5, 1, height_ratios=[3, 3, 3, 3, 1], hspace=0.3)
        
        # Create user axes
        self.user_axes = {
            "panel1": self.fig.add_subplot(gs[0, 0]),
            "panel2": self.fig.add_subplot(gs[1, 0]),
        }
        for ax in list(self.user_axes.values())[1:]:
            ax.sharex(self.user_axes["panel1"])
        
        # Create strip axis
        self.strip_ax = self.fig.add_subplot(gs[4, 0], sharex=self.user_axes["panel1"])
        self.strip_ax.set_ylabel("Labels", fontsize=9)
        self.strip_ax.set_ylim(0, 1)
        self.strip_ax.set_yticks([])
        
        # NOW apply date formatting (all axes exist)
        for ax in list(self.user_axes.values()) + [self.strip_ax]:
            self._apply_time_axis_format(ax)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, parent)  # noqa: F841
        toolbar.update()

        # Rectangle selector on first panel (shared x makes it consistent)
        self.rect_selector = RectangleSelector(
            self.user_axes["panel1"],
            onselect=self._on_rectangle_select,
            useblit=True,
            button=[1],  # left mouse
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
            props=dict(facecolor="yellow", edgecolor="orange",
                       alpha=0.3, linestyle="--", linewidth=2),
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

    # --------------------- Event handlers ---------------------

    def _update_time_window(self) -> None:
        try:
            new_t0 = pd.to_datetime(self.start_time_entry.get())  # type: ignore[union-attr]
            new_t1 = pd.to_datetime(self.end_time_entry.get())    # type: ignore[union-attr]
            if new_t0 >= new_t1:
                messagebox.showerror("Invalid Range", "Start time must be before end time.")
                return
            self.t0 = max(new_t0, self.data_start)
            self.t1 = min(new_t1, self.data_end)
            # Normalize entries (clipped)
            self.start_time_entry.delete(0, tk.END)  # type: ignore[union-attr]
            self.start_time_entry.insert(0, str(self.t0))  # type: ignore[union-attr]
            self.end_time_entry.delete(0, tk.END)  # type: ignore[union-attr]
            self.end_time_entry.insert(0, str(self.t1))  # type: ignore[union-attr]
            self._update_plot()
            self.status_var.set(f"Window updated: {self.t0.strftime('%H:%M:%S')} → {self.t1.strftime('%H:%M:%S')}")  # type: ignore[union-attr]
        except Exception as e:
            messagebox.showerror("Invalid Time Format", f"Could not parse time: {e}")

    def _on_interval_tree_select(self, _event) -> None:
        sel = self.intervals_tree.selection()  # type: ignore[union-attr]
        if not sel:
            self.selected_interval = None
            return
        item = sel[0]
        try:
            idx = int(self.intervals_tree.item(item)["text"]) - 1  # type: ignore[union-attr]
            if 0 <= idx < len(self.intervals):
                self.selected_interval = self.intervals[idx]
                iv = self.selected_interval
                self.status_var.set(
                    f"Selected: {iv.label} [{iv.start.strftime('%H:%M:%S')} → {iv.end.strftime('%H:%M:%S')}]"
                )  # type: ignore[union-attr]
                self._update_strip()
                self.canvas.draw()  # type: ignore[union-attr]
        except Exception:
            self.selected_interval = None

    def _on_rectangle_select(self, eclick, erelease) -> None:
        """Handle drag selection over the plot."""
        # Guard: if selection happens outside data area, xdata can be None
        if eclick.xdata is None or erelease.xdata is None:
            return
        x1, x2 = sorted([eclick.xdata, erelease.xdata])

        # Convert safely to naive timestamps (avoid tz-aware/naive mismatches)
        def _to_naive_ts(x: float) -> pd.Timestamp:
            dt = mdates.num2date(x)
            # num2date may return tz-aware; normalize to naive for DatetimeIndex
            if getattr(dt, "tzinfo", None) is not None:
                dt = dt.replace(tzinfo=None)
            return pd.Timestamp(dt)

        t_start, t_end = _to_naive_ts(x1), _to_naive_ts(x2)

        # Apply snapping if enabled
        if self.snap_var.get():  # type: ignore[union-attr]
            t_start, t_end = self._snap_to_samples(t_start, t_end)

        self.current_selection = (t_start, t_end)
        self.status_var.set(f"Selected: {t_start.strftime('%H:%M:%S')} → {t_end.strftime('%H:%M:%S')}")  # type: ignore[union-attr]
        self._update_strip()
        self.canvas.draw()  # type: ignore[union-attr]

    def _on_strip_click(self, event) -> None:
        """Handle clicks on rectangles in the strip (pick_event)."""
        if event.artist not in self.strip_ax.patches:  # type: ignore[union-attr]
            return
        if event.mouseevent.xdata is None:
            return
        dt = mdates.num2date(event.mouseevent.xdata)
        if getattr(dt, "tzinfo", None) is not None:
            dt = dt.replace(tzinfo=None)
        click_ts = pd.Timestamp(dt)

        for iv in self.intervals:
            if iv.contains(click_ts):
                self.selected_interval = iv
                self.status_var.set(
                    f"Selected: {iv.label} [{iv.start.strftime('%H:%M:%S')} → {iv.end.strftime('%H:%M:%S')}]"
                )  # type: ignore[union-attr]
                self._update_strip()
                self.canvas.draw()  # type: ignore[union-attr]
                break

    def _on_key_press(self, event) -> None:
        key = event.keysym

        # Class selection with digits 1..9
        if key.isdigit() and int(key) > 0:
            idx = int(key) - 1
            if idx < len(self.classes):
                self.current_class_var.set(self.classes[idx])  # type: ignore[union-attr]
                self.status_var.set(f"Selected class: {self.classes[idx]}")  # type: ignore[union-attr]

        # Navigation
        elif key in ("n", "N", "Right"):
            self._next_window()
        elif key in ("p", "P", "Left"):
            self._prev_window()

        # Actions
        elif key in ("a", "A", "Return"):
            self._add_interval()
        elif key in ("d", "D", "Delete"):
            self._delete_interval()
        elif key in ("u", "U"):
            if "UNKNOWN" in self.classes:
                self.current_class_var.set("UNKNOWN")  # type: ignore[union-attr]
                self.status_var.set("Selected class: UNKNOWN")  # type: ignore[union-attr]

        # Save / export (Ctrl+S / Ctrl+E)
        elif key in ("s", "S") and event.state & 0x4:
            self._save_session()
        elif key in ("e", "E") and event.state & 0x4:
            self._export_intervals()

        # Undo / Redo (Ctrl+Z / Ctrl+Y) + Backspace ergonomics
        elif (key == "z" and event.state & 0x4) or key == "BackSpace":
            self._undo()
        elif (key == "y" and event.state & 0x4) or (key == "BackSpace" and event.state & 0x1):
            # Shift+Backspace → redo (state bit 0x1 is Shift)
            self._redo()

    # --------------------- Plot updates ---------------------

    def _update_plot(self) -> None:
        """Redraw user panels and strip."""
        # Clear user axes
        for ax in self.user_axes.values():
            ax.clear()
    
        # Call user plot function with sliced data
        try:
            sub_df = self.df.loc[self.t0:self.t1]
            self.plot_fn(self.user_axes, sub_df, self.t0, self.t1)
        except Exception as e:  # be robust to user code
            for ax in self.user_axes.values():
                ax.text(0.5, 0.5, f"Plot error:\n{e}", transform=ax.transAxes,
                        ha="center", va="center")
    
        # Apply x-lims + date formatting to every panel and the strip
        axes = list(self.user_axes.values())
        if self.strip_ax is not None:
            axes.append(self.strip_ax)
    
        for ax in axes:
            ax.set_xlim(self.t0, self.t1)
            self._apply_time_axis_format(ax)
            ax.margins(x=0.01)
    
        self._update_strip()
        self._update_intervals_list()
    
        self.fig.tight_layout()  # type: ignore[union-attr]
        self.canvas.draw()       # type: ignore[union-attr]

    def _update_strip(self) -> None:
        """Redraw the annotation strip with intervals + current selection preview."""
        ax = self.strip_ax  # type: ignore[assignment]
        ax.clear()
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel("Labels", fontsize=9)
    
        # Clearing resets formatters/locators; restore them and limits here.
        ax.set_xlim(self.t0, self.t1)
        self._apply_time_axis_format(ax)
    
        # Draw labeled intervals overlapping the window
        for iv in self.intervals:
            if iv.end <= self.t0 or iv.start >= self.t1:
                continue
            s = max(iv.start, self.t0)
            e = min(iv.end, self.t1)
    
            color = self.class_colors.get(iv.label, "#cccccc")
            alpha = 0.8 if iv == self.selected_interval else 0.6
            edgecolor = "red" if iv == self.selected_interval else "black"
            lw = 2 if iv == self.selected_interval else 0.5
    
            rect = Rectangle(
                (mdates.date2num(s), 0.1),
                mdates.date2num(e) - mdates.date2num(s),
                0.8,
                facecolor=color,
                edgecolor=edgecolor,
                linewidth=lw,
                alpha=alpha,
                picker=True,
            )
            ax.add_patch(rect)
    
        # Current selection preview
        if self.current_selection:
            s, e = self.current_selection
            rect = Rectangle(
                (mdates.date2num(s), 0.05),
                mdates.date2num(e) - mdates.date2num(s),
                0.9,
                facecolor="yellow",
                edgecolor="orange",
                linewidth=2,
                alpha=0.3,
                linestyle="--",
            )
            ax.add_patch(rect)


    def _update_intervals_list(self) -> None:
        """Refresh the sidebar list + stats."""
        tree = self.intervals_tree  # type: ignore[assignment]
        # Clear
        for item in tree.get_children():
            tree.delete(item)

        # Refill
        for i, iv in enumerate(self.intervals):
            dur = iv.end - iv.start
            start_str = iv.start.strftime("%H:%M:%S")
            end_str = iv.end.strftime("%H:%M:%S")
            dur_str = str(dur).split(".")[0]
            item_id = tree.insert(
                "", "end", text=str(i + 1),
                values=(start_str, end_str, iv.label, dur_str),
                tags=(iv.label,),
            )
            # Apply per-label background color
            tree.tag_configure(iv.label, background=self.class_colors.get(iv.label, "#cccccc"))

        self._update_statistics()

    def _update_statistics(self) -> None:
        txt = self.stats_text  # type: ignore[assignment]
        txt.config(state="normal")
        txt.delete(1.0, tk.END)

        if not self.intervals:
            txt.insert(tk.END, "No intervals labeled yet.")
            txt.config(state="disabled")
            return

        total = self.data_end - self.data_start
        labeled = sum((iv.end - iv.start for iv in self.intervals), pd.Timedelta(0))
        pct = (labeled / total * 100) if total > pd.Timedelta(0) else 0

        # Aggregate by label
        counts: Dict[str, int] = {}
        durations: Dict[str, pd.Timedelta] = {}
        for iv in self.intervals:
            counts[iv.label] = counts.get(iv.label, 0) + 1
            durations[iv.label] = durations.get(iv.label, pd.Timedelta(0)) + (iv.end - iv.start)

        txt.insert(tk.END, f"Total Intervals: {len(self.intervals)}\n")
        txt.insert(tk.END, f"Labeled: {labeled} / {total}\n")
        txt.insert(tk.END, f"Coverage: {pct:.1f}%\n\n")
        txt.insert(tk.END, "By Label:\n")
        for label in sorted(counts):
            lpct = (durations[label] / total * 100) if total > pd.Timedelta(0) else 0
            txt.insert(tk.END, f"  {label}: {counts[label]} intervals, {lpct:.1f}%\n")

        txt.config(state="disabled")

    # --------------------- Commands & ops ---------------------

    def _add_interval(self) -> None:
        if not self.current_selection:
            messagebox.showwarning("No Selection", "Drag on a plot to select a time range first.")
            return
        s, e = self.current_selection
        label = self.current_class_var.get()  # type: ignore[union-attr]
        cmd = AddIntervalCommand(self, Interval(s, e, label))
        self._execute_command(cmd)
        self.current_selection = None
        self.status_var.set(f"Added {label} interval")  # type: ignore[union-attr]
        self._update_plot()
        self._maybe_autosave()

    def _relabel_interval(self) -> None:
        if not self.selected_interval:
            messagebox.showwarning("No Selection", "Select an interval (strip or list) first.")
            return
        new_label = self.current_class_var.get()  # type: ignore[union-attr]
        cmd = RelabelIntervalCommand(self, self.selected_interval, new_label)
        self._execute_command(cmd)
        self.status_var.set(f"Relabeled → {new_label}")  # type: ignore[union-attr]
        self._update_plot()
        self._maybe_autosave()

    def _delete_interval(self) -> None:
        if not self.selected_interval:
            messagebox.showwarning("No Selection", "Select an interval to delete.")
            return
        cmd = DeleteIntervalCommand(self, self.selected_interval)
        self._execute_command(cmd)
        self.selected_interval = None
        self.status_var.set("Deleted interval")  # type: ignore[union-attr]
        self._update_plot()
        self._maybe_autosave()

    def _assign_remainder(self) -> None:
        """Label unlabeled gaps in the current window as UNKNOWN."""
        if "UNKNOWN" not in self.classes:
            messagebox.showwarning("No UNKNOWN Class", "UNKNOWN class is not defined.")
            return

        covered: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for iv in self.intervals:
            if iv.end <= self.t0 or iv.start >= self.t1:
                continue
            covered.append((max(iv.start, self.t0), min(iv.end, self.t1)))

        covered.sort()
        # Merge overlaps
        merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for s, e in covered:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # Find gaps
        gaps: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        cur = self.t0
        for s, e in merged:
            if cur < s:
                gaps.append((cur, s))
            cur = max(cur, e)
        if cur < self.t1:
            gaps.append((cur, self.t1))

        # Add UNKNOWN intervals
        for s, e in gaps:
            self._execute_command(AddIntervalCommand(self, Interval(s, e, "UNKNOWN")))

        self.status_var.set(f"Assigned {len(gaps)} UNKNOWN intervals")  # type: ignore[union-attr]
        self._update_plot()
        self._maybe_autosave()

    def _execute_command(self, cmd: Command) -> None:
        cmd.execute()
        self.undo_stack.append(cmd)
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.modified = True

    def _undo(self) -> None:
        if not self.undo_stack:
            self.status_var.set("Nothing to undo")  # type: ignore[union-attr]
            return
        cmd = self.undo_stack.pop()
        cmd.undo()
        self.redo_stack.append(cmd)
        self.status_var.set("Undo")  # type: ignore[union-attr]
        self._update_plot()
        self._maybe_autosave()

    def _redo(self) -> None:
        if not self.redo_stack:
            self.status_var.set("Nothing to redo")  # type: ignore[union-attr]
            return
        cmd = self.redo_stack.pop()
        cmd.execute()
        self.undo_stack.append(cmd)
        self.status_var.set("Redo")  # type: ignore[union-attr]
        self._update_plot()
        self._maybe_autosave()

    def _remove_overlapping_intervals(
        self, new_interval: Interval
    ) -> Tuple[List[Interval], List[Interval]]:
        """
        Remove/trim intervals that overlap `new_interval`.

        Returns
        -------
        removed : list[Interval]
            The original intervals that were removed due to overlap.
        trims : list[Interval]
            The *new* trimmed intervals we added back (non-overlapping parts).
        """
        removed: List[Interval] = []
        trims: List[Interval] = []

        for iv in self.intervals[:]:
            if not iv.overlaps(new_interval):
                continue

            # Remove the overlapping original
            self.intervals.remove(iv)
            removed.append(iv)

            # Add left piece
            if iv.start < new_interval.start:
                trims.append(Interval(iv.start, new_interval.start, iv.label, iv.notes))

            # Add right piece
            if iv.end > new_interval.end:
                trims.append(Interval(new_interval.end, iv.end, iv.label, iv.notes))

        # Append trimmed pieces now
        self.intervals.extend(trims)
        return removed, trims

    def _sort_and_merge_intervals(self) -> None:
        """Sort by start time and merge adjacent intervals with same label."""
        if not self.intervals:
            return
        self.intervals.sort(key=lambda x: x.start)
        merged = [self.intervals[0]]
        for iv in self.intervals[1:]:
            last = merged[-1]
            if iv.start == last.end and iv.label == last.label:
                last.end = iv.end
            else:
                merged.append(iv)
        self.intervals = merged

    def _snap_to_samples(
        self, t_start: pd.Timestamp, t_end: pd.Timestamp
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Snap timestamps to nearest samples within the current window."""
        sub = self.df.loc[self.t0:self.t1]
        if len(sub.index) == 0:
            return t_start, t_end
        idx_start = sub.index[sub.index.get_indexer([t_start], method="nearest")[0]]
        idx_end = sub.index[sub.index.get_indexer([t_end], method="nearest")[0]]
        return idx_start, idx_end

    # -------------------- Formatting --------------------
    
    def _apply_time_axis_format(self, ax):
        """
        Ensure the x-axis is treated and formatted as dates.
    
        Important: do NOT call ticklabel_format(...) here; that installs a
        ScalarFormatter and nukes the date formatter.
        """
        if ax is None:
            return
        import matplotlib.dates as mdates
    
        ax.xaxis_date()
        locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)



    # --------------------- File ops ---------------------

    def _save_session(self, path: Optional[str] = None) -> None:
        """Save session to JSON (GUI button)."""
        target = Path(path) if path else self.autosave_path
        if target is None:
            chosen = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            if not chosen:
                return
            target = Path(chosen)
            self.autosave_path = target

        data = {
            "version": 1,
            "classes": self.classes,
            "class_colors": self.class_colors,
            "window": str(self.window),
            "step": str(self.step),
            "data_start": self.data_start.isoformat(),
            "data_end": self.data_end.isoformat(),
            "intervals": [iv.to_dict() for iv in self.intervals],
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        self.modified = False
        self.status_var.set(f"Saved to {target}")  # type: ignore[union-attr]

    def _load_session(self, path: Optional[str] = None) -> None:
        """Load session from JSON (GUI button)."""
        if path is None:
            chosen = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not chosen:
                return
            path = chosen

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.classes = list(data["classes"])
        self.class_colors = dict(data["class_colors"])
        self.window = pd.Timedelta(data["window"])
        self.step = pd.Timedelta(data["step"])
        self.intervals = [Interval.from_dict(d) for d in data["intervals"]]

        self.autosave_path = Path(path)
        self.modified = False

        # Refresh class combo values safely
        if self.class_combo is not None and self.current_class_var is not None:
            self.class_combo["values"] = self.classes
            if self.current_class_var.get() not in self.classes:
                self.current_class_var.set(self.classes[0])

        # Refresh entries
        if self.start_time_entry and self.end_time_entry and self.step_entry:
            self.start_time_entry.delete(0, tk.END)
            self.start_time_entry.insert(0, str(self.t0))
            self.end_time_entry.delete(0, tk.END)
            self.end_time_entry.insert(0, str(self.t1))
            self.step_entry.delete(0, tk.END)
            self.step_entry.insert(0, str(self.step))

        self._update_plot()
        self.status_var.set(f"Loaded from {path}")  # type: ignore[union-attr]

    def _export_intervals(self) -> None:
        if not self.intervals:
            messagebox.showwarning("No Data", "No intervals to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Parquet files", "*.parquet"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        rows = [
            {"start": iv.start, "end": iv.end, "label": iv.label, "notes": iv.notes}
            for iv in self.intervals
        ]
        df_export = pd.DataFrame(rows)
        if path.endswith(".parquet"):
            df_export.to_parquet(path, index=False)
        else:
            df_export.to_csv(path, index=False)
        self.status_var.set(f"Exported to {path}")  # type: ignore[union-attr]
        messagebox.showinfo("Export Complete", f"Intervals exported to {path}")

    def _export_per_sample(self) -> None:
        if not self.intervals:
            messagebox.showwarning("No Data", "No intervals to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Parquet files", "*.parquet"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        labels: List[Optional[str]] = []
        for ts in self.df.index:
            lbl = None
            for iv in self.intervals:
                if iv.contains(ts):
                    lbl = iv.label
                    break
            labels.append(lbl if lbl is not None else "UNKNOWN")
        df_export = pd.DataFrame({"label": labels}, index=self.df.index)

        if path.endswith(".parquet"):
            df_export.to_parquet(path)
        else:
            df_export.to_csv(path)

        self.status_var.set(f"Exported to {path}")  # type: ignore[union-attr]
        messagebox.showinfo("Export Complete", f"Per-sample labels exported to {path}")

    def _clear_all_intervals(self) -> None:
        if not self.intervals:
            return
        if messagebox.askyesno("Clear All", f"Delete all {len(self.intervals)} intervals?"):
            self.intervals.clear()
            self.selected_interval = None
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.modified = True
            self._update_plot()
            self._update_intervals_list()
            self.status_var.set("All intervals cleared")  # type: ignore[union-attr]

    # --------------------- Navigation ---------------------

    def _prev_window(self) -> None:
        try:
            self.step = pd.Timedelta(self.step_entry.get())  # type: ignore[union-attr]
        except Exception:
            pass
        self.window = self.t1 - self.t0
        self.t0 = max(self.data_start, self.t0 - self.step)
        self.t1 = min(self.t0 + self.window, self.data_end)
        self._sync_entries_and_plot()

    def _next_window(self) -> None:
        try:
            self.step = pd.Timedelta(self.step_entry.get())  # type: ignore[union-attr]
        except Exception:
            pass
        self.window = self.t1 - self.t0
        self.t0 = self.t0 + self.step
        self.t1 = self.t0 + self.window
        if self.t1 > self.data_end:
            self.t1 = self.data_end
            self.t0 = max(self.t1 - self.window, self.data_start)
        self._sync_entries_and_plot()

    def _sync_entries_and_plot(self) -> None:
        self.start_time_entry.delete(0, tk.END)  # type: ignore[union-attr]
        self.start_time_entry.insert(0, str(self.t0))  # type: ignore[union-attr]
        self.end_time_entry.delete(0, tk.END)  # type: ignore[union-attr]
        self.end_time_entry.insert(0, str(self.t1))  # type: ignore[union-attr]
        self._update_plot()
        self.status_var.set(f"Window: {self.t0.strftime('%H:%M:%S')} → {self.t1.strftime('%H:%M:%S')}")  # type: ignore[union-attr]

    # --------------------- Close ---------------------

    def _maybe_autosave(self) -> None:
        if self.autosave_path and self.modified:
            self._save_session(str(self.autosave_path))

    def _on_closing(self) -> None:
        if self.modified:
            resp = messagebox.askyesnocancel("Save Changes?", "Save before closing?")
            if resp is None:  # Cancel
                return
            elif resp:        # Yes
                self._save_session()
        self.root.destroy()  # type: ignore[union-attr]
