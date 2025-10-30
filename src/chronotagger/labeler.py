"""
Time Interval Labeler for Time-Series Plots

A Tkinter + Matplotlib GUI tool for interactively labeling time intervals
over multi-panel scientific plots.

Features:
- Text entry boxes for precise time range specification
- Drag-box selection across multiple axes
- Sidebar with intervals list showing counts and percentages
- Consistent axis alignment across all panels
- Undo/redo functionality
- Session save/load
- Export to CSV/Parquet

Author: Claude
Date: 2025-10-30
"""

import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
from matplotlib.dates import date2num, num2date
import matplotlib.dates as mdates


@dataclass
class Interval:
    """Represents a single labeled time interval."""
    start: pd.Timestamp
    end: pd.Timestamp
    label: str
    notes: Optional[str] = None
    
    def __post_init__(self):
        """Ensure start <= end."""
        if self.start > self.end:
            self.start, self.end = self.end, self.start
    
    def overlaps(self, other: 'Interval') -> bool:
        """Check if this interval overlaps with another."""
        return not (self.end <= other.start or self.start >= other.end)
    
    def contains(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within this interval."""
        return self.start <= timestamp < self.end
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'start': self.start.isoformat(),
            'end': self.end.isoformat(),
            'label': self.label,
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Interval':
        """Create Interval from dictionary."""
        return cls(
            start=pd.Timestamp(d['start']),
            end=pd.Timestamp(d['end']),
            label=d['label'],
            notes=d.get('notes')
        )


class Command:
    """Base class for undo/redo commands."""
    
    def execute(self):
        """Execute the command."""
        raise NotImplementedError
    
    def undo(self):
        """Undo the command."""
        raise NotImplementedError


class AddIntervalCommand(Command):
    """Command to add an interval."""
    
    def __init__(self, labeler: 'TimeIntervalLabeler', interval: Interval):
        self.labeler = labeler
        self.interval = interval
        self.removed_intervals = []
    
    def execute(self):
        """Add interval, removing overlaps."""
        self.removed_intervals = self.labeler._remove_overlapping_intervals(self.interval)
        self.labeler.intervals.append(self.interval)
        self.labeler._sort_and_merge_intervals()
    
    def undo(self):
        """Restore previous intervals."""
        self.labeler.intervals.remove(self.interval)
        self.labeler.intervals.extend(self.removed_intervals)
        self.labeler._sort_and_merge_intervals()


class DeleteIntervalCommand(Command):
    """Command to delete an interval."""
    
    def __init__(self, labeler: 'TimeIntervalLabeler', interval: Interval):
        self.labeler = labeler
        self.interval = interval
    
    def execute(self):
        """Remove the interval."""
        self.labeler.intervals.remove(self.interval)
    
    def undo(self):
        """Restore the interval."""
        self.labeler.intervals.append(self.interval)
        self.labeler._sort_and_merge_intervals()


class RelabelIntervalCommand(Command):
    """Command to relabel an interval."""
    
    def __init__(self, labeler: 'TimeIntervalLabeler', interval: Interval, new_label: str):
        self.labeler = labeler
        self.interval = interval
        self.old_label = interval.label
        self.new_label = new_label
    
    def execute(self):
        """Change the label."""
        self.interval.label = self.new_label
        self.labeler._sort_and_merge_intervals()
    
    def undo(self):
        """Restore old label."""
        self.interval.label = self.old_label
        self.labeler._sort_and_merge_intervals()


class TimeIntervalLabeler:
    """
    Interactive time interval labeler for time-series data.
    
    This tool provides a Tkinter GUI with embedded Matplotlib plots for
    labeling time intervals in multi-panel scientific visualizations.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex containing the time-series data.
    plot_fn : callable
        Function with signature plot_fn(axs_dict, df, t0, t1) that draws
        the user's panels. axs_dict contains matplotlib axes for plotting.
    classes : list[str], optional
        List of label class names (default: standard magnetosphere regions).
    class_colors : dict[str, str], optional
        Mapping of class names to hex colors. If None, uses default palette.
    window : pd.Timedelta, optional
        Initial window duration (default: 30 minutes).
    step : pd.Timedelta, optional
        Step size for next/previous navigation (default: 15 minutes).
    start : pd.Timestamp, optional
        Initial window start time. If None, uses df.index[0].
    end : pd.Timestamp, optional
        Data end boundary. If None, uses df.index[-1].
    autosave_path : str or Path, optional
        Path for automatic session saves. If None, prompts on first save.
    
    Examples
    --------
    >>> def my_plot(axs, df, t0, t1):
    ...     sub = df.loc[t0:t1]
    ...     axs["panel1"].plot(sub.index, sub["data"])
    >>> labeler = TimeIntervalLabeler(df, my_plot)
    >>> labeler.run()
    """
    
    # Default color palette (from Tableau 10)
    DEFAULT_COLORS = [
        '#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac'
    ]
    
    def __init__(
        self,
        df: pd.DataFrame,
        plot_fn: Callable,
        classes: List[str] = None,
        class_colors: Optional[Dict[str, str]] = None,
        window: pd.Timedelta = pd.Timedelta("30min"),
        step: pd.Timedelta = pd.Timedelta("15min"),
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        autosave_path: Optional[str] = None,
    ):
        # Validate inputs
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame must have a DatetimeIndex")
        
        if not callable(plot_fn):
            raise TypeError("plot_fn must be callable")
        
        # Core data
        self.df = df
        self.plot_fn = plot_fn
        
        # Classes and colors
        if classes is None:
            classes = ["PlasmaSheet", "Lobe", "Magnetosheath", "SolarWind", "UNKNOWN"]
        self.classes = classes
        
        if class_colors is None:
            class_colors = {
                cls: self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)]
                for i, cls in enumerate(classes)
            }
        self.class_colors = class_colors
        
        # Time window settings
        self.window = window
        self.step = step
        self.data_start = df.index[0]
        self.data_end = df.index[-1]
        
        # Current window
        if start is None:
            start = self.data_start
        self.t0 = max(start, self.data_start)
        self.t1 = min(self.t0 + window, self.data_end)
        
        # Labeled intervals
        self.intervals: List[Interval] = []
        self.selected_interval: Optional[Interval] = None
        
        # Undo/redo
        self.undo_stack: List[Command] = []
        self.redo_stack: List[Command] = []
        self.max_undo = 20
        
        # Persistence
        self.autosave_path = Path(autosave_path) if autosave_path else None
        self.modified = False
        
        # GUI state
        self.root = None
        self.fig = None
        self.canvas = None
        self.user_axes = {}
        self.strip_ax = None
        self.rect_selector = None
        self.current_selection = None  # (start, end) timestamps
        self.current_class_var = None
        
        # GUI widgets
        self.start_time_entry = None
        self.end_time_entry = None
        self.step_entry = None
        self.intervals_tree = None
        self.stats_text = None
        self.snap_var = None
        
    def run(self):
        """Start the Tkinter main loop."""
        self._build_gui()
        self._update_plot()
        self.root.mainloop()
    
    def _build_gui(self):
        """Construct the Tkinter GUI."""
        self.root = tk.Tk()
        self.root.title("Time Interval Labeler")
        self.root.geometry("1600x900")
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Top controls
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self._build_top_controls(top_frame)
        
        # Main layout: left=plot, right=sidebar
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left: Matplotlib figure
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self._build_plot(plot_frame)
        
        # Right: Sidebar with labels and controls
        sidebar_frame = ttk.Frame(main_frame, width=300)
        sidebar_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        sidebar_frame.pack_propagate(False)
        
        self._build_sidebar(sidebar_frame)
        
        # Keyboard shortcuts
        self.root.bind('<Key>', self._on_key_press)
    
    def _build_top_controls(self, parent):
        """Build the top control bar with time range selection."""
        # Time range frame
        range_frame = ttk.LabelFrame(parent, text="Time Range", padding=5)
        range_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(range_frame, text="Start:").grid(row=0, column=0, padx=2)
        self.start_time_entry = ttk.Entry(range_frame, width=20)
        self.start_time_entry.insert(0, str(self.t0))
        self.start_time_entry.grid(row=0, column=1, padx=2)
        
        ttk.Label(range_frame, text="End:").grid(row=0, column=2, padx=2)
        self.end_time_entry = ttk.Entry(range_frame, width=20)
        self.end_time_entry.insert(0, str(self.t1))
        self.end_time_entry.grid(row=0, column=3, padx=2)
        
        ttk.Button(range_frame, text="Update Window", 
                  command=self._update_time_window).grid(row=0, column=4, padx=5)
        
        # Navigation buttons
        nav_frame = ttk.Frame(parent)
        nav_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(nav_frame, text="◄◄ Prev", command=self._prev_window).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Next ►►", command=self._next_window).pack(side=tk.LEFT, padx=2)
        
        # Quick navigation
        ttk.Label(nav_frame, text="Step:").pack(side=tk.LEFT, padx=(10, 2))
        self.step_entry = ttk.Entry(nav_frame, width=10)
        self.step_entry.insert(0, str(self.step))
        self.step_entry.pack(side=tk.LEFT, padx=2)
        
        # Class selection frame
        class_frame = ttk.LabelFrame(parent, text="Current Label", padding=5)
        class_frame.pack(side=tk.LEFT, padx=5)
        
        self.current_class_var = tk.StringVar(value=self.classes[0])
        class_combo = ttk.Combobox(class_frame, textvariable=self.current_class_var, 
                                   values=self.classes, state='readonly', width=15)
        class_combo.pack(side=tk.LEFT, padx=2)
        
        # Quick action buttons
        action_frame = ttk.Frame(parent)
        action_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(action_frame, text="Add Label", command=self._add_interval).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Delete", command=self._delete_interval).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Undo", command=self._undo).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Redo", command=self._redo).pack(side=tk.LEFT, padx=2)
    
    def _build_plot(self, parent):
        """Build the matplotlib figure and canvas."""
        # Create figure with user panels + annotation strip
        self.fig = plt.Figure(figsize=(14, 8))
        
        # Grid: 4 rows for user, 1 for annotation strip
        gs = self.fig.add_gridspec(5, 1, height_ratios=[3, 3, 3, 3, 1], hspace=0.3)
        
        # User axes (customizable; here we provide 2 standard panels)
        self.user_axes = {
            'panel1': self.fig.add_subplot(gs[0, 0]),
            'panel2': self.fig.add_subplot(gs[1, 0]),
        }
        
        # Share x-axis among all user axes
        for i, (key, ax) in enumerate(list(self.user_axes.items())[1:]):
            ax.sharex(self.user_axes['panel1'])
        
        # Annotation strip
        self.strip_ax = self.fig.add_subplot(gs[4, 0], sharex=self.user_axes['panel1'])
        self.strip_ax.set_ylabel('Labels', fontsize=9)
        self.strip_ax.set_ylim(0, 1)
        self.strip_ax.set_yticks([])
        self.strip_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()
        
        # Rectangle selector on the first user panel (but works across all due to shared x-axis)
        self.rect_selector = RectangleSelector(
            self.user_axes['panel1'],
            self._on_rectangle_select,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=False,
            props=dict(facecolor='yellow', edgecolor='orange', 
                      alpha=0.3, linestyle='--', linewidth=2)
        )
        
        # Store for rectangle coordinates
        self.current_selection = None
    
    def _build_sidebar(self, parent):
        """Build the sidebar with intervals list and controls."""
        # Intervals list
        intervals_frame = ttk.LabelFrame(parent, text="Labeled Intervals", padding=5)
        intervals_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Create treeview for intervals
        columns = ('Start', 'End', 'Label', 'Duration')
        self.intervals_tree = ttk.Treeview(intervals_frame, columns=columns, 
                                          show='tree headings', height=15)
        
        # Column headings
        self.intervals_tree.heading('#0', text='#')
        self.intervals_tree.heading('Start', text='Start')
        self.intervals_tree.heading('End', text='End')
        self.intervals_tree.heading('Label', text='Label')
        self.intervals_tree.heading('Duration', text='Duration')
        
        # Column widths
        self.intervals_tree.column('#0', width=30)
        self.intervals_tree.column('Start', width=80)
        self.intervals_tree.column('End', width=80)
        self.intervals_tree.column('Label', width=80)
        self.intervals_tree.column('Duration', width=60)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(intervals_frame, orient=tk.VERTICAL, 
                                 command=self.intervals_tree.yview)
        self.intervals_tree.configure(yscrollcommand=scrollbar.set)
        
        self.intervals_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event
        self.intervals_tree.bind('<<TreeviewSelect>>', self._on_interval_tree_select)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(parent, text="Statistics", padding=5)
        stats_frame.pack(fill=tk.X, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=8, width=30, state='disabled')
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Action buttons frame
        action_frame = ttk.LabelFrame(parent, text="Actions", padding=5)
        action_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(action_frame, text="Relabel Selected", 
                  command=self._relabel_interval).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Delete Selected", 
                  command=self._delete_interval).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Assign Remainder → UNKNOWN", 
                  command=self._assign_remainder).pack(fill=tk.X, pady=2)
        
        ttk.Separator(action_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        ttk.Button(action_frame, text="Clear All Intervals", 
                  command=self._clear_all_intervals).pack(fill=tk.X, pady=2)
        
        # File operations
        file_frame = ttk.LabelFrame(parent, text="File Operations", padding=5)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_frame, text="Save Session", 
                  command=self._save_session).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Load Session", 
                  command=self._load_session).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Export Intervals", 
                  command=self._export_intervals).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Export Per-Sample", 
                  command=self._export_per_sample).pack(fill=tk.X, pady=2)
        
        # Options
        options_frame = ttk.LabelFrame(parent, text="Options", padding=5)
        options_frame.pack(fill=tk.X, pady=5)
        
        self.snap_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Snap to samples", 
                       variable=self.snap_var).pack(anchor=tk.W)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(parent, textvariable=self.status_var, 
                               relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _update_time_window(self):
        """Update the time window from text entry boxes."""
        try:
            new_t0 = pd.to_datetime(self.start_time_entry.get())
            new_t1 = pd.to_datetime(self.end_time_entry.get())
            
            if new_t0 >= new_t1:
                messagebox.showerror("Invalid Range", "Start time must be before end time.")
                return
            
            self.t0 = max(new_t0, self.data_start)
            self.t1 = min(new_t1, self.data_end)
            
            # Update entry boxes with clipped values
            self.start_time_entry.delete(0, tk.END)
            self.start_time_entry.insert(0, str(self.t0))
            self.end_time_entry.delete(0, tk.END)
            self.end_time_entry.insert(0, str(self.t1))
            
            self._update_plot()
            self.status_var.set(f"Window updated: {self.t0.strftime('%H:%M:%S')} to {self.t1.strftime('%H:%M:%S')}")
        except Exception as e:
            messagebox.showerror("Invalid Time Format", f"Could not parse time: {e}")
    
    def _on_interval_tree_select(self, event):
        """Handle selection in the intervals tree."""
        selection = self.intervals_tree.selection()
        if not selection:
            self.selected_interval = None
            return
        
        # Get the selected interval index
        item = selection[0]
        try:
            idx = int(self.intervals_tree.item(item)['text']) - 1
            if 0 <= idx < len(self.intervals):
                self.selected_interval = self.intervals[idx]
                self.status_var.set(f"Selected: {self.selected_interval.label} "
                                  f"[{self.selected_interval.start.strftime('%H:%M:%S')} to {self.selected_interval.end.strftime('%H:%M:%S')}]")
                self._update_strip()
                self.canvas.draw()
        except (ValueError, IndexError):
            self.selected_interval = None
    
    def _clear_all_intervals(self):
        """Clear all intervals after confirmation."""
        if not self.intervals:
            return
        
        response = messagebox.askyesno("Clear All", 
                                      f"Are you sure you want to delete all {len(self.intervals)} intervals?")
        if response:
            self.intervals.clear()
            self.selected_interval = None
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.modified = True
            self._update_plot()
            self._update_intervals_list()
            self.status_var.set("All intervals cleared")
    
    def _update_intervals_list(self):
        """Update the intervals treeview with current intervals."""
        # Clear existing items
        for item in self.intervals_tree.get_children():
            self.intervals_tree.delete(item)
        
        # Add intervals
        for i, interval in enumerate(self.intervals):
            duration = interval.end - interval.start
            
            # Get color for this label
            color = self.class_colors.get(interval.label, '#cccccc')
            
            # Format times
            start_str = interval.start.strftime('%H:%M:%S')
            end_str = interval.end.strftime('%H:%M:%S')
            duration_str = str(duration).split('.')[0]  # Remove microseconds
            
            item_id = self.intervals_tree.insert('', 'end', text=str(i+1),
                                                values=(start_str, end_str, 
                                                       interval.label, duration_str),
                                                tags=(interval.label,))
            
            # Color code by label
            self.intervals_tree.tag_configure(interval.label, background=color)
        
        # Update statistics
        self._update_statistics()
    
    def _update_statistics(self):
        """Update the statistics text widget."""
        self.stats_text.config(state='normal')
        self.stats_text.delete(1.0, tk.END)
        
        if not self.intervals:
            self.stats_text.insert(tk.END, "No intervals labeled yet.")
            self.stats_text.config(state='disabled')
            return
        
        # Calculate statistics
        total_duration = self.data_end - self.data_start
        labeled_duration = sum((iv.end - iv.start for iv in self.intervals), 
                              pd.Timedelta(0))
        
        # Count by label
        label_counts = {}
        label_durations = {}
        for interval in self.intervals:
            label = interval.label
            duration = interval.end - interval.start
            label_counts[label] = label_counts.get(label, 0) + 1
            label_durations[label] = label_durations.get(label, pd.Timedelta(0)) + duration
        
        # Display
        self.stats_text.insert(tk.END, f"Total Intervals: {len(self.intervals)}\n")
        self.stats_text.insert(tk.END, f"Labeled: {labeled_duration} / {total_duration}\n")
        pct = (labeled_duration / total_duration * 100) if total_duration > pd.Timedelta(0) else 0
        self.stats_text.insert(tk.END, f"Coverage: {pct:.1f}%\n\n")
        
        self.stats_text.insert(tk.END, "By Label:\n")
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            duration = label_durations[label]
            pct = (duration / total_duration * 100) if total_duration > pd.Timedelta(0) else 0
            self.stats_text.insert(tk.END, f"  {label}: {count} intervals, {pct:.1f}%\n")
        
        self.stats_text.config(state='disabled')
    
    def _on_rectangle_select(self, eclick, erelease):
        """Callback when user finishes dragging a rectangle."""
        # Get x coordinates (time) from the rectangle
        x1, x2 = sorted([eclick.xdata, erelease.xdata])
        
        # Convert matplotlib date numbers to timestamps
        t_start = pd.Timestamp(num2date(x1))
        t_end = pd.Timestamp(num2date(x2))
        
        # Apply snapping if enabled
        if self.snap_var.get():
            t_start, t_end = self._snap_to_samples(t_start, t_end)
        
        # Store selection
        self.current_selection = (t_start, t_end)
        self.status_var.set(f"Selected: {t_start.strftime('%H:%M:%S')} to {t_end.strftime('%H:%M:%S')}")
        
        # Update strip to show preview
        self._update_strip()
        self.canvas.draw()
    
    def _update_plot(self):
        """Redraw the user panels and annotation strip."""
        # Clear user axes
        for ax in self.user_axes.values():
            ax.clear()
        
        # Call user's plot function
        try:
            sub_df = self.df.loc[self.t0:self.t1]
            self.plot_fn(self.user_axes, sub_df, self.t0, self.t1)
        except Exception as e:
            print(f"Error in plot_fn: {e}")
            for ax in self.user_axes.values():
                ax.text(0.5, 0.5, f"Plot error:\n{e}", 
                       transform=ax.transAxes, ha='center', va='center')
        
        # CRITICAL: Ensure all axes have the same x-axis limits
        xlim = mdates.date2num([self.t0, self.t1])
        for ax in list(self.user_axes.values()) + [self.strip_ax]:
            ax.set_xlim(xlim)
        
        # Format time axis consistently
        for ax in self.user_axes.values():
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            # Add some padding so data doesn't hug edges
            ax.margins(x=0.01)
        
        # Update annotation strip
        self._update_strip()
        
        # Update intervals list
        self._update_intervals_list()
        
        # Ensure tight layout
        self.fig.tight_layout()
        
        self.canvas.draw()
    
    def _update_strip(self):
        """Redraw the annotation strip with current intervals."""
        self.strip_ax.clear()
        self.strip_ax.set_ylim(0, 1)
        self.strip_ax.set_yticks([])
        self.strip_ax.set_ylabel('Labels', fontsize=9)
        
        # Draw intervals that overlap current window
        for interval in self.intervals:
            if interval.end <= self.t0 or interval.start >= self.t1:
                continue  # Outside window
            
            # Clip to window
            start = max(interval.start, self.t0)
            end = min(interval.end, self.t1)
            
            color = self.class_colors.get(interval.label, '#cccccc')
            
            # Highlight if selected
            alpha = 0.8 if interval == self.selected_interval else 0.6
            edgecolor = 'red' if interval == self.selected_interval else 'black'
            linewidth = 2 if interval == self.selected_interval else 0.5
            
            rect = Rectangle(
                (mdates.date2num(start), 0.1),
                mdates.date2num(end) - mdates.date2num(start),
                0.8,
                facecolor=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
                picker=True
            )
            self.strip_ax.add_patch(rect)
        
        # Draw current selection preview (if any)
        if self.current_selection:
            start, end = self.current_selection
            rect = Rectangle(
                (mdates.date2num(start), 0.05),
                mdates.date2num(end) - mdates.date2num(start),
                0.9,
                facecolor='yellow',
                edgecolor='orange',
                linewidth=2,
                alpha=0.3,
                linestyle='--'
            )
            self.strip_ax.add_patch(rect)
        
        # Handle clicks on strip
        self.canvas.mpl_connect('pick_event', self._on_strip_click)
    
    def _snap_to_samples(self, t_start: pd.Timestamp, t_end: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Snap timestamps to nearest sample times in the current window."""
        sub = self.df.loc[self.t0:self.t1]
        if len(sub) == 0:
            return t_start, t_end
        
        # Find nearest samples
        idx_start = sub.index[sub.index.get_indexer([t_start], method='nearest')[0]]
        idx_end = sub.index[sub.index.get_indexer([t_end], method='nearest')[0]]
        
        return idx_start, idx_end
    
    def _on_strip_click(self, event):
        """Handle clicks on interval rectangles in the strip."""
        if event.artist not in self.strip_ax.patches:
            return
        
        # Find which interval was clicked
        click_time = pd.Timestamp(num2date(event.mouseevent.xdata))
        
        for interval in self.intervals:
            if interval.contains(click_time):
                self.selected_interval = interval
                self.status_var.set(f"Selected: {interval.label} [{interval.start.strftime('%H:%M:%S')} to {interval.end.strftime('%H:%M:%S')}]")
                self._update_strip()
                self.canvas.draw()
                break
    
    def _add_interval(self):
        """Add a new interval from current selection."""
        if not self.current_selection:
            messagebox.showwarning("No Selection", "Please drag on the plot to select a time range.")
            return
        
        start, end = self.current_selection
        label = self.current_class_var.get()
        
        interval = Interval(start, end, label)
        
        # Execute command
        cmd = AddIntervalCommand(self, interval)
        self._execute_command(cmd)
        
        # Clear selection
        self.current_selection = None
        self.status_var.set(f"Added {label} interval")
        
        self._update_plot()
        self._maybe_autosave()
    
    def _relabel_interval(self):
        """Relabel the selected interval."""
        if not self.selected_interval:
            messagebox.showwarning("No Selection", "Please select an interval from the list.")
            return
        
        new_label = self.current_class_var.get()
        
        cmd = RelabelIntervalCommand(self, self.selected_interval, new_label)
        self._execute_command(cmd)
        
        self.status_var.set(f"Relabeled to {new_label}")
        self._update_plot()
        self._maybe_autosave()
    
    def _delete_interval(self):
        """Delete the selected interval."""
        if not self.selected_interval:
            messagebox.showwarning("No Selection", "Please select an interval from the list.")
            return
        
        cmd = DeleteIntervalCommand(self, self.selected_interval)
        self._execute_command(cmd)
        
        self.selected_interval = None
        self.status_var.set("Deleted interval")
        self._update_plot()
        self._maybe_autosave()
    
    def _assign_remainder(self):
        """Label all unlabeled time in current window as UNKNOWN."""
        if "UNKNOWN" not in self.classes:
            messagebox.showwarning("No UNKNOWN Class", "UNKNOWN class not defined.")
            return
        
        # Find covered regions
        covered = []
        for interval in self.intervals:
            if interval.end <= self.t0 or interval.start >= self.t1:
                continue
            start = max(interval.start, self.t0)
            end = min(interval.end, self.t1)
            covered.append((start, end))
        
        # Sort and merge
        covered.sort()
        merged = []
        for start, end in covered:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        
        # Find gaps
        gaps = []
        current = self.t0
        for start, end in merged:
            if current < start:
                gaps.append((current, start))
            current = max(current, end)
        if current < self.t1:
            gaps.append((current, self.t1))
        
        # Add UNKNOWN intervals
        for start, end in gaps:
            interval = Interval(start, end, "UNKNOWN")
            cmd = AddIntervalCommand(self, interval)
            self._execute_command(cmd)
        
        self.status_var.set(f"Assigned {len(gaps)} UNKNOWN intervals")
        self._update_plot()
        self._maybe_autosave()
    
    def _execute_command(self, cmd: Command):
        """Execute a command and add to undo stack."""
        cmd.execute()
        self.undo_stack.append(cmd)
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.modified = True
    
    def _undo(self):
        """Undo the last operation."""
        if not self.undo_stack:
            self.status_var.set("Nothing to undo")
            return
        
        cmd = self.undo_stack.pop()
        cmd.undo()
        self.redo_stack.append(cmd)
        
        self.status_var.set("Undo")
        self._update_plot()
        self._maybe_autosave()
    
    def _redo(self):
        """Redo the last undone operation."""
        if not self.redo_stack:
            self.status_var.set("Nothing to redo")
            return
        
        cmd = self.redo_stack.pop()
        cmd.execute()
        self.undo_stack.append(cmd)
        
        self.status_var.set("Redo")
        self._update_plot()
        self._maybe_autosave()
    
    def _remove_overlapping_intervals(self, new_interval: Interval) -> List[Interval]:
        """
        Remove parts of existing intervals that overlap with new_interval.
        Returns list of removed/modified intervals for undo.
        """
        removed = []
        to_add = []
        
        for interval in self.intervals[:]:
            if not interval.overlaps(new_interval):
                continue
            
            # Remove the overlapping interval
            self.intervals.remove(interval)
            removed.append(interval)
            
            # Add back non-overlapping parts
            if interval.start < new_interval.start:
                # Left part
                to_add.append(Interval(interval.start, new_interval.start, interval.label, interval.notes))
            
            if interval.end > new_interval.end:
                # Right part
                to_add.append(Interval(new_interval.end, interval.end, interval.label, interval.notes))
        
        # Add the trimmed intervals
        self.intervals.extend(to_add)
        
        return removed
    
    def _sort_and_merge_intervals(self):
        """Sort intervals and merge adjacent ones with the same label."""
        if not self.intervals:
            return
        
        # Sort by start time
        self.intervals.sort(key=lambda x: x.start)
        
        # Merge adjacent same-label intervals
        merged = [self.intervals[0]]
        
        for interval in self.intervals[1:]:
            last = merged[-1]
            
            # Check if adjacent and same label
            if (interval.start == last.end and 
                interval.label == last.label):
                # Merge
                last.end = interval.end
            else:
                merged.append(interval)
        
        self.intervals = merged
    
    def _prev_window(self):
        """Navigate to previous window."""
        # Get step from entry
        try:
            self.step = pd.Timedelta(self.step_entry.get())
        except:
            pass
        
        # Calculate current window size
        self.window = self.t1 - self.t0
        
        self.t0 -= self.step
        self.t1 = self.t0 + self.window
        
        # Clip to data bounds
        if self.t0 < self.data_start:
            self.t0 = self.data_start
            self.t1 = min(self.t0 + self.window, self.data_end)
        
        # Update entry boxes
        self.start_time_entry.delete(0, tk.END)
        self.start_time_entry.insert(0, str(self.t0))
        self.end_time_entry.delete(0, tk.END)
        self.end_time_entry.insert(0, str(self.t1))
        
        self._update_plot()
        self.status_var.set(f"Window: {self.t0.strftime('%H:%M:%S')} to {self.t1.strftime('%H:%M:%S')}")
    
    def _next_window(self):
        """Navigate to next window."""
        # Get step from entry
        try:
            self.step = pd.Timedelta(self.step_entry.get())
        except:
            pass
        
        # Calculate current window size
        self.window = self.t1 - self.t0
        
        self.t0 += self.step
        self.t1 = self.t0 + self.window
        
        # Clip to data bounds
        if self.t1 > self.data_end:
            self.t1 = self.data_end
            self.t0 = max(self.t1 - self.window, self.data_start)
        
        # Update entry boxes
        self.start_time_entry.delete(0, tk.END)
        self.start_time_entry.insert(0, str(self.t0))
        self.end_time_entry.delete(0, tk.END)
        self.end_time_entry.insert(0, str(self.t1))
        
        self._update_plot()
        self.status_var.set(f"Window: {self.t0.strftime('%H:%M:%S')} to {self.t1.strftime('%H:%M:%S')}")
    
    def _on_key_press(self, event):
        """Handle keyboard shortcuts."""
        key = event.keysym
        
        # Class selection: 1-9
        if key.isdigit() and int(key) > 0:
            idx = int(key) - 1
            if idx < len(self.classes):
                self.current_class_var.set(self.classes[idx])
                self.status_var.set(f"Selected class: {self.classes[idx]}")
        
        # Navigation
        elif key in ('n', 'N', 'Right'):
            self._next_window()
        elif key in ('p', 'P', 'Left'):
            self._prev_window()
        
        # Actions
        elif key in ('a', 'A', 'Return'):
            self._add_interval()
        elif key in ('d', 'D', 'Delete'):
            self._delete_interval()
        elif key in ('u', 'U'):
            if "UNKNOWN" in self.classes:
                self.current_class_var.set("UNKNOWN")
                self.status_var.set("Selected class: UNKNOWN")
        elif key in ('s', 'S') and event.state & 0x4:  # Ctrl+S
            self._save_session()
        elif key in ('e', 'E') and event.state & 0x4:  # Ctrl+E
            self._export_intervals()
        
        # Undo/Redo
        elif key == 'z' and event.state & 0x4:  # Ctrl+Z
            self._undo()
        elif key == 'y' and event.state & 0x4:  # Ctrl+Y
            self._redo()
    
    def _save_session(self, path: Optional[str] = None):
        """Save session to JSON."""
        if path is None:
            path = self.autosave_path
        
        if path is None:
            path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not path:
                return
            self.autosave_path = Path(path)
        
        data = {
            'version': 1,
            'classes': self.classes,
            'class_colors': self.class_colors,
            'window': str(self.window),
            'step': str(self.step),
            'data_start': self.data_start.isoformat(),
            'data_end': self.data_end.isoformat(),
            'intervals': [iv.to_dict() for iv in self.intervals]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.modified = False
        self.status_var.set(f"Saved to {path}")
    
    def _load_session(self, path: Optional[str] = None):
        """Load session from JSON."""
        if path is None:
            path = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not path:
                return
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Restore settings
        self.classes = data['classes']
        self.class_colors = data['class_colors']
        self.window = pd.Timedelta(data['window'])
        self.step = pd.Timedelta(data['step'])
        
        # Restore intervals
        self.intervals = [Interval.from_dict(iv) for iv in data['intervals']]
        
        self.autosave_path = Path(path)
        self.modified = False
        
        # Update GUI entry boxes
        self.start_time_entry.delete(0, tk.END)
        self.start_time_entry.insert(0, str(self.t0))
        self.end_time_entry.delete(0, tk.END)
        self.end_time_entry.insert(0, str(self.t1))
        self.step_entry.delete(0, tk.END)
        self.step_entry.insert(0, str(self.step))
        
        self._update_plot()
        self.status_var.set(f"Loaded from {path}")
    
    def _maybe_autosave(self):
        """Autosave if path is set."""
        if self.autosave_path and self.modified:
            self._save_session(str(self.autosave_path))
    
    def _export_intervals(self):
        """Export intervals to CSV or Parquet."""
        if not self.intervals:
            messagebox.showwarning("No Data", "No intervals to export.")
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Parquet files", "*.parquet"), ("All files", "*.*")]
        )
        if not path:
            return
        
        # Create DataFrame
        data = []
        for iv in self.intervals:
            data.append({
                'start': iv.start,
                'end': iv.end,
                'label': iv.label,
                'notes': iv.notes
            })
        
        df_export = pd.DataFrame(data)
        
        # Save
        if path.endswith('.parquet'):
            df_export.to_parquet(path, index=False)
        else:
            df_export.to_csv(path, index=False)
        
        self.status_var.set(f"Exported to {path}")
        messagebox.showinfo("Export Complete", f"Intervals exported to {path}")
    
    def _export_per_sample(self):
        """Export per-sample labels aligned to dataframe index."""
        if not self.intervals:
            messagebox.showwarning("No Data", "No intervals to export.")
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Parquet files", "*.parquet"), ("All files", "*.*")]
        )
        if not path:
            return
        
        # Assign label to each timestamp
        labels = []
        for ts in self.df.index:
            label = None
            for iv in self.intervals:
                if iv.contains(ts):
                    label = iv.label
                    break
            labels.append(label if label else "UNKNOWN")
        
        df_export = pd.DataFrame({'label': labels}, index=self.df.index)
        
        # Save
        if path.endswith('.parquet'):
            df_export.to_parquet(path)
        else:
            df_export.to_csv(path)
        
        self.status_var.set(f"Exported to {path}")
        messagebox.showinfo("Export Complete", f"Per-sample labels exported to {path}")
    
    def _on_closing(self):
        """Handle window close event."""
        if self.modified:
            response = messagebox.askyesnocancel("Save Changes?", "Do you want to save changes before closing?")
            if response is None:  # Cancel
                return
            elif response:  # Yes
                self._save_session()
        
        self.root.destroy()
    
    def go_to_window(self, t0: pd.Timestamp):
        """Jump to a specific window start time."""
        self.t0 = max(t0, self.data_start)
        self.t1 = min(self.t0 + self.window, self.data_end)
        
        # Update entry boxes
        if self.start_time_entry:
            self.start_time_entry.delete(0, tk.END)
            self.start_time_entry.insert(0, str(self.t0))
            self.end_time_entry.delete(0, tk.END)
            self.end_time_entry.insert(0, str(self.t1))
        
        self._update_plot()
    
    def save(self, path: Optional[str] = None):
        """Public API: Save session."""
        self._save_session(path)
    
    def load(self, path: str):
        """Public API: Load session."""
        self._load_session(path)
    
    def export_intervals(self, path: str, fmt: str = "parquet"):
        """
        Public API: Export intervals.
        
        Parameters
        ----------
        path : str
            Output file path.
        fmt : str, optional
            Format: 'parquet' or 'csv' (default: 'parquet').
        """
        if not self.intervals:
            print("No intervals to export.")
            return
        
        data = [{'start': iv.start, 'end': iv.end, 'label': iv.label, 'notes': iv.notes}
                for iv in self.intervals]
        df_export = pd.DataFrame(data)
        
        if fmt == "parquet":
            df_export.to_parquet(path, index=False)
        else:
            df_export.to_csv(path, index=False)
        
        print(f"Exported intervals to {path}")
    
    def export_per_sample(self, path: str, fmt: str = "parquet", label_on_uncovered: str = "UNKNOWN"):
        """
        Public API: Export per-sample labels.
        
        Parameters
        ----------
        path : str
            Output file path.
        fmt : str, optional
            Format: 'parquet' or 'csv' (default: 'parquet').
        label_on_uncovered : str or None, optional
            Label for uncovered timestamps (default: 'UNKNOWN').
        """
        labels = []
        for ts in self.df.index:
            label = None
            for iv in self.intervals:
                if iv.contains(ts):
                    label = iv.label
                    break
            labels.append(label if label else label_on_uncovered)
        
        df_export = pd.DataFrame({'label': labels}, index=self.df.index)
        
        if fmt == "parquet":
            df_export.to_parquet(path)
        else:
            df_export.to_csv(path)
        
        print(f"Exported per-sample labels to {path}")


if __name__ == "__main__":
    # Simple test
    print("Time Interval Labeler module loaded successfully.")
    print("Import this module and use TimeIntervalLabeler class.")
