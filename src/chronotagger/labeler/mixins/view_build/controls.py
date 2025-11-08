"""
Top control bar construction mixin.

Responsibilities:
- Build the top control bar with all its sections
- Create time range controls, navigation buttons, label actions, I/O, and help sections

This mixin provides the _build_top_controls method that creates the entire top toolbar.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ControlsMixin:
    """
    Mixin providing top control bar construction.

    This mixin expects the following attributes/methods to be available on self:
    - t0: float - Initial start time
    - t1: float - Initial end time
    - step: float - Initial step size
    - classes: list[str] - List of label classes
    - root: tk.Tk - The main tkinter root window
    - _update_time_window() - Callback for updating time window
    - _double_time_window() - Callback for doubling time window
    - _halve_time_window() - Callback for halving time window
    - _prev_window() - Callback for previous window
    - _next_window() - Callback for next window
    - _apply_step_entry() - Callback for applying step entry
    - _double_step() - Callback for doubling step
    - _halve_step() - Callback for halving step
    - _relabel_interval() - Callback for relabeling interval
    - _add_interval() - Callback for adding interval
    - _undo() - Callback for undo
    - _open_label_manager() - Callback for opening label manager
    - _open_label_unassigned_dialog() - Callback for opening fill gaps dialog
    - _delete_interval() - Callback for deleting interval
    - _redo() - Callback for redo
    - _open_label_by_rule_dialog() - Callback for opening by-rule dialog
    - _open_clear_intervals_dialog() - Callback for opening clear intervals dialog
    - _save_session() - Callback for saving session
    - _load_session() - Callback for loading session
    - _export_labels_dialog() - Callback for exporting labels
    - _open_help_dialog() - Callback for opening help dialog
    - _reset_all_yscales() - Callback for resetting all y-scales
    - _toggle_sidebar() - Callback for toggling sidebar
    - _create_tooltip(widget, text) - Method to create tooltips

    Attributes created:
    - start_time_entry: ttk.Entry - Entry for start time
    - end_time_entry: ttk.Entry - Entry for end time
    - step_entry: ttk.Entry - Entry for step size
    - current_class_var: tk.StringVar - StringVar for current class selection
    - class_combo: ttk.Combobox - Combobox for class selection
    - sidebar_toggle_btn: ttk.Button - Button for toggling sidebar
    """

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
