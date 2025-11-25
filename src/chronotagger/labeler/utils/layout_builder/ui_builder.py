"""
UI Builder Mixin for Layout Builder

This module provides functionality for building and managing the user interface
components of the layout builder dialog, including grid controls, variable
assignment, and panel editing controls.

Classes:
    UIBuilderMixin: Handles UI construction and updates
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import tkinter as tk
from tkinter import ttk

if TYPE_CHECKING:
    from .models import PanelConfig


class UIBuilderMixin:
    """
    Mixin providing UI building and management functionality.

    This mixin handles the construction of the dialog's user interface,
    including:
    1. Main layout with grid canvas and control panels
    2. Grid size controls
    3. Variable assignment UI (changes based on selected role)
    4. Panel editing controls

    The mixin expects the parent class to provide:
    - self.nrows_var: tk.IntVar
    - self.ncols_var: tk.IntVar
    - self.role_var: tk.StringVar
    - self.numeric_columns: List[str]
    - self.canvas: tk.Canvas
    - self.panel_listbox: tk.Listbox
    - Various callback methods
    """

    def _build_ui(self):
        """Build the complete dialog UI."""
        # Main container
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # === LEFT PANEL: Grid builder ===
        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Grid controls
        controls = ttk.Frame(left)
        controls.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(controls, text="Grid Size:", font=('', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(controls, text="Rows:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(
            controls,
            from_=2, to=10,  # Minimum 2 rows (1 for user, 1 for Labels)
            textvariable=self.nrows_var,
            width=5,
            command=self._on_grid_size_changed
        ).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(controls, text="Columns:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(
            controls,
            from_=1, to=5,
            textvariable=self.ncols_var,
            width=5,
            command=self._on_grid_size_changed
        ).pack(side=tk.LEFT, padx=(0, 15))

        tk.Button(controls, text="Clear All", command=self._clear_all_panels).pack(side=tk.RIGHT)

        # Instructions
        instructions = ttk.Label(
            left,
            text="📌 Click panel to select • Drag empty cells to create new panel\nBottom row (Labels) is auto-managed",
            font=('', 9, 'italic'),
            foreground='#666',
            justify=tk.LEFT
        )
        instructions.pack(fill=tk.X, pady=(0, 5))

        # Canvas for grid
        canvas_frame = ttk.Frame(left, relief=tk.SUNKEN, borderwidth=2)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            canvas_frame,
            bg='white',
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events for drag-to-span
        self.canvas.bind('<Button-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)

        # === RIGHT PANEL: Panel configuration ===
        right = ttk.Frame(main, width=320)
        right.pack(side=tk.RIGHT, fill=tk.BOTH)
        right.pack_propagate(False)

        # Panel list
        list_frame = ttk.LabelFrame(right, text="Panels", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.panel_listbox = tk.Listbox(list_frame, height=8)
        self.panel_listbox.pack(fill=tk.BOTH, expand=True)
        self.panel_listbox.bind('<<ListboxSelect>>', self._on_panel_select)

        # Add/Edit panel controls (DUAL PURPOSE)
        add_frame = ttk.LabelFrame(right, text="Add/Edit Panel", padding=10)
        add_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(add_frame, text="1. Select role and variable").pack(anchor='w', pady=(0, 5))
        ttk.Label(add_frame, text="2. Drag to add OR click Update to edit").pack(anchor='w', pady=(0, 10))

        # Role selection
        ttk.Label(add_frame, text="Role:", font=('', 9, 'bold')).pack(anchor='w', pady=(5, 2))
        self.role_var = tk.StringVar(value="time")

        role_frame = ttk.Frame(add_frame)
        role_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Radiobutton(
            role_frame,
            text="Time Series",
            value="time",
            variable=self.role_var,
            command=self._on_role_changed
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(
            role_frame,
            text="Cross-Plot",
            value="not-time",
            variable=self.role_var,
            command=self._on_role_changed
        ).pack(side=tk.LEFT)

        # Variable assignment frame
        self.vars_frame = ttk.Frame(add_frame)
        self.vars_frame.pack(fill=tk.X)

        # Edit panel controls
        edit_frame = ttk.LabelFrame(right, text="Edit Selected Panel", padding=10)
        edit_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Button(edit_frame, text="Update Panel", command=self._update_selected_panel).pack(fill=tk.X, pady=(0, 5))
        tk.Button(edit_frame, text="Delete Panel", command=self._delete_selected_panel).pack(fill=tk.X)

        # Bottom buttons
        bottom_frame = ttk.Frame(right)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Button(bottom_frame, text="Preview", command=self._show_preview, width=12).pack(
            side=tk.LEFT, padx=2
        )
        tk.Button(bottom_frame, text="Done", command=self._on_done, width=12).pack(
            side=tk.RIGHT, padx=2
        )
        tk.Button(bottom_frame, text="Cancel", command=self._on_cancel, width=12).pack(
            side=tk.RIGHT, padx=2
        )

        # Initialize variable assignment UI
        self._rebuild_vars_ui()

        # Draw initial grid
        self._redraw_grid()

    def _rebuild_vars_ui(self):
        """Rebuild variable assignment UI based on selected role."""
        # Clear existing widgets
        for widget in self.vars_frame.winfo_children():
            widget.destroy()

        role = self.role_var.get()

        if role == "time":
            # Time plot: only need Y variable (X is always df.index)
            ttk.Label(self.vars_frame, text="Y-axis variable:", font=('', 9, 'bold')).pack(anchor='w', pady=(0, 2))
            self.y_var = tk.StringVar()
            ttk.Combobox(
                self.vars_frame,
                textvariable=self.y_var,
                values=self.numeric_columns,
                state='readonly',
                width=22
            ).pack(fill=tk.X)

        else:  # not-time
            # Cross-plot: need both X and Y variables
            ttk.Label(self.vars_frame, text="X-axis variable:", font=('', 9, 'bold')).pack(anchor='w', pady=(5, 2))
            self.x_var = tk.StringVar()
            ttk.Combobox(
                self.vars_frame,
                textvariable=self.x_var,
                values=self.numeric_columns,
                state='readonly',
                width=22
            ).pack(fill=tk.X)

            ttk.Label(self.vars_frame, text="Y-axis variable:", font=('', 9, 'bold')).pack(anchor='w', pady=(5, 2))
            self.y2_var = tk.StringVar()
            ttk.Combobox(
                self.vars_frame,
                textvariable=self.y2_var,
                values=self.numeric_columns,
                state='readonly',
                width=22
            ).pack(fill=tk.X)

    def _on_role_changed(self):
        """Handle role radio button change."""
        self._rebuild_vars_ui()

    def _on_grid_size_changed(self):
        """Handle grid size changes - update Labels panel position."""
        self._update_labels_panel()
        self._redraw_grid()
