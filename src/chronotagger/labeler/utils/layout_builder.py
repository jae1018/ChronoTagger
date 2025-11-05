"""
Visual Layout Builder for ChronoTagger

A Tkinter-based GUI that allows users to interactively design their plot layout
by dragging on a grid. This eliminates the need to manually write layout_spec 
dictionaries.

Key Features:
- Visual grid builder with drag-to-span panels
- Variable assignment (which columns to plot)
- Role selection (time vs not-time plots)
- Panel editing capability
- Generates both layout_spec and plot_config

Usage:
    from chronotagger.labeler.utils import build_layout
    
    layout_spec, plot_config = build_layout(df)
    # Returns None, None if user cancels

Author: ChronoTagger Team
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Any
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


@dataclass
class PanelConfig:
    """
    Configuration for a single panel in the layout.
    
    Attributes:
        key: Unique identifier (e.g., "panel_1" or "labels")
        row: Row position in grid (0-indexed)
        col: Column position in grid (0-indexed)
        rowspan: Number of rows this panel spans (default: 1)
        colspan: Number of columns this panel spans (default: 1)
        role: "time", "not-time", or "labels"
        locked: If True, panel cannot be edited or deleted (for Labels strip)
        y_column: Column name for y-axis (time plots) or None
        x_column: Column name for x-axis (not-time plots) or None
        y_column_2: Column name for y-axis (not-time plots) or None
    """
    key: str
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    role: str = "time"
    locked: bool = False
    y_column: Optional[str] = None
    x_column: Optional[str] = None
    y_column_2: Optional[str] = None
    
    def overlaps(self, other: 'PanelConfig') -> bool:
        """Check if this panel overlaps with another panel."""
        # Get the set of cells occupied by each panel
        self_rows = set(range(self.row, self.row + self.rowspan))
        self_cols = set(range(self.col, self.col + self.colspan))
        other_rows = set(range(other.row, other.row + other.rowspan))
        other_cols = set(range(other.col, other.col + other.colspan))
        
        # Panels overlap if they share any cells
        return bool(self_rows & other_rows and self_cols & other_cols)


class LayoutBuilderDialog(tk.Toplevel):
    """
    Interactive dialog for building plot layouts.
    
    This dialog presents a visual grid where users can:
    1. Set grid dimensions (rows × columns)
    2. Drag on cells to create spanning panels OR click to select existing panels
    3. Assign DataFrame columns to each panel
    4. Set panel roles (time-series vs cross-plot)
    5. Edit existing panels
    6. Delete panels as needed
    
    The dialog returns both layout_spec (for TimeIntervalLabeler) and
    plot_config (for automatic plot function generation).
    """
    
    # Visual constants
    CELL_SIZE = 80  # pixels per grid cell
    GRID_PADDING = 20  # padding around grid
    COLOR_EMPTY = "#f0f0f0"
    COLOR_TIME = "#cce5ff"  # Light blue
    COLOR_NOT_TIME = "#d4edda"  # Light green
    COLOR_LABELS = "#e0e0e0"  # Light gray for locked Labels strip
    COLOR_HOVER = "#fff3cd"  # Light yellow
    COLOR_SELECTED_OUTLINE = "#ff6600"  # Orange for selected panel
    
    def __init__(self, parent: tk.Tk, df: pd.DataFrame):
        """
        Initialize the layout builder dialog.
        
        Args:
            parent: Parent Tk window
            df: DataFrame to analyze for available columns
        """
        super().__init__(parent)
        self.title("ChronoTagger - Layout Builder")
        self.geometry("1000x700")
        
        # Store data
        self.df = df
        self.result_layout_spec: Optional[Dict] = None
        self.result_plot_config: Optional[Dict] = None
        
        # Get numeric columns from DataFrame
        try:
            import numpy as np
            self.numeric_columns = list(df.select_dtypes(include=[np.number]).columns)
        except Exception:
            self.numeric_columns = list(df.columns)
        
        if not self.numeric_columns:
            messagebox.showerror(
                "No Columns",
                "DataFrame has no numeric columns to plot.",
                parent=self
            )
            self.destroy()
            return
        
        # State
        self.nrows_var = tk.IntVar(value=3)
        self.ncols_var = tk.IntVar(value=2)
        self.panels: List[PanelConfig] = []
        self.selected_panel: Optional[PanelConfig] = None
        self.next_panel_id = 1
        
        # Drag state for spanning panels
        self.drag_start_cell: Optional[Tuple[int, int]] = None
        self.drag_current_cell: Optional[Tuple[int, int]] = None
        self.drag_preview_id: Optional[int] = None
        
        # Preview window reference (only one at a time)
        self.preview_window: Optional[tk.Toplevel] = None
        
        # Build UI
        self._build_ui()
        
        # Auto-create Labels panel (must be after UI build)
        self._create_labels_panel()
        
        # Bind window close
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
    
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
        
        ttk.Button(controls, text="Clear All", command=self._clear_all_panels).pack(side=tk.RIGHT)
        
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
        
        ttk.Button(edit_frame, text="Update Panel", command=self._update_selected_panel).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(edit_frame, text="Delete Panel", command=self._delete_selected_panel).pack(fill=tk.X)
        
        # Bottom buttons
        bottom_frame = ttk.Frame(right)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        ttk.Button(bottom_frame, text="Preview", command=self._show_preview, width=12).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(bottom_frame, text="Done", command=self._on_done, width=12).pack(
            side=tk.RIGHT, padx=2
        )
        ttk.Button(bottom_frame, text="Cancel", command=self._on_cancel, width=12).pack(
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
    
    def _create_labels_panel(self):
        """Create the auto-managed Labels strip panel."""
        # Labels panel always occupies the bottom row
        labels_panel = PanelConfig(
            key="labels",
            row=self.nrows_var.get() - 1,  # Bottom row (0-indexed)
            col=0,
            rowspan=1,
            colspan=1,  # Start with 1 column (will be updated later)
            role="labels",
            locked=True
        )
        self.panels.append(labels_panel)
        self._redraw_grid()
    
    def _on_grid_size_changed(self):
        """Handle grid size changes - update Labels panel position."""
        self._update_labels_panel()
        self._redraw_grid()
    
    def _update_labels_panel(self):
        """Update Labels panel position to match current grid size."""
        # Find Labels panel
        labels = next((p for p in self.panels if p.role == "labels"), None)
        if labels is None:
            return
        
        # Update position to bottom row
        labels.row = self.nrows_var.get() - 1
        
        # For now, keep colspan as-is (will be updated when we calculate time plot span)
        # Ensure it doesn't exceed grid width
        max_colspan = self.ncols_var.get()
        if labels.col + labels.colspan > max_colspan:
            labels.colspan = max_colspan - labels.col
    
    def _redraw_grid(self):
        """Redraw the entire grid and all panels."""
        self.canvas.delete('all')
        
        nrows = self.nrows_var.get()
        ncols = self.ncols_var.get()
        
        # Calculate canvas size
        width = ncols * self.CELL_SIZE + 2 * self.GRID_PADDING
        height = nrows * self.CELL_SIZE + 2 * self.GRID_PADDING
        self.canvas.config(width=width, height=height, scrollregion=(0, 0, width, height))
        
        # Draw grid cells
        for row in range(nrows):
            for col in range(ncols):
                x0 = self.GRID_PADDING + col * self.CELL_SIZE
                y0 = self.GRID_PADDING + row * self.CELL_SIZE
                x1 = x0 + self.CELL_SIZE
                y1 = y0 + self.CELL_SIZE
                
                self.canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill=self.COLOR_EMPTY,
                    outline='#ccc',
                    tags='grid_cell'
                )
        
        # Draw existing panels
        for panel in self.panels:
            self._draw_panel(panel)
    
    def _draw_panel(self, panel: PanelConfig):
        """Draw a panel on the canvas, handling rowspan and colspan."""
        x0 = self.GRID_PADDING + panel.col * self.CELL_SIZE
        y0 = self.GRID_PADDING + panel.row * self.CELL_SIZE
        x1 = x0 + panel.colspan * self.CELL_SIZE
        y1 = y0 + panel.rowspan * self.CELL_SIZE
        
        # Color based on role
        if panel.role == "labels":
            fill_color = self.COLOR_LABELS
        elif panel.role == "time":
            fill_color = self.COLOR_TIME
        else:
            fill_color = self.COLOR_NOT_TIME
        
        # Highlight selected panel with different outline (but not for locked panels)
        outline_color = '#333'
        outline_width = 2
        
        if panel == self.selected_panel and not panel.locked:
            outline_color = self.COLOR_SELECTED_OUTLINE
            outline_width = 4
        
        # Draw rectangle
        rect_id = self.canvas.create_rectangle(
            x0, y0, x1, y1,
            fill=fill_color,
            outline=outline_color,
            width=outline_width,
            tags=f'panel_{panel.key}'
        )
        
        # Add stipple pattern for locked Labels panel
        if panel.locked:
            self.canvas.itemconfig(rect_id, stipple='gray50')
        
        # Draw label
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        
        # Different label text for Labels panel
        if panel.role == "labels":
            label_text = "⏱️ Labels Strip\n(Auto-managed)"
        else:
            label_text = panel.key
            if panel.role == "time" and panel.y_column:
                label_text += f"\n{panel.y_column}"
            elif panel.role == "not-time" and panel.x_column and panel.y_column_2:
                label_text += f"\n{panel.x_column}\nvs\n{panel.y_column_2}"
            
            label_text += f"\n({panel.role})"
            
            # Add span info if not 1x1
            if panel.rowspan > 1 or panel.colspan > 1:
                label_text += f"\n[{panel.rowspan}×{panel.colspan}]"
        
        self.canvas.create_text(
            cx, cy,
            text=label_text,
            font=('', 8 if not panel.locked else 9),
            fill='#666' if panel.locked else '#000',
            tags=f'panel_{panel.key}_label'
        )
    
    def _pixel_to_cell(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """Convert canvas pixel coordinates to grid cell coordinates."""
        nrows = self.nrows_var.get()
        ncols = self.ncols_var.get()
        
        # Adjust for padding
        x -= self.GRID_PADDING
        y -= self.GRID_PADDING
        
        if x < 0 or y < 0:
            return None
        
        col = x // self.CELL_SIZE
        row = y // self.CELL_SIZE
        
        if row >= nrows or col >= ncols:
            return None
        
        return (row, col)
    
    def _on_mouse_down(self, event):
        """Handle mouse button press - select existing panel or start drag."""
        cell = self._pixel_to_cell(event.x, event.y)
        if cell is None:
            return
        
        row, col = cell
        
        # Check if clicking on an existing panel - if yes, select it (unless locked)
        for panel in self.panels:
            if (row >= panel.row and row < panel.row + panel.rowspan and
                col >= panel.col and col < panel.col + panel.colspan):
                # Clicked on existing panel
                if panel.locked:
                    # Can't select locked panels (Labels)
                    messagebox.showinfo(
                        "Auto-Managed Panel",
                        "The Labels strip is auto-managed and cannot be edited.\n"
                        "It will align with your time-series plots.",
                        parent=self
                    )
                    return
                
                # Select non-locked panel
                self.selected_panel = panel
                # Update listbox selection
                try:
                    idx = [p for p in self.panels if not p.locked].index(panel)
                    self.panel_listbox.selection_clear(0, tk.END)
                    self.panel_listbox.selection_set(idx)
                    self.panel_listbox.see(idx)
                except:
                    pass
                # Load settings
                self.role_var.set(panel.role)
                self._rebuild_vars_ui()
                if panel.role == "time" and hasattr(self, 'y_var') and panel.y_column:
                    self.y_var.set(panel.y_column)
                elif panel.role == "not-time":
                    if hasattr(self, 'x_var') and panel.x_column:
                        self.x_var.set(panel.x_column)
                    if hasattr(self, 'y2_var') and panel.y_column_2:
                        self.y2_var.set(panel.y_column_2)
                self._redraw_grid()
                return
        
        # Not clicking on existing panel - start drag to create new one
        self.drag_start_cell = cell
        self.drag_current_cell = cell
    
    def _on_mouse_drag(self, event):
        """Handle mouse drag - show preview rectangle."""
        if self.drag_start_cell is None:
            return
        
        current_cell = self._pixel_to_cell(event.x, event.y)
        if current_cell is None or current_cell == self.drag_current_cell:
            return
        
        self.drag_current_cell = current_cell
        
        # Remove old preview
        if self.drag_preview_id is not None:
            self.canvas.delete(self.drag_preview_id)
        
        # Calculate preview bounds
        start_row, start_col = self.drag_start_cell
        curr_row, curr_col = current_cell
        
        row_min = min(start_row, curr_row)
        row_max = max(start_row, curr_row)
        col_min = min(start_col, curr_col)
        col_max = max(start_col, curr_col)
        
        # Draw preview rectangle
        x0 = self.GRID_PADDING + col_min * self.CELL_SIZE
        y0 = self.GRID_PADDING + row_min * self.CELL_SIZE
        x1 = self.GRID_PADDING + (col_max + 1) * self.CELL_SIZE
        y1 = self.GRID_PADDING + (row_max + 1) * self.CELL_SIZE
        
        self.drag_preview_id = self.canvas.create_rectangle(
            x0, y0, x1, y1,
            fill=self.COLOR_HOVER,
            outline='#ff6600',
            width=3,
            stipple='gray50',
            tags='drag_preview'
        )
    
    def _on_mouse_up(self, event):
        """Handle mouse release - create panel."""
        if self.drag_start_cell is None:
            return
        
        # Remove preview
        if self.drag_preview_id is not None:
            self.canvas.delete(self.drag_preview_id)
            self.drag_preview_id = None
        
        # Get final cell
        end_cell = self._pixel_to_cell(event.x, event.y)
        if end_cell is None:
            end_cell = self.drag_current_cell or self.drag_start_cell
        
        # Calculate panel bounds
        start_row, start_col = self.drag_start_cell
        end_row, end_col = end_cell
        
        row = min(start_row, end_row)
        col = min(start_col, end_col)
        rowspan = abs(end_row - start_row) + 1
        colspan = abs(end_col - start_col) + 1
        
        # Reset drag state
        self.drag_start_cell = None
        self.drag_current_cell = None
        
        # Get current role and variables
        role = self.role_var.get()
        
        # Validate variables are selected
        if role == "time":
            if not hasattr(self, 'y_var') or not self.y_var.get():
                messagebox.showwarning(
                    "No Variable Selected",
                    "Please select a Y-axis variable first.",
                    parent=self
                )
                return
            y_col = self.y_var.get()
            x_col = None
            y2_col = None
        else:  # not-time
            if not hasattr(self, 'x_var') or not hasattr(self, 'y2_var'):
                messagebox.showwarning(
                    "No Variables Selected",
                    "Please select both X and Y variables first.",
                    parent=self
                )
                return
            if not self.x_var.get() or not self.y2_var.get():
                messagebox.showwarning(
                    "No Variables Selected",
                    "Please select both X and Y variables first.",
                    parent=self
                )
                return
            x_col = self.x_var.get()
            y_col = None
            y2_col = self.y2_var.get()
        
        # Create new panel
        new_panel = PanelConfig(
            key=f"panel_{self.next_panel_id}",
            row=row,
            col=col,
            rowspan=rowspan,
            colspan=colspan,
            role=role,
            y_column=y_col,
            x_column=x_col,
            y_column_2=y2_col
        )
        
        # Check for overlaps with existing panels
        for existing in self.panels:
            if new_panel.overlaps(existing):
                # Special message if overlapping with Labels
                if existing.role == "labels":
                    messagebox.showwarning(
                        "Cannot Overlap Labels",
                        f"Cannot place panel in the Labels row (bottom row).\n"
                        f"The Labels strip is auto-managed.",
                        parent=self
                    )
                else:
                    messagebox.showwarning(
                        "Overlap Detected",
                        f"New panel overlaps with {existing.key}",
                        parent=self
                    )
                return
        
        # Add panel
        self.panels.append(new_panel)
        self.next_panel_id += 1
        
        # Clear selection since we just added a new panel
        self.selected_panel = None
        self.panel_listbox.selection_clear(0, tk.END)
        
        # Redraw and update list
        self._redraw_grid()
        self._update_panel_list()
    
    def _update_panel_list(self):
        """Update the panel listbox (excludes locked Labels panel)."""
        self.panel_listbox.delete(0, tk.END)
        # Only show non-locked panels (user-created panels)
        for panel in self.panels:
            if panel.locked:
                continue  # Skip Labels panel
            span_info = f"{panel.rowspan}×{panel.colspan}" if (panel.rowspan > 1 or panel.colspan > 1) else "1×1"
            display = f"{panel.key} [{panel.row},{panel.col}] {span_info} ({panel.role})"
            self.panel_listbox.insert(tk.END, display)
    
    def _renumber_panels(self):
        """Renumber all user panels sequentially from 1 to N (skips Labels)."""
        # Only renumber non-locked panels
        user_panels = [p for p in self.panels if not p.locked]
        for i, panel in enumerate(user_panels, start=1):
            panel.key = f"panel_{i}"
        # Set next_panel_id to N+1
        self.next_panel_id = len(user_panels) + 1
    
    def _on_panel_select(self, event):
        """Handle panel selection from listbox - load settings for editing."""
        selection = self.panel_listbox.curselection()
        if not selection:
            # No selection - do nothing, don't clear
            return
        
        idx = selection[0]
        # Get user panels only (excluding locked)
        user_panels = [p for p in self.panels if not p.locked]
        if idx >= len(user_panels):
            # Invalid index
            return
        
        self.selected_panel = user_panels[idx]
        
        # Load panel settings into controls for editing
        self.role_var.set(self.selected_panel.role)
        self._rebuild_vars_ui()  # Rebuild UI for the selected role
        
        # Populate variable dropdowns
        if self.selected_panel.role == "time":
            if hasattr(self, 'y_var') and self.selected_panel.y_column:
                self.y_var.set(self.selected_panel.y_column)
        else:  # not-time
            if hasattr(self, 'x_var') and self.selected_panel.x_column:
                self.x_var.set(self.selected_panel.x_column)
            if hasattr(self, 'y2_var') and self.selected_panel.y_column_2:
                self.y2_var.set(self.selected_panel.y_column_2)
        
        # Redraw to highlight selected panel
        self._redraw_grid()
    
    def _update_selected_panel(self):
        """Update the selected panel with current control values."""
        if self.selected_panel is None:
            messagebox.showinfo("No Selection", "Please select a panel to edit.", parent=self)
            return
        
        # Get current role and variables
        role = self.role_var.get()
        
        # Validate variables are selected
        if role == "time":
            if not hasattr(self, 'y_var') or not self.y_var.get():
                messagebox.showwarning(
                    "No Variable Selected",
                    "Please select a Y-axis variable.",
                    parent=self
                )
                return
            y_col = self.y_var.get()
            x_col = None
            y2_col = None
        else:  # not-time
            if not hasattr(self, 'x_var') or not hasattr(self, 'y2_var'):
                messagebox.showwarning(
                    "No Variables Selected",
                    "Please select both X and Y variables.",
                    parent=self
                )
                return
            if not self.x_var.get() or not self.y2_var.get():
                messagebox.showwarning(
                    "No Variables Selected",
                    "Please select both X and Y variables.",
                    parent=self
                )
                return
            x_col = self.x_var.get()
            y_col = None
            y2_col = self.y2_var.get()
        
        # Update panel
        self.selected_panel.role = role
        self.selected_panel.y_column = y_col
        self.selected_panel.x_column = x_col
        self.selected_panel.y_column_2 = y2_col
        
        # Redraw and update list
        self._redraw_grid()
        self._update_panel_list()
        
        messagebox.showinfo("Updated", f"{self.selected_panel.key} updated successfully.", parent=self)
    
    def _delete_selected_panel(self):
        """Delete the selected panel."""
        if self.selected_panel is None:
            selection = self.panel_listbox.curselection()
            if not selection:
                messagebox.showinfo("No Selection", "Please select a panel first.", parent=self)
                return
            idx = selection[0]
            user_panels = [p for p in self.panels if not p.locked]
            if idx >= len(user_panels):
                return
            self.selected_panel = user_panels[idx]
        
        # Check if trying to delete locked panel
        if self.selected_panel.locked:
            messagebox.showinfo(
                "Cannot Delete",
                "The Labels strip cannot be deleted.\nIt is auto-managed.",
                parent=self
            )
            return
        
        # Confirm
        if not messagebox.askyesno(
            "Confirm Delete",
            f"Delete {self.selected_panel.key}?",
            parent=self
        ):
            return
        
        # Remove
        self.panels.remove(self.selected_panel)
        self.selected_panel = None
        
        # Renumber all panels sequentially
        self._renumber_panels()
        
        # Redraw
        self._redraw_grid()
        self._update_panel_list()
    
    def _clear_all_panels(self):
        """Clear all user panels (keeps Labels panel)."""
        # Count only user panels
        user_panels = [p for p in self.panels if not p.locked]
        if not user_panels:
            return
        
        if not messagebox.askyesno(
            "Confirm Clear",
            "Delete all panels?",
            parent=self
        ):
            return
        
        # Keep only locked panels (Labels)
        self.panels = [p for p in self.panels if p.locked]
        self.selected_panel = None
        
        # Renumber (will reset next_panel_id to 1)
        self._renumber_panels()
        
        self._redraw_grid()
        self._update_panel_list()
    
    def _show_preview(self):
        """Show matplotlib preview of the current layout."""
        # Validate that panels exist
        if not self.panels:
            messagebox.showinfo(
                "No Panels",
                "Please create at least one panel before previewing.",
                parent=self
            )
            return
        
        # Close existing preview window if open
        if self.preview_window is not None and self.preview_window.winfo_exists():
            self.preview_window.destroy()
        
        # Create new preview window
        self.preview_window = tk.Toplevel(self)
        self.preview_window.title("Layout Preview")
        self.preview_window.geometry("800x600")
        
        # Generate layout spec from current panels
        layout_spec = self._generate_layout_spec()
        
        # Create matplotlib figure with GridSpec
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(
            nrows=layout_spec['nrows'],
            ncols=layout_spec['ncols'],
            hspace=layout_spec.get('hspace', 0.15),
            wspace=layout_spec.get('wspace', 0.12),
            figure=fig
        )
        
        # Create a subplot for each panel
        for panel in self.panels:
            # Use GridSpec slice notation for spanning panels
            ax = fig.add_subplot(gs[
                panel.row:panel.row + panel.rowspan,
                panel.col:panel.col + panel.colspan
            ])
            
            # Set background color based on role (match grid colors)
            if panel.role == "labels":
                bg_color = self.COLOR_LABELS
            elif panel.role == "time":
                bg_color = self.COLOR_TIME
            else:
                bg_color = self.COLOR_NOT_TIME
            ax.set_facecolor(bg_color)
            
            # Remove ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Build info text to display in panel
            if panel.role == "labels":
                # Special rendering for Labels panel
                info_lines = ["⏱️ Labels Strip", "(Auto-managed)"]
            else:
                info_lines = [panel.key]
                
                # Add variable information
                if panel.role == "time":
                    info_lines.append(f"Y: {panel.y_column}")
                else:  # not-time
                    info_lines.append(f"X: {panel.x_column}")
                    info_lines.append(f"Y: {panel.y_column_2}")
                
                # Add role
                info_lines.append(f"({panel.role})")
                
                # Add span info if not 1x1
                if panel.rowspan > 1 or panel.colspan > 1:
                    info_lines.append(f"[{panel.rowspan}×{panel.colspan}]")
            
            # Display text in center of panel
            info_text = "\n".join(info_lines)
            ax.text(
                0.5, 0.5, info_text,
                ha='center', va='center',
                fontsize=10,
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        # Embed matplotlib figure in Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.preview_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add close button at bottom
        button_frame = ttk.Frame(self.preview_window)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            button_frame,
            text="Close",
            command=self.preview_window.destroy,
            width=12
        ).pack()
    
    def _generate_layout_spec(self) -> Dict:
        """Generate layout_spec dictionary from current panels."""
        nrows = self.nrows_var.get()
        ncols = self.ncols_var.get()
        
        areas = []
        for panel in self.panels:
            area = {
                'key': panel.key,
                'row': panel.row,
                'col': panel.col,
                'role': panel.role,
            }
            if panel.rowspan > 1:
                area['rowspan'] = panel.rowspan
            if panel.colspan > 1:
                area['colspan'] = panel.colspan
            areas.append(area)
        
        return {
            'nrows': nrows,
            'ncols': ncols,
            'hspace': 0.15,
            'wspace': 0.12,
            'areas': areas,
        }
    
    def _generate_plot_config(self) -> Dict:
        """Generate plot_config dictionary from current panels."""
        config = {}
        for panel in self.panels:
            panel_cfg = {
                'role': panel.role,
            }
            if panel.role == "time":
                panel_cfg['y_column'] = panel.y_column
            else:
                panel_cfg['x_column'] = panel.x_column
                panel_cfg['y_column'] = panel.y_column_2
            
            config[panel.key] = panel_cfg
        
        return config
    
    def _validate_layout(self) -> bool:
        """Validate layout before accepting."""
        # Check that user has created at least one data panel
        user_panels = [p for p in self.panels if not p.locked]
        if not user_panels:
            messagebox.showwarning(
                "No Data Panels",
                "Please create at least one time-series or cross-plot panel.",
                parent=self
            )
            return False
        
        # Check all user panels have variables assigned
        for panel in user_panels:
            if panel.role == "time" and not panel.y_column:
                messagebox.showwarning(
                    "Incomplete Panel",
                    f"{panel.key} has no Y variable assigned.",
                    parent=self
                )
                return False
            elif panel.role == "not-time" and (not panel.x_column or not panel.y_column_2):
                messagebox.showwarning(
                    "Incomplete Panel",
                    f"{panel.key} has no X/Y variables assigned.",
                    parent=self
                )
                return False
        
        return True
    
    def _on_done(self):
        """Handle Done button - validate and close."""
        if not self._validate_layout():
            return
        
        # Generate outputs
        self.result_layout_spec = self._generate_layout_spec()
        self.result_plot_config = self._generate_plot_config()
        
        self.destroy()
    
    def _on_cancel(self):
        """Handle Cancel button."""
        if self.panels and not messagebox.askyesno(
            "Confirm Cancel",
            "Discard layout and close?",
            parent=self
        ):
            return
        
        self.result_layout_spec = None
        self.result_plot_config = None
        self.destroy()


def build_layout(df: pd.DataFrame, parent: Optional[tk.Tk] = None) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Launch interactive layout builder dialog.
    
    This is the main entry point for users who want to visually design their
    plot layout instead of manually writing layout_spec dictionaries.
    
    Args:
        df: DataFrame containing the data to be plotted. Used to extract
            available columns for assignment.
        parent: Optional parent Tk window. If None, creates temporary root.
    
    Returns:
        Tuple of (layout_spec, plot_config):
        - layout_spec: Dictionary suitable for TimeIntervalLabeler constructor
        - plot_config: Dictionary suitable for generate_plot_fn()
        - Both are None if user cancels
    
    Example:
        >>> import pandas as pd
        >>> from chronotagger.labeler.utils import build_layout, generate_plot_fn
        >>> from chronotagger.labeler import TimeIntervalLabeler
        >>> 
        >>> # Load data
        >>> df = pd.read_csv('data.csv', index_col=0, parse_dates=True)
        >>> 
        >>> # Build layout interactively
        >>> layout_spec, plot_config = build_layout(df)
        >>> 
        >>> if layout_spec is not None:
        >>>     # Generate plot function from config
        >>>     plot_fn = generate_plot_fn(plot_config)
        >>>     
        >>>     # Create labeler
        >>>     app = TimeIntervalLabeler(
        >>>         df=df,
        >>>         plot_fn=plot_fn,
        >>>         layout_spec=layout_spec
        >>>     )
        >>>     app.run()
    """
    # Create temporary root if needed
    owns_root = False
    if parent is None:
        parent = tk.Tk()
        parent.withdraw()  # Hide root window
        owns_root = True
    
    try:
        # Launch dialog
        dialog = LayoutBuilderDialog(parent, df)
        
        # Wait for dialog to close
        parent.wait_window(dialog)
        
        # Get results
        layout_spec = dialog.result_layout_spec
        plot_config = dialog.result_plot_config
        
        return layout_spec, plot_config
    
    finally:
        # Clean up temporary root
        if owns_root:
            parent.destroy()
