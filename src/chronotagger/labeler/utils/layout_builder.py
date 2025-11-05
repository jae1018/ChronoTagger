"""
Visual Layout Builder for ChronoTagger

A Tkinter-based GUI that allows users to interactively design their plot layout
by clicking on a grid. This eliminates the need to manually write layout_spec 
dictionaries.

Key Features:
- Visual grid builder with click-to-add panels
- Variable assignment (which columns to plot)
- Role selection (time vs not-time plots)
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


@dataclass
class PanelConfig:
    """
    Configuration for a single panel in the layout.
    
    Attributes:
        key: Unique identifier (e.g., "panel_1")
        row: Row position in grid (0-indexed)
        col: Column position in grid (0-indexed)
        rowspan: Number of rows this panel spans (default: 1)
        colspan: Number of columns this panel spans (default: 1)
        role: "time" or "not-time"
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
    y_column: Optional[str] = None
    x_column: Optional[str] = None
    y_column_2: Optional[str] = None


class LayoutBuilderDialog(tk.Toplevel):
    """
    Interactive dialog for building plot layouts.
    
    This dialog presents a visual grid where users can:
    1. Set grid dimensions (rows × columns)
    2. Click cells to add panels
    3. Assign DataFrame columns to each panel
    4. Set panel roles (time-series vs cross-plot)
    5. Delete panels as needed
    
    The dialog returns both layout_spec (for TimeIntervalLabeler) and
    plot_config (for automatic plot function generation).
    """
    
    # Visual constants
    CELL_SIZE = 80  # pixels per grid cell
    GRID_PADDING = 20  # padding around grid
    COLOR_EMPTY = "#f0f0f0"
    COLOR_TIME = "#cce5ff"  # Light blue
    COLOR_NOT_TIME = "#d4edda"  # Light green
    COLOR_HOVER = "#fff3cd"  # Light yellow
    
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
        
        # Build UI
        self._build_ui()
        
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
            from_=1, to=10,
            textvariable=self.nrows_var,
            width=5,
            command=self._redraw_grid
        ).pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Label(controls, text="Columns:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(
            controls,
            from_=1, to=5,
            textvariable=self.ncols_var,
            width=5,
            command=self._redraw_grid
        ).pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Button(controls, text="Clear All", command=self._clear_all_panels).pack(side=tk.RIGHT)
        
        # Instructions
        instructions = ttk.Label(
            left,
            text="📌 Click on a grid cell to add a panel",
            font=('', 9, 'italic'),
            foreground='#666'
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
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self._on_canvas_click)
        
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
        
        # Add panel controls
        add_frame = ttk.LabelFrame(right, text="Add Panel", padding=10)
        add_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(add_frame, text="1. Select role and variable").pack(anchor='w', pady=(0, 5))
        ttk.Label(add_frame, text="2. Click a grid cell to add").pack(anchor='w', pady=(0, 10))
        
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
        
        ttk.Button(edit_frame, text="Delete Panel", command=self._delete_selected_panel).pack(fill=tk.X)
        
        # Bottom buttons
        bottom_frame = ttk.Frame(right)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
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
        """Draw a panel on the canvas."""
        x0 = self.GRID_PADDING + panel.col * self.CELL_SIZE
        y0 = self.GRID_PADDING + panel.row * self.CELL_SIZE
        x1 = x0 + self.CELL_SIZE
        y1 = y0 + self.CELL_SIZE
        
        # Color based on role
        if panel.role == "time":
            fill_color = self.COLOR_TIME
        else:
            fill_color = self.COLOR_NOT_TIME
        
        # Draw rectangle
        self.canvas.create_rectangle(
            x0, y0, x1, y1,
            fill=fill_color,
            outline='#333',
            width=2,
            tags=f'panel_{panel.key}'
        )
        
        # Draw label
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        
        label_text = panel.key
        if panel.role == "time" and panel.y_column:
            label_text += f"\n{panel.y_column}"
        elif panel.role == "not-time" and panel.x_column and panel.y_column_2:
            label_text += f"\n{panel.x_column}\nvs\n{panel.y_column_2}"
        
        label_text += f"\n({panel.role})"
        
        self.canvas.create_text(
            cx, cy,
            text=label_text,
            font=('', 8),
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
    
    def _on_canvas_click(self, event):
        """Handle click on canvas - add panel."""
        cell = self._pixel_to_cell(event.x, event.y)
        if cell is None:
            return
        
        row, col = cell
        
        # Check if cell already occupied
        for panel in self.panels:
            if panel.row == row and panel.col == col:
                messagebox.showinfo(
                    "Cell Occupied",
                    f"Cell [{row},{col}] already contains {panel.key}",
                    parent=self
                )
                return
        
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
            role=role,
            y_column=y_col,
            x_column=x_col,
            y_column_2=y2_col
        )
        
        self.panels.append(new_panel)
        self.next_panel_id += 1
        
        # Redraw and update list
        self._redraw_grid()
        self._update_panel_list()
    
    def _update_panel_list(self):
        """Update the panel listbox."""
        self.panel_listbox.delete(0, tk.END)
        for panel in self.panels:
            display = f"{panel.key} [{panel.row},{panel.col}] ({panel.role})"
            self.panel_listbox.insert(tk.END, display)
    
    def _on_panel_select(self, event):
        """Handle panel selection from listbox."""
        selection = self.panel_listbox.curselection()
        if not selection:
            return
        
        idx = selection[0]
        self.selected_panel = self.panels[idx]
    
    def _delete_selected_panel(self):
        """Delete the selected panel."""
        if self.selected_panel is None:
            selection = self.panel_listbox.curselection()
            if not selection:
                messagebox.showinfo("No Selection", "Please select a panel first.", parent=self)
                return
            idx = selection[0]
            self.selected_panel = self.panels[idx]
        
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
        
        # Redraw
        self._redraw_grid()
        self._update_panel_list()
    
    def _clear_all_panels(self):
        """Clear all panels."""
        if not self.panels:
            return
        
        if not messagebox.askyesno(
            "Confirm Clear",
            "Delete all panels?",
            parent=self
        ):
            return
        
        self.panels.clear()
        self.selected_panel = None
        self.next_panel_id = 1
        
        self._redraw_grid()
        self._update_panel_list()
    
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
        if not self.panels:
            messagebox.showwarning("No Panels", "Please create at least one panel.", parent=self)
            return False
        
        # Check all panels have variables assigned
        for panel in self.panels:
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
