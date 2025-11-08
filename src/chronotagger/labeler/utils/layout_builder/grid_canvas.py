"""
Grid Canvas Mixin for Layout Builder

This module provides grid/canvas rendering functionality for the visual layout
builder. It handles drawing the interactive grid, panels, and coordinate conversions,
as well as panel management operations.

Key Features:
- Grid rendering with customizable size
- Panel visualization with role-based colors
- Pixel-to-cell coordinate conversion
- Selected panel highlighting
- Panel list management and editing
- Labels panel auto-management

Author: ChronoTagger Team
"""

from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING
import tkinter as tk
from tkinter import messagebox

if TYPE_CHECKING:
    from .models import PanelConfig


class GridCanvasMixin:
    """
    Mixin class providing grid/canvas methods for the Layout Builder.

    This mixin handles all visual rendering of the grid and panels on a Tkinter
    canvas. It should be mixed into a class that has the following attributes:
    - canvas: tk.Canvas instance
    - nrows_var: tk.IntVar for number of rows
    - ncols_var: tk.IntVar for number of columns
    - panels: List[PanelConfig] of panel configurations
    - selected_panel: Optional[PanelConfig] currently selected panel

    And the following constants:
    - CELL_SIZE: Size of each grid cell in pixels
    - GRID_PADDING: Padding around the grid in pixels
    - COLOR_EMPTY: Color for empty cells
    - COLOR_TIME: Color for time-series panels
    - COLOR_NOT_TIME: Color for cross-plot panels
    - COLOR_LABELS: Color for locked labels strip
    - COLOR_SELECTED_OUTLINE: Color for selected panel outline
    """

    # Visual constants (expected to be defined in the class using this mixin)
    CELL_SIZE: int = 80
    GRID_PADDING: int = 20
    COLOR_EMPTY: str = "#f0f0f0"
    COLOR_TIME: str = "#cce5ff"
    COLOR_NOT_TIME: str = "#d4edda"
    COLOR_LABELS: str = "#e0e0e0"
    COLOR_SELECTED_OUTLINE: str = "#ff6600"

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

    def _create_labels_panel(self):
        """Create the auto-managed Labels strip panel."""
        # Import here to avoid circular dependency
        from .models import PanelConfig

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
