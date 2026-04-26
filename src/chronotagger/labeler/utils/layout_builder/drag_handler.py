"""
Drag Handler Mixin for Layout Builder

This module provides drag-and-drop functionality for creating spanning panels
in the layout builder grid. It handles mouse events for selecting existing
panels or creating new ones by dragging across grid cells.

Classes:
    DragHandlerMixin: Handles mouse down, drag, and release events
"""

from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING
import tkinter as tk
from tkinter import messagebox

if TYPE_CHECKING:
    from .models import PanelConfig


class DragHandlerMixin:
    """
    Mixin providing drag-and-drop functionality for the layout builder.

    This mixin handles three primary interactions:
    1. Clicking on existing panels to select them for editing
    2. Starting a drag operation on empty cells
    3. Creating new spanning panels by dragging across multiple cells

    The mixin expects the parent class to provide:
    - self.panels: List[PanelConfig]
    - self.selected_panel: Optional[PanelConfig]
    - self.drag_start_cell: Optional[Tuple[int, int]]
    - self.drag_current_cell: Optional[Tuple[int, int]]
    - self.drag_preview_id: Optional[int]
    - self.canvas: tk.Canvas
    - self.role_var: tk.StringVar
    - self.panel_listbox: tk.Listbox
    - Various UI variables and methods
    """

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

        # Import here to avoid circular dependency
        from .models import PanelConfig

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

        # Keep the auto-managed Labels strip aligned with the first time panel
        self._update_labels_panel()

        # Redraw and update list
        self._redraw_grid()
        self._update_panel_list()
