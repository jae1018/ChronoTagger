"""
Layout Builder Dialog

This module provides the main dialog class for the interactive layout builder,
combining all mixins through multiple inheritance. The dialog allows users to
visually design plot layouts by dragging on a grid.

Classes:
    LayoutBuilderDialog: Main dialog class combining all layout builder functionality
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, List
import tkinter as tk
from tkinter import messagebox
import pandas as pd

from .models import PanelConfig
from .grid_canvas import GridCanvasMixin
from .drag_handler import DragHandlerMixin
from .preview import PreviewMixin
from .ui_builder import UIBuilderMixin


class LayoutBuilderDialog(
    UIBuilderMixin,
    DragHandlerMixin,
    PreviewMixin,
    GridCanvasMixin,
    tk.Toplevel
):
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

    This class combines functionality from multiple mixins:
    - UIBuilderMixin: UI construction and management
    - DragHandlerMixin: Mouse event handling for drag-to-create
    - PreviewMixin: Layout preview and config generation
    - GridCanvasMixin: Grid drawing and panel rendering

    Attributes:
        df: DataFrame to analyze for available columns
        result_layout_spec: Generated layout specification (set on Done)
        result_plot_config: Generated plot configuration (set on Done)
        numeric_columns: List of numeric column names from DataFrame
        panels: List of all panel configurations
        selected_panel: Currently selected panel for editing
        next_panel_id: Counter for generating unique panel IDs
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

        # Check that no two panels overlap in the grid. _clip_panels_to_grid
        # may have collapsed two previously-disjoint panels onto the same
        # cell when the user shrank the grid; rather than silently accept
        # a degenerate layout, refuse and ask the user to fix it.
        for i, p_a in enumerate(self.panels):
            for p_b in self.panels[i + 1:]:
                if self._panels_overlap(p_a, p_b):
                    messagebox.showwarning(
                        "Overlapping Panels",
                        f"{p_a.key} and {p_b.key} overlap on the grid. "
                        f"Please move or resize one of them so the panels "
                        f"don't share any cells.",
                        parent=self,
                    )
                    return False

        return True

    @staticmethod
    def _panels_overlap(a, b) -> bool:
        """True if two PanelConfigs share any grid cell."""
        a_row_end = a.row + a.rowspan
        a_col_end = a.col + a.colspan
        b_row_end = b.row + b.rowspan
        b_col_end = b.col + b.colspan
        rows_overlap = a.row < b_row_end and b.row < a_row_end
        cols_overlap = a.col < b_col_end and b.col < a_col_end
        return rows_overlap and cols_overlap

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
