"""
Data Models and Constants for ChronoTagger Layout Builder

This module contains the core data structures and visual constants used by the
layout builder system. It provides the PanelConfig dataclass for representing
individual panel configurations and defines the visual appearance constants
for the grid-based layout interface.

Author: ChronoTagger Team
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Visual constants
CELL_SIZE = 80  # pixels per grid cell
GRID_PADDING = 20  # padding around grid
COLOR_EMPTY = "#f0f0f0"
COLOR_TIME = "#cce5ff"  # Light blue
COLOR_NOT_TIME = "#d4edda"  # Light green
COLOR_LABELS = "#e0e0e0"  # Light gray for locked Labels strip
COLOR_HOVER = "#fff3cd"  # Light yellow
COLOR_SELECTED_OUTLINE = "#ff6600"  # Orange for selected panel


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
