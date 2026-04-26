"""
Configuration data structures for quick-start wizard.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class PlotConfig:
    """
    Configuration for auto-generated plot function.

    This class stores all information needed to recreate a plot
    function from user selections in the quick-start wizard.
    """

    # Data source
    data_file: str
    """Path to the data file (CSV, Parquet, etc.)"""

    # Column selection
    selected_columns: List[str]
    """List of column names to plot"""

    # Layout configuration
    layout_type: str = "vertical_stack"
    """Type of layout: 'vertical_stack' or 'custom'"""

    layout_spec: Optional[Dict[str, Any]] = None
    """Layout specification dict (for custom layouts)"""

    # Metadata
    created_at: Optional[str] = None
    """ISO timestamp when configuration was created"""

    def to_dict(self) -> dict:
        """Serialize to JSON-friendly dictionary."""
        return {
            'data_file': self.data_file,
            'selected_columns': self.selected_columns,
            'layout_type': self.layout_type,
            'layout_spec': self.layout_spec,
            'created_at': self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'PlotConfig':
        """Deserialize from dictionary."""
        return cls(
            data_file=d['data_file'],
            selected_columns=d['selected_columns'],
            layout_type=d.get('layout_type', 'vertical_stack'),
            layout_spec=d.get('layout_spec'),
            created_at=d.get('created_at'),
        )

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if valid, False otherwise
        """
        # Check data file exists
        if not Path(self.data_file).exists():
            return False

        # Check at least one column selected
        if not self.selected_columns:
            return False

        # Check valid layout type
        if self.layout_type not in ['vertical_stack', 'custom']:
            return False

        # If custom layout, must have layout_spec
        if self.layout_type == 'custom' and not self.layout_spec:
            return False

        return True
