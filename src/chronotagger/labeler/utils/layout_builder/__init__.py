"""
ChronoTagger Layout Builder Package

A modular, refactored implementation of the interactive layout builder for
ChronoTagger. This package provides a visual grid-based interface for designing
plot layouts without manually writing layout_spec dictionaries.

The package is organized into the following modules:

- models: Core data structures (PanelConfig) and visual constants
- grid_canvas: Grid rendering and panel management
- drag_handler: Mouse event handling for drag-to-create functionality
- preview: Layout preview and configuration generation
- ui_builder: UI construction and management
- dialog: Main dialog class combining all mixins

Main Entry Points:
    LayoutBuilderDialog: The main dialog class for interactive layout building
    PanelConfig: Data class for panel configuration
    build_layout: Convenience function for launching the layout builder

Example:
    >>> from chronotagger.labeler.utils.layout_builder import build_layout
    >>> layout_spec, plot_config = build_layout(df)
    >>> if layout_spec:
    >>>     # Use the generated configuration
    >>>     app = TimeIntervalLabeler(df, plot_fn, layout_spec)
    >>>     app.run()

Author: ChronoTagger Team
"""

from .models import PanelConfig
from .dialog import LayoutBuilderDialog
from .builder import build_layout

__all__ = [
    'PanelConfig',
    'LayoutBuilderDialog',
    'build_layout',
]
