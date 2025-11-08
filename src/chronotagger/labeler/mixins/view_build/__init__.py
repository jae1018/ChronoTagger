"""
GUI construction mixins for ChronoTagger.

This module combines all UI building mixins into a single ViewBuildMixin
for backward compatibility. The package is organized into:

- window: Main window orchestration
- controls: Top control bar (time range, navigation, label actions)
- sidebar: Right panel (intervals list, statistics, options)
- canvas: Matplotlib figure embedding and event wiring
- widgets: Shared widget utilities (tooltips, etc.)

All mixins are combined through multiple inheritance to provide the complete
GUI construction functionality for TimeIntervalLabeler.
"""

from __future__ import annotations

from .window import WindowMixin
from .controls import ControlsMixin
from .sidebar import SidebarMixin
from .canvas import CanvasMixin
from .widgets import WidgetsMixin


class ViewBuildMixin(
    WindowMixin,
    ControlsMixin,
    SidebarMixin,
    CanvasMixin,
    WidgetsMixin,
):
    """
    Combined view building mixin.
    
    This class combines all GUI construction functionality for TimeIntervalLabeler.
    It maintains backward compatibility while organizing the code into logical components.
    
    Method Resolution Order (MRO):
    1. WindowMixin - Main window orchestration
    2. ControlsMixin - Top control bar construction
    3. SidebarMixin - Right panel construction
    4. CanvasMixin - Matplotlib figure and canvas
    5. WidgetsMixin - Shared widget utilities
    
    All methods from the original view_build.py are preserved and accessible
    through this combined mixin.
    """
    pass


# Maintain backward compatibility
__all__ = ['ViewBuildMixin']
