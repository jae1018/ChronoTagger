"""
Event handling subpackage for TimeIntervalLabeler.

This subpackage organizes event handling logic into focused mixins:
- EventsBaseMixin: Base utility methods for time conversion and UI updates
- OverlaysMixin: Visual overlays and highlighting
- StripInteractionMixin: Strip editing and preview
- SelectionMixin: Point/interval/box selection
- MouseEventsMixin: Mouse event handlers
- KeyboardEventsMixin: Keyboard shortcuts and key handlers

The EventsMixin class combines all submixins to maintain backward compatibility
with the original monolithic events.py implementation.
"""

from __future__ import annotations

from .base import EventsBaseMixin
from .overlays import OverlaysMixin
from .strip import StripInteractionMixin
from .selection import SelectionMixin
from .mouse import MouseEventsMixin
from .keyboard import KeyboardEventsMixin


class EventsMixin(
    KeyboardEventsMixin,
    MouseEventsMixin,
    SelectionMixin,
    StripInteractionMixin,
    OverlaysMixin,
    EventsBaseMixin,
):
    """
    Combined event handling mixin providing complete backward compatibility.

    This class inherits from all event handling submixins through multiple
    inheritance, allowing the TimeIntervalLabeler to access all event handling
    methods through a single mixin.

    Method Resolution Order (MRO):
    1. KeyboardEventsMixin - Keyboard shortcuts and key events
    2. MouseEventsMixin - Mouse clicks, motion, drag/drop
    3. SelectionMixin - Selection mechanisms (box, two-click, strip)
    4. StripInteractionMixin - Strip preview and editing
    5. OverlaysMixin - Visual overlays and highlighting
    6. EventsBaseMixin - Base utility methods for time conversion and UI updates

    All methods from the original events.py are preserved and accessible
    through this combined mixin.
    """
    pass


# Export all mixins for direct import if needed
__all__ = [
    'EventsMixin',
    'EventsBaseMixin',
    'OverlaysMixin',
    'StripInteractionMixin',
    'SelectionMixin',
    'MouseEventsMixin',
    'KeyboardEventsMixin',
]
