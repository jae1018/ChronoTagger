"""
Interval management mixins for ChronoTagger.

This module combines all interval-related mixins into a single IntervalsMixin
for backward compatibility.
"""

from __future__ import annotations

from .crud import IntervalCRUDMixin
from .commands import IntervalCommandsMixin
from .gaps import IntervalGapsMixin
from .validation import IntervalValidationMixin
from .merge import IntervalMergeMixin


class IntervalsMixin(
    IntervalCRUDMixin,
    IntervalCommandsMixin,
    IntervalGapsMixin,
    IntervalValidationMixin,
    IntervalMergeMixin,
):
    """
    Combined interval management mixin.
    
    This class combines all interval-related functionality for the TimeIntervalLabeler.
    It maintains backward compatibility while organizing the code into logical components:
    
    - IntervalCRUDMixin: Basic interval create, read, update, delete operations
    - IntervalCommandsMixin: Undo/redo command pattern implementation
    - IntervalGapsMixin: Gap detection and unassigned region labeling
    - IntervalValidationMixin: Overlap detection and resolution
    - IntervalMergeMixin: Merging adjacent intervals with same label
    
    All methods from the original intervals.py are preserved and accessible
    through this combined mixin.
    """
    pass


# Maintain backward compatibility
__all__ = ['IntervalsMixin']
