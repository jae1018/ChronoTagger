"""
ChronoTagger package

Public API:
    - Interval (core)
    - TimeIntervalLabeler (GUI)
"""
from .core import Interval
from .labeler import TimeIntervalLabeler

__all__ = ["Interval", "TimeIntervalLabeler"]
