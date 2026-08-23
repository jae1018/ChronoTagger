"""
ChronoTagger package

Public API:
    - Interval (core)
    - TimeIntervalLabeler (GUI)
"""
import logging

# Library hygiene (Pack 4): without a NullHandler, an unconfigured
# logger's warnings spew raw to stderr via logging.lastResort (measured:
# 676 bytes of traceback noise with the message lost on a cp1252 console).
# Plain `import logging`, no alias: `_logging` would collide with the
# sibling module chronotagger._logging on the package object (V1 fold 6).
logging.getLogger("chronotagger").addHandler(logging.NullHandler())

from .core import Interval
from .labeler import TimeIntervalLabeler

__all__ = ["Interval", "TimeIntervalLabeler"]
