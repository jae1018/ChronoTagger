"""
Package logging (Pack 4): the forensic channel.

Layer A (``__init__.py``): a NullHandler on the package logger -- library
hygiene, and measured necessity: logging.lastResort otherwise writes
unconfigured warnings raw to stderr.

Layer B (here): one utf-8 file handler beside the autosave, attached at
labeler construction via configure_file_logging(). Idempotent by sentinel:
twenty constructions in a test session yield ONE handler. Level INFO;
CHRONOTAGGER_DEBUG=1 raises to DEBUG (same shape as CHRONOTAGGER_STRICT).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

LOGGER_NAME = "chronotagger"
LOG_FILENAME = "chronotagger.log"
_SENTINEL = "_chronotagger_file_handler"

logger = logging.getLogger(LOGGER_NAME)


def _level_from_env() -> int:
    return (logging.DEBUG
            if os.environ.get("CHRONOTAGGER_DEBUG") == "1"
            else logging.INFO)


def configure_file_logging(folder: Union[str, Path]) -> Optional[Path]:
    """
    Attach the single utf-8 file handler for this process (idempotent).

    Same folder again: no-op. Different folder: the old handler is closed
    and replaced (test isolation: each labeler's tmp folder gets a live
    handler). The folder is created if missing; an unwritable location
    returns None -- a broken log must never take the app down with it.
    The handler opens EAGERLY: with delay=True the open happens at the
    first record, OUTSIDE this guard, and the guard is dead code
    (verifier B2 -- executed: construction died with PermissionError).
    Idempotence compares ABSOLUTE paths: FileHandler.baseFilename is
    always absolute, a relative autosave_folder is not (verifier M2).
    """
    path = Path(folder) / LOG_FILENAME
    target = os.path.abspath(str(path))

    for h in list(logger.handlers):
        if getattr(h, _SENTINEL, False):
            if getattr(h, "baseFilename", None) == target:
                return path
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    except (OSError, ValueError):
        return None
    setattr(handler, _SENTINEL, True)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(_level_from_env())
    logger.info("=== session start %s ===",
                datetime.now().isoformat(timespec="seconds"))
    return path
