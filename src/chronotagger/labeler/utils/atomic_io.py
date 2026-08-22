"""
Atomic file writes for ChronoTagger's persistence layer.

Truncate-then-write (open(path, "w")) destroys the old file before the
new content exists; a crash mid-write leaves an empty file or a
valid-looking prefix. Everything here writes to a temp file IN THE
TARGET'S DIRECTORY (os.replace is not atomic across volumes on
Windows), fsyncs, then swaps with os.replace -- so the target always
holds either the complete old content or the complete new content.

Platform notes (all MEASURED -- Windows dev machine and the Linux Dell
node; see edit_pack/evidence/pack2_dell_crossplatform_report.md):
- The SAME-DIRECTORY tmp rule is a hard requirement on BOTH platforms:
  cross-filesystem os.replace raises WinError 17 on Windows and EXDEV
  (errno 18) on Linux.
- os.replace over a target held open: Windows raises PermissionError
  (OneDrive/antivirus share locks) and the bounded retry rides it out;
  Linux always succeeds, so the retry is simply dormant there.
- Directory fsync (full POSIX rename-durability) is deliberately
  SKIPPED: it raises on Windows, and on ext4 it measured +90% autosave
  cost (+9ms per gesture) for crash-durability of the rename alone.
  Atomicity holds without it; if rename-durability is ever wanted for
  the user-initiated writes (session save, exports), that is a Pack 5
  opt-in, not a per-gesture default.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Union

_REPLACE_TRIES = 5
_REPLACE_DELAY_S = 0.05


def _tmp_for(target: Path) -> Path:
    """
    Temp sibling that PRESERVES the target's final suffix:
    labels.csv.gz -> labels.csv.tmp.gz. pandas infers compression from
    the final suffix, so target.name + '.tmp' would silently write a
    PLAIN file under a compressed name (fold V1/V2/V3-M .gz). The
    autosave case keeps its .gitignore prefix:
    chronotagger_autosave_<hex>.json -> chronotagger_autosave_<hex>.tmp.json.
    """
    return target.with_name(target.stem + ".tmp" + target.suffix)


def _atomic_replace(tmp: Path, target: Path) -> None:
    """os.replace with a short bounded retry for transient share locks
    (OneDrive/antivirus). On final failure the caller PRESERVES tmp and
    names it in the raised error -- the new content is in there."""
    for attempt in range(_REPLACE_TRIES):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if attempt == _REPLACE_TRIES - 1:
                raise
            time.sleep(_REPLACE_DELAY_S)


def _backup_existing_json(target: Path) -> None:
    """Best-effort .bak copy of the current target -- but only if it
    still parses as JSON: promoting an externally-corrupted main over
    the last good backup would destroy the one recoverable copy (fold
    V1/V2/V3). copy2 (not rename) so the target never ceases to exist."""
    if not target.exists():
        return
    try:
        with open(target, "r", encoding="utf-8") as f:
            json.load(f)
    except Exception:
        return
    try:
        shutil.copy2(target, target.with_name(target.name + ".bak"))
    except OSError:
        pass


def _finish(tmp: Path, target: Path, backup: bool) -> None:
    """Shared tail: optional validated backup, then the atomic swap.
    Cleanup contract: tmp is removed on success; on a replace failure
    tmp is KEPT and named in the error (it holds the only copy of the
    new content -- evidence map section 7.3)."""
    if backup:
        _backup_existing_json(target)
    try:
        _atomic_replace(tmp, target)
    except PermissionError as e:
        raise PermissionError(
            f"{e} -- the new content is preserved at {tmp}") from e
    try:
        tmp.unlink()
    except OSError:
        pass


def _cleanup_tmp(tmp: Path) -> None:
    try:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        pass


def atomic_write_json(target: Union[str, Path], obj: Any,
                      backup: bool = False) -> None:
    """Serialize obj fully to a same-directory temp file, fsync, swap."""
    target = Path(target)
    tmp = _tmp_for(target)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        _cleanup_tmp(tmp)
        raise
    _finish(tmp, target, backup)


def atomic_write_path(target: Union[str, Path],
                      write_to: Callable[[str], None],
                      backup: bool = False) -> None:
    """
    For writers that need a PATH (DataFrame.to_csv / to_parquet):
    write_to(tmp_path) must produce the complete file; it is then
    fsynced and swapped in.
    """
    target = Path(target)
    tmp = _tmp_for(target)
    try:
        write_to(str(tmp))
        # Reopen read-write for a writable handle purely to fsync.
        # 'r+b' (not 'ab') so a writer that silently produced NOTHING
        # raises FileNotFoundError here instead of installing a 0-byte
        # file over good content (fold V2).
        with open(tmp, "r+b") as f:
            os.fsync(f.fileno())
    except BaseException:
        _cleanup_tmp(tmp)
        raise
    _finish(tmp, target, backup)
