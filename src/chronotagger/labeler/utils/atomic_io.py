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
- Directory fsync (full POSIX rename-durability) is OPT-IN via
  sync_dir=True, and Pack 5 (R8) wires it to the writes a USER
  initiated -- session save, Save As, exports -- and never to the
  per-gesture autosave, where ext4 measured +8.94 ms per write (+90% of
  the autosave path). Atomicity holds without it either way.
  CORRECTION to what this docstring used to claim: on Windows the idiom
  fails one step EARLIER than "os.fsync raises". It is
  os.open(dir, O_RDONLY) that raises PermissionError (errno 13); fsync
  is never reached, and os.O_DIRECTORY does not exist on the platform at
  all (measured, pack5_g1 5c / S8). The cost there is one raised and
  caught OSError -- 0.0675 ms per call, measured over 2,000 -- so the
  guard has to sit around the open.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Union

logger = logging.getLogger(__name__)

_REPLACE_TRIES = 5
_REPLACE_DELAY_S = 0.05

# Pack 6 F7: warn-once per SESSION, module level -- the fastdraw.py:9-11
# shape. A directory-fsync failure is a property of the volume, so a
# per-write record would spam an autosave loop on a failing disk.
_dirsync_warn_logged = False


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


def _sync_dir(target: Path) -> None:
    """Best-effort directory fsync -- rename-durability for USER-initiated
    writes only (Pack 5 R8). On Windows the open itself raises
    PermissionError before fsync is reachable, so the cost there is the
    cost of raising and catching one OSError: MEASURED 0.0675 ms per call
    over 2,000 calls -- effectively free, not literally zero. On the
    Dell's ext4 the fsync does run, +8.94 ms per write. Never raises: a
    durability nicety may not fail a save whose bytes are already on
    disk."""
    try:
        fd = os.open(str(target.parent), os.O_RDONLY)
    except (OSError, ValueError):
        return
    try:
        os.fsync(fd)
    except OSError:
        # Pack 6 F7. On POSIX an EIO here means the rename may not be
        # durable -- exactly the guarantee sync_dir=True was added (Pack 5
        # R8) to give a USER-initiated save. Before this line the failure
        # was completely invisible: injecting errno 5 at this call
        # produced no exception, no log record, and a user told the save
        # had succeeded. Still never raises -- a durability nicety may not
        # fail a save whose bytes are already on disk -- but it is on the
        # record now.
        #
        # Deliberately NOT around the os.open above: that raises
        # PermissionError on EVERY Windows call by design (0.0675 ms per
        # call, measured over 2,000), so a warning there would fire once
        # per session on every Windows box and mean nothing.
        global _dirsync_warn_logged
        if not _dirsync_warn_logged:
            _dirsync_warn_logged = True
            logger.warning(
                "directory fsync failed for %s; the rename may not be "
                "durable on this filesystem", target.parent, exc_info=True)
    finally:
        os.close(fd)


def _finish(tmp: Path, target: Path, backup: bool,
            sync_dir: bool = False) -> None:
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
    if sync_dir:
        _sync_dir(target)
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
                      backup: bool = False, sync_dir: bool = False) -> None:
    """Serialize obj fully to a same-directory temp file, fsync, swap.
    sync_dir=True adds the directory fsync (see _sync_dir): pass it for
    writes the USER asked for, never for the autosave."""
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
    _finish(tmp, target, backup, sync_dir)


def atomic_write_path(target: Union[str, Path],
                      write_to: Callable[[str], None],
                      backup: bool = False, sync_dir: bool = False) -> None:
    """
    For writers that need a PATH (DataFrame.to_csv / to_parquet):
    write_to(tmp_path) must produce the complete file; it is then
    fsynced and swapped in. sync_dir=True adds the directory fsync
    (see _sync_dir) -- user-initiated writes only.
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
    _finish(tmp, target, backup, sync_dir)
