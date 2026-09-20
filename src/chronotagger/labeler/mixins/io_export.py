"""
Session I/O and export mixin.

All persistence writes go through utils.atomic_io: the target file
always holds either the complete old content or the complete new
content, never a truncated hybrid (Pack 2).
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List

import hashlib
import json
import logging
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np

from chronotagger.core.models import Interval
from chronotagger.core.tracks import (
    DEFAULT_TRACK_ID,
    Track,
    active_id_of,
    active_track_of,
    default_table,
    find_track,
    intervals_on,
    lane_merge_note,
    merge_track_tables,
    stray_tracks,
    table_of,
    union_covered,
)
from chronotagger.core.commands import check_interval_invariants
from ..utils.atomic_io import atomic_write_json, atomic_write_path

logger = logging.getLogger(__name__)


def _norm_iso(ts: pd.Timestamp) -> str:
    """tz-normalized isoformat: tz-aware timestamps are converted to
    UTC and made naive, so a naive frame and its UTC-localized twin
    produce IDENTICAL strings. Used by BOTH the fingerprint and the
    saved/compared time_range, so the identity check can never
    contradict the fingerprint about timezones (fold V1/V2/V3 tz)."""
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.isoformat()


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """
    12-hex identity used in the autosave filename: sha1 of sorted
    column names + tz-normalized index bounds + row count (grill Q1,
    recipe R1). Stable across column reorder, dtype casts, and value
    edits; changes when columns are added/renamed or the time range
    changes. HONEST LIMIT (evidence map 3b): two datasets with
    identical column names AND identical time coverage (e.g. two
    spacecraft through a shared loader) share a fingerprint -- the
    source_name comparison in _check_autosave is the guard for that
    case, when a source name is known.
    """
    key = "|".join([
        ",".join(sorted(str(c) for c in df.columns)),
        _norm_iso(df.index[0]),
        _norm_iso(df.index[-1]),
        str(len(df.index)),
    ])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def write_label_map_sidecar(classes, data_path: str, what: str,
                            stem_suffix: str = "") -> None:
    """Write ``<stem>_label_map<stem_suffix>.json`` beside ``data_path``.

    Pack 6 F14. Pack 5's R7 made ``export_per_sample`` write integer ids,
    which was right -- it separated class 0 ``UNKNOWN`` from unlabelled
    ``-1``, which the old string column could not -- but the id -> name
    mapping then lived only in ``self.classes``, in memory, and was
    written nowhere. A programmatic export was a column of integers with
    nothing on disk saying what they meant.

    The GUI "Export Labels" path already did the right thing; this is
    that block, lifted so both callers share one implementation.

    Written AFTER the data file, so the only reachable partial state is
    "complete data, missing or stale map" -- and that state is reported
    rather than swallowed. The mapping follows the current class
    ordering, for stability.

    Pack M1 adds ``stem_suffix``: ONE SIDECAR PER TRACK. It is the empty
    string for the default track, so a single-track session's sidecar
    keeps its exact filename and its exact bytes, and it is
    ``"__<track id>"`` for every other track. That asymmetry is the whole
    reason the acceptance floor survives -- measured on a 115,339-row
    real ARTEMIS frame with 85 real hand-drawn intervals, the
    single-track sidecar is byte-identical
    (``fdb43a898e0468434f4af4b97d5a0a86f2d1cb8c30f1fa2dd24f29a0d7d3b3c8``).

    Module-level, not a method, deliberately: tests/test_persistence_
    safety.py binds a NAMED LIST of mixin methods onto a GUI-free host,
    so a new method would have had to be added to that list. A free
    function keeps that harness untouched, which is part of the evidence
    that this refactor changes nothing.
    """
    label_to_id = {label: i for i, label in enumerate(classes)}
    sidecar = Path(data_path).with_name(
        Path(data_path).stem + "_label_map" + stem_suffix + ".json")
    try:
        atomic_write_json(sidecar, label_to_id, sync_dir=True)
    except Exception as e:
        raise RuntimeError(
            f"The {what} was written to {data_path}, but the label "
            f"map sidecar failed: {e}") from e


# ---------------------------------------------------------------- Pack M1
# Version 2 of the session and autosave payloads, the gate that refuses a
# newer one, and the per-track naming and accounting helpers. All
# module-level and all free functions, for the write_label_map_sidecar
# reason above: the GUI-free hosts in tests/ bind a named list of METHODS,
# and nothing here needs to be on that list.

SESSION_VERSION = 2


class SessionVersionError(RuntimeError):
    """A session or autosave was written by a NEWER ChronoTagger.

    Raised on the HEAD-LESS path only. In a GUI session the same refusal
    is a messagebox, because a modal in a display-less script hangs
    forever -- the two-channel split _save_session already uses.
    """


def validate_track_table(rows, what):
    """The table rules that must hold wherever a table ARRIVES.

    v2 fold F2. _build_track_table (app.py) enforces these for a
    caller's tracks= argument, and DR13 leans on that refusal for the
    whole enforcement of "the id 'default' is reserved". A table read
    off DISK arrives by a different door and needs the same refusals,
    or a hand-edited or foreign session re-creates exactly the failure
    the constructor exists to prevent: measured on v1 of this pack, two
    rows sharing one id LOADED, gave the two lanes ONE export column
    and ONE sidecar, and made the orphan gate report the FIRST row's
    perfectly legal label as an orphan because it keys a dict by id and
    the last row wins. A classless row loaded and then died on a bare
    IndexError out of self.current_class_var.set(self.classes[0]).

    Pack M3.1: the rules themselves moved to
    `core.tracks.check_track_table`, which _build_track_table (app.py)
    now calls too, so there is ONE copy of them and the Manage Lanes
    box did not add a third. The MESSAGES are unchanged: passing `what`
    selects this door's voice.
    """
    from chronotagger.core.tracks import check_track_table
    return check_track_table(rows, what)


def label_id_column(track_id) -> str:
    """The per-sample export column name for one track.

    ``label_id`` for the track whose id is ``default``,
    ``label_id__<track id>`` for every other track. This is the ONLY
    scheme under which a single-track session's per-sample export is
    byte-for-byte today's file: measured on a 115,339-row real ARTEMIS
    frame, parquet
    ``c14add1527f58ef0d13c5b7e91dd8782773c9502e71c5e062934c5b9207cf35a``
    and CSV ``0ce94caed27b208d06218e22c223c7fd869a1c4898737b32baa48b862ee916f6``
    both ways round. A uniform ``label_id__<id>`` for every track is
    tidier and breaks that floor on day one.

    The price is an asymmetry: ``label_id`` and ``label_id__quality`` in
    one file look like two different kinds of thing and are not. The
    price is worth paying because a track id is IMMUTABLE, so renaming a
    track can never silently change a consumer's column name.
    """
    return ("label_id" if track_id == DEFAULT_TRACK_ID
            else "label_id__%s" % track_id)


def label_map_stem_suffix(track_id) -> str:
    """The sidecar filename suffix for one track. See label_id_column."""
    return "" if track_id == DEFAULT_TRACK_ID else "__%s" % track_id


def payload_version(data):
    """Read a payload's format version. Returns (found, readable).

    ABSENT MEANS 1. That is not a guess: the autosave has never carried
    a version key at all, and all 33 of the live autosaves on disk are
    exactly that shape (one shape, no variants).

    A bool is refused even though bool is an int subclass, because
    ``"version": true`` is a malformed file, not version 1.
    """
    if not isinstance(data, dict) or "version" not in data:
        return 1, True
    found = data.get("version")
    if isinstance(found, bool) or not isinstance(found, int):
        return found, False
    return found, 1 <= found <= SESSION_VERSION


def refuse_future_version(host, found, what, path) -> None:
    """Refuse a payload this build cannot read, loudly, having touched
    nothing.

    Two channels, the split _save_session already uses: a GUI session
    gets one ``showerror``; a head-less one gets SessionVersionError,
    because a modal would hang a display-less script forever.

    The caller must not have written ANY live state before calling this.
    """
    # v2 fold F7: DR5 refuses version 0 and below as well, and they were
    # being refused with the words "written by a NEWER ChronoTagger" --
    # the right refusal with the wrong diagnosis.
    kind = ("a NEWER ChronoTagger"
            if isinstance(found, int) and not isinstance(found, bool)
            and found > SESSION_VERSION
            else "something this build does not recognise")
    body = ("This %s file was written by %s.\n\n"
            "    file format version: %s\n"
            "    this build reads:    %s\n\n"
            "It has NOT been loaded. Nothing in this session changed.\n"
            "Upgrade ChronoTagger, or open the file with the build that "
            "wrote it.\n\n%s"
            % (what, kind, found, SESSION_VERSION, path))
    status = ("Refused: %s file format version %s (this build reads %s)"
              % (what, found, SESSION_VERSION))
    sv = getattr(host, "status_var", None)
    if getattr(host, "root", None) is None:
        if sv is not None:
            sv.set(status)
        raise SessionVersionError(body)
    messagebox.showerror("Cannot Open: Newer File Format", body)
    if sv is not None:
        sv.set(status)


def orphan_labels_by_track(intervals, tracks):
    """Two kinds of orphan, kept apart. Returns (by_track, orphan_tracks).

    by_track      {track id: [labels that track's class set does not hold]}
    orphan_tracks [track ids no row in the table carries]

    They are separate because an interval on a track that does not exist
    has NO class set to check its label against, so the label gate
    cannot even run on it -- and silently dropping it re-creates the
    ``-1`` ambiguity one level up.

    The flat version of this gate does not merely under-report: measured,
    with an active track that does not know another track's perfectly
    legal class, it BLOCKS EVERY EXPORT.
    """
    known = {t.id: set(t.classes) for t in tracks}
    by_track = {}
    for iv in intervals:
        if iv.track not in known:
            continue
        if iv.label not in known[iv.track]:
            by_track.setdefault(iv.track, set()).add(iv.label)
    return ({k: sorted(v) for k, v in sorted(by_track.items())},
            stray_tracks(intervals, tracks))


def format_orphans(by_track, orphan_tracks) -> str:
    """One human-readable block naming the track for every orphan."""
    lines = ["  %s: %s" % (tid, ", ".join(labels))
             for tid, labels in sorted(by_track.items())]
    if orphan_tracks:
        lines.append("  intervals on tracks that do not exist: %s"
                     % ", ".join(orphan_tracks))
    return "\n".join(lines)


def refuse_export_orphans(intervals, tracks, what) -> None:
    """Refuse a PROGRAMMATIC export rather than write a silent -1.

    Two distinct refusals, because they are two different faults:
      * a label outside its OWN track's class set;
      * an interval on a track that is not in the table at all.

    A raise, not a dialog: export_intervals already sets that precedent
    ("Scripts must fail loudly: a print-and-return leaves a pipeline with
    no exception and no file"). Measured on the shipped tree, these two
    paths had NO gate at all -- export_per_sample wrote 1,160,926 bytes
    with ZERO dialogs, 85 orphan-labelled rows rendered as -1, and the
    orphan absent from the sidecar. That is exactly the corrupted
    training data the GUI gate's own error text describes, on the paths a
    driver script actually calls.

    The cost, stated: this can break a script that is relying on today's
    silence.
    """
    by_track, orphan_tracks = orphan_labels_by_track(intervals, tracks)
    if orphan_tracks:
        raise ValueError(
            "Refusing the %s: %d interval(s) sit on tracks that are not in "
            "the track table (%s). An interval on a track that does not "
            "exist has no class set to check its label against, so it "
            "cannot be exported honestly."
            % (what, sum(1 for iv in intervals if iv.track in orphan_tracks),
               ", ".join(orphan_tracks)))
    if by_track:
        raise ValueError(
            ("Refusing the %s: these interval labels are not in their own "
             "track's class set and would be written as -1 (unlabeled), "
             "which is indistinguishable from unlabeled in the output and "
             "absent from the label-map sidecar:" % (what,))
            + "\n" + format_orphans(by_track, []))


def label_column_collision(df, tracks):
    """Source column names this export would overwrite. Pack M1.

    Today a source column already named ``label_id`` is SILENTLY
    overwritten -- measured, source ``[0,1,2,0,1,2,...]`` written as
    ``[-1,-1,...]`` with zero dialogs. The column ingest makes that
    likely rather than exotic: the natural workflow is "ingest the
    label_id column from my model's predictions parquet, then export the
    full frame plus my hand labels", and that IS this collision.
    """
    taken = {label_id_column(t.id) for t in tracks}
    return sorted(c for c in (str(x) for x in df.columns) if c in taken)


def label_id_series_for_track(intervals, index, track) -> pd.Series:
    """
    THE ONE per-sample label-id writer (Pack M1).

    Build a vectorized per-sample label_id series for ONE track,
    aligned to `index`.

    Mapping:
      that track's classes -> ids (0..N-1), in the track's own order
      rows no interval OF THAT TRACK covers -> -1

    Uses the smallest feasible integer dtype to keep CSVs compact. Two
    tracks in one file can therefore carry DIFFERENT dtypes, because
    the dtype follows each track's own class count (3-127 classes ->
    int8, 128+ -> int16). Today's one-column contract never had to say
    this out loud.

    `index` IS AN ARGUMENT, not self.df.index, so that the export
    dialog's PREVIEW is a SLICE of exactly these bytes. The preview used
    to be a SECOND implementation with a DIFFERENT arbitration rule --
    FIRST-match-wins against this function's LAST-writer-wins -- and on a
    deliberately overlapping pair the two disagreed on 20 of 60 previewed
    rows (0 of 60 on a non-overlapping control). Tracks make cross-track
    overlap legal, so that divergence became reachable the day this pack
    shipped; the second implementation is deleted.

    Module-level, like every other helper in this block, for the
    write_label_map_sidecar reason -- no mock host's bound-method list
    needs a new name for it -- and because it needs nothing from a
    labeler but three values.
    """
    # Stable, deterministic mapping from this track's class ordering
    label_to_id = {label: i for i, label in enumerate(track.classes)}
    unknown_id = -1

    # Smallest int dtype that fits the number of classes
    n = len(label_to_id)
    if n <= np.iinfo(np.int8).max:
        dtype = np.int8
    elif n <= np.iinfo(np.int16).max:
        dtype = np.int16
    else:
        dtype = np.int32

    idx = index
    ids = np.full(len(idx), fill_value=unknown_id, dtype=dtype)

    # Index.searchsorted does NOT validate sortedness. On a frame that
    # is not monotonic -- two spacecraft concatenated without a
    # re-sort, say -- it returned a bogus slice and MISLABELLED:
    # measured, 95 rows set where 185 are contained, and not even a
    # subset of the right ones (pack5_g1 6b / S9). The slice stays for
    # the monotonic case, where [searchsorted(start), searchsorted(end))
    # IS exactly {ts : start <= ts < end}; otherwise the half-open
    # boolean mask, which agrees with Interval.contains by
    # construction. Fixing it (rather than raising) is the ruling:
    # export has no current raise behaviour to preserve, it has a
    # silently wrong one to correct (Pack 5 R7).
    monotonic = bool(getattr(idx, "is_monotonic_increasing", False))

    # WITHIN one track the intervals are non-overlapping by
    # construction -- the strict invariant is per track and that is
    # what enforces it. ACROSS tracks they may overlap, which is why
    # this loop filters instead of trusting the whole list.
    for iv in intervals:
        if iv.track != track.id:
            continue
        code = label_to_id.get(iv.label, unknown_id)
        if monotonic:
            s = idx.searchsorted(iv.start, side="left")
            e = idx.searchsorted(iv.end, side="left")
            if s < e:
                ids[s:e] = code
        else:
            mask = (idx >= iv.start) & (idx < iv.end)
            if mask.any():
                ids[mask] = code

    return pd.Series(ids, index=idx, name=label_id_column(track.id))


def label_stats_by_track(intervals, tracks):
    """{track id: {label: {count, duration_hours}}}. Pack M1.

    The flat shape summed two lanes that share a class name into one
    entry -- measured, five keys for two tracks -- so both the count and
    the hours were silently wrong. Every track in the table gets a key,
    including an empty one, so a reader can tell "no intervals" from "no
    such track".
    """
    out = {t.id: {} for t in tracks}
    for iv in intervals:
        bucket = out.setdefault(iv.track, {})
        entry = bucket.setdefault(iv.label, {'count': 0, 'duration_hours': 0})
        entry['count'] += 1
        entry['duration_hours'] += (
            (iv.end - iv.start).total_seconds() / 3600)
    return out


def flatten_label_stats(label_stats):
    """Sum a label_stats payload down to {label: {count, duration_hours}}.

    Accepts BOTH shapes, because the recovery dialog can be handed
    either: v1's flat ``{label: {...}}`` from an autosave already on
    disk, and v2's ``{track: {label: {...}}}``. A leaf is recognised by
    carrying a 'count' or a 'duration_hours' key, which no track id can.

    The dialog renders the union of the lanes rather than one lane,
    because "Intervals by Label" is a summary and a per-lane breakdown is
    M2's Track column.
    """
    out = {}

    def add(label, s):
        if not isinstance(s, dict):
            return
        leaf = out.setdefault(str(label), {'count': 0, 'duration_hours': 0})
        leaf['count'] += s.get('count', 0) or 0
        leaf['duration_hours'] += s.get('duration_hours', 0) or 0

    for key, value in (label_stats or {}).items():
        if not isinstance(value, dict):
            continue
        if 'count' in value or 'duration_hours' in value:
            add(key, value)
        else:
            for label, s in value.items():
                add(label, s)
    return out


class IOExportMixin:
    def _dataset_fingerprint(self) -> str:
        """Fingerprint of the currently loaded DataFrame (see
        dataset_fingerprint)."""
        return dataset_fingerprint(self.df)

    # ---- Public convenience wrappers ----
    def save(self, path: Optional[str] = None) -> None:
        self._save_session(path)

    def load(self, path: str) -> None:
        self._load_session(path)

    def export_intervals(self, path: str, fmt: str = "parquet") -> None:
        if not self.intervals:
            # Scripts must fail loudly: a print-and-return leaves a
            # pipeline with no exception and no file (grill Q7).
            raise ValueError("No intervals to export.")
        # Pack M1: the two PROGRAMMATIC export paths had NO orphan gate at
        # all. Measured, this one wrote the orphan string verbatim and
        # export_per_sample wrote 85 rows as -1 with ZERO dialogs and the
        # orphan absent from the sidecar -- the corrupted training data the
        # GUI gate's own error text describes, on the paths a driver script
        # actually calls. A raise, not a dialog, for the reason two lines
        # above.
        refuse_export_orphans(self.intervals, table_of(self),
                              "intervals export")
        rows = [
            # Pack M1: `track` is present ALWAYS, not only when there is
            # more than one lane. export_intervals has no byte-identity
            # floor -- R2 grants one only to the per-sample column and its
            # sidecar -- and a CONDITIONAL schema is a worse contract than
            # a changed one, because it forces every reader to branch on
            # the file's own shape.
            {"start": iv.start, "end": iv.end, "track": iv.track,
             "label": iv.label, "notes": iv.notes}
            for iv in self.intervals
        ]
        df_export = pd.DataFrame(rows)
        if fmt.lower() == "parquet":
            atomic_write_path(path, lambda p: df_export.to_parquet(p, index=False),
                              sync_dir=True)
        else:
            atomic_write_path(path, lambda p: df_export.to_csv(p, index=False),
                              sync_dir=True)
        logger.info("Exported intervals to %s", path)

    def export_per_sample(self, path: str, fmt: str = "parquet") -> None:
        """
        Write one integer label id per row of ``self.df``.

        Output contract (Pack 5 R7, a CLEAN BREAK from the old string
        column):

            label_id = index into ``self.classes`` for a covered row,
                       and -1 for a row no interval covers.

        The old shape wrote label STRINGS and spelled uncovered rows
        ``label_on_uncovered="UNKNOWN"`` -- which is also the default name
        of class 0, so a genuinely UNKNOWN-labelled sample and an
        unlabelled one were the same characters in the file and could not
        be told apart. Ids separate them: 0 against -1. The
        ``label_on_uncovered`` argument is gone with the ambiguity it
        encoded.

        It is also 31,780x faster at 100 intervals, which is the reason
        this edit exists. The old loop was O(rows x intervals) in PYTHON:
        measured 41.518 s on the user's 1,464,070-row frame with 100
        intervals and seven and a half MINUTES with 2,000, against
        1.31 ms and 19.19 ms for the vectorized path that already shipped
        in this file (pack5_g1 section 6).
        """
        if not self.intervals:
            raise ValueError("No intervals to export.")

        tracks = table_of(self)
        # Pack M1: the programmatic path gets the gate the GUI has always
        # had, as a raise rather than a dialog.
        refuse_export_orphans(self.intervals, tracks, "per-sample export")

        # ONE COLUMN PER TRACK (R2). With a single default track the frame
        # is exactly {"label_id": <series>} and the FILE IS BYTE-FOR-BYTE
        # TODAY'S: measured on a 115,339-row real ARTEMIS frame with 85
        # real hand-drawn intervals, parquet
        # c14add1527f58ef0d13c5b7e91dd8782773c9502e71c5e062934c5b9207cf35a,
        # CSV 0ce94caed27b208d06218e22c223c7fd869a1c4898737b32baa48b862ee916f6,
        # sidecar fdb43a898e0468434f4af4b97d5a0a86f2d1cb8c30f1fa2dd24f29a0d7d3b3c8
        # -- with a same-frame-twice determinism control in the same run,
        # so that identity is a meaningful pin and not an accident.
        df_export = pd.DataFrame(
            {label_id_column(t.id): label_id_series_for_track(
                self.intervals, self.df.index, t)
             for t in tracks},
            index=self.df.index)
        if fmt.lower() == "parquet":
            atomic_write_path(path, lambda p: df_export.to_parquet(p),
                              sync_dir=True)
        else:
            atomic_write_path(path, lambda p: df_export.to_csv(p),
                              sync_dir=True)
        # Pack 6 F14: both branches write ids, so the map is not optional.
        # Pack M1: ONE SIDECAR PER TRACK. The default track keeps the name
        # <stem>_label_map.json; every other track gets
        # <stem>_label_map__<id>.json.
        for t in tracks:
            write_label_map_sidecar(t.classes, path, "per-sample export",
                                    label_map_stem_suffix(t.id))
        logger.info("Exported per-sample labels to %s", path)

    # ---- GUI-connected ops ----
    def _save_session(self, path: Optional[str] = None) -> bool:
        """Save the session JSON atomically. Returns True only if the
        file was actually written (False on dialog cancel or failure) --
        _on_closing depends on this to never close on a failed save."""
        target = Path(path) if path else None
        if target is None:
            chosen = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            if not chosen:
                return False
            target = Path(chosen)

        tracks = table_of(self)
        data = {
            "version": SESSION_VERSION,
            # Pack M1 v2: the TRACK TABLE is the only schema authority in
            # the file. The old top-level "classes" / "class_colors" keys
            # are GONE, deliberately, and the consequence was measured
            # both ways round: a pre-M1 build handed a v2 session raises
            # KeyError: 'classes' and loads nothing -- it fails CLOSED.
            # Keeping a mirror would put two copies of one schema in one
            # file, and the only thing it would buy is an old build
            # loading the file and flattening every track into one, which
            # is the outcome the version gate exists to prevent.
            "tracks": [t.to_dict() for t in tracks],
            # View state that must persist, so it lives at the top level
            # and not on a row. M1 ships no control that changes it.
            "active_track": active_id_of(self),
            "window": str(self.window),
            "step": str(self.step),
            "data_start": self.data_start.isoformat(),
            "data_end": self.data_end.isoformat(),
            "intervals": [iv.to_dict() for iv in self.intervals],
            "layout_spec": self.layout_spec,  # Save layout configuration
            # Multi-pane metadata
            "multi_pane_mode": getattr(self, 'multi_pane_mode', False),
            "active_pane_idx": getattr(self, 'active_pane_idx', 0) if getattr(self, 'multi_pane_mode', False) else 0,
            "panes": [
                {
                    "title": pane.title,
                    "layout_spec": getattr(pane, 'layout_spec', None),
                }
                for pane in getattr(self, 'panes', [])
            ] if getattr(self, 'multi_pane_mode', False) else [],
        }
        try:
            # sync_dir: the user asked for this write (Ctrl+S / Save As),
            # so it gets rename-durability too -- 0 ms on Windows,
            # +8.94 ms on ext4. The autosave deliberately does not
            # (Pack 5 R8).
            atomic_write_json(target, data, sync_dir=True)
        except Exception as e:
            # GUI session: surface in a dialog and report failure.
            # Headless/library use: RE-RAISE -- a modal here would hang
            # a display-less script forever, and swallowing would be
            # the exact silent-failure shape this pack exists to kill
            # (fold V3-M: executed, it hangs).
            if getattr(self, 'root', None) is None:
                raise
            messagebox.showerror(
                "Save Failed",
                f"Could not save session:\n{e}\n\n"
                f"The previous file (if any) is unchanged.")
            if getattr(self, 'status_var', None) is not None:
                self.status_var.set("Save failed")
            return False

        self.modified = False
        if getattr(self, 'status_var', None) is not None:
            self.status_var.set(f"Saved to {target}")
        return True

    def _load_session(self, path: Optional[str] = None) -> None:
        if path is None:
            chosen = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not chosen:
                return
            path = chosen

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Pack M1: THE VERSION GATE, and it is the FIRST statement after
        # json.load for a measured reason. With the gate sitting below the
        # layout-compatibility question, a refused file first asks the
        # user to accept a layout RISK -- and if they answer No the file
        # is refused for the WRONG REASON and they never learn the real
        # one. Hoisted here, exactly one dialog appears and it is the
        # refusal. Nothing live has been touched at this point, which is
        # the other half of the contract.
        found, ok = payload_version(data)
        if not ok:
            refuse_future_version(self, found, "session", path)
            return
        migrated_from_v1 = (found == 1)

        # Check if session has layout_spec (backward compatibility)
        saved_layout = data.get("layout_spec", None)
        
        if saved_layout is not None and self.layout_spec is not None:
            # Validate against current layout
            if not self._layouts_compatible(saved_layout, self.layout_spec):
                # Show warning dialog
                result = messagebox.askyesno(
                    "Layout Mismatch",
                    "This session was saved with a different layout configuration.\n\n"
                    f"Saved:   {self._describe_layout(saved_layout)}\n"
                    f"Current: {self._describe_layout(self.layout_spec)}\n\n"
                    "Loading this session may cause display issues.\n"
                    "Continue anyway?",
                    icon='warning'
                )
                if not result:
                    self.status_var.set("Load cancelled - layout mismatch")  # type: ignore[union-attr]
                    return  # User cancelled

        # Pack M1: BUILD every piece of new state in locals, VALIDATE the
        # locals against each other, and only THEN publish. The old shape
        # assigned self.intervals first and validated last, so a corrupt
        # session left the labeler holding the bad set -- measured,
        # "previous set intact: False", and on the recovery path it left
        # raw dicts in self.intervals. There is no try/except and no saved
        # copy here because nothing is written until every check has
        # passed, which is strictly simpler than a rollback.
        if migrated_from_v1:
            # A v1 session -- or one with no version key at all -- loads
            # in MEMORY as ONE default track whose vocabulary is the
            # file's own flat "classes" list. THE FILE IS NOT REWRITTEN;
            # the next explicit Save writes v2. Say the consequence out
            # loud: the AUTOSAVE beside it goes to v2 on the first
            # gesture, so for a while one folder holds a v1 session next
            # to a v2 autosave. That is an argument for the gate on the
            # recovery path, not against the asymmetry.
            new_tracks = default_table(list(data["classes"]),
                                       dict(data["class_colors"]))
        else:
            # v2 fold F2: the two table rules _build_track_table refuses
            # for a caller's tracks= argument must also hold for a table
            # that arrives OFF DISK. Measured on v1: two rows sharing
            # one id loaded, then collapsed two lanes into one export
            # column and one sidecar.
            new_tracks = validate_track_table(
                [Track.from_dict(t) for t in data["tracks"]], "Session")
        new_active = data.get("active_track") or new_tracks[0].id
        new_intervals = [Interval.from_dict(d) for d in data["intervals"]]
        unknown = stray_tracks(new_intervals, new_tracks)
        if unknown:
            # Pack M2.7: TWO CHANNELS, the pattern refuse_future_version
            # above already uses. A GUI session got a bare traceback here
            # while the RECOVERY path caught the identical error and made
            # it a message (app.py, Pack M2.6): "a refused load is a
            # message, not a traceback" has to hold on both doors. A
            # head-less caller -- app.load(path), and the GUI-free hosts
            # in tests/test_persistence_safety.py -- still gets the raise,
            # because a modal would hang a display-less script forever.
            # Nothing has been published at this point, so live state is
            # untouched either way.
            _why = ("Session has intervals on tracks that are not in its "
                    "track table: " + ", ".join(unknown))
            if getattr(self, "root", None) is None:
                raise ValueError(_why)
            messagebox.showerror(
                "Load Failed",
                "%s\n\nIt has NOT been loaded. Nothing in this session "
                "changed.\n\n%s" % (_why, path))
            if getattr(self, "status_var", None) is not None:
                self.status_var.set(
                    "Refused %s -- it holds intervals on tracks its own "
                    "track table does not list" % (path,))
            return
        # Pack M3.0: THE FILE'S LANES, THEN THE LANES ONLY THE DRIVER
        # KNOWS. The stray-interval refusal above is deliberately keyed
        # to the FILE'S OWN table and runs BEFORE this: a file naming a
        # lane its own table lacks is internally inconsistent and is
        # refused exactly as it is today, whether or not the live table
        # happens to carry a row of that id. The merged table is then
        # validated in its own right, and nothing has been published at
        # this point, so a refusal leaves the session untouched.
        new_tracks, _n_file, _n_kept = merge_track_tables(
            new_tracks, table_of(self))
        new_tracks = validate_track_table(new_tracks, "Session")
        self._lane_merge_note = lane_merge_note(_n_file, _n_kept)
        # Validate the MERGED table against the intervals being
        # INSTALLED, not against the live table.
        check_interval_invariants(new_intervals, new_tracks)
        new_window = pd.Timedelta(data["window"])
        new_step = pd.Timedelta(data["step"])

        # Publish. The table is slice-assigned when one already exists so
        # a reference taken before the load still points at the live list.
        if getattr(self, "tracks", None) is None:
            self.tracks = new_tracks
        else:
            self.tracks[:] = new_tracks
        # Pack M2.7: ONE SENTENCE AT THE END, and this is where its first
        # clause is earned. _load_session's closing status_var.set is the
        # LAST write on this path and erases anything set before it -- the
        # "ONE SENTENCE, NOT TWO" lesson Pack M2.6 took on the lane
        # switch. So everything this load has to tell the user is
        # collected here and said once, at the bottom.
        _load_note = []
        # Pack M3.0: the merge says what it did, in the SAME sentence as
        # everything else this load reports. Empty when it kept nothing,
        # so a session saved from this driver reads exactly as it does
        # today.
        if getattr(self, "_lane_merge_note", ""):
            _load_note.append(self._lane_merge_note)
        if find_track(self.tracks, new_active):
            self._active_track_id = new_active
        else:
            # A saved active_track that names no lane used to fall back to
            # tracks[0], which can be a HIDDEN row -- a state the GUI
            # itself refuses to reach (_toggle_active_lane_visible moves
            # the active lane rather than hide it, and _cycle_active_lane
            # visits visible lanes only). resolve_active_row is Pack
            # M2.6's one authority and picks the first VISIBLE lane,
            # exactly as the painter's fallback does. The id is cleared
            # first so the resolution cannot latch onto the PREVIOUS
            # session's active lane when the new table happens to carry a
            # row of that name.
            from chronotagger.core.tracks import resolve_active_row
            self._active_track_id = None
            _row = resolve_active_row(self)
            self._active_track_id = _row.id
            _load_note.append(" -- lane '%s' no longer exists -- active "
                              "lane is now '%s'"
                              % (new_active, _row.name or _row.id))
        self._migrated_from_v1 = migrated_from_v1
        self.window = new_window
        self.step = new_step
        self.intervals = new_intervals

        # A loaded session invalidates the undo history and any selection
        # made against the previous session's interval objects. The strict
        # validation that used to sit at the END of this method is gone:
        # it ran AFTER the assignment, which is the defect above.
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.selected_interval = None
        if hasattr(self, '_clear_selected_interval_highlights'):
            self._clear_selected_interval_highlights()
        # Pack M2.7: A LOAD DROPS A STAGED RULE, exactly as a lane switch
        # does (Pack M2.6, lane_controls.py). The spans in `_commit_spans`
        # and the yellow in `current_spans` were carved against the
        # session being replaced, so one press of Add after a load
        # committed the PREVIOUS session's geometry onto the loaded one --
        # measured, and the Replace branch wrote three UNKNOWN intervals
        # over the loaded lane and destroyed its one interval
        # (probe_s4_save_load Q2).
        if (getattr(self, "_commit_spans", None)
                or getattr(self, "current_spans", None)):
            if hasattr(self, "_clear_preview_state"):
                self._clear_preview_state()
                _load_note.append(" -- rule preview cleared")

        self.modified = False

        if self.class_combo is not None and self.current_class_var is not None:
            self.class_combo["values"] = self.classes
            if self.current_class_var.get() not in self.classes:
                self.current_class_var.set(self.classes[0])

        if self.start_time_entry and self.end_time_entry and self.step_entry:
            self.start_time_entry.delete(0, tk.END)
            self.start_time_entry.insert(0, str(self.t0))
            self.end_time_entry.delete(0, tk.END)
            self.end_time_entry.insert(0, str(self.t1))
            self.step_entry.delete(0, tk.END)
            self.step_entry.insert(0, str(self.step))

        # Restore active tab if multi-pane
        if getattr(self, 'multi_pane_mode', False) and "active_pane_idx" in data:
            idx = data["active_pane_idx"]
            if hasattr(self, 'notebook') and hasattr(self, 'panes'):
                if 0 <= idx < len(self.panes):
                    self.active_pane_idx = idx
                    self.notebook.select(idx)

        # Restore pane titles if saved
        if getattr(self, 'multi_pane_mode', False) and "panes" in data:
            saved_panes = data["panes"]
            for i, saved_pane in enumerate(saved_panes):
                if i < len(self.panes) and "title" in saved_pane:
                    self.panes[i].title = saved_pane["title"]
                    if hasattr(self, 'notebook'):
                        self.notebook.tab(i, text=saved_pane["title"])

        self._update_plot()
        # Pack M2.7: THE AUTOSAVE FOLLOWS THE LOAD. Without this the
        # autosave on disk still described the DISCARDED pre-load session,
        # and because the load also sets modified=False a close right
        # afterwards asks nothing -- so the next launch offered the work
        # the user had just loaded away from, with "Recover Session" as
        # the focused default button (probe_s4_save_load Q3/Q5, refuter
        # case_D). The signature skip in _save_autosave makes this free
        # whenever the load changed nothing.
        self._save_autosave()
        # Pack M1: a migrated v1 session SAYS SO, once, where the user is
        # already looking. It is the only signal that the file on disk is
        # still v1 and that the lane being labeled was synthesized from
        # the file's flat "classes" list.
        # Pack M2.7: and it says it in the SAME sentence as everything
        # else this load has to report -- a saved active lane that no
        # longer exists, a staged rule that was dropped. This set() is the
        # last write on the path and would erase any earlier line.
        if getattr(self, "_migrated_from_v1", False):
            _said = (f"Loaded from {path} as a single track "
                     f"('{DEFAULT_TRACK_ID}')")
        else:
            _said = f"Loaded from {path}"
        self.status_var.set(_said + "".join(_load_note))  # type: ignore[union-attr]
        
    def _compute_label_id_series(self) -> pd.Series:
        """The ACTIVE track's per-sample label ids.

        Kept under its old name and its old signature, because the GUI
        export path and eight existing tests call it. With one default
        track it is exactly what it always was. With more than one it is
        the ACTIVE lane, which is the R6 reading: the alternative -- a
        flat last-writer-wins pass over every lane -- was measured to
        write 1,200 of 2,400 human-labeled rows as -1, because an imported
        track's interval overwrote them with a class the active track does
        not even contain.
        """
        return label_id_series_for_track(
            self.intervals, self.df.index, active_track_of(self))

    def _export_labels_dialog(self) -> None:
        """
        Enhanced modal with live preview that lets the user choose:
          - Scope: Full dataset  |  Labeled rows only (active lane)
          - Content: Index + labels (CSV)  |  Full DF + labels (CSV)
        
        Shows real-time preview of first 10 rows and estimated total.
        Writes a CSV plus a sidecar '<chosen_name>_label_map.json'.
        """
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
    
        if self.df is None or len(self.df.index) == 0:
            messagebox.showwarning("No Data", "There is no data to export.")
            return

        # Orphan labels would render as -1 in the preview and the CSV;
        # refuse at the door so the preview never lies (grill Q4).
        # Pack M1: PER TRACK, and the message names the track. The flat
        # version did not merely under-report: measured, it BLOCKED EVERY
        # EXPORT the moment two tracks had different vocabularies, because
        # one lane's perfectly legal class is not in the other lane's
        # class set.
        _tracks = table_of(self)
        _by_track, _orphan_tracks = orphan_labels_by_track(self.intervals,
                                                           _tracks)
        if _by_track or _orphan_tracks:
            messagebox.showerror(
                "Export Blocked",
                "These interval labels are not in their own track's label "
                "schema and would be exported as -1 (unlabeled):\n\n"
                f"{format_orphans(_by_track, _orphan_tracks)}\n\n"
                "Fix them via Manage Labels..., then export again.")
            return
    
        # Modal container - larger size for side-by-side layout
        dlg = tk.Toplevel(self.root)
        dlg.title("Export Labels")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(True, True)
        dlg.geometry("800x500")  # Wider for preview
    
        # Main frame with two sides
        main_frame = ttk.Frame(dlg)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side: Options (30% width)
        options_frame = ttk.Frame(main_frame)
        options_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Right side: Preview (70% width)
        preview_frame = ttk.LabelFrame(main_frame, text="Preview")
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
        # --- Scope Options ---
        scope_var = tk.StringVar(value="full")
        scope_grp = ttk.LabelFrame(options_frame, text="Scope")
        scope_grp.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Radiobutton(
            scope_grp, text="Full dataset (unlabeled = -1)",
            variable=scope_var, value="full"
        ).pack(anchor="w", pady=2, padx=5)
        
        ttk.Radiobutton(
            # Pack M2.7: THE WORDS, NOT THE VALUE. This scope has never
            # meant "the intervals you selected" -- app.selected_interval
            # is never consulted. It means the rows the ACTIVE lane has
            # labeled, which is what _export_labels_do writes. The VALUE
            # stays "selected": seven assertions in the suite pass that
            # literal string into the writer and the preview, and a wire
            # value is not a sentence.
            scope_grp, text="Labeled rows only (active lane)",
            variable=scope_var, value="selected"
        ).pack(anchor="w", pady=2, padx=5)
    
        # --- Content Options ---
        content_var = tk.StringVar(value="index_labels_csv")
        content_grp = ttk.LabelFrame(options_frame, text="Content")
        content_grp.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Radiobutton(
            content_grp, text="Index + labels (CSV)",
            variable=content_var, value="index_labels_csv"
        ).pack(anchor="w", pady=2, padx=5)
        
        ttk.Radiobutton(
            content_grp, text="Full DataFrame + labels (CSV)",
            variable=content_var, value="full_df_labels_csv"
        ).pack(anchor="w", pady=2, padx=5)
    
        # --- Preview Panel ---
        preview_text = tk.Text(
            preview_frame, 
            font=("Courier", 9), 
            state="disabled",
            wrap=tk.NONE,
            bg="#f8f8f8"
        )
        
        # Add scrollbars to preview
        preview_scroll_y = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=preview_text.yview)
        preview_scroll_x = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=preview_text.xview)
        preview_text.configure(yscrollcommand=preview_scroll_y.set, xscrollcommand=preview_scroll_x.set)
        
        preview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        preview_scroll_y.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        preview_scroll_x.pack(side=tk.BOTTOM, fill=tk.X, padx=(5, 0))
    
        # --- Action Buttons ---
        btns = ttk.Frame(options_frame)
        btns.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
    
        def do_export_and_close():
            # Ask for CSV location
            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save labels CSV as…",
            )
            if not path:
                return
            try:
                # _export_labels_do returns True only when the CSV (and
                # sidecar) were actually written; its early-return paths
                # have already told the user why (grill Q7).
                if self._export_labels_do(path, scope_var.get(), content_var.get()):
                    messagebox.showinfo("Export Complete", f"Exported:\n{path}")
                    dlg.destroy()
            except Exception as e:
                messagebox.showerror("Export Failed", f"{e}")
    
        tk.Button(btns, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT, padx=(5, 0))
        tk.Button(btns, text="Export", command=do_export_and_close).pack(side=tk.RIGHT)
        
        # --- Preview Update Functions ---
        def update_preview():
            """Update preview based on current dialog options."""
            try:
                preview_df, total_estimate, info = self._generate_export_preview(
                    scope_var.get(), 
                    content_var.get(), 
                    limit=10
                )
                preview_content = self._format_dataframe_preview(preview_df, total_estimate, info)
                
                # Update preview text
                preview_text.config(state="normal")
                preview_text.delete(1.0, tk.END)
                preview_text.insert(1.0, preview_content)
                preview_text.config(state="disabled")
                
            except Exception as e:
                # Show error in preview
                error_msg = f"Preview Error:\n{str(e)}\n\nThis might indicate no data matches your selection."
                preview_text.config(state="normal")
                preview_text.delete(1.0, tk.END)
                preview_text.insert(1.0, error_msg)
                preview_text.config(state="disabled")
        
        # Connect option changes to preview updates
        scope_var.trace_add("write", lambda *args: update_preview())
        content_var.trace_add("write", lambda *args: update_preview())
        
        # Initial preview
        dlg.after(100, update_preview)  # Small delay to ensure UI is ready
        
        # Center over parent
        dlg.update_idletasks()
        if self.root is not None:
            rx = self.root.winfo_rootx()
            ry = self.root.winfo_rooty()
            rw = self.root.winfo_width()
            rh = self.root.winfo_height()
            dw = dlg.winfo_width()
            dh = dlg.winfo_height()
            dlg.geometry(f"+{rx + (rw - dw)//2}+{ry + (rh - dh)//2}")

    def _export_labels_do(self, csv_path: str, scope: str, content: str) -> bool:
        """
        Core writer for labels CSV + sidecar label_map.json.
    
        Parameters
        ----------
        csv_path : str
            Destination CSV path (chosen by user).
        scope : {"full","selected"}
            "full"     → all rows included, unlabeled rows get -1.
            "selected" → only rows that fall within labeled intervals (label_id != -1).
        content : {"index_labels_csv","full_df_labels_csv"}
            Index + labels only, or full DF with labels appended.

        Returns True only if the CSV (and sidecar) were written.
        """
        import json
        from pathlib import Path
        import pandas as pd

        # Refuse to export labels the schema does not contain: they
        # would silently collapse to -1 (= unlabeled) in the CSV and be
        # absent from the sidecar -- corrupted training data with no
        # warning (grill Q4; reproduced in the Pack 2 evidence map).
        # Pack M1: PER TRACK, and the message names the track.
        tracks = table_of(self)
        by_track, orphan_tracks = orphan_labels_by_track(self.intervals,
                                                         tracks)
        if by_track or orphan_tracks:
            from tkinter import messagebox
            messagebox.showerror(
                "Export Blocked",
                "These interval labels are not in their own track's label "
                "schema and would be exported as -1 (unlabeled):\n\n"
                f"{format_orphans(by_track, orphan_tracks)}\n\n"
                "Fix them via Manage Labels..., then export again.")
            return False

        # Build one label_id column per track. Pack M1: with a single
        # default track this is the same single column under the same
        # name, so the file is byte-identical to today's.
        columns = {label_id_column(t.id): label_id_series_for_track(
            self.intervals, self.df.index, t) for t in tracks}

        # The ACTIVE track decides what "selected" means. With K lanes the
        # test "is this row labeled" has K answers, and the union answer
        # produces rows that are -1 in most columns -- the same "is this
        # unlabeled, or is it outside this lane" ambiguity Pack 5 R7 spent
        # a clean break to remove.
        active = active_track_of(self)
        label_id = columns[label_id_column(active.id)]

        if scope == "selected":
            mask = label_id.values != -1
            if not mask.any():
                from tkinter import messagebox
                messagebox.showwarning("No Labeled Samples", "There are no labeled samples in the current data.")
                return False
            idx = self.df.index[mask]
            columns = {k: v.loc[idx] for k, v in columns.items()}
            label_id = columns[label_id_column(active.id)]
            df_source = self.df.loc[idx]
        else:
            # full dataset (unlabeled = -1)
            df_source = self.df

        # Assemble output frame
        if content == "index_labels_csv":
            out = pd.DataFrame(columns, index=label_id.index)
            out.index.name = "time"
        else:  # "full_df_labels_csv"
            # Pack M1: REFUSE rather than overwrite. A source column
            # already named label_id used to be silently overwritten --
            # measured, source first ten [0,1,2,0,1,2,0,1,2,0] written as
            # [-1,-1,...] with zero dialogs. The column ingest makes that
            # likely rather than exotic.
            clash = label_column_collision(df_source, tracks)
            if clash:
                from tkinter import messagebox
                messagebox.showerror(
                    "Export Blocked",
                    "The data already has a column named "
                    f"{', '.join(clash)}, and this export would overwrite "
                    "it.\n\nRename the source column, or choose "
                    "'Index + labels' instead, then export again.")
                return False
            out = df_source.copy()
            for _name, _series in columns.items():
                out[_name] = _series.astype(_series.dtype)
            if out.index.name is None:
                out.index.name = "time"

        # Write CSV atomically (complete file or no change, never a
        # valid-looking truncated training set)
        atomic_write_path(csv_path, lambda p: out.to_csv(p), sync_dir=True)

        # Write the sidecar mapping, atomically, AFTER the CSV.
        # (Pack 6 F14: this block moved into the module-level
        # write_label_map_sidecar() so export_per_sample could share
        # it. The RuntimeError text is
        # unchanged -- "The labels CSV was written to ..." -- because it
        # is what a user reads when a save half-fails.)
        # Pack M1: one sidecar per track; the default track's keeps its
        # name and its bytes.
        for t in tracks:
            write_label_map_sidecar(t.classes, csv_path, "labels CSV",
                                    label_map_stem_suffix(t.id))
        return True

    def _generate_export_preview(self, scope: str, content: str, limit: int = 10):
        """
        Generate efficient preview of export data using only first few rows.
        
        Parameters
        ----------
        scope : {"full", "selected"}
            Export scope selection
        content : {"index_labels_csv", "full_df_labels_csv"}
            Content type selection  
        limit : int
            Maximum number of rows to include in preview
            
        Returns
        -------
        tuple
            (preview_df, total_estimate, info_dict)
        """
        import pandas as pd
        
        # Build label_id efficiently for preview
        if scope == "full":
            # Take first `limit` rows from original DataFrame
            preview_input = self.df.head(limit)
            total_estimate = len(self.df)
            info = {
                "scope_desc": "Full dataset",
                "total_unlabeled": "Some rows may be unlabeled (-1)"
            }
        elif scope == "selected":
            # Get first `limit` rows that fall within labeled intervals
            preview_input, total_estimate = self._get_first_labeled_rows(limit)
            if len(preview_input) == 0:
                raise ValueError("No labeled intervals found in current data range")
            _act = active_track_of(self)
            info = {
                "scope_desc": ("Labeled rows only (active lane: %s)"
                               % (_act.name or _act.id,)),
                "total_unlabeled": "All rows are labeled"
            }
        else:
            raise ValueError(f"Unknown scope: {scope}")
        
        # Generate label_id series for preview data. Pack M1: a SLICE of
        # the real writer's output, not a second implementation.
        # _get_first_labeled_rows concatenates rows interval by interval,
        # so once cross-track overlap is legal its index can carry
        # DUPLICATES -- and a duplicated index makes searchsorted slices
        # ambiguous. Drop them first; this is a real edge the preview has
        # never had to face.
        _pidx = preview_input.index
        if not _pidx.is_unique:
            # .loc[a de-duplicated index] on a frame whose index HAS
            # duplicates returns EVERY matching row again -- measured, 19
            # rows for 14 unique stamps in pandas 2.3.3 and 3.0.2 alike --
            # so the FRAME is masked, not re-selected. Without the mask the
            # "full DataFrame + labels" preview still showed the
            # duplicated rows while "index + labels" showed 14.
            preview_input = preview_input[~_pidx.duplicated()]
            _pidx = preview_input.index
        # Pack M2.7: THE PREVIEW NAMES ITS LABEL COLUMN THE WAY THE FILE
        # NAMES IT. It was the bare "label_id", which label_id_column
        # reserves for the track whose id is "default" -- a column no
        # K-lane session can produce. Measured: preview ['a','label_id']
        # against a file ['time','a','label_id','label_id__human',
        # 'label_id__auto'] (probe_s4_export Q5).
        _preview_track = active_track_of(self)
        _label_col = label_id_column(_preview_track.id)
        info["label_column"] = _label_col
        # And what the FILE will actually cost per row, for the size
        # estimate below: the writer emits one label column PER LANE
        # beside the index, and the full-DF content carries every source
        # column as well. About 30 bytes for the ISO timestamp, 3 for a
        # label id, 20 for a source value. The old estimate multiplied the
        # PREVIEW's one label column by a row total that was itself wrong
        # -- ~0.2 MB announced for a 1,004-byte file and ~0.2 MB for a
        # 374,502-byte one.
        _n_label_cols = len(table_of(self))
        _n_data_cols = (len(self.df.columns)
                        if content == "full_df_labels_csv" else 0)
        info["est_bytes_per_row"] = (30 + 3 * _n_label_cols
                                     + 20 * _n_data_cols)
        label_id = label_id_series_for_track(
            self.intervals, _pidx, _preview_track)
        
        # Apply content formatting
        if content == "index_labels_csv":
            preview_df = pd.DataFrame({_label_col: label_id}, index=label_id.index)
            preview_df.index.name = "time"
            info["content_desc"] = "Index + labels only"
        elif content == "full_df_labels_csv":
            preview_df = preview_input.copy()
            preview_df[_label_col] = label_id.astype(label_id.dtype)
            if preview_df.index.name is None:
                preview_df.index.name = "time"
            info["content_desc"] = "Full DataFrame + labels"
        else:
            raise ValueError(f"Unknown content type: {content}")
            
        return preview_df, total_estimate, info
    
    def _get_first_labeled_rows(self, limit: int = 10):
        """
        Get first N rows that fall within labeled intervals.
        
        Parameters
        ----------
        limit : int
            Maximum number of rows to return
            
        Returns
        -------
        tuple
            (preview_df, total_labeled_count)
        """
        import pandas as pd

        if not self.intervals:
            return pd.DataFrame(), 0

        # Pack M2.7: THE PREVIEW ANSWERS THE SAME QUESTION THE WRITER
        # ANSWERS. This walked EVERY lane's intervals, so the preview for
        # a scope that writes ONE lane showed rows from lanes the file
        # never selects -- Agent rows a full day before any Wake interval.
        # Its `limit` counted interval CHUNKS, not rows, so a limit of 10
        # returned 37. And its total was a per-interval sum across every
        # lane -- 9,633 against a 9,602-row frame and 25 rows actually
        # written (probe_s4_export Q2). All three fall out of asking
        # _export_labels_do's own question instead:
        #     mask = (the ACTIVE lane's label id series) != -1
        # which is that method's `scope == "selected"` branch verbatim, so
        # the preview's total is the file's row count by construction and
        # cannot drift from it again.
        active = active_track_of(self)
        label_id = label_id_series_for_track(
            self.intervals, self.df.index, active)
        mask = label_id.values != -1
        total_labeled_count = int(mask.sum())
        if not total_labeled_count:
            return pd.DataFrame(), 0
        preview_df = self.df.loc[mask].head(limit)

        return preview_df, total_labeled_count
    
    # Pack M1: _compute_label_id_series_for_subset IS GONE.
    #
    # It was a SECOND per-sample implementation with a DIFFERENT
    # arbitration rule -- FIRST-match-wins here against the real writer's
    # LAST-writer-wins -- and on a deliberately overlapping pair the two
    # disagreed on 20 of 60 previewed rows (0 of 60 on a non-overlapping
    # control). Tracks make cross-track overlap legal, so that divergence
    # became reachable the day this pack shipped: the dialog's live
    # preview and the file the user then got would have disagreed about
    # the same rows, with nothing saying so.
    #
    # It was also the O(rows x intervals) pure-Python loop Pack 5 R7
    # deleted from the writer, kept alive here for three more packs.
    #
    # _generate_export_preview now calls the module-level
    # label_id_series_for_track() with the preview's own index: one
    # implementation, one arbitration rule, one dtype rule, one orphan
    # answer.
    
    def _format_dataframe_preview(self, preview_df, total_estimate: int, info: dict) -> str:
        """
        Format DataFrame for display in preview text widget.
        
        Parameters
        ----------
        preview_df : pd.DataFrame
            Preview data to format
        total_estimate : int
            Estimated total rows in full export
        info : dict
            Additional information about the export
            
        Returns
        -------
        str
            Formatted text for preview display
        """
        if preview_df.empty:
            return (
                "No data to preview.\n\n"
                "This might occur if:\n"
                "• No intervals are labeled\n"
                "• No data exists in the current time range\n"
                "• Selected intervals don't contain any data points"
            )
        
        lines = []
        
        # Header with summary
        if len(preview_df) < total_estimate:
            lines.append(f"Preview (first {len(preview_df)} of ~{total_estimate:,} rows):")
        else:
            lines.append(f"Preview (all {len(preview_df)} rows):")
        lines.append("")
        
        # Data preview - limit columns for readability
        display_df = preview_df.copy()
        columns_truncated = False
        
        # For wide DataFrames, show only first few columns + label_id
        max_cols = 6
        if len(display_df.columns) > max_cols:
            # Keep the label column if it exists, otherwise just take the
            # first max_cols-1. Pack M2.7: the NAME comes from the same
            # helper the preview built it with (label_id_column of the
            # active lane). The literal 'label_id' matched nothing in any
            # session whose track id is not 'default', so the wide preview
            # silently dropped the one column the user opened it for.
            _label_col = info.get("label_column", "label_id")
            if _label_col in display_df.columns:
                other_cols = [col for col in display_df.columns
                              if col != _label_col]
                cols_to_show = other_cols[:max_cols-1] + [_label_col]
            else:
                cols_to_show = list(display_df.columns[:max_cols])
            
            display_df = display_df[cols_to_show]
            columns_truncated = True
            lines.append(f"(Showing {len(cols_to_show)} of {len(preview_df.columns)} columns)")
            lines.append("")
        
        # Add visual separator before DataFrame
        lines.append("─" * 60)  # Horizontal line
        
        # Format the DataFrame with manual column truncation indication
        df_str = display_df.to_string(max_rows=20, max_cols=max_cols)
        
        # Add ellipsis to column headers if truncated
        if columns_truncated:
            df_lines = df_str.split('\n')
            if len(df_lines) > 0:
                # Add "..." to the header line
                header_line = df_lines[0]
                if not header_line.endswith('...'):
                    df_lines[0] = header_line + "  ..."
                
                # Add "..." to each data row
                for i in range(1, len(df_lines)):
                    if df_lines[i].strip() and not df_lines[i].endswith('...'):
                        df_lines[i] = df_lines[i] + "  ..."
                
                df_str = '\n'.join(df_lines)
        
        lines.append(df_str)
        
        # Add row continuation indicator if we have more rows
        if len(preview_df) < total_estimate:
            lines.append("...")
            lines.append(f"(+ {total_estimate - len(preview_df):,} more rows)")
        
        # Add visual separator after DataFrame
        lines.append("─" * 60)  # Horizontal line
        lines.append("")
        
        # Summary information
        lines.append("Export Settings:")
        lines.append(f"  Scope: {info['scope_desc']}")
        lines.append(f"  Content: {info['content_desc']}")
        if total_estimate > 1000:
            # Pack M2.7: the TRUE row total -- the writer's own mask.sum()
            # -- times what a row of the FILE costs, not what a row of the
            # PREVIEW costs. The old line multiplied the row count by the
            # preview's single label column and a flat 20 bytes, and read
            # "~0.2 MB" for a 1,004-byte file and "~0.2 MB" for a
            # 374,502-byte one.
            _per_row = info.get("est_bytes_per_row",
                                20 * len(display_df.columns))
            est_size_mb = total_estimate * _per_row / (1024 * 1024)
            lines.append(f"  Estimated file size: ~{est_size_mb:.1f} MB")
        
        # Label mapping preview. Pack M1: the ACTIVE track's map, and it
        # SAYS WHOSE MAP IT IS, because with K lanes an unnamed "Label ID
        # Mapping" is the wrong map for every non-default column.
        _tracks = table_of(self)
        _active = active_track_of(self)
        if _active.classes:
            lines.append("")
            lines.append(f"Label ID Mapping ({_active.name}):")
            for i, label in enumerate(_active.classes[:8]):  # Show first 8
                lines.append(f"  {i}: {label}")
            if len(_active.classes) > 8:
                lines.append(f"  ... and {len(_active.classes) - 8} more")
            lines.append(f"  -1: UNLABELED")
            if len(_tracks) > 1:
                lines.append(
                    f"  (+{len(_tracks) - 1} more track(s), each exporting "
                    f"its own label_id__<id> column and sidecar)")
        
        return "\n".join(lines)


    def _export_intervals(self) -> None:
        if not self.intervals:
            messagebox.showwarning("No Data", "No intervals to export.")
            return
        # Pack M2.7: THE SAME ORPHAN GATE THE TWIN HAS. export_intervals
        # (above) calls refuse_export_orphans and this copy never did --
        # two copies of one contract, drifting, which is the exact thing
        # this method's own comment below says must not happen. Pack M2.7
        # puts a BUTTON on this path, and it would otherwise have been the
        # one export control in the window with no orphan check behind it.
        # Same gate, GUI channel: a message instead of a traceback, and no
        # file.
        try:
            refuse_export_orphans(self.intervals, table_of(self),
                                  "intervals export")
        except ValueError as exc:
            messagebox.showerror("Export Blocked", str(exc))
            if getattr(self, "status_var", None) is not None:
                self.status_var.set(
                    "Intervals export refused -- labels outside their own "
                    "track's schema")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Parquet files", "*.parquet"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        rows = [
            # Pack M1: the SECOND, independent row builder for the same
            # contract. It gains the same `track` column export_intervals
            # does -- two copies of one contract must not be allowed to
            # drift, which is exactly what happened to the orphan gate.
            {"start": iv.start, "end": iv.end, "track": iv.track,
             "label": iv.label, "notes": iv.notes}
            for iv in self.intervals
        ]
        df_export = pd.DataFrame(rows)
        try:
            if path.lower().endswith(".parquet"):
                atomic_write_path(path, lambda p: df_export.to_parquet(p, index=False),
                                  sync_dir=True)
            else:
                atomic_write_path(path, lambda p: df_export.to_csv(p, index=False),
                                  sync_dir=True)
        except Exception as e:
            messagebox.showerror(
                "Export Failed",
                f"Could not export intervals:\n{e}\n\n"
                f"The previous file (if any) is unchanged.")
            return
        self.status_var.set(f"Exported to {path}")  # type: ignore[union-attr]
        messagebox.showinfo("Export Complete", f"Intervals exported to {path}")

    def _save_autosave(self) -> None:
        """Atomically save current state (intervals + label schema +
        dataset identity) to the fingerprinted autosave file, keeping
        one .bak generation."""
        from datetime import datetime

        # Use instance autosave file path
        autosave_path = self.autosave_file

        tracks = table_of(self)

        # Pack M1: SKIP THE WRITE when neither the interval set nor the
        # track table has changed since the last successful one. Autosave
        # fires per GESTURE, and one ingested rule track takes a single
        # write from 14.8 ms / 14,693 B to 64.7 ms / 392,453 B -- measured
        # on the user's own 2016 rule-label column, 2,349 intervals. The
        # signature is a tuple of field values and costs about a
        # millisecond at that size. The file's continued EXISTENCE is part
        # of the condition, so deleting the autosave and calling this
        # again still rewrites it, and the signature is only recorded
        # after a write that actually succeeded.
        # v2 fold F3: active_id_of is in the SIGNATURE because it is in
        # the PAYLOAD (DR4). M1 ships no control that changes it, so
        # this costs nothing today; the moment M2 ships a lane switch, a
        # signature without it makes the skip swallow that change and
        # the autosave re-opens on the wrong lane. Measured on v1 of
        # this pack: the file still said 'a' while the labeler said 'b'.
        signature = (
            tuple((iv.start, iv.end, iv.label, iv.notes, iv.track,
                   repr(iv.meta) if iv.meta else "")
                  for iv in self.intervals),
            tuple((t.id, t.name, tuple(t.classes),
                   tuple(sorted(t.class_colors.items())), t.kind,
                   t.locked, t.visible, t.order) for t in tracks),
            active_id_of(self),
        )
        if (signature == getattr(self, '_autosave_signature', None)
                and autosave_path.exists()):
            return

        # Calculate statistics, PER TRACK (Pack M1). The flat shape summed
        # two lanes that share a class name into one entry -- measured,
        # five keys for two tracks -- so both the count and the hours were
        # silently wrong.
        label_stats = label_stats_by_track(self.intervals, tracks)

        # Calculate coverage percentage as the UNION, not the sum. Summing
        # durations across lanes reported 100.8 % of the record on one
        # hand track plus one ingested rule track, and the recovery dialog
        # renders that number verbatim -- a user offered a recovery that
        # claims to have labelled more than all of time. With ONE track
        # the union IS the sum, to the nanosecond.
        data_duration = (self.data_end - self.data_start).total_seconds() / 3600
        union_hours = union_covered(self.intervals).total_seconds() / 3600
        coverage_percent = (union_hours / data_duration * 100) if data_duration > 0 else 0
        coverage_by_track = {}
        for _t in tracks:
            _hrs = union_covered(
                intervals_on(self.intervals, _t.id)).total_seconds() / 3600
            coverage_by_track[_t.id] = round(
                (_hrs / data_duration * 100) if data_duration > 0 else 0, 1)

        # Build autosave data structure
        autosave_data = {
            'metadata': {
                'data_columns': [str(c) for c in self.df.columns],  # For matching validation
                'dtypes': {str(c): str(t) for c, t in self.df.dtypes.items()},
                'n_rows': int(len(self.df.index)),
                'fingerprint': self._dataset_fingerprint(),
                'source_name': getattr(self, 'source_name', None),
                'autosave_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_intervals': len(self.intervals),
                'coverage_percent': round(coverage_percent, 1),
                'coverage_percent_by_track': coverage_by_track,
                # tz-NORMALIZED, matching the fingerprint and the load-time
                # comparison, so a naive/UTC-localized pair of the same
                # dataset can never self-flag as a mismatch (fold tz).
                'time_range': {
                    'start': _norm_iso(self.data_start),
                    'end': _norm_iso(self.data_end)
                }
            },
            # Pack M1 v2. The TRACK TABLE travels with the intervals so
            # recovery can restore the schema -- without a schema,
            # recovered labels outside the session's default vocabulary
            # silently export as -1 (grill Q4). The top-level 'classes' /
            # 'class_colors' mirror is GONE, and the version key is NEW:
            # this payload never had one, which is exactly why a pre-M1
            # build reading it would fail OPEN rather than closed.
            'version': SESSION_VERSION,
            'tracks': [t.to_dict() for t in tracks],
            'active_track': active_id_of(self),
            'intervals': [iv.to_dict() for iv in self.intervals],  # Convert Interval objects to dicts
            'label_stats': label_stats
        }

        # Atomic write; keep one .bak generation as the last line of
        # defence (a crash mid-write can no longer destroy recovery).
        try:
            atomic_write_json(autosave_path, autosave_data, backup=True)
        except Exception as e:
            # Silent data-loss was reported through the most ephemeral
            # channel in the app, and only if the statusbar existed
            # (Pack 4). Now: always logged; statusbar when present; a
            # dialog ONCE per session (R13) -- autosave fires per
            # gesture, so a persistent failure must not dialog-spam.
            logger.error("autosave write failed: %s", e, exc_info=True)
            if getattr(self, 'status_var', None) is not None:
                self.status_var.set(f"Autosave failed: {e}")
            # Real-Tk sessions only: MockPersistHost sets root=object()
            # on purpose, and an unstubbed dialog from the GUI-free suite
            # deadlocks the whole run (verifier B1 -- observed live).
            if (isinstance(getattr(self, 'root', None), tk.Misc)
                    and not getattr(self, '_autosave_failure_dialog_shown',
                                    False)):
                self._autosave_failure_dialog_shown = True
                try:
                    messagebox.showerror(
                        "Autosave Failed",
                        "Autosave is not working -- your labels are NOT "
                        "being backed up.\n\n"
                        f"{e}\n\n"
                        "Further failures this session are logged "
                        "(chronotagger.log) without this dialog."
                    )
                except Exception:
                    pass  # headless: the log line above is the record
        else:
            # Pack M1: only a write that SUCCEEDED may suppress the next
            # one. A failed write leaves the signature alone, so the very
            # next gesture retries exactly as it did before this pack.
            self._autosave_signature = signature

    def _check_autosave(self):
        """
        Look for THIS dataset's autosave (fingerprinted filename) and
        return its parsed contents plus identity annotations, or None.

        Clean break (Pack 2, grill Q2): only the fingerprinted JSON
        name is consulted. Pre-fingerprint files and the old pickle
        format are never read. If the main file is corrupt, the .bak
        generation written by _save_autosave is tried before giving up.
        """
        from chronotagger.core.models import Interval

        main = self.autosave_file
        bak = main.with_name(main.name + ".bak")
        main_existed = main.exists()

        # The .bak is consulted ONLY when the main file exists but is
        # unreadable. If the user deliberately deleted the named main
        # file, the .bak must not resurrect it (fold V3).
        candidates = [main]
        if main_existed:
            candidates.append(bak)

        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                with open(candidate, 'r', encoding='utf-8') as f:
                    autosave_data = json.load(f)
                # Pack M1: THE VERSION GATE, before the identity block
                # below. An AUTOSAVE HAS NEVER CARRIED A VERSION KEY --
                # all 33 of the live ones on disk are exactly that shape,
                # one shape with no variants -- so absent reads as 1.
                # This is the half of the gate that matters most:
                # measured, a pre-M1 build handed a v2 autosave recovers
                # it SILENTLY, flattens two tracks into one overlapping
                # list, keeps the live schema, and then exports 240 of
                # 240 rows as -1. The session path fails closed; the
                # autosave path fails OPEN, and only a version comparison
                # closes it.
                found, ok = payload_version(autosave_data)
                if not ok:
                    refuse_future_version(self, found, "autosave",
                                          str(candidate))
                    return None
                autosave_data['_migrated_from_v1'] = (found == 1)
                # Convert interval dicts back to Interval objects
                autosave_data['intervals'] = [
                    Interval.from_dict(d) for d in autosave_data.get('intervals', [])
                ]
            except SessionVersionError:
                # A refusal is not a corrupt file. It must not be logged
                # as "unreadable", and on the head-less path it must not
                # be swallowed by the handler below.
                raise
            except Exception as e:
                # A corrupt candidate is worth telling the user about --
                # silently pretending no autosave exists converted
                # "recoverable" into "lost" before this pack. The status
                # line is overwritten before the first frame renders
                # (Pack 4 G2 2.1), so the log carries it now too.
                logger.warning("autosave unreadable (%s): %s",
                               candidate.name, e, exc_info=True)
                if getattr(self, 'status_var', None) is not None:
                    self.status_var.set(
                        f"Autosave unreadable ({candidate.name}): {e}")
                continue

            # Identity check against the currently loaded DataFrame.
            # Wrapped in its own try/except: a parseable-but-wrong-SHAPED
            # file must degrade to a warning, never kill the app at
            # launch (fold V2-M: {"metadata": 5} used to raise out of
            # run()). Everything here validates what _save_autosave
            # writes: fingerprint (exact identity, subsumes columns +
            # bounds + count), then human-readable diffs for the dialog.
            warns = []
            try:
                metadata = autosave_data.get('metadata', {})
                if not isinstance(metadata, dict):
                    metadata = {}
                    warns.append("WARNING: autosave metadata is malformed")

                saved_fp = metadata.get('fingerprint')
                if saved_fp and str(saved_fp) != self._dataset_fingerprint():
                    warns.append("WARNING: dataset fingerprint differs from current data")

                saved_columns = [str(c) for c in (metadata.get('data_columns') or [])]
                current_columns = [str(c) for c in self.df.columns]
                if saved_columns and sorted(saved_columns) != sorted(current_columns):
                    from collections import Counter
                    saved_c = Counter(saved_columns)
                    cur_c = Counter(current_columns)
                    missing = sorted((saved_c - cur_c).keys())
                    extra = sorted((cur_c - saved_c).keys())
                    warns.append("WARNING: saved columns differ from current data")
                    if missing:
                        warns.append(f"  only in autosave: {', '.join(missing)}")
                    if extra:
                        warns.append(f"  only in current:  {', '.join(extra)}")
                    if not missing and not extra:
                        warns.append("  (duplicate column name counts differ)")

                n_rows = metadata.get('n_rows')
                if n_rows is not None and int(n_rows) != len(self.df.index):
                    warns.append(
                        f"WARNING: saved row count ({n_rows}) differs from "
                        f"current data ({len(self.df.index)})")

                tr = metadata.get('time_range')
                if not isinstance(tr, dict):
                    tr = {}
                if tr.get('start') and str(tr['start']) != _norm_iso(self.data_start):
                    warns.append("WARNING: saved time range differs from current data")
                elif tr.get('end') and str(tr['end']) != _norm_iso(self.data_end):
                    warns.append("WARNING: saved time range differs from current data")

                # Same-schema, same-window datasets (two spacecraft via a
                # shared loader) share a fingerprint -- the source name is
                # the tiebreaker when both sides know one (evidence 3b).
                saved_src = metadata.get('source_name')
                live_src = getattr(self, 'source_name', None)
                if saved_src and live_src and str(saved_src) != str(live_src):
                    warns.append(
                        "WARNING: autosave came from a different source file")
                    warns.append(f"  autosave: {saved_src}")
                    warns.append(f"  current:  {live_src}")
            except Exception:
                warns.append("WARNING: autosave metadata is malformed")

            lines = list(warns)
            if candidate is not candidates[0]:
                lines.append("NOTE: loaded from the .bak backup copy "
                             "(the main autosave file is corrupt)")
            autosave_data['_identity'] = {'mismatch': bool(warns), 'lines': lines}
            autosave_data['_loaded_path'] = str(candidate)
            return autosave_data

        # Clean break notice (not a fallback): if a pre-fingerprint
        # autosave sits in this folder, say so once instead of silently
        # ignoring what a returning user may believe is their session.
        legacy = self.autosave_folder / "chronotagger_autosave.json"
        if legacy.exists() and getattr(self, 'status_var', None) is not None:
            self.status_var.set(
                "Note: a pre-2.x autosave (chronotagger_autosave.json) "
                "exists here and is no longer read.")
        return None

    def _apply_recovered_autosave(self, autosave_data: dict) -> None:
        """
        Install a recovered autosave: the intervals plus the label
        schema they were made with. Invalidates the undo history and
        selection, validates in strict mode, and marks the session
        modified (recovered work is unsaved work). GUI refresh and pane
        sync are the caller's job.
        """
        # Pack M1: BUILD, VALIDATE, PUBLISH -- the same discipline
        # _load_session now uses, and the reason it matters more here.
        # This method used to assign self.intervals FIRST and validate
        # LAST with no rollback, so a bad payload left the labeler holding
        # it: measured, raw dicts installed with no exception in normal
        # mode, and in strict mode an AttributeError raised from the
        # validation call AFTER self.intervals had already been replaced.
        raw = autosave_data.get('intervals', []) or []
        new_intervals = [d if isinstance(d, Interval) else Interval.from_dict(d)
                         for d in raw]

        # The track table, or the v1 fallback. The fallback is four lines
        # and it is KEPT on purpose: this is a NAMED ENTRY POINT that the
        # suite calls directly with hand-built payloads, and a
        # future importer will hand it a payload it built itself. A
        # payload with neither 'tracks' nor 'classes' keeps the live
        # schema, exactly as it did before.
        saved_tracks = autosave_data.get('tracks')
        saved_classes = autosave_data.get('classes')
        if saved_tracks:
            # v2 fold F2, the same door on the path a user meets without
            # choosing to.
            new_tracks = validate_track_table(
                [Track.from_dict(t) for t in saved_tracks], "Autosave")
        elif saved_classes:
            new_tracks = default_table(
                list(saved_classes),
                dict(autosave_data.get('class_colors') or {})
                if 'class_colors' in autosave_data else None)
        else:
            new_tracks = [Track.from_dict(t.to_dict())
                          for t in table_of(self)]
        new_active = autosave_data.get('active_track') or new_tracks[0].id

        unknown = stray_tracks(new_intervals, new_tracks)
        if unknown:
            raise ValueError(
                "Autosave has intervals on tracks that are not in its track "
                "table: " + ", ".join(unknown))
        # Pack M3.0: THE SAME MERGE THE LOAD PATH DOES, for the same
        # reason -- a recovered autosave written before the driver grew a
        # lane must not delete it. The stray check above stays keyed to
        # the payload's own table.
        new_tracks, _n_file, _n_kept = merge_track_tables(
            new_tracks, table_of(self))
        new_tracks = validate_track_table(new_tracks, "Autosave")
        self._lane_merge_note = lane_merge_note(_n_file, _n_kept)
        check_interval_invariants(new_intervals, new_tracks)

        if getattr(self, "tracks", None) is None:
            self.tracks = new_tracks
        else:
            self.tracks[:] = new_tracks
        # Pack M3.0: THE FALLBACK THE LOAD PATH ALREADY USES. A saved
        # active lane naming no row fell back to `tracks[0]`, which can be
        # a HIDDEN row -- measured, recovery landed the active lane on a
        # hidden lane, a state the GUI itself refuses to reach.
        # `resolve_active_row` picks the first VISIBLE lane, which is what
        # the painter falls back to as well. The id is cleared first so
        # the resolution cannot latch onto the PREVIOUS session's active
        # lane when the merged table happens to carry a row of that name.
        if find_track(self.tracks, new_active):
            self._active_track_id = new_active
        else:
            from chronotagger.core.tracks import resolve_active_row
            self._active_track_id = None
            self._active_track_id = resolve_active_row(self).id
        self.intervals = new_intervals
        # getattr guards: this method is a named entry point and must
        # not assume the GUI widgets exist yet (fold V2).
        combo = getattr(self, 'class_combo', None)
        var = getattr(self, 'current_class_var', None)
        if combo is not None and var is not None:
            combo["values"] = self.classes
            if var.get() not in self.classes and self.classes:
                var.set(self.classes[0])
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.selected_interval = None
        if hasattr(self, '_clear_selected_interval_highlights'):
            self._clear_selected_interval_highlights()
        self.modified = True

    def _show_recovery_dialog(self, autosave_data):
        """
        Show recovery dialog with autosave information.

        Args:
            autosave_data: Dict containing autosave metadata and intervals

        Returns:
            str: User choice - 'recover', 'start_fresh', 'save_backup', or 'cancel'
        """
        import tkinter as tk
        from tkinter import ttk, messagebox
        from datetime import datetime
        import shutil

        # Identity annotations from _check_autosave (may be absent when
        # the dialog is driven directly, e.g. in tests)
        identity = autosave_data.get('_identity', {}) or {}

        # Create modal dialog. The BUTTONS are structurally protected
        # from overflow (packed bottom-first, see below), so the
        # geometry only decides how much info shows before clipping;
        # vertical resize is the escape hatch. Width covers the row
        # content at 150% DPI scaling (recheck: reqwidth 626 > 620).
        dialog = tk.Toplevel(self.root)
        dialog.title("Autosave Found")
        dialog.geometry("640x700")
        dialog.transient(self.root)
        dialog.grab_set()

        # Pack M3.0: OVER THE MAIN WINDOW, not over the middle of the
        # primary display. The size is passed IN because this box's
        # content is built AFTER this line and its geometry is the fixed
        # 640x700 set three lines up, so asking the window to measure
        # itself here can only ever describe an empty box.
        from chronotagger.labeler.dialogs._placement import center_on_parent
        center_on_parent(dialog, self.root, 640, 700)

        # Horizontally fixed for consistent appearance; vertically
        # resizable as the escape hatch against content overflow
        dialog.resizable(False, True)

        # Store result
        result = {'choice': None}

        # Main container with padding
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill='both', expand=True)

        # Reserve the button area FIRST (packed side='bottom' before
        # any content packs with expand=True): pack priority follows
        # pack order, so overflowing info content can never push the
        # buttons off-screen. The buttons themselves are created later,
        # after their callbacks are defined.
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side='bottom', fill='x', pady=(10, 0))

        top_button_frame = ttk.Frame(button_frame)
        top_button_frame.pack(fill='x', pady=(0, 5))

        bottom_button_frame = ttk.Frame(button_frame)
        bottom_button_frame.pack(fill='x')

        # Header with icon and title
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill='x', pady=(0, 15))

        # Title
        title_label = ttk.Label(
            header_frame,
            text=("Autosave Found -- Identity Mismatch"
                  if identity.get('mismatch')
                  else "Autosave Found for This Data File"),
            font=('Segoe UI', 12, 'bold')
        )
        title_label.pack()

        # Info container with light background
        info_frame = ttk.LabelFrame(main_frame, text="Session Information", padding="15")
        info_frame.pack(fill='both', expand=True, pady=(0, 15))

        metadata = autosave_data.get('metadata', {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        label_stats = flatten_label_stats(
            autosave_data.get('label_stats', {}) or {})


        # Autosave file actually loaded (main or .bak); wraplength so a
        # long absolute path wraps instead of clipping at the fixed width
        loaded_path = autosave_data.get('_loaded_path')
        if loaded_path:
            ttk.Label(
                info_frame,
                text=f"Autosave File: {loaded_path}",
                font=('Segoe UI', 9),
                wraplength=560,
                justify='left'
            ).pack(anchor='w', pady=(0, 5))

        # Source dataset, if the wizard recorded one
        source_name = metadata.get('source_name')
        if source_name:
            ttk.Label(
                info_frame,
                text=f"Source Data: {source_name}",
                font=('Segoe UI', 9),
                wraplength=560,
                justify='left'
            ).pack(anchor='w', pady=(0, 5))

        # Dataset fingerprint (matches the 12-hex in the filename)
        saved_fp = metadata.get('fingerprint')
        if saved_fp:
            ttk.Label(
                info_frame,
                text=f"Dataset ID: {saved_fp}",
                font=('Segoe UI', 9)
            ).pack(anchor='w', pady=(0, 5))

        # Autosave date
        date_label = ttk.Label(
            info_frame,
            text=f"Autosave Date: {metadata.get('autosave_timestamp', 'unknown')}",
            font=('Segoe UI', 9)
        )
        date_label.pack(anchor='w', pady=(0, 5))

        # Saved time range (the strongest identity signal on disk)
        tr = metadata.get('time_range')
        if not isinstance(tr, dict):
            tr = {}
        if tr.get('start') and tr.get('end'):
            ttk.Label(
                info_frame,
                text=f"Time Range: {tr['start']}  to  {tr['end']}",
                font=('Segoe UI', 9),
                wraplength=560,
                justify='left'
            ).pack(anchor='w', pady=(0, 5))

        # Coverage
        coverage_label = ttk.Label(
            info_frame,
            text=f"Coverage: {metadata.get('coverage_percent', '?')}% of time range labeled",
            font=('Segoe UI', 9)
        )
        coverage_label.pack(anchor='w', pady=(0, 10))

        # Identity warnings (fingerprint / column diff / time-range /
        # source mismatch / loaded from .bak) -- rendered, not print()ed
        # to a console nobody sees
        if identity.get('lines'):
            for line in identity['lines']:
                ttk.Label(
                    info_frame,
                    text=line,
                    font=('Segoe UI', 9, 'bold'),
                    foreground='#8b2e2e',
                    wraplength=560,
                    justify='left'
                ).pack(anchor='w')
            if identity.get('mismatch'):
                ttk.Label(
                    info_frame,
                    text="Recovering into a different dataset is NOT recommended.",
                    font=('Segoe UI', 9, 'bold'),
                    foreground='#8b2e2e'
                ).pack(anchor='w', pady=(0, 8))

        # Separator
        ttk.Separator(info_frame, orient='horizontal').pack(fill='x', pady=(0, 10))

        # Intervals by label header
        intervals_header = ttk.Label(
            info_frame,
            text="Intervals by Label:",
            font=('Segoe UI', 9, 'bold')
        )
        intervals_header.pack(anchor='w', pady=(0, 5))

        # Create frame for interval list with padding
        intervals_container = ttk.Frame(info_frame)
        intervals_container.pack(fill='both', expand=True, pady=(0, 10))

        # Display each label's stats
        shown = 0
        for label, stats in sorted(label_stats.items()):
            if not isinstance(stats, dict):
                continue
            if shown >= 8:
                ttk.Label(
                    intervals_container,
                    text=f"  ... and {len(label_stats) - shown} more label(s)",
                    font=('Segoe UI', 9),
                    foreground='#666666'
                ).pack(anchor='w', pady=2)
                break
            shown += 1
            count = stats.get('count', 0)
            hours = stats.get('duration_hours', 0)

            interval_frame = ttk.Frame(intervals_container)
            interval_frame.pack(fill='x', pady=2)

            # Label name (left-aligned)
            label_text = ttk.Label(
                interval_frame,
                text=f"  {label}:",
                font=('Segoe UI', 9),
                width=20,
                anchor='w'
            )
            label_text.pack(side='left')

            # Stats (right side)
            stats_text = ttk.Label(
                interval_frame,
                text=f"{count} intervals ({hours:.1f} hours)",
                font=('Segoe UI', 9),
                foreground='#666666'
            )
            stats_text.pack(side='left')

        # Separator
        ttk.Separator(info_frame, orient='horizontal').pack(fill='x', pady=(5, 10))

        # Total summary (tolerant reads: a missing key must not kill the
        # app at launch)
        total_intervals = metadata.get('total_intervals',
                                       len(autosave_data.get('intervals', [])))
        total_hours = sum(s.get('duration_hours', 0) for s in label_stats.values())

        total_label = ttk.Label(
            info_frame,
            text=f"Total: {total_intervals} intervals covering {total_hours:.1f} hours",
            font=('Segoe UI', 9, 'bold')
        )
        total_label.pack(anchor='w')

        # Button callbacks
        def on_recover():
            result['choice'] = 'recover'
            dialog.destroy()

        def on_start_fresh():
            confirm = messagebox.askyesno(
                "Confirm Start Fresh",
                f"Are you sure you want to start fresh?\n\n"
                f"This will NOT load the autosave with {total_intervals} intervals.\n"
                f"The autosave file will remain and can be recovered later.",
                parent=dialog
            )
            if confirm:
                result['choice'] = 'start_fresh'
                dialog.destroy()

        def on_save_backup():
            # Copy the file that was ACTUALLY loaded (main or .bak), and
            # never overwrite an existing backup from the same second.
            src = autosave_data.get('_loaded_path') or str(self.autosave_file)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.autosave_folder / f'chronotagger_autosave_backup_{timestamp}.json'
            n = 1
            while backup_file.exists():
                backup_file = self.autosave_folder / f'chronotagger_autosave_backup_{timestamp}_{n}.json'
                n += 1

            try:
                shutil.copy(src, backup_file)
                messagebox.showinfo(
                    "Backup Saved",
                    f"Autosave backed up to:\n{backup_file.name}",
                    parent=dialog
                )
                result['choice'] = 'save_backup'
                dialog.destroy()
            except Exception as e:
                messagebox.showerror(
                    "Backup Failed",
                    f"Could not save backup:\n{e}",
                    parent=dialog
                )

        def on_cancel():
            result['choice'] = 'cancel'
            dialog.destroy()

        # Buttons go into the frames reserved at the top of this method
        # (042l): bottom-packed first, so they always stay on screen.

        # Top row buttons
        recover_btn = tk.Button(
            top_button_frame,
            text="Recover Session",
            command=on_recover,
            width=25
        )
        recover_btn.pack(side='left', expand=True, padx=(0, 5))

        fresh_btn = tk.Button(
            top_button_frame,
            text="Start Fresh",
            command=on_start_fresh,
            width=25
        )
        fresh_btn.pack(side='left', expand=True, padx=(5, 0))

        # Bottom row buttons
        backup_btn = tk.Button(
            bottom_button_frame,
            text="Save & Start Fresh",
            command=on_save_backup,
            width=25
        )
        backup_btn.pack(side='left', expand=True, padx=(0, 5))

        cancel_btn = tk.Button(
            bottom_button_frame,
            text="Exit ChronoTagger",
            command=on_cancel,
            width=25
        )
        cancel_btn.pack(side='left', expand=True, padx=(5, 0))

        # Closing the dialog with the window X means cancel (exit), the
        # same as Escape -- never the silent no-branch limbo it was (D3)
        dialog.protocol("WM_DELETE_WINDOW", on_cancel)

        # Default button: Recover -- unless the identity check flagged a
        # mismatch, in which case Start Fresh is the safe default.
        if identity.get('mismatch'):
            fresh_btn.focus_set()
            dialog.bind('<Return>', lambda e: on_start_fresh())
        else:
            recover_btn.focus_set()
            dialog.bind('<Return>', lambda e: on_recover())
        dialog.bind('<Escape>', lambda e: on_cancel())

        # Wait for dialog to close
        dialog.wait_window()

        return result['choice']

    def _on_closing(self) -> None:
        if self.modified:
            resp = messagebox.askyesnocancel("Save Changes?", "Save before closing?")
            if resp is None:
                return
            elif resp:
                # _save_session returns True only if the file was
                # actually written. On cancel/failure, offer the
                # close-anyway choice Q7 ruled -- never close silently
                # on a failed save, never trap the user either.
                if not self._save_session():
                    if not messagebox.askyesno(
                            "Close Anyway?",
                            "The session was not saved. Close anyway and "
                            "discard unsaved changes?"):
                        return
        self.root.destroy()  # type: ignore[union-attr]
    
    def _layouts_compatible(self, layout1: dict, layout2: dict) -> bool:
        """
        Check if two layout_specs are compatible.
        
        Args:
            layout1: First layout specification
            layout2: Second layout specification
            
        Returns:
            True if layouts are structurally compatible, False otherwise
        """
        if layout1 is None or layout2 is None:
            return True  # Can't validate, allow load
        
        # Compare key structural elements
        nrows_match = layout1.get("nrows") == layout2.get("nrows")
        ncols_match = layout1.get("ncols") == layout2.get("ncols")
        areas_match = len(layout1.get("areas", [])) == len(layout2.get("areas", []))
        
        return nrows_match and ncols_match and areas_match
    
    def _describe_layout(self, layout: dict) -> str:
        """
        Create human-readable layout description.
        
        Args:
            layout: Layout specification dictionary
            
        Returns:
            String describing the layout (e.g., "3 rows × 2 columns (4 panels)")
        """
        if layout is None:
            return "Unknown layout"
        
        nrows = layout.get("nrows", "?")
        ncols = layout.get("ncols", "?")
        n_panels = len(layout.get("areas", []))
        
        return f"{nrows} rows × {ncols} columns ({n_panels} panels)"
