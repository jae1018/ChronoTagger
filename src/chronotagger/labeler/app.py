"""
Main application class composed from focused mixins.

This keeps a single public entry point (TimeIntervalLabeler) while letting
implementation details live in small, testable files.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector

# ABSOLUTE imports from core (moved here)
from chronotagger.core.models import Interval
from chronotagger.core.tracks import (
    Track,
    active_id_of,
    active_track_of,
    colors_for,
    default_table,
    find_track,
    track_ids,
)
from chronotagger.core.commands import (
    Command,
    AddIntervalCommand,
    DeleteIntervalCommand,
    RelabelIntervalCommand,
)

# Relative imports to labeler submodules (unchanged)
from .tab_pane import TabPane
from .sync import PaneSyncManager
from .mixins.view_build import ViewBuildMixin
from .mixins.plotting import PlottingMixin
from .mixins.events import EventsMixin
from .mixins.rules import RulesMixin
from .mixins.navigation import NavigationMixin
from .mixins.help import HelpMixin
from .mixins.zoom import ZoomMixin
from .mixins.intervals import IntervalsMixin
from .mixins.stats import StatsMixin
from .mixins.io_export import IOExportMixin
from .mixins.labels import LabelsMixin
from .mixins.lane_controls import LaneControlMixin

logger = logging.getLogger(__name__)


class TimeIntervalLabeler(
    ViewBuildMixin,
    PlottingMixin,
    EventsMixin,
    RulesMixin,
    NavigationMixin,
    HelpMixin,
    ZoomMixin,
    IntervalsMixin,
    LabelsMixin,
    LaneControlMixin,
    StatsMixin,
    IOExportMixin,
):
    
    # Default label colors (Tableau-like)
    DEFAULT_COLORS = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc949", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ac",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        plot_fn: Optional[Callable] = None,
        classes: Optional[List[str]] = None,
        class_colors: Optional[Dict[str, str]] = None,
        window: pd.Timedelta = pd.Timedelta("30min"),
        step: pd.Timedelta = pd.Timedelta("15min"),
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        autosave_folder: str = ".",
        source_name: Optional[str] = None,
        *,
        # Pack M1: the multi-track spelling. KEYWORD-ONLY on purpose --
        # nothing passes it positionally, so inserting it here cannot
        # shift any of the ten positional parameters above it. Mutually
        # exclusive with classes= / class_colors=; see the track-table
        # build further down.
        tracks: Optional[List[Any]] = None,
        layout_spec: Optional[Dict[str, Any]] = None,
        panes: Optional[List[Dict[str, Any]]] = None,
        parent: Optional[tk.Misc] = None,
        decimate: bool = True,
    ) -> None:
        # --- Validate inputs ---
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame must have a DatetimeIndex.")

        # Pack 6 R9. An unsorted index used to be accepted in silence and
        # produce nonsense: data_start / data_end are read blind from
        # index[0] / index[-1], so data_end lands BEFORE data_start, t1 <
        # t0, and df.loc[t0:t1] falls back to a boolean mask that returns
        # the WHOLE dataset -- every "window" contains everything. Measured
        # on a fully shuffled 200-row frame: constructor accepted,
        # _build_gui, _update_plot, _next_window and the redraw after it
        # all succeeded, with no exception anywhere.
        #
        # Sorting rather than raising: the EXPORT path handles
        # non-monotonic frames correctly and deliberately
        # (io_export.py:303-324, Pack 5 R7), so raising here would reject
        # data one half of the package gets right. Sorting is what a caller
        # means. Loud, because silently reordering someone's frame is its
        # own surprise -- and once per construction, not per frame, so a
        # plain warning is the right level (Pack 4 doctrine).
        #
        # The sentence this comment used to open with -- "the quick-start
        # wizard already refuses such a frame (file_loader.py:457)" -- was
        # stale twice over: the refusal had moved to file_loader.py:473,
        # and Pack 8 R14 REMOVED it. All three paths now sort. This
        # constructor warns, the wizard loader says so on its status line,
        # and an emitted driver sorts unconditionally.
        #
        # IDENTITY CONSEQUENCE, and the reason the warning below carries a
        # second sentence. dataset_fingerprint() (io_export.py:38) hashes
        # index[0] and index[-1]; on an unsorted frame those are arbitrary
        # rows, and after the sort they are the true bounds. So the Pack-2
        # autosave FILENAME moves -- measured A/B on one shuffled 200-row
        # frame, chronotagger_autosave_b1179b6c4918.json before and
        # chronotagger_autosave_fb09764d0822.json after -- and Pack 2 made
        # the fingerprinted name the only name _check_autosave consults.
        # An autosave written from the unsorted frame is therefore not
        # offered for recovery after this change. Correct on the merits --
        # the old fingerprint encoded nonsense bounds and those sessions
        # were already windowing over the whole dataset -- but the user has
        # to be TOLD, not left to discover it.
        #
        # Scope: monotonicity only. Duplicate and NaT index entries are NOT
        # in this pack.
        if not df.index.is_monotonic_increasing:
            logger.warning(
                "DataFrame index is not monotonically increasing (%d rows); "
                "sorting by index. Windowing, navigation and per-sample "
                "export all assume ascending time. This also changes the "
                "dataset fingerprint, so an autosave written from the "
                "unsorted frame will not be offered for recovery.",
                len(df.index))
            df = df.sort_index()

        # Validate panes vs plot_fn usage
        if panes is not None and plot_fn is not None:
            raise ValueError(
                "Cannot specify both 'panes' and 'plot_fn'. "
                "Use 'panes' for multi-pane mode or 'plot_fn' for single-pane mode."
            )
        if panes is None and plot_fn is None:
            raise ValueError(
                "Must specify either 'panes' (for multi-pane mode) or 'plot_fn' (for single-pane mode)."
            )

        # Determine mode and convert to unified internal representation
        if panes is not None:
            # Multi-pane mode
            self.multi_pane_mode = True
            self._panes_config: List[Dict[str, Any]] = panes
            # Validate that each pane has a plot_fn
            for i, pane in enumerate(self._panes_config):
                if "plot_fn" not in pane or not callable(pane["plot_fn"]):
                    raise ValueError(f"Pane at index {i} must have a callable 'plot_fn'.")
        else:
            # Single-pane mode (backward compatibility)
            if not callable(plot_fn):
                raise TypeError("plot_fn must be callable.")
            self.multi_pane_mode = False
            self._panes_config: List[Dict[str, Any]] = [
                {
                    "title": "Main",
                    "plot_fn": plot_fn,
                    "layout_spec": layout_spec,
                }
            ]

        # Convert config dicts to TabPane objects
        self.panes: List[TabPane] = [
            TabPane(
                title=config["title"],
                plot_fn=config["plot_fn"],
                layout_spec=config.get("layout_spec"),
            )
            for config in self._panes_config
        ]

        # Track active pane
        self.active_pane_idx: int = 0

        # Create sync manager for multi-pane coordination
        self.sync_manager = PaneSyncManager(self)

        # Core data / plotting contract
        self.df = df
        # Note: plot_fn and layout_spec are now properties that delegate to active_pane

        # Draw-only decimation (Pack 5 R4b/R11). ON by default: a window
        # holding more samples than the panel has pixels is drawn from an
        # envelope of ORIGINAL rows -- per pixel column, per numeric
        # column, the argmin and argmax rows -- so single-sample spikes
        # survive and nothing is averaged or synthesised. Selection,
        # rules, labeling and export always read the full-resolution
        # frame. decimate=False draws every sample; see the README's
        # known-limitations note for what that buys and costs.
        self.decimate: bool = bool(decimate)

        # Label TRACKS (Pack M1). classes= / class_colors= stay as the
        # SINGLE-TRACK spelling and build a one-row table; tracks= is the
        # multi-track spelling. Passing both is a ValueError, in exactly
        # the shape this constructor already uses for panes vs plot_fn
        # sixty lines above. This is the one place Pack M1 does NOT take
        # a clean break, and the reason is that a driver is a file the
        # user owns: 47 files on disk pass classes=CLASSES and 151
        # construct a labeler outside src/ and tests/. classes= is not a
        # compatibility layer, it is the one-lane name for the same
        # thing.
        if tracks is not None and (classes is not None
                                   or class_colors is not None):
            raise ValueError(
                "Cannot specify both 'tracks' and 'classes'/'class_colors'. "
                "Use 'tracks' for the multi-track table, or 'classes' "
                "(optionally with 'class_colors') for a single default track."
            )
        if tracks is not None:
            self.tracks: List[Track] = self._build_track_table(tracks)
        else:
            if classes is None:
                classes = ["UNKNOWN", "label_1", "label_2"]
            self.tracks = default_table(
                list(classes),
                dict(class_colors) if class_colors is not None else None,
                palette=self.DEFAULT_COLORS,
            )
        # The ACTIVE track. View state -- but it persists, so it lives at
        # the session's top level as "active_track" rather than on a row.
        # Pack M1 ships no control that changes it, so it is the first
        # row for every session the user opens; self.classes and
        # self.class_colors are properties over it (see below).
        self._active_track_id: str = self.tracks[0].id

        # Time bounds & window
        self.data_start: pd.Timestamp = df.index[0]
        self.data_end: pd.Timestamp = df.index[-1]
        self.window: pd.Timedelta = window
        self.step: pd.Timedelta = step

        if start is None:
            start = self.data_start
        self.t0: pd.Timestamp = max(start, self.data_start)
        self.t1: pd.Timestamp = min(self.t0 + window, self.data_end)

        # Wheel zoom/pan config
        self.zoom_sensitivity: float = 0.20
        self.pan_sensitivity: float = 0.20
        self.min_window: pd.Timedelta = pd.Timedelta("5s")
        self.max_window: pd.Timedelta = self.data_end - self.data_start

        # Intervals & selection
        self.intervals: List[Interval] = []
        self.selected_interval: Optional[Interval] = None
        self.current_selection: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None

        # Undo/redo
        self.undo_stack: List[Command] = []
        self.redo_stack: List[Command] = []
        self.max_undo: int = 50  # gesture-level entries; each holds two list snapshots

        # Persistence - Autosave configuration. The filename carries a
        # dataset fingerprint (sorted columns + time bounds + row
        # count), so differently-shaped datasets sharing one folder or
        # CWD do not collide at recovery. Identical-schema datasets over
        # the same window (two spacecraft via one loader) DO share a
        # name -- the source_name check in _check_autosave is the guard
        # for that case. Clean break (Pack 2 grill Q2): pre-fingerprint
        # autosave files are ignored. The fingerprint is fixed at
        # construction; do not add/rename df columns on a live labeler.
        self.source_name: Optional[str] = source_name
        self.autosave_folder = Path(autosave_folder)
        self.autosave_folder.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist
        # Forensic log lives BESIDE the autosave: it inherits the writable-
        # dir guarantee, the suite's CWD isolation, and the examples'
        # gitignored autosave dirs for free (Pack 4 R2/R3). Idempotent.
        from .._logging import configure_file_logging
        self._log_path = configure_file_logging(self.autosave_folder)
        self.autosave_file = (
            self.autosave_folder
            / f"chronotagger_autosave_{self._dataset_fingerprint()}.json"
        )
        # Pack M2.7: THE RECOVERY QUESTION HAS NOT BEEN ASKED YET.
        # add_track_from_column ends in an unconditional _save_autosave(),
        # and a driver that ingests a model column BEFORE run() therefore
        # wrote an agent-only autosave over the previous session's file
        # before the user was ever offered it. Measured end to end
        # (probe_s4_recovery_clobber Q1-Q3): session one left 259
        # intervals on disk, relaunch one rewrote the main file to 256
        # agent-only and demoted the good file to .bak -- which
        # _check_autosave reads ONLY when main is unreadable -- and after
        # relaunch two the hand labels were on no file in the folder.
        # run() sets this True the moment the recovery question is
        # settled; until then the INGEST does not write. Nothing else is
        # gated: the ten other _save_autosave call sites are GUI gesture
        # handlers, reachable only once the window is up.
        self._recovery_resolved: bool = False
        self.modified: bool = False

        # GUI state.  When `parent` is provided (e.g. the quick-start wizard),
        # the labeler will mount itself as a tk.Toplevel under that parent so
        # only one tk.Tk root exists per process.  When parent is None, the
        # labeler creates its own tk.Tk root (standalone use).
        self._parent: Optional[tk.Misc] = parent
        self.root: Optional[tk.Misc] = None
        self.notebook: Optional[ttk.Notebook] = None  # Only created if multi_pane_mode

        # Layout & axes metadata.  layout_spec supports a "role" field for each
        # axis area:
        #   - "time":     Time-series data with time on x-axis (box-select
        #                 uses time coords).
        #   - "not-time": Non-time plots like position/phase space (box-select
        #                 maps point order to time).
        # axes_meta mirrors active_pane.axes_meta for legacy access; figures
        # / canvases / per-axis zoom state live on the TabPane and are
        # accessed via the @property delegates below.
        self.axes_meta: Dict[str, Dict[str, Any]] = {}            # key -> {role, row, col, ...}
        self._time_axis_keys: List[str] = []                       # which keys are "time"
        self._primary_time_key: Optional[str] = None               # first time axis in col 0

        # Matplotlib connections
        self.rect_selectors: Dict[str, RectangleSelector] = {}
        self.pick_cid: Optional[int] = None
        self._scroll_cid: Optional[int] = None

        # Drag/resize/move state
        self._drag_mode: Optional[str] = None
        self._drag_iv: Optional[Interval] = None
        self._drag_initial: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
        self._drag_offset: Optional[pd.Timedelta] = None
        self._drag_preview: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
        self._press_cid: Optional[int] = None
        self._motion_cid: Optional[int] = None
        self._release_cid: Optional[int] = None

        # Two-click time selection
        self.two_click_mode: bool = True         # default on; disable to use drag-selector
        self.two_click_auto_add: bool = False    # if True, auto-creates interval on 2nd click

        # Point highlighting (performance optimization)
        self.enable_point_highlighting: bool = True  # disable for large datasets
        for pane in self.panes:
            pane.enable_point_highlighting = self.enable_point_highlighting

        self._pick_anchor_ts: Optional[pd.Timestamp] = None  # first click time
        self._twoclick_motion_cid: Optional[int] = None      # (legacy) preview wire-up
        self._time_click_cid: Optional[int] = None           # mpl connection for clicks
        self._time_motion_cid: Optional[int] = None          # mpl connection for motion

        # Minimal interval duration inference
        try:
            diffs = self.df.index.to_series().diff().dropna()
            med = diffs.median()
            if not isinstance(med, pd.Timedelta) or med <= pd.Timedelta(0):
                med = pd.Timedelta(seconds=1)
            self.min_duration: pd.Timedelta = med
        except Exception:
            # Gates interval validation; a silent 1 s changes the
            # minimum-length rule's meaning (Pack 4 A9).
            # (Pack 6: same logger, same name -- EDIT 185 gave the module
            # one, so the function-local import is redundant.)
            logger.warning(
                "min_duration inference failed; defaulting to 1 s",
                exc_info=True)
            self.min_duration = pd.Timedelta(seconds=1)

        # Multi-span preview state from box-select / rule preview
        self.current_spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        self._commit_spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

        # Click-vs-drag arbitration (pixel slop)
        self.CLICK_DRAG_SLOP_PX: int = 6
        self._press_xy_px: Optional[Tuple[int, int]] = None
        self._dragging_box: bool = False

        # Label-rule policy
        self._overlap_policy: str = "skip"

        # Per-axis zoom state for Y-zoom and cross-plot X/Y zoom.
        # Keyed by matplotlib Axes so they survive plot rebuilds.
        # (Pack 6 R4: a `self._manual_zooms` lived here too, recording
        # which axes the user had zoomed and in which direction across 13
        # sites. Every read in the tree was `if ax not in self._manual_
        # zooms:` immediately before a write; nothing iterated it, tested
        # membership to make a decision, or read the set contents. The two
        # reset paths that look like consumers -- plotting.py:496 and
        # zoom.py:226 -- iterate _auto_ylims / _auto_xlims and reset
        # EVERYTHING, then clear a record they never consulted. Selective
        # reset would be a feature; this was not scaffolding for one.)
        self._auto_xlims: Dict[plt.Axes, Tuple[float, float]] = {}  # Auto X limits for cross-plots
        self._auto_ylims: Dict[plt.Axes, Tuple[float, float]] = {}  # Auto Y limits for all plots
        self._time_range_dirty: bool = False                        # Flag: time range changed

    # -------- Pane configuration properties --------

    @property
    def active_pane_config(self) -> Dict[str, Any]:
        """Return the configuration dictionary for the currently active pane."""
        return self._panes_config[self.active_pane_idx]

    @property
    def active_pane(self) -> TabPane:
        """Get the currently active TabPane object."""
        return self.panes[self.active_pane_idx]

    # -------- Backward compatibility delegation properties --------
    # These delegate to the active pane so existing code continues to work

    @property
    def fig(self) -> Optional[plt.Figure]:
        """Delegate to active pane for backward compatibility."""
        return self.active_pane.fig

    @fig.setter
    def fig(self, value: Optional[plt.Figure]) -> None:
        """Set fig on active pane."""
        self.active_pane.fig = value

    @property
    def canvas(self) -> Optional[FigureCanvasTkAgg]:
        """Delegate to active pane for backward compatibility."""
        return self.active_pane.canvas

    @canvas.setter
    def canvas(self, value: Optional[FigureCanvasTkAgg]) -> None:
        """Set canvas on active pane."""
        self.active_pane.canvas = value

    @property
    def user_axes(self) -> Dict[str, plt.Axes]:
        """Delegate to active pane for backward compatibility."""
        return self.active_pane.user_axes

    @user_axes.setter
    def user_axes(self, value: Dict[str, plt.Axes]) -> None:
        """Set user_axes on active pane."""
        self.active_pane.user_axes = value

    @property
    def strip_ax(self) -> Optional[plt.Axes]:
        """Delegate to active pane for backward compatibility."""
        return self.active_pane.strip_ax

    @strip_ax.setter
    def strip_ax(self, value: Optional[plt.Axes]) -> None:
        """Set strip_ax on active pane."""
        self.active_pane.strip_ax = value

    @property
    def plot_fn(self) -> Callable:
        """Delegate to active pane for backward compatibility."""
        return self.active_pane.plot_fn

    @property
    def layout_spec(self) -> Optional[Dict[str, Any]]:
        """Delegate to active pane for backward compatibility."""
        return self.active_pane.layout_spec

    # -------- Pack M1: the track table --------

    def _build_track_table(self, tracks) -> List[Track]:
        """Normalize a caller's tracks= argument into a Track list.

        Accepts Track objects and plain dicts in any mixture, because a
        driver's TRACKS block is a dict literal a human edits.

        Three refusals, each of them something that would otherwise fail
        much later and much more quietly:
          * an empty table -- a session with no lane cannot label;
          * a duplicate id -- an id names an export column and a sidecar
            file, so two rows sharing one id would write one column and
            lose the other in silence. This is also the whole
            enforcement of "the id 'default' is reserved": the default
            table already holds that row, so a second one collides;
          * a track with no classes -- every lane needs its own
            vocabulary, and an empty one exports a column of -1.
        The id charset itself is enforced in Track.__post_init__.

        `order` is NORMALISED to the list position, so "the order I
        wrote them in the driver is the lane order" is true without the
        caller keeping two things in sync. A caller-supplied `order` is
        therefore overridden; that is a drafter ruling and it is stated
        as one in the pack.
        """
        rows = [t if isinstance(t, Track) else Track.from_dict(t)
                for t in (tracks or [])]
        if not rows:
            raise ValueError(
                "'tracks' must contain at least one track; pass "
                "classes=[...] for the single-track default instead.")
        seen = set()
        for i, row in enumerate(rows):
            if row.id in seen:
                raise ValueError(
                    "duplicate track id %r: a track id names an export "
                    "column and a label-map sidecar, so it must be "
                    "unique" % (row.id,))
            seen.add(row.id)
            row.order = i
            if not row.classes:
                raise ValueError(
                    "track %r has no classes; every track needs its own "
                    "vocabulary" % (row.id,))
            if not row.class_colors:
                row.class_colors = colors_for(row.classes,
                                              self.DEFAULT_COLORS)
        return rows

    @property
    def active_track(self) -> Track:
        """The row every gesture writes to and the strip paints.

        Falls back to the first row rather than raising when the active
        id is stale, because a stale id is a view-state bug and must not
        take the window down.
        """
        return active_track_of(self)

    @property
    def active_track_id(self) -> str:
        """The active track's id."""
        return active_id_of(self)

    def track_by_id(self, track_id) -> Optional[Track]:
        """The row with this id, or None."""
        return find_track(self.tracks, track_id)

    @property
    def track_ids(self) -> List[str]:
        """Every track id, in lane order."""
        return track_ids(self.tracks)

    @property
    def classes(self) -> List[str]:
        """The ACTIVE track's vocabulary.

        Pack M1 turned this from a plain attribute into a view over one
        row of the track table. With one default track it is the same
        list it always was, which is why 22 of the 44 code lines that
        read self.classes needed no edit at all. The setter writes
        through to the active row, so Manage Labels..., a session load
        and an autosave recovery all keep working unchanged.
        """
        return self.active_track.classes

    @classes.setter
    def classes(self, value) -> None:
        self.active_track.classes = [str(c) for c in value]

    @property
    def class_colors(self) -> Dict[str, str]:
        """The ACTIVE track's colour map. See `classes`."""
        return self.active_track.class_colors

    @class_colors.setter
    def class_colors(self, value) -> None:
        self.active_track.class_colors = dict(value)

    # -------- Pack M1: column -> locked track ingest (R5) --------

    def add_track_from_column(self, column, track_id, name=None,
                              gap_tolerance=None, locked=True,
                              class_order=None):
        """Turn a per-sample label COLUMN into a label TRACK.

        This is the door the user's own workflow has been waiting for: his
        driver hand-builds a model-label lane as a fake one-row
        `pcolormesh` in all four of his pane layouts, because the tool has
        one strip. That lane costs him a gridspec row per pane and cannot
        be clicked, edited, undone, selected, saved or exported. This makes
        it a real track.

        column        a column NAME in self.df, or a pd.Series aligned to
                      self.df.index.
        track_id      the new (or existing) track's opaque id, validated
                      to [A-Za-z0-9_-]{1,32}.
        name          the display name; defaults to the id.
        gap_tolerance anything pd.Timedelta accepts, or None for the
                      default `3 x p95(dt)` of the frame's own index. See
                      core/ingest.py for why `k x median` is the wrong
                      statistic on this user's bimodal ARTEMIS cadence --
                      it inflates the interval count 14.7x.
        locked        the new track is locked BY CONSTRUCTION. "The tool
                      must not let you edit this, it came from a file" is a
                      fact about provenance, not about the current window,
                      so it is MODEL state and it persists.
        class_order   the vocabulary in the order the ids should take.
                      When this ingest CREATES the track, an absent
                      class_order means the column's distinct values,
                      sorted. When it ingests into an EXISTING track, an
                      unknown label is REFUSED -- there the class set is a
                      declared schema, and a silent auto-add would make
                      Manage Labels, the strip legend and the sidebar tags
                      drift without a gesture.

        Returns the Track.

        IT RUNS INSIDE A GESTURE, so one Ctrl+Z removes the track AND its
        intervals. That is the whole reason the track table rides in the
        gesture snapshot: an ingest adds no interval to an existing lane,
        so an intervals-only no-op test would push no undo entry at all.

        IT NEVER WRITES A COLUMN INTO self.df. Measured: adding one column
        moves dataset_fingerprint from d5dce8d8910a to 7520c0a76b11, which
        changes the autosave FILENAME, which makes every prior autosave for
        that dataset unreachable and pops the identity-mismatch dialog.
        That is a rule, not a note.
        """
        from chronotagger.core.ingest import intervals_from_column

        if isinstance(column, str):
            if self.df is None or column not in self.df.columns:
                raise ValueError(
                    "no column %r in the frame; pass a column name that "
                    "exists, or a Series aligned to df.index" % (column,))
            values = self.df[column]
            source = "column:%s" % column
        else:
            values = column
            if len(values) != len(self.df.index):
                raise ValueError(
                    "the series has %d rows and the frame has %d; an "
                    "ingested column must be aligned to df.index"
                    % (len(values), len(self.df.index)))
            source = "column:%s" % (getattr(column, "name", None) or "series")

        existing = self.track_by_id(track_id)
        if existing is None:
            # Validate the id before anything else touches the model: the
            # id becomes an export column name and a sidecar filename.
            Track(id=track_id)

        ivs, info = intervals_from_column(
            self.df.index, values, track_id,
            gap_tolerance=gap_tolerance,
            source=source,
            class_order=class_order,
            end_after_inclusive=self._end_after_inclusive,
            data_end=self.data_end,
        )

        with self._gesture("import track %s from %s" % (track_id, source)):
            if existing is None:
                classes = list(info["classes"])
                row = Track(
                    id=track_id,
                    name=name or track_id,
                    classes=classes,
                    class_colors=colors_for(classes, self.DEFAULT_COLORS),
                    locked=bool(locked),
                    order=len(self.tracks),
                )
                self.tracks.append(row)
            else:
                unknown = sorted(set(info["present"])
                                 - set(existing.classes))
                if unknown:
                    raise ValueError(
                        "track %r already exists and its class set does "
                        "not contain %s; ingesting into an EXISTING track "
                        "must not silently change its schema. Pass a new "
                        "track_id, or add the class through Manage "
                        "Labels first." % (track_id, ", ".join(unknown)))
                row = existing
                if locked:
                    row.locked = True
            self.intervals.extend(ivs)
            self._sort_and_merge_intervals()

        if getattr(self, 'status_var', None) is not None:
            self.status_var.set(
                "Imported %d interval(s) into track '%s' from %s "
                "(gap tolerance %s, %d run(s), %d split(s), %d unlabeled "
                "sample(s) skipped)"
                % (len(ivs), track_id, source, info["gap_tolerance"],
                   info["runs"], info["splits"], info["unlabeled_rows"]))
        if getattr(self, 'canvas', None) is not None:
            self._update_plot()
        # Pack M2.7: THE LAUNCH-TIME INGEST MUST NOT OVERWRITE THE
        # RECOVERABLE AUTOSAVE. This call sits OUTSIDE the `with
        # self._gesture(...)` block above, so holding it back leaves the
        # ingest fully undoable and the in-memory state exactly as it was
        # -- only the disk write waits. It waits for run() to settle the
        # recovery question, because a driver that ingests before
        # app.run() (test_drivers/drive_multilabel_thb.py:173, ahead of
        # app.run() at :201) otherwise writes a model-only autosave over
        # the very file the recovery dialog is about to offer.
        if getattr(self, "_recovery_resolved", False):
            self._save_autosave()
        return row

    # -------- Public entrypoint --------

    def run(self) -> None:
        """Start the Tkinter main loop with autosave recovery."""
        # Build GUI first
        self._build_gui()

        # Check for autosave BEFORE starting mainloop
        autosave_data = self._check_autosave()

        # Pack M2.7: NOTHING TO OFFER IS ALSO AN ANSWER. _check_autosave
        # returns None when no autosave exists, when the only candidate
        # was unreadable, and when a FUTURE-VERSION file was refused with
        # "It has NOT been loaded. Nothing in this session changed." In
        # all three the user has been told everything he is going to be
        # told, so the ingest may write from here on.
        if autosave_data is None:
            self._recovery_resolved = True

        if autosave_data is not None:
            choice = self._show_recovery_dialog(autosave_data)

            if choice == 'recover':
                # Restore intervals + the label schema they were made
                # with; invalidate undo history/selection; validate in
                # strict mode; mark modified. The testable core lives in
                # IOExportMixin._apply_recovered_autosave (Pack 2 D1).
                #
                # Pack M2.6: A REFUSED RECOVERY IS A MESSAGE, NOT A
                # TRACEBACK. _apply_recovered_autosave RAISES ValueError
                # when the payload carries an interval on a track its own
                # saved table does not hold, and nothing caught it -- so
                # the window never opened and the user got a traceback in
                # the terminal. That payload is exactly what the stale-id
                # fail-open above used to write, and one driver autosave
                # on this machine still holds one. The root fix stops NEW
                # ones being written; this makes an OLD one survivable:
                # say why, then carry on exactly as "start fresh" does,
                # leaving the file untouched for a later look.
                try:
                    self._apply_recovered_autosave(autosave_data)
                except ValueError as exc:
                    from tkinter import messagebox
                    messagebox.showerror(
                        "Recovery Failed",
                        "This autosave could not be recovered:\n\n%s\n\n"
                        "Starting a fresh session instead. The autosave "
                        "file has been left where it is." % (exc,))
                    self.status_var.set(
                        "Recovery failed -- starting fresh session "
                        "(autosave not loaded)")
                else:
                    # Sync intervals across all panes
                    self.sync_manager.sync_intervals_changed()
                    # Refresh UI to show loaded intervals
                    self._update_plot()
                    if hasattr(self, '_update_intervals_list'):
                        self._update_intervals_list()
                    # Pack M3.0: and it says whether the merge kept a lane
                    # the autosave had never heard of.
                    self.status_var.set(
                        "Recovered %d intervals from autosave%s"
                        % (len(self.intervals),
                           getattr(self, "_lane_merge_note", "") or ""))

            elif choice == 'start_fresh':
                # Don't load autosave, keep empty intervals
                # Autosave file remains for potential future recovery
                self.status_var.set("Starting fresh session (autosave not loaded)")

            elif choice == 'save_backup':
                # Already handled in dialog callback
                self.status_var.set("Starting fresh session (backup saved)")

            elif choice == 'cancel':
                # User wants to exit
                self.root.destroy()
                return  # Exit without starting mainloop

            # Pack M2.7: EVERY BRANCH THAT REACHES THIS LINE KEEPS THE
            # WINDOW -- recover (whether it loaded or showed "Recovery
            # Failed"), start fresh, and save-backup-then-start-fresh. The
            # recovery question has been answered, so the ingest may write
            # again. 'cancel' returned above and never gets here, which is
            # right: there is no session left to autosave.
            self._recovery_resolved = True

        # Update plot and start GUI event loop.
        # When the labeler is a child Toplevel (launched from the wizard),
        # mainloop() is already running on the parent's Tk.  We block on the
        # Toplevel via wait_window() instead so labeler.run() returns when
        # the user closes the labeler, but the parent's mainloop keeps going.
        self._update_plot()
        if isinstance(self.root, tk.Tk):
            self.root.mainloop()  # type: ignore[union-attr]
        else:
            self.root.wait_window()  # type: ignore[union-attr]
