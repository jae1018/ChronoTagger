# src/chronotagger/labeler/mixins/plotting.py
"""
Plotting & axis formatting mixin.

Responsibilities:
- Redraw user panels and strip
- Keep all x-axes aligned and formatted as dates
"""

from __future__ import annotations

import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

from contextlib import contextmanager
from pathlib import Path

import logging

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from ..utils.timeaxis import apply_time_axis_format
from ..utils.overlays import draw_interval_bands
from ..utils.decimate import plan_decimation
from ..utils.spectrogram import refresh_colorbars, take_layout_dirty

logger = logging.getLogger(__name__)

# Layout re-solves are supposed to be rare (a window resize, the F9
# sidebar). If they stop being rare the freeze is buying nothing, which is
# a degrade worth exactly one record per session (Pack 4 R14).
_LAYOUT_INVALIDATIONS = 0
_LAYOUT_STORM_WARNED = False
_LAYOUT_STORM_AT = 500


def _note_layout_invalidation() -> None:
    global _LAYOUT_INVALIDATIONS, _LAYOUT_STORM_WARNED
    _LAYOUT_INVALIDATIONS += 1
    if _LAYOUT_INVALIDATIONS > _LAYOUT_STORM_AT and not _LAYOUT_STORM_WARNED:
        _LAYOUT_STORM_WARNED = True
        logger.warning(
            "constrained-layout re-solve requested %d times this session; "
            "the layout freeze is not holding", _LAYOUT_INVALIDATIONS)


class PlottingMixin:
    # Expects on self: fig, canvas, user_axes, strip_ax, class_colors, etc.

    @contextmanager
    def _squelch_xlim_events(self):
        prev = getattr(self, "_squelch_xlim", False)
        self._squelch_xlim = True
        try:
            yield
        finally:
            self._squelch_xlim = prev

    def _apply_time_axis_format(self, ax) -> None:
        apply_time_axis_format(ax)

    def _overlays_enabled(self) -> bool:
        """Safe check for overlay toggle; default True if toggle not built yet."""
        try:
            return bool(self.overlays_var.get())  # created in view_build Options
        except Exception:
            return True

    def _warn_once(self, key: str, msg: str) -> None:
        """
        WARNING with traceback, once per session per site: these sites
        fire on every redraw, and a repeating fallback is one fact, not
        a log flood (Pack 4 level doctrine R14).
        """
        warned = getattr(self, "_redraw_warnings", None)
        if warned is None:
            warned = set()
            self._redraw_warnings = warned
        if key in warned:
            return
        warned.add(key)
        logger.warning(msg, exc_info=True)

    def _plot_error_text(self, exc: BaseException) -> str:
        """
        Three-line on-axes summary (Pack 4 R6): exception type + message,
        the user's own file:line (last traceback frame OUTSIDE this
        package -- their code is the frame that matters), and the log
        pointer. The full traceback goes to the log, not the axes.
        """
        import traceback
        lines = [f"Plot error: {type(exc).__name__}: {exc}"]
        try:
            pkg_root = str(Path(__file__).resolve().parents[2])
            # FORWARD walk, first frame outside the package: that is the
            # user's own plot_fn frame. A reversed walk returns the
            # DEEPEST foreign frame -- pandas internals (verifier B3:
            # `base.py, line 3819, in get_loc` instead of the user).
            for frame in traceback.extract_tb(exc.__traceback__):
                fname = frame.filename
                try:
                    fname = str(Path(fname).resolve())
                except Exception:
                    pass
                if not fname.startswith(pkg_root):
                    lines.append(
                        f"  {Path(frame.filename).name}, line "
                        f"{frame.lineno}, in {frame.name}")
                    break
        except Exception:
            pass
        lines.append("  full traceback -> chronotagger.log")
        return "\n".join(lines)

    # ---------- Pack 5: draw-only decimation (R4b / R11) ----------

    def _decimation_enabled(self) -> bool:
        """
        Whether draw decimation may run for the CURRENT pane.

        Off when the user asked for raw drawing (``decimate=False``), and
        off in the two shapes where selecting rows would break something
        real instead of speeding it up:

        - ``df.attrs`` carries companion arrays -- a spectrogram's energy
          table, for one -- that plot_fn indexes against ``df.index``.
          ``_build_window_attrs_view`` slices those to the FULL window, so
          handing plot_fn a shorter frame beside a full-length array is a
          shape error, not a speedup. That is also the case decimation
          cannot help anyway: a mesh is not a line (README, known limits).
        - a ``not-time`` panel is present. Min/max per time bin is a
          line-envelope transform; on an X-Y cross plot it means nothing,
          and the highlight extractor for those panels indexes the full
          window frame positionally.
        """
        if not getattr(self, "decimate", True):
            return False
        if getattr(self, "_decim_suspend", False):
            # A box selection is reading the artists; see
            # _on_rectangle_select. Nothing that decides which samples get
            # LABELLED may ever look at a decimated trace.
            return False
        try:
            if getattr(self.df, "attrs", None):
                return False
        except Exception:
            return False
        meta = getattr(self, "axes_meta", None)
        if isinstance(meta, dict):
            for m in meta.values():
                try:
                    if str(m.get("role", "time")).lower() != "time":
                        return False
                except Exception:
                    return False
        return True

    def _draw_width_px(self) -> int:
        """Pixel width of a data panel -- the bin count decimation aims at."""
        axes = getattr(self, "user_axes", None) or {}
        for ax in axes.values():
            try:
                w = int(ax.bbox.width)
            except Exception:
                continue
            if w > 1:
                return w
        try:
            return int(self.fig.get_size_inches()[0] * self.fig.dpi)
        except Exception:
            return 0

    def _decimate_for_draw(self, sub_df):
        """
        Return the frame plot_fn should draw, and record whether it was
        decimated (the selection path asks).

        sub_df comes back untouched whenever decimation is off or a no-op:
        there is no half-decimated state.
        """
        self._decim_active = False

        if not self._decimation_enabled():
            return sub_df

        n_px = self._draw_width_px()
        if n_px <= 1:
            return sub_df

        try:
            plan = plan_decimation(sub_df, n_px)
        except Exception:
            # A rendering optimisation may never take the redraw down.
            self._warn_once(
                "decimation-failed",
                "envelope decimation failed; drawing the full window")
            return sub_df
        if plan is None:
            return sub_df

        kept = plan[0]
        self._decim_active = True
        return sub_df.take(kept)

    # ---------- Pack 5: constrained-layout freeze (R4a) ----------

    def _freeze_layout_after_draw(self, pane=None) -> None:
        """
        Turn the constrained-layout solver off once it has produced a
        geometry.

        The solver runs on EVERY canvas.draw() and its cost is flat, not
        per-point: an empty figure measures 122.5 ms with
        constrained_layout against 46.3 ms frozen, and a 43k-point frame
        332 ms against 210 ms. Freezing installs matplotlib's
        PlaceHolderLayoutEngine, which preserves the solved geometry
        exactly (verified), and re-solving is one call away.
        """
        if pane is None:
            pane = getattr(self, "active_pane", None)
        fig = getattr(pane, "fig", None) if pane is not None else None
        if fig is None or getattr(pane, "_layout_frozen", False):
            return
        if not getattr(pane, "_layout_constrained", False):
            pane._layout_frozen = True
            return
        try:
            fig.set_layout_engine("none")
        except Exception:
            self._warn_once("layout-freeze",
                            "could not freeze the constrained-layout engine")
            return
        pane._layout_frozen = True

    def _invalidate_layout_freeze(self, pane=None) -> None:
        """
        Ask for ONE more constrained-layout solve.

        A genuine layout change -- a figure resize (window resize, the F9
        sidebar toggle, a DPI change) or a rebuilt view -- must re-solve or
        the panels keep the geometry of the old window. Everything else
        keeps the frozen geometry, which is the entire point.
        """
        if pane is None:
            pane = getattr(self, "active_pane", None)
        if pane is None:
            return
        fig = getattr(pane, "fig", None)
        if fig is None or not getattr(pane, "_layout_constrained", False):
            return
        if not getattr(pane, "_layout_frozen", False):
            return
        try:
            fig.set_layout_engine("constrained")
        except Exception:
            self._warn_once("layout-resolve",
                            "could not restore the constrained-layout engine")
            return
        pane._layout_frozen = False
        _note_layout_invalidation()

    def _update_plot(self) -> None:
        """Redraw user panels and strip, preserving two-click preview overlays."""
        import pandas as pd

        # Clear data panels (but keep colorbars/inset axes if user created separately)
        with self._squelch_xlim_events():
            for ax in self.user_axes.values():
                ax.clear()
                
        # Identify all time-role axes and force them into date mode *before* plotting.
        if getattr(self, "_time_axis_keys", None):
            for k in self._time_axis_keys:
                ax = self.user_axes.get(k)
                if ax is None:
                    continue
                ax.xaxis_date()                 # tell Matplotlib this axis is dates
                self._apply_time_axis_format(ax)
                # Set the bounds now so autoscale can't switch units later
                ax.set_xlim(self.t0, self.t1, emit=False)
    
        # --- Window the dataframe ---
        try:
            sub_df = self.df.loc[self.t0:self.t1]
        except Exception:
            # Unsorted/duplicated index: fall back to the FULL frame --
            # recorded now instead of silent (Pack 4 A4). The stale-cache
            # hazard this used to compound is closed by EDIT 087, which
            # caches sub_df.index regardless of render outcome.
            self._warn_once(
                "window-fallback",
                "windowing df.loc[t0:t1] failed; plotting the FULL frame")
            sub_df = self.df
    
        # Also window any time-like arrays carried in df.attrs so plot_fn can use them.
        # (Uses half-open [j0, j1) semantics.)
        try:
            if len(self.df.index) and len(sub_df.index):
                # j0 = index position of sub_df's first row; j1 = j0 + len(window)
                j0 = self.df.index.get_indexer([sub_df.index[0]])[0]
                j0 = max(0, int(j0))
                j1 = j0 + len(sub_df.index)
            else:
                j0, j1 = 0, 0
        except Exception:
            # Window positions default to the array START: attrs arrays
            # handed to plot_fn are misaligned by the window offset (A5).
            self._warn_once(
                "attrs-window-positions",
                "df.attrs window positions failed; attrs arrays are "
                "sliced from position 0 (misaligned)")
            j0, j1 = 0, len(sub_df.index)
    
        try:
            # shallow copy so we can set attrs without touching self.df
            sub_df = sub_df.copy(deep=False)
            sub_df.attrs = self._build_window_attrs_view(j0, j1)
        except Exception:
            # plot_fn receives FULL-LENGTH attrs beside a windowed frame
            # (shallow copy inherits self.df.attrs) -- recorded (A6).
            self._warn_once(
                "attrs-window-view",
                "df.attrs windowed view failed; plot_fn receives "
                "full-length attrs arrays")
    
        # --- Draw-only decimation (Pack 5 R4b / R11) ---
        # Chooses WHICH original rows to draw; never averages or
        # synthesises one. Selection, rules, labeling and export keep
        # reading self.df at full resolution -- pinned by
        # test_decimation_is_draw_only.
        sub_df = self._decimate_for_draw(sub_df)

        # --- User plot function ---
        # Cache the windowed index BEFORE rendering (Pack 4 R7): the
        # windowing above succeeded independently of plot_fn, and a stale
        # cache mis-mapped box selections onto the PREVIOUS window --
        # measured: an interval committed 1 hour off, with a success
        # message (evidence pack4_repro_T1). It caches the frame plot_fn
        # actually receives, so Pack 3's artist-ordinal condition (artist
        # point count == len(windowed_idx)) still holds when that frame
        # has been decimated.
        self._last_windowed_index = sub_df.index.copy()

        # Pack 8.5-B B1: cache the FRAME beside its index. The highlight
        # extractor (selection.py `_extract_data_at_indices`) used to
        # rebuild its own reference window from `self.df.loc[t0:t1]`,
        # which is the FULL window even when this frame was decimated --
        # so its artist-length gate rejected every drawn line and the red
        # and blue marks silently vanished. Measured on the 1M-row frame
        # at a 2-day window: 44,572 rows in the full window, 36,316 drawn,
        # 0 marks extracted. The two names are written together here so
        # they can never describe different frames.
        self._last_windowed_frame = sub_df

        if len(sub_df.index) == 0:
            # Zoomed finer than the data cadence: say so instead of
            # handing user code an empty frame and calling the explosion
            # a "Plot error" (Pack 4, ledger 7.x cheap fix).
            first_ax = next(iter(self.user_axes.values()), None)
            if first_ax is not None:
                first_ax.text(
                    0.5, 0.5,
                    "No samples in this time range\n(zoom out or pan)",
                    transform=first_ax.transAxes, ha="center", va="center"
                )
        else:
            try:
                self.plot_fn(self.user_axes, sub_df, self.t0, self.t1)
            except Exception as e:
                # Full traceback to the forensic channel; a three-line
                # honest summary on ONE axis; a statusbar pointer (R6).
                logger.exception("plot_fn raised during redraw")
                first_ax = next(iter(self.user_axes.values()), None)
                if first_ax is not None:
                    first_ax.text(
                        0.02, 0.98, self._plot_error_text(e),
                        transform=first_ax.transAxes, ha="left", va="top",
                        fontsize=8, family="monospace", wrap=True
                    )
                if getattr(self, "status_var", None) is not None:
                    self.status_var.set(
                        "Plot error -- details in chronotagger.log")

        # --- Re-point registered colorbars (Pack 8.5 SP-R5) ---
        # The ax.clear() at the top of this method destroyed the artist
        # every registered colorbar was built from -- measured: the
        # artist leaves ax.get_children() and its .figure goes None. The
        # colorbar, its axes and its geometry all survive (position
        # byte-identical across redraws, a pan, a zoom and a
        # constrained-layout re-solve); only the pointer is stale. One
        # generic sweep here, after plot_fn drew the new artist and
        # before canvas.draw(), closes it with no per-frame user code,
        # and is a no-op on every pane that never called
        # utils.spectrogram.attach_colorbar.
        try:
            refresh_colorbars(self.user_axes.values())
            if take_layout_dirty(self.user_axes.values()):
                # Pack 8.5-B B4: a gutter colorbar axes was just born,
                # inside plot_fn, after the solver had already placed the
                # panels -- so it is sitting in its raw gridspec cell.
                # Ask for ONE more solve; the canvas.draw() at the end of
                # this method performs it and _freeze_layout_after_draw
                # locks the result. Once per bar, not once per frame.
                self._invalidate_layout_freeze()
        except Exception:
            # A colour scale may never take the redraw down (Pack 5
            # doctrine, and the decimation fallback above).
            self._warn_once(
                "colorbar-refresh",
                "re-pointing a registered colorbar failed; its colour "
                "scale may be stale")

        # Capture auto limits after plot_fn() completes
        self._capture_auto_limits()
        
        # If time range changed, reset all manual zooms
        if self._time_range_dirty:
            self._reset_limits_to_auto()
            self._time_range_dirty = False
    
        # Partition axes into time vs non-time (grid-only requires explicit keys)
        if not getattr(self, "_time_axis_keys", None):
            raise RuntimeError("No time axes registered; check your layout_spec.")
        time_axes = {k: self.user_axes[k] for k in self._time_axis_keys if k in self.user_axes}
    
        # Align limits + date formatting for time axes (single pass)
        with self._squelch_xlim_events():
            for ax in time_axes.values():
                # zero horizontal padding
                try:
                    ax.set_xmargin(0.0)
                except Exception:
                    pass
                try:
                    ax.margins(x=0.0)
                except Exception:
                    pass
                # hard-limit to the window
                ax.set_xlim(self.t0, self.t1, emit=False)
                self._apply_time_axis_format(ax)
    
        # Compact x-labels per column (unchanged policy)
        self._apply_time_xlabel_policy()
    
        # Background interval overlays across time panels only
        if self._overlays_enabled() and time_axes:
            from ..utils.overlays import draw_interval_bands
            from chronotagger.core.tracks import active_id_of, intervals_on
            # Pack M1: the ACTIVE track only. self.class_colors IS the
            # active track's colour map, so painting every lane through it
            # paints an imported machine label in the colour of a
            # hand-drawn one: measured with two tracks through this very
            # call, an imported `sw` band came out in EXACTLY the human
            # track's `sw` colour -- visually indistinguishable from a hand
            # label -- and an `umbra` band the active track does not know
            # came out grey. Grey at least looks wrong. The other lanes are
            # still in the model, the autosave, the session and the export;
            # M2 gives them their own bands.
            draw_interval_bands(
                time_axes,
                intervals_on(self.intervals, active_id_of(self)),
                self.t0, self.t1,
                self.class_colors,
                selected_interval=self.selected_interval,
                preview=self.current_selection,
                preview_spans=getattr(self, "current_spans", None),
                alpha=0.15,
                alpha_selected=0.16,
                alpha_preview=0.12,
                zorder=0.05,
            )
    
        # Strip + sidebar
        self._update_strip()
        # Pack M2: and EVERY OTHER PANE's strip, because the lanes are a
        # property of the session and a pane the user tabs back to must not
        # show the lane table as it was three gestures ago. The inactive
        # CANVASES are deliberately NOT drawn: building all three strips
        # costs 30.0 ms at 120 intervals and 76.7 ms at 2,000, while
        # drawing the other two canvases costs 253 ms on top -- and
        # _on_tab_changed (view_build/window.py) already calls _update_plot
        # when the user actually gets there. Pack 6 deleted the dirty flag
        # that would have made this lazy; this is the cheap half of it.
        for _p in (getattr(self, "panes", None) or []):
            if _p is self.active_pane:
                continue
            if getattr(_p, "strip_ax", None) is None:
                continue
            try:
                self._update_strip(_p)
            except Exception:
                # A stale pane's strip is a cosmetic problem; the active
                # pane's paint has already happened and must not be lost.
                pass
        if hasattr(self, 'intervals_tree'):
            self._update_intervals_list()
    
        # --- Keep two-click preview overlays alive across redraws (existing logic) ---
        if getattr(self, "two_click_mode", False):
            try:
                # Pack 6 R3: an `_update_time_overlays(self._two_click_t0,
                # last)` used to follow, with `last` read from
                # _two_click_last_x. That name is written in four places
                # and EVERY write is None, so the read always produced
                # None and the call raised TypeError at `min(x0, x1)` --
                # before touching a single artist -- straight into the
                # `except Exception: pass` below. The feature ("the
                # two-click preview band survives a zoom/pan redraw") has
                # never once run. Making it work is a FEATURE and belongs
                # in a feature pack, not here.
                #
                # The rebuild call BELOW stays. It is this block's only
                # live statement and that method's only caller in the tree:
                # it recreates the preview patches after a redraw that
                # cleared the axes. Measured -- delete it and every
                # two-click band detaches on the first redraw (3/3 attached
                # with the call, 0/3 without), with the whole suite green.
                self._rebuild_time_overlays_if_needed()
            except Exception:
                pass
        
        # --- Restore point highlights if there's an active selection preview ---
        # This ensures highlights persist across zoom/pan/time range changes
        if hasattr(self, '_show_selected_point_highlights'):
            # Check if there's an active selection that needs highlighting
            has_selection = (
                (hasattr(self, 'current_selection') and self.current_selection is not None) or
                (hasattr(self, 'current_spans') and self.current_spans)
            )
            
            if has_selection:
                try:
                    self._show_selected_point_highlights(redraw=False)
                except Exception:
                    # Highlights vanishing with no reason given was a
                    # named ledger target (Pack 4 B3). Fallback unchanged.
                    self._warn_once(
                        "highlight-restore-preview",
                        "restoring selection point highlights failed")
        
        # --- Restore selected interval highlights if there's a selected interval ---
        if hasattr(self, '_show_selected_interval_highlights'):
            if hasattr(self, 'selected_interval') and self.selected_interval is not None:
                try:
                    self._show_selected_interval_highlights()
                except Exception:
                    self._warn_once(
                        "highlight-restore-interval",
                        "restoring selected interval highlights failed")
    
        self.canvas.draw()
        # The constrained-layout solver has now produced a geometry for
        # this figure; freeze it (Pack 5 R4a). Genuine layout changes call
        # _invalidate_layout_freeze and the next draw re-solves.
        self._freeze_layout_after_draw()

    def _capture_auto_limits(self) -> None:
        """
        Capture current axis limits as 'auto' state after plot_fn() renders.
        
        For time-series plots (role='time'):
            - X limits are managed by time window (not captured)
            - Y limits are captured for zoom reset
        
        For cross-plots (role='not-time'):
            - Both X and Y limits are captured for zoom reset
        """
        self._auto_xlims.clear()
        self._auto_ylims.clear()
        
        for key, ax in self.user_axes.items():
            role = self.axes_meta.get(key, {}).get('role', 'time').lower()
            
            # Always capture Y limits (all plots support Y-zoom)
            self._auto_ylims[ax] = ax.get_ylim()
            
            # Capture X limits only for cross-plots (not-time)
            if role == 'not-time':
                self._auto_xlims[ax] = ax.get_xlim()
    
    def _reset_limits_to_auto(self) -> None:
        """
        Reset all manually-zoomed axes back to auto limits.
        Called when time range changes via navigation or time zoom.
        """
        for ax, (ymin, ymax) in self._auto_ylims.items():
            ax.set_ylim(ymin, ymax)
        
        for ax, (xmin, xmax) in self._auto_xlims.items():
            ax.set_xlim(xmin, xmax)

    def _apply_time_xlabel_policy(self) -> None:
        """
        Keep Matplotlib defaults everywhere, except:
          • hide x tick labels and xlabel for time-role axes in column 0
            (the strip owns the time axis).
        """
        meta = getattr(self, "axes_meta", None)
        if not isinstance(meta, dict) or not meta:
            return
    
        for key, m in meta.items():
            ax = self.user_axes.get(key)
            if ax is None:
                continue
    
            col = int(m.get("col", 0))
            role = str(m.get("role", "time")).lower()
    
            if col == 0 and role == "time":
                ax.tick_params(axis="x", labelbottom=False)
                ax.set_xlabel("")
                fmt = ax.xaxis.get_major_formatter()
                if hasattr(fmt, "show_offset"):
                    fmt.show_offset = False
                ax.xaxis.get_offset_text().set_visible(False)
            else:
                # Show normal labels and make sure they aren't clipped by their own axes box
                ax.tick_params(axis="x", labelbottom=True)
                for lbl in ax.get_xticklabels():
                    lbl.set_clip_on(False)
                    
    def _rebuild_time_overlays_if_needed(self) -> None:
        """
        Ensure the two-click preview rectangles exist on each time axis + strip
        after a redraw that cleared axes.
        """
        import matplotlib.patches as mpatches
        from matplotlib.transforms import blended_transform_factory
    
        if not getattr(self, "two_click_mode", False):
            return
    
        if not hasattr(self, "_time_overlays"):
            # First-time init from EventsMixin
            if hasattr(self, "_init_time_overlays"):
                self._init_time_overlays()
            return
    
        axes = []
        if getattr(self, "_time_axis_keys", None):
            axes.extend(self.user_axes[k] for k in self._time_axis_keys if k in self.user_axes)
        if getattr(self, "strip_ax", None) is not None:
            axes.append(self.strip_ax)
    
        # Recreate missing patches
        for ax in axes:
            rect = self._time_overlays.get(ax)
            if rect is None or rect not in ax.patches:
                trans = blended_transform_factory(ax.transData, ax.transAxes)
                r = mpatches.Rectangle(
                    (0, 0), 0, 1,
                    transform=trans,
                    facecolor="tab:orange",
                    edgecolor="none",
                    alpha=0.25,
                    zorder=ax.get_zorder() + 10,
                    visible=False,
                )
                ax.add_patch(r)
                self._time_overlays[ax] = r
    
        # Remove stale entries (axes that no longer exist)
        stale = [ax for ax in self._time_overlays.keys() if ax not in axes]
        for ax in stale:
            self._time_overlays.pop(ax, None)
            
    def _hook_time_xlim(self) -> None:
        """Connect 'xlim_changed' on the primary time axis only."""
        if getattr(self, "_xlim_cb_cid", None) is not None:
            return
        key = getattr(self, "_primary_time_key", None)
        if key is None:
            return
        ax = self.user_axes.get(key)
        if ax is None:
            return
        self._xlim_cb_cid = ax.callbacks.connect("xlim_changed", self._on_time_xlim_changed)
    
    
    def _on_time_xlim_changed(self, ax) -> None:
        """Toolbar zoom/pan -> update t0/t1 -> full redraw. Guard against feedback."""
        if getattr(self, "_squelch_xlim", False):
            return
    
        import matplotlib.dates as mdates
        import pandas as pd
    
        lo, hi = ax.get_xlim()
        try:
            t0_new = pd.Timestamp(mdates.num2date(lo)).tz_localize(None)
            t1_new = pd.Timestamp(mdates.num2date(hi)).tz_localize(None)
        except Exception:
            return
    
        # No-op if the limits didn't really change (avoid float churn)
        if getattr(self, "t0", None) is not None and getattr(self, "t1", None) is not None:
            if self.t0 == t0_new and self.t1 == t1_new:
                return
    
        self.t0, self.t1 = t0_new, t1_new
    
        # keep Start/End UI in sync if present
        try:
            self.start_time_entry.delete(0, "end"); self.start_time_entry.insert(0, str(self.t0))
            self.end_time_entry.delete(0, "end");   self.end_time_entry.insert(0, str(self.t1))
        except Exception:
            pass

        # Toolbar zoom/pan is a CONTINUOUS gesture: matplotlib emits
        # xlim_changed for every intermediate limit. Coalesced (Pack 5
        # R4d): the drag renders once, at the limits it ended on.
        self._request_redraw()

    def _build_window_attrs_view(self, j0: int, j1: int) -> dict:
        """
        Build a dict for sub_df.attrs where array-likes that look 'time-like' are
        sliced to the current window. Handles:
          • 1D arrays length N == len(index)          -> v[j0:j1]
          • ND arrays with last axis N == len(index)  -> v[..., j0:j1]
          • tuple/list of such arrays                  -> elementwise sliced
        If N != len(index) but N is plausibly time-like, we apply a
        *proportional* slice: map [j0, j1] from base length to N.
        Everything else is passed through unchanged.
        """
        import numpy as _np
    
        base = getattr(self.df, "attrs", {}) or {}
        base_len = len(self.df.index)
        out: dict = {}
    
        def _proj_bounds(n: int) -> tuple[int, int]:
            """
            Proportionally project [j0, j1] (in base_len units) to [p0, p1] in n units.
            Keeps at least one element when possible.
            """
            if base_len <= 0 or n <= 0:
                return 0, 0
            p0 = int(round(j0 * n / base_len))
            p1 = int(round(j1 * n / base_len))
            if p1 <= p0:
                p1 = min(p0 + 1, n)
            p0 = max(0, min(p0, n))
            p1 = max(0, min(p1, n))
            return p0, p1
    
        def _slice_like(v):
            # numpy / array-ish with ndim
            if hasattr(v, "shape") and getattr(v, "ndim", 0) >= 1:
                n_last = v.shape[-1]
                # exact last-axis alignment
                if base_len and n_last == base_len:
                    return v[..., j0:j1]
                # proportional fallback if length is plausibly time-like (not wildly different)
                ratio = (n_last / base_len) if base_len else 0
                if base_len and n_last >= 8 and 0.1 <= ratio <= 10.0:
                    p0, p1 = _proj_bounds(n_last)
                    return v[..., p0:p1]
                return v
    
            # generic sequences (lists/tuples) — leave slicing to the container case below
            return v
    
        for k, v in base.items():
            try:
                # Handle tuples/lists by element
                if isinstance(v, (list, tuple)):
                    sliced_elems = []
                    changed = False
                    for item in v:
                        new_item = _slice_like(item)
                        if new_item is not item:
                            changed = True
                        sliced_elems.append(new_item)
                    out[k] = type(v)(sliced_elems) if changed else v
                    continue
    
                # 1D pandas Series treated like arrays (len check)
                if hasattr(v, "__len__") and not hasattr(v, "shape"):
                    n = len(v)
                    if base_len and n == base_len:
                        return_v = v[j0:j1]
                        out[k] = return_v
                        continue
                    if base_len and n >= 8 and 0.1 <= (n / base_len) <= 10.0:
                        p0, p1 = _proj_bounds(n)
                        out[k] = v[p0:p1]
                        continue
    
                out[k] = _slice_like(v)
            except Exception:
                out[k] = v
    
        return out

    def _update_strip(self, pane=None) -> None:
        """Redraw annotation strip (intervals + current selection preview).

        Pack M2: `pane` names WHICH pane's strip to repaint, defaulting to
        the active one, which is what every one of the fifteen existing
        callers means. It is a parameter rather than an active-pane swap
        because the swap would change what every other reader on `self`
        sees for the duration of the paint, and this method writes four
        attributes onto the pane it paints.
        """
        import matplotlib.dates as mdates
        # Pack 5 R14. Function-scoped on purpose: TOOL_GID_PREFIX lives in
        # mixins.events.base, and this module is imported BY that package,
        # so a module-level import would be a cycle.
        import numpy as np
        import pandas as pd
        from matplotlib.collections import PolyCollection
        from matplotlib.colors import to_rgba
        from matplotlib.transforms import blended_transform_factory
        from .events.base import TOOL_GID_PREFIX

        _pane = pane
        if _pane is None:
            _pane = self.active_pane if hasattr(self, "active_pane") else self
        ax = getattr(_pane, "strip_ax", None)
        if ax is None:
            ax = self.strip_ax  # type: ignore[assignment]

        # ---------------------------------------------------------- Pack M2
        # THE LANE TABLE FOR THIS PAINT. Computed ONCE, here, and stored on
        # the PANE, because both hit tests and the drag preview must read
        # exactly what was painted. The gather measured what happens when a
        # second reader keeps its own copy: paint pane 2 with one lane
        # hidden while pane 0 still shows three and the active pane's hit
        # test uses K=2 against an 18-face collection -- every click lands
        # on the wrong lane, silently. `_strip_preview_pool` already lives
        # on the pane (events/strip.py) for the same reason.
        #
        # AT K == 1 EVERYTHING BELOW IS A NO-OP AND THAT IS THE POINT: no
        # lane name, no focus ring, no legend, pad 0.1, preview 0.05..0.95
        # -- the strip renders byte-for-byte as it did before this pack
        # (whole-figure PNG sha256, six conditions, measured). With one
        # lane there is also nothing to switch to, so there is nothing for
        # a lane name or a ring to tell you.
        from chronotagger.core.lanes import active_band, lane_layout
        from chronotagger.core.tracks import table_of, track_ids as _track_ids
        _lay = lane_layout(self)
        _lanes = _lay["lanes"]
        _row_of = _lay["row_of"]
        _K = _lay["k"]
        _pad = _lay["pad"]
        _active_row = _lay["active_row"]
        _pane._strip_lane_count = _K
        _pane._strip_pad_frac = _pad
        _pane._strip_lane_ids = [t.id for t in _lanes]
        _pane._strip_active_row = _active_row
        # Pack M2.1: the lane names and the active lane's legend are
        # artists the constrained-layout solver had never seen when this
        # figure's geometry was frozen after its first draw (Pack 5 R4a),
        # so on a layout with a narrow margin they fall off the figure --
        # measured on the C05 feel-test driver: 43-62 px of every lane
        # name and 78 % of the legend invisible on the Context tab. Ask
        # for ONE more solve whenever the thing the solver must make room
        # for changes (the lane count, the lane set, or the active lane,
        # whose legend is the widest artist), and only above one lane, so
        # a single-lane figure keeps its byte-identical geometry. The
        # draw at the end of _update_plot re-solves and refreezes.
        _sig = (_K, tuple(t.id for t in _lanes), self.active_track_id)
        if _K > 1 and getattr(_pane, "_strip_layout_sig", None) != _sig:
            self._invalidate_layout_freeze(_pane)
        _pane._strip_layout_sig = _sig
        # The active lane's PREVIEW band. The two painted preview
        # rectangles below read it, so one paint asks for it once. The DRAG
        # path deliberately does NOT read it and computes its own: a cached
        # band is a stale band after any lane switch that does not repaint,
        # and the live call measures 0.0023 ms against a 1.78 ms blit.
        _pane._strip_active_band = active_band(self)
        _known_ids = set(_track_ids(table_of(self)))
        # The focus colour: the ACTIVE lane's colour for the class the
        # dropdown is on right now, so the lane name and the ring say both
        # WHERE the next Add lands and IN WHAT COLOUR. Falls back to a
        # neutral blue when the lane does not declare that class (which is
        # the state _repoint_class_controls exists to prevent).
        _cur_class = ""
        if getattr(self, "current_class_var", None) is not None:
            try:
                _cur_class = self.current_class_var.get()
            except Exception:
                _cur_class = ""
        _focus_color = "#1f77b4"
        if _active_row is not None:
            _focus_color = (_lanes[_active_row].class_colors.get(_cur_class)
                            or "#1f77b4")
        if hasattr(self, "_refresh_lane_controls"):
            self._refresh_lane_controls()

        with self._squelch_xlim_events():
            ax.clear()
            ax.set_ylim(0, 1)
            if _K > 1:
                # ONE tick per lane, at the centre of its band, labelled
                # with the lane's NAME -- 0 extra artists, and the only
                # focus indicator that is visible when the active lane
                # holds no interval in this window (a heavier per-face
                # edge is invisible exactly then; measured).
                from chronotagger.core.lanes import lane_band
                _ticks = []
                for _r in range(_K):
                    _lo, _hi = lane_band(_r, _K, _pad)
                    _ticks.append(0.5 * (_lo + _hi))
                ax.set_yticks(_ticks)
                ax.set_yticklabels([(t.name or t.id) for t in _lanes],
                                   fontsize=7)
                for _r, _lbl in enumerate(ax.get_yticklabels()):
                    if _r == _active_row:
                        _lbl.set_fontweight("bold")
                        _lbl.set_color(_focus_color)
                    else:
                        _lbl.set_color("#555555")
            else:
                ax.set_yticks([])
            ax.set_ylabel("Labels", fontsize=9)
        
            # Reset limits/formatting because clearing resets formatter
            ax.set_xlim(self.t0, self.t1, emit=False)
            self._apply_time_axis_format(ax)
            
            # zero horizontal padding so the strip matches time panels exactly
            try:
                ax.set_xmargin(0.0)
            except Exception:
                pass
            try:
                ax.margins(x=0.0)
            except Exception:
                pass

        # Labeled intervals in strip -- ONE PolyCollection, not one
        # Rectangle per interval (Pack 5 R14). Structurally the same
        # disease R4c cured on the data panels: at 2,000 intervals this
        # loop built 2,000 pickable patches every frame and measured
        # 1,419.9 ms of a 2,251.4 ms redraw; the collection measures
        # 27.0 ms and takes the whole frame to 730.9 ms.
        # Per-face facecolor / edgecolor / linewidth keep the
        # selected-interval emphasis exactly as the Rectangles had it.
        #
        # Pack M2: STILL ONE COLLECTION AT EVERY K. The lanes are 4K faces
        # with per-vertex y, not K artists -- measured flat in K: at 2,000
        # intervals the build is 23.6-25.1 ms across K=1..4 against the
        # one-lane painter's 24.3 ms. A per-lane artist would put R14's
        # 1,419.9 ms back one lane at a time.
        spans = []
        faces = []
        edges = []
        widths = []
        lane_rows = []      # the visual row of each face, parallel to spans
        band_ivs = []       # the Interval behind each face, parallel to spans
        strays = 0          # intervals on a track this table does not hold
        for iv in self.intervals:
            # Each lane paints in ITS OWN colours: an imported machine
            # label must not borrow the human lane's colour for a shared
            # class name, and must not be grey because the ACTIVE lane
            # does not know its class (both measured under M1's painter).
            _t_row = _row_of.get(iv.track)
            if _t_row is None:
                # Either a hidden lane -- not painted, and therefore
                # unreachable by both hit tests, which is the whole
                # mechanism -- or a stray, counted and reported once.
                if iv.track not in _known_ids:
                    strays += 1
                continue
            if iv.end <= self.t0 or iv.start >= self.t1:
                continue
            s = max(iv.start, self.t0)
            e = min(iv.end, self.t1)

            selected = iv == self.selected_interval
            color = _lanes[_t_row].class_colors.get(iv.label, "#cccccc")
            spans.append((s, e))
            faces.append(to_rgba(color, 0.8 if selected else 0.6))
            edges.append(to_rgba("red" if selected else "black", 1.0))
            widths.append(2.0 if selected else 0.5)
            lane_rows.append(_t_row)
            band_ivs.append(iv)

        if spans:
            x0 = mdates.date2num(
                pd.DatetimeIndex([s for s, _ in spans]).to_numpy())
            x1 = mdates.date2num(
                pd.DatetimeIndex([e for _, e in spans]).to_numpy())
            verts = np.empty((len(spans), 4, 2), dtype=float)
            verts[:, 0, 0] = x0
            verts[:, 1, 0] = x1
            verts[:, 2, 0] = x1
            verts[:, 3, 0] = x0
            # Pack M2: per-face y, from ONE authority. At K == 1 with
            # pad 0.1 lane_band(0, 1, 0.1) is exactly (0.1, 0.9) -- the
            # four constants this block used to carry -- which is the
            # first of the three rules the single-lane byte identity
            # needs. Vectorised per lane: K slices, not N lookups.
            from chronotagger.core.lanes import lane_band
            _rows = np.asarray(lane_rows, dtype=int)
            for _r in range(_K):
                _lo, _hi = lane_band(_r, _K, _pad)
                _m = _rows == _r
                verts[_m, 0, 1] = _lo
                verts[_m, 1, 1] = _lo
                verts[_m, 2, 1] = _hi
                verts[_m, 3, 1] = _hi
            bands = PolyCollection(
                verts, facecolors=faces, edgecolors=edges,
                linewidths=widths, picker=True)
            # Pack M2: a pick must DECLINE in the gutter between two lanes
            # and above/below every band, or the two hit paths disagree.
            # At the default radius a click 5.1 px outside a band still
            # picks the nearest face and names the wrong lane (measured at
            # K=4: pick_row 2 where the geometry says row 3). Zero makes
            # the collection answer for its own paint, and the press path's
            # `lane_strict` declines in exactly the same places.
            bands.set_pickradius(0)
            # y in AXES coordinates -- the strip's ylim is pinned to (0, 1)
            # a few lines above, so this is the same geometry the
            # Rectangles had, and it survives any future ylim change.
            bands.set_transform(
                blended_transform_factory(ax.transData, ax.transAxes))
            bands.set_gid(TOOL_GID_PREFIX + "strip-bands")
            ax.add_collection(bands, autolim=False)

        # Pack M2: the two arrays the hit tests resolve `.ind` through, on
        # the PANE, written on EVERY paint including the empty one -- a
        # stale array from the previous window would name an interval that
        # is no longer under the cursor.
        _pane._strip_band_ivs = band_ivs
        _pane._strip_band_rows = lane_rows

        # Pack M2: the dashed preview lives ON THE ACTIVE LANE, bracketing
        # its band by half the gutter. At K == 1 `preview_band(0, 1, 0.1)`
        # is exactly (0.05, 0.9) -- the two constants this block used to
        # carry -- and that bracket is the third of the three rules the
        # single-lane byte identity needs: the shipped band is 0.1..0.9 and
        # the shipped preview is 0.05..0.95, so the preview is NOT the
        # band. It was the one condition that failed before the gather
        # found it.
        from chronotagger.core.lanes import (strip_focus_gid,
                                             strip_preview_gid)
        _pv_y, _pv_h = _pane._strip_active_band   # recorded a moment ago

        # single-span preview
        if self.current_selection:
            s, e = self.current_selection
            rect = Rectangle(
                (mdates.date2num(s), _pv_y),
                mdates.date2num(e) - mdates.date2num(s),
                _pv_h,
                facecolor="yellow",
                edgecolor="orange",
                linewidth=2,
                alpha=0.3,
                linestyle="--",
            )
            rect.set_gid(strip_preview_gid())
            ax.add_patch(rect)

        # multi-span preview
        if getattr(self, "current_spans", None):
            for (s, e) in self.current_spans:
                rect = Rectangle(
                    (mdates.date2num(s), _pv_y),
                    mdates.date2num(e) - mdates.date2num(s),
                    _pv_h,
                    facecolor="yellow",
                    edgecolor="orange",
                    linewidth=2,
                    alpha=0.3,
                    linestyle="--",
                )
                rect.set_gid(strip_preview_gid())
                ax.add_patch(rect)

        # ---------------------------------------------------------- Pack M2
        # THE FOCUS RING: ONE Rectangle in transAxes over the active lane's
        # band, full width. Constant in K -- it is not a per-lane artist --
        # and NOT PICKABLE (a Patch with no picker can never produce a
        # PickEvent, so _on_strip_click's `strip_ax.patches` gate never
        # sees it; measured). It is the only indicator that marks the
        # lane's whole width, and together with the bold tick label it is
        # visible even when the active lane holds nothing in this window,
        # which is when the user most needs to know where an Add would go.
        if _K > 1 and _active_row is not None:
            from chronotagger.core.lanes import lane_band
            _rlo, _rhi = lane_band(_active_row, _K, _pad)
            ring = Rectangle(
                (0.0, _rlo), 1.0, _rhi - _rlo,
                transform=ax.transAxes,
                facecolor="none",
                edgecolor=_focus_color,
                linewidth=1.6,
                linestyle="-",
                zorder=3.0,
                clip_on=False,
            )
            ring.set_gid(strip_focus_gid())
            ax.add_patch(ring)

            # ONE legend, for the ACTIVE lane, titled with the lane's name,
            # OUTSIDE the axes. Measured +0.90 ms and 0 strip pixels. One
            # legend PER LANE was the alternative and it is not survivable:
            # 13 entries at fontsize 5 want ~91 px of stacked text, which is
            # taller than the WHOLE strip on two of his four real layouts
            # (78.3 px and 76.3 px).
            _act = _lanes[_active_row]
            _handles = [Rectangle((0, 0), 1, 1,
                                  facecolor=to_rgba(
                                      _act.class_colors.get(c, "#cccccc"),
                                      0.6),
                                  edgecolor="black", linewidth=0.5)
                        for c in _act.classes]
            if _handles:
                _leg = ax.legend(
                    _handles, list(_act.classes),
                    title=(_act.name or _act.id),
                    loc="upper left", bbox_to_anchor=(1.002, 1.0),
                    borderaxespad=0.0, fontsize=6, title_fontsize=7,
                    handlelength=1.0, handleheight=0.8, labelspacing=0.25,
                    frameon=False)
                _leg.set_zorder(3.5)

        # Strays: an interval on a track the table does not hold paints
        # nothing and is reachable by nothing. It is not silent -- the two
        # load paths refuse it and the export refuses it -- but a repaint
        # is where the user is looking, so say it here too, once, and do
        # NOT invent a grey orphan lane for it: that would change K, and K
        # changes every band's geometry.
        if strays and getattr(self, "status_var", None) is not None:
            try:
                self.status_var.set(
                    "%d interval(s) sit on tracks this table does not hold"
                    % (strays,))
            except Exception:
                pass

        # Pack M2 v2 fold: THE ACTIVE LANE IS HIDDEN. DR11 keeps this case
        # distinct on purpose and the painter must not re-point the model,
        # but silence here is the exact shape of the bug this pack exists
        # to remove: measured, an undo across a visibility toggle leaves
        # `a` active and hidden (K=2, active_row None, no ring, no legend,
        # the preview back across the whole strip), and the next Add lands
        # on a lane the user cannot see and says "Added 1 x interval(s)".
        # Say it, once per paint, on the ACTIVE pane only.
        elif (_K and _active_row is None
                and _pane is getattr(self, "active_pane", _pane)
                and getattr(self, "status_var", None) is not None):
            try:
                from chronotagger.core.lanes import track_display_name
                self.status_var.set(
                    "the active lane '%s' is HIDDEN -- an edit would land "
                    "where you cannot see it (Ctrl+H shows it again)"
                    % (track_display_name(self, _lay["active_id"]),))
            except Exception:
                pass
