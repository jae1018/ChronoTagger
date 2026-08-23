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

logger = logging.getLogger(__name__)


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
    
        # --- User plot function ---
        # Cache the windowed index BEFORE rendering (Pack 4 R7): the
        # windowing above succeeded independently of plot_fn, and a stale
        # cache mis-mapped box selections onto the PREVIOUS window --
        # measured: an interval committed 1 hour off, with a success
        # message (evidence pack4_repro_T1).
        self._last_windowed_index = sub_df.index.copy()

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
            draw_interval_bands(
                time_axes,
                self.intervals,
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
        if hasattr(self, 'intervals_tree'):
            self._update_intervals_list()
    
        # --- Keep two-click preview overlays alive across redraws (existing logic) ---
        if getattr(self, "two_click_mode", False):
            try:
                self._rebuild_time_overlays_if_needed()
                if getattr(self, "_two_click_active", False) and getattr(self, "_two_click_t0", None) is not None:
                    last = getattr(self, "_two_click_last_x", self._two_click_t0)
                    self._update_time_overlays(self._two_click_t0, last)
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
        
        # Clear manual zoom tracking
        self._manual_zooms.clear()
        
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
    
        self._update_plot()
            
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

    def _update_strip(self) -> None:
        """Redraw annotation strip (intervals + current selection preview)."""
        import matplotlib.dates as mdates
        
        ax = self.strip_ax  # type: ignore[assignment]
        with self._squelch_xlim_events():
            ax.clear()
            ax.set_ylim(0, 1)
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

        # Labeled intervals in strip
        for iv in self.intervals:
            if iv.end <= self.t0 or iv.start >= self.t1:
                continue
            s = max(iv.start, self.t0)
            e = min(iv.end, self.t1)

            color = self.class_colors.get(iv.label, "#cccccc")
            alpha = 0.8 if iv == self.selected_interval else 0.6
            edgecolor = "red" if iv == self.selected_interval else "black"
            lw = 2 if iv == self.selected_interval else 0.5

            rect = Rectangle(
                (mdates.date2num(s), 0.1),
                mdates.date2num(e) - mdates.date2num(s),
                0.8,
                facecolor=color,
                edgecolor=edgecolor,
                linewidth=lw,
                alpha=alpha,
                picker=True,
            )
            ax.add_patch(rect)

        # single-span preview
        if self.current_selection:
            s, e = self.current_selection
            rect = Rectangle(
                (mdates.date2num(s), 0.05),
                mdates.date2num(e) - mdates.date2num(s),
                0.9,
                facecolor="yellow",
                edgecolor="orange",
                linewidth=2,
                alpha=0.3,
                linestyle="--",
            )
            ax.add_patch(rect)

        # multi-span preview
        if getattr(self, "current_spans", None):
            for (s, e) in self.current_spans:
                rect = Rectangle(
                    (mdates.date2num(s), 0.05),
                    mdates.date2num(e) - mdates.date2num(s),
                    0.9,
                    facecolor="yellow",
                    edgecolor="orange",
                    linewidth=2,
                    alpha=0.3,
                    linestyle="--",
                )
                ax.add_patch(rect)
