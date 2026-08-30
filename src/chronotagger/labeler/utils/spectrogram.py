"""
Image-panel helpers: time rebinning, the fast draw, and a colorbar that
survives the redraw loop (Pack 8.5).

WHY THIS MODULE EXISTS. A spectrogram panel drawn with ``pcolormesh`` at
native resolution costs ~0.41 us per quad, and the quad count grows with
the window: measured on real THEMIS-B / ARTEMIS ESA ion energy flux at a
1400x800 figure, one 32-channel panel costs 50 ms of a 166 ms redraw at
3,736 rows and 1,115 ms of a 1,282 ms redraw at 89,128 rows -- 68-87 % of
every frame wider than a few hours, and the only term left that grows
with the window. Rebinning the window onto the panel's pixel columns
makes that cost flat FOR THE RULED DEFAULT AGGREGATOR ``max`` -- 1.5 ms
at 3,736 rows and 1.9 ms at 89,128, a 1.3x rise over a 24x rise in rows
-- and, at the same time, MANUFACTURES the uniform time grid that makes
``imshow`` -- the cheapest primitive -- legal at all. ``mean`` and
``logmean`` stay far under the mesh (19 ms and 43 ms at 89,128 rows,
against 1,098 ms) but do NOT go flat: over that same 24x range they rise
11.9x and 12.1x. Flat is a property of the default, not of rebinning.

THE THREE THINGS TO KNOW BEFORE USING IT
----------------------------------------

1. ``imshow`` on a NATIVE spectra grid is WRONG, always. Real spin-period
   cadence drifts systematically: 99.98 % of a THEMIS day's sample gaps
   sit within 1 % of the median, and the residual still accumulates until
   the worst sample lands 91 cells (6.5 % of the panel) from where a
   uniform grid would put it -- and the spin period itself changes between
   days (3.8551 s on 2011-08-14, 4.0733 s on 2011-10-07). After
   :func:`rebin` the grid is uniform to 5.5e-12 days, i.e. float
   round-off, and ``imshow(extent=...)`` is exact. Never hand raw
   timestamps to ``imshow``.

2. THE DEFAULT AGGREGATOR IS ``max``, AND IT BRIGHTENS THE PANEL.
   See :data:`AGGREGATORS` below for the measured trade. In one line: a
   labeler must see rare events at their true magnitude, and ``max`` is
   the only aggregator that shows them there -- at a disclosed cost of
   about +0.30 dex of overall brightness against the within-bin median.

3. ``imshow`` IS AN AFFINE ARTIST: it maps its ``extent`` rectangle
   through the axes transform by its corners only, so a LOG y scale
   stretches the image linearly and puts every channel in the wrong
   place. For an energy axis, pass ``y_centers=np.log10(energies)`` and
   label the ticks -- ``examples/spectrogram_multipane.py`` shows the
   three lines that does.

THE COLORBAR. ``TimeIntervalLabeler._update_plot`` calls ``ax.clear()``
on every user panel every frame, which destroys the artist a colorbar was
built from; and a ``fig.colorbar(...)`` call inside ``plot_fn`` leaks one
axes per redraw while shrinking the image panel by ~19 % of its remaining
width each time, until its x axis is silently no longer the strip's x
axis. :func:`attach_colorbar` closes both: it is IDEMPOTENT per owner
axes (so calling it from inside ``plot_fn`` creates exactly one), it
steals space from the whole shared-x group -- every time panel AND the
Labels strip -- so the panels stay aligned, and it registers itself on
the owner axes so that ``_update_plot``'s sweep
(:func:`refresh_colorbars`) re-points it after each redraw with no
per-frame user code.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple

import numpy as np

#: The aggregators :func:`rebin` accepts, in the order they were measured.
#:
#: MEASURED ON WHAT A ``plot_fn`` ACTUALLY PASSES -- real THEMIS-B ESA
#: reduced-mode ion energy flux, one day (thb, 2011-08-14), 22,282 rows x
#: 32 channels, rebinned to 1,400 pixel columns (a median of 16 native
#: samples per column), NOT lifted to any display floor. Level drift is
#: the median of ``log10(cell) - log10(within-bin median)`` with both
#: sides floored at 1.0 eflux unit so that zero-flux cells count as
#: agreeing rather than dropping out; the burst column injects a 2.0-dex
#: burst 1 to 3 NATIVE columns wide and reports how much of its amplitude
#: survives against the neighbouring column under the same aggregator:
#:
#: ===========  ======================  ====================  ==========
#: aggregator   keeps a 1-3 column      median level drift    cost, that
#:              2-dex burst at          vs the within-bin     same day
#:                                      median                and panel
#: ===========  ======================  ====================  ==========
#: ``max``      **100 % of amplitude**  **+0.3010 dex**       1.5 ms
#: ``mean``     6-19 %                  +0.0105 dex           4.9 ms
#: ``logmean``  0.6-2 %                 -0.0047 dex           12.1 ms
#: ===========  ======================  ====================  ==========
#:
#: Cells landing within 0.1 dex of the within-bin median: ``max`` 25.8 %,
#: ``mean`` 86.0 %, ``logmean`` 85.2 %. The cost column is one coherent
#: measurement on the day above; only ``max``'s stays put as the window
#: grows (1.5 ms at 3,736 rows, 1.9 ms at 89,128, against 19 ms and
#: 43 ms for ``mean`` and ``logmean`` at 89,128).
#:
#: ``max`` is the default BY RULING: a labelling tool exists to let a
#: human see rare events, and only ``max`` shows one at its true value.
#: The cost is disclosed, not hidden -- ``max`` brightens the whole panel
#: by about a third of a decade and only a quarter of its cells land
#: within 0.1 dex of the within-bin median. Note also that this is a
#: change of CONVENTION, not a preservation of one: today's undecimated
#: ``pcolormesh``, blended by Agg at ~16 sub-pixel quads per pixel,
#: already shows a CENTRAL statistic (it matches the within-bin median to
#: 0.126 dex and ``max`` only to 0.311 dex).
#:
#: ZEROS AND NEGATIVES. 22.0 % of the raw samples on that real day are
#: exactly 0.0 -- normal for the high-energy channels of an ESA sweep.
#: ``max`` and ``mean`` pass zeros and negatives straight through;
#: ``logmean`` cannot take their logarithm, so it averages the positive
#: samples in a bin and reports **0.0** for a bin whose samples are all
#: non-positive (the geometric mean of a bin of zeros IS zero). That
#: matters because it keeps ``nan`` meaning exactly ONE thing out of
#: :func:`rebin`, for every aggregator: NO SAMPLE FELL IN THIS BIN.
#: Without it ``logmean`` returned ``nan`` for 8,538 of that panel's
#: 44,800 cells (19.1 %) against 160 (0.4 %) for ``max`` and ``mean`` --
#: 8,378 cells, 18.7 % of the panel, painted at the bottom of the scale
#: by two aggregators and transparent under the third, where a hole
#: reads as a data gap that is not there.
#:
#: ``aggregator="logmean"`` is the closest of the three to the central
#: statistic the old undecimated render happened to show, and it is the
#: only unbiased one; it is NOT a pixel-for-pixel reproduction of that
#: render, and on a burst it keeps under 2 % of the amplitude (+0.14 dex
#: of contrast on a 2-dex, one-column burst -- dimmed to nearly nothing).
#: ``aggregator="mean"`` is the cheap middle.
AGGREGATORS = ("max", "mean", "logmean")

#: What :func:`draw_spectrogram` bins to when the axes cannot tell it a
#: pixel width, i.e. when its bbox is still sub-pixel. Measured: this
#: does NOT happen inside the tool -- a real labeler panel already
#: reports 1342 px on its very first ``plot_fn`` call, and even a bare
#: ``fig.add_subplot(111)`` reports 465 -- so the fallback is for
#: hand-built axes with no laid-out width, and 1400 is the panel width
#: every measurement in this pack was taken at.
DEFAULT_N_COLS = 1400

#: A window whose first and last timestamp are equal has no width to lay
#: bins across. Rather than emit a zero-width image (invisible, and it
#: takes the axes limits with it) the edges span one second -- and the
#: window is drawn as exactly ONE column holding every sample in it,
#: because one instant is one column. Spreading it over the requested
#: n_cols would paint the last column and leave the other n-1 nan, which
#: reads as a panel that is almost all data gap.
_ZERO_SPAN_DAYS = 1.0 / 86400.0

# Attribute names used to register a colorbar and its source artist on
# the OWNER axes. Deliberately NOT the Pack 6 names: `colorbar.py`,
# `ensure_lane_colorbar` and `time_lane_cbar_gutter` are pinned absent by
# tests/test_pack6_cleanup.py and reviving them is not a silent option.
_CB_ATTR = "_ct_colorbar"
_SRC_ATTR = "_ct_colorbar_source"
#: Set on the colorbar AXES this module creates in a gutter column, so
#: the free-column scan does not count its own bars as occupants.
_GUTTER_ATTR = "_ct_colorbar_gutter"
#: Set on the OWNER axes when a gutter bar has just been created. The
#: cax is born inside plot_fn, i.e. AFTER the constrained-layout solver
#: has already placed the panels, so it keeps its raw gridspec cell until
#: something asks for one more solve. `_update_plot` consumes this flag
#: through `take_layout_dirty` and does exactly that, once.
_DIRTY_ATTR = "_ct_colorbar_layout_dirty"


# --------------------------------------------------------------- the core

def rebin(
    index: Any,
    values: Any,
    n_cols: int,
    aggregator: str = "max",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate ``values`` onto ``n_cols`` uniform time bins.

    PURE: no matplotlib artist is created and no axes is touched. Callers
    who want to keep their own ``pcolormesh`` use this and draw the result
    themselves; :func:`draw_spectrogram` is the batteries-included path.

    CONTRACT ON ``index``: it must be NON-DECREASING. Duplicated
    timestamps are fine (real THEMIS survey data contains them);
    out-of-order ones are refused, loudly, because binning them anyway
    puts the data in the wrong columns and silently disagrees with
    ``utils/decimate._bin_edges``, which falls back to equal-row bins on
    the same input. ``TimeIntervalLabeler`` sorts its frame at
    construction (``app.py:131``), so the tool never trips this; a direct
    caller with a two-spacecraft frame loaded without a re-sort will.

    Parameters
    ----------
    index :
        The window's time index -- a ``pandas.DatetimeIndex``,
        non-decreasing, length ``n``.
    values :
        ``(n_channels, n)`` array, channels down the rows. This is the
        transpose of a wide dataframe slice: ``df[cols].to_numpy().T``.
    n_cols :
        Requested bin count, normally the panel's pixel width. CLAMPED to
        ``1 <= n_cols <= n`` -- asking for more bins than there are
        samples would manufacture empty columns between real ones, which
        reads as data gaps that are not in the data -- and clamped to 1
        on a zero-span window (see ``_ZERO_SPAN_DAYS``).
    aggregator :
        One of :data:`AGGREGATORS`. See that constant for the measured
        trade-off, and for how each one treats zeros; ``max`` is the
        ruled default.

    Returns
    -------
    (edges, out) :
        ``edges`` is ``n_cols + 1`` matplotlib date numbers, uniform to
        float round-off -- which is what makes ``imshow`` legal on
        ``out``. ``out`` is ``(n_channels, n_cols)``; ``nan`` means one
        thing and one thing only, FOR EVERY AGGREGATOR: no sample fell in
        that bin. matplotlib renders those in the colormap's "bad"
        colour, transparent by default, exactly as a masked
        ``pcolormesh`` cell would be.

    Notes
    -----
    Bin membership is ``index.searchsorted`` against ``pd.date_range``
    edges -- the SAME construction ``utils/decimate.py:_bin_edges`` uses
    for its pixel columns, so the rebin grid and the decimation grid line
    up by design, and the monotonicity contract above is what keeps that
    true. NaN samples are ignored rather than propagated, so a gappy
    channel still yields the extreme (or the mean) of its real samples.
    """
    import matplotlib.dates as mdates
    import pandas as pd

    if aggregator not in AGGREGATORS:
        raise ValueError(
            "aggregator must be one of %s; got %r"
            % (", ".join(repr(a) for a in AGGREGATORS), aggregator))

    idx = index if hasattr(index, "searchsorted") else pd.Index(index)
    n = len(idx)
    if n == 0:
        raise ValueError("rebin needs a non-empty index")
    if n > 1:
        mono = getattr(idx, "is_monotonic_increasing", None)
        if mono is None:
            arr = np.asarray(idx)
            mono = bool(np.all(arr[1:] >= arr[:-1]))
        if not mono:
            raise ValueError(
                "rebin needs a non-decreasing index (duplicate timestamps "
                "are fine, out-of-order samples are not); this one runs "
                "%r .. %r and is not sorted. TimeIntervalLabeler sorts "
                "its frame at construction (app.py:131), so sort yours: "
                "df = df.sort_index(). Binning it anyway would put the "
                "data in the wrong columns and disagree silently with "
                "utils/decimate._bin_edges, which falls back to "
                "equal-row bins on this input." % (idx[0], idx[-1]))

    try:
        Z = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "rebin needs channel values it can read as numbers; numpy "
            "refused this array (%s). A dataframe slice that still "
            "carries a string, categorical or datetime column arrives "
            "here as object dtype -- select the channel columns first: "
            "df[channel_cols].to_numpy(dtype=float).T" % exc) from exc
    if Z.ndim != 2:
        raise ValueError("values must be 2-D (n_channels, n_samples); got "
                         "shape %r" % (Z.shape,))
    if Z.shape[1] != n:
        raise ValueError(
            "values has %d samples but the index has %d; values must be "
            "(n_channels, n_samples) -- did you forget the .T?"
            % (Z.shape[1], n))

    n_cols = int(n_cols)
    if n_cols < 1:
        raise ValueError("n_cols must be >= 1; got %d" % n_cols)
    n_cols = min(n_cols, n)
    if n > 1 and not idx[-1] > idx[0]:
        # Zero span: every timestamp in the window is the same instant,
        # so the window IS one column. See _ZERO_SPAN_DAYS.
        n_cols = 1

    t_edges = pd.date_range(idx[0], idx[-1], periods=n_cols + 1)
    starts = np.asarray(idx.searchsorted(t_edges[:-1], side="left"),
                        dtype=np.int64)
    ends = np.asarray(idx.searchsorted(t_edges[1:], side="left"),
                      dtype=np.int64)
    ends[-1] = n

    edges = np.asarray(mdates.date2num(t_edges.to_numpy()), dtype=float)
    if not edges[-1] > edges[0]:
        # Zero span again, now in date numbers: give the single column a
        # second of width so imshow has something to draw.
        edges = np.linspace(edges[0], edges[0] + _ZERO_SPAN_DAYS,
                            n_cols + 1)

    out = np.full((Z.shape[0], n_cols), np.nan, dtype=float)
    nz = np.flatnonzero(ends > starts)
    if nz.size:
        # reduceat over EQUAL successive starts returns the single element
        # at that position instead of a reduction; dropping empty bins
        # first is what keeps the start array strictly increasing.
        s = starts[nz]
        if aggregator == "max":
            red = np.fmax.reduceat(Z, s, axis=1)
        elif aggregator == "mean":
            valid = ~np.isnan(Z)
            num = np.add.reduceat(np.where(valid, Z, 0.0), s, axis=1)
            den = np.add.reduceat(valid.astype(float), s, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                red = np.where(den > 0, num / den, np.nan)
        else:                                     # "logmean"
            with np.errstate(invalid="ignore", divide="ignore"):
                L = np.log10(Z)
            valid = np.isfinite(L)
            num = np.add.reduceat(np.where(valid, L, 0.0), s, axis=1)
            den = np.add.reduceat(valid.astype(float), s, axis=1)
            # A bin can hold samples and still have nothing to average --
            # every one zero or negative. Its geometric mean is 0.0, not
            # "no data": see AGGREGATORS. nan is reserved for empty bins.
            seen = np.logical_or.reduceat(np.isfinite(Z), s, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                red = np.where(den > 0, 10.0 ** (num / den),
                               np.where(seen, 0.0, np.nan))
        out[:, nz] = red
    return edges, out


def edges_from_centers(centers: Any) -> np.ndarray:
    """
    Turn ``n`` channel centres into ``n + 1`` cell edges.

    GEOMETRIC midpoints when every centre is positive (an energy table is
    log-spaced, and the arithmetic midpoint of 1 keV and 10 keV is not
    where the eye expects the boundary), arithmetic midpoints otherwise.
    The two outer edges are extrapolated by reflecting the first and last
    interval.

    Every centre must be FINITE. One ``nan`` or ``inf`` left in an energy
    table would otherwise poison every edge in the array, because each
    interior edge is built from its two neighbours and the outer two are
    extrapolated from those; the result is a ``nan`` ``imshow`` extent
    and a panel that draws nothing. Refusing costs one line and says
    which table is broken. (A FINITE fill value -- the ``-1e31`` a CDF
    hands you -- cannot be told from data and is not caught; it will
    merely turn the table non-positive and get arithmetic midpoints.)
    """
    c = np.asarray(centers, dtype=float).ravel()
    if c.size == 0:
        raise ValueError("edges_from_centers needs at least one centre")
    bad = int(np.count_nonzero(~np.isfinite(c)))
    if bad:
        raise ValueError(
            "edges_from_centers needs finite centres; %d of %d are not "
            "(nan or inf). One fill value poisons EVERY edge -- each "
            "edge is built from its neighbours -- and a nan extent makes "
            "imshow draw nothing at all. Drop or mask the bad channels "
            "before you pass the table." % (bad, int(c.size)))
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5], dtype=float)
    logspace = bool(np.all(c > 0))
    v = np.log(c) if logspace else c
    e = np.empty(v.size + 1, dtype=float)
    e[1:-1] = 0.5 * (v[1:] + v[:-1])
    e[0] = v[0] - 0.5 * (v[1] - v[0])
    e[-1] = v[-1] + 0.5 * (v[-1] - v[-2])
    return np.exp(e) if logspace else e


# --------------------------------------------------------------- the draw

def draw_spectrogram(
    ax: Any,
    index: Any,
    values: Any,
    *,
    y_edges: Optional[Any] = None,
    y_centers: Optional[Any] = None,
    n_cols: Optional[int] = None,
    aggregator: str = "max",
    norm: Optional[Any] = None,
    cmap: Optional[Any] = None,
    **imshow_kwargs: Any
) -> Any:
    """
    Rebin ``values`` onto the panel's pixel columns and draw it with
    ``imshow``. Returns the ``AxesImage``.

    ALWAYS REBINS. There is no native-draw fallback below the density at
    which rebinning stops paying for itself (~4 native rows per pixel
    column, where the measured penalty is 22 ms on a 3,736-row window):
    one consistent aggregator across every zoom level is worth more to a
    labeller than 22 ms, because an aggregator that silently changes when
    you zoom in is an aggregator you cannot reason about. When the window
    holds fewer rows than the panel is wide, :func:`rebin` clamps the bin
    count to the row count, so a zoomed-in window is drawn one column per
    sample -- never coarser than the data.

    Parameters
    ----------
    ax :
        The image panel. It is NOT cleared here: ``_update_plot`` already
        cleared it before calling ``plot_fn``.
    index, values, aggregator :
        As :func:`rebin` -- including its non-decreasing index contract.
    y_edges :
        ``n_channels + 1`` cell edges. Mutually exclusive with
        ``y_centers``.
    y_centers :
        ``n_channels`` channel centres -- an energy table, typically --
        converted with :func:`edges_from_centers`. REMEMBER (3) in the
        module docstring: for a log-spaced table pass
        ``np.log10(energies)``, not the energies, and format the ticks;
        ``imshow`` cannot draw on a log axis.
    n_cols :
        Bin count. Defaults to the axes' pixel width, or
        :data:`DEFAULT_N_COLS` when the panel's bbox is still sub-pixel
        because the figure has not been laid out.
    norm, cmap :
        Passed to ``imshow``. For energy flux the house convention is
        ``LogNorm(1e3, 1e8)`` with ``cmap="jet"``.
    **imshow_kwargs :
        Anything else ``imshow`` takes. ``origin``, ``aspect``,
        ``interpolation`` and ``extent`` are set here and may not be
        overridden -- they are what make the picture correct.
    """
    if y_edges is not None and y_centers is not None:
        raise ValueError("pass y_edges or y_centers, not both")
    for reserved in ("origin", "aspect", "interpolation", "extent"):
        if reserved in imshow_kwargs:
            raise ValueError(
                "%r is set by draw_spectrogram and may not be overridden"
                % reserved)

    if n_cols is None:
        try:
            width = int(ax.bbox.width)
        except Exception:
            width = 0
        n_cols = width if width > 1 else DEFAULT_N_COLS

    edges, Zr = rebin(index, values, n_cols, aggregator=aggregator)
    n_ch = Zr.shape[0]

    if y_centers is not None:
        y = np.asarray(edges_from_centers(y_centers), dtype=float)
    elif y_edges is not None:
        y = np.asarray(y_edges, dtype=float).ravel()
    else:
        y = np.arange(n_ch + 1, dtype=float)
    if y.size != n_ch + 1:
        raise ValueError(
            "y edges must have n_channels + 1 = %d entries; got %d"
            % (n_ch + 1, y.size))

    im = ax.imshow(
        Zr,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(edges[0], edges[-1], y[0], y[-1]),
        norm=norm,
        cmap=cmap,
        **imshow_kwargs
    )
    # The sweep reads this to re-point a registered colorbar; see
    # refresh_colorbars.
    setattr(ax, _SRC_ATTR, im)
    return im


# ----------------------------------------------------------- the colorbar

def shared_x_axes(ax: Any) -> List[Any]:
    """
    Every axes sharing ``ax``'s x axis, ``ax`` included.

    On a labeler pane that is exactly "the time panels plus the Labels
    strip" -- ``canvas.py`` shares x from the primary time axis to every
    other time axis AND to the strip, and to nothing else. That set is
    precisely what a colorbar must steal width from: take it from the time
    panels only and the strip keeps its old width, the x axes stop being
    the same axis, and a box drawn on the image no longer means the time
    it appears to mean.
    """
    try:
        sibs = list(ax.get_shared_x_axes().get_siblings(ax))
    except Exception:
        sibs = []
    if not sibs:
        sibs = [ax]
    if ax not in sibs:
        sibs.append(ax)
    return sibs


def current_mappable(ax: Any) -> Optional[Any]:
    """
    The live colour-mapped artist on ``ax``, or None.

    Prefers the artist :func:`draw_spectrogram` registered, but only while
    it is still attached: ``ax.clear()`` detaches an artist from
    ``ax.get_children()`` (measured -- and sets its ``.axes`` and
    ``.figure`` to None), so a stale registration is recognised as stale.
    Falls back to the last image, then the last colour-mapped collection,
    which is what makes the sweep work for a caller who used :func:`rebin`
    and drew their own ``pcolormesh``.
    """
    art = getattr(ax, _SRC_ATTR, None)
    if art is not None:
        try:
            if art in ax.get_children():
                return art
        except Exception:
            pass
    for seq in (getattr(ax, "images", ()), getattr(ax, "collections", ())):
        for candidate in reversed(list(seq)):
            try:
                if candidate.get_array() is not None:
                    return candidate
            except Exception:
                continue
    return None


def gutter_column(owner_ax: Any) -> Optional[int]:
    """
    The gridspec column a per-owner colorbar for ``owner_ax`` may use, or
    ``None`` if the pane reserved no room for one.

    A GUTTER is a column of the pane's own gridspec that NO panel
    occupies -- ``layout_spec`` has accepted ``ncols`` and
    ``width_ratios`` all along, so reserving one is two keys and no new
    schema::

        "ncols": 3,
        "width_ratios": [1.0, 0.05, 1.0],     # column 1 is the gutter
        "areas": [ ... col 0 and col 2 only ... ]

    The answer is the FIRST free column to the right of the owner, so a
    bar lands beside the panel it belongs to rather than out past
    whatever else the pane holds. Bars this module has already built in a
    gutter do not count as occupants; every other axes on the same
    gridspec does, including the Labels strip.

    Returns ``None`` for a pane with no free column -- which is not an
    error, it is the shape :func:`attach_colorbar` falls back to the
    shared-x group for.
    """
    try:
        ss = owner_ax.get_subplotspec()
        if ss is None:
            return None
        gs = ss.get_gridspec()
        ncols = int(gs.ncols)
        first_free = int(ss.colspan.stop)
    except Exception:
        return None
    if first_free >= ncols:
        return None

    occupied = set()
    try:
        siblings = list(getattr(owner_ax.get_figure(), "axes", ()))
    except Exception:
        return None
    for other in siblings:
        if getattr(other, _GUTTER_ATTR, False):
            continue
        try:
            oss = other.get_subplotspec()
            if oss is None or oss.get_gridspec() is not gs:
                continue
            occupied.update(range(int(oss.colspan.start),
                                  int(oss.colspan.stop)))
        except Exception:
            continue

    for col in range(first_free, ncols):
        if col not in occupied:
            return col
    return None


def _make_gutter_axes(owner_ax: Any, col: int) -> Any:
    """A real subplot in ``col``, spanning exactly the owner's rows."""
    ss = owner_ax.get_subplotspec()
    gs = ss.get_gridspec()
    rows = ss.rowspan
    cax = owner_ax.get_figure().add_subplot(
        gs[int(rows.start):int(rows.stop), int(col)])
    setattr(cax, _GUTTER_ATTR, True)
    return cax


def take_layout_dirty(axes: Iterable[Any]) -> bool:
    """
    Consume the "a gutter bar was just born" flag; True at most once per
    bar.

    ``attach_colorbar`` is called from inside ``plot_fn``, which runs
    AFTER the constrained-layout solver has placed the panels, so a cax
    created there keeps its raw gridspec cell position and Pack 5's
    layout freeze then locks that in -- measured, a bar at
    ``[0.8635, 0.6316, 0.9, 0.88]`` against an owner at
    ``[0.0256, 0.6935, 0.9369, 0.9879]``. One more solve lands it exactly
    on the owner's rows and it never moves again (redraw drift 0.0, pan
    drift 0.0, resize round trip 0.000000). ``_update_plot`` asks for
    that solve when this returns True.
    """
    dirty = False
    for ax in axes:
        if getattr(ax, _DIRTY_ATTR, False):
            dirty = True
            try:
                delattr(ax, _DIRTY_ATTR)
            except Exception:
                try:
                    setattr(ax, _DIRTY_ATTR, False)
                except Exception:
                    pass
    return dirty


def attach_colorbar(
    owner_ax: Any,
    mappable: Optional[Any] = None,
    *,
    axes: Optional[Iterable[Any]] = None,
    gutter: Any = None,
    label: Optional[str] = None,
    fraction: float = 0.04,
    pad: float = 0.01,
    **colorbar_kwargs: Any
) -> Any:
    """
    Create ONE colorbar for ``owner_ax``, or re-point the one it has.

    IDEMPOTENT BY OWNER AXES, which is the whole point: it is safe -- and
    intended -- to call this from inside ``plot_fn``, where the mappable
    you just drew is in hand. The first call creates the colorbar; every
    later call re-points it and returns the same object, so the figure
    never grows an axes and the panels never lose width.

    What the naive alternative does, measured on this tree through a real
    ``_update_plot`` on a pane of THREE time panels plus the Labels strip
    at 1400x800: a bare ``fig.colorbar(mesh, ax=ax)`` inside ``plot_fn``
    adds one axes per redraw and takes ~19 % of the image panel's
    REMAINING width each time -- x1 = 0.7927, 0.6393, 0.5166, 0.4184,
    0.3398, 0.2770 over six frames, while the line panels and the strip
    stay at 0.9845, so the x axes silently stop being the same axis. The
    constants are layout-specific; the shape (about a fifth of what is
    left, every frame, forever) is not.

    A colorbar that has been ``remove()``d is treated as gone: the dead
    registration is dropped and the next call builds a new bar, rather
    than handing back a colorbar that is no longer in the figure.

    WHERE THE BAR GOES (Pack 8.5-B B4). Two placements, and the pane's
    own layout picks between them:

    * **GUTTER (preferred).** If the pane reserved a gridspec column that
      no panel occupies -- ``"ncols": 3, "width_ratios": [1.0, 0.05,
      1.0]`` with areas in columns 0 and 2 -- the bar is built there as a
      real subplot spanning exactly the owner's rows. Measured over five
      redraws, a pan and a 14x8 -> 10x6 -> 14x8 resize: bar height /
      owner height **1.0000** at every stage, position drift
      **0.000000** across the resize round trip, and the panels' width
      FLAT in the number of bars (0.8741 of the figure at N = 1, 2 and 3
      spectrograms).
    * **SHARED-X STEAL (fallback).** A pane with no free column keeps
      what Pack 8.5 shipped: ``fig.colorbar(mappable, ax=shared_x_axes)``.
      Every x axis stays identical, which is the point, but the bar is
      laid out against the WHOLE group -- measured 2.118 / 3.338 / 4.666
      times its own panel's height at N = 1 / 2 / 3, and each extra bar
      costs the panels another ~4.5 % of the figure width. Reserve a
      gutter column to get out of it.

    Parameters
    ----------
    owner_ax :
        The image panel the bar belongs to. The colorbar is registered on
        it, and ``_update_plot``'s sweep re-points it from there.
    mappable :
        The artist to build from. Defaults to :func:`current_mappable` of
        ``owner_ax``.
    axes :
        Which axes give up width IN THE FALLBACK PLACEMENT. Passing it
        also SELECTS the fallback, because asking for particular axes to
        be charged is asking for the steal. Defaults to
        :func:`shared_x_axes` of ``owner_ax`` -- the time panels and the
        Labels strip -- which is the only choice that keeps every x axis
        identical. An empty list, or axes belonging to another figure, is
        refused.
    gutter :
        Which gridspec column to build the bar in. ``None`` (the default)
        means :func:`gutter_column` decides; an ``int`` names a column
        explicitly; ``False`` forces the shared-x steal even on a pane
        that has a free column.
    label :
        Colorbar label, e.g. ``"ion eflux (eV/cm^2-s-sr-eV)"``.
    fraction, pad :
        Passed to ``fig.colorbar`` IN THE FALLBACK PLACEMENT only -- in a
        gutter the width is stated by the layout's ``width_ratios``. The
        defaults are narrower than matplotlib's (0.15 / 0.05), which on a
        stacked time pane cost ~17-19 % of the panel width against ~4 %
        for these.
    """
    fig = owner_ax.get_figure()

    cb = getattr(owner_ax, _CB_ATTR, None)
    if cb is not None and getattr(cb, "ax", None) is not None:
        try:
            live = cb.ax in list(getattr(fig, "axes", ()))
        except Exception:
            live = True
        if not live:
            # Somebody called cb.remove(). Handing the dead bar back
            # would cost this panel its colour scale permanently and
            # silently, so forget it and build a new one.
            cb = None
            try:
                delattr(owner_ax, _CB_ATTR)
            except Exception:
                pass

    if mappable is None:
        mappable = current_mappable(owner_ax)
    if cb is not None:
        if mappable is not None and mappable is not cb.mappable:
            cb.update_normal(mappable)
        return cb
    if mappable is None:
        raise ValueError(
            "attach_colorbar found no colour-mapped artist on this axes; "
            "draw the image first, or pass mappable=")

    # Pack 8.5-B B4: a bar belongs to ONE panel and must span ONE panel.
    # If the pane reserved a free gridspec column, build the bar there as
    # a real subplot over the owner's rows -- measured bar-height /
    # owner-height 1.000 at every stage of a five-redraw + pan + resize
    # round trip, against 2.12 / 3.34 / 4.67 for the shared-group steal
    # this replaces at N = 1 / 2 / 3 spectrograms. An explicit `axes=`
    # still means "steal from these", because that is what asking for it
    # says.
    if gutter is None and axes is None:
        gutter = gutter_column(owner_ax)
    if gutter is not None and gutter is not False:
        cax = _make_gutter_axes(owner_ax, int(gutter))
        cb = fig.colorbar(mappable, cax=cax, **colorbar_kwargs)
        if label:
            cb.set_label(label)
        setattr(owner_ax, _CB_ATTR, cb)
        # The cax was born after the solver ran; ask _update_plot for one
        # more solve (take_layout_dirty).
        setattr(owner_ax, _DIRTY_ATTR, True)
        return cb

    if axes is None:
        parents = shared_x_axes(owner_ax)
    else:
        parents = list(axes)
        if not parents:
            raise ValueError(
                "attach_colorbar needs at least one axes to take the "
                "colorbar's width from; axes=[] gives matplotlib nothing "
                "to steal from and it fails with a bare IndexError from "
                "make_axes. Omit axes= to use the owner's shared-x "
                "group.")
        outside = [a for a in parents if a.get_figure() is not fig]
        if outside:
            raise ValueError(
                "attach_colorbar was given %d of %d axes from a "
                "DIFFERENT figure. The bar would be built over there "
                "while registered here, so _update_plot's sweep would "
                "keep re-pointing a colorbar nobody can see. Pass axes "
                "from the owner's own figure."
                % (len(outside), len(parents)))

    cb = fig.colorbar(mappable, ax=parents, fraction=fraction, pad=pad,
                      **colorbar_kwargs)
    if label:
        cb.set_label(label)
    setattr(owner_ax, _CB_ATTR, cb)
    return cb


def registered_colorbar(ax: Any) -> Optional[Any]:
    """The colorbar :func:`attach_colorbar` registered on ``ax``, or None."""
    return getattr(ax, _CB_ATTR, None)


def refresh_colorbars(axes: Iterable[Any]) -> int:
    """
    Re-point every registered colorbar at the artist its owner now holds.
    Returns how many were re-pointed.

    ``_update_plot`` calls this once per frame, after ``plot_fn`` has
    drawn and before ``canvas.draw()``. A NO-OP when nothing is
    registered, which is every pane that has never called
    :func:`attach_colorbar`.

    The problem it solves: ``_update_plot`` clears every user panel, which
    destroys the artist the colorbar was built from. The colorbar, its
    axes and its geometry all survive that -- measured, byte-identical
    position across redraws, a pan, a zoom and a re-solve -- but its
    ``mappable`` is a dead artist, so any change of norm or colormap stops
    reaching the bar. One ``update_normal`` per frame closes it, and it
    costs nothing when the artist is unchanged.
    """
    n = 0
    for ax in axes:
        cb = getattr(ax, _CB_ATTR, None)
        if cb is None:
            continue
        art = current_mappable(ax)
        if art is None or art is cb.mappable:
            continue
        cb.update_normal(art)
        n += 1
    return n
