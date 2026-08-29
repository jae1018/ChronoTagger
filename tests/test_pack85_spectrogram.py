"""
Pack 8.5 (spectrogram) regression tests.

These pin BEHAVIOUR, never timing -- Pack 5's rule, and the reason this
pack's speed claims live in `edit_pack/evidence/` benchmark scripts and
not in an assert that flakes on a loaded machine.

What each group owns:
  rebin core        -- values, edges, all three aggregators, the clamp,
                       the index contract, what `nan` is allowed to mean
  burst survival    -- SP-R1's headline: max keeps a sub-pixel burst whole
  alignment         -- imshow on the rebinned grid lands where pcolormesh
                       lands, proven by rasterising both (not by eye)
  draw helper       -- always rebins, sets the geometry it promises,
                       registers its artist, refuses the kwargs that
                       would break the picture
  colorbar          -- the A/B: what a bare fig.colorbar does inside
                       plot_fn today, and that the helper does not do it;
                       plus its refusals, its width and its label
  the sweep         -- SP-R5 re-points across real redraws, never hands
                       back a dead artist, and is a no-op on a pane that
                       never asked for a colorbar
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from chronotagger.labeler.utils.spectrogram import (
    AGGREGATORS,
    DEFAULT_N_COLS,
    attach_colorbar,
    current_mappable,
    draw_spectrogram,
    edges_from_centers,
    rebin,
    refresh_colorbars,
    registered_colorbar,
    shared_x_axes,
    _CB_ATTR,
    _SRC_ATTR,
)


@pytest.fixture(autouse=True)
def _stub_messagebox(monkeypatch):
    """STANDING RULE (Pack 3): any dialog-reachable path, real-Tk or not --
    tkinter.messagebox creates its own root on demand."""
    import tkinter.messagebox as mb
    calls = []
    for kind in ("showinfo", "showwarning", "showerror", "askyesno",
                 "askyesnocancel"):
        monkeypatch.setattr(
            mb, kind, lambda *a, _k=kind, **kw: calls.append(_k) or True)
    return calls


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------- fixtures

N_CH = 8
CH = ["C%d" % i for i in range(N_CH)]
VMIN, VMAX = 1e3, 1e8

#: The gather's burst protocol, reused verbatim in shape
#: (`pack_spec_g1_probe_4_maxmean.py`, item 3d B): a burst 1, 2 or 3
#: NATIVE columns wide, 2.0 dex above the local background, injected in
#: the middle energy channels at the CENTRE of one pixel column so that
#: no aggregator gets an edge advantage.
BURST_DEX = 2.0


@pytest.fixture
def spec_index():
    """4,000 samples at a real ESA reduced-mode cadence."""
    return pd.date_range("2011-08-14", periods=4000, freq="3851ms")


@pytest.fixture
def spec_values(spec_index):
    """(8, 4000) of smooth, positive, channel-ordered energy flux."""
    n = len(spec_index)
    base = np.logspace(4.0, 6.0, N_CH)[:, None]
    ripple = 1.0 + 0.25 * np.sin(np.linspace(0, 12, n))[None, :]
    return base * ripple


def _spec_frame(idx, Z):
    data = {"ion_n": np.linspace(0.1, 3.0, len(idx)),
            "Bx": np.cos(np.linspace(0, 20, len(idx)))}
    for i, c in enumerate(CH):
        data[c] = Z[i]
    return pd.DataFrame(data, index=idx)


LAYOUT = {
    "nrows": 4, "ncols": 1,
    "areas": [
        {"key": "spectrogram", "row": 0, "col": 0, "role": "time"},
        {"key": "ion_n", "row": 1, "col": 0, "role": "time"},
        {"key": "Bx", "row": 2, "col": 0, "role": "time"},
        {"key": "labels", "row": 3, "col": 0, "role": "labels"},
    ],
}


def _make_labeler(df, plot_fn, tmp_path):
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(
        df=df, plot_fn=plot_fn, layout_spec=LAYOUT,
        classes=["UNKNOWN", "PS"], decimate=False,
        window=pd.Timedelta("1h"), step=pd.Timedelta("30min"),
        autosave_folder=str(tmp_path),
    )
    lbl._build_gui()
    lbl.root.withdraw()
    lbl.fig.set_size_inches(14, 8, forward=False)
    return lbl


def _xspan(ax):
    p = ax.get_position()
    return (round(p.x0, 5), round(p.x1, 5))


# ---------------------------------------------------------- the rebin core

def test_rebin_returns_uniform_edges_spanning_the_window(spec_index,
                                                         spec_values):
    """The edges are what makes imshow legal: uniform to float round-off.

    A REAL spectra time axis is not uniform -- 99.98 % of a THEMIS day's
    gaps sit within 1 % of the median and the residual still accumulates
    to 91 cells of error over one day -- so the rebin has to MANUFACTURE
    the regularity, not assume it."""
    import matplotlib.dates as mdates

    edges, out = rebin(spec_index, spec_values, 200)
    assert edges.shape == (201,)
    assert out.shape == (N_CH, 200)
    assert edges[0] == pytest.approx(mdates.date2num(spec_index[0]))
    assert edges[-1] == pytest.approx(mdates.date2num(spec_index[-1]))
    spacing = np.diff(edges)
    assert spacing.min() > 0
    # Measured in the gather on a real THEMIS day at 1400 columns: the
    # edge spacing spread is 5.457e-12 DAYS, i.e. float round-off. The
    # bound here is 1e-9 days (86 microseconds) -- three orders above the
    # round-off and six below any real cadence drift, so it passes on a
    # rebinned grid and fails on a native one.
    assert (spacing.max() - spacing.min()) < 1e-9


@pytest.mark.parametrize("aggregator", list(AGGREGATORS))
def test_rebin_computes_the_aggregate_it_names(aggregator):
    """Hand-computable: 3 channels x 12 samples into 3 bins of 4."""
    idx = pd.date_range("2020-01-01", periods=12, freq="1min")
    Z = np.array([
        [1.0, 10.0, 100.0, 1000.0] * 3,
        [2.0, 2.0, 2.0, 2.0] * 3,
        [1e3, 1e4, 1e5, 1e6] * 3,
    ])
    _, out = rebin(idx, Z, 3, aggregator=aggregator)
    assert out.shape == (3, 3)
    if aggregator == "max":
        want_first = [1000.0, 2.0, 1e6]
    elif aggregator == "mean":
        want_first = [(1 + 10 + 100 + 1000) / 4.0, 2.0,
                      (1e3 + 1e4 + 1e5 + 1e6) / 4.0]
    else:
        want_first = [10.0 ** 1.5, 2.0, 10.0 ** 4.5]
    # every bin holds the same 4 values, so every column is the same
    for col in range(3):
        assert out[:, col] == pytest.approx(want_first, rel=1e-9)


def test_rebin_ignores_nan_samples_and_leaves_empty_bins_nan():
    """A gappy channel still yields the extreme of its REAL samples, and a
    bin with no sample at all is nan -- which matplotlib renders in the
    colormap's `bad` colour, exactly as a masked pcolormesh cell."""
    left = pd.date_range("2020-01-01 00:00", periods=6, freq="1min")
    right = pd.date_range("2020-01-01 02:00", periods=6, freq="1min")
    idx = left.append(right)
    Z = np.ones((2, 12)) * 5.0
    Z[0, 0] = np.nan
    Z[0, 1] = 50.0
    edges, out = rebin(idx, Z, 24)
    assert np.isnan(out).any(), "a two-hour gap must leave empty bins"
    assert out[0, 0] == 50.0            # the NaN neighbour did not win
    assert not np.isnan(out[1, 0])


def test_nan_out_of_rebin_means_no_sample_and_nothing_else():
    """The invariant the three aggregators are only useful if they share.

    22.0 % of a real THEMIS day's eflux samples are exactly 0.0 -- normal
    for the high-energy channels of an ESA sweep. `max` and `mean` paint
    those bins at the bottom of the scale. Before this was pinned,
    `logmean` could not take log10(0) and returned nan for them, which
    matplotlib draws as a HOLE: on that real day 8,378 cells -- 18.7 % of
    the panel -- were painted under `max` and transparent under
    `logmean`, and a hole reads as a data gap that is not there.

    So: a bin that HELD samples always yields a number, whatever the
    aggregator; the geometric mean of a bin of zeros is zero. `nan` is
    reserved for a bin no sample fell in, and the nan MASKS of the three
    aggregators are identical."""
    idx = pd.date_range("2020-01-01", periods=12, freq="1min")
    Z = np.array([
        [0.0] * 12,                     # a channel that measured zero
        [0.0, 0.0, 100.0, 0.0] * 3,     # one positive sample per bin
        [np.nan] * 12,                  # genuinely no data
        [-5.0, -3.0, -1.0, -2.0] * 3,   # negative calibration artefacts
    ])
    got = {}
    for agg in AGGREGATORS:
        _, out = rebin(idx, Z, 3, aggregator=agg)
        got[agg] = out
        assert out[0, 0] == 0.0, (
            "%s turned a measured zero into %r" % (agg, out[0, 0]))
        assert np.isnan(out[2, 0]), "an all-nan bin is genuinely empty"
        assert not np.isnan(out[3, 0]), (
            "%s dropped a bin of negative samples" % agg)
    assert got["logmean"][1, 0] == pytest.approx(100.0), (
        "logmean averages the samples it CAN log, and 100 is the only one")
    ref = np.isnan(got["max"])
    for agg in AGGREGATORS:
        assert np.array_equal(np.isnan(got[agg]), ref), (
            "%s puts its holes somewhere else than max does" % agg)

    # and an EMPTY bin is nan for all three, identically
    left = pd.date_range("2020-01-01 00:00", periods=6, freq="1min")
    right = pd.date_range("2020-01-01 02:00", periods=6, freq="1min")
    gap = left.append(right)
    holes = set()
    for agg in AGGREGATORS:
        _, out = rebin(gap, np.ones((1, 12)), 24, aggregator=agg)
        holes.add(int(np.count_nonzero(np.isnan(out))))
    assert len(holes) == 1 and holes.pop() > 0


def test_rebin_clamps_the_bin_count_to_the_row_count(spec_index,
                                                     spec_values):
    """SP-R3's degenerate clamp. Asking for more bins than there are
    samples would manufacture empty columns between real ones, which
    reads as data gaps that are not in the data."""
    idx = spec_index[:37]
    Z = spec_values[:, :37]
    edges, out = rebin(idx, Z, 1400)
    assert out.shape == (N_CH, 37)
    assert edges.shape == (38,)
    assert not np.isnan(out).any()


def test_rebin_refuses_what_it_cannot_do(spec_index, spec_values):
    with pytest.raises(ValueError, match="aggregator"):
        rebin(spec_index, spec_values, 100, aggregator="median")
    with pytest.raises(ValueError, match="did you forget the .T"):
        rebin(spec_index, spec_values.T, 100)
    with pytest.raises(ValueError, match="2-D"):
        rebin(spec_index, spec_values[0], 100)
    with pytest.raises(ValueError, match="non-empty"):
        rebin(spec_index[:0], spec_values[:, :0], 100)
    with pytest.raises(ValueError, match="n_cols"):
        rebin(spec_index, spec_values, 0)


@pytest.mark.parametrize("shape", ["descending", "shuffled",
                                   "shuffled_but_endpoints_in_order"])
def test_rebin_refuses_an_index_that_is_not_sorted(spec_index, spec_values,
                                                   shape):
    """The silent-damage case, refused.

    `rebin` copies `utils/decimate.py:_bin_edges`' bin construction, and
    that function GUARDS on monotonicity and falls back to equal-row bins
    when it fails -- a two-spacecraft frame loaded without a re-sort is
    the scenario its own comment names. Copying the construction without
    the guard means the two grids silently stop agreeing: measured on the
    unguarded code, a shuffled index put 9 of 10 columns' worth of data
    into the last column, and a DESCENDING one made `edges[-1] > edges[0]`
    false so the zero-span branch fired and drew the whole window across
    one second, with no warning of any kind.

    The third case is the one an endpoint check would miss: only the
    MIDDLE is scrambled, so first < last still holds.

    Refusing rather than sorting is deliberate -- `rebin` is a pure core
    that must not silently reorder the caller's array -- and the scan
    costs 0.11 ms at 44,564 rows against the rebin's own 1.7 ms."""
    rng = np.random.RandomState(0)
    if shape == "descending":
        idx = spec_index[::-1]
    elif shape == "shuffled":
        idx = spec_index[rng.permutation(len(spec_index))]
    else:
        mid = list(spec_index[1:-1])
        rng.shuffle(mid)
        idx = pd.DatetimeIndex([spec_index[0]] + mid + [spec_index[-1]])
        assert idx[0] < idx[-1], "this case must survive an endpoint check"
    with pytest.raises(ValueError, match="non-decreasing"):
        rebin(idx, spec_values, 10)
    # and it refuses BEFORE any artist exists, through the draw helper
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    with pytest.raises(ValueError, match="non-decreasing"):
        draw_spectrogram(ax, idx, spec_values, n_cols=10)
    assert not ax.images, "a refused draw may not leave half a picture"


def test_rebin_accepts_duplicate_timestamps():
    """SP-R9's crash is NOT inherited, and the monotonicity contract does
    not accidentally forbid it: real THEMIS ESA survey data carries
    duplicated timestamps (1 in 22,282 for thb_20110814), and a
    non-DECREASING index is what `rebin` asks for, not a strictly
    increasing one."""
    idx = pd.DatetimeIndex(
        list(pd.date_range("2020-01-01", periods=6, freq="1min"))
        + [pd.Timestamp("2020-01-01 00:02:00")]).sort_values()
    assert idx.has_duplicates and idx.is_monotonic_increasing
    Z = np.arange(2 * 7, dtype=float).reshape(2, 7)
    edges, out = rebin(idx, Z, 3)
    assert out.shape == (2, 3)
    assert not np.isnan(out).any()
    assert out[0].tolist() == [1.0, 4.0, 6.0]


def test_rebin_refuses_channels_it_cannot_read_as_numbers(spec_index):
    """A wide slice that still carries a label column arrives as object
    dtype. numpy's own message for that is `could not convert string to
    float: 'a'`, which names neither the argument nor the fix."""
    obj = np.array([["a", "b"], ["c", "d"]], dtype=object)
    with pytest.raises(ValueError, match="read as numbers"):
        rebin(spec_index[:2], obj, 2)
    mixed = np.array([[1.0, "b"], [3.0, 4.0]], dtype=object)
    with pytest.raises(ValueError, match="channel columns first"):
        rebin(spec_index[:2], mixed, 2)


def test_a_zero_span_window_is_drawn_as_one_column():
    """F9. A window whose timestamps are all one instant has no width to
    lay bins across, and an imshow with a zero-width extent draws nothing
    AND takes the axis limits with it. So the edges get one second --
    and the window gets exactly ONE column, holding every sample.

    The one-column half matters as much as the width: spreading one
    instant over the requested n_cols painted the LAST column and left
    the other n-1 nan, i.e. a panel that reads as almost entirely data
    gap. One instant is one column."""
    idx = pd.DatetimeIndex(["2020-01-01"] * 6)
    Z = np.arange(2 * 6, dtype=float).reshape(2, 6)
    edges, out = rebin(idx, Z, 400)
    assert out.shape == (2, 1)
    assert edges.shape == (2,)
    assert (edges[-1] - edges[0]) * 86400.0 == pytest.approx(1.0)
    assert not np.isnan(out).any(), "one instant is one PAINTED column"
    assert out[:, 0].tolist() == [5.0, 11.0], "max of every sample"
    _, mean_out = rebin(idx, Z, 400, aggregator="mean")
    assert mean_out[:, 0].tolist() == [2.5, 8.5]

    # a single row is the same shape of thing
    edges1, out1 = rebin(pd.DatetimeIndex(["2020-01-01"]), np.ones((3, 1)),
                         10)
    assert out1.shape == (3, 1)
    assert (edges1[-1] - edges1[0]) * 86400.0 == pytest.approx(1.0)

    # and through the draw helper the extent is one second wide
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    im = draw_spectrogram(ax, idx, Z, n_cols=400)
    x0, x1, _, _ = im.get_extent()
    assert (x1 - x0) * 86400.0 == pytest.approx(1.0)
    assert im.get_array().shape == (2, 1)


def test_edges_from_centers_is_geometric_for_a_positive_table():
    """An energy table is log-spaced: the boundary between 1 keV and
    10 keV belongs at 3.16 keV, not 5.5 keV."""
    e = edges_from_centers([1.0, 10.0, 100.0])
    assert e.shape == (4,)
    assert e[1] == pytest.approx(10.0 ** 0.5)
    assert e[2] == pytest.approx(10.0 ** 1.5)
    lin = edges_from_centers([-2.0, 0.0, 2.0])
    assert lin == pytest.approx([-3.0, -1.0, 1.0, 3.0])


def test_edges_from_centers_refuses_a_non_finite_table():
    """ONE nan or inf centre poisons EVERY edge -- each interior edge is
    built from its two neighbours and the outer two are extrapolated from
    those -- and a nan extent makes imshow draw nothing at all. A fill
    value in an energy table is an ordinary space-physics accident, so
    this refuses instead of returning an all-nan array."""
    with pytest.raises(ValueError, match="finite centres"):
        edges_from_centers([1.0, np.nan, 100.0])
    with pytest.raises(ValueError, match="finite centres"):
        edges_from_centers([1.0, np.inf, 100.0])
    with pytest.raises(ValueError, match="at least one centre"):
        edges_from_centers([])


# -------------------------------------------------------- burst survival

@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("channel", [0, 3, N_CH - 1])
def test_a_sub_pixel_burst_survives_max_rebin_at_full_amplitude(
        spec_index, spec_values, width, channel):
    """SP-R1's headline guarantee, in the shape of Pack 5's spike-survival
    pin, on the gather's protocol: a burst 1-3 NATIVE columns wide and
    2.0 dex above the local background, injected at the CENTRE of one
    pixel column so no aggregator gets an edge advantage.

    Measured on real unclipped eflux at 16 native rows per pixel column:
    max keeps 100.00 % of the amplitude at every width, mean keeps
    6.3-18.8 % and logmean 0.6-2.0 %. This asserts the ruling's half of
    that -- max keeps ALL of it -- and that the others do not, so the pin
    fails if the default ever quietly becomes an averaging aggregator."""
    n_cols = 250
    Z = spec_values.copy()
    edges, plain = rebin(spec_index, Z, n_cols)
    # the pixel column containing the middle sample, and its centre row
    per_col = len(spec_index) / float(n_cols)
    col = n_cols // 2
    row0 = int((col + 0.5) * per_col)
    background = float(np.median(Z[channel]))
    amplitude = background * 10.0 ** BURST_DEX
    Z[channel, row0:row0 + width] = amplitude

    _, hot = rebin(spec_index, Z, n_cols)
    assert hot[channel, col] == pytest.approx(amplitude, rel=1e-12), (
        "max must show the burst at its TRUE value")
    assert hot[channel, col] / amplitude == pytest.approx(1.0, rel=1e-12)
    # and it is confined to its own column: the neighbours are untouched
    assert hot[channel, col - 1] == pytest.approx(plain[channel, col - 1])
    assert hot[channel, col + 1] == pytest.approx(plain[channel, col + 1])

    _, mean_r = rebin(spec_index, Z, n_cols, aggregator="mean")
    _, log_r = rebin(spec_index, Z, n_cols, aggregator="logmean")
    assert mean_r[channel, col] < 0.5 * amplitude
    assert log_r[channel, col] < mean_r[channel, col]


def test_the_measured_cost_of_the_max_default_is_a_brighter_panel(
        spec_index, spec_values):
    """The disclosed half of SP-R1. On real data max sits ~+0.30 dex above
    the within-bin median while logmean is unbiased; this pins the SIGN
    and the ORDER, which is what a user reading the docstring is
    promised. (The 0.30 is real-data-specific and is NOT asserted.)"""
    n_cols = 200
    _, mx = rebin(spec_index, spec_values, n_cols, aggregator="max")
    _, mn = rebin(spec_index, spec_values, n_cols, aggregator="mean")
    _, lg = rebin(spec_index, spec_values, n_cols, aggregator="logmean")
    ok = np.isfinite(mx) & np.isfinite(lg)
    assert np.all(mx[ok] >= mn[ok] - 1e-9)
    assert np.all(mn[ok] >= lg[ok] - 1e-9)
    assert np.median(np.log10(mx[ok]) - np.log10(lg[ok])) > 0.0


# ------------------------------------------------------------- alignment

def _blank_axes(size=(4.0, 1.0), dpi=100):
    fig = plt.figure(figsize=size, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    return fig, ax


def _readback(fig, ax, x, y):
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(int)
    plt.close(fig)
    return buf


def _hot_span(buf, row):
    """Columns of `buf` whose pixel at `row` is at the top of `jet`."""
    line = buf[row, :, :]
    hot = np.flatnonzero((line[:, 0] > 120) & (line[:, 2] < 60))
    return (int(hot.min()), int(hot.max())) if hot.size else None


@pytest.mark.parametrize("hot_frac", [0.0, 0.05, 0.5, 0.999])
def test_imshow_on_the_rebinned_grid_lands_where_pcolormesh_lands(
        spec_index, hot_frac):
    """PROVEN BY RASTERISING, not by eye, and PROVEN THROUGH THE SHIPPED
    HELPER.

    Leg A is `draw_spectrogram` end to end -- its rebin, its extent, its
    imshow. Leg B is `pcolormesh` of the SAME `rebin` output on the same
    edges, which is the ground truth `imshow` has to reproduce. An
    off-by-half-cell extent moves leg A by half a cell (5 px at this
    geometry) and nothing else in the suite would notice, which is
    exactly why this test drives the helper and not a hand-rolled
    imshow."""
    n_cols, n_rows = 40, 5
    idx = spec_index[:800]
    Z = np.full((n_rows, len(idx)), VMIN)
    Z[0, int(hot_frac * (len(idx) - 1))] = VMAX
    edges, Zr = rebin(idx, Z, n_cols)
    y = np.arange(n_rows + 1, dtype=float)
    assert np.count_nonzero(Zr[0] > VMIN) == 1, "one hot column, by design"

    figA, axA = _blank_axes()
    draw_spectrogram(axA, idx, Z, n_cols=n_cols, y_edges=y,
                     norm=LogNorm(VMIN, VMAX), cmap="jet")
    a = _readback(figA, axA, edges, y)

    figB, axB = _blank_axes()
    axB.pcolormesh(edges, y, Zr, norm=LogNorm(VMIN, VMAX), cmap="jet")
    b = _readback(figB, axB, edges, y)

    assert a.shape == b.shape
    # bottom row of the canvas -- channel 0 is at the BOTTOM under
    # origin="lower", which is the other half of the claim
    row = a.shape[0] - 3
    span_a, span_b = _hot_span(a, row), _hot_span(b, row)
    assert span_a is not None, "the hot column did not render at all"
    assert span_a == span_b, (
        "helper put the hot column at %r, pcolormesh at %r"
        % (span_a, span_b))
    # and nowhere else in the image differs by more than colour rounding
    assert np.abs(a - b).max() <= 1


def test_the_lowest_channel_is_drawn_at_the_bottom(spec_index):
    """origin="lower": row 0 of the array is the LOWEST energy and belongs
    at the bottom of the panel. Getting this wrong flips every
    spectrogram in the tool upside down and nothing else notices."""
    Z = np.full((4, 400), VMIN)
    Z[0, :] = VMAX
    fig = plt.figure(figsize=(4.0, 1.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    draw_spectrogram(ax, spec_index[:400], Z, n_cols=40,
                     norm=LogNorm(VMIN, VMAX), cmap="jet")
    ax.set_ylim(0, 4)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(int)
    plt.close(fig)
    bottom = buf[buf.shape[0] - 3, buf.shape[1] // 2]
    top = buf[2, buf.shape[1] // 2]
    assert bottom[0] > 120 and bottom[2] < 60, "hot row is not at the bottom"
    assert not (top[0] > 120 and top[2] < 60)


# ----------------------------------------------------------- draw helper

def test_draw_spectrogram_sets_the_geometry_it_promises(spec_index,
                                                        spec_values):
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    im = draw_spectrogram(ax, spec_index, spec_values, n_cols=300,
                          norm=LogNorm(VMIN, VMAX), cmap="jet")
    edges, out = rebin(spec_index, spec_values, 300)
    assert im.get_array().shape == out.shape
    x0, x1, y0, y1 = im.get_extent()
    # EXACT, not approx: these are matplotlib DATE NUMBERS (~1.5e4), so a
    # relative tolerance of 1e-6 would swallow a shift of half a pixel
    # column and this pin would be vacuous. The helper passes edges[0]
    # and edges[-1] straight through, so equality is the right test.
    assert x0 == edges[0] and x1 == edges[-1]
    assert (y0, y1) == (0.0, float(N_CH))
    assert im.origin == "lower"
    assert im.get_interpolation() == "nearest"
    assert ax.get_aspect() == "auto"
    assert im in ax.get_children()


def test_draw_spectrogram_registers_its_artist_for_the_sweep(spec_index,
                                                             spec_values):
    """Half of F6, and it was unpinned until a mutant said so: delete the
    registration and `current_mappable` still answers, because its
    FALLBACK (the last image on the axes) happens to give the same answer
    on a panel holding one image.

    So this test builds the case where the two disagree -- a registered
    spectrogram and a LATER, unrelated image on the same axes -- and
    pins that the registered one wins. That is the whole point of
    registering: a caller may draw more than one colour-mapped thing, and
    the colorbar belongs to the one the helper drew."""
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    im = draw_spectrogram(ax, spec_index, spec_values, n_cols=100)
    assert getattr(ax, _SRC_ATTR, None) is im
    assert current_mappable(ax) is im
    other = ax.imshow(np.ones((2, 2)))
    assert ax.images[-1] is other, "the fallback would pick this one"
    assert current_mappable(ax) is im, (
        "the REGISTERED artist must win over a later image")


def test_draw_spectrogram_always_rebins(spec_index, spec_values):
    """SP-R3: no native-draw fallback below the density threshold. One
    aggregator at every zoom level beats the measured 22 ms the fallback
    would have saved on a narrow window, because an aggregator that
    changes when you zoom is one you cannot reason about."""
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    # 400 rows into a 1400-column panel: far below 4 rows per pixel
    im = draw_spectrogram(ax, spec_index[:400], spec_values[:, :400],
                          n_cols=1400)
    assert im.get_array().shape == (N_CH, 400), (
        "the clamp must bin one column per sample, not draw natively")
    assert not ax.collections, "no QuadMesh fallback may be drawn"
    assert len(ax.images) == 1


def test_an_axes_with_no_laid_out_width_bins_to_the_default(spec_index,
                                                            spec_values):
    """DEFAULT_N_COLS, which nothing pinned: a mutant set it to 7 and the
    whole suite stayed green.

    1400 is not arbitrary -- it is the panel width every measurement in
    this pack was taken at, and the number the disclosure table in
    `AGGREGATORS` is quoted for. Measured: the fallback does NOT fire in
    the tool (a real labeler panel reports 1342 px on its first plot_fn
    call and a bare add_subplot reports 465), so it exists for hand-built
    axes with no laid-out width."""
    assert DEFAULT_N_COLS == 1400, (
        "the disclosure table in AGGREGATORS is quoted at 1,400 columns")
    idx = pd.date_range("2020-01-01", periods=3000, freq="1min")
    Z = np.ones((3, 3000))
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_axes([0.1, 0.1, 1e-4, 0.8])       # sub-pixel on purpose
    assert ax.bbox.width <= 1.0
    im = draw_spectrogram(ax, idx, Z)
    assert im.get_array().shape[1] == DEFAULT_N_COLS

    # a laid-out axes uses its own width instead
    fig2 = plt.figure(figsize=(6, 4))
    ax2 = fig2.add_subplot(111)
    assert ax2.bbox.width > 1.0
    im2 = draw_spectrogram(ax2, idx, Z)
    assert im2.get_array().shape[1] == int(ax2.bbox.width)


def test_draw_spectrogram_takes_an_energy_table(spec_index, spec_values):
    energies = np.logspace(0, 4.5, N_CH)
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    im = draw_spectrogram(ax, spec_index, spec_values, n_cols=100,
                          y_centers=np.log10(energies))
    _, _, y0, y1 = im.get_extent()
    want = edges_from_centers(np.log10(energies))
    assert (y0, y1) == pytest.approx((want[0], want[-1]))

    ax2 = fig.add_subplot(212)
    im2 = draw_spectrogram(ax2, spec_index, spec_values, n_cols=100,
                           y_edges=np.arange(N_CH + 1) * 2.0)
    assert im2.get_extent()[2:] == pytest.approx((0.0, 2.0 * N_CH))


def test_draw_spectrogram_refuses_what_would_break_the_picture(
        spec_index, spec_values):
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    with pytest.raises(ValueError, match="not both"):
        draw_spectrogram(ax, spec_index, spec_values, n_cols=10,
                         y_edges=np.arange(N_CH + 1),
                         y_centers=np.arange(N_CH))
    for reserved in ("origin", "aspect", "interpolation", "extent"):
        with pytest.raises(ValueError, match=reserved):
            draw_spectrogram(ax, spec_index, spec_values, n_cols=10,
                             **{reserved: "upper"})
    with pytest.raises(ValueError, match="n_channels \\+ 1"):
        draw_spectrogram(ax, spec_index, spec_values, n_cols=10,
                         y_edges=np.arange(3))


# ------------------------------------------------------------- colorbar

def test_shared_x_axes_is_the_time_panels_plus_the_labels_strip(
        spec_index, spec_values, tmp_path):
    """The colorbar must steal width from exactly this set. Take it from
    the time panels only and the strip keeps its width, the x axes stop
    being the same axis, and a box drawn on the image no longer means the
    time it appears to mean."""
    df = _spec_frame(spec_index, spec_values)

    def plot_fn(axs, d, t0, t1):
        axs["ion_n"].plot(d.index, d["ion_n"])

    lbl = _make_labeler(df, plot_fn, tmp_path)
    try:
        lbl._update_plot()
        sibs = shared_x_axes(lbl.user_axes["spectrogram"])
        want = set(id(lbl.user_axes[k]) for k in lbl._time_axis_keys)
        want.add(id(lbl.strip_ax))
        assert set(id(a) for a in sibs) == want
        assert id(lbl.strip_ax) in [id(a) for a in sibs]
    finally:
        lbl.root.destroy()


def test_the_naive_colorbar_leaks_and_the_helper_does_not(
        spec_index, spec_values, tmp_path):
    """THE A/B, driven through the real pipeline and counted.

    Leg A is what a user writes today: `fig.colorbar(im, ax=ax)` inside
    plot_fn. Measured on this tree at 3274eb9, redraw by redraw:
    fig.axes 5,6,7,8,9,10 while the image panel keeps 0.7927, 0.6393,
    0.5166, 0.4184, 0.3398, 0.2770 of the figure width and the line
    panels and strip stay at 0.9845 -- the x axes silently stop being the
    same axis. Leg B is the helper. Same pipeline, same number of
    redraws."""
    df = _spec_frame(spec_index, spec_values)
    Zcols = list(CH)

    def naive_plot_fn(axs, d, t0, t1):
        axs["ion_n"].plot(d.index, d["ion_n"])
        ax = axs["spectrogram"]
        im = draw_spectrogram(ax, d.index, d[Zcols].to_numpy(float).T,
                              n_cols=200, norm=LogNorm(VMIN, VMAX),
                              cmap="jet")
        ax.figure.colorbar(im, ax=ax)          # the naive thing

    def helper_plot_fn(axs, d, t0, t1):
        axs["ion_n"].plot(d.index, d["ion_n"])
        ax = axs["spectrogram"]
        im = draw_spectrogram(ax, d.index, d[Zcols].to_numpy(float).T,
                              n_cols=200, norm=LogNorm(VMIN, VMAX),
                              cmap="jet")
        attach_colorbar(ax, im, label="eflux")

    lbl = _make_labeler(df, naive_plot_fn, tmp_path)
    try:
        counts, widths = [], []
        for _ in range(4):
            lbl._update_plot()
            counts.append(len(lbl.fig.axes))
            widths.append(_xspan(lbl.user_axes["spectrogram"])[1])
        assert counts == sorted(counts) and counts[-1] > counts[0], (
            "leg A is supposed to leak; if it stopped, this A/B is stale")
        assert widths[-1] < widths[0]
        assert (_xspan(lbl.user_axes["spectrogram"])
                != _xspan(lbl.strip_ax)), "leg A must break the alignment"
    finally:
        lbl.root.destroy()

    lbl = _make_labeler(df, helper_plot_fn, tmp_path)
    try:
        counts, spans = [], []
        for _ in range(4):
            lbl._update_plot()
            counts.append(len(lbl.fig.axes))
            spans.append((_xspan(lbl.user_axes["spectrogram"]),
                          _xspan(lbl.user_axes["ion_n"]),
                          _xspan(lbl.strip_ax)))
        assert len(set(counts)) == 1, (
            "the helper must create exactly one colorbar axes: %r" % counts)
        for spec, line, strip in spans:
            assert spec == line == strip, (
                "every time panel and the strip must keep one x extent")
        assert len(set(spans)) == 1, "the geometry must not drift"
    finally:
        lbl.root.destroy()


def test_the_colorbar_is_re_pointed_across_real_redraws(
        spec_index, spec_values, tmp_path):
    """SP-R5, driven through the real pipeline.

    `_update_plot` clears every user panel, which detaches the artist the
    colorbar was built from (measured: it leaves ax.get_children() and
    its .figure goes None). Without the sweep the bar points at a dead
    artist for the rest of the session; with it, cb.mappable is the image
    on screen after every single redraw."""
    df = _spec_frame(spec_index, spec_values)
    Zcols = list(CH)
    seen = []

    def plot_fn(axs, d, t0, t1):
        axs["ion_n"].plot(d.index, d["ion_n"])
        ax = axs["spectrogram"]
        im = draw_spectrogram(ax, d.index, d[Zcols].to_numpy(float).T,
                              n_cols=200, norm=LogNorm(VMIN, VMAX),
                              cmap="jet")
        seen.append(im)
        attach_colorbar(ax, im, label="eflux")

    lbl = _make_labeler(df, plot_fn, tmp_path)
    try:
        lbl._update_plot()
        ax = lbl.user_axes["spectrogram"]
        cb = registered_colorbar(ax)
        assert cb is not None
        for i in range(5):
            lbl._update_plot()
            live = current_mappable(ax)
            assert live is seen[-1]
            assert live in ax.get_children()
            assert cb.mappable is live, (
                "redraw %d left the colorbar on a dead artist" % (i + 1))
        # a pan and a zoom are redraws too
        lbl._next_window()
        assert registered_colorbar(ax).mappable is current_mappable(ax)
        lbl._halve_time_window()
        assert registered_colorbar(ax).mappable is current_mappable(ax)
        assert registered_colorbar(ax) is cb, "and it is still ONE bar"
    finally:
        lbl.root.destroy()


def test_attach_colorbar_is_idempotent_and_re_points(spec_index,
                                                     spec_values):
    """Called twice on the same axes it returns the SAME colorbar and adds
    no axes -- which is what makes it safe to call from inside plot_fn."""
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    im1 = draw_spectrogram(ax, spec_index, spec_values, n_cols=100,
                           norm=LogNorm(VMIN, VMAX), cmap="jet")
    cb = attach_colorbar(ax, im1, label="eflux")
    n_axes = len(fig.axes)
    ax.clear()
    im2 = draw_spectrogram(ax, spec_index, spec_values, n_cols=100,
                           norm=LogNorm(VMIN, VMAX), cmap="jet")
    again = attach_colorbar(ax, im2)
    assert again is cb
    assert len(fig.axes) == n_axes
    assert cb.mappable is im2


def test_attach_colorbar_rebuilds_a_colorbar_that_was_removed(
        spec_index, spec_values):
    """Idempotence must not outlive the object it protects.

    `cb.remove()` takes the bar's axes out of the figure but leaves the
    registration on the owner pointing at it. Handing that dead object
    back means the panel loses its colour scale for the rest of the
    session, silently, and the sweep neither raises nor recovers. No
    ChronoTagger path calls `cb.remove()` -- `_update_plot` clears axes
    and there is no `fig.clear()` anywhere -- so this is a robustness
    pin, not a bug repro."""
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    im = draw_spectrogram(ax, spec_index, spec_values, n_cols=100,
                          norm=LogNorm(VMIN, VMAX), cmap="jet")
    cb = attach_colorbar(ax, im, label="eflux")
    n_with = len(fig.axes)
    cb.remove()
    assert len(fig.axes) == n_with - 1
    assert registered_colorbar(ax) is cb, "the dead registration is there"

    cb2 = attach_colorbar(ax, im, label="eflux")
    assert cb2 is not cb, "a removed colorbar must not be handed back"
    assert len(fig.axes) == n_with
    assert cb2.ax in list(fig.axes)
    assert registered_colorbar(ax) is cb2
    assert refresh_colorbars([ax]) == 0, "and the sweep is content"


def test_attach_colorbar_refuses_a_degenerate_axes_list(spec_index,
                                                        spec_values):
    """`axes=` is the escape hatch for a pane that is not shaped like a
    labeler pane, so it has to say what it will not take. Measured on the
    unguarded code: `axes=[]` came back as a bare `IndexError: list index
    out of range` from matplotlib's `make_axes`, and an axes belonging to
    a DIFFERENT figure was accepted silently -- the bar built over there
    while registered here, so the sweep would spend the session
    re-pointing a colorbar nobody can see."""
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    im = draw_spectrogram(ax, spec_index, spec_values, n_cols=100,
                          norm=LogNorm(VMIN, VMAX), cmap="jet")
    n_axes = len(fig.axes)
    with pytest.raises(ValueError, match="at least one axes"):
        attach_colorbar(ax, im, axes=[])

    fig2 = plt.figure(figsize=(4, 3))
    ax2 = fig2.add_subplot(111)
    with pytest.raises(ValueError, match="DIFFERENT figure"):
        attach_colorbar(ax, im, axes=[ax2])
    with pytest.raises(ValueError, match="DIFFERENT figure"):
        attach_colorbar(ax, im, axes=[ax, ax2])   # one bad one spoils it
    assert len(fig.axes) == n_axes, "a refusal may not leave half a bar"
    assert len(fig2.axes) == 1
    assert registered_colorbar(ax) is None

    # the legitimate override still works
    cb = attach_colorbar(ax, im, axes=[ax], label="eflux")
    assert cb is not None and registered_colorbar(ax) is cb


def test_attach_colorbar_is_narrower_than_matplotlibs_default(spec_index,
                                                              spec_values):
    """F5, which nothing pinned: a mutant reverted fraction/pad to
    matplotlib's 0.15 / 0.05 and the whole suite stayed green.

    Measured on a four-panel stacked pane: the helper's 0.04 / 0.01 costs
    the image panel 4.3 % of its width, matplotlib's defaults cost
    17.2 %. On the labeler's own pane the same comparison is 0.9845 ->
    0.9365 against 0.9845 -> 0.7927, i.e. 19 % of the figure for one
    colorbar."""
    idx = pd.date_range("2020-01-01", periods=400, freq="1min")
    Z = np.abs(np.random.RandomState(4).randn(4, 400)) * 1e5 + 1e3

    def cost(**kw):
        fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
        im = draw_spectrogram(axes[0], idx, Z, n_cols=200,
                              norm=LogNorm(VMIN, VMAX), cmap="jet")
        for a in axes[1:]:
            a.plot(idx, np.arange(400))
        before = float(axes[0].get_position().x1)
        attach_colorbar(axes[0], im, **kw)
        fig.canvas.draw()
        after = float(axes[0].get_position().x1)
        plt.close(fig)
        return before, before - after

    before, helper = cost()
    _, mpl = cost(fraction=0.15, pad=0.05)
    assert helper < 0.08 * before, (
        "the helper's defaults must stay a thin bar: lost %.4f of %.4f"
        % (helper, before))
    assert mpl > 0.12 * before, (
        "if matplotlib's defaults stopped being expensive this pin is "
        "stale")
    assert mpl > 3.0 * helper, (
        "matplotlib's defaults cost %.4f, the helper's %.4f -- the "
        "helper is supposed to be several times narrower" % (mpl, helper))


def test_attach_colorbar_sets_the_label_it_is_given(spec_index,
                                                    spec_values):
    """The label the flagship passes, which nothing pinned: a mutant
    dropped it silently and the suite stayed green."""
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    im = draw_spectrogram(ax, spec_index, spec_values, n_cols=100,
                          norm=LogNorm(VMIN, VMAX), cmap="jet")
    cb = attach_colorbar(ax, im, label="ion eflux (eV/cm^2-s-sr-eV)")
    assert cb.ax.get_ylabel() == "ion eflux (eV/cm^2-s-sr-eV)"
    # and a bar asked for no label does not invent one
    fig2 = plt.figure(figsize=(6, 4))
    ax2 = fig2.add_subplot(111)
    im2 = draw_spectrogram(ax2, spec_index, spec_values, n_cols=100)
    assert attach_colorbar(ax2, im2).ax.get_ylabel() == ""


def test_attach_colorbar_says_so_when_there_is_nothing_to_map(spec_index):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])
    with pytest.raises(ValueError, match="no colour-mapped artist"):
        attach_colorbar(ax)


def test_the_sweep_finds_a_hand_drawn_pcolormesh_too(spec_index,
                                                     spec_values):
    """A caller who wants their own pcolormesh uses the core `rebin` and
    draws it themselves; the sweep still re-points their colorbar, so
    SP-R5 costs them no per-frame code either."""
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    edges, Zr = rebin(spec_index, spec_values, 100)
    y = np.arange(N_CH + 1, dtype=float)
    mesh = ax.pcolormesh(edges, y, Zr, norm=LogNorm(VMIN, VMAX), cmap="jet")
    cb = attach_colorbar(ax, mesh)
    ax.clear()
    mesh2 = ax.pcolormesh(edges, y, Zr, norm=LogNorm(VMIN, VMAX), cmap="jet")
    assert refresh_colorbars([ax]) == 1
    assert cb.mappable is mesh2


# ------------------------------------------------------------- the sweep

def test_current_mappable_never_returns_a_dead_artist(spec_index,
                                                      spec_values):
    """F6's staleness test, which the module docstring calls exact rather
    than heuristic -- and which nothing pinned: a mutant deleted it and
    the suite stayed green, because every other test happens to redraw
    something before asking.

    `ax.clear()` detaches the artist (it leaves get_children() and its
    .figure goes None) but does NOT remove the registration, so without
    the check the sweep would feed a dead artist to `cb.update_normal`
    every frame."""
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    im = draw_spectrogram(ax, spec_index, spec_values, n_cols=100)
    assert current_mappable(ax) is im
    ax.clear()
    assert getattr(ax, _SRC_ATTR, None) is im, "the registration survives"
    assert im not in ax.get_children()
    assert current_mappable(ax) is None, (
        "a detached artist is not the current mappable")
    # and after a REDRAW the new artist is what the sweep sees
    im2 = draw_spectrogram(ax, spec_index, spec_values, n_cols=100)
    assert current_mappable(ax) is im2


def test_the_sweep_is_a_no_op_when_nothing_is_registered(spec_index,
                                                         spec_values):
    """Every pane that never calls attach_colorbar must pay nothing and
    notice nothing -- including panes holding images, collections or
    plain lines."""
    fig = plt.figure(figsize=(6, 4))
    ax_line = fig.add_subplot(311)
    ax_line.plot([0, 1], [0, 1])
    ax_img = fig.add_subplot(312)
    draw_spectrogram(ax_img, spec_index, spec_values, n_cols=50)
    ax_empty = fig.add_subplot(313)
    assert refresh_colorbars([ax_line, ax_img, ax_empty]) == 0
    assert refresh_colorbars([]) == 0
    assert registered_colorbar(ax_img) is None


def test_a_pane_with_no_colorbar_redraws_exactly_as_before(
        spec_index, spec_values, tmp_path):
    """The sweep may not touch any other plotting behaviour: same axes
    count, same geometry, same artists, before and after."""
    df = _spec_frame(spec_index, spec_values)
    Zcols = list(CH)

    def plot_fn(axs, d, t0, t1):
        axs["ion_n"].plot(d.index, d["ion_n"])
        axs["Bx"].plot(d.index, d["Bx"])
        draw_spectrogram(axs["spectrogram"], d.index,
                         d[Zcols].to_numpy(float).T, n_cols=200)

    lbl = _make_labeler(df, plot_fn, tmp_path)
    try:
        lbl._update_plot()
        before = (len(lbl.fig.axes),
                  _xspan(lbl.user_axes["spectrogram"]),
                  _xspan(lbl.strip_ax),
                  len(lbl.user_axes["spectrogram"].images))
        for _ in range(4):
            lbl._update_plot()
        after = (len(lbl.fig.axes),
                 _xspan(lbl.user_axes["spectrogram"]),
                 _xspan(lbl.strip_ax),
                 len(lbl.user_axes["spectrogram"].images))
        assert before == after
        assert before[1] == before[2]
    finally:
        lbl.root.destroy()


def test_update_plot_runs_the_sweep(spec_index, spec_values, tmp_path):
    """The wiring itself, pinned separately from the behaviour: delete the
    call from _update_plot and this is what says so."""
    import chronotagger.labeler.mixins.plotting as plotting_mod

    df = _spec_frame(spec_index, spec_values)
    calls = []
    real = plotting_mod.refresh_colorbars

    def spy(axes):
        axes = list(axes)
        calls.append(len(axes))
        return real(axes)

    def plot_fn(axs, d, t0, t1):
        axs["ion_n"].plot(d.index, d["ion_n"])

    lbl = _make_labeler(df, plot_fn, tmp_path)
    try:
        plotting_mod.refresh_colorbars = spy
        lbl._update_plot()
        lbl._update_plot()
    finally:
        plotting_mod.refresh_colorbars = real
        lbl.root.destroy()
    assert len(calls) == 2
    assert calls == [3, 3], "the sweep must see every user axes"


def test_a_broken_colorbar_cannot_take_the_redraw_down(
        spec_index, spec_values, tmp_path):
    """Pack 5 doctrine, applied to the new sweep: a colour scale may never
    be the thing that kills a frame.

    The registration is made through the module's own `_CB_ATTR`, not
    through a copy of the string: with the string hardcoded here, renaming
    the constant left this test setting an attribute nobody reads -- the
    redraw then trivially did not raise and the pin passed while
    verifying nothing."""
    df = _spec_frame(spec_index, spec_values)

    class Exploding(object):
        mappable = None

        def update_normal(self, m):
            raise RuntimeError("boom")

    def plot_fn(axs, d, t0, t1):
        ax = axs["spectrogram"]
        ax.imshow(np.ones((2, 2)))
        setattr(ax, _CB_ATTR, Exploding())

    lbl = _make_labeler(df, plot_fn, tmp_path)
    try:
        lbl._update_plot()          # must not raise
        assert lbl.user_axes["spectrogram"].images
        assert isinstance(
            registered_colorbar(lbl.user_axes["spectrogram"]), Exploding), (
            "the sweep must actually have READ the registration")
    finally:
        lbl.root.destroy()


# ------------------------------------------------- Pack 6 absence holds

def test_the_new_module_did_not_revive_the_deleted_names():
    """tests/test_pack6_cleanup.py pins `utils/colorbar.py`,
    `ensure_lane_colorbar` and `time_lane_cbar_gutter` absent from src/.
    This module is new code doing the same job under new names, and that
    is deliberate -- reviving the old ones is not a silent option."""
    import ast
    import importlib.util
    from pathlib import Path
    import chronotagger.labeler.utils.spectrogram as mod

    assert importlib.util.find_spec(
        "chronotagger.labeler.utils.colorbar") is None
    # Pack 6's own detector shape: code INCLUDING string literals, with
    # comments dropped -- this module names both deleted symbols in a
    # comment on purpose, and a plain substring scan would match the
    # rationale rather than the code.
    src = Path(mod.__file__).read_text(encoding="utf-8")
    code = ast.unparse(ast.parse(src))
    assert "ensure_lane_colorbar" not in code
    assert "time_lane_cbar_gutter" not in code
