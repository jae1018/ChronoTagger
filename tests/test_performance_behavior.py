"""
Pack 5 (measured performance) regression tests.

These pin BEHAVIOUR, never timing. Pack 5's speed claims are verified by
the rerunnable benchmark scripts under edit_pack/evidence/ -- a timing
assert inside a unit suite flakes on a loaded machine and teaches the
suite to lie, which is the opposite of what this pack is for (R9).

What each group owns:
  vectorized index mapping  -- same positions AND the same raise (R1/R2)
  decimation                -- spikes survive, draws only, escapes (R11)
  redraw coalescing         -- a burst collapses, a lone gesture does not
  layout freeze             -- frozen after a draw, re-solved on a resize
  export                    -- the -1 contract and the non-monotonic fix
  sync_dir                  -- the Windows guard swallows exactly errno 13
"""

import os

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

BIG_ROWS = 20000


@pytest.fixture
def df_big():
    """Wide enough that decimation engages: the fixture figure is 1400 px,
    the early exit is 4 samples per pixel, so ~5,000 rows is the floor."""
    idx = pd.date_range("2015-01-03 00:00:00", periods=BIG_ROWS, freq="1s")
    t = np.linspace(0, 200, BIG_ROWS)
    return pd.DataFrame(
        {"BX": np.sin(t) * 10, "BY": np.cos(t) * 5, "BZ": np.linspace(-7, 3, BIG_ROWS)},
        index=idx,
    )


@pytest.fixture
def big_labeler(df_big, tmp_path):
    from chronotagger.labeler import TimeIntervalLabeler

    layout_spec = {
        "nrows": 3, "ncols": 1,
        "areas": [
            {"key": "panel1", "row": 0, "col": 0, "rowspan": 1, "colspan": 1,
             "role": "time"},
            {"key": "panel2", "row": 1, "col": 0, "rowspan": 1, "colspan": 1,
             "role": "time"},
            {"key": "labels", "row": 2, "col": 0, "rowspan": 1, "colspan": 1,
             "role": "labels"},
        ],
    }

    def plot_fn(axs, df, t0, t1):
        axs["panel1"].plot(df.index, df["BX"])
        axs["panel2"].plot(df.index, df["BY"])
        axs["panel2"].plot(df.index, df["BZ"])

    lbl = TimeIntervalLabeler(
        df=df_big, plot_fn=plot_fn,
        window=df_big.index[-1] - df_big.index[0],
        autosave_folder=str(tmp_path), layout_spec=layout_spec)
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    yield lbl
    lbl.root.destroy()


@pytest.fixture
def raw_labeler(df_big, tmp_path):
    """The same frame with decimation OFF -- the escape-hatch control.
    A fixture, not an in-test construction, so a Tk-init flake surfaces
    as an ERROR rather than a FAILED (verifier V2, M9)."""
    from chronotagger.labeler import TimeIntervalLabeler

    layout_spec = {
        "nrows": 2, "ncols": 1,
        "areas": [
            {"key": "panel1", "row": 0, "col": 0, "role": "time"},
            {"key": "labels", "row": 1, "col": 0, "role": "labels"},
        ],
    }
    lbl = TimeIntervalLabeler(
        df=df_big, plot_fn=lambda axs, df, t0, t1: axs["panel1"].plot(
            df.index, df["BX"]),
        window=df_big.index[-1] - df_big.index[0],
        autosave_folder=str(tmp_path), layout_spec=layout_spec, decimate=False)
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    yield lbl
    lbl.root.destroy()


def _nonmono_frame():
    """Two 'spacecraft' concatenated without a re-sort -- the frame that
    made searchsorted mislabel and get_indexer raise (pack5_g1 2e/6b)."""
    a = pd.date_range("2015-01-03 00:00:00", periods=300, freq="30s")
    b = pd.date_range("2015-01-03 00:00:15", periods=300, freq="30s")
    idx = a.append(b)
    return pd.DataFrame({"v": np.arange(len(idx), dtype=float)}, index=idx)


# ------------------------------------------------- R1/R2 index mapping

def _scalar_nearest(idx, timestamps):
    """The loop Pack 5 replaces, verbatim (selection.py at 354be67)."""
    out = []
    for ts in timestamps:
        j = idx.get_indexer([ts], method="nearest")[0]
        if 0 <= j < len(idx):
            out.append(int(j))
    return out


def _scalar_exact_then_nearest(idx, timestamps):
    """The _timestamps_to_indices ladder, verbatim."""
    out = []
    for ts in timestamps:
        try:
            j = idx.get_loc(ts)
            if isinstance(j, slice):
                j = j.start
            if j is not None and 0 <= j < len(idx):
                out.append(int(j))
        except Exception:
            try:
                j = idx.get_indexer([ts], method="nearest")[0]
                if 0 <= j < len(idx):
                    out.append(int(j))
            except Exception:
                continue
    return out


def test_positions_nearest_matches_the_scalar_loop(df_big):
    """R1: bit-exact, including on probes that came back through
    matplotlib's float day numbers a median 484 ns off the true sample."""
    import matplotlib.dates as mdates
    from chronotagger.labeler.utils.fastindex import positions_nearest

    idx = df_big.index
    take = np.arange(0, len(idx), 37)
    probes = []
    for xf in mdates.date2num(idx[take].to_numpy()):
        dt = mdates.num2date(float(xf))
        if getattr(dt, "tzinfo", None) is not None:
            dt = dt.replace(tzinfo=None)
        probes.append(pd.Timestamp(dt))

    assert positions_nearest(idx, probes) == _scalar_nearest(idx, probes)
    # and it really is the rows we asked for, not merely self-consistent
    assert positions_nearest(idx, probes) == [int(j) for j in take]


def test_positions_nearest_drops_out_of_range_instead_of_keeping_minus_one():
    """Invariant V1: a vectorized get_indexer returns -1 for a probe it
    cannot place, and -1 kept in a position list indexes the LAST row.

    The probe has to be one that ACTUALLY produces -1 (verifier V2, M1):
    an exact index member never can, so an all-members test passes with
    the mask deleted. An EMPTY index does produce -1 for every probe --
    measured `pd.DatetimeIndex([]).get_indexer(probes, "nearest")` ->
    `[-1, -1, -1]`.
    """
    from chronotagger.labeler.utils.fastindex import positions_nearest
    idx = pd.date_range("2015-01-03", periods=10, freq="30s")
    assert positions_nearest(idx, []) == []
    assert -1 not in positions_nearest(idx, list(idx))

    empty = pd.DatetimeIndex([])
    probes = list(pd.date_range("2015-01-03", periods=3, freq="30s"))
    assert list(empty.get_indexer(pd.DatetimeIndex(probes),
                                  method="nearest")) == [-1, -1, -1]
    assert positions_nearest(empty, probes) == []


def test_vectorized_mapping_raises_exactly_as_the_scalar_loop_did():
    """R1/R3: behaviour-preserving means the FAILURE is preserved too.
    np.searchsorted was rejected precisely because it answers here."""
    from chronotagger.labeler.utils.fastindex import positions_nearest
    idx = _nonmono_frame().index
    probes = list(idx[:50])

    with pytest.raises(ValueError) as scalar_exc:
        _scalar_nearest(idx, probes)
    with pytest.raises(ValueError) as vector_exc:
        positions_nearest(idx, probes)
    assert str(vector_exc.value) == str(scalar_exc.value)


def test_exact_then_nearest_ladder_matches_including_non_monotonic():
    """R2: _timestamps_to_indices tried get_loc first and fell back to
    nearest per probe. Same answers, and the same silent skip when the
    nearest pass cannot run."""
    from chronotagger.labeler.utils.fastindex import positions_exact_then_nearest
    for idx in (pd.date_range("2015-01-03", periods=500, freq="30s"),
                _nonmono_frame().index):
        probes = list(idx[::7]) + [idx[3] + pd.Timedelta("1s")]
        assert (positions_exact_then_nearest(idx, probes)
                == _scalar_exact_then_nearest(idx, probes))


def test_exact_then_nearest_keeps_the_scalar_ladder_on_a_duplicated_index():
    """get_loc returns a slice on a non-unique index and the old code took
    .start; get_indexer refuses outright. Vectorizing that would turn a
    working answer into an exception."""
    from chronotagger.labeler.utils.fastindex import positions_exact_then_nearest
    base = pd.date_range("2015-01-03", periods=20, freq="30s")
    idx = base.append(base[5:10])
    assert not idx.is_unique
    probes = list(base[4:12])
    assert (positions_exact_then_nearest(idx, probes)
            == _scalar_exact_then_nearest(idx, probes))


def test_preview_timestamps_equal_the_unclipped_per_span_masks(big_labeler):
    """R2: _get_preview_timestamps clips to the window ONCE instead of
    ANDing a full-length window mask in per span. Same rows, same order."""
    lbl = big_labeler
    idx = lbl.df.index
    spans = [(idx[i], idx[i + 300]) for i in range(0, 9000, 900)]
    lbl._commit_spans = spans

    expected = []
    for s, e in spans:
        mask = (idx >= s) & (idx < e)
        mask &= (idx >= lbl.t0) & (idx <= lbl.t1)
        expected.extend(idx[mask].tolist())

    assert lbl._get_preview_timestamps() == expected


# ------------------------------------------------------- R11 decimation

@pytest.mark.parametrize("col", ["BX", "BY", "BZ"])
@pytest.mark.parametrize("row", [0, 1, 6543, 12345, BIG_ROWS - 1])
def test_a_single_sample_spike_survives_decimation(df_big, col, row):
    """R11's headline guarantee. Per pixel bin, per NUMERIC column
    independently, the argmin and argmax rows are kept -- so a one-sample
    spike is that column's extremum in its own bin and cannot be dropped,
    whichever column it lands in."""
    from chronotagger.labeler.utils.decimate import plan_decimation

    spiked = df_big.copy()
    spiked.iloc[row, spiked.columns.get_loc(col)] = 1e9
    plan = plan_decimation(spiked, 1340)
    assert plan is not None, "expected decimation to be active at this size"
    kept = plan[0]
    assert row in set(int(k) for k in kept)

    spiked.iloc[row, spiked.columns.get_loc(col)] = -1e9
    assert row in set(int(k) for k in plan_decimation(spiked, 1340)[0])


def test_decimation_emits_original_rows_only(df_big):
    """Nothing averaged, nothing synthesised: every retained row is an
    original sample, with its true timestamp and its true values."""
    from chronotagger.labeler.utils.decimate import plan_decimation
    kept = plan_decimation(df_big, 1340)[0]
    out = df_big.take(kept)
    assert len(out) < len(df_big)
    assert out.index.is_monotonic_increasing
    assert out.index.isin(df_big.index).all()
    assert np.array_equal(out.to_numpy(), df_big.loc[out.index].to_numpy())


def test_decimation_is_a_no_op_when_points_per_pixel_is_small(df_big):
    """R11's early exit: below ~4 samples per pixel there is nothing to
    win, so zooming in genuinely reveals raw data.

    DISCRIMINATING shape (verifier V2, M2): the obvious assertions here
    are satisfied by the LATER `kept.size >= n` guard, so deleting the
    early exit left them green. This case is below the threshold AND
    would produce a real plan without it -- measured, one numeric column
    at 3.5 samples per pixel yields 200 kept rows of 350 when the early
    exit is removed, and None when it is present.
    """
    from chronotagger.labeler.utils.decimate import plan_decimation
    assert plan_decimation(df_big.iloc[:2000], 1340) is None
    assert plan_decimation(df_big.iloc[:1], 1340) is None

    one_column = df_big.iloc[:350][["BX"]]
    assert len(one_column) < 4.0 * 100, "the case must be below the exit"
    assert plan_decimation(one_column, 100) is None
    # and just above the threshold a plan IS produced, so the exit is not
    # simply refusing everything
    assert plan_decimation(df_big.iloc[:401][["BX"]], 100) is not None


def test_decimation_survives_an_all_nan_column(df_big):
    from chronotagger.labeler.utils.decimate import plan_decimation
    frame = df_big.copy()
    frame["dead"] = np.nan
    frame.iloc[9999, frame.columns.get_loc("BX")] = 1e9
    kept = plan_decimation(frame, 1340)[0]
    assert 9999 in set(int(k) for k in kept)


def test_a_lone_real_sample_in_an_all_nan_bin_survives(df_big):
    """fmin/fmax IGNORE NaN; minimum/maximum PROPAGATE it. With the
    propagating pair a gappy column's bin extreme is NaN, matches
    nothing, and the one real sample in that bin is dropped -- measured
    (verifier V2, M3: kept=True with fmin/fmax, kept=False with
    minimum/maximum). Gappy instrument data is the normal case here, so
    this is the NaN pin with real correctness value."""
    from chronotagger.labeler.utils.decimate import plan_decimation
    frame = df_big.copy()
    frame["gappy"] = np.nan
    row = 12345
    frame.iloc[row, frame.columns.get_loc("gappy")] = 5.0

    kept = set(int(k) for k in plan_decimation(frame, 1340)[0])
    assert row in kept, "the only real sample of a gappy column was dropped"


def test_the_highlight_marker_cap_runs_before_the_mapping(big_labeler,
                                                          monkeypatch):
    """EDITs 119/120. Two things the suite did not catch (verifier V2,
    M4): the cap VALUE could be changed 200x with everything green, and
    nothing pinned that the cap now runs BEFORE the mapping -- which is
    the entire point of the reorder (pack5_g1 S6). Watch what
    `_timestamps_to_indices` is actually handed."""
    seen = []
    real = big_labeler._timestamps_to_indices
    monkeypatch.setattr(big_labeler, "_timestamps_to_indices",
                        lambda ts: seen.append(len(ts)) or real(ts))

    idx = big_labeler.df.index
    big_labeler._commit_spans = [(idx[0], idx[15000])]
    assert len(big_labeler._get_preview_timestamps()) == 15000

    big_labeler._show_selected_point_highlights(redraw=False)

    assert seen, "the mapping was never reached"
    assert seen[0] <= 2000, \
        "the expensive step still sees the FULL preview set: %d" % seen[0]
    # 15,000 timestamps -> step = 15000 // 1000 = 15 -> ~1000 probes
    assert 900 <= seen[0] <= 1100, "the ~1000-marker cap moved: %d" % seen[0]

    # Pack 8.5-B B5. Everything above watches what goes INTO the mapping
    # and nothing watched what came out, so this test drove the B1 defect
    # on every run and passed on it: decimation active, 1000 timestamps
    # in, ZERO highlight artists out. The fixture must stay in the
    # decimating regime or this pin stops covering the thing it is for.
    assert big_labeler._decim_active is True, (
        "big_labeler no longer decimates; this pin covers the "
        "under-decimation case and needs a frame that engages it")
    marks = [c for ax in big_labeler.user_axes.values() for c in ax.collections
             if (c.get_gid() or "") == "chronotagger:preview-highlight"]
    assert len(marks) == 2, (
        "one highlight artist per time panel; got %d" % len(marks))
    total = sum(int(c.get_offsets().shape[0]) for c in marks)
    assert total == 3 * seen[0], (
        "three drawn lines x %d probes should be %d marks; got %d"
        % (seen[0], 3 * seen[0], total))


def test_redraw_decimates_and_the_window_cache_follows_the_artists(big_labeler):
    """The drawn frame is decimated and _last_windowed_index caches THAT
    frame, so Pack 3's artist-ordinal condition still holds."""
    lbl = big_labeler
    assert lbl._decim_active is True
    drawn = len(lbl.user_axes["panel1"].lines[0].get_xdata())
    assert drawn < len(lbl.df)
    assert len(lbl._last_windowed_index) == drawn


def test_decimation_is_draw_only(big_labeler):
    """THE invariant (R11): decimation changes what is DRAWN and nothing
    else. self.df is untouched, and everything that decides which samples
    get labelled reads self.df."""
    lbl = big_labeler
    assert len(lbl.df) == BIG_ROWS
    assert lbl.df.index.is_monotonic_increasing

    from chronotagger.core.models import Interval
    lbl.intervals = [Interval(lbl.df.index[10], lbl.df.index[9000], "UNKNOWN")]
    ids = lbl._compute_label_id_series()
    assert len(ids) == BIG_ROWS
    assert int((ids.values != -1).sum()) == 8990


def test_box_select_never_reads_a_decimated_artist(big_labeler, monkeypatch):
    """R11, the load-bearing half: a box select re-renders at full
    resolution before scanning. Measured, a thin y-band against a
    decimated trace had recall 0.687 and precision 0.125 against the raw
    scan -- silent mislabelling."""
    import matplotlib.dates as mdates
    lbl = big_labeler
    assert lbl._decim_active is True

    seen = {}
    real_scan = lbl._finalize_box_selection

    def spy(picked_ts):
        seen["decim_active"] = lbl._decim_active
        seen["drawn"] = len(lbl.user_axes["panel1"].lines[0].get_xdata())
        return real_scan(picked_ts)

    monkeypatch.setattr(lbl, "_finalize_box_selection", spy)

    ax = lbl.user_axes["panel1"]
    y0, y1 = ax.get_ylim()
    x0, x1 = ax.get_xlim()
    e1 = type("E", (), {"xdata": x0 + (x1 - x0) * 0.2, "ydata": y0,
                        "inaxes": ax, "button": 1})
    e2 = type("E", (), {"xdata": x0 + (x1 - x0) * 0.6,
                        "ydata": y0 + (y1 - y0) * 0.6,
                        "inaxes": ax, "button": 1})
    lbl._on_rectangle_select(e1, e2, lbl.active_pane)

    assert seen.get("decim_active") is False, "artists were still decimated"
    assert seen.get("drawn") == len(lbl.df), "scan saw a decimated trace"


def test_decimate_false_draws_every_sample(raw_labeler):
    """The escape hatch is a real escape, not a hint.

    Construction lives in the `raw_labeler` FIXTURE, not in the test body
    (verifier V2, M9): this machine's known `tcl_findLibrary` Tk-init
    flake then surfaces as an ERROR like every other affected test,
    instead of as a FAILED that reads like a real regression in a log."""
    lbl = raw_labeler
    assert lbl._decim_active is False
    assert len(lbl.user_axes["panel1"].lines[0].get_xdata()) == BIG_ROWS
    assert len(lbl._last_windowed_index) == BIG_ROWS


def test_decimation_stands_down_for_companion_arrays_and_cross_plots(
        big_labeler):
    """Two shapes where selecting rows breaks something real: df.attrs
    arrays are windowed to the FULL window (a spectrogram's energy table
    would no longer align), and a min/max-per-time-bin envelope means
    nothing on an X-Y cross plot."""
    lbl = big_labeler
    assert lbl._decimation_enabled() is True

    lbl.df.attrs["energy"] = np.arange(len(lbl.df), dtype=float)
    assert lbl._decimation_enabled() is False
    lbl.df.attrs.clear()
    assert lbl._decimation_enabled() is True

    lbl.axes_meta["panel2"]["role"] = "not-time"
    assert lbl._decimation_enabled() is False


# ------------------------------------------------ R4d redraw coalescing

def test_a_burst_of_redraw_requests_collapses_to_one_render(labeler):
    """R4d/R12: ten requests, ONE render -- measured, ten requests produce
    ten full Agg renders today (7.1 s at 43k points)."""
    renders = []
    real = labeler._update_plot
    labeler._update_plot = lambda: renders.append(1) or real()

    for _ in range(10):
        labeler._request_redraw()
    assert renders == [], "a coalesced request must not render inline"

    labeler.root.update_idletasks()
    assert len(renders) == 1


def test_a_lone_request_still_renders_at_idle(labeler):
    """No timer, no debounce: the single-gesture case pays no latency."""
    renders = []
    real = labeler._update_plot
    labeler._update_plot = lambda: renders.append(1) or real()

    labeler._request_redraw()
    labeler.root.update_idletasks()
    assert len(renders) == 1


def test_flush_renders_now_and_cancels_the_queued_one(labeler):
    """Gestures that READ what _update_plot writes flush first, so a click
    that lands before the idle handler cannot map onto the old window.

    The render count alone does NOT pin the cancel (verifier V2, M11a):
    `_redraw_pending` is cleared before rendering, so an uncancelled
    callback would fire and return early, leaving the count at 1. Spy on
    the CALLBACK instead -- with the cancel it never runs at all."""
    renders = []
    real = labeler._update_plot
    labeler._update_plot = lambda: renders.append(1) or real()
    callbacks = []
    real_cb = labeler._run_pending_redraw
    labeler._run_pending_redraw = lambda: callbacks.append(1) or real_cb()

    labeler._request_redraw()
    labeler._flush_pending_redraw()
    assert len(renders) == 1
    assert labeler._redraw_idle_id is None
    labeler.root.update_idletasks()
    assert len(renders) == 1, "the cancelled idle callback fired anyway"
    assert callbacks == [], "the idle callback was never cancelled"


def test_request_redraw_is_synchronous_without_a_tk_root():
    """Headless hosts and the mixin harnesses have no idle queue. Their
    _update_plot must still be called, in line, exactly once."""
    from chronotagger.labeler.mixins.events.base import EventsBaseMixin

    class Host:
        root = None

        def __init__(self):
            self.calls = 0

        def _update_plot(self):
            self.calls += 1

    host = Host()
    EventsBaseMixin._request_redraw(host)
    assert host.calls == 1
    assert host._redraw_pending is False
    assert host._redraw_idle_id is None
    EventsBaseMixin._flush_pending_redraw(host)
    assert host.calls == 1, "nothing was pending; flush must be a no-op"


# ------------------------------------------- R4d rerouted call sites

def test_update_time_window_coalesces_and_moves_the_window(labeler):
    """EDIT 131. Reverting the reroute turned NOTHING red (verifier V2,
    M6), so the site gets its own pin: the button must move the window
    and render once, at idle."""
    renders = []
    real = labeler._update_plot
    labeler._update_plot = lambda: renders.append(1) or real()
    idx = labeler.df.index
    labeler.start_time_entry.delete(0, "end")
    labeler.start_time_entry.insert(0, str(idx[5]))
    labeler.end_time_entry.delete(0, "end")
    labeler.end_time_entry.insert(0, str(idx[40]))

    labeler._update_time_window()
    assert renders == [], "the button must not render inline any more"
    labeler.root.update_idletasks()
    assert len(renders) == 1
    assert labeler.t0 == idx[5] and labeler.t1 == idx[40]


def test_strip_release_commits_then_coalesces(labeler, monkeypatch):
    """EDIT 134. The resize commit and the autosave still happen on the
    release; only the RENDER defers.

    Recheck finding F-2: the old `>= 1` could not see the reroute at all
    -- reverting EDIT 134 left it green. Measured, the reroute moves the
    render across the idle boundary WITHOUT changing the total, so the
    phase is the observable and the total is not:

        EDIT 134 as drafted : 0 renders inline, 1 after update_idletasks
        EDIT 134 reverted   : 1 render  inline, 0 after

    (The old docstring's premise was also wrong: the command path does
    not render on its own here -- inline is 0.) Count-based, with no time
    dependence: update_idletasks drains Tk's idle queue synchronously."""
    from chronotagger.core.models import Interval
    idx = labeler.df.index
    iv = Interval(idx[10], idx[30], "PS")
    labeler.intervals = [iv]
    saves = []
    monkeypatch.setattr(labeler, "_save_autosave", lambda: saves.append(1))
    renders = []
    real = labeler._update_plot
    labeler._update_plot = lambda: renders.append(1) or real()

    labeler._drag_mode = "resize_end"
    labeler._drag_iv = iv
    labeler._drag_preview = (idx[10], idx[45])
    labeler.current_selection = (idx[10], idx[45])

    labeler._on_strip_release(object(), labeler.active_pane)

    # the COMMIT half stays synchronous
    assert labeler.current_selection is None
    assert saves == [1], "autosave must still fire on the release"
    assert labeler._drag_mode is None
    # the RENDER half is the reroute: nothing inline, exactly one at idle
    assert renders == [], "the release must not render inline any more"
    assert getattr(labeler, "_redraw_pending", False) is True, (
        "the release must leave a redraw queued on the idle queue")
    labeler.root.update_idletasks()
    assert len(renders) == 1, "the queued redraw must render exactly once"
    assert getattr(labeler, "_redraw_pending", False) is False, (
        "the idle handler must clear the pending flag")


def test_the_wheel_junction_coalesces(labeler):
    """EDIT 132 is R4d's headline -- both wheel time paths funnel through
    it -- and reverting it turned nothing red (verifier V2, M6). Ten
    notches over empty canvas must move the window and render ONCE."""
    renders = []
    real = labeler._update_plot
    labeler._update_plot = lambda: renders.append(1) or real()
    before = labeler.t1 - labeler.t0

    for _ in range(10):
        ev = type("Evt", (), {"inaxes": None, "xdata": None, "button": "up",
                              "step": 1, "key": None})
        labeler._on_scroll_zoom(ev, labeler.active_pane)

    assert renders == [], "each notch must not render inline"
    labeler.root.update_idletasks()
    assert len(renders) == 1, "a 10-notch burst must collapse to one render"
    assert (labeler.t1 - labeler.t0) < before, "the window did not move"


# --------------------------------------------------- R4a layout freeze

def test_layout_engine_is_frozen_after_the_first_draw(labeler):
    """R4a: the constrained-layout solver runs on EVERY draw and costs a
    flat 110-160 ms; freeze it once it has produced a geometry."""
    from matplotlib.layout_engine import PlaceHolderLayoutEngine

    pane = labeler.active_pane
    assert pane._layout_constrained is True
    assert pane._layout_frozen is True
    assert isinstance(pane.fig.get_layout_engine(), PlaceHolderLayoutEngine)


def test_a_genuine_layout_change_re_solves_then_re_freezes(labeler):
    """A resize must re-solve or the panels keep the old window's
    geometry. Everything else keeps the frozen geometry."""
    from matplotlib.layout_engine import (ConstrainedLayoutEngine,
                                          PlaceHolderLayoutEngine)
    pane = labeler.active_pane

    labeler._invalidate_layout_freeze()
    assert pane._layout_frozen is False
    assert isinstance(pane.fig.get_layout_engine(), ConstrainedLayoutEngine)

    labeler._update_plot()
    assert pane._layout_frozen is True
    assert isinstance(pane.fig.get_layout_engine(), PlaceHolderLayoutEngine)


def test_interval_bands_span_the_full_axes_height(labeler):
    """EDIT 129's blended transform is the one band detail that really is
    load-bearing, and deleting it left the whole suite green while the
    bands collapsed to a sliver (verifier V2, M10).

    EVERY user panel is measured, not just the first (recheck finding
    F-1). With the transform deleted the bands become DATA-coordinate
    rectangles spanning y = 0..1, so the cover depends entirely on the
    panel's y range: measured 1.2020 on the log10n panel (which passes a
    lower-bound-only check) and 0.0455 on the BX/BY/BZ panel. Checking
    one panel, and checking only a lower bound, each miss the bug on
    their own. The blended transform makes the cover exactly 1.0 by
    construction on every panel at any DPI, so both bounds are safe."""
    from chronotagger.core.models import Interval
    idx = labeler.df.index
    labeler.intervals = [Interval(idx[10], idx[40], "PS")]
    labeler._update_plot()

    assert labeler.user_axes, "the fixture must draw at least one user panel"
    for key, ax in labeler.user_axes.items():
        bands = [c for c in ax.collections
                 if str(c.get_gid() or "").endswith("interval-bands")]
        assert len(bands) == 1, f"{key}: expected exactly one band collection"
        pts = bands[0].get_transform().transform(
            bands[0].get_paths()[0].vertices)
        cover = ((pts[:, 1].max() - pts[:, 1].min())
                 / ax.get_window_extent().height)
        assert 0.95 < cover < 1.05, (
            f"{key}: interval bands must span the panel exactly -- neither a "
            f"sliver nor an overshoot (cover={cover:.4f})")


def test_freeze_never_installs_a_solver_the_figure_opted_out_of(labeler):
    """A lane-gutter figure is built with constrained_layout=False; the
    freeze/re-solve pair must leave it alone in both directions."""
    pane = labeler.active_pane
    pane._layout_constrained = False
    pane._layout_frozen = False
    engine_before = pane.fig.get_layout_engine()

    labeler._invalidate_layout_freeze()
    assert pane.fig.get_layout_engine() is engine_before
    labeler._freeze_layout_after_draw()
    assert pane.fig.get_layout_engine() is engine_before


# -------------------------------------------------------- R7 the export

def test_export_per_sample_writes_ids_with_minus_one_for_uncovered(
        labeler, tmp_path):
    """R7 CLEAN BREAK: uncovered rows are int -1, never the string
    'UNKNOWN' -- which was also the default name of class 0, so the two
    could not be told apart in the file."""
    from chronotagger.core.models import Interval
    idx = labeler.df.index
    labeler.classes = ["UNKNOWN", "PS", "LOBE"]
    labeler.intervals = [Interval(idx[10], idx[20], "PS")]

    out = tmp_path / "per_sample.csv"
    labeler.export_per_sample(str(out), fmt="csv")

    got = pd.read_csv(out, index_col=0)
    assert list(got.columns) == ["label_id"]
    assert got["label_id"].dtype.kind == "i"
    assert int((got["label_id"] == -1).sum()) == len(idx) - 10
    assert int((got["label_id"] == 1).sum()) == 10
    assert "UNKNOWN" not in out.read_text(encoding="utf-8")


def test_export_per_sample_still_refuses_an_empty_interval_list(labeler,
                                                                tmp_path):
    with pytest.raises(ValueError):
        labeler.export_per_sample(str(tmp_path / "x.csv"), fmt="csv")


def test_label_ids_agree_with_interval_contains_on_a_non_monotonic_frame(
        labeler):
    """R7's second half. Index.searchsorted does not validate sortedness:
    on this frame it set 95 rows where 185 are contained, and not even a
    subset of the right ones (pack5_g1 6b). Export has no raise to
    preserve here -- it has a silently wrong answer to correct."""
    from chronotagger.core.models import Interval

    frame = _nonmono_frame()
    labeler.df = frame
    labeler.classes = ["UNKNOWN", "PS", "LOBE"]
    iv = Interval(frame.index[50], frame.index[400], "PS")
    labeler.intervals = [iv]

    ids = labeler._compute_label_id_series()
    contained = np.array([iv.contains(t) for t in frame.index])
    assert contained.sum() > 0
    assert np.array_equal(ids.values != -1, contained)
    assert set(ids.values[contained].tolist()) == {1}


def test_label_ids_unchanged_on_a_monotonic_frame(labeler):
    """The fast slice is kept for the monotonic case; it must still agree
    with Interval.contains exactly."""
    from chronotagger.core.models import Interval
    idx = labeler.df.index
    labeler.classes = ["UNKNOWN", "PS", "LOBE"]
    ivs = [Interval(idx[5], idx[15], "PS"), Interval(idx[40], idx[60], "LOBE")]
    labeler.intervals = ivs

    ids = labeler._compute_label_id_series().values
    truth = np.full(len(idx), -1, dtype=int)
    for one in ivs:
        truth[np.array([one.contains(t) for t in idx])] = \
            labeler.classes.index(one.label)
    assert np.array_equal(ids, truth)


# ------------------------------------------------------ R5 the wizard cap

def _wizard_first_window(monkeypatch, index):
    """Drive the real _launch_labeler with TimeIntervalLabeler stubbed, and
    report how many samples its first frame would hold. (The pack v1 said
    this edit had no unit pin because it needs a >200k-row frame; verifier
    V2 showed the frame only needs an INDEX, so it is cheap after all.)"""
    import chronotagger.quickstart.wizard as wiz_mod

    captured = {}

    class _StubLabeler:
        def __init__(self, **kw):
            captured["window"] = kw["window"]

        def run(self):
            pass

    class _StubRoot:
        def withdraw(self):
            pass

        def destroy(self):
            pass

        def deiconify(self):
            pass

    # _launch_labeler does `from chronotagger import TimeIntervalLabeler`
    # at call time, so the package attribute is the one to replace.
    import chronotagger
    monkeypatch.setattr(chronotagger, "TimeIntervalLabeler", _StubLabeler)

    wiz = wiz_mod.QuickStartWizard()
    wiz.df = pd.DataFrame({"BX": np.zeros(len(index))}, index=index)
    wiz.root = _StubRoot()
    wiz.tabs_config = [{"title": "t"}]
    # Pack 7 W1: _build_tab_plot returns (plot_fn, layout_spec,
    # plot_config) now -- the wizard keeps the config beside the closure
    # so a driver file can be emitted from the same collected state.
    monkeypatch.setattr(
        wiz, "_build_tab_plot",
        lambda tab: (lambda axs, df, t0, t1: None, {"nrows": 1}, {}))
    monkeypatch.setattr(wiz, "_show_tab_planner", lambda: None)

    wiz._launch_labeler()
    window = captured["window"]
    return int(((index >= index[0]) & (index < index[0] + window)).sum())


def test_wizard_caps_the_first_window_at_200k_samples(monkeypatch):
    """R5. 3.0M rows at 1 s = a 300,000-point first frame under the plain
    10%-of-range rule; the cap brings it to exactly 200,000."""
    idx = pd.date_range("2015-01-01", periods=3_000_000, freq="1s")
    assert _wizard_first_window(monkeypatch, idx) == 200_000


def test_wizard_cap_is_inert_below_the_threshold(monkeypatch):
    """It only ever SHRINKS: a frame whose 10% window already holds fewer
    than 200,000 samples is untouched."""
    idx = pd.date_range("2015-01-01", periods=1_000_000, freq="1s")
    assert _wizard_first_window(monkeypatch, idx) == 100_000


def test_wizard_cap_asks_the_index_not_the_cadence(monkeypatch):
    """The edit's comment claims it asks the INDEX where sample 200,000
    sits rather than assuming a uniform cadence. A two-cadence frame is
    the case that tells the two apart."""
    fast = pd.date_range("2015-01-01", periods=400_000, freq="10ms")
    slow = pd.date_range(fast[-1] + pd.Timedelta("10s"), periods=400_000,
                         freq="10s")
    idx = fast.append(slow)
    assert _wizard_first_window(monkeypatch, idx) == 200_000


# ------------------------------------------------------- R8 the dir sync

def test_sync_dir_is_silent_and_free_on_windows(tmp_path):
    """R8, measured: on Windows os.open(dir, O_RDONLY) raises
    PermissionError errno 13 BEFORE fsync is reachable (the atomic_io
    docstring used to blame os.fsync), and os.O_DIRECTORY does not exist
    at all. The guard has to sit around the OPEN, and it costs 0 ms.
    The POSIX half of this -- fsync actually runs, +8.94 ms per save on
    ext4 -- is the Dell lane's to execute."""
    from chronotagger.labeler.utils.atomic_io import _sync_dir

    target = tmp_path / "session.json"
    target.write_text("{}", encoding="utf-8")
    _sync_dir(target)            # must not raise on any platform

    if os.name == "nt":
        with pytest.raises(PermissionError) as exc:
            fd = os.open(str(tmp_path), os.O_RDONLY)
            os.close(fd)
        assert exc.value.errno == 13
        assert not hasattr(os, "O_DIRECTORY")


def test_sync_dir_is_actually_invoked_for_a_user_write(tmp_path, monkeypatch):
    """Deleting `if sync_dir: _sync_dir(target)` left the suite green
    (verifier V2, M5) -- R8 could ship as plumbing that never syncs, and
    only the ext4 lane, where the fsync IS the feature, would notice.
    Spy on the call itself."""
    import chronotagger.labeler.utils.atomic_io as aio

    seen = []
    real = aio._sync_dir
    monkeypatch.setattr(aio, "_sync_dir",
                        lambda target: seen.append(str(target)) or real(target))

    aio.atomic_write_json(tmp_path / "plain.json", {"a": 1})
    assert seen == [], "a default write must not touch the directory"

    aio.atomic_write_json(tmp_path / "user.json", {"a": 1}, sync_dir=True)
    assert len(seen) == 1 and seen[0].endswith("user.json")


def test_sync_dir_never_takes_a_save_down(tmp_path, monkeypatch):
    """A durability nicety may not fail a write whose bytes are on disk."""
    import chronotagger.labeler.utils.atomic_io as aio

    def boom(*a, **kw):
        raise OSError("no")

    monkeypatch.setattr(aio.os, "open", boom)
    target = tmp_path / "s.json"
    aio.atomic_write_json(target, {"a": 1}, sync_dir=True)
    assert target.exists()


def test_user_initiated_saves_ask_for_the_dir_sync_and_autosave_does_not(
        labeler, tmp_path, monkeypatch):
    """R8's scope: Ctrl+S / Save As / export yes, per-gesture autosave no."""
    import chronotagger.labeler.mixins.io_export as io_mod

    seen = []
    real_json = io_mod.atomic_write_json

    def spy(target, obj, backup=False, sync_dir=False):
        seen.append((os.path.basename(str(target)), sync_dir))
        return real_json(target, obj, backup=backup, sync_dir=sync_dir)

    monkeypatch.setattr(io_mod, "atomic_write_json", spy)

    labeler._save_session(str(tmp_path / "session.json"))
    labeler._save_autosave()

    by_name = dict(seen)
    assert by_name["session.json"] is True
    autosaves = [v for k, v in seen if k.startswith("chronotagger_autosave")]
    assert autosaves and not any(autosaves)
