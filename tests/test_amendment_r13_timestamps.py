"""
DRAFT AMENDMENT R13 (Pack 5 v2) -- the vectorized matplotlib-date ->
Timestamp conversion that closes acceptance gate 1.

Severable: if J.E. declines R13, EDITs 155-161 and this file are struck
together and nothing else in the pack changes.

The contract is BIT-EXACTNESS, not accuracy. A more accurate conversion
exists (read the timestamps off the drawn index and skip the float round
trip) and is deliberately NOT taken -- it deviates by up to 780 ns from
what the shipped loop produces, which is a behaviour change under R3.
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _stub_messagebox(monkeypatch):
    """STANDING RULE (Pack 3): any dialog-reachable path, real-Tk or not."""
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


def _scalar_loop(xs):
    """The per-point loop R13 replaces, verbatim (selection.py at 354be67)."""
    out = []
    for xf in xs:
        dt = mdates.num2date(float(xf))
        if getattr(dt, "tzinfo", None) is not None:
            dt = dt.replace(tzinfo=None)
        out.append(pd.Timestamp(dt))
    return out


def _deviation_ns(got, ref):
    a = pd.DatetimeIndex(got).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    b = pd.DatetimeIndex(ref).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    assert a.shape == b.shape
    return int(np.abs(a - b).max()) if a.size else 0


CASES = {
    # the ordinary case: a 30 s cadence in the 2010s
    "modern_30s": pd.date_range("2015-01-03 00:00:00", periods=4000, freq="30s"),
    # exercises matplotlib's |x| > 70*365 nearest-20-microsecond fixup
    "post_2040": pd.date_range("2044-06-01 00:00:00", periods=4000, freq="30s"),
    # negative day numbers relative to the 1970 epoch
    "pre_1900": pd.date_range("1885-03-01 00:00:00", periods=4000, freq="30s"),
    # sub-microsecond structure, where any rounding slip shows up first
    "cadence_100us": pd.date_range("2015-01-03", periods=4000, freq="100us"),
    # a millisecond cadence, the other side of the rounding boundary
    "cadence_3ms": pd.date_range("2015-01-03", periods=4000, freq="3ms"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_vectorized_conversion_is_bit_exact_with_the_scalar_loop(case):
    """R13's whole contract: 0 ns of deviation, on every branch of
    matplotlib's own conversion."""
    from chronotagger.labeler.utils.fasttime import naive_timestamps_from_num

    idx = CASES[case]
    xs = mdates.date2num(idx.to_numpy())
    got = naive_timestamps_from_num(xs)
    ref = _scalar_loop(xs)

    assert len(got) == len(ref)
    assert _deviation_ns(got, ref) == 0, case
    assert got.tz is None


def test_conversion_returns_an_index_not_a_list():
    """Downstream (positions_nearest, the component dialog) now receives a
    DatetimeIndex; EDIT 110's _probe_index passes an Index through
    untouched, which is why nothing below had to change."""
    from chronotagger.labeler.utils.fasttime import naive_timestamps_from_num
    xs = mdates.date2num(CASES["modern_30s"].to_numpy())
    got = naive_timestamps_from_num(xs)
    assert isinstance(got, pd.DatetimeIndex)
    assert got.dtype == np.dtype("datetime64[us]")


def test_empty_input_is_an_empty_index_not_an_error():
    from chronotagger.labeler.utils.fasttime import naive_timestamps_from_num
    got = naive_timestamps_from_num(np.array([], dtype=float))
    assert isinstance(got, pd.DatetimeIndex)
    assert len(got) == 0


def test_box_select_output_is_unchanged_by_the_amendment(labeler,
                                                        monkeypatch):
    """The end-to-end guarantee: R13 changes how picked_ts is BUILT, never
    what it contains, so the timestamps the mapping is handed -- and the
    spans that get committed -- are the same to the NANOSECOND. Driven
    through the real rectangle-select entry point.

    Recheck finding F-8: the old form asserted only that SOMETHING was
    selected, and survived a +1-DAY poison of the kernel (probes that
    land outside the frame are clamped by method="nearest", so a span
    still existed and its boundary was still a real sample). The kernel
    is now compared against the per-point loop it replaces ON THE
    GESTURE'S OWN x-values, in line, during ONE real gesture -- which
    catches a 1-us slip that the committed spans alone cannot see,
    because a 1-us slip on a 30-s cadence still maps to the same sample.
    (Running the gesture twice and diffing does NOT work: the first
    gesture leaves highlight ink on the panel, so the second scan picks
    a different number of points -- measured 27 then 54.)"""
    import chronotagger.labeler.mixins.events.selection as selmod

    idx = labeler.df.index
    ax = labeler.user_axes["panel1"]
    y0, y1 = ax.get_ylim()

    real_kernel = selmod.naive_timestamps_from_num
    deviations = []
    converted = []

    def _checked(xs):
        got = real_kernel(xs)
        converted.append(len(got))
        deviations.append(_deviation_ns(got, _scalar_loop(xs)))
        return got

    monkeypatch.setattr(selmod, "naive_timestamps_from_num", _checked)

    e1 = type("E", (), {"xdata": mdates.date2num(idx[10]), "ydata": y0,
                        "inaxes": ax, "button": 1})
    e2 = type("E", (), {"xdata": mdates.date2num(idx[40]),
                        "ydata": y0 + (y1 - y0) * 0.6,
                        "inaxes": ax, "button": 1})
    labeler.snap_var.set(True)
    labeler._on_rectangle_select(e1, e2, labeler.active_pane)

    # the kernel really ran, on real picked points
    assert converted, "the amendment's kernel was never reached"
    assert sum(converted) > 0, "the gesture converted no timestamps"
    # and every chunk it produced is bit-identical to the scalar loop
    assert max(deviations) == 0, (
        f"R13 must be 0 ns from the loop it replaces; saw {max(deviations)} ns")

    # and the end-to-end result is a real, sample-aligned selection
    spans = labeler._commit_spans
    assert spans, "the gesture must still select something"
    for s, e in spans:
        assert s in idx
    covered = sum(int(((idx >= s) & (idx < e)).sum()) for s, e in spans)
    assert covered > 0


def test_an_empty_box_still_clears_the_preview(labeler):
    """The emptiness gate became `len(picked_ts) == 0` because a pandas
    Index is not truth-testable. Pin that the gate still fires."""
    idx = labeler.df.index
    ax = labeler.user_axes["panel1"]
    y0, y1 = ax.get_ylim()
    # a y-band far above the data: nothing is inside the box
    e1 = type("E", (), {"xdata": mdates.date2num(idx[10]), "ydata": y1 + 100.0,
                        "inaxes": ax, "button": 1})
    e2 = type("E", (), {"xdata": mdates.date2num(idx[40]), "ydata": y1 + 140.0,
                        "inaxes": ax, "button": 1})
    labeler.current_spans = [(idx[0], idx[1])]
    labeler._on_rectangle_select(e1, e2, labeler.active_pane)

    assert labeler._commit_spans == []
    assert labeler.current_selection is None
    assert "No points in selection" in labeler.status_var.get()
