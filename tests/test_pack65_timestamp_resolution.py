"""Pack 6.5 -- pins for resolution-aware tail timestamps (R65-1..R65-3).

Two bugs, one pack.

R65-1  The Pack 3 last-sample end cap was a hardcoded +1 NANOSECOND. pandas
       3.0 makes MICROSECOND the default resolution for pd.date_range, for
       CSV through pd.to_datetime and for parquet, and on a microsecond
       index that cap is not representable: the end Timestamp promotes to
       ns and idx.searchsorted(end) raises
       ValueError("Cannot losslessly convert units"). Box-select to the end
       of the data, then export per-sample, and the export CRASHED. The cap
       is now one step of the index's OWN resolution.

R65-2  The same cap now also guards the two commit producers Pack 3 parked
       at its Q7: the rule-aware padding tail and the strip drag-resize
       tail. Both used to end a tail span exactly ON the final sample,
       which under half-open [start, end) exports that sample as -1.

R65-4  The tail cap must not RATCHET. _on_strip_press reads the drag's
       CLOSED start/end back out of the stored HALF-OPEN interval, so a
       capped tail end used to leak its epsilon into the drag width and
       every later interior drag covered one extra sample -- permanently,
       with snapping OFF. Folded from the v1 adversarial verifier's
       MAJOR-1.

Every ns-resolution pin below is a BIT-EXACTNESS pin: on the nanosecond
frames this tool is actually driven with, Pack 6.5 must change nothing.

No timing assertion appears anywhere in this file.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytest

from chronotagger.labeler import TimeIntervalLabeler
from chronotagger.core.models import Interval


LAYOUT = {
    "nrows": 2,
    "ncols": 1,
    "areas": [
        {"key": "p1", "row": 0, "col": 0, "rowspan": 1, "colspan": 1,
         "role": "time"},
        {"key": "labels", "row": 1, "col": 0, "rowspan": 1, "colspan": 1,
         "role": "labels"},
    ],
}

# The four pandas datetime64 resolutions, with the epsilon each one implies.
UNIT_STEP = {
    "ns": pd.Timedelta(nanoseconds=1),
    "us": pd.Timedelta(microseconds=1),
    "ms": pd.Timedelta(milliseconds=1),
    "s": pd.Timedelta(seconds=1),
}


def make_frame(unit=None, n=120, values=None):
    """n samples every 30s. unit=None keeps the pandas DEFAULT, which is
    what every other fixture in this suite gets and what the wizard's CSV
    and parquet paths produce."""
    idx = pd.date_range("2015-01-03 00:00:00", periods=n, freq="30s")
    if unit is not None:
        idx = idx.as_unit(unit)
    a = np.linspace(0.0, 1.0, n) if values is None else np.asarray(values,
                                                                  dtype=float)
    return pd.DataFrame({"a": a}, index=idx)


def make_labeler(df, tmp_path, window="60min"):
    lbl = TimeIntervalLabeler(
        df=df,
        plot_fn=lambda axs, dd, t0, t1: axs["p1"].plot(dd.index, dd["a"]),
        window=pd.Timedelta(window),
        autosave_folder=str(tmp_path),
        layout_spec=LAYOUT,
    )
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    return lbl


@pytest.fixture
def labeler_factory(tmp_path):
    made = []

    def _make(unit=None, n=120, values=None, window="60min"):
        lbl = make_labeler(make_frame(unit, n, values), tmp_path, window)
        made.append(lbl)
        return lbl

    yield _make
    for lbl in made:
        lbl.root.destroy()


# =====================================================================
# R65-1 -- the crash, and the cap that fixes it
# =====================================================================

def test_end_cap_matches_the_environments_default_unit(labeler_factory):
    """The premise, restated so it holds on every pandas line CI builds.

    pandas 3.x makes MICROSECOND the default resolution -- for
    pd.date_range, for CSV through pd.to_datetime and for parquet -- and
    that shift is what this whole file is about. pandas 2.x, which is what
    still installs on python 3.9, defaults to NANOSECOND. So the premise
    cannot be "the default is us". It is the CANARY: whatever unit the
    environment's default construction yields, the epsilon is one step of
    THAT unit and the tail end stays placeable in the index.

    If this ever goes red the rest of the file is about a resolution
    pandas no longer hands out, not about a live bug."""
    lbl = labeler_factory()
    idx = lbl.df.index
    unit = idx.unit
    assert unit in UNIT_STEP, "unrecognised default resolution %r" % unit
    assert lbl._index_unit_epsilon() == UNIT_STEP[unit]

    end = lbl._end_after_inclusive(idx[-1])
    assert end == idx[-1] + UNIT_STEP[unit]
    assert end.as_unit(unit, round_ok=False) == end
    assert int(idx.searchsorted(end, side="left")) == len(idx)


def test_tail_box_select_then_export_per_sample_does_not_raise(
        labeler_factory, tmp_path):
    """R65-1, the headline. Box-select to the END of a MICROSECOND frame,
    commit, then export per-sample. Before Pack 6.5 both the id series and
    the export raised ValueError('Cannot losslessly convert units').

    The microsecond frame is FORCED with as_unit instead of taken from
    pandas' construction default: that default is us on pandas 3.x but ns
    on pandas 2.x, and this pin is about the us frame specifically."""
    lbl = labeler_factory(unit="us")
    df = lbl.df
    assert df.index.unit == "us"
    lbl.t0 = df.index[90]
    lbl.t1 = lbl.data_end
    lbl._update_plot()
    lbl.snap_var.set(True)
    lbl._finalize_box_selection([t for t in df.index[109:120]])
    lbl._add_interval()

    iv = lbl.intervals[0]
    assert iv.contains(df.index[-1])
    # REPRESENTABLE in the index's own microseconds -- the property the
    # crash was actually about -- stated as a lossless round trip rather
    # than as iv.end.unit == "us". pandas 2.x builds every numeric
    # Timedelta at ns resolution (pd.Timedelta(1, unit="us").unit ==
    # "ns"), so it labels the correctly-VALUED end "ns"; pandas 3.x labels
    # it "us". The round trip holds on both lines and a +1ns end raises on
    # both lines (measured).
    assert iv.end.as_unit("us", round_ok=False) == iv.end

    ids = lbl._compute_label_id_series()          # raised before Pack 6.5
    assert int(ids.iloc[-1]) != -1

    out = tmp_path / "per_sample.csv"
    lbl.export_per_sample(str(out), fmt="csv")    # raised before Pack 6.5
    assert out.exists()


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_end_cap_is_one_step_of_the_index_resolution(labeler_factory, unit):
    """R65-1 across every pandas datetime64 resolution: the cap is one step
    of the index's own unit, and the capped end stays REPRESENTABLE in that
    unit, so searchsorted and get_indexer both accept it.

    Representability is a lossless as_unit round trip, not end.unit ==
    unit, because pandas 2.x builds every numeric Timedelta at ns
    resolution: the same correct VALUE is labelled "ns" there and
    "us"/"ms"/"s" on pandas 3.x. The round trip and the searchsorted below
    both hold on either line, and both go red on either line if the cap
    reverts to a hardcoded +1ns (measured on pandas 2.3.3 and 3.0.2)."""
    lbl = labeler_factory(unit=unit)
    idx = lbl.df.index
    assert idx.unit == unit

    end = lbl._end_after_inclusive(idx[-1])
    assert end == idx[-1] + UNIT_STEP[unit]
    assert end.as_unit(unit, round_ok=False) == end
    # the whole point: the index can place it
    assert int(idx.searchsorted(end, side="left")) == len(idx)


def test_ns_frame_end_cap_is_bit_exact_one_nanosecond(labeler_factory):
    """BIT-EXACTNESS on the frames this tool is actually driven with. On a
    NANOSECOND index Pack 6.5 must reproduce Pack 3's +1ns to the bit, on
    every path it touches."""
    lbl = labeler_factory(unit="ns")
    idx = lbl.df.index
    one_ns = pd.Timedelta(nanoseconds=1)

    # the helper
    assert lbl._end_after_inclusive(idx[-1]) == idx[-1] + one_ns
    assert lbl._index_unit_epsilon() == one_ns
    # the span converter
    assert (lbl._exact_spans_to_half_open([(idx[115], idx[-1])])
            == [(idx[115], idx[-1] + one_ns)])
    # the runs converter
    assert (lbl._runs_to_half_open_intervals(idx, [(115, len(idx) - 1)])
            == [(idx[115], idx[-1] + one_ns)])
    # the box-select padding lane
    padded = lbl._apply_localized_padding_to_intervals([(idx[115], idx[-1])])
    assert padded[0][1] == idx[-1] + one_ns
    # the value, to the nanosecond
    assert (lbl._end_after_inclusive(idx[-1]).value - idx[-1].value) == 1


def test_us_frame_end_cap_is_one_microsecond_not_one_nanosecond(
        labeler_factory):
    """The behaviour change, stated as a value: on a microsecond index the
    epsilon is 1us, and the old 1ns end is NOT what is produced. The us
    frame is FORCED -- pandas 2.x would otherwise hand this test an ns
    index, on which 1ns IS the right answer."""
    lbl = labeler_factory(unit="us")
    idx = lbl.df.index
    end = lbl._end_after_inclusive(idx[-1])
    assert end == idx[-1] + pd.Timedelta(microseconds=1)
    assert end != idx[-1] + pd.Timedelta(nanoseconds=1)


def test_runs_converter_tail_is_representable_on_a_us_frame(labeler_factory):
    """The second converter (two-click / rule runs) shares the cap. The us
    frame is FORCED for the same reason, and 'representable' is a lossless
    as_unit round trip rather than e.unit, which pandas 2.x reports as
    'ns' even when the VALUE is an exact microsecond."""
    lbl = labeler_factory(unit="us")
    idx = lbl.df.index
    (s, e), = lbl._runs_to_half_open_intervals(idx, [(115, len(idx) - 1)])
    assert e.as_unit("us", round_ok=False) == e
    lbl.intervals = [Interval(s, e, lbl.classes[0])]
    assert int(lbl._compute_label_id_series().iloc[-1]) != -1


def test_epsilon_falls_back_to_ns_without_a_datetime_index():
    """A host with no usable df.index must still produce a cap rather than
    raise -- the helper is on the redraw path."""
    class _Bare:
        _index_unit_epsilon = (
            TimeIntervalLabeler._index_unit_epsilon)
        _end_after_inclusive = (
            TimeIntervalLabeler._end_after_inclusive)

    bare = _Bare()
    assert bare._index_unit_epsilon() == pd.Timedelta(nanoseconds=1)
    bare.df = pd.DataFrame({"a": [1, 2, 3]})      # RangeIndex, no .unit
    assert bare._index_unit_epsilon() == pd.Timedelta(nanoseconds=1)


# =====================================================================
# R65-2 -- the two commit paths Pack 3 parked
# =====================================================================

def _tail_rule_frame(n=120):
    v = np.zeros(n)
    v[108:] = 1.0            # true through the FINAL sample
    return v


def _rule_result(policy="skip", scope="dataset"):
    from chronotagger.labeler.dialogs.label_by_rule import (
        LabelByRuleResult, RuleCondition)
    return LabelByRuleResult(
        conditions=[RuleCondition(column="a", op=">=", value=0.5)],
        combine_mode="AND",
        nan_as_true=False,
        overlap_policy=policy,
        scope=scope,
    )


def test_rule_commit_spans_label_the_final_sample(labeler_factory):
    """R65-2 through the REAL rule entry point: the spans that 'Add Label'
    consumes must cover the last sample the rule matched."""
    lbl = labeler_factory(values=_tail_rule_frame())
    lbl._rule_preview_apply(_rule_result())
    lbl.intervals = [Interval(s, e, lbl.classes[0])
                     for s, e in lbl._commit_spans]
    assert int(lbl._compute_label_id_series().iloc[-1]) != -1


def test_rule_skip_carve_is_not_widened_into_an_existing_interval(
        labeler_factory):
    """The 'skip' policy carves a rule span so it stops exactly AT an
    existing interval's first sample, which under half-open [s, e) is
    already correct. Routing the carved span through
    _exact_spans_to_half_open -- the census's F10 proposal -- pushes that
    end one sample FORWARD and the commit then reaches into the interval it
    was supposed to skip (Pack 6.5, measured)."""
    n = 200
    v = np.zeros(n)
    v[100:121] = 1.0
    lbl = labeler_factory(n=n, values=v, window="120min")
    idx = lbl.df.index
    existing = Interval(idx[103], idx[110], lbl.classes[0])
    lbl.intervals = [existing]
    lbl._rule_preview_apply(_rule_result(policy="skip"))

    assert lbl._commit_spans, "the rule must produce at least one span"
    for s, e in lbl._commit_spans:
        reached = [t for t in idx if s <= t < e and existing.contains(t)]
        assert reached == [], (
            "a 'skip' carve must not reach into the existing interval; "
            "span (%s, %s) covers %s" % (s, e, reached[:3]))


def test_rule_tail_labels_the_final_sample(labeler_factory):
    """R65-2 / census F10b. A rule true through the last sample used to pad
    to exactly data_end, and the half-open export then wrote -1 for the very
    sample the rule matched."""
    lbl = labeler_factory(values=_tail_rule_frame())
    df = lbl.df
    mask = (df["a"].values >= 0.5)
    runs = lbl._mask_to_runs(mask)
    padded = lbl._apply_rule_aware_padding(runs, mask, df.index)

    (s, e), = padded
    assert e > df.index[-1], "the tail cap must clear the final sample"
    assert e.unit == df.index.unit

    lbl.intervals = [Interval(s, e, lbl.classes[0])]
    ids = lbl._compute_label_id_series()
    assert int(ids.iloc[-1]) != -1


def test_rule_interior_spans_are_untouched(labeler_factory):
    """The cap is the TAIL only. Interior rule spans keep their padded ends
    exactly -- this is the pin that would go red if the whole span were
    routed through _exact_spans_to_half_open instead (Pack 6.5)."""
    n = 200
    v = np.zeros(n)
    v[20:31] = 1.0
    v[100:121] = 1.0
    lbl = labeler_factory(n=n, values=v, window="120min")
    df = lbl.df
    mask = (df["a"].values >= 0.5)
    runs = lbl._mask_to_runs(mask)
    padded = lbl._apply_rule_aware_padding(runs, mask, df.index)

    half = (df.index[1] - df.index[0]) / 2
    assert len(padded) == 2
    for (s, e), (i0, i1) in zip(padded, runs):
        # exactly half the local cadence past the run, to the unit: no cap,
        # no converter, no epsilon anywhere near an interior end
        assert e == df.index[i1] + half
        assert s == df.index[i0] - half
        assert e not in df.index


def test_rule_tail_cap_is_bit_exact_on_a_ns_frame(labeler_factory):
    """Bit-exactness of the new rules cap on a nanosecond index."""
    lbl = labeler_factory(unit="ns", values=_tail_rule_frame())
    df = lbl.df
    mask = (df["a"].values >= 0.5)
    padded = lbl._apply_rule_aware_padding(
        lbl._mask_to_runs(mask), mask, df.index)
    assert padded[0][1] == df.index[-1] + pd.Timedelta(nanoseconds=1)


def test_strip_resize_onto_the_final_sample_labels_it(labeler_factory):
    """R65-2 / census F10c. Dragging the right handle onto the last sample
    committed [s, data_end), which does not label data_end."""
    lbl = labeler_factory()
    idx = lbl.df.index
    lbl.snap_var.set(True)
    iv = Interval(idx[100], idx[110], lbl.classes[0])
    lbl.intervals = [iv]
    lbl.selected_interval = iv

    s_new, e_new = lbl._apply_snap_clamp(iv.start, idx[-1])
    assert e_new == lbl.data_end          # the shipped clamp pins it here
    lbl._drag_mode = "resize_right"
    lbl._drag_iv = iv
    lbl._drag_initial = (iv.start, iv.end)
    lbl._drag_preview = (s_new, e_new)
    lbl._on_strip_release(object(), lbl.active_pane)

    out, = lbl.intervals
    assert out.contains(idx[-1])
    assert out.end == idx[-1] + pd.Timedelta(1, unit=idx.unit)
    assert int(lbl._compute_label_id_series().iloc[-1]) != -1


def test_strip_resize_tail_cap_is_bit_exact_on_a_ns_frame(labeler_factory):
    """Bit-exactness of the new strip cap on a nanosecond index."""
    lbl = labeler_factory(unit="ns")
    idx = lbl.df.index
    iv = Interval(idx[100], idx[110], lbl.classes[0])
    lbl.intervals = [iv]
    lbl.selected_interval = iv
    lbl._drag_mode = "resize_right"
    lbl._drag_iv = iv
    lbl._drag_initial = (iv.start, iv.end)
    lbl._drag_preview = (idx[100], lbl.data_end)
    lbl._on_strip_release(object(), lbl.active_pane)
    out, = lbl.intervals
    assert out.end == idx[-1] + pd.Timedelta(nanoseconds=1)


def test_strip_drag_away_from_the_tail_is_byte_unchanged(labeler_factory):
    """The strip cap is the TAIL only: an interior drag commits exactly the
    preview it was handed, with no widening. This is the anti-creep pin --
    routing the whole span through the converter makes it red."""
    lbl = labeler_factory()
    idx = lbl.df.index
    iv = Interval(idx[100], idx[110], lbl.classes[0])
    lbl.intervals = [iv]
    lbl.selected_interval = iv
    lbl._drag_mode = "resize_right"
    lbl._drag_iv = iv
    lbl._drag_initial = (iv.start, iv.end)
    lbl._drag_preview = (idx[100], idx[115])
    lbl._on_strip_release(object(), lbl.active_pane)
    out, = lbl.intervals
    assert (out.start, out.end) == (idx[100], idx[115])


def test_repeated_strip_moves_do_not_widen_the_interval(labeler_factory):
    """The regression the census's whole-span proposal would have shipped:
    each MOVE fed the previous half-open end back in as a closed one and the
    interval grew by a sample per drag. Measured 10 -> 11 -> 12 -> ... under
    both snap states. It must stay flat."""
    lbl = labeler_factory(n=200, window="120min")
    idx = lbl.df.index
    gap = idx[1] - idx[0]
    iv = Interval(idx[20], idx[30], lbl.classes[0])
    lbl.intervals = [iv]
    lbl.selected_interval = iv

    covered = []
    for _ in range(5):
        cur = lbl.intervals[0]
        width = cur.end - cur.start
        new_start = cur.start + 5 * gap
        s_new, e_new = lbl._apply_snap_clamp(new_start, new_start + width)
        lbl._drag_mode = "move"
        lbl._drag_iv = lbl.intervals[0]
        lbl._drag_initial = (cur.start, cur.end)
        lbl._drag_offset = pd.Timedelta(0)
        lbl._drag_preview = (s_new, e_new)
        lbl._on_strip_release(object(), lbl.active_pane)
        lbl.selected_interval = lbl.intervals[0]
        covered.append(int(sum(lbl.intervals[0].contains(t) for t in idx)))

    assert len(set(covered)) == 1, (
        "a repeated move must not change how many samples the interval "
        "covers; got %s" % covered)


# =====================================================================
# R65-4 -- the tail cap must not RATCHET (verifier v1, MAJOR-1)
# =====================================================================

class _Ev:
    """Minimal stand-in for a matplotlib MouseEvent."""


def _press_event(lbl, ts):
    ax = lbl.active_pane.strip_ax
    ev = _Ev()
    ev.button = 1
    ev.inaxes = ax
    ev.xdata = mdates.date2num(pd.Timestamp(ts).to_pydatetime())
    ev.ydata = 0.5
    ev.x = float(ax.transData.transform((ev.xdata, 0.5))[0])
    return ev


def _move_gesture(lbl, delta):
    """One MOVE gesture: a REAL _on_strip_press (which is where the
    anti-ratchet fix lives), the arithmetic _on_strip_motion performs
    verbatim, then a REAL _on_strip_release.

    The mouse POSITION is not round-tripped through
    mdates.date2num/num2date on purpose: that round trip is lossy at about
    0.3 us on a 2015 timestamp, which would make a pin about a 1 us
    epsilon flaky for a reason that has nothing to do with what is being
    tested. Everything the fix touches is real."""
    iv = lbl.selected_interval
    grab = iv.start + (iv.end - iv.start) / 2
    lbl._on_strip_press(_press_event(lbl, grab), lbl.active_pane)
    assert lbl._drag_mode == "move", "expected a move, got %r" % lbl._drag_mode

    s0, e0 = lbl._drag_initial                 # <- the fix's output
    width = e0 - s0                            # <- _on_strip_motion, verbatim
    new_start = s0 + delta
    new_end = new_start + width
    lbl._drag_preview = lbl._apply_snap_clamp(new_start, new_end)

    lbl._on_strip_release(_Ev(), lbl.active_pane)
    lbl.selected_interval = lbl.intervals[0]
    return lbl.intervals[0]


def _covered(lbl):
    iv = lbl.intervals[0]
    return int(sum(iv.contains(t) for t in lbl.df.index))


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_tail_visit_does_not_ratchet_a_later_interior_drag(
        labeler_factory, unit):
    """R65-4 / verifier MAJOR-1, BOTH directions in one pin.

    Dragging an interval so its end lands on data_end caps that end one
    index-unit past it, which is R65-2 working: the tail commit labels the
    final sample. But _on_strip_press used to read that capped end back as
    the drag's CLOSED end, so the epsilon entered the width and every
    later interior drag covered one extra sample -- permanently, with
    snapping OFF (measured 10 -> 11 -> 11, 11, 11 on all four
    resolutions). The cap must apply only to the commit that reaches the
    final sample."""
    from chronotagger.core.models import Interval
    lbl = labeler_factory(unit=unit, n=200, window="200min")
    idx = lbl.df.index
    gap = idx[1] - idx[0]
    lbl.snap_var.set(False)                    # the un-immune case
    lbl.intervals = [Interval(idx[20], idx[30], lbl.classes[0])]
    lbl.selected_interval = lbl.intervals[0]
    assert _covered(lbl) == 10

    # direction 1: the tail commit still labels the last sample
    _move_gesture(lbl, lbl.data_end - lbl.intervals[0].end)
    assert lbl.intervals[0].contains(idx[-1]), (
        "the tail commit must still label the final sample")
    assert lbl.intervals[0].end == idx[-1] + UNIT_STEP[unit]
    assert _covered(lbl) == 11

    # direction 2: six interior moves afterwards, flat -- no ratchet
    after = []
    for _ in range(6):
        _move_gesture(lbl, -10 * gap)
        after.append(_covered(lbl))
    assert after == [10] * 6, (
        "a tail visit must not widen later interior drags; got %s" % after)


def test_tail_visit_does_not_ratchet_with_snapping_on(labeler_factory):
    """The snap-ON lane was already immune (the re-snap threw the epsilon
    away). It must stay immune -- a fix that traded one ratchet for
    another would be caught here."""
    from chronotagger.core.models import Interval
    lbl = labeler_factory(n=200, window="200min")
    idx = lbl.df.index
    gap = idx[1] - idx[0]
    lbl.snap_var.set(True)
    lbl.intervals = [Interval(idx[20], idx[30], lbl.classes[0])]
    lbl.selected_interval = lbl.intervals[0]

    _move_gesture(lbl, lbl.data_end - lbl.intervals[0].end)
    assert lbl.intervals[0].contains(idx[-1])
    after = [(_move_gesture(lbl, -10 * gap), _covered(lbl))[1]
             for _ in range(6)]
    assert after == [10] * 6, after


def test_press_uncaps_a_tail_interval_and_leaves_others_alone(
        labeler_factory):
    """The mechanism itself, as a unit pin: _on_strip_press hands the drag
    a CLOSED end. For a tail-capped interval that is data_end exactly, not
    data_end plus the epsilon; for any other interval it is the stored end
    unchanged, byte for byte."""
    from chronotagger.core.models import Interval
    lbl = labeler_factory(n=200, window="200min")
    idx = lbl.df.index

    tail = Interval(idx[180], lbl._end_after_inclusive(lbl.data_end),
                    lbl.classes[0])
    lbl.intervals = [tail]
    lbl.selected_interval = tail
    grab = tail.start + (tail.end - tail.start) / 2
    lbl._on_strip_press(_press_event(lbl, grab), lbl.active_pane)
    assert lbl._drag_initial == (idx[180], lbl.data_end)
    assert lbl._drag_initial[1] != tail.end
    lbl._drag_mode = None

    interior = Interval(idx[20], idx[30], lbl.classes[0])
    lbl.intervals = [interior]
    lbl.selected_interval = interior
    grab = interior.start + (interior.end - interior.start) / 2
    lbl._on_strip_press(_press_event(lbl, grab), lbl.active_pane)
    assert lbl._drag_initial == (interior.start, interior.end)


# =====================================================================
# absence / anchor checks
# =====================================================================

def test_no_hardcoded_nanosecond_end_cap_survives_in_src():
    """The three +1ns end-cap CALLS the census anchored are gone from src/.

    AST, not grep: _index_unit_epsilon's docstring names the old expression
    on purpose, and a grep-shaped version of this test would trip on it and
    would not discriminate a real regression from a comment."""
    import ast

    import chronotagger
    root = Path(chronotagger.__file__).resolve().parent
    hits = []
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and fn.attr == "Timedelta"):
                continue
            if any(kw.arg == "nanoseconds" for kw in node.keywords):
                hits.append("%s:%d" % (p.name, node.lineno))
    assert hits == [], "hardcoded +1ns Timedelta calls left in src/: %s" % hits
