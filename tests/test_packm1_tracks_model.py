"""
Pack M1 pins -- THE MODEL: the track table, the two new Interval fields,
the per-track invariant, the per-track fuse, the track-scoped add path,
all ELEVEN Interval(...) construction sites -- and, at the end, the four
SCREEN READERS of PART G, which are the only pins in this pack that need a
real axes and a real Treeview.

Every test here is GUI-free or runs on the withdrawn-window `labeler`
fixture. CHRONOTAGGER_STRICT is 1 for the whole suite (conftest.py), so
the strict invariants below are live.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from chronotagger.core.commands import (
    AddIntervalCommand,
    IntervalInvariantError,
    ResizeIntervalCommand,
    check_interval_invariants,
    copy_intervals,
    copy_meta,
)
from chronotagger.core.models import Interval
from chronotagger.core.tracks import (
    DEFAULT_TRACK_ID,
    DEFAULT_TRACK_NAME,
    Track,
    active_id_of,
    colors_for,
    copy_tracks,
    default_table,
    find_track,
    active_track_of,
    intervals_on,
    snapshot_tracks,
    table_of,
    stray_tracks,
    union_covered,
    validate_track_id,
)
from chronotagger.labeler.mixins.intervals import IntervalsMixin


# --------------------------------------------------------------- harness

class _Var:
    def __init__(self, v=""):
        self._v = v

    def set(self, v):
        self._v = v

    def get(self):
        return self._v


class _Sync:
    def sync_intervals_changed(self):
        pass


class Host:
    """GUI-free host binding the real interval mixin methods, with a
    track table -- the shape TimeIntervalLabeler has."""

    _BOUND = [
        "_execute_command", "_gesture",
        "_check_interval_invariants", "_check_interval_invariants_on",
        "_repoint_selected_interval", "_undo", "_redo",
        "_remove_overlapping_intervals", "_sort_and_merge_intervals",
        "_carve_existing_for_new_span", "_apply_overlap_policy_to_spans",
        "_subtract_overlaps_from_span", "_count_overlapping_intervals",
        "_clear_intervals_in_range", "_assign_gaps_to_label",
        "_add_intervals_with_policy",
    ]

    def __init__(self, df, tracks=None):
        self.df = df
        self.intervals = []
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo = 100
        self.modified = False
        self.selected_interval = None
        self.tracks = tracks if tracks is not None else default_table(
            ["UNKNOWN", "A", "B"])
        self._active_track_id = self.tracks[0].id
        self.t0 = df.index[0]
        self.t1 = df.index[-1]
        self.data_start = df.index[0]
        self.data_end = df.index[-1]
        self.status_var = _Var()
        self.sync_manager = _Sync()
        for name in self._BOUND:
            setattr(self, name, getattr(IntervalsMixin, name).__get__(self))

    def _update_plot(self):
        pass

    def _save_autosave(self):
        pass


def _frame(periods=200, freq="1min"):
    idx = pd.date_range("2024-01-01 00:00:00", periods=periods, freq=freq)
    return pd.DataFrame({"v": np.linspace(0.0, 1.0, periods)}, index=idx)


def T(hhmm):
    return pd.Timestamp("2024-01-01 %s:00" % hhmm)


@pytest.fixture
def host():
    return Host(_frame())


@pytest.fixture
def two_track_host():
    tracks = [
        Track(id="human", name="Human", classes=["UNKNOWN", "sw", "msh"]),
        Track(id="rules", name="Rules", classes=["sw", "msh", "umbra"],
              locked=True),
    ]
    return Host(_frame(), tracks=tracks)


# ============================================================= the fields

def test_interval_has_exactly_six_fields():
    """The staleness guard copy_intervals buys itself.

    copy_intervals names all six fields explicitly. A SEVENTH field added
    without touching it would be erased by every undo, in silence. This
    turns that into a RED test.
    """
    names = [f.name for f in dataclasses.fields(Interval)]
    assert names == ["start", "end", "label", "notes", "track", "meta"]
    assert len(dataclasses.fields(Interval)) == 6


def test_the_interval_track_default_is_the_default_track_id():
    """Load-bearing: three of the eleven construction sites used to pass
    three positional arguments, and every bare Interval(a, b, "A") in the
    suite still does."""
    assert DEFAULT_TRACK_ID == "default"
    fields = {f.name: f for f in dataclasses.fields(Interval)}
    assert fields["track"].default == DEFAULT_TRACK_ID
    assert Interval(T("00:10"), T("00:20"), "A").track == DEFAULT_TRACK_ID
    assert Interval(T("00:10"), T("00:20"), "A").meta == {}
    assert default_table(["A"])[0].id == DEFAULT_TRACK_ID
    assert default_table(["A"])[0].name == DEFAULT_TRACK_NAME


def test_two_intervals_with_different_tracks_are_different_values():
    """Interval.__eq__ is field-wise, and the gesture no-op test depends
    on track and meta being part of it."""
    a = Interval(T("00:10"), T("00:20"), "A", track="human")
    b = Interval(T("00:10"), T("00:20"), "A", track="rules")
    assert a != b
    c = Interval(T("00:10"), T("00:20"), "A", meta={"x": 1})
    d = Interval(T("00:10"), T("00:20"), "A")
    assert c != d


# ======================================================= the serializer

def test_to_dict_emits_track_always_and_meta_only_when_present():
    plain = Interval(T("00:10"), T("00:20"), "A").to_dict()
    assert plain["track"] == DEFAULT_TRACK_ID
    assert "meta" not in plain
    assert sorted(plain) == ["end", "label", "notes", "start", "track"]

    tagged = Interval(T("00:10"), T("00:20"), "A",
                      meta={"import.source": "column:x"}).to_dict()
    assert tagged["meta"] == {"import.source": "column:x"}


def test_from_dict_reads_a_v1_interval_as_the_default_track():
    """A four-key v1 interval is a DEFAULT for an absent field, in the
    same shape `notes` has always had."""
    v1 = {"start": "2024-01-01T00:10:00", "end": "2024-01-01T00:20:00",
          "label": "A", "notes": "kept"}
    iv = Interval.from_dict(v1)
    assert iv.track == DEFAULT_TRACK_ID
    assert iv.meta == {}
    assert iv.notes == "kept"


def test_to_dict_hands_out_a_COPY_of_meta_not_the_live_dict():
    """v2 fold F8. R2 makes these two methods the choke point all 16
    serialization sites route through, so a caller that post-processes
    the payload must not be able to edit the MODEL through it. Measured
    on v1 of this pack: d["meta"] is iv.meta was True, and appending to
    d["meta"]["k"] changed the interval."""
    iv = Interval(T("00:10"), T("00:20"), "A", None, "human", {"k": [1]})
    d = iv.to_dict()
    assert d["meta"] == {"k": [1]}
    assert d["meta"] is not iv.meta
    d["meta"]["k"].append(2)
    assert iv.meta == {"k": [1]}


def test_from_dict_does_not_keep_the_payloads_nested_values():
    """v2 fold F8, the other half of the same door: a parsed JSON
    payload that is mutated afterwards must not reach into the interval
    it produced."""
    src = {"k": [1], "flat": 3}
    d = {"start": "2024-01-01T00:10:00", "end": "2024-01-01T00:20:00",
         "label": "A", "notes": None, "track": "human", "meta": src}
    iv = Interval.from_dict(d)
    assert iv.meta == src
    assert iv.meta["k"] is not src["k"]
    src["k"].append(2)
    assert iv.meta == {"k": [1], "flat": 3}


def test_meta_round_trips_verbatim_including_nested_content():
    meta = {"zzz_nonsense_key": [1, {"deep": "value"}], "pi": 3.5}
    iv = Interval(T("00:10"), T("00:20"), "A", "n", "human", meta)
    back = Interval.from_dict(iv.to_dict())
    assert back.meta == meta
    assert back.track == "human"


# ========================================================= copy_intervals

def test_copy_intervals_carries_track_and_meta():
    src = [Interval(T("00:10"), T("00:20"), "A", "n", "human",
                    {"k": [1, 2]})]
    cp = copy_intervals(src)
    assert cp[0].track == "human"
    assert cp[0].meta == {"k": [1, 2]}
    assert cp == src


def test_copy_intervals_deep_copies_meta_so_a_snapshot_cannot_be_edited():
    """A SHALLOW dict(iv.meta) shares nested lists between snapshots,
    which lets an undo edit the future. Measured, and refused."""
    src = [Interval(T("00:10"), T("00:20"), "A", meta={"k": ["value"]})]
    cp = copy_intervals(src)
    cp[0].meta["k"].append("mutated")
    assert src[0].meta == {"k": ["value"]}
    assert copy_meta({}) == {}
    assert copy_meta(None) == {}


def test_a_flat_meta_is_copied_not_shared():
    """The ingest stamps a FLAT three-scalar meta on every interval it
    makes, so that shape is copied 2,349 times per gesture on a session
    with an imported track. copy_meta takes a SHALLOW copy there --
    measured at 2,000 such intervals, 6.58 ms deep against 3.13 ms
    shallow -- which is safe precisely because a flat dict holds nothing
    mutable. The dict itself must still be a NEW one, or two snapshots
    would write into each other.
    """
    flat = {"import.source": "column:rule_label",
            "import.gap_tolerance": "P0DT0H27M31S",
            "import.run_rows": 84}
    src = [Interval(T("00:10"), T("00:20"), "A", meta=dict(flat))]
    cp = copy_intervals(src)
    assert cp[0].meta == flat
    assert cp[0].meta is not src[0].meta
    cp[0].meta["import.run_rows"] = 999
    assert src[0].meta["import.run_rows"] == 84
    # ... and one nested value anywhere in it sends the whole dict deep
    nested = dict(flat)
    nested["n"] = [1, 2]
    out = copy_meta(nested)
    out["n"].append(3)
    assert nested["n"] == [1, 2]


def test_copy_intervals_keeps_value_equality_for_a_noop_gesture(host):
    host.intervals = [Interval(T("00:10"), T("00:20"), "A",
                               meta={"k": [1]})]
    before = copy_intervals(host.intervals)
    after = copy_intervals(host.intervals)
    assert after == before


# ============================================================ Track rules

def test_a_track_id_is_validated_to_a_filename_and_column_safe_charset():
    assert validate_track_id("rules-2") == "rules-2"
    for bad in ["", "a/b", "a.b", "a b", "x" * 33, 7, None]:
        with pytest.raises(ValueError):
            validate_track_id(bad)
    with pytest.raises(ValueError):
        Track(id="a/b", classes=["A"])


def test_a_track_kind_must_be_interval_or_point():
    assert Track(id="p", classes=["A"], kind="point").kind == "point"
    with pytest.raises(ValueError):
        Track(id="p", classes=["A"], kind="blob")


def test_a_point_track_round_trips_through_the_table():
    """kind="point" CONSTRUCTS and round-trips. Nothing renders it: a
    point interval paints zero pixels and Interval.contains is
    start <= ts < end, so it is unselectable. That is M4's work and it is
    deliberate -- reserving a value you refuse to construct is not a
    reservation."""
    t = Track(id="xings", name="Crossings", classes=["shock"], kind="point")
    back = Track.from_dict(t.to_dict())
    assert back.kind == "point"
    assert back == t


def test_copy_tracks_is_a_value_copy():
    src = [Track(id="human", classes=["A"], class_colors={"A": "#111111"})]
    cp = copy_tracks(src)
    cp[0].classes.append("B")
    cp[0].class_colors["A"] = "#222222"
    assert src[0].classes == ["A"]
    assert src[0].class_colors == {"A": "#111111"}


def test_track_helpers_answer_the_questions_the_readers_ask():
    tbl = [Track(id="human", classes=["A"]), Track(id="rules", classes=["A"])]
    assert find_track(tbl, "rules").id == "rules"
    assert find_track(tbl, "ghost") is None
    ivs = [Interval(T("00:10"), T("00:20"), "A", track="human"),
           Interval(T("00:30"), T("00:40"), "A", track="ghost")]
    assert [iv.track for iv in intervals_on(ivs, "human")] == ["human"]
    assert stray_tracks(ivs, tbl) == ["ghost"]
    assert colors_for(["A", "B"])["A"].startswith("#")
    # the caller's palette is what gets used, and it wraps at its length
    assert colors_for(["A", "B", "C"], palette=["#111111", "#222222"]) == {
        "A": "#111111", "B": "#222222", "C": "#111111"}
    assert DEFAULT_TRACK_NAME == "Labels"


def test_union_covered_is_the_union_and_not_the_sum():
    """With one track the union IS the sum, to the nanosecond. With two
    it is bounded by the record."""
    one = [Interval(T("00:00"), T("00:10"), "A"),
           Interval(T("00:20"), T("00:30"), "A")]
    assert union_covered(one) == pd.Timedelta("20min")
    two = one + [Interval(T("00:05"), T("00:25"), "A", track="rules")]
    assert union_covered(two) == pd.Timedelta("30min")
    assert union_covered([]) == pd.Timedelta(0)


# ================================================= the labeler's contract

def _labeler(df, **kw):
    from chronotagger.labeler import TimeIntervalLabeler

    def fn(axs, sub, t0, t1):
        pass

    return TimeIntervalLabeler(df=df, plot_fn=fn, autosave_folder=".", **kw)


def test_classes_is_a_view_over_the_active_track():
    lbl = _labeler(_frame(), classes=["UNKNOWN", "PS", "LOBE"])
    assert lbl.tracks[0].id == DEFAULT_TRACK_ID
    assert lbl.classes == ["UNKNOWN", "PS", "LOBE"]
    assert lbl.active_track_id == DEFAULT_TRACK_ID
    lbl.tracks[0].classes.append("NEW")
    assert lbl.classes[-1] == "NEW"


def test_the_classes_setter_writes_through_to_the_active_track():
    lbl = _labeler(_frame(), classes=["UNKNOWN", "PS"])
    lbl.classes = ["UNKNOWN", "X"]
    assert lbl.tracks[0].classes == ["UNKNOWN", "X"]
    lbl.class_colors = {"UNKNOWN": "#000000", "X": "#ffffff"}
    assert lbl.tracks[0].class_colors["X"] == "#ffffff"


def test_the_constructor_refuses_both_tracks_and_classes():
    with pytest.raises(ValueError) as e:
        _labeler(_frame(), classes=["A"],
                 tracks=[{"id": "human", "classes": ["A"]}])
    assert "tracks" in str(e.value) and "classes" in str(e.value)


def test_the_tracks_argument_builds_the_table_and_normalises_order():
    lbl = _labeler(_frame(), tracks=[
        {"id": "human", "name": "Human", "classes": ["UNKNOWN", "sw"],
         "order": 99},
        Track(id="rules", classes=["sw", "umbra"], locked=True),
    ])
    assert [t.id for t in lbl.tracks] == ["human", "rules"]
    assert [t.order for t in lbl.tracks] == [0, 1]
    assert lbl.tracks[1].locked is True
    # colours are filled in from the palette, per class, per track
    assert set(lbl.tracks[0].class_colors) == {"UNKNOWN", "sw"}
    assert lbl.active_track_id == "human"
    assert lbl.classes == ["UNKNOWN", "sw"]
    assert lbl.track_ids == ["human", "rules"]
    assert lbl.track_by_id("rules").name == "rules"


def test_the_tracks_argument_refuses_a_duplicate_id_or_an_empty_table():
    with pytest.raises(ValueError):
        _labeler(_frame(), tracks=[])
    with pytest.raises(ValueError) as e:
        _labeler(_frame(), tracks=[{"id": "x", "classes": ["A"]},
                                   {"id": "x", "classes": ["B"]}])
    assert "duplicate track id" in str(e.value)
    with pytest.raises(ValueError) as e2:
        _labeler(_frame(), tracks=[{"id": "x", "classes": []}])
    assert "no classes" in str(e2.value)


def test_two_tracks_over_one_vocabulary_share_colours_until_m3():
    """Stated, not hidden: the default palette is DEFAULT_COLORS[i % 10]
    per class PER TRACK, so two tracks over the same class names get the
    same colours. Set class_colors per track to tell two annotators apart
    by colour."""
    lbl = _labeler(_frame(), tracks=[{"id": "a", "classes": ["X", "Y"]},
                                     {"id": "b", "classes": ["X", "Y"]}])
    assert lbl.tracks[0].class_colors == lbl.tracks[1].class_colors


# ================================================== the strict invariants

def test_a_cross_track_overlap_is_legal(two_track_host):
    h = two_track_host
    h.intervals = [
        Interval(T("00:00"), T("02:00"), "sw", track="human"),
        Interval(T("01:00"), T("03:00"), "umbra", track="rules"),
    ]
    h._check_interval_invariants()          # must not raise


def test_a_within_track_overlap_raises_and_names_the_track(two_track_host):
    h = two_track_host
    h.intervals = [
        Interval(T("00:00"), T("02:00"), "sw", track="rules"),
        Interval(T("01:00"), T("03:00"), "umbra", track="rules"),
    ]
    with pytest.raises(IntervalInvariantError) as e:
        h._check_interval_invariants()
    assert "track 'rules'" in str(e.value)


def test_an_interval_on_a_track_the_table_does_not_know_raises(two_track_host):
    h = two_track_host
    h.intervals = [Interval(T("00:00"), T("01:00"), "sw", track="ghost")]
    with pytest.raises(IntervalInvariantError) as e:
        h._check_interval_invariants()
    assert "not in the track table" in str(e.value)
    assert "ghost" in str(e.value)


def test_the_per_track_scan_sorts_before_it_compares():
    """v2 fold F12, CORRECTED AT THE SEAL.

    Lane 1 asked for a REVERSED OVERLAPPING pair here. Measured: that
    pins nothing, because the pair test `a.end > b.start` is not
    symmetric -- fed [later, earlier] it compares the LATER interval's
    end against the EARLIER one's start and raises with or without the
    sort. (The seal ran it: 0 RED.) And no reversed-overlap case can
    ever be MISSED, because a list whose consecutive pairs all satisfy
    end <= next.start is necessarily sorted and overlap-free.

    What the sort actually buys is the absence of a FALSE POSITIVE, and
    that is what this pins. The invariant also runs on a set read OFF
    DISK, whose interval ORDER is whatever the file says, and without
    the sort a perfectly LEGAL set in the wrong list order raises
    "overlapping intervals" at a user who has none.
    """
    a = Interval(T("00:00"), T("00:10"), "A", None, DEFAULT_TRACK_ID, {})
    b = Interval(T("00:20"), T("00:30"), "A", None, DEFAULT_TRACK_ID, {})
    tbl = default_table(["A"])
    # REVERSED and LEGAL: this must not raise. Without the sort it does.
    check_interval_invariants([b, a], tbl)
    # ... and a real overlap is still caught whatever the list order.
    c = Interval(T("00:05"), T("00:25"), "A", None, DEFAULT_TRACK_ID, {})
    with pytest.raises(IntervalInvariantError):
        check_interval_invariants([c, a], tbl)
    with pytest.raises(IntervalInvariantError):
        check_interval_invariants([a, c], tbl)


def test_the_membership_clause_uses_the_table_being_installed():
    """The first prototype validated a LOADED set against the OLD table
    and a legal v1 migration failed. The table argument is the fix."""
    new_tracks = default_table(["A"])
    ivs = [Interval(T("00:00"), T("01:00"), "A")]
    check_interval_invariants(ivs, new_tracks)       # must not raise
    with pytest.raises(IntervalInvariantError):
        check_interval_invariants(ivs, [Track(id="human", classes=["A"])])


def test_the_membership_clause_is_skipped_when_there_is_no_table():
    """A host with no table has nothing to be a member of; the per-track
    scan still runs."""
    ivs = [Interval(T("00:00"), T("01:00"), "A", track="whatever")]
    check_interval_invariants(ivs, None)
    with pytest.raises(IntervalInvariantError):
        check_interval_invariants(
            ivs + [Interval(T("00:30"), T("01:30"), "A", track="whatever")],
            None)


# =========================================== the gesture and the table

def test_the_gesture_snapshot_carries_the_track_table(host):
    with host._gesture("add a track"):
        host.tracks.append(Track(id="rules", classes=["A"]))
    assert len(host.undo_stack) == 1
    assert host.undo_stack[-1].tracks_after is not None
    host._undo()
    assert [t.id for t in host.tracks] == [DEFAULT_TRACK_ID]
    host._redo()
    assert [t.id for t in host.tracks] == [DEFAULT_TRACK_ID, "rules"]


def test_creating_a_track_alone_still_pushes_an_undo_entry(host):
    """It adds no interval, so under an intervals-only no-op test it would
    push nothing and be silently non-undoable."""
    before = len(host.undo_stack)
    with host._gesture("create"):
        host.tracks.append(Track(id="rules", classes=["A"]))
    assert len(host.undo_stack) == before + 1


def test_a_gesture_that_changes_nothing_still_pushes_nothing(host):
    host.intervals = [Interval(T("00:10"), T("00:20"), "A")]
    before = len(host.undo_stack)
    with host._gesture("noop"):
        pass
    assert len(host.undo_stack) == before


def test_a_failing_gesture_rolls_back_the_table_and_the_intervals(host):
    host.intervals = [Interval(T("00:10"), T("00:20"), "A")]
    with pytest.raises(RuntimeError):
        with host._gesture("bad"):
            host.tracks.append(Track(id="rules", classes=["A"]))
            host.intervals.append(Interval(T("00:30"), T("00:40"), "A"))
            raise RuntimeError("boom")
    assert [t.id for t in host.tracks] == [DEFAULT_TRACK_ID]
    assert len(host.intervals) == 1


def test_a_membership_violation_inside_a_gesture_rolls_back(two_track_host):
    """The safety net for a missed construction site: the locked track
    comes out unchanged."""
    h = two_track_host
    h.intervals = [
        Interval(T("00:10"), T("00:30"), "umbra", track="rules"),
        Interval(T("00:40"), T("00:50"), "sw", track="rules"),
    ]
    with pytest.raises(IntervalInvariantError):
        with h._gesture("a missed site"):
            h.intervals.append(Interval(T("01:00"), T("01:10"), "sw",
                                        track="ghost"))
    assert [(iv.track, iv.label) for iv in h.intervals] == [
        ("rules", "umbra"), ("rules", "sw")]


def test_a_host_without_a_table_snapshots_none(host):
    del host.tracks
    assert snapshot_tracks(host) is None
    with host._gesture("add"):
        host.intervals.append(Interval(T("00:10"), T("00:20"), "A"))
    assert host.undo_stack[-1].tracks_before is None
    host._undo()
    assert host.intervals == []


# ================================================================ the fuse

def test_abutting_same_label_intervals_on_different_tracks_do_not_fuse(
        two_track_host):
    """merge.py:28 used to ABSORB one track into another: a `storm` on
    humanA ending where a `storm` on modelB begins fused into ONE interval
    carrying humanA. That is the ensemble case this program exists for."""
    h = two_track_host
    h.intervals = [
        Interval(T("00:00"), T("01:00"), "sw", track="human"),
        Interval(T("01:00"), T("02:00"), "sw", track="rules"),
    ]
    h._sort_and_merge_intervals()
    assert len(h.intervals) == 2
    assert sorted(iv.track for iv in h.intervals) == ["human", "rules"]


def test_abutting_same_label_intervals_on_the_same_track_do_fuse(host):
    host.intervals = [
        Interval(T("00:00"), T("01:00"), "A"),
        Interval(T("01:00"), T("02:00"), "A"),
        Interval(T("02:00"), T("03:00"), "B"),
    ]
    host._sort_and_merge_intervals()
    assert [(iv.start, iv.end, iv.label) for iv in host.intervals] == [
        (T("00:00"), T("02:00"), "A"), (T("02:00"), T("03:00"), "B")]


def test_the_merge_leaves_the_list_sorted_by_start(two_track_host):
    """The fuse walks in (track, start) order and must RE-SORT by start
    before it rebinds. This case is chosen so the two orders DISAGREE --
    `human` sorts first by track but LAST by start -- because a case where
    they agree cannot tell the re-sort from its absence.
    """
    h = two_track_host
    h.intervals = [
        Interval(T("03:00"), T("04:00"), "sw", track="human"),
        Interval(T("00:00"), T("01:00"), "umbra", track="rules"),
        Interval(T("01:30"), T("02:00"), "msh", track="rules"),
    ]
    h._sort_and_merge_intervals()
    starts = [iv.start for iv in h.intervals]
    assert starts == [T("00:00"), T("01:30"), T("03:00")]
    assert starts == sorted(starts)


# ============================================================ the add path

def test_a_cross_track_add_counts_no_overlap_and_a_within_track_one_does(
        two_track_host):
    """Unscoped, 200 of 200 adds aimed at one track over a fully covered
    other track raised the overlap modal. Scoped, 0 of 200."""
    h = two_track_host
    h.intervals = [Interval(T("00:00"), T("03:00"), "umbra", track="rules")]
    spans = [(T("01:00"), T("02:00"))]
    h._active_track_id = "human"
    assert h._count_overlapping_intervals(spans) == 0
    assert h._count_overlapping_intervals(spans, track="rules") == 1
    h._active_track_id = "rules"
    assert h._count_overlapping_intervals(spans) == 1


def test_a_cross_track_add_leaves_the_locked_track_untouched(two_track_host):
    """The measured disaster: the locked track's interval was trimmed,
    deleted and replaced by two fragments on a track called 'default'
    that was not even in the table."""
    h = two_track_host
    h._active_track_id = "human"
    h.intervals = [
        Interval(T("00:10"), T("00:30"), "umbra", track="rules"),
        Interval(T("00:40"), T("00:50"), "sw", track="rules"),
    ]
    h._add_intervals_with_policy([(T("00:16"), T("00:23"))], "sw", "replace")
    on_rules = [(iv.start, iv.end, iv.label) for iv in h.intervals
                if iv.track == "rules"]
    assert on_rules == [(T("00:10"), T("00:30"), "umbra"),
                        (T("00:40"), T("00:50"), "sw")]
    on_human = [(iv.start, iv.end, iv.label) for iv in h.intervals
                if iv.track == "human"]
    assert on_human == [(T("00:16"), T("00:23"), "sw")]


def test_the_skip_policy_only_subtracts_the_target_track(two_track_host):
    h = two_track_host
    h._active_track_id = "human"
    h.intervals = [Interval(T("00:00"), T("03:00"), "umbra", track="rules")]
    pieces = h._apply_overlap_policy_to_spans(
        [(T("01:00"), T("02:00"))], "skip")
    assert pieces == [(T("01:00"), T("02:00"))]
    assert h._subtract_overlaps_from_span(
        T("01:00"), T("02:00"), track="rules") == []


# ================================== the ELEVEN construction sites, by site

def test_site_1_undo_and_redo_carry_track_and_meta(two_track_host):
    h = two_track_host
    h._active_track_id = "human"
    with h._gesture("add"):
        h.intervals.append(Interval(T("00:10"), T("00:20"), "sw", "n",
                                    "human", {"src": "probe"}))
    h._undo()
    assert h.intervals == []
    h._redo()
    assert h.intervals[0].track == "human"
    assert h.intervals[0].meta == {"src": "probe"}


def test_site_2_a_resize_keeps_the_track_and_the_meta(two_track_host):
    h = two_track_host
    h._active_track_id = "human"
    iv = Interval(T("00:10"), T("00:20"), "sw", "n", "human", {"k": [1]})
    h.intervals = [iv]
    h._execute_command(ResizeIntervalCommand(h, iv, T("00:12"), T("00:22")))
    assert len(h.intervals) == 1
    assert h.intervals[0].track == "human"
    assert h.intervals[0].meta == {"k": [1]}
    assert (h.intervals[0].start, h.intervals[0].end) == (T("00:12"),
                                                          T("00:22"))


def test_sites_3_and_4_an_add_lands_on_the_active_track(two_track_host):
    h = two_track_host
    h._active_track_id = "human"
    h._add_intervals_with_policy([(T("00:10"), T("00:20"))], "sw", "skip")
    h._add_intervals_with_policy([(T("01:10"), T("01:20"))], "msh",
                                 "replace")
    assert sorted(iv.track for iv in h.intervals) == ["human", "human"]


def test_sites_5_and_6_a_clear_range_split_inherits_track_and_meta(
        two_track_host):
    h = two_track_host
    h._active_track_id = "human"
    h.intervals = [Interval(T("00:00"), T("03:00"), "sw", "note", "human",
                            {"import.source": "column:x"})]
    h._clear_intervals_in_range(T("01:00"), T("02:00"))
    frags = sorted(h.intervals, key=lambda iv: iv.start)
    assert len(frags) == 2
    for f in frags:
        assert f.track == "human"
        assert f.meta == {"import.source": "column:x"}
        assert f.notes == "note"
    # a DEEP copy: editing one fragment's meta must not touch the other
    frags[0].meta["import.source"] = "edited"
    assert frags[1].meta["import.source"] == "column:x"


def test_site_7_a_gap_fill_lands_on_the_active_track(two_track_host):
    h = two_track_host
    h._active_track_id = "human"
    h._assign_gaps_to_label([(T("00:10"), T("00:20"))], "sw")
    assert [iv.track for iv in h.intervals] == ["human"]


def test_sites_8_and_9_a_carve_fragment_inherits_track_and_meta(
        two_track_host):
    h = two_track_host
    h._active_track_id = "human"
    h.intervals = [Interval(T("00:00"), T("03:00"), "sw", "note", "human",
                            {"k": ["v"]})]
    h._carve_existing_for_new_span(T("01:00"), T("02:00"))
    frags = sorted(h.intervals, key=lambda iv: iv.start)
    assert len(frags) == 2
    for f in frags:
        assert f.track == "human"
        assert f.meta == {"k": ["v"]}
    frags[0].meta["k"].append("mutated")
    assert frags[1].meta == {"k": ["v"]}


def test_sites_10_and_11_a_trim_inherits_track_and_meta(two_track_host):
    h = two_track_host
    h._active_track_id = "human"
    h.intervals = [Interval(T("00:00"), T("03:00"), "sw", "note", "human",
                            {"k": ["v"]})]
    removed, trims = h._remove_overlapping_intervals(
        Interval(T("01:00"), T("02:00"), "msh", track="human"))
    assert len(removed) == 1 and len(trims) == 2
    for t in trims:
        assert t.track == "human"
        assert t.meta == {"k": ["v"]}
    trims[0].meta["k"].append("mutated")
    assert trims[1].meta == {"k": ["v"]}


def test_the_trim_never_touches_another_track(two_track_host):
    h = two_track_host
    h.intervals = [Interval(T("00:00"), T("03:00"), "umbra", track="rules")]
    removed, trims = h._remove_overlapping_intervals(
        Interval(T("01:00"), T("02:00"), "sw", track="human"))
    assert removed == [] and trims == []
    assert len(h.intervals) == 1


def test_a_host_with_no_table_reads_as_one_default_track(host):
    """The tolerant reader: a host built before tracks existed behaves as
    exactly ONE default track whose vocabulary is its own `classes`,
    rather than raising AttributeError."""
    host.classes = ["UNKNOWN", "A", "B"]      # what such a host HAS
    host.class_colors = {"UNKNOWN": "#cccccc"}
    del host.tracks
    del host._active_track_id
    assert active_id_of(host) == DEFAULT_TRACK_ID
    tbl = table_of(host)
    assert [t.id for t in tbl] == [DEFAULT_TRACK_ID]
    assert tbl[0].classes == ["UNKNOWN", "A", "B"]
    assert tbl[0].class_colors == {"UNKNOWN": "#cccccc"}
    assert active_track_of(host).id == DEFAULT_TRACK_ID


# ============================== THE SCREEN READERS (PART G)
#
# Four pins on a REAL labeler with its GUI built and its window
# withdrawn -- the conftest `labeler` fixture's own recipe. They are here
# rather than in a fourth file because R8 names three, and they are the
# only pins in the pack that need an axes and a Treeview.

def _gui_labeler(tracks=None, classes=None):
    from chronotagger.labeler import TimeIntervalLabeler

    idx = pd.date_range("2024-01-01 00:00:00", periods=120, freq="30s")
    df = pd.DataFrame({"v": np.linspace(0.0, 1.0, 120)}, index=idx)
    layout = {
        "nrows": 2, "ncols": 1,
        "areas": [
            {"key": "p1", "row": 0, "col": 0, "rowspan": 1, "colspan": 1,
             "role": "time"},
            {"key": "labels", "row": 1, "col": 0, "rowspan": 1,
             "colspan": 1, "role": "labels"},
        ],
    }
    kw = {"tracks": tracks} if tracks is not None else {"classes": classes}
    lbl = TimeIntervalLabeler(
        df=df, plot_fn=lambda axs, sub, t0, t1: None,
        window=pd.Timedelta("1h"), autosave_folder=".",
        layout_spec=layout, **kw)
    lbl._build_gui()
    lbl.root.withdraw()
    return lbl


def _two_lane(lbl):
    idx = lbl.df.index
    lbl.intervals = [
        Interval(idx[0], idx[60], "sw", None, "human"),
        Interval(idx[30], idx[90], "umbra", None, "rules"),
    ]


TWO_LANES = [{"id": "human", "name": "Human", "classes": ["sw"]},
             {"id": "rules", "name": "Rules", "classes": ["umbra"]}]


def test_the_strip_paints_every_VISIBLE_lane_in_its_OWN_colours():
    """AMENDED BY PACK M2, which is what M1's own docstring promised.

    M1 painted the ACTIVE lane only and said so in this very test: "M2
    adds the lanes." The rule M1 was protecting is intact and is the half
    that matters -- a lane is painted in ITS OWN class_colors, never in
    the active lane's -- while the count of painted faces is now every
    visible lane's.
    """
    lbl = _gui_labeler(tracks=TWO_LANES)
    try:
        _two_lane(lbl)
        lbl._update_plot()
        bands = [c for c in lbl.strip_ax.collections
                 if str(c.get_gid() or "").endswith("strip-bands")]
        assert len(bands) == 1, "ONE PolyCollection at every K (Pack 5 R14)"
        assert len(bands[0].get_paths()) == 2      # both lanes now
        assert bands[0].get_pickradius() == 0      # the gutter declines
        # one band ROW per lane, and the active lane does not move the paint
        ys = sorted({round(float(p.vertices[0][1]), 6)
                     for p in bands[0].get_paths()})
        assert len(ys) == 2
        lbl._active_track_id = "rules"
        lbl._update_plot()
        bands = [c for c in lbl.strip_ax.collections
                 if str(c.get_gid() or "").endswith("strip-bands")]
        assert len(bands[0].get_paths()) == 2
        ys2 = sorted({round(float(p.vertices[0][1]), 6)
                      for p in bands[0].get_paths()})
        assert ys2 == ys, "only the ring, the legend and the bold lane " \
                          "name follow the active lane"
    finally:
        lbl.root.destroy()


def test_the_panel_bands_paint_the_ACTIVE_track_only(monkeypatch):
    import chronotagger.labeler.utils.overlays as ov
    seen = {}
    real = ov.draw_interval_bands

    def spy(axes, intervals, *a, **kw):
        seen["tracks"] = [iv.track for iv in intervals]
        return real(axes, intervals, *a, **kw)

    monkeypatch.setattr(ov, "draw_interval_bands", spy)
    lbl = _gui_labeler(tracks=TWO_LANES)
    try:
        _two_lane(lbl)
        lbl._update_plot()
        assert seen.get("tracks") == ["human"]
    finally:
        lbl.root.destroy()


def test_the_statistics_box_reports_UNION_coverage():
    """The record is 59.5 minutes. Two 30-minute lanes overlapping by 15
    sum to 60 minutes -- 100.8 % of all time, which is what the old
    arithmetic printed. The union is 45 minutes, 75.6 %."""
    lbl = _gui_labeler(tracks=TWO_LANES)
    try:
        _two_lane(lbl)
        lbl._update_intervals_list()
        txt = lbl.stats_text.get("1.0", "end")
        pct = float(txt.split("Coverage:")[1].split("%")[0])
        assert pct < 100.0
        assert abs(pct - 45.0 / 59.5 * 100.0) < 0.2
        assert "0 days 00:45:00" in txt
    finally:
        lbl.root.destroy()


def test_the_sidebar_configures_a_PER_LANE_tag_for_every_row_it_shows():
    """AMENDED BY PACK M2: this is DR14, closed.

    M1 pinned the row tag to the BARE LABEL, so one label name could only
    have one colour in the whole list -- measured on two lanes sharing
    `solar_wind`, 3 of 6 rows were painted in the wrong lane's colour or
    in none at all. M2's tag is "<track>|<label>" and its colour comes
    from that lane's own map. The list also defaults to the ACTIVE lane,
    so the second lane's row is not listed until the filter is opened --
    which is the other half of what makes this list honest.
    """
    lbl = _gui_labeler(tracks=TWO_LANES)
    try:
        _two_lane(lbl)
        configured = []
        real = lbl.intervals_tree.tag_configure

        def spy(tagname, *a, **kw):
            if a or kw:
                configured.append(tagname)
            return real(tagname, *a, **kw)

        lbl.intervals_tree.tag_configure = spy
        lbl._update_intervals_list()
        assert configured == ["human|sw"]
        assert len(lbl.intervals_tree.get_children()) == 1
        # open the filter and BOTH lanes are listed, each with its own tag
        configured[:] = []
        lbl.interval_track_scope_var.set("all")
        lbl._update_intervals_list()
        assert sorted(configured) == ["human|sw", "rules|umbra"]
        assert len(lbl.intervals_tree.get_children()) == 2
    finally:
        lbl.root.destroy()
