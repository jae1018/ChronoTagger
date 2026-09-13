"""Pack M2 -- THE LANE ARITHMETIC, on its own, with no GUI at all.

Every wrong-lane bug the M2 gather found was a DISAGREEMENT BETWEEN TWO
COPIES of this arithmetic: the painter said a band was here, a hit test
said it was there, and a click selected an interval on a lane the painter
had not drawn.  So the arithmetic lives in exactly one module and this
file is the pin that says the pieces are exact inverses of each other.

Three groups:

  1. `lane_band` / `lane_from_frac` / `lane_strict` -- the geometry, at
     k = 1..8, over 1,001 fractions, including every boundary.
  2. The THREE COMPAT RULES the single-lane byte identity needs: pad 0.1
     at k == 1, `lane_band(0, 1, 0.1) == (0.1, 0.9)` exactly, and
     `preview_band` BRACKETING the band rather than equalling it.
  3. The host-side readers -- `visible_lanes`, `lane_layout`,
     `active_band`, `refuse_if_locked` -- against a GUI-FREE host, which
     is the shape every mock in this suite has.
"""

import pytest

from chronotagger.core import lanes as L
from chronotagger.core.tracks import Track

FRACS = [i / 1000.0 for i in range(1001)]


# --------------------------------------------------------------- helpers

class Host(object):
    """The smallest thing the lane readers accept: a table and a status."""

    def __init__(self, tracks, active=None, status=True):
        self.tracks = list(tracks)
        if active is not None:
            self._active_track_id = active
        if status:
            self.status_var = Var()


class Var(object):
    def __init__(self, value=""):
        self._v = value

    def get(self):
        return self._v

    def set(self, v):
        self._v = v


def T(tid, **kw):
    kw.setdefault("classes", ["a", "b"])
    return Track(id=tid, name=kw.pop("name", tid.title()), **kw)


# ============================================================ 1. geometry


def test_row_zero_is_the_top_band_at_every_k():
    for k in range(1, 9):
        tops = [L.lane_band(r, k)[1] for r in range(k)]
        assert tops == sorted(tops, reverse=True), (
            "row 0 must be the TOP band; Track.order is top to bottom")


def test_every_band_is_inside_the_axes_and_bands_do_not_touch():
    for k in range(1, 9):
        pad = L.pad_for(k)
        bands = [L.lane_band(r, k, pad) for r in range(k)]
        for lo, hi in bands:
            assert 0.0 <= lo < hi <= 1.0
        for i in range(k - 1):
            # bands[i] sits ABOVE bands[i+1] and there is a gutter
            assert bands[i][0] > bands[i + 1][1], "no gutter between lanes"


def test_the_band_height_is_uniform_at_every_k():
    for k in range(1, 9):
        hs = [round(hi - lo, 12) for lo, hi in
              (L.lane_band(r, k) for r in range(k))]
        assert len(set(hs)) == 1, "M2 ships UNIFORM lane height"


def test_lane_from_frac_is_the_inverse_of_lane_band_at_every_k():
    """1,001 fractions x 8 lane counts: the loose form always names a row."""
    for k in range(1, 9):
        pad = L.pad_for(k)
        for f in FRACS:
            r = L.lane_from_frac(f, k)
            assert 0 <= r < k
            lo, hi = L.lane_band(r, k, pad)
            # the fraction is either inside this row's band or in the
            # gutter that belongs to it -- never inside ANOTHER row's band
            for other in range(k):
                if other == r:
                    continue
                olo, ohi = L.lane_band(other, k, pad)
                assert not (olo <= f <= ohi), (
                    "frac %.3f named row %d but sits in row %d's band"
                    % (f, r, other))


def test_a_band_centre_round_trips_at_every_k():
    for k in range(1, 9):
        pad = L.pad_for(k)
        for r in range(k):
            lo, hi = L.lane_band(r, k, pad)
            mid = 0.5 * (lo + hi)
            assert L.lane_from_frac(mid, k) == r
            assert L.lane_strict(mid, k, pad) == r


def test_lane_strict_is_none_in_every_gutter_and_outside_every_band():
    for k in range(1, 9):
        pad = L.pad_for(k)
        inside = 0
        for f in FRACS:
            r = L.lane_strict(f, k, pad)
            if r is None:
                # must NOT be inside any band
                for other in range(k):
                    olo, ohi = L.lane_band(other, k, pad)
                    assert not (olo <= f <= ohi)
            else:
                lo, hi = L.lane_band(r, k, pad)
                assert lo <= f <= hi
                inside += 1
        assert inside > 0


def test_lane_strict_declines_above_the_top_and_below_the_bottom():
    """The third of the gather's three defects: a click above every band
    used to select lane 0's interval."""
    for k in range(1, 9):
        assert L.lane_strict(0.999, k) is None
        assert L.lane_strict(0.001, k) is None
        assert L.lane_strict(1.5, k) is None
        assert L.lane_strict(-0.5, k) is None


def test_the_k4_gutter_table_is_what_the_gather_measured():
    pad = L.pad_for(4)
    want = {0.005: None, 0.125: 3, 0.240: None, 0.250: None, 0.260: None,
            0.375: 2, 0.490: None, 0.500: None, 0.510: None, 0.995: None}
    got = dict((f, L.lane_strict(f, 4, pad)) for f in want)
    assert got == want


def test_lane_band_refuses_a_row_outside_the_table_and_k_below_one():
    with pytest.raises(ValueError):
        L.lane_band(0, 0)
    with pytest.raises(ValueError):
        L.lane_band(-1, 3)
    with pytest.raises(ValueError):
        L.lane_band(3, 3)
    with pytest.raises(ValueError):
        L.lane_from_frac(0.5, 0)
    assert L.lane_strict(0.5, 0) is None


# ============================================ 2. the three compat rules


def test_rule_one_the_pad_is_point_one_at_one_lane_and_point_zero_six_above():
    assert L.pad_for(1) == 0.1
    assert L.pad_for(0) == 0.1
    for k in range(2, 9):
        assert L.pad_for(k) == 0.06


def test_rule_two_one_lane_is_exactly_the_pre_m2_band():
    """0.1 and 0.9 are the four constants plotting.py used to write."""
    lo, hi = L.lane_band(0, 1, 0.1)
    assert (round(lo, 12), round(hi, 12)) == (0.1, 0.9)
    assert L.lane_band(0, 1) == L.lane_band(0, 1, 0.1)


def test_rule_three_the_preview_brackets_the_band_by_half_the_gutter():
    """Shipped band 0.1..0.9, shipped preview 0.05..0.95: NOT the band.

    This is the one condition that failed before the gather found it.
    """
    y, h = L.preview_band(0, 1, 0.1)
    assert (round(y, 12), round(h, 12)) == (0.05, 0.9)
    assert round(y + h, 12) == 0.95
    for k in range(1, 9):
        pad = L.pad_for(k)
        for r in range(k):
            lo, hi = L.lane_band(r, k, pad)
            y, h = L.preview_band(r, k, pad)
            half = pad * (1.0 / k) / 2.0
            assert round(y, 12) == round(lo - half, 12)
            assert round(y + h, 12) == round(hi + half, 12)
            assert y < lo and (y + h) > hi


def test_the_preview_never_leaves_the_axes():
    for k in range(1, 9):
        for r in range(k):
            y, h = L.preview_band(r, k)
            assert -1e-9 <= y and y + h <= 1.0 + 1e-9


def test_the_labels_row_height_is_one_at_one_lane_and_grows_at_three_quarters():
    assert L.labels_row_height(1) == 1.0
    assert L.labels_row_height(0) == 1.0
    assert L.labels_row_height(2) == 1.5
    assert L.labels_row_height(3) == 2.25
    assert L.labels_row_height(4) == 3.0


def test_the_band_gid_is_the_one_the_painter_writes():
    """Spelled twice on purpose -- core may not import labeler -- so it is
    pinned equal here, in both directions."""
    from chronotagger.labeler.mixins.events.base import TOOL_GID_PREFIX
    assert L.strip_band_gid() == TOOL_GID_PREFIX + "strip-bands"
    assert L.STRIP_BAND_GID == "chronotagger:strip-bands"


# ============================================== 3. the host-side readers


def test_visible_lanes_is_ordered_top_to_bottom_and_drops_hidden_rows():
    h = Host([T("c", order=2), T("a", order=0), T("b", order=1)])
    assert [t.id for t in L.visible_lanes(h)] == ["a", "b", "c"]
    h.tracks[1].visible = False
    assert [t.id for t in L.visible_lanes(h)] == ["b", "c"]


def test_lane_layout_reports_k_the_rows_and_the_active_row():
    h = Host([T("a"), T("b", order=1), T("c", order=2)], active="b")
    lay = L.lane_layout(h)
    assert lay["k"] == 3
    assert lay["row_of"] == {"a": 0, "b": 1, "c": 2}
    assert lay["active_row"] == 1
    assert lay["pad"] == 0.06


def test_a_hidden_active_lane_has_no_row_and_keeps_its_id():
    h = Host([T("a"), T("b", order=1)], active="b")
    h.tracks[1].visible = False
    lay = L.lane_layout(h)
    assert lay["k"] == 1
    assert lay["active_id"] == "b"
    assert lay["active_row"] is None


def test_an_active_id_the_table_does_not_hold_falls_back_to_the_first_lane():
    """After an undo removes an ingested lane, `_active_track_id` still
    names it: the layout must not answer None for a lane that is GONE, or
    the screen and self.classes disagree."""
    h = Host([T("a"), T("b", order=1)], active="ghost")
    lay = L.lane_layout(h)
    assert lay["active_id"] == "a"
    assert lay["active_row"] == 0


def test_lane_layout_survives_every_lane_being_hidden():
    h = Host([T("a"), T("b", order=1)], active="a")
    for t in h.tracks:
        t.visible = False
    lay = L.lane_layout(h)
    assert lay["k"] == 0 and lay["active_row"] is None
    assert L.active_band(h) == (0.05, 0.9)


def test_active_band_is_the_pre_m2_rectangle_at_one_lane():
    h = Host([T("a")], active="a")
    y, hh = L.active_band(h)
    assert (round(y, 12), round(hh, 12)) == (0.05, 0.9)


def test_the_k4_band_table_is_what_the_gather_measured():
    """The gather's K=4 table -- (y, h) = (0.7650, 0.2200) for row 0,
    (0.2650, 0.2200) for row 2, (0.0150, 0.2200) for row 3 -- is the BAND.
    The shipping preview BRACKETS it (rule three), which is why the
    preview's own numbers are half a gutter wider on each side."""
    pad = L.pad_for(4)
    for row, want in ((0, (0.7650, 0.2200)), (2, (0.2650, 0.2200)),
                      (3, (0.0150, 0.2200))):
        lo, hi = L.lane_band(row, 4, pad)
        assert (round(lo, 4), round(hi - lo, 4)) == want


def test_active_band_follows_the_active_lane():
    h = Host([T("a"), T("b", order=1), T("c", order=2), T("d", order=3)],
             active="d")
    pad = L.pad_for(4)
    for tid, row in (("d", 3), ("a", 0), ("c", 2)):
        h._active_track_id = tid
        y, hh = L.active_band(h)
        want = L.preview_band(row, 4, pad)
        assert (round(y, 6), round(hh, 6)) == (round(want[0], 6),
                                               round(want[1], 6))
        lo, hi = L.lane_band(row, 4, pad)
        assert y < lo and y + hh > hi, "the preview brackets the band"


def test_the_readers_work_on_a_host_with_no_table_at_all():
    class Bare(object):
        classes = ["x", "y"]
        class_colors = {"x": "#111111"}

    b = Bare()
    assert [t.id for t in L.visible_lanes(b)] == ["default"]
    assert L.lane_layout(b)["k"] == 1
    assert L.is_locked(b) is False
    assert L.refuse_if_locked(b) is False


# ------------------------------------------------------------- the guard


def test_refuse_if_locked_is_false_and_silent_on_an_unlocked_lane():
    h = Host([T("a")], active="a")
    assert L.refuse_if_locked(h, what="delete") is False
    assert h.status_var.get() == ""


def test_refuse_if_locked_names_the_lane_the_key_and_the_action():
    h = Host([T("a", name="Agent (C-MMAE)", locked=True)], active="a")
    assert L.refuse_if_locked(h, what="clear range") is True
    msg = h.status_var.get()
    assert "Agent (C-MMAE)" in msg
    assert "locked" in msg
    assert "Ctrl+L" in msg
    assert "clear range refused" in msg


def test_refuse_if_locked_judges_the_lane_it_is_given_not_the_active_one():
    h = Host([T("a"), T("b", locked=True, order=1)], active="a")
    assert L.refuse_if_locked(h) is False
    assert L.refuse_if_locked(h, "b", what="drag") is True


def test_the_guard_opens_no_dialog(monkeypatch):
    import tkinter.messagebox as mb
    calls = []
    for kind in ("showinfo", "showwarning", "showerror", "askyesno"):
        monkeypatch.setattr(mb, kind,
                            lambda *a, _k=kind, **kw: calls.append(_k))
    h = Host([T("a", locked=True)], active="a")
    for what in ("clear range", "delete", "relabel", "drag", "add",
                 "fill gaps", "edit the label schema"):
        assert L.refuse_if_locked(h, what=what) is True
    assert calls == [], "a refusal is a STATUS LINE, never a modal"


def test_the_guard_never_raises_on_a_host_with_no_status_bar():
    h = Host([T("a", locked=True)], active="a", status=False)
    assert L.refuse_if_locked(h, what="delete") is True
    assert L.set_status(h, "anything") is False


def test_track_display_name_prefers_the_name_and_falls_back_to_the_id():
    h = Host([T("a", name="Region (human)"), T("b", name="", order=1)])
    assert L.track_display_name(h, "a") == "Region (human)"
    assert L.track_display_name(h, "b") == "b"
    assert L.track_display_name(h, "ghost") == "ghost"


def test_frac_from_event_returns_none_without_a_pixel_position():
    class E(object):
        x = None
        y = None
    assert L.frac_from_event(None, None) is None
    assert L.frac_from_event(object(), E()) is None
