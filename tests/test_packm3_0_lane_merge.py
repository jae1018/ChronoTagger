"""Pack M3.0, item D -- LOAD SESSION AND RECOVER MERGE THE LANE LIST.

Both paths REPLACED `self.tracks` with the file's table, so a lane only
the DRIVER declared vanished with its name, its classes and its colours.
Measured at base (probe_m30.py D.1, D.3): a two-lane file loaded into a
three-lane app left two lanes; a v1 file left ONE, called `default`.

The rule, ratified: the FILE WINS for every lane id it holds. Every live
lane the file has never heard of is KEPT -- empty of the file's
intervals, because the interval list is the file's -- appended AFTER the
file's lanes, with `order` renumbered so the painted order is "file
lanes, then kept lanes". The bar says so in one sentence, and says
nothing new when nothing was kept. A kept lane arrives EMPTY: the
interval list a load publishes has always been the file's, and the file
has never heard of that lane. Measured, and pinned here, because the
prose alone could be reversed by a later pack without anything going
red.

Recover's fallback active lane was `tracks[0]`, which can be a HIDDEN row
-- measured at base, D.4. It is now the first VISIBLE lane, which is what
Load Session already used.

The all-or-nothing contract is unchanged: the stray-interval refusal
stays keyed to the FILE'S OWN table (a file naming a lane its own table
lacks is internally inconsistent whatever the live table holds), the
merged table is validated in its own right, and a refusal publishes
nothing.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from chronotagger.core.models import Interval

DAY = "2015-01-03 "

REGION = {"id": "region", "name": "Region (human)",
          "classes": ["sw", "msh", "UNKNOWN"], "order": 0}
WAKE = {"id": "wake", "name": "Wake (umbra)",
        "classes": ["umbra", "UNKNOWN"], "order": 1}
AGENT = {"id": "agent", "name": "Agent (C-MMAE)", "classes": ["0", "1"],
         "locked": True, "order": 2}

LAYOUT = {
    "nrows": 3, "ncols": 1,
    "areas": [
        {"key": "panel1", "row": 0, "col": 0, "role": "time"},
        {"key": "panel2", "row": 1, "col": 0, "role": "time"},
        {"key": "labels", "row": 2, "col": 0, "role": "labels"},
    ],
}


def ts(hhmmss):
    return pd.Timestamp(DAY + hhmmss)


def frame():
    idx = pd.date_range(DAY + "00:00:00", periods=120, freq="30s")
    return pd.DataFrame({"a": np.linspace(0.0, 1.0, len(idx))}, index=idx)


def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["a"])
    axs["panel2"].plot(df.index, df["a"])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def build(tracks, tmp_path):
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(
        df=frame(), plot_fn=plot_fn, layout_spec=dict(LAYOUT),
        window=pd.Timedelta("60min"), autosave_folder=str(tmp_path),
        tracks=[dict(t) for t in tracks])
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    return lbl


def session_payload(tmp_path, tracks, ivs=()):
    """A real v2 session file, written by the app's own writer."""
    src = build(tracks, tmp_path)
    try:
        src.intervals[:] = list(ivs)
        p = tmp_path / "written.json"
        assert src._save_session(str(p)) is True
        return json.loads(p.read_text(encoding="utf-8"))
    finally:
        src.root.destroy()


def write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


@pytest.fixture
def three(tmp_path):
    lbl = build([REGION, WAKE, AGENT], tmp_path)
    yield lbl
    lbl.root.destroy()


def ids(app):
    return [t.id for t in app.tracks]


# ========================================== 1. Load Session merges


def test_a_two_lane_file_keeps_the_lane_only_the_driver_declares(
        three, tmp_path):
    data = session_payload(tmp_path, [REGION, WAKE],
                           [Interval(start=ts("00:10:00"),
                                     end=ts("00:15:00"),
                                     label="sw", track="region")])
    three._load_session(write(tmp_path, "two.json", data))
    assert ids(three) == ["region", "wake", "agent"]
    assert [t.order for t in three.tracks] == [0, 1, 2]
    assert len(three.intervals) == 1
    kept = three.tracks[2]
    assert kept.name == "Agent (C-MMAE)" and kept.locked is True
    assert list(kept.classes) == ["0", "1"], \
        "the kept lane keeps its own vocabulary"
    assert three.status_var.get().endswith(
        " -- loaded 2 lanes from the file; kept 1 lane from the driver")


def test_a_kept_lane_arrives_EMPTY_of_its_live_intervals(three, tmp_path):
    """The ruled shape: "kept, empty". The interval list is the FILE'S."""
    three.intervals[:] = [Interval(start=ts("00:40:00"),
                                   end=ts("00:45:00"),
                                   label="0", track="agent")]
    data = session_payload(tmp_path, [REGION, WAKE])
    three._load_session(write(tmp_path, "kept_empty.json", data))
    assert ids(three) == ["region", "wake", "agent"]
    assert three.intervals == [], (
        "the kept lane's live intervals are the file's business, and the "
        "file has never heard of that lane")


def test_the_file_wins_every_field_for_a_lane_it_knows(three, tmp_path):
    other = dict(REGION)
    other["name"] = "Renamed in the file"
    other["locked"] = True
    data = session_payload(tmp_path, [other, WAKE, AGENT])
    three._load_session(write(tmp_path, "won.json", data))
    assert three.tracks[0].name == "Renamed in the file"
    assert three.tracks[0].locked is True
    assert three.status_var.get().startswith("Loaded from ")
    assert "kept" not in three.status_var.get()


def test_when_nothing_is_kept_the_status_line_is_todays_exactly(
        three, tmp_path):
    data = session_payload(tmp_path, [REGION, WAKE, AGENT])
    path = write(tmp_path, "same.json", data)
    three._load_session(path)
    assert ids(three) == ["region", "wake", "agent"]
    assert three.status_var.get() == "Loaded from %s" % path


def test_a_lane_only_the_file_knows_still_becomes_a_row(three, tmp_path):
    extra = {"id": "extra", "name": "Extra", "classes": ["e"], "order": 3}
    data = session_payload(tmp_path, [REGION, WAKE, AGENT, extra])
    three._load_session(write(tmp_path, "extra.json", data))
    assert ids(three) == ["region", "wake", "agent", "extra"]
    assert "kept" not in three.status_var.get()


def test_a_v1_file_lands_beside_the_drivers_lanes(three, tmp_path):
    v1 = {"version": 1, "classes": ["a", "b"],
          "class_colors": {"a": "#111111", "b": "#222222"},
          "window": "0 days 01:00:00", "step": "0 days 00:30:00",
          "data_start": str(three.data_start),
          "data_end": str(three.data_end),
          "intervals": [], "layout_spec": three.layout_spec}
    three._load_session(write(tmp_path, "v1.json", v1))
    assert ids(three) == ["default", "region", "wake", "agent"]
    assert [t.order for t in three.tracks] == [0, 1, 2, 3]
    assert three.active_track_id == "default"
    assert list(three.classes) == ["a", "b"]
    assert " -- loaded 1 lane from the file; kept 3 lanes from the driver" \
        in three.status_var.get()


def test_the_painted_order_of_the_files_own_lanes_is_the_files_order(
        three, tmp_path):
    """A file whose `order` values are scrambled paints as it does today."""
    data = session_payload(tmp_path, [REGION, WAKE])
    assert [t["id"] for t in data["tracks"]] == ["region", "wake"]
    data["tracks"][0]["order"] = 7
    data["tracks"][1]["order"] = 2
    three._load_session(write(tmp_path, "scrambled.json", data))
    assert ids(three) == ["wake", "region", "agent"]
    assert [t.order for t in three.tracks] == [0, 1, 2]


def test_a_refused_file_publishes_nothing(three, tmp_path):
    data = session_payload(tmp_path, [REGION, WAKE])
    data["intervals"] = [{"start": str(ts("00:20:00")),
                          "end": str(ts("00:25:00")),
                          "label": "sw", "track": "ghost"}]
    before_ids = ids(three)
    before_active = three.active_track_id
    three._load_session(write(tmp_path, "bad.json", data))
    assert ids(three) == before_ids
    assert three.active_track_id == before_active
    assert three.intervals == []
    assert "Refused" in three.status_var.get()


def test_a_stray_is_refused_even_when_a_kept_lane_carries_that_id(
        three, tmp_path):
    """`agent` is live but is NOT in this file's table: still refused."""
    data = session_payload(tmp_path, [REGION, WAKE])
    data["intervals"] = [{"start": str(ts("00:20:00")),
                          "end": str(ts("00:25:00")),
                          "label": "0", "track": "agent"}]
    three._load_session(write(tmp_path, "inconsistent.json", data))
    assert ids(three) == ["region", "wake", "agent"]
    assert three.intervals == []
    assert "Refused" in three.status_var.get()


# ========================================== 2. Recover merges too


def test_recovery_keeps_the_drivers_lane(three, tmp_path):
    data = session_payload(tmp_path, [REGION, WAKE])
    three._apply_recovered_autosave(
        {"version": 2, "tracks": data["tracks"], "intervals": [],
         "active_track": "region"})
    assert ids(three) == ["region", "wake", "agent"]
    assert [t.order for t in three.tracks] == [0, 1, 2]
    assert three.active_track_id == "region"
    assert three._lane_merge_note == \
        " -- loaded 2 lanes from the file; kept 1 lane from the driver"


def test_recovery_of_a_v1_payload_lands_beside_the_drivers_lanes(three):
    three._apply_recovered_autosave(
        {"version": 2, "classes": ["a", "b"], "intervals": []})
    assert ids(three) == ["default", "region", "wake", "agent"]


def test_recovery_with_no_table_keeps_the_live_one_and_says_nothing(
        three):
    three._apply_recovered_autosave({"version": 2, "intervals": []})
    assert ids(three) == ["region", "wake", "agent"]
    assert three._lane_merge_note == ""


def test_recovery_lands_on_the_first_VISIBLE_lane(three, tmp_path):
    hidden = dict(REGION)
    hidden["visible"] = False
    data = session_payload(tmp_path, [hidden, WAKE])
    assert data["tracks"][0]["visible"] is False
    three._apply_recovered_autosave(
        {"version": 2, "tracks": data["tracks"], "intervals": [],
         "active_track": "no such lane"})
    assert three.tracks[0].id == "region"
    assert three.tracks[0].visible is False
    assert three.active_track_id == "wake", \
        "row 0 is hidden; the active lane must be the first VISIBLE one"


def test_a_refused_recovery_publishes_nothing(three, tmp_path):
    data = session_payload(tmp_path, [REGION, WAKE])
    before = ids(three)
    with pytest.raises(ValueError):
        three._apply_recovered_autosave(
            {"version": 2, "tracks": data["tracks"],
             "intervals": [{"start": str(ts("00:20:00")),
                            "end": str(ts("00:25:00")),
                            "label": "sw", "track": "ghost"}]})
    assert ids(three) == before
    assert three.intervals == []


# ========================================== 3. the merge, on its own


def test_merge_track_tables_is_file_first_then_kept_renumbered():
    from chronotagger.core.tracks import Track, merge_track_tables
    f = [Track(id="b", name="B", classes=["x"], order=5),
         Track(id="a", name="A", classes=["x"], order=1)]
    live = [Track(id="a", name="live A", classes=["q"], order=0),
            Track(id="z", name="Z", classes=["q"], order=9)]
    merged, n_file, n_kept = merge_track_tables(f, live)
    assert [t.id for t in merged] == ["a", "b", "z"]
    assert [t.order for t in merged] == [0, 1, 2]
    assert n_file == 2 and n_kept == 1
    assert merged[0].name == "A", "the file wins for a lane it holds"
    assert merged[2].name == "Z"
    assert merged[2] is not live[1], "the kept row is a copy"


def test_the_note_is_empty_when_nothing_was_kept():
    from chronotagger.core.tracks import lane_merge_note
    assert lane_merge_note(3, 0) == ""
    assert lane_merge_note(1, 1) == \
        " -- loaded 1 lane from the file; kept 1 lane from the driver"
    assert lane_merge_note(2, 3) == \
        " -- loaded 2 lanes from the file; kept 3 lanes from the driver"
