# ============================================================
# PACK M2 FEEL TEST -- THREE LANES ON ONE STRIP.
#
# Run it:     python test_drivers/drive_multilabel_thb.py
# Read first: test_drivers/FEEL_TEST_MULTILABEL.md
#
# WHAT THIS IS. One window, three label lanes on the Labels strip:
#
#   Region (human)    your six region classes -- the lane you draw on
#   Wake (umbra)      one class, `umbra` -- the thing that used to force
#                     you to cut a region crossing into three pieces
#   Agent (C-MMAE)    256 intervals built from the frame's own
#                     `prediction` column, LOCKED, read-only
#
# It is the first driver in this project where the model's answer and your
# answer are on screen at the same time, at the same x, and neither can
# overwrite the other.
#
# WHY C05. This is the ONLY frame on disk with a per-row model column:
# c05_frame.parquet carries `prediction` (int64, values -1/0/1/2/3).
# C03's THB/GMOM frame has 100 columns and no prediction at all, so there
# is nothing there to ingest.
#
# THE DATA AND THE PANES COME FROM THE C05 DRIVER NEXT DOOR
# (C05_cmmae/drive_c05.py): the same three panes, the same spectrogram, the
# same layouts. Everything this file adds is lanes, the window it opens on,
# and its own case directory.
#
# NOTHING HERE WRITES INTO THE PRODUCT TREE, and nothing writes into
# C05_cmmae/: the autosave and every export land in
# test_drivers/_out/M2_feel_test/ (gitignored).
# ============================================================

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# The C05 case (its wizard-emitted driver and the 30-day frame with the
# per-row `prediction` column) is battle-test campaign evidence and stays
# under the gitignored edit_pack/campaign/; this driver only reads it.
C05_DIR = os.path.join(REPO, "edit_pack", "campaign", "C05_cmmae")
if not os.path.isdir(C05_DIR):
    raise SystemExit("the C05 case directory is missing: %s -- this feel test "
                     "needs its c05_frame.parquet and drive_c05.py" % C05_DIR)
if C05_DIR not in sys.path:
    sys.path.insert(0, C05_DIR)

from chronotagger.labeler import TimeIntervalLabeler        # noqa: E402
import drive_c05 as C05                                     # noqa: E402

DATA_PATH = os.path.join(C05_DIR, "c05_frame.parquet")
CASE_DIR = os.path.join(HERE, "_out", "M2_feel_test")
AUTOSAVE_FOLDER = os.path.join(CASE_DIR, "chronotagger_autosave")
SOURCE_NAME = "c05_frame_m2"

# ---------------- THE THREE LANES ---------------------------
# `region` is ACTIVE because it is first in the table and nothing moves
# the active lane at construction. Its vocabulary and colours are the ones
# from the ARTEMIS sessions.
REGION_CLASSES = ["UNKNOWN", "solar_wind", "magnetosheath", "plasma_sheet",
                  "lobe", "mantle"]
REGION_COLORS = {
    "UNKNOWN": "#7f7f7f",
    "solar_wind": "#4e79a7",
    "magnetosheath": "#f28e2b",
    "plasma_sheet": "#e15759",
    "lobe": "#76b7b2",
    "mantle": "#59a14f",
}

# One class. That is the point: `umbra` is an INDEPENDENT dimension, so it
# no longer has to interrupt a region interval to be recorded.
WAKE_CLASSES = ["umbra"]
WAKE_COLORS = {"umbra": "#333333"}

# The agent lane is DECLARED HERE rather than created by the ingest, for
# one measured reason: the Labels strip's height is set when the figure is
# built (max(1.0, 0.75 * K), Pack M2 R5), so a lane that appears AFTER
# construction gets no height of its own. Declaring all three up front
# gives the strip its full three-lane height, and the ingest then fills an
# EXISTING locked lane -- which the ingest supports and which refuses an
# unknown label instead of silently growing the vocabulary.
AGENT_CLASSES = ["0", "1", "2", "3"]
AGENT_COLORS = {"0": "#4e79a7", "1": "#f28e2b", "2": "#e15759",
                "3": "#59a14f"}

TRACKS = [
    {"id": "region", "name": "Region (human)", "classes": REGION_CLASSES,
     "class_colors": REGION_COLORS, "locked": False, "order": 0},
    {"id": "wake", "name": "Wake (umbra)", "classes": WAKE_CLASSES,
     "class_colors": WAKE_COLORS, "locked": False, "order": 1},
    {"id": "agent", "name": "Agent (C-MMAE)", "classes": AGENT_CLASSES,
     "class_colors": AGENT_COLORS, "locked": True, "order": 2},
]

# ---------------- THE INGEST --------------------------------
# `1h`, EXPLICITLY, and this is not a taste decision. The shipped default
# is 3 x p95(dt) = 27m36s and on THIS frame it RAISES: the record has 31
# real data gaps above 15 minutes, 27 of them inside a run, so the guard's
# "more than 10 % of runs would be split" fires on real holes rather than
# on a too-small tolerance. Measured over ten candidates, every tolerance
# from 5 min to 45 min is refused and 1h is the first that is accepted:
# 256 intervals, 239 runs, 17 splits, 0 per-sample mismatches, and only 5
# intervals span a hole longer than 15 min (0.9 % of the record).
GAP_TOLERANCE = "1h"
PREDICTION_COLUMN = "prediction"

# ---------------- THE WINDOW --------------------------------
# DO NOT OPEN ON DAY 1. The agent intervals are wildly uneven per day --
# 2020-09-01 has ONE, 09-02 has 54, 09-03 has 52 -- so a window at the
# record start puts a single band on the strip and the feel test shows
# nothing. 6h at 2020-09-02 06:00 puts 33 bands on screen.
WINDOW = pd.Timedelta("6h")
STEP = pd.Timedelta("3h")
OPEN_AT = pd.Timestamp("2020-09-02 06:00:00")

# OPEN ON A CLASS THAT HAS A COLOUR (Pack M2.5). The class dropdown opens on
# the active lane's FIRST class, and `region`'s first class is UNKNOWN, whose
# colour is grey #7f7f7f -- so the ACTIVE lane's bold name and its focus ring
# came up PALER than the two inactive names (#555555) and only the bold
# weight told you where you were. `solar_wind` is the first real class and
# its #4e79a7 blue reads at a glance. The lane's own class list is NOT
# changed: UNKNOWN is still in it, one keystroke (`u`) away.
OPEN_CLASS = "solar_wind"

PANES = [
    {"title": "Ion Spectra C0-C30", "plot_fn": C05.plot_fn_1,
     "layout_spec": C05.LAYOUT_1},
    {"title": "Context", "plot_fn": C05.plot_fn_2,
     "layout_spec": C05.LAYOUT_2},
    {"title": "Orbit + Fields", "plot_fn": C05.plot_fn_3,
     "layout_spec": C05.LAYOUT_3},
]


def load_dataframe():
    """The C05 frame from the PARQUET, index sorted, time index named."""
    df = pd.read_parquet(DATA_PATH)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)
    df.index.name = "time"
    return df.sort_index()


def build(run=True):
    """Build the labeler, ingest the agent lane, open at OPEN_AT.

    `build(run=False)` hands back the labeler with its GUI built and the
    window positioned, which is what a headless check drives.
    """
    if not os.path.isdir(CASE_DIR):
        os.makedirs(CASE_DIR)
    df = load_dataframe()

    app = TimeIntervalLabeler(
        df=df,
        panes=PANES,
        tracks=TRACKS,
        window=WINDOW,
        step=STEP,
        autosave_folder=AUTOSAVE_FOLDER,
        source_name=SOURCE_NAME,
        decimate=True,
    )
    app._build_gui()

    # The model's answer becomes a real lane: 256 undoable intervals on a
    # locked track, each carrying where it came from in its `meta`. ONE
    # Ctrl+Z removes the whole import.
    app.add_track_from_column(PREDICTION_COLUMN, "agent",
                              name="Agent (C-MMAE)",
                              gap_tolerance=GAP_TOLERANCE,
                              locked=True)

    # Open where the bands are.
    app.t0 = OPEN_AT
    app.t1 = OPEN_AT + WINDOW
    app.current_class_var.set(OPEN_CLASS)
    # Pack M2.5: _sync_entries_and_plot, NOT _update_plot. The Start and End
    # boxes are written by the navigation mixin and by nothing else, so
    # setting t0/t1 by hand and repainting left the two boxes reading the
    # START OF THE RECORD (2020-09-01 00:00:38) while the strip showed
    # 2020-09-02 06:00 -- measured in the live window, and the first thing a
    # live tester noticed. This is the one call that writes the boxes, the
    # plot and the status line together.
    #
    # And then FLUSH. _sync_entries_and_plot asks for a COALESCED redraw
    # (Pack 5 R4d), which a real Tk root defers to its idle queue -- and this
    # driver is also run headless with run=False, where no mainloop ever
    # services that queue, so without the flush every headless consumer
    # (the pins, the button sweep, the screenshot probes) would measure the
    # figure at the record start. The live path is unaffected: the flush
    # renders the same frame the idle callback would have.
    app._sync_entries_and_plot()
    app._flush_pending_redraw()

    if run:
        app.run()
    return app


def main(run=True):
    return build(run=run)


def export_per_lane(app, stem="m2_feel_test"):
    """Both exports, as CSV, into the case directory.

    `fmt="csv"` EXPLICITLY: the default is parquet and the file extension
    is ignored, so `export_intervals("x.csv")` writes PAR1 bytes into a
    file called .csv without complaining.
    """
    out = []
    p1 = os.path.join(CASE_DIR, stem + "_intervals.csv")
    app.export_intervals(p1, fmt="csv")
    out.append(p1)
    p2 = os.path.join(CASE_DIR, stem + "_per_sample.csv")
    app.export_per_sample(p2, fmt="csv")
    out.append(p2)
    return out


if __name__ == "__main__":
    main()
