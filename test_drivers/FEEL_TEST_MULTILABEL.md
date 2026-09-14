# FEEL TEST -- THREE LANES ON ONE STRIP (Pack M2)

**Run it**


    conda activate chronotagger_test
    python test_drivers/drive_multilabel_thb.py

One window, three tabs, and a Labels strip that is now three lanes deep.
Everything below was measured on this driver, headless, before you ran it.

---

## WHAT YOU SEE

The Labels strip at the bottom of every tab is divided into three
horizontal lanes, top to bottom, each with its NAME on the y axis where
the strip used to have nothing:


      Region (human)      <-- your six region classes; this is where you draw
      Wake (umbra)        <-- one class: umbra
      Agent (C-MMAE)      <-- 256 bands built from the frame's own
                              `prediction` column.  LOCKED.

The **active lane** -- the one an Add lands on -- is marked three ways at
once, because the gather measured that any one of them alone fails in some
window:

- its name on the y axis is **bold** and coloured with the colour of the
  class the dropdown is on right now;
- a thin **rectangle outlines the whole width** of its lane (this is the
  one that still tells you where you are when the active lane has nothing
  in this window);
- a **legend sits to the right of the strip**, titled with the lane's
  name, listing that lane's classes and their colours -- six entries for
  `Region (human)`, one for `Wake (umbra)`, four for `Agent (C-MMAE)`.

Measured on the three panes, at the window the driver opens on:

| pane | strip | lanes | pixels per lane |
|---|---|---|---|
| Ion Spectra C0-C30 | 1043.5 x 290.3 px | 3 | **96.8** |
| Context | 1342.4 x 169.3 px | 3 | **56.4** |
| Orbit + Fields | 603.5 x 288.0 px | 3 | **96.0** |

The strip got taller on its own: with lanes on screen the Labels row of
the figure asks for `max(1.0, 0.75 x lanes)` of a row instead of 1.0, so
pane 0's `height_ratios` came out `[1.0, 1.0, 1.0, 2.25]`. The data
panels each lost about a fifth of their height to pay for it. That is
the trade, and it only happens when lanes exist -- open a single-lane
session and the figure is exactly the one you have today.

**33 bands** are on the strip when the window opens: the agent lane's
intervals in the first six hours of 2020-09-02.

---

## HOW TO SWITCH LANES

| | |
|---|---|
| **Ctrl+Down** | next lane down |
| **Ctrl+Up** | next lane up |
| **the Lane list** in the sidebar | pick any lane by name |
| **click a band** | activates that lane -- unless it is locked |

Watch the **class dropdown** while you do it. It re-points at the lane
you switched to: six region classes, then `umbra`, then `0 1 2 3`. That
is not decoration -- before this pack the dropdown stayed on the old
lane's vocabulary and the next Add built an interval whose label the new
lane does not declare, which is the exact state that makes the whole
session refuse to export.

`Ctrl+Up` and `Ctrl+Down` were chosen because they are the only keys that
keep working after you have clicked the class dropdown once. Plain `l`,
`h` and `v` are swallowed while a combobox has focus -- that is a
pre-existing trap in the key handler and this pack does not change it.

---

## DRAWING ON EACH LANE

1. **Region.** Drag on a data panel or two-click on a data panel; the
   strip is for selecting and dragging existing intervals. Pick
   `solar_wind`, press Add (or `a`). The band lands in lane 1. Measured:
   `Added 1 solar_wind interval(s)`.
2. **Wake.** `Ctrl+Down` to `Wake (umbra)`. Draw a short interval INSIDE
   the region interval you just made. It lands in lane 2 and **the region
   interval is not cut**. That is the whole point of the pack: 643 of
   your 2,255 hand-drawn intervals are an `umbra` sandwiched between two
   intervals carrying the SAME region label, because the old tool had one
   label slot and the physics has two independent dimensions.
3. **Agent.** `Ctrl+Down` again and try to Add. The status bar says


       track 'Agent (C-MMAE)' is locked (press Ctrl+L to unlock) -- add refused

   No dialog. NINE paths refuse the same way -- Add, the rules commit,
   Re-label, Delete, the Clear Range WARNING BOX, the clear itself, Fill
   Gaps, Manage Labels, and a strip drag -- in seven distinct sentences,
   and all nine were measured with **zero modals**. (Label Unassigned's
   gap finder is SCOPED to the active lane instead of refused: a locked
   lane's gaps are a fair question. The drag's own hit test gets a LANE
   gate, not a lock.)

**The preview follows the lane.** The dashed yellow rectangle a
selection or a drag paints now sits on the ACTIVE lane's band, not across
the whole strip.

**The gutter is dead space on purpose.** Click in the gap between two
lanes, or above the top band, or below the bottom one, and nothing is
selected. Before this pack a click above every band selected the top
lane's interval.

---

## HIDE AND LOCK

| | |
|---|---|
| **Ctrl+H** | hide / show the active lane |
| **Ctrl+L** | lock / unlock the active lane |
| the **visible** and **locked** checkboxes | the same two things, in the sidebar |

Hiding the active lane MOVES the active lane and the status bar says
where to:


    lane 'Region (human)' hidden -- active lane is now 'Wake (umbra)'

The strip becomes two lanes and each gets taller. The hidden lane's
intervals are **still there**: still held, still saved, still exported --
just not painted, and not clickable. To get it back, pick it in the Lane
list, where it is listed as `Region (human) (hidden)`.

The **last** visible lane refuses to hide. A strip with no lanes is not
a state worth being able to reach.

`Ctrl+L` on `Agent (C-MMAE)` unlocks it, and then you can correct a
prediction by hand -- relabel it, drag its edge, delete it. Lock it again
when you are done. The lock persists in the session file, because "this
came from a model" is a fact about the data, not about this window.

---

## THE SIDEBAR

The interval list has a **Lane** column (last, so the four columns you
read every second keep their widths), and a **Lanes: active / all**
filter right under `Show: all / window`.

**The filter defaults to `active`**, and that is what keeps the list the
size it has always been. Measured on the real widget at 8,000 intervals
across four lanes: unfiltered 230.8 ms per refill and 8,000 rows of four
different vocabularies; filtered to the active lane 60.6 ms and 2,000
rows -- which is what a one-lane list costs today. Click `all` when you
want to compare lanes side by side.

Two lanes that share a class name now get **their own colours** in the
list. Under Pack M1 they shared one, and with two lanes sharing
`solar_wind` three of six rows were painted in the wrong lane's colour or
in none at all.

If you select an interval and then switch to a lane that filters it out,
the status bar tells you where it went:


    the selected interval sits on lane 'Agent (C-MMAE)', which this list is not showing

It is still selected, and Delete still works on it.

---

## EXPORT


    app.export_intervals("x.csv", fmt="csv")     # fmt= is NOT optional
    app.export_per_sample("y.csv", fmt="csv")

`fmt` defaults to **parquet** and the file extension is ignored, so a
`.csv` path silently gets PAR1 bytes. The driver's
`export_per_lane(app)` passes `fmt="csv"` for you and writes both files
into `test_drivers/_out/M2_feel_test/` (gitignored).

Measured on this session:


    m2_probe_intervals.csv     18,102 bytes   start,end,track,label,notes
    m2_probe_per_sample.csv   374,504 bytes   time,label_id__region,label_id__wake,label_id__agent
    sidecars                  m2_probe_per_sample_label_map__{region,wake,agent}.json

One column per lane, one label-map sidecar per lane. The agent lane
exports whether or not it is hidden and whether or not it is locked.

---

## THE NUMBERS, MEASURED HEADLESS BEFORE YOU RAN IT


    build + GUI + the ingest                    1.8-4.0 s  (see below)
    ingest        256 interval(s) into 'agent' from column:prediction
                  gap tolerance 0 days 01:00:00, 239 run(s), 17 split(s),
                  3 unlabeled sample(s) skipped
    panes         3          lanes 3          bands on screen 33
    lane names    ['Region (human)', 'Wake (umbra)', 'Agent (C-MMAE)']
    legend        6 entries, title 'Region (human)'
    after one region interval and one umbra inside it:
                  intervals per lane {'region': 1, 'wake': 1, 'agent': 256}
                  cross-lane overlapping pairs 13
    _update_plot at 3 lanes, 258 intervals      203-435 ms  (see below)
    the strip alone                              11.0-27.8 ms
    dialogs                                      0

THE THREE RANGES WITH "see below" ARE MACHINE-LOAD RANGES, NOT VARIANCE.
The seal measured this driver twice on the same tree: on an idle machine
1.79 s / 203.3 ms / 11.0 ms, and with four pytest suites running beside it
4.02 s / 435.0 ms / 27.8 ms. The drafter's own run (2.3-2.7 s / 216-283 ms
/ 12.1-16.4 ms) sits between them. Nothing about the frame changed; the
CPU did. Read the low end as what you will get.

---

## FOUR THINGS THIS DRIVER GETS RIGHT ON PURPOSE

1. **`gap_tolerance="1h"`, explicitly.** The shipped default is
   `3 x p95(dt)` = 27m36s and on THIS frame it RAISES: the record has 31
   real data gaps above 15 minutes, 27 of them inside a run, so the
   ingest's "more than 10 % of runs would be split" guard fires on real
   holes. Every tolerance from 5 min to 45 min is refused; 1h is the
   first that is accepted. (Pack M2 does not change the guard -- it
   changes its MESSAGE, which now names a holey record as the other
   cause.)
2. **It opens at 2020-09-02 06:00, not at the start.** The agent
   intervals are wildly uneven per day: 2020-09-01 has ONE, 09-02 has 54,
   09-03 has 52. A window at the record start puts a single band on the
   strip.
3. **All three lanes are declared up front**, and the ingest fills the
   existing `agent` lane rather than creating it. The strip's height is
   set when the figure is built, so a lane that appears afterwards gets no
   height of its own.
4. **Its own case directory.** Autosave and both exports land in
   `test_drivers/_out/M2_feel_test/`; nothing is written into
   `edit_pack/campaign/C05_cmmae/` (the frame and the C05 driver it
   reads) and nothing is written into the product tree.

---

## WHAT TO TELL ME AFTERWARDS

- Is 56 px per lane enough on the Context pane, or does the strip need
  more than `0.75 x lanes`?
- Is the bold coloured lane name + the ring + the legend too much at
  once, or is one of them doing all the work?
- `Ctrl+Up` / `Ctrl+Down`: right keys, right direction?
- The Lane list lives in the sidebar. Would you rather have it next to
  the class dropdown in the controls strip?
- Hiding the active lane moves the active lane. Would you rather it
  refused?
- Does the locked lane feel locked -- or does the status line get lost?
- **BACKLOG P9t, the "no date on the strip" half:** the strip already
  carries a date at every window width, in the offset text at the right
  end of the x axis (`2020-Sep-02` at 6h, `2020-Sep` at 3D). Nothing was
  changed for it. Did you mean it is missing, or that it is easy to miss?
