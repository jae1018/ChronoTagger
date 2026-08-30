# Writing a fast `plot_fn`

Your `plot_fn` runs on every redraw: every pan, every zoom notch, every
commit, every window change. A highlight refresh does NOT re-enter it --
the red and blue marks are drawn over the frame you already drew. Six
rules, each with the measurement that earns it.

**How these were measured.** Branch `refactor` @ `575fa28`, on a real
22,281-row THEMIS-B day (44 float64 columns), AMD Ryzen 7 5800X,
matplotlib 3.10.9 on TkAgg, with the figure forced to 14 x 8 in at
100 dpi so each panel is 1,342 px wide. Every number below is a FULL
REDRAW through a real labeler -- `_update_plot()` end to end, not a bare
matplotlib call -- median of 5 reps after 3 warmups (fewer for the
heaviest modes), each block bracketed by a CPU calibration guard, and a
block that never ran clean was never reported. Bigger frames are real
samples, concatenated. Numbers taken at a different figure size or on a
different frame are not interchangeable with these.

## 1. One artist, not many

40,000 points in ONE artist cost **+26.0 ms** above the redraw floor.
One thousand artists holding 1,000 points between them cost
**+804.4 ms** -- 31x the price for one fortieth of the data.

The per-artist cost is **~0.78 ms, and flat**: 0.770 / 0.787 / 0.804 ms
at N = 50 / 200 / 1,000. It does not improve with scale. The per-point
cost is ~0.00065 ms, so **one extra artist costs about what 1,200 extra
points cost.** That is the exchange rate worth memorising.

The merged form of the same picture is **12.5x cheaper**: 1,000
`ax.axvline` calls cost +786.5 ms, while ONE `ax.vlines` carrying the
same 1,000 x positions costs +62.7 ms.

~~~python
# NO -- one artist per event
for t in event_times:
    ax.axvline(t, color="k", lw=0.5)

# YES -- one artist, same picture.  vlines takes its y limits in DATA
# coordinates, so span the axes with the x-axis transform;
# *ax.get_ylim() freezes the limit at call time and the merged line
# then falls short of the axes edges.
ax.vlines(event_times, 0, 1, transform=ax.get_xaxis_transform(),
          color="k", lw=0.5)
~~~

A realistic mixture -- a 40,000-point trace plus 200 event markers --
costs +179.9 ms, of which the trace is 26.0 ms and the markers are the
rest. **The markers cost 6x the data.**

## 2. Stay vectorized, windowed, and off the disk

Deriving a quantity inside `plot_fn` is **free when it is vectorized
over the window you were handed.** At 44,000 rows,
`np.sqrt(Bx**2 + By**2 + Bz**2)` measures **+1.9 ms** and
`rolling(2001).std()` measures **-8.9 ms**; every vectorized derive
tested landed within **+/-13 ms** of simply reading a precomputed
column, and at that size the sign is not meaningful.

What costs is work that is per-row, wrong-scale, or on disk:

| inside `plot_fn`, 44,000 rows | penalty per redraw |
|---|---|
| `.apply(axis=1)` | **+582.2 ms** (+850 %) |
| an explicit Python loop over rows | **+25.4 ms** |
| `sort_values` | +12.7 ms |
| deriving over the WHOLE frame, then slicing | **+37.3 ms** (1.59x) |
| `pd.read_parquet` per frame | **+51.2 ms** (1.81x) |

So derive freely, provided pandas or numpy runs the loop, the input is
the window you were given, and nothing reaches the filesystem.

## 3. `scatter` costs more than `plot`, and a colour array costs far more

Bare `ax.scatter` is **1.24x-2.37x** the above-floor cost of
`ax.plot(linestyle="none", marker=".")`: +10.2 ms at 40,000 points,
+21.1 ms at 10,000. Real, but not dramatic.

A per-point colour array is the dramatic one.
`ax.scatter(x, y, s=3, c=<array>, cmap=...)` costs **+535.6 ms** at
40,000 points -- **12.5x** the equivalent marker line, turning a 108 ms
frame into a 601 ms one. Marker size matters too: `ms=1` is +23.4 ms
against `ms=3`'s +42.8 ms at 40,000 points.

Scatter-drawn `role="time"` panels are otherwise first-class: selection
and both highlight families read a `PathCollection` exactly as they read
a line.

## 4. Use `draw_spectrogram`, not a raw mesh

A naive `ax.pcolormesh` over the native columns of a 44,000-row window
draws 1,408,000 quads and costs **997.1 ms of a 1,119.9 ms frame -- 89 %
of every redraw.** `draw_spectrogram(aggregator="max")` rebins to the
panel's 1,342 pixel columns (42,944 cells) and costs **32.0 ms**:
**31.2x cheaper**, same panel, same window. `mean` is level with `max`
(33.5 ms); `logmean` costs about double (64.2 ms).

The README's `#### Known limitations` measures this effect on a
different figure geometry and a different frame; the two sets of numbers
are not interchangeable.

## 5. Alpha is free. A thousand artists are not.

Overlapping translucent bands are not the expensive thing -- artist
count is. Drawn over a real 40,000-point line, at N = 1,000 the
artist-count effect is **+697.7 ms** while the alpha effect is
**-24.2 ms**, i.e. nothing: across N it measures +5.9, +4.9, -63.4 and
-24.2 ms, changing sign, which is noise.

1,000 `ax.axvspan` calls cost **+753.6 ms**. The same 1,000 quads
handed to ONE `PolyCollection` cost **+55.9 ms** -- **13.5x**. That
per-artist rate is 0.75 ms, which is rule 1's 0.78 ms again: rule 5 is
rule 1 wearing a hat.

This is the cost of bands YOU draw. The tool's own interval bands are
already one collection per panel, and the figure the README quotes for
those is a different measurement.

## 6. Draw the DataFrame you were handed

`_update_plot` decimates `sub_df` before handing it to you, so draw
decimation can only reach the frame in your `df` argument. Whatever your
`plot_fn` fetches for itself is drawn at full resolution.

On a 200,000-row frame with 3 numeric columns and a 20,000-row window,
decimation hands you **1,337 of 19,987 rows**, and going around it
costs **+32.9 ms (1.52x)** through a closure re-slice, **+51.2 ms
(1.81x)** through a per-frame file read, and **+37.3 ms (1.59x)** by
deriving over the whole frame before slicing.

How much that is worth is set by the frame's **numeric column count**,
because the plan unions the minimum and maximum row per pixel bin over
EVERY numeric column, not only the plotted ones. Same frame, same
window: 3 numeric columns keeps 1,337 rows (6.7 %), while 44 numeric
columns keeps 19,668 (98.4 %) and there is almost nothing left to lose.
On the real 22,281-row day the same law reads 6.0 % against 97.1 %.

That is a fact about the plan, not an instruction to strip columns: a
generated driver loads every column on purpose, so the `[YOURS]` block
can use them. Whether decimation should look only at the plotted columns
is parked as SP-R6.

What decimation is, and the two conditions that switch it off pane-wide,
are in the README's `### Performance Optimizations`. This page is only
about what your `plot_fn` should do about it.

## Two knobs the rules do not cover

- **`labeler.enable_point_highlighting = False`** turns off the red
  preview and blue interval marks. They are recomputed per gesture, per
  panel, and capped at about 1,000 markers per panel; on a very large
  frame this is the first thing to switch off.
- **Never call `fig.colorbar()` inside `plot_fn`.** It leaks one axes
  per redraw and shrinks the image panel by about 19 % of its remaining
  width each time, until that panel's x axis is no longer the Labels
  strip's. Use `attach_colorbar()`, which is idempotent per owner axes.

## A `plot_fn` that follows all six

`examples/timeseries_only.py`: one artist per trace, no per-frame
derives, no scatter, no translucent stacking, nothing read from disk,
and a layout whose areas are all `time` or `labels` -- which is what
makes it eligible for decimation end to end.
