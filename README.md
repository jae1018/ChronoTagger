# ChronoTagger — Time-interval labeling for time-series (Tk + Matplotlib)

ChronoTagger is a small, focused GUI for interactively labeling time intervals on top of your own time-series plots. You provide the data (a `pandas.DataFrame` with a `DatetimeIndex`) and a plotting function; ChronoTagger handles selection, labeling, overlay visualization, undo/redo, and exporting labels in ML-friendly formats.

## TL;DR (quick start)

**Requirements:** 

Python 3.9+, pandas, matplotlib, tkinter.

**Your plot function:** 

implements `plot_fn(axes, df, t0, t1)` and draws on `axes["panel1"]`, `axes["panel2"]`, …

**Launch:**

```python
import pandas as pd
from chronotagger import TimeIntervalLabeler

def plot_fn(axs, df, t0, t1):
    ax = axs["panel1"]
    df["log10n"].plot(ax=ax)
    ax.set_ylabel("log10(n) [cm^-3]")

    ax2 = axs["panel2"]
    df[["BX","BY","BZ"]].plot(ax=ax2)
    ax2.set_ylabel("B [nT]")

labeler = TimeIntervalLabeler(df=df, plot_fn=plot_fn, window=pd.Timedelta("1h"))
labeler.run()
```

Select & label: drag across a plot to select, select **Add Label**.
Manage label names/colors via **Manage Labels…**.

Export: **Export Labels…** (CSV + *_label_map.json sidecar), or **Export Intervals** (CSV/Parquet).

## Features

- Works with your own plotting code (Matplotlib).
- Multiple panels (auto-detected or specified), all x-aligned on time.
- Drag to create a time selection; faint overlays show labeled/preview spans across all panels.
- Bottom strip shows labeled intervals; click to select; drag edges to resize or drag the box to move.
- Mouse wheel zoom around cursor; Shift + wheel to pan; Prev/Next navigation with configurable step.
- Undo/redo with a command stack (add, delete, relabel, resize/move).
- **Manage Labels…** dialog (add/rename/delete/reorder/change color, with reassign for in-use labels).
- Save/load session (JSON).
- Exports for ML: per-sample `label_id` CSV (with sidecar mapping), or interval lists.

## Installation

Use your existing environment where you run Matplotlib + Tk:

```bash
pip install pandas matplotlib
# If you run via Conda on Windows:
conda install -c anaconda tk  # ensures a working Tcl/Tk
```

Then add this repository to your Python path (editable install, or run scripts from the repo).

## Data & plotting contract

### Input data
- `pandas.DataFrame` with a `DatetimeIndex` spanning your dataset.
- All columns you plot must exist in `df`.

### Plot function

Your function must accept four arguments and draw onto the provided axes dict:

```python
def plot_fn(axs: dict[str, matplotlib.axes.Axes],
            df_window: pd.DataFrame,
            t0: pd.Timestamp,
            t1: pd.Timestamp) -> None:
    # df_window is already sliced to [t0, t1]
    axs["panel1"].plot(df_window.index, df_window["log10n"])
    axs["panel2"].plot(df_window.index, df_window[["BX","BY","BZ"]])
```

### How many panels?

ChronoTagger resolves the number of panels in this order:

1. `TimeIntervalLabeler(..., n_panels=K)` if you pass it.  
2. `plot_fn.n_panels` if you set it on your function.  
3. **Probe:** ChronoTagger calls your `plot_fn` on a temporary figure and counts panels that drew anything.  
4. **Default:** 2 panels.

## Running the demo

You can run the example script located in examples/synthetic_demo.py.

## UI guide

### Top bar
- **Time Range:** Start/End entries and **Update Window**.
- **Navigation:** **Prev / Next**, a **Step** entry, quick buttons **x2** and **/2** to adjust the step.
- **Current Label:** a read-only combobox of labels and a **Manage Labels…** button.
- **Quick Actions:** **Add Label**, **Delete**, **Undo**, **Redo**, **Export Labels…**, **Help (F1)**.

### Plots & strip
- Your plots appear in stacked panels, sharing the same time axis.
- A bottom **Labels** strip shows colored rectangles for each labeled interval.
- **Interval overlays:** subtle vertical bands across your data panels reflect labeled intervals in the current view, the selected interval (slightly stronger), and the current drag selection (yellow tint), all drawn behind the data.

### Sidebar
- **Labeled Intervals** list (start/end/label/duration). Click to select.
- **Statistics** summarizing coverage and counts per label.
- **Actions:** Relabel, Delete, Assign Remainder -> UNKNOWN, Clear All Intervals.
- **File Operations:** Save, Load, Export Intervals, Export Per-Sample, Export Labels…

## Keyboard & mouse

### Keyboard
- `1–9`: select label by index
- `n` / Right Arrow: next window
- `p` / Left Arrow: previous window
- `a` / Enter: add interval from current selection
- `d` / Delete: delete selected interval
- `u`: select UNKNOWN label (if present)
- `Ctrl+S`: save session
- `Ctrl+E`: export (labels dialog)
- `Ctrl+Z` / Backspace: undo
- `Ctrl+Y` / Shift+Backspace: redo

> When an Entry/Combobox has focus (e.g., editing Start/End/Step), arrows and text keys act normally; global shortcuts require Ctrl.

### Mouse
- **Drag on a data panel:** create a selection (press `a`/Enter or click **Add Label** to commit).
- **Wheel:** zoom around cursor (20%/notch by default).
- **Shift + Wheel:** pan left/right (20%/notch).
- **Strip (bottom):** click an interval to select; drag edges to resize; drag center to move.
- **Snapping:** Toggle **Snap to samples** to snap selection edges to the nearest timestamps in the current window.

## Saving & loading sessions
- **Save Session** writes a JSON file with label schema, window/step, and the list of labeled intervals.
- **Load Session** restores a prior session and redraws.
- **Autosave** can be enabled by passing `autosave_path` to `TimeIntervalLabeler`.

## Exporting labels

ChronoTagger provides three exporting paths:

### Export Labels… (ML-friendly per-sample export)
**Scope:** Full dataset (unlabeled = -1) or Selected intervals only.

**Content:**
- **Index + labels (CSV)** — two columns: time (index) and `label_id`
- **Full DataFrame + labels (CSV)** — original columns + `label_id`

**Writes:**
- The chosen CSV
- A sidecar `*_label_map.json` with `{label_name: id}` (0..N-1, order = current label list)

The `label_id` column uses the smallest signed integer dtype that fits the number of classes; unlabeled samples are `-1`.

### Export Intervals (CSV/Parquet)
One row per labeled interval with `start`, `end`, `label`, and optional `notes`.

### Export Per-Sample (CSV/Parquet)
Per-index label strings (legacy; use **Export Labels…** for ML-friendly integers and mapping).

## API (programmatic)

```python
from chronotagger import TimeIntervalLabeler

labeler = TimeIntervalLabeler(
    df: pd.DataFrame,                 # must have DatetimeIndex
    plot_fn: Callable,                # signature: plot_fn(axs, df_window, t0, t1)
    n_panels: int | None = None,      # optional; see panel resolution above
    classes: list[str] | None = None, # default: ["PlasmaSheet","Lobe","Magnetosheath","SolarWind","UNKNOWN"]
    class_colors: dict[str,str] | None = None,
    window: pd.Timedelta = "30min",
    step: pd.Timedelta   = "15min",
    start: pd.Timestamp | None = None,
    end:   pd.Timestamp | None = None,
    autosave_path: str | Path | None = None,
)

labeler.run()               # start Tk main loop
labeler.save(path=None)     # save session (JSON)
labeler.load(path)          # load session (JSON)
labeler.export_intervals(path, fmt="parquet"|"csv")
labeler.export_per_sample(path, fmt="parquet"|"csv")  # legacy, label strings

# Useful navigation helper
labeler.go_to_window(t0: pd.Timestamp)
```

## Project layout

```
chronotagger/
  __init__.py
  core/
    commands.py      # Command pattern for undo/redo: add/delete/relabel/resize
    models.py        # Interval dataclass
  labeler/
    app.py           # TimeIntervalLabeler composition
    dialogs/
      label_manager.py   # Manage Labels… modal (add/rename/delete/reorder/colors)
    mixins/
      events.py      # key/mouse bindings, selection, strip click, drag/resize/move
      help.py        # F1 help dialog (shortcuts)
      intervals.py   # add/delete/relabel, assign remainder, merge logic, undo/redo
      io_export.py   # save/load, ML-friendly export dialog & writers
      labels.py      # plumbing for label schema updates
      navigation.py  # prev/next, step parsing, entry sync
      plotting.py    # redraw data panels + strip; overlays
      stats.py       # sidebar list & stats
      view_build.py  # Tk layout & controls; Matplotlib embedding
      zoom.py        # wheel zoom/pan
    utils/
      overlays.py    # faint interval bands across panels
      timeaxis.py    # consistent time-axis formatting
```

The GUI class composes these mixins to keep individual files small and testable while presenting a single `TimeIntervalLabeler` type to users.

## Development & testing

Recommended: create a Conda env or venv with `pandas`, `matplotlib`, and `tk`.

If you use `pytest`, run:

```bash
python -m pytest -q
```

For image tests, ensure Matplotlib’s non-interactive backend is set (e.g., `MPLBACKEND=Agg`) and that your environment has a working Tk (or skip GUI tests in pure headless CI).

## Troubleshooting

### Tk/Tcl not found (Windows + Conda)
If you see errors like “Can’t find a usable init.tcl”, install Tk inside your env:

```bash
conda install -c anaconda tk
```

Also ensure you launch Python from the same environment where `tk` is installed.

### Matplotlib backends
This app embeds Matplotlib in Tk via `FigureCanvasTkAgg`. You don’t need to set a global backend, but avoid forcing a non-Tk backend in your environment when running the GUI.

### Large datasets
- Per-sample exports compute `label_id` by vectorized search over the index; performance is suitable for large indices (uses `searchsorted`).
- CSV size is kept in check by using the smallest integer dtype for `label_id` and a compact `{label: id}` JSON sidecar.

## Roadmap (ideas)
- Multiple rectangle selectors (one per panel) for selection anywhere.
- Optional class balancing / coverage guides.
- Plugin points for custom export targets.
- Hotkeys overlay toggles and step presets.

## License
Specify a license here (e.g., MIT) and include the license file in the repository.

## Acknowledgments
Thanks to the Matplotlib and pandas communities for the great foundations this tool builds upon.
