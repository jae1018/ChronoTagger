# ChronoTagger — Advanced Time-Series Interval Labeling Tool

ChronoTagger is a powerful, interactive GUI for labeling time intervals on matplotlib plots. Built for scientific data analysis, it provides a seamless workflow for annotating time-series data with ML-ready label exports. With temporal data, ChronoTagger can adapt to your custom plotting code.

![ChronoTagger quick-start](docs/assets/quickstart.gif)

*From a CSV to labeled intervals in under a minute. `chronotagger` launches a guided wizard that handles file loading, column selection, layout, and the labeling UI.*

## Key Features

- **Works with your matplotlib code** - Bring your own plot functions
- **Interactive layout builder** - Visually design multi-panel layouts without coding  
- **Mixed plot types** - Combine time-series with position/phase-space plots
- **Multiple selection modes** - Drag, two-click, or box selection with snap-to-samples
- **Rule-based labeling** - Apply labels using data conditions (AND/OR logic)
- **Flexible label assignment** - Fill gaps with any label via dialog
- **Overlap resolution** - Smart handling when selections overlap existing intervals
- **ML-ready exports** - Integer labels with JSON mapping for direct model training
- **Performance optimized** - PolyCollection overlays for 10-30× faster rendering
- **Complete undo/redo** - Full command stack with keyboard shortcuts
- **Professional label management** - Add, rename, recolor, reorder, reassign labels

## See it in action

### Multi-tab labeling

![Multi-tab labeling](docs/assets/multi-tab.gif)

Configure several tabs of plots in one wizard pass — each with its own column selection and layout — then label across them. Intervals stay synchronized: a label drawn on tab 1's strip shows up on tab 2's strip too.

### Custom grid designer

![Custom grid designer](docs/assets/custom-grid.gif)

For layouts the vertical-stack default doesn't cover, drop into the visual grid designer. Drag panels onto cells, switch a panel from time-series to cross-plot, pick its X/Y columns from dropdowns. The output is the same `layout_spec` dict you'd write by hand — without writing it.

### Rule-based labeling

![Rule-based labeling](docs/assets/by-rule.gif)

Skip the click-and-drag for systematic patterns: type `feat_1 >= 0`, hit Preview, see every matching span on screen, then commit. Combine multiple conditions with AND/OR, choose how to resolve overlaps with already-labeled intervals (skip vs replace), and apply to the current window or the entire dataset.

## Quick Start

### Option 1 — GUI wizard (no code)

After installing (see below), launch the wizard from any terminal:

```bash
chronotagger
```

The wizard walks you through file loading (CSV/Parquet), column selection, layout configuration, and drops you straight into the labeler. No Python required.

### Option 2 — Python API

For custom plot functions (e.g. spectrograms, multi-trace overlays, domain-specific overlays), construct the labeler directly:

```python
import pandas as pd
from chronotagger import TimeIntervalLabeler

# Your data (DataFrame with DatetimeIndex)
df = pd.read_csv("data.csv", index_col=0, parse_dates=True)

# Your custom plot function -- panel keys must match layout_spec.areas[*].key
def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["temperature"])
    axs["panel1"].set_ylabel("Temperature (°C)")

    axs["panel2"].plot(df.index, df["pressure"])
    axs["panel2"].set_ylabel("Pressure (hPa)")

# Layout: two time-series panels stacked, plus the auto-managed Labels strip
layout_spec = {
    "nrows": 3, "ncols": 1,
    "areas": [
        {"key": "panel1", "row": 0, "col": 0, "role": "time"},
        {"key": "panel2", "row": 1, "col": 0, "role": "time"},
        {"key": "labels", "row": 2, "col": 0, "role": "labels"},
    ],
}

app = TimeIntervalLabeler(df=df, plot_fn=plot_fn, layout_spec=layout_spec)
app.run()
```

## Installation

### Requirements
- Python 3.9+
- pandas
- numpy
- matplotlib
- pyarrow (for Parquet I/O)
- tkinter (usually included with Python)

### Install from source
```bash
git clone <repository-url>
cd chronotagger
pip install -e .
```

### Platform Notes

**Windows with Conda:**
```bash
conda install -c anaconda tk  # Ensures working Tcl/Tk
```

**Linux/Mac:**
Tkinter typically comes with Python. If missing:
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Mac with Homebrew
brew install python-tk
```

## Core Concepts

### Selection Modes

ChronoTagger offers multiple ways to select time intervals:

1. **Drag Selection** (Time-series plots)
   - Click and drag horizontally to select a time range
   - Works on any time-series panel
   - Creates single continuous interval

2. **Two-Click Selection** (Alternative for time-series)
   - Click once for start time
   - Click again for end time
   - Shows yellow preview between clicks

3. **Box Selection** (Time or position plots)
   - Drag a rectangle on time-series plots to select by time AND value
   - Drag on position plots (`role="not-time"`) to select spatial regions
   - Can create multiple disjoint time spans from single selection

4. **Snap to Samples**
   - Toggle in UI to snap selection edges to nearest data points
   - Ensures selections align with actual timestamps in data
   - Useful for precise interval boundaries

### Overlap Resolution

When adding intervals that overlap existing ones:

1. **Automatic Detection** - System detects overlaps before adding
2. **Resolution Dialog** - Choose how to handle:
   - **Skip** - Only label unassigned regions (preserve existing)
   - **Replace** - Delete conflicting intervals (new takes priority)
3. **Preview** - See result before confirming
4. **Undo Support** - All operations reversible with Ctrl+Z

## Advanced Features

### Interactive Layout Builder

Design complex layouts visually without writing layout code:

```python
from chronotagger.labeler.utils import build_layout, generate_plot_fn

# Launch interactive layout designer
layout_spec, plot_config = build_layout(df)

# Auto-generate plot function from your design
plot_fn = generate_plot_fn(plot_config)

# Use with labeler
app = TimeIntervalLabeler(df=df, plot_fn=plot_fn, layout_spec=layout_spec)
app.run()
```

The layout builder provides:
- Visual grid-based panel arrangement
- Drag to create spanning panels (e.g., 2×1, 1×2)
- Variable assignment via dropdowns
- Live matplotlib preview
- Auto-managed Labels strip at bottom
- Support for time-series and cross-plots

### Mixed Plot Types

Combine time-series with position/phase-space plots:

```python
layout_spec = {
    "nrows": 3, "ncols": 2,
    "areas": [
        {"key": "timeseries",  "row": 0, "col": 0, "colspan": 2, "role": "time"},
        {"key": "xy_position", "row": 1, "col": 0, "role": "not-time"},
        {"key": "xz_position", "row": 1, "col": 1, "role": "not-time"},
        {"key": "labels",      "row": 2, "col": 0, "colspan": 2, "role": "labels"},
    ]
}

def plot_fn(axs, df, t0, t1):
    # Time-series plot
    axs["timeseries"].plot(df.index, df["value"])

    # Position plots (CRITICAL: preserve temporal ordering!)
    # Point N must correspond to df.index[N]
    axs["xy_position"].scatter(df["x"], df["y"], s=5)
    axs["xz_position"].scatter(df["x"], df["z"], s=5)

app = TimeIntervalLabeler(df=df, plot_fn=plot_fn, layout_spec=layout_spec)
```

**Key insight:** For position plots, maintain DataFrame order - point N maps to `df.index[N]`, enabling box selection in position space to create time intervals.

### Label Management Dialog

Comprehensive label control via "Manage Labels..." button:

- **Add Labels** - Create new label classes with custom colors
- **Rename** - Change label names (updates all existing intervals)
- **Delete** - Remove unused labels
- **Reorder** - Drag to change display order (affects number shortcuts)
- **Recolor** - Pick colors via color dialog or hex input
- **Reassign** - Replace all instances of one label with another
- **Validation** - Prevents duplicate names, ensures at least one label exists

### Rule-Based Labeling

Apply labels based on data conditions via "Label by Rule..." button:

```python
# Dialog supports complex rules like:
# - Single condition: BX > 5
# - Multiple with AND: BX > 5 AND density < 10
# - Multiple with OR: X_GSE > 0 OR velocity > 400

# Features:
# - Add unlimited conditions
# - Combine with AND/OR logic
# - NaN handling (treat as True or False)
# - Overlap policies (skip existing or replace)
# - Scope control (current window/entire dataset/custom range)
# - Preview before applying
# - Full undo support
```

### Flexible Gap Filling

The "Label Unassigned..." button opens a dialog to:
- Select any existing label for unassigned gaps
- Preview gap count before applying ("Will create N interval(s)")
- See affected time range
- Double-click or Enter to quickly confirm
- Maintains full undo/redo support

### Performance Optimizations

- **PolyCollection overlays** - Multi-span overlays use single collection (10-30× faster)
- **Fast draw mode** - Optimized redraw for large datasets
- **Vectorized operations** - NumPy-based label assignment
- **Smart caching** - Reuses matplotlib artists where possible
- **Efficient slicing** - Pandas datetime indexing for data windows

### Multi-Pane Interface

ChronoTagger supports multiple tabbed panes for viewing different visualizations of the same data simultaneously. This is useful when labeling requires many plots that won't fit comfortably in a single figure.

#### Basic Usage

Create multiple panes with different plot functions:

```python
import pandas as pd
from chronotagger import TimeIntervalLabeler

# Load your data
df = pd.read_parquet("magnetosphere_data.parquet")

# Define plot functions for each pane
def plot_overview(axs, df, t0, t1):
    axs['density'].plot(df.index, df['n_density'])
    axs['density'].set_ylabel('Density (cm⁻³)')

    axs['temp'].semilogy(df.index, df['temperature'])
    axs['temp'].set_ylabel('Temperature (K)')

def plot_fields(axs, df, t0, t1):
    for component in ['Bx', 'By', 'Bz']:
        axs['b_field'].plot(df.index, df[component], label=component)
    axs['b_field'].set_ylabel('B (nT)')
    axs['b_field'].legend()

def plot_positions(axs, df, t0, t1):
    axs['xy'].scatter(df['X_GSE'], df['Y_GSE'], s=1)
    axs['xy'].set_xlabel('X (RE)')
    axs['xy'].set_ylabel('Y (RE)')

# Define layouts for each pane
layout_overview = {
    "nrows": 3,
    "ncols": 1,
    "areas": [
        {"key": "density", "row": 0, "col": 0, "role": "time"},
        {"key": "temp", "row": 1, "col": 0, "role": "time"},
        {"key": "labels", "row": 2, "col": 0, "role": "labels"},
    ]
}

layout_fields = {
    "nrows": 2,
    "ncols": 1,
    "areas": [
        {"key": "b_field", "row": 0, "col": 0, "role": "time"},
        {"key": "labels", "row": 1, "col": 0, "role": "labels"},
    ]
}

layout_positions = {
    "nrows": 2,
    "ncols": 1,
    "areas": [
        {"key": "xy", "row": 0, "col": 0, "role": "not-time"},
        {"key": "labels", "row": 1, "col": 0, "role": "labels"},
    ]
}

# Create multi-pane labeler
panes = [
    {
        "title": "Overview",
        "plot_fn": plot_overview,
        "layout_spec": layout_overview,
    },
    {
        "title": "Magnetic Field",
        "plot_fn": plot_fields,
        "layout_spec": layout_fields,
    },
    {
        "title": "Position",
        "plot_fn": plot_positions,
        "layout_spec": layout_positions,
    },
]

labeler = TimeIntervalLabeler(
    df=df,
    panes=panes,  # Use 'panes' instead of 'plot_fn'
    window=pd.Timedelta("4h"),
    step=pd.Timedelta("30min"),
)

labeler.run()
```

#### Features

**Tab Navigation:**
- Click tabs to switch between panes
- `Ctrl+Tab` / `Ctrl+Shift+Tab` - Next/previous tab
- `Ctrl+1` through `Ctrl+9` - Jump directly to tabs 1-9
- `Ctrl+0` - Jump to tab 10

**Tab Management:**
- Right-click tab → Rename, refresh
- Custom tab names persist in saved sessions

**Synchronized State:**
- All panes share the same intervals and labels
- Time window synchronized across panes
- Labeling on any pane affects all panes

**Performance:**
- Only active pane updates in real-time
- Inactive panes update when switched to
- Efficient blitting for fast overlay rendering

#### When to Use Multi-Pane

✅ **Good use cases:**
- Need to see 10+ plots for accurate labeling
- Different views of same data (time series + position plots)
- Comparing different parameter combinations
- Separating overview from detailed plots

❌ **Not recommended:**
- Only need 2-3 plots (use single pane with subplots)
- Very slow plot functions (will multiply startup time)
- More than 5-6 panes (gets cluttered)

#### Migrating from Single to Multi-Pane

Existing single-pane code still works:

```python
# Old way (still supported)
labeler = TimeIntervalLabeler(
    df=df,
    plot_fn=my_plot_function,
    layout_spec=my_layout,
)

# New multi-pane way
panes = [
    {"title": "Main View", "plot_fn": my_plot_function, "layout_spec": my_layout}
]
labeler = TimeIntervalLabeler(df=df, panes=panes)
```

See `examples/dual_pane_demo.py` and `examples/multi_pane_magnetosphere.py` for complete examples.

#### Real-world example: cislunar plasma classification

A two-pane setup driving real ARTEMIS-mission ion data. Pane 1 shows a 32-channel ion energy-flux spectrogram alongside density, B-field, and spacecraft potential time series; pane 2 shows the derived thermodynamic + velocity quantities. Both panes share the same orbital cross-plot panels — the GSE (Earth-centered) and SSE (Moon-centered) frames with bow shock and magnetopause boundaries overlaid — and any label drawn on either pane appears on both panes' Labels strips.

| Pane 1: spectra + fields | Pane 2: thermo + dynamics |
|:---:|:---:|
| ![Pane 1](docs/assets/spectrogram-flagship.png) | ![Pane 2](docs/assets/spectrogram-pane2.png) |

Driver code: `examples/spectrogram_multipane.py`.

## User Interface

### Main Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Time Range │ Navigation │ Current Label │ Actions           │  ← Top Bar
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Your matplotlib plots (stacked vertically)                 │  ← Plot Area
│  - Drag to select                                           │
│  - Scroll to zoom                                           │
│  - Box select for value-based selection                     │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  ████████  ██████  █████████  ████                          │  ← Labels Strip
└─────────────────────────────────────────────────────────────┘
                                              ┌───────────────┐
                                              │ Intervals     │  ← Sidebar
                                              │ • start→end   │
                                              │ • label       │
                                              │ • duration    │
                                              ├───────────────┤
                                              │ Statistics    │
                                              │ • Coverage %  │
                                              │ • Label counts│
                                              ├───────────────┤
                                              │ Actions       │
                                              │ Export        │
                                              └───────────────┘
```

### Navigation Controls

- **Time Range** - Direct entry fields for start/end times
- **Step Size** - Adjustable step for navigation
- **Quick Adjust** - `x2` and `/2` buttons to double/halve step
- **Prev/Next** - Navigate by current step size
- **Snap Toggle** - Enable/disable snap-to-samples

### Keyboard Shortcuts

| Key | Action | | Key | Action |
|-----|--------|-|-----|--------|
| `1-9` | Select label by index | | `n` / `→` | Next window |
| `a` / `Enter` | Add interval | | `p` / `←` | Previous window |
| `d` / `Delete` | Delete selected | | `Ctrl+Z` / `Backspace` | Undo |
| `u` | Select UNKNOWN label | | `Ctrl+Y` / `Shift+Backspace` | Redo |
| `r` | Relabel selected | | `Ctrl+S` | Save session |
| `m` | Manage labels | | `Ctrl+E` | Export dialog |
| `F1` | Help dialog | | `Escape` | Clear selection |

**Multi-Pane Shortcuts (when using tabbed interface):**

| Key | Action |
|-----|--------|
| `Ctrl+Tab` | Switch to next tab |
| `Ctrl+Shift+Tab` | Switch to previous tab |
| `Ctrl+1` to `Ctrl+9` | Jump to tab 1-9 |
| `Ctrl+0` | Jump to tab 10 |
| Right-click tab | Show tab menu (rename, refresh) |

### Mouse Controls

- **Drag on plot** - Create time selection
- **Drag rectangle** - Box selection (time+value or position)
- **Scroll wheel** - Zoom in/out (20% per notch)
- **Shift + wheel** - Pan left/right
- **Click interval** - Select in labels strip
- **Drag edges** - Resize intervals
- **Drag center** - Move intervals
- **Double-click** - Quick actions in dialogs

### Sidebar Features

**Labeled Intervals List:**
- Shows all intervals in current window
- Format: `start -> end: label (duration)`
- Click to select
- Double-click to jump to interval

**Statistics Panel:**
- Total/labeled/unlabeled point counts
- Coverage percentage
- Per-label statistics (count, coverage %)
- Updates live as you label

**Action Buttons:**
- Relabel - Change selected interval's label
- Delete - Remove selected interval
- Label Unassigned... - Fill gaps with chosen label
- Clear All Intervals - Remove all labels (with confirmation)

## Data Requirements

### Input DataFrame
- Must have a `DatetimeIndex`
- All plotted columns must exist in the DataFrame
- Numeric columns for plotting (automatically detected)
- No missing index values (continuous time series preferred)

### Plot Function Contract

Your plot function receives:
```python
def plot_fn(
    axs: dict[str, matplotlib.axes.Axes],  # Panel key → axis
    df: pd.DataFrame,                       # Data slice for [t0, t1]
    t0: pd.Timestamp,                       # Window start
    t1: pd.Timestamp                        # Window end
) -> None:
    # Draw on the provided axes
    # df is already sliced to the time window
    axs["panel1"].plot(df.index, df["column"])
```

### Panel Count Resolution

The set of panels is fully determined by `layout_spec.areas`: every entry with a unique `key` becomes a panel and is passed to your `plot_fn` as `axs[key]`. Add entries to grow or shrink the layout — there is no separate panel-count argument.

## Exporting

![Export Labels dialog](docs/assets/export-dialog.png)

### ML-Ready Export (Recommended)

Via "Export Labels..." button or `app.export_labels()`:

**Options:**
- Scope: Full dataset or selected intervals only
- Format: Index + labels only, or full DataFrame + labels
- Output: CSV with integer labels + JSON mapping

```python
# Produces two files:
# 1. data_labels.csv:
#    timestamp,label_id
#    2024-01-01 00:00:00,-1    # Unlabeled
#    2024-01-01 00:05:00,0     # PlasmaSheet
#    2024-01-01 00:10:00,1     # Magnetosheath

# 2. data_labels_label_map.json:
#    {"PlasmaSheet": 0, "Magnetosheath": 1, "SolarWind": 2}
```

- Uses smallest integer dtype (int8/int16/int32) for efficiency
- Unlabeled samples marked as -1
- Label order matches current display order

### Interval Export

```python
# CSV or Parquet with columns:
# start, end, label, notes (optional)
app.export_intervals("intervals.parquet", fmt="parquet")
```

### Session Save/Load

```python
# Save complete session (labels, intervals, settings)
app.save("session.json")

# Resume later
app.load("session.json")

# Autosave-on-change to a folder of your choice (default: ".")
# Each modification rewrites <autosave_folder>/chronotagger_autosave.json,
# which the labeler offers to recover from on the next launch.
app = TimeIntervalLabeler(..., autosave_folder="autosaves")
```

## API Reference

### Main Class

```python
TimeIntervalLabeler(
    df: pd.DataFrame,                      # Data with DatetimeIndex (required)

    # Single-pane mode
    plot_fn: Callable = None,              # Plot function: fn(axs, df, t0, t1)
    classes: List[str] = None,             # Label names (default: ["UNKNOWN", "label_1", "label_2"])
    class_colors: Dict[str, str] = None,   # Label -> color (auto-generated if omitted)
    window: pd.Timedelta = "30min",        # Initial visible window
    step: pd.Timedelta = "15min",          # Prev/Next navigation step
    start: pd.Timestamp = None,            # Initial window start (default: df.index[0])
    autosave_folder: str = ".",            # Folder for chronotagger_autosave.json

    # Keyword-only:
    *,
    layout_spec: dict = None,              # Panel layout (single-pane); see below
    panes: List[Dict] = None,              # Pane configs (multi-pane); see below
    parent: tk.Misc = None,                # Existing Tk root to mount under (used by the wizard).
                                           # If None, creates its own tk.Tk().
)

# Either `plot_fn` (single-pane) or `panes` (multi-pane) is required, not both.

# Pane configuration (for multi-pane mode):
panes = [
    {
        "title": str,                      # Tab label
        "plot_fn": Callable,               # Plot function for this pane
        "layout_spec": dict,               # Layout for this pane
    },
    # ... up to several panes
]
```

### Layout Specification

```python
layout_spec = {
    "nrows": 3,                            # Grid rows
    "ncols": 2,                            # Grid columns
    "height_ratios": [1, 1, 0.5],          # Relative heights (optional)
    "width_ratios": [2, 1],                # Relative widths (optional)
    "hspace": 0.15,                        # Vertical spacing (optional)
    "wspace": 0.12,                        # Horizontal spacing (optional)
    "areas": [                             # Panel definitions
        {
            "key": "panel_name",           # Unique identifier (required)
            "row": 0,                      # Grid position (required)
            "col": 0,                      # Grid position (required)
            "rowspan": 2,                  # Span multiple rows (optional, default: 1)
            "colspan": 1,                  # Span multiple columns (optional, default: 1)
            "role": "time"                 # "time" or "not-time" (optional, default: "time")
        }
    ]
}
```

### Methods

The Python surface is intentionally small — most user actions are driven through the GUI (label management, interval editing, by-rule labeling, ML-ready export, etc.).

```python
# Lifecycle
app.run()                                  # Build the GUI and start the event loop

# Session save/load (JSON)
app.save(path: str = None)                 # Save session; prompts via file dialog if path=None
app.load(path: str)                        # Load a session JSON

# Export
app.export_intervals(path: str, fmt="parquet")  # Per-interval rows: start, end, label, notes
app.export_per_sample(path: str, fmt="parquet", label_on_uncovered="UNKNOWN")
                                           # Per-row labels for the entire df.index

# Navigation
app.go_to_window(t0: pd.Timestamp)         # Jump the visible window so it begins at t0
```

The richer ML-ready export (integer label IDs + JSON label-map) is reachable from the **Export Labels...** button in the GUI; programmatic access is on the roadmap.

## Examples

See the `examples/` directory for complete demonstrations:

- `timeseries_only.py` - Basic time-series labeling
- `mixed_layout.py` - Combined time and position plots
- `layout_wizard_demo.py` - Interactive layout builder with sample data
- `layout_wizard_simple.py` - Minimal layout builder for CSV files
- `simple_layout_test.py` - Testing layout specifications
- `dual_pane_demo.py` - Simple 2-tab multi-pane example
- `multi_pane_magnetosphere.py` - Comprehensive 4-tab space physics example
- `spectrogram_multipane.py` - Real-world cislunar plasma classification driver: ion energy spectrogram (pcolormesh), B-field components, and orbital cross-plots in both Earth- and Moon-centered frames. Optional `geospacefronts` overlay for bow-shock/magnetopause boundaries. Bring your own dataset via the `CHRONOTAGGER_EXAMPLE_DATA` env var.

## Project Structure

```
src/chronotagger/
├── launcher.py                 # `chronotagger` console-script entry point
├── core/
│   ├── commands.py             # Undo/redo command pattern
│   └── models.py               # Interval data model
├── quickstart/                 # GUI wizard (file loader → tab planner → labeler)
│   ├── wizard.py
│   ├── file_loader.py
│   ├── tab_planner.py
│   ├── plot_builder.py
│   └── config.py
└── labeler/
    ├── app.py                  # TimeIntervalLabeler (composes all mixins)
    ├── tab_pane.py             # Per-pane state for multi-pane mode
    ├── sync.py                 # Keeps intervals/window in sync across panes
    ├── dialogs/
    │   ├── label_manager.py        # Manage labels (add/rename/recolor/...)
    │   ├── label_by_rule.py        # Rule-based labeling
    │   └── overlap_resolution.py   # Skip-vs-replace dialog
    ├── mixins/
    │   ├── events/                 # Mouse / keyboard / selection / overlays / strip
    │   ├── intervals/              # CRUD, validation, gap-fill, merge
    │   ├── view_build/             # canvas, controls, sidebar, window, widgets
    │   ├── help.py
    │   ├── io_export.py            # Save/load/export
    │   ├── labels.py               # Label schema management
    │   ├── navigation.py           # Time-window navigation
    │   ├── plotting.py             # Plot rendering and updates
    │   ├── rules.py                # Rule evaluation engine
    │   ├── stats.py                # Coverage statistics
    │   └── zoom.py                 # Zoom / pan
    └── utils/
        ├── colorbar.py
        ├── fastdraw.py
        ├── layout_builder/         # Interactive grid designer (build_layout)
        ├── overlays.py
        ├── plot_generator.py       # Auto-generate plot_fn from a layout config
        └── timeaxis.py
```

## Testing

```bash
# Run test suite
python -m pytest tests/

# Run with coverage
python -m pytest --cov=chronotagger tests/

# Run specific test file
python -m pytest tests/test_interval_operations.py

# Run with verbose output
python -m pytest -v tests/
```

## Performance Tips

- **Large datasets** (>1M points)
  - Use appropriate window/step sizes to limit visible data
  - Consider downsampling for visualization
  - Export to Parquet for faster I/O

- **Many panels** (>6)
  - Use layout builder to optimize arrangement
  - Consider reducing panel count or using tabs

- **Memory optimization**
  - Export large label sets as Parquet (compressed)
  - Use categorical dtype for label columns

## License

MIT License - See LICENSE file for details

## Acknowledgments

ChronoTagger builds upon the excellent foundations provided by:
- [matplotlib](https://matplotlib.org/) - Plotting library
- [pandas](https://pandas.pydata.org/) - Data manipulation
- [numpy](https://numpy.org/) - Numerical computing
- [tkinter](https://docs.python.org/3/library/tkinter.html) - GUI framework