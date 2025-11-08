# ChronoTagger — Advanced Time-Series Interval Labeling Tool

ChronoTagger is a powerful, interactive GUI for labeling time intervals on matplotlib plots. Built for scientific data analysis, it provides a seamless workflow for annotating time-series data with ML-ready label exports. Whether you're analyzing spacecraft data, sensor measurements, or any temporal data, ChronoTagger adapts to your custom plotting code.

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

## Quick Start

```python
import pandas as pd
from chronotagger import TimeIntervalLabeler

# Your data (DataFrame with DatetimeIndex)
df = pd.read_csv("data.csv", index_col=0, parse_dates=True)

# Your custom plot function
def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["temperature"])
    axs["panel1"].set_ylabel("Temperature (°C)")
    
    axs["panel2"].plot(df.index, df["pressure"])  
    axs["panel2"].set_ylabel("Pressure (hPa)")

# Launch the labeler
app = TimeIntervalLabeler(df=df, plot_fn=plot_fn)
app.run()
```

## Installation

### Requirements
- Python 3.9+
- pandas
- matplotlib  
- numpy
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
    "nrows": 2, "ncols": 2,
    "areas": [
        {"key": "timeseries", "row": 0, "col": 0, "colspan": 2, "role": "time"},
        {"key": "xy_position", "row": 1, "col": 0, "role": "not-time"},
        {"key": "xz_position", "row": 1, "col": 1, "role": "not-time"},
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

ChronoTagger determines panel count in this order:
1. **Explicit:** `TimeIntervalLabeler(..., n_panels=3)`
2. **Function attribute:** `plot_fn.n_panels = 3`
3. **Auto-probe:** Calls plot_fn and counts axes with content
4. **Default:** 2 panels

## Exporting

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

# Enable autosave (saves on each change)
app = TimeIntervalLabeler(..., autosave_path="auto_backup.json")
```

## API Reference

### Main Class

```python
TimeIntervalLabeler(
    df: pd.DataFrame,                      # Data with DatetimeIndex
    plot_fn: Callable,                     # Your plot function
    layout_spec: dict = None,              # Panel layout specification
    n_panels: int = None,                  # Override panel count
    classes: List[str] = None,             # Label names (default: common plasma regions)
    class_colors: Dict[str, str] = None,   # Label colors (auto-generated if not provided)
    window: pd.Timedelta = "30min",        # View window size
    step: pd.Timedelta = "15min",          # Navigation step
    start: pd.Timestamp = None,            # Initial start time (default: df.index[0])
    end: pd.Timestamp = None,              # Initial end time (default: start + window)
    autosave_path: Path = None,            # Auto-save location
)
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

```python
# Core operations
app.run()                                  # Start the GUI main loop
app.add_interval(start, end, label)        # Programmatically add interval
app.delete_interval(interval)              # Remove specific interval
app.clear_all_intervals()                  # Remove all intervals

# Navigation
app.go_to_window(t0: pd.Timestamp)         # Jump to specific time
app.set_window_size(window: pd.Timedelta)  # Change window duration
app.set_step_size(step: pd.Timedelta)      # Change navigation step

# Import/Export
app.save(path: str = None)                 # Save session to JSON
app.load(path: str)                        # Load session from JSON
app.export_intervals(path: str, fmt: str)  # Export as CSV/Parquet
app.export_per_sample(path: str, fmt: str) # Legacy string labels export
app.export_labels(path: str, ...)          # ML-ready integer labels

# Label management
app.add_label(name: str, color: str)       # Add new label class
app.rename_label(old: str, new: str)       # Rename label (updates intervals)
app.delete_label(name: str)                # Remove unused label
app.recolor_label(name: str, color: str)   # Change label color
```

## Examples

See the `examples/` directory for complete demonstrations:

- `timeseries_only.py` - Basic time-series labeling
- `mixed_layout.py` - Combined time and position plots
- `layout_wizard_demo.py` - Interactive layout builder with sample data
- `layout_wizard_simple.py` - Minimal layout builder for CSV files
- `simple_layout_test.py` - Testing layout specifications

## Project Structure

```
chronotagger/
├── core/
│   ├── commands.py         # Undo/redo command pattern implementation
│   └── models.py          # Interval data model
├── labeler/
│   ├── app.py             # Main TimeIntervalLabeler class
│   ├── dialogs/           
│   │   ├── label_manager.py      # Comprehensive label management
│   │   ├── label_by_rule.py      # Rule-based labeling with conditions
│   │   └── overlap_resolution.py # Handle overlapping intervals
│   ├── mixins/            
│   │   ├── events.py      # Mouse/keyboard event handling
│   │   ├── intervals.py   # Interval CRUD operations
│   │   ├── io_export.py   # Save/load/export functionality
│   │   ├── labels.py      # Label schema management
│   │   ├── navigation.py  # Time window navigation
│   │   ├── plotting.py    # Plot rendering and updates
│   │   ├── rules.py       # Rule evaluation engine
│   │   ├── stats.py       # Statistics calculation
│   │   ├── view_build.py  # UI construction
│   │   └── zoom.py        # Zoom/pan functionality
│   └── utils/             
│       ├── colorbar.py    # Colorbar utilities for plots
│       ├── fastdraw.py    # Performance optimizations
│       ├── layout_builder.py      # Interactive layout designer
│       ├── overlays.py    # Interval overlay rendering
│       ├── plot_generator.py      # Auto-generate plot functions
│       └── timeaxis.py    # Time axis formatting
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