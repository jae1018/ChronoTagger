# ChronoTagger Architecture

This is the bird's-eye view of the source tree for contributors. End-users do not need to read this — the README + the API reference are enough.

## Package layout

```
src/chronotagger/
├── launcher.py                 # `chronotagger` console-script entry point
├── core/
│   ├── commands.py             # Undo/redo command pattern
│   └── models.py               # Interval data model
├── quickstart/                 # GUI wizard (file loader -> tab planner -> labeler)
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

## How the pieces fit together

- **`launcher.py`** is the `chronotagger` console script. It launches `quickstart.wizard.run()`, which guides the user through file loading and tab/layout configuration, then constructs a `TimeIntervalLabeler` with the configured panes and runs it.

- **`labeler/app.TimeIntervalLabeler`** is the heavyweight class. Behavior is split across mixins so any one file stays readable:
  - `core.commands` provides gesture-based undo/redo: each user gesture is captured as one `GestureCommand` holding before/after snapshots of the interval list, and undo/redo restore those snapshots wholesale. Session loads, autosave recovery, and label-schema edits invalidate the undo history instead of participating in it.
  - `mixins.intervals` owns interval CRUD, overlap policy resolution, gap-fill, and merging.
  - `mixins.events` (subpackage) handles mouse/keyboard input, drag-selection, two-click selection, box selection on time and not-time axes, and highlight overlays.
  - `mixins.view_build` (subpackage) builds the Tk window, the matplotlib canvas, the toolbar, and the sidebar.
  - `mixins.plotting`, `mixins.navigation`, `mixins.zoom` handle the visible-window state.
  - `mixins.rules` evaluates the by-rule conditions against the dataframe and emits preview spans.
  - `mixins.io_export` saves/loads sessions and exports interval CSVs / Parquet.

- **Multi-pane mode** is mediated by `labeler.tab_pane.TabPane` (per-pane figure/canvas/axes state) and `labeler.sync.PaneSyncManager` (broadcasts interval changes and window-time changes across panes).

- **`utils.layout_builder`** is the interactive grid designer reachable from the wizard's Custom Grid path. It produces the same `layout_spec` dict you'd write by hand and a corresponding `plot_config` that `utils.plot_generator.generate_plot_fn` can compile into a runnable `plot_fn`.

## Tk root invariant

Only one `tk.Tk()` ever exists in a running process. The wizard creates it; when the labeler is launched from the wizard it mounts itself as a `tk.Toplevel(parent=wizard.root)` rather than calling `tk.Tk()` again. When the labeler is launched standalone (`TimeIntervalLabeler(...).run()` from a script), it creates its own `tk.Tk()` and `run()` blocks via `mainloop()` instead of `wait_window()`.

This invariant is load-bearing: `tk.StringVar()` and friends bind to `tk._default_root` (set on first Tk creation), so cross-`Tk` references silently break textvariable wiring. Maintain it.
