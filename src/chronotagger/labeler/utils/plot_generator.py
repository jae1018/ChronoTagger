"""
Plot Function Generator for ChronoTagger

This module provides utilities to automatically generate plot functions from
configuration dictionaries. This eliminates the need for users to manually
write plot_fn code for simple plotting scenarios.

The generated plot functions support:
- Time-series plots (line and scatter)
- Cross-plots / scatter plots (any column vs any column)
- A log or linear y scale
- Basic styling and labeling

For more complex plotting needs, users should write custom plot functions.

THIS IS THE PACKAGE'S ONE PLOT GENERATOR (Pack 7 W1). The quick-start
wizard used to carry a second one, `quickstart.plot_builder`, which
could not express a cross-plot at all, accumulated artists on repeat
renders, and used its own `panel0` key scheme. It was folded in here:
`vertical_stack_config` IS that preset, emitting the same shapes the
interactive grid designer emits, so the live wizard, the designer and
the generated driver file all agree on one vocabulary.

Usage:
    from chronotagger.labeler.utils import build_layout, generate_plot_fn

    layout_spec, plot_config = build_layout(df)             # designer
    plot_fn = generate_plot_fn(plot_config)

    from chronotagger.labeler.utils import vertical_stack_config
    layout_spec, plot_config = vertical_stack_config(cols)  # preset

Author: ChronoTagger Team
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple
import matplotlib.pyplot as plt
import pandas as pd


def generate_plot_fn(plot_config: Dict[str, Dict[str, Any]]) -> Callable:
    """
    Generate a plot function from a plot configuration dictionary.
    
    This function creates a plotting function that can be passed directly to
    TimeIntervalLabeler. The generated function handles basic plotting for
    time-series and cross-plot panels.
    
    Args:
        plot_config: Dictionary mapping panel keys to their configurations.
            Each panel config should have:
            - 'role': Either "time" or "not-time"
            - 'y_column': Column name for y-axis (time plots)
            - 'x_column': Column name for x-axis (not-time plots)
            - 'y_column': Column name for y-axis (not-time plots)
            
            Optional fields:
            - 'style': "line" or "scatter" (default: "line" for time, "scatter" for not-time)
            - 'color': Color for the plot (default: "blue" for time, "green" for not-time)
            - 'ylabel': Custom y-axis label (default: column name)
            - 'xlabel': Custom x-axis label (default: column name for not-time plots)
            - 'title': Custom panel title (default: none)
            - 'grid': Whether to show grid (default: True)
    
    Returns:
        A plotting function with signature: plot_fn(axs, df, t0, t1)
        This function can be passed to TimeIntervalLabeler constructor.
    
    Example:
        >>> plot_config = {
        ...     'panel_1': {
        ...         'role': 'time',
        ...         'y_column': 'density',
        ...         'ylabel': 'Density [cm^-3]',
        ...     },
        ...     'panel_2': {
        ...         'role': 'not-time',
        ...         'x_column': 'X_GSE',
        ...         'y_column': 'Y_GSE',
        ...         'xlabel': 'X (GSE) [RE]',
        ...         'ylabel': 'Y (GSE) [RE]',
        ...     }
        ... }
        >>> plot_fn = generate_plot_fn(plot_config)
        >>> # Now use plot_fn with TimeIntervalLabeler
    
    Notes:
        - The generated function uses simple line/scatter plots
        - For complex visualizations (multiple traces, colormaps, etc.),
          write a custom plot_fn instead
        - Generated functions handle missing data gracefully
    """
    
    def generated_plot_fn(axs: Dict[str, plt.Axes], df: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp):
        """
        Auto-generated plotting function.
        
        Args:
            axs: Dictionary of matplotlib axes (key -> Axes)
            df: DataFrame containing the data (already filtered to [t0, t1])
            t0: Start time of current window
            t1: End time of current window
        """
        # Loop through each panel configuration
        for panel_key, config in plot_config.items():
            # Get the corresponding axis
            if panel_key not in axs:
                continue  # Panel might not exist if layout changed
            
            ax = axs[panel_key]
            # Remove only data artists. Calling ax.clear() here would also
            # reset shared-axis relationships and the datetime unit converter,
            # which the labeler's _update_plot already set up before invoking
            # this plot_fn. Resetting them here makes the labeler's
            # subsequent set_xlim(t0, t1) only apply to whichever axis it
            # touches first, leaving the other axis with its auto-fit
            # (60-year-wide) range and rendering its line invisibly.
            for ln in list(ax.lines):
                ln.remove()
            for coll in list(ax.collections):
                coll.remove()
            for patch in list(ax.patches):
                patch.remove()
            for txt in list(ax.texts):
                txt.remove()
            
            role = config.get('role', 'time')
            
            # === TIME-SERIES PLOTS ===
            if role == 'time':
                y_col = config.get('y_column')
                if y_col is None or y_col not in df.columns:
                    # Column doesn't exist, show error message
                    ax.text(
                        0.5, 0.5,
                        f"Error: Column '{y_col}' not found",
                        ha='center', va='center',
                        transform=ax.transAxes,
                        color='red'
                    )
                    continue
                
                # Get data
                x_data = df.index
                y_data = df[y_col]
                
                # Check for empty data
                if len(x_data) == 0:
                    ax.text(
                        0.5, 0.5,
                        "No data in window",
                        ha='center', va='center',
                        transform=ax.transAxes,
                        color='gray'
                    )
                    continue
                
                # Plot style
                style = config.get('style', 'line')
                color = config.get('color', '#1f77b4')  # matplotlib default blue
                
                if style == 'scatter':
                    ax.scatter(x_data, y_data, s=3, c=color, alpha=0.7)
                else:  # line
                    ax.plot(x_data, y_data, color=color, linewidth=1.0)

                # Axis scale (Pack 7 W7). The runtime function and the
                # emitted driver must be able to express the SAME
                # figure, and a log y is the first thing a density or
                # flux panel needs. An absent key leaves matplotlib's
                # default untouched, so every existing plot_config
                # renders byte-identically.
                yscale = config.get('yscale')
                if yscale:
                    ax.set_yscale(yscale)

                # Labels
                ylabel = config.get('ylabel', y_col)
                ax.set_ylabel(ylabel, fontsize=9)
                
                # Grid
                if config.get('grid', True):
                    ax.grid(alpha=0.3, linewidth=0.5)
                
                # Title (optional)
                title = config.get('title')
                if title:
                    ax.set_title(title, fontsize=10)
            
            # === NOT-TIME (CROSS-PLOT) ===
            elif role == 'not-time':
                x_col = config.get('x_column')
                y_col = config.get('y_column')
                
                # Validate columns exist
                if x_col is None or x_col not in df.columns:
                    ax.text(
                        0.5, 0.5,
                        f"Error: Column '{x_col}' not found",
                        ha='center', va='center',
                        transform=ax.transAxes,
                        color='red'
                    )
                    continue
                
                if y_col is None or y_col not in df.columns:
                    ax.text(
                        0.5, 0.5,
                        f"Error: Column '{y_col}' not found",
                        ha='center', va='center',
                        transform=ax.transAxes,
                        color='red'
                    )
                    continue
                
                # Get data
                x_data = df[x_col]
                y_data = df[y_col]
                
                # Check for empty data
                if len(x_data) == 0:
                    ax.text(
                        0.5, 0.5,
                        "No data in window",
                        ha='center', va='center',
                        transform=ax.transAxes,
                        color='gray'
                    )
                    continue
                
                # Plot style
                style = config.get('style', 'scatter')
                color = config.get('color', '#2ca02c')  # matplotlib default green
                
                if style == 'line':
                    ax.plot(x_data, y_data, color=color, linewidth=1.0)
                else:  # scatter
                    ax.scatter(x_data, y_data, s=5, c=color, alpha=0.6)

                # Axis scale (Pack 7 W7), same key as the time branch.
                yscale = config.get('yscale')
                if yscale:
                    ax.set_yscale(yscale)

                # Labels
                xlabel = config.get('xlabel', x_col)
                ylabel = config.get('ylabel', y_col)
                ax.set_xlabel(xlabel, fontsize=9)
                ax.set_ylabel(ylabel, fontsize=9)
                
                # Grid
                if config.get('grid', True):
                    ax.grid(alpha=0.3, linewidth=0.5)
                
                # Title (optional)
                title = config.get('title')
                if title:
                    ax.set_title(title, fontsize=10)
                
                # Equal aspect for position plots (optional)
                if config.get('equal_aspect', False):
                    ax.set_aspect('equal', adjustable='box')

    return generated_plot_fn


def vertical_stack_config(
    selected_columns: Sequence[str],
    *,
    hspace: float = 0.15,
    wspace: float = 0.12,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build (layout_spec, plot_config) for one time panel per column.

    This is the wizard's "Vertical Stack" choice, expressed as a PRESET
    over this module's vocabulary rather than as a second generator
    (Pack 7 W1). What the deleted `quickstart.plot_builder` produced and
    what this produces differ in four measured ways, all of them the
    designer's shape winning:

    - keys are `panel_1..panel_N`, not `panel0..panel_{N-1}`;
    - `rowspan` / `colspan` are emitted only when they exceed 1;
    - `hspace` / `wspace` are present;
    - a `plot_config` comes back too, so the same columns can drive a
      generated driver file and not just a runtime closure.

    The labels strip is appended in the bottom row, spanning the single
    column the stack occupies. No `labels` entry is put in plot_config:
    the labeler holds that strip separately from `axs`, and a config
    entry for it is what made the old code emitter raise KeyError.

    Args:
        selected_columns: Column names, one time panel each, in order.
        hspace / wspace: GridSpec spacing, matching the designer's
            defaults so a stack and a designed grid look alike.

    Returns:
        (layout_spec, plot_config), the same pair `build_layout()`
        returns.

    Raises:
        ValueError: If no columns were selected.
    """
    columns = list(selected_columns)
    if not columns:
        raise ValueError(
            "vertical_stack_config needs at least one column; got none.")

    areas: list = []
    plot_config: Dict[str, Any] = {}
    for i, col in enumerate(columns, start=1):
        key = 'panel_%d' % i
        areas.append({
            'key': key,
            'row': i - 1,
            'col': 0,
            'role': 'time',
        })
        plot_config[key] = {'role': 'time', 'y_column': col}

    areas.append({
        'key': 'labels',
        'row': len(columns),
        'col': 0,
        'role': 'labels',
    })

    layout_spec = {
        'nrows': len(columns) + 1,
        'ncols': 1,
        'hspace': hspace,
        'wspace': wspace,
        'areas': areas,
    }
    return layout_spec, plot_config


def normalize_time_columns(layout_spec: Dict[str, Any]) -> None:
    """
    Coerce every 'time' and 'labels' area in layout_spec to share the same
    column extent as the FIRST 'time' area found. Modifies layout_spec in
    place. No-op if layout_spec has no 'time' areas.

    Within a single pane, the user expects time-series panels and the
    labels strip to be horizontally aligned. The interactive layout
    designer doesn't enforce this constraint as the user designs, so we
    coerce after the fact: whatever the user picked for the first time
    panel's col/colspan becomes the target, and every other time/labels
    area is moved/resized to match.

    Cross-plot ('not-time') areas are left alone -- those are typically
    side panels with their own column layout.

    (Moved here from the deleted `quickstart.plot_builder` in Pack 7 W1,
    byte-for-byte. It is a layout helper, not a wizard screen, and the
    designer's preview already documents it as the export-time rule.)

    Args:
        layout_spec: A layout_spec dict (as produced by build_layout()) with
            an 'areas' list. Each area must have at minimum 'role' and 'col';
            'colspan' defaults to 1 if absent.
    """
    areas = layout_spec.get("areas", [])
    if not areas:
        return

    # Pick the first 'time' area's column extent as the target
    target_col = None
    target_colspan = None
    for area in areas:
        if str(area.get("role", "")).lower() == "time":
            target_col = int(area.get("col", 0))
            target_colspan = int(area.get("colspan", 1))
            break

    if target_col is None:
        return  # No time panels to normalize against

    # Apply the target to every 'time' and 'labels' area
    for area in areas:
        role = str(area.get("role", "")).lower()
        if role in ("time", "labels"):
            area["col"] = target_col
            area["colspan"] = target_colspan


def validate_plot_inputs(df: pd.DataFrame, selected_columns: List[str]) -> None:
    """
    Validate that selected columns exist in DataFrame.

    (Moved here from the deleted `quickstart.plot_builder` in Pack 7 W1,
    byte-for-byte.)

    Args:
        df: DataFrame containing the data
        selected_columns: List of column names to validate

    Raises:
        ValueError: If any selected column is missing from DataFrame
    """
    missing_columns = [col for col in selected_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Selected columns not found in DataFrame: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )
