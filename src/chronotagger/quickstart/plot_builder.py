"""
Plot function and layout builder for ChronoTagger quick-start wizard.

Dynamically generates plot functions and layout specifications from
user's column selections.
"""

from typing import List, Dict, Any, Callable
import pandas as pd


def build_plot_function(selected_columns: List[str]) -> Callable:
    """
    Build a plot function from selected columns.

    Creates a plot function that plots each selected column in its
    own panel, compatible with TimeIntervalLabeler's plot_fn signature.

    Args:
        selected_columns: List of column names to plot

    Returns:
        Plot function with signature: fn(axs, df, t0, t1)
        where axs is a dict of axis keys to matplotlib axes
    """
    def plot_fn(axs, df, t0, t1):
        """
        Plot selected columns in their respective panels.

        Args:
            axs: Dict mapping panel keys to matplotlib axes
            df: DataFrame with data to plot
            t0: Start time (not used, included for compatibility)
            t1: End time (not used, included for compatibility)
        """
        # Plot each selected column in its corresponding panel
        for i, col in enumerate(selected_columns):
            panel_key = f'panel{i}'

            if panel_key in axs:
                ax = axs[panel_key]

                # Plot the data
                ax.plot(df.index, df[col], label=col, linewidth=1.0)

                # Set y-axis label
                ax.set_ylabel(col)

                # Add legend
                ax.legend(loc='upper right', fontsize=8)

                # Grid for better readability
                ax.grid(True, alpha=0.3)

    return plot_fn


def build_layout_spec(selected_columns: List[str], layout_type: str) -> Dict[str, Any]:
    """
    Build layout specification for TimeIntervalLabeler.

    Creates a layout spec dict that defines how panels are arranged
    in the TimeIntervalLabeler grid layout.

    Args:
        selected_columns: List of column names to plot
        layout_type: 'vertical_stack' or 'custom_grid'

    Returns:
        Layout spec dict with keys:
        - nrows: Number of rows in grid
        - ncols: Number of columns in grid
        - areas: List of area dicts defining panel placement

    Raises:
        NotImplementedError: If layout_type is not 'vertical_stack'
    """
    if layout_type == 'vertical_stack':
        # Create vertical stack layout
        # One row per selected column, plus one row for labels panel
        nrows = len(selected_columns) + 1
        ncols = 1
        areas = []

        # Add data panels (one per selected column)
        for i, col in enumerate(selected_columns):
            areas.append({
                'key': f'panel{i}',
                'row': i,
                'col': 0,
                'rowspan': 1,
                'colspan': 1,
                'role': 'time'
            })

        # Add labels panel at bottom
        areas.append({
            'key': 'labels',
            'row': len(selected_columns),
            'col': 0,
            'rowspan': 1,
            'colspan': 1,
            'role': 'labels'
        })

        return {
            'nrows': nrows,
            'ncols': ncols,
            'areas': areas
        }
    else:
        raise NotImplementedError(
            f"Layout type '{layout_type}' not yet implemented. "
            f"Only 'vertical_stack' is currently supported."
        )


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
