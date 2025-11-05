"""
Plot Function Generator for ChronoTagger

This module provides utilities to automatically generate plot functions from
configuration dictionaries. This eliminates the need for users to manually
write plot_fn code for simple plotting scenarios.

The generated plot functions support:
- Time-series plots (line and scatter)
- Cross-plots / scatter plots (any column vs any column)
- Basic styling and labeling

For more complex plotting needs, users should write custom plot functions.

Usage:
    from chronotagger.labeler.utils import build_layout, generate_plot_fn
    
    layout_spec, plot_config = build_layout(df)
    plot_fn = generate_plot_fn(plot_config)

Author: ChronoTagger Team
"""

from __future__ import annotations

from typing import Dict, Callable, Any
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
            ax.clear()  # Clear previous plot
            
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


def generate_plot_code(plot_config: Dict[str, Dict[str, Any]]) -> str:
    """
    Generate Python code for a plot function from configuration.
    
    This is useful for users who want to see what the auto-generated function
    does, or who want to start with generated code and customize it.
    
    Args:
        plot_config: Same format as generate_plot_fn()
    
    Returns:
        Python code as a string that defines a plot_fn function
    
    Example:
        >>> code = generate_plot_code(plot_config)
        >>> print(code)
        >>> # User can copy-paste and modify this code
    """
    lines = [
        "def plot_fn(axs, df, t0, t1):",
        "    \"\"\"Auto-generated plotting function.\"\"\"",
        "",
    ]
    
    for panel_key, config in plot_config.items():
        role = config.get('role', 'time')
        
        lines.append(f"    # Panel: {panel_key}")
        lines.append(f"    ax = axs['{panel_key}']")
        lines.append(f"    ax.clear()")
        lines.append("")
        
        if role == 'time':
            y_col = config.get('y_column', 'COLUMN_NAME')
            style = config.get('style', 'line')
            color = config.get('color', '#1f77b4')
            ylabel = config.get('ylabel', y_col)
            
            if style == 'scatter':
                lines.append(f"    ax.scatter(df.index, df['{y_col}'], s=3, c='{color}', alpha=0.7)")
            else:
                lines.append(f"    ax.plot(df.index, df['{y_col}'], color='{color}', linewidth=1.0)")
            
            lines.append(f"    ax.set_ylabel('{ylabel}')")
            lines.append(f"    ax.grid(alpha=0.3)")
            
        else:  # not-time
            x_col = config.get('x_column', 'X_COLUMN')
            y_col = config.get('y_column', 'Y_COLUMN')
            style = config.get('style', 'scatter')
            color = config.get('color', '#2ca02c')
            xlabel = config.get('xlabel', x_col)
            ylabel = config.get('ylabel', y_col)
            
            if style == 'line':
                lines.append(f"    ax.plot(df['{x_col}'], df['{y_col}'], color='{color}', linewidth=1.0)")
            else:
                lines.append(f"    ax.scatter(df['{x_col}'], df['{y_col}'], s=5, c='{color}', alpha=0.6)")
            
            lines.append(f"    ax.set_xlabel('{xlabel}')")
            lines.append(f"    ax.set_ylabel('{ylabel}')")
            lines.append(f"    ax.grid(alpha=0.3)")
        
        lines.append("")
    
    return "\n".join(lines)


def print_plot_code(plot_config: Dict[str, Dict[str, Any]]) -> None:
    """
    Print the generated plot function code to console.
    
    Convenience function for users who want to see and copy the generated code.
    
    Args:
        plot_config: Same format as generate_plot_fn()
    
    Example:
        >>> layout_spec, plot_config = build_layout(df)
        >>> print_plot_code(plot_config)
        >>> # Copy the printed code and modify as needed
    """
    code = generate_plot_code(plot_config)
    print("=" * 60)
    print("Generated Plot Function Code:")
    print("=" * 60)
    print(code)
    print("=" * 60)
    print("\nCopy this code and modify as needed!")
