"""
Layout Builder Entry Point

This module provides the main entry point function for launching the interactive
layout builder dialog. It handles the creation and cleanup of the Tkinter root
window and returns the generated layout configuration.

Functions:
    build_layout: Launch the layout builder dialog and return configurations
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict
import tkinter as tk
import pandas as pd

from .dialog import LayoutBuilderDialog


def build_layout(df: pd.DataFrame, parent: Optional[tk.Tk] = None) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Launch interactive layout builder dialog.

    This is the main entry point for users who want to visually design their
    plot layout instead of manually writing layout_spec dictionaries.

    Args:
        df: DataFrame containing the data to be plotted. Used to extract
            available columns for assignment.
        parent: Optional parent Tk window. If None, creates temporary root.

    Returns:
        Tuple of (layout_spec, plot_config):
        - layout_spec: Dictionary suitable for TimeIntervalLabeler constructor
        - plot_config: Dictionary suitable for generate_plot_fn()
        - Both are None if user cancels

    Example:
        >>> import pandas as pd
        >>> from chronotagger.labeler.utils import build_layout, generate_plot_fn
        >>> from chronotagger.labeler import TimeIntervalLabeler
        >>>
        >>> # Load data
        >>> df = pd.read_csv('data.csv', index_col=0, parse_dates=True)
        >>>
        >>> # Build layout interactively
        >>> layout_spec, plot_config = build_layout(df)
        >>>
        >>> if layout_spec is not None:
        >>>     # Generate plot function from config
        >>>     plot_fn = generate_plot_fn(plot_config)
        >>>
        >>>     # Create labeler
        >>>     app = TimeIntervalLabeler(
        >>>         df=df,
        >>>         plot_fn=plot_fn,
        >>>         layout_spec=layout_spec
        >>>     )
        >>>     app.run()
    """
    # Create temporary root if needed
    owns_root = False
    if parent is None:
        parent = tk.Tk()
        parent.withdraw()  # Hide root window
        owns_root = True

    try:
        # Launch dialog
        dialog = LayoutBuilderDialog(parent, df)

        # Wait for dialog to close
        parent.wait_window(dialog)

        # Get results
        layout_spec = dialog.result_layout_spec
        plot_config = dialog.result_plot_config

        return layout_spec, plot_config

    finally:
        # Clean up temporary root
        if owns_root:
            parent.destroy()
