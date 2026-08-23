"""
Quick-Start Wizard for ChronoTagger.

Provides a GUI-based workflow for loading data and configuring plots
without requiring users to write Python code.
"""

import tkinter as tk
from tkinter import messagebox
import sys
from typing import Optional
import pandas as pd


class QuickStartWizard:
    """
    Main wizard orchestrator for ChronoTagger quick-start.

    Guides users through:
    1. Loading data file
    2. Selecting columns to plot
    3. Choosing layout
    4. Launching TimeIntervalLabeler
    """

    def __init__(self):
        """Initialize wizard state."""
        self.root: Optional[tk.Tk] = None
        self.df = None
        # Configuration produced by the tab planner -- a list of dicts,
        # one per labeler tab. Each entry has 'title', 'columns',
        # 'layout_type', and (for custom_grid) 'layout_spec' +
        # 'plot_config'. See chronotagger.quickstart.tab_planner.
        self.tabs_config: Optional[list] = None

    def run(self):
        """
        Run the quick-start wizard.

        This is the main entry point called by launcher.py.
        """
        # Create root window
        self.root = tk.Tk()
        self.root.title("ChronoTagger Quick Start")
        self.root.geometry("700x600")  # Larger for file dialog

        # Center window on screen
        self._center_window()

        # Start the wizard flow
        self._show_file_loader()

        # Start Tkinter main loop
        self.root.mainloop()

    def _center_window(self):
        """Center the window on screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _show_file_loader(self):
        """Show file loading dialog."""
        from chronotagger.quickstart.file_loader import FileLoaderDialog

        file_dialog = FileLoaderDialog(self.root)
        df = file_dialog.run()

        if df is None:
            # User cancelled
            self._on_cancel()
            return

        # Store loaded DataFrame, and remember where it came from so
        # the labeler's autosave metadata / recovery dialog can name it
        # (the path was previously known here and thrown away).
        self.df = df
        self.source_name = getattr(file_dialog, 'current_file', None)

        # Proceed to tab planner
        self._show_tab_planner()

    def _show_tab_planner(self):
        """Show the tab planner dialog."""
        from chronotagger.quickstart.tab_planner import TabPlannerDialog

        planner = TabPlannerDialog(self.root, self.df)
        result = planner.run()

        if result is None:
            # User cancelled
            self._on_cancel()
            return

        self.tabs_config = result["tabs"]
        self._launch_labeler()

    def _build_tab_plot(self, tab: dict):
        """
        Build (plot_fn, layout_spec) for a single tab config dict.

        For 'vertical_stack' tabs, generates the plot function and layout
        spec from the column selection on the fly. For 'custom_grid' tabs,
        the user pre-designed the layout in the planner -- we reuse the
        stored layout_spec / plot_config and just normalize.
        """
        from chronotagger.quickstart.plot_builder import (
            build_layout_spec,
            build_plot_function,
            normalize_time_columns,
            validate_plot_inputs,
        )

        columns = tab["columns"]
        layout_type = tab["layout_type"]

        if layout_type == "vertical_stack":
            validate_plot_inputs(self.df, columns)
            plot_fn = build_plot_function(columns)
            layout_spec = build_layout_spec(columns, layout_type)
        elif layout_type == "custom_grid":
            from chronotagger.labeler.utils.plot_generator import (
                generate_plot_fn,
            )
            layout_spec = tab["layout_spec"]
            plot_config = tab["plot_config"]
            normalize_time_columns(layout_spec)
            plot_fn = generate_plot_fn(plot_config)
        else:
            raise ValueError(
                f"Unknown layout type: {layout_type!r}. "
                f"Expected 'vertical_stack' or 'custom_grid'."
            )

        return plot_fn, layout_spec

    def _launch_labeler(self):
        """Launch TimeIntervalLabeler with the configured tabs."""
        from chronotagger import TimeIntervalLabeler

        try:
            # Build (plot_fn, layout_spec) for each tab
            pane_configs = []
            for tab in self.tabs_config:
                plot_fn, layout_spec = self._build_tab_plot(tab)
                pane_configs.append({
                    "title": tab["title"],
                    "plot_fn": plot_fn,
                    "layout_spec": layout_spec,
                })

            # Calculate a reasonable default window (10% of data range)
            time_range = self.df.index[-1] - self.df.index[0]
            default_window = time_range * 0.1

            # Pass parent=self.root so the labeler mounts itself as a
            # tk.Toplevel under the wizard's Tk root, instead of creating a
            # second tk.Tk() (which would land tk.StringVar/IntVar/BooleanVar
            # in a different Tcl interpreter and silently break textvariable
            # links throughout the labeler).
            if len(pane_configs) == 1:
                # Single-pane API (preserves the historical surface for
                # users who only configured one tab)
                only = pane_configs[0]
                labeler = TimeIntervalLabeler(
                    df=self.df,
                    plot_fn=only["plot_fn"],
                    layout_spec=only["layout_spec"],
                    window=default_window,
                    source_name=getattr(self, 'source_name', None),
                    parent=self.root,
                )
            else:
                # Multi-pane API
                labeler = TimeIntervalLabeler(
                    df=self.df,
                    panes=pane_configs,
                    window=default_window,
                    source_name=getattr(self, 'source_name', None),
                    parent=self.root,
                )

            # Hide wizard window while labeler is up; the labeler's
            # Toplevel runs under the wizard's existing mainloop and
            # blocks via wait_window() until the user closes it.
            self.root.withdraw()
            labeler.run()

            # Labeler closed -- tear down the wizard root to exit
            # mainloop and return from QuickStartWizard.run().
            self.root.destroy()

        except Exception as e:
            # Show error and return to the tab planner. The dialog is
            # transient; the traceback is not (Pack 4 R6c).
            import logging
            logging.getLogger("chronotagger.quickstart.wizard").exception(
                "labeler launch failed")
            self.root.deiconify()
            messagebox.showerror(
                "Error Launching Labeler",
                f"Failed to launch TimeIntervalLabeler:\n\n{str(e)}\n\n"
                f"Please check your tab configurations and try again.",
                parent=self.root,
            )
            self._show_tab_planner()

    def _on_cancel(self):
        """Handle cancellation."""
        result = messagebox.askyesno(
            "Exit Wizard",
            "Exit the wizard?",
            parent=self.root
        )
        if result:
            self.root.destroy()
            sys.exit(0)


def run():
    """
    Run the quick-start wizard.

    This function is called by launcher.py.
    """
    wizard = QuickStartWizard()
    wizard.run()


if __name__ == "__main__":
    # Allow running wizard directly for testing
    run()
