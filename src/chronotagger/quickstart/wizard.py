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
        self.selected_columns = None
        self.layout_type = None
        self.plot_config = None

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

        # Phase 2: Start with file loading
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

        # Store loaded DataFrame
        self.df = df

        # Proceed to Phase 3 (stub for now)
        self._show_column_selector()

    def _show_column_selector(self):
        """Show column selection dialog."""
        from chronotagger.quickstart.column_selector import ColumnSelectorDialog

        col_dialog = ColumnSelectorDialog(self.root, self.df)
        selection = col_dialog.run()

        if selection is None:
            # User cancelled
            self._on_cancel()
            return

        # Store selection
        self.selected_columns = selection['columns']
        self.layout_type = selection['layout_type']

        # Proceed to Phase 4 (launch labeler)
        self._launch_labeler()

    def _launch_labeler(self):
        """Launch TimeIntervalLabeler with configured data and plots."""
        from chronotagger import TimeIntervalLabeler
        from chronotagger.quickstart.plot_builder import (
            build_plot_function,
            build_layout_spec,
            validate_plot_inputs
        )

        try:
            # Validate inputs
            validate_plot_inputs(self.df, self.selected_columns)

            # Build plot function and layout
            plot_fn = build_plot_function(self.selected_columns)
            layout_spec = build_layout_spec(self.selected_columns, self.layout_type)

            # Calculate a reasonable default window (10% of data range)
            time_range = self.df.index[-1] - self.df.index[0]
            default_window = time_range * 0.1

            # Create TimeIntervalLabeler
            labeler = TimeIntervalLabeler(
                df=self.df,
                plot_fn=plot_fn,
                layout_spec=layout_spec,
                window=default_window
            )

            # Hide wizard window
            self.root.withdraw()

            # Run labeler (this blocks until labeler window is closed)
            labeler.run()

            # When labeler closes, destroy wizard
            self.root.destroy()

        except Exception as e:
            # Show error and return to wizard
            self.root.deiconify()  # Show wizard again
            messagebox.showerror(
                "Error Launching Labeler",
                f"Failed to launch TimeIntervalLabeler:\n\n{str(e)}\n\n"
                f"Please check your selections and try again.",
                parent=self.root
            )
            # Return to column selector to try again
            self._show_column_selector()

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
