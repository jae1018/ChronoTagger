"""
Quick-Start Wizard for ChronoTagger.

Provides a GUI-based workflow for loading data and configuring plots
without requiring users to write Python code.
"""

import tkinter as tk
from tkinter import messagebox
import sys
from typing import Optional


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
        self.plot_config = None

    def run(self):
        """
        Run the quick-start wizard.

        This is the main entry point called by launcher.py.
        """
        # Create root window
        self.root = tk.Tk()
        self.root.title("ChronoTagger Quick Start")
        self.root.geometry("600x400")

        # Center window on screen
        self._center_window()

        # For Phase 1: Just show welcome message
        self._show_welcome()

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

    def _show_welcome(self):
        """
        Show welcome screen (Phase 1 stub).

        In future phases, this will be replaced with actual wizard UI.
        """
        import tkinter.ttk as ttk

        # Main frame
        main_frame = ttk.Frame(self.root, padding="40")
        main_frame.pack(fill='both', expand=True)

        # Title
        title = ttk.Label(
            main_frame,
            text="ChronoTagger Quick Start",
            font=('Arial', 16, 'bold')
        )
        title.pack(pady=(0, 20))

        # Welcome message
        message = ttk.Label(
            main_frame,
            text=(
                "Welcome to ChronoTagger Quick Start!\n\n"
                "Phase 1: Foundation is complete.\n"
                "The wizard will guide you through loading data\n"
                "and configuring plots in future phases.\n\n"
                "For now, this is a placeholder to verify\n"
                "the entry point works correctly."
            ),
            justify='center',
            font=('Arial', 11)
        )
        message.pack(pady=20)

        # Info about programmatic usage
        info = ttk.Label(
            main_frame,
            text=(
                "To use ChronoTagger programmatically:\n"
                "from chronotagger import TimeIntervalLabeler\n"
                "app = TimeIntervalLabeler(df, plot_fn, labels)\n"
                "app.run()"
            ),
            justify='center',
            font=('Courier', 9),
            foreground='gray'
        )
        info.pack(pady=20)

        # Close button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)

        close_button = ttk.Button(
            button_frame,
            text="Close",
            command=self._on_close
        )
        close_button.pack()

    def _on_close(self):
        """Handle close button click."""
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
