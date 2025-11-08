"""
Main window construction mixin.

Responsibilities:
- Build the main tkinter window
- Orchestrate construction of top controls, plot, and sidebar
- Set up global key bindings

This mixin provides the top-level _build_gui method that creates the entire UI.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class WindowMixin:
    """
    Mixin providing main window construction.

    This mixin expects the following attributes/methods to be available on self:
    - _on_closing() - Callback for window close event
    - _build_top_controls(parent) - Method to build top controls
    - _build_sidebar(parent) - Method to build sidebar
    - _build_plot(parent) - Method to build plot area
    - _on_key_press(event) - Callback for key press events
    - _open_help_dialog(event) - Callback for opening help dialog
    - _toggle_sidebar() - Callback for toggling sidebar visibility

    Attributes created:
    - root: tk.Tk - The main tkinter root window
    - sidebar_frame: ttk.Frame - The sidebar frame
    - plot_frame: ttk.Frame - The plot frame

    Methods provided:
    - _build_gui() - Build the complete GUI (window, controls, plot, sidebar)
    """

    def _build_gui(self) -> None:
        self.root = tk.Tk()
        self.root.title("ChronoTagger - Time Interval Labeler")
        self.root.geometry("1600x900")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self._build_top_controls(top)

        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        # need to create sidebar BEFORE creating full plot, otherwise it
        # doesn't show up
        self.sidebar_frame = ttk.Frame(main, width=320)
        self.sidebar_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        self.sidebar_frame.pack_propagate(False)
        self._build_sidebar(self.sidebar_frame)

        self.plot_frame = ttk.Frame(main)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_plot(self.plot_frame)

        self.root.bind("<Key>", self._on_key_press)
        # F1 opens Help
        self.root.bind("<F1>", self._open_help_dialog)
        # F9 toggles sidebar visibility
        self.root.bind("<F9>", lambda e: self._toggle_sidebar())
