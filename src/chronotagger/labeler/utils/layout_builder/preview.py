"""
Preview Mixin for Layout Builder

This module provides functionality for generating and displaying a matplotlib
preview of the current layout configuration. It also includes methods for
generating the layout_spec and plot_config dictionaries.

Classes:
    PreviewMixin: Handles preview generation and output configuration
"""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

if TYPE_CHECKING:
    from .models import PanelConfig


class PreviewMixin:
    """
    Mixin providing preview and configuration generation functionality.

    This mixin provides three key features:
    1. Visual matplotlib preview of the layout
    2. Generation of layout_spec dictionary for TimeIntervalLabeler
    3. Generation of plot_config dictionary for automatic plot function creation

    The mixin expects the parent class to provide:
    - self.panels: List[PanelConfig]
    - self.nrows_var: tk.IntVar
    - self.ncols_var: tk.IntVar
    - self.preview_window: Optional[tk.Toplevel]
    - Color constants (COLOR_TIME, COLOR_NOT_TIME, COLOR_LABELS)
    """

    def _show_preview(self):
        """Show matplotlib preview of the current layout."""
        # Validate that panels exist
        if not self.panels:
            messagebox.showinfo(
                "No Panels",
                "Please create at least one panel before previewing.",
                parent=self
            )
            return

        # Close existing preview window if open
        if self.preview_window is not None and self.preview_window.winfo_exists():
            self.preview_window.destroy()

        # Create new preview window
        self.preview_window = tk.Toplevel(self)
        self.preview_window.title("Layout Preview")
        self.preview_window.geometry("800x600")

        # Generate layout spec from current panels
        layout_spec = self._generate_layout_spec()

        # Create matplotlib figure with GridSpec
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(
            nrows=layout_spec['nrows'],
            ncols=layout_spec['ncols'],
            hspace=layout_spec.get('hspace', 0.15),
            wspace=layout_spec.get('wspace', 0.12),
            figure=fig
        )

        # Pick the first 'time' panel's column extent as the target.
        # Both the Labels strip AND every other 'time' panel are coerced
        # to this target during preview rendering, matching the post-hoc
        # normalization applied on export (see chronotagger.quickstart
        # .plot_builder.normalize_time_columns). 'not-time' panels keep
        # their authored col/colspan.
        target_col = None
        target_colspan = None
        for p in self.panels:
            if p.role == "time":
                target_col = p.col
                target_colspan = p.colspan
                break

        # Create a subplot for each panel
        for panel in self.panels:
            if panel.role in ("time", "labels") and target_col is not None:
                col = target_col
                colspan = target_colspan
            else:
                col = panel.col
                colspan = panel.colspan

            # Use GridSpec slice notation for spanning panels
            ax = fig.add_subplot(gs[
                panel.row:panel.row + panel.rowspan,
                col:col + colspan
            ])

            # Set background color based on role (match grid colors)
            if panel.role == "labels":
                bg_color = self.COLOR_LABELS
            elif panel.role == "time":
                bg_color = self.COLOR_TIME
            else:
                bg_color = self.COLOR_NOT_TIME
            ax.set_facecolor(bg_color)

            # Remove ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])

            # Build info text to display in panel
            if panel.role == "labels":
                # Special rendering for Labels panel
                info_lines = ["⏱️ Labels Strip", "(Auto-managed)"]
            else:
                info_lines = [panel.key]

                # Add variable information
                if panel.role == "time":
                    info_lines.append(f"Y: {panel.y_column}")
                else:  # not-time
                    info_lines.append(f"X: {panel.x_column}")
                    info_lines.append(f"Y: {panel.y_column_2}")

                # Add role
                info_lines.append(f"({panel.role})")

                # Add span info if not 1x1
                if panel.rowspan > 1 or panel.colspan > 1:
                    info_lines.append(f"[{panel.rowspan}×{panel.colspan}]")

            # Display text in center of panel
            info_text = "\n".join(info_lines)
            ax.text(
                0.5, 0.5, info_text,
                ha='center', va='center',
                fontsize=10,
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

        # Embed matplotlib figure in Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.preview_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add close button at bottom
        button_frame = ttk.Frame(self.preview_window)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(
            button_frame,
            text="Close",
            command=self.preview_window.destroy,
            width=12
        ).pack()

    def _generate_layout_spec(self) -> Dict:
        """Generate layout_spec dictionary from current panels."""
        nrows = self.nrows_var.get()
        ncols = self.ncols_var.get()

        areas = []
        for panel in self.panels:
            area = {
                'key': panel.key,
                'row': panel.row,
                'col': panel.col,
                'role': panel.role,
            }
            if panel.rowspan > 1:
                area['rowspan'] = panel.rowspan
            if panel.colspan > 1:
                area['colspan'] = panel.colspan
            areas.append(area)

        return {
            'nrows': nrows,
            'ncols': ncols,
            'hspace': 0.15,
            'wspace': 0.12,
            'areas': areas,
        }

    def _generate_plot_config(self) -> Dict:
        """Generate plot_config dictionary from current panels."""
        config = {}
        for panel in self.panels:
            panel_cfg = {
                'role': panel.role,
            }
            if panel.role == "time":
                panel_cfg['y_column'] = panel.y_column
            else:
                panel_cfg['x_column'] = panel.x_column
                panel_cfg['y_column'] = panel.y_column_2

            config[panel.key] = panel_cfg

        return config
