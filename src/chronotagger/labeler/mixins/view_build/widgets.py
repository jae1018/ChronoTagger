"""
Widget utility methods for the labeler view.

Responsibilities:
- Create and manage tooltips
- Update tooltip text dynamically

This mixin provides reusable widget utility methods used throughout the labeler UI.
"""

from __future__ import annotations

import tkinter as tk


class WidgetsMixin:
    """
    Mixin providing widget utility methods.

    This mixin expects the following attributes/methods to be available on self:
    - root: tk.Tk - The main tkinter root window
    - sidebar_toggle_btn: tk.Button - The sidebar toggle button (for tooltip updates)

    Methods provided:
    - _create_tooltip(widget, text) - Create a simple tooltip for a widget
    - _update_tooltip_text(new_text) - Update the sidebar toggle button tooltip
    """

    def _create_tooltip(self, widget, text):
        """
        Create a simple tooltip for a widget.

        Args:
            widget: The widget to add tooltip to
            text: The tooltip text to display
        """
        # Store initial tooltip text
        widget.tooltip_text = text

        def on_enter(event):
            # Use stored text (which may have been updated)
            current_text = getattr(widget, 'tooltip_text', text)

            # Create tooltip window
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")

            # Add tooltip text
            label = tk.Label(
                tooltip,
                text=current_text,
                background="lightyellow",
                relief="solid",
                borderwidth=1,
                font=("TkDefaultFont", 8)
            )
            label.pack()

            # Store reference to tooltip
            widget.tooltip = tooltip

        def on_leave(event):
            # Destroy tooltip
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip

        # Bind events
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def _update_tooltip_text(self, new_text):
        """
        Update the tooltip text for the sidebar toggle button.

        Args:
            new_text: The new tooltip text to display
        """
        # Store the new text so future tooltip displays will use it
        self.sidebar_toggle_btn.tooltip_text = new_text
