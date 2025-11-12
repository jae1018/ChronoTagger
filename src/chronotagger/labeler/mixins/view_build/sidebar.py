"""
Sidebar construction and management mixin.

Responsibilities:
- Build the sidebar with scrollable content
- Create sidebar sections (intervals list, stats, options, status)
- Handle sidebar scrolling behavior
- Toggle sidebar visibility
- Manage sidebar layout updates

This mixin provides all sidebar-related functionality including construction,
scrolling, and visibility toggling.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class SidebarMixin:
    """
    Mixin providing sidebar construction and management.

    This mixin expects the following attributes/methods to be available on self:
    - classes: list[str] - List of label classes
    - sidebar_frame: ttk.Frame - The parent frame for the sidebar
    - plot_frame: ttk.Frame - The plot frame (for layout management)
    - root: tk.Tk - The main tkinter root window
    - sidebar_toggle_btn: ttk.Button - The sidebar toggle button
    - _on_interval_tree_select(event) - Callback for interval tree selection
    - _update_tooltip_text(text) - Method to update tooltip text

    Attributes created:
    - sidebar_collapsed: bool - Whether the sidebar is currently collapsed
    - sidebar_expanded_width: int - Width of the sidebar when expanded
    - sidebar_canvas: tk.Canvas - Canvas for scrolling sidebar content
    - sidebar_interior: ttk.Frame - Interior frame holding sidebar content
    - sidebar_canvas_window: int - Canvas window ID for the interior frame
    - intervals_tree: ttk.Treeview - Treeview for displaying labeled intervals
    - stats_text: tk.Text - Text widget for statistics display
    - snap_var: tk.BooleanVar - BooleanVar for snap-to-samples option
    - overlays_var: tk.BooleanVar - BooleanVar for interval overlays option
    - status_var: tk.StringVar - StringVar for status messages

    Methods provided:
    - _build_sidebar(parent) - Build the sidebar with scrollable content
    - _build_sidebar_sections(parent) - Build all sidebar sections
    - _on_sidebar_configure(event) - Update scroll region on size changes
    - _update_sidebar_scroll_region() - Update the canvas scroll region
    - _bind_sidebar_mousewheel() - Bind mouse wheel for scrolling
    - _toggle_sidebar() - Toggle sidebar visibility
    - _refresh_sidebar_layout() - Force complete refresh of sidebar layout
    """

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        """
        Build the right sidebar with scrollable content and collapse functionality.

        Uses Canvas + Scrollbar pattern for smooth scrolling when content
        exceeds available vertical space. All sidebar sections are placed
        inside a scrollable interior frame.
        """
        # Initialize collapse state
        self.sidebar_collapsed = False
        self.sidebar_expanded_width = 320
        # No collapsed width needed since we completely hide/show

        # Create canvas for scrolling
        self.sidebar_canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0)

        # Create scrollbar
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.sidebar_canvas.yview)

        # Create interior frame (holds all content)
        self.sidebar_interior = ttk.Frame(self.sidebar_canvas)

        # Add interior frame to canvas
        self.sidebar_canvas_window = self.sidebar_canvas.create_window(
            (0, 0), window=self.sidebar_interior, anchor="nw"
        )

        # Configure canvas scrolling
        self.sidebar_canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Build all sections INSIDE the interior frame
        self._build_sidebar_sections(self.sidebar_interior)

        # Bind events for scroll region updates
        self.sidebar_interior.bind("<Configure>", self._on_sidebar_configure)

        # Bind mouse wheel for smooth scrolling
        self._bind_sidebar_mousewheel()

    def _build_sidebar_sections(self, parent: ttk.Frame) -> None:
        """
        Build all sidebar sections (intervals, stats, options, etc.).

        This is separated from _build_sidebar() to keep the scrolling
        setup clean and modular.

        Args:
            parent: The interior frame inside the canvas
        """
        # Intervals list (no toggle button here anymore - moved to top bar)
        frame = ttk.LabelFrame(parent, text="Labeled Intervals", padding=5)
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Treeview for intervals
        columns = ("Start", "End", "Label", "Duration")
        self.intervals_tree = ttk.Treeview(
            frame, columns=columns, show="tree headings", height=15
        )
        self.intervals_tree.heading("#0", text="#")
        for col in columns:
            self.intervals_tree.heading(col, text=col)
        self.intervals_tree.column("#0", width=30)
        self.intervals_tree.column("Start", width=80)
        self.intervals_tree.column("End", width=80)
        self.intervals_tree.column("Label", width=90)
        self.intervals_tree.column("Duration", width=70)

        sb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.intervals_tree.yview)
        self.intervals_tree.configure(yscrollcommand=sb.set)
        self.intervals_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.intervals_tree.bind("<<TreeviewSelect>>", self._on_interval_tree_select)

        # Stats
        stats = ttk.LabelFrame(parent, text="Statistics", padding=5)
        stats.pack(fill=tk.X, pady=5)
        self.stats_text = tk.Text(stats, height=8, width=30, state="disabled")
        self.stats_text.pack(fill=tk.BOTH, expand=True)

        # Options
        opts = ttk.LabelFrame(parent, text="Options", padding=5)
        opts.pack(fill=tk.X, pady=5)
        self.snap_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Snap to samples", variable=self.snap_var).pack(anchor=tk.W)

        # Overlay toggle
        self.overlays_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opts, text="Show interval overlays on panels", variable=self.overlays_var
        ).pack(anchor=tk.W)

        # Point highlighting toggle (performance optimization)
        self.highlight_points_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opts, text="Highlight Points", variable=self.highlight_points_var,
            command=self._on_highlight_points_toggle
        ).pack(anchor=tk.W)

        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.BOTTOM, fill=tk.X
        )

    # ========== Sidebar Scrolling Helper Methods ==========

    def _on_sidebar_configure(self, event=None) -> None:
        """
        Update scroll region when sidebar interior frame size changes.

        This is called automatically whenever the interior frame is resized
        (e.g., when intervals are added/deleted, window is resized, etc.).
        """
        self._update_sidebar_scroll_region()

    def _update_sidebar_scroll_region(self) -> None:
        """
        Update the canvas scroll region to match interior frame size.

        Call this manually if you programmatically change sidebar content
        and need to update scrolling immediately.
        """
        if hasattr(self, 'sidebar_canvas') and hasattr(self, 'sidebar_interior'):
            # Update scroll region to encompass all interior content
            self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

            # Update canvas window width to match canvas width (prevents horizontal scrolling)
            canvas_width = self.sidebar_canvas.winfo_width()
            if canvas_width > 1:  # Only update if canvas has been rendered
                self.sidebar_canvas.itemconfig(self.sidebar_canvas_window, width=canvas_width)

    def _bind_sidebar_mousewheel(self) -> None:
        """
        Bind mouse wheel to scroll the sidebar canvas.

        Supports Windows, Mac, and Linux scroll events.
        Only scrolls when mouse is over the sidebar.
        """
        def on_mouse_wheel(event):
            """Handle mouse wheel scroll events."""
            # Determine scroll direction (cross-platform)
            if event.num == 5 or event.delta < 0:  # Scroll down
                self.sidebar_canvas.yview_scroll(1, "units")
            elif event.num == 4 or event.delta > 0:  # Scroll up
                self.sidebar_canvas.yview_scroll(-1, "units")

        def on_enter(event):
            """Enable scrolling when mouse enters sidebar."""
            # Bind mouse wheel events (cross-platform)
            self.sidebar_canvas.bind_all("<MouseWheel>", on_mouse_wheel)  # Windows/Mac
            self.sidebar_canvas.bind_all("<Button-4>", on_mouse_wheel)    # Linux scroll up
            self.sidebar_canvas.bind_all("<Button-5>", on_mouse_wheel)    # Linux scroll down

        def on_leave(event):
            """Disable scrolling when mouse leaves sidebar."""
            # Unbind mouse wheel events to avoid interfering with plot zooming
            self.sidebar_canvas.unbind_all("<MouseWheel>")
            self.sidebar_canvas.unbind_all("<Button-4>")
            self.sidebar_canvas.unbind_all("<Button-5>")

        # Bind enter/leave events to canvas and interior frame
        self.sidebar_canvas.bind("<Enter>", on_enter)
        self.sidebar_canvas.bind("<Leave>", on_leave)
        self.sidebar_interior.bind("<Enter>", on_enter)
        self.sidebar_interior.bind("<Leave>", on_leave)

    def _toggle_sidebar(self) -> None:
        """
        Toggle the sidebar between expanded and collapsed states.

        The toggle button is now in the top bar and always visible.
        When collapsed, the entire sidebar is hidden.
        """
        if self.sidebar_collapsed:
            # Expand sidebar
            self.sidebar_collapsed = False

            # Update button appearance
            self.sidebar_toggle_btn.configure(text="Hide Panel ▶")

            # To ensure proper layout, temporarily unpack plot frame
            # then repack sidebar and plot frame in correct order
            self.plot_frame.pack_forget()

            # Pack sidebar first (right side)
            self.sidebar_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
            self.sidebar_frame.pack_propagate(False)

            # Then pack plot frame (left side, fills remaining space)
            self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Restore normal width
            self.sidebar_frame.configure(width=self.sidebar_expanded_width)

            # Force immediate geometry update
            self.sidebar_frame.update_idletasks()
            self.plot_frame.update_idletasks()

            # Force layout updates and refresh
            self.root.after_idle(self._refresh_sidebar_layout)

            # Update tooltip
            self._update_tooltip_text("Hide sidebar (F9)")

        else:
            # Collapse sidebar
            self.sidebar_collapsed = True

            # Update button appearance
            self.sidebar_toggle_btn.configure(text="◀ Show Panel")

            # Hide the entire sidebar frame
            self.sidebar_frame.pack_forget()

            # Update tooltip
            self._update_tooltip_text("Show sidebar (F9)")

    def _refresh_sidebar_layout(self):
        """
        Force a complete refresh of the sidebar layout.

        This is called after expanding the sidebar to ensure all widgets
        properly recalculate their sizes and positions.
        """
        # Only refresh if sidebar is expanded and exists
        if not self.sidebar_collapsed and hasattr(self, 'sidebar_canvas') and hasattr(self, 'sidebar_interior'):
            # Force geometry updates
            self.sidebar_frame.update_idletasks()
            self.sidebar_interior.update_idletasks()
            self.sidebar_canvas.update_idletasks()

            # Reconfigure canvas window width
            canvas_width = self.sidebar_canvas.winfo_width()
            if canvas_width > 1:
                self.sidebar_canvas.itemconfig(self.sidebar_canvas_window, width=canvas_width)

            # Update scroll region
            self._update_sidebar_scroll_region()

            # Final geometry update
            self.sidebar_canvas.update_idletasks()
            self.sidebar_interior.update_idletasks()

    def _on_highlight_points_toggle(self) -> None:
        """
        Handle Highlight Points checkbox toggle.

        Updates the enable_point_highlighting state and refreshes the plot
        to immediately show or hide point highlights.
        """
        # Update state from checkbox
        if hasattr(self, 'highlight_points_var'):
            self.enable_point_highlighting = self.highlight_points_var.get()

        # Sync state to all panes (for multi-pane mode)
        if hasattr(self, 'panes'):
            for pane in self.panes:
                pane.enable_point_highlighting = self.enable_point_highlighting

        # If disabled, clear existing highlights immediately
        if not self.enable_point_highlighting:
            if hasattr(self, '_clear_selected_point_highlights'):
                self._clear_selected_point_highlights()
            if hasattr(self, '_clear_selected_interval_highlights'):
                self._clear_selected_interval_highlights()

            # Redraw canvas to remove highlights
            canvas = self.active_pane.canvas if hasattr(self, 'active_pane') else getattr(self, 'canvas', None)
            if canvas is not None:
                canvas.draw_idle()
        else:
            # If enabled, show highlights for current selection (if any)
            # First check if there's an active selection/preview
            has_preview = (
                getattr(self, 'current_selection', None) is not None or
                bool(getattr(self, 'current_spans', None))
            )

            if has_preview:
                # Show highlights for active preview
                if hasattr(self, '_show_selected_point_highlights'):
                    self._show_selected_point_highlights(redraw=True)

            # Check if there's a selected interval
            if hasattr(self, 'selected_interval') and self.selected_interval is not None:
                if hasattr(self, '_show_selected_interval_highlights'):
                    self._show_selected_interval_highlights()

            # Redraw canvas to show highlights (especially for interval highlights)
            canvas = self.active_pane.canvas if hasattr(self, 'active_pane') else getattr(self, 'canvas', None)
            if canvas is not None:
                canvas.draw_idle()
