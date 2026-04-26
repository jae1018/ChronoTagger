"""
Column selection dialog for ChronoTagger quick-start wizard.

Provides a GUI for selecting which numeric columns to plot and
choosing the layout type.
"""

import tkinter as tk
from tkinter import messagebox
import tkinter.ttk as ttk
from typing import Optional, List
import pandas as pd


class ColumnSelectorDialog:
    """
    Dialog for selecting columns to plot and layout configuration.

    Features:
    - Display available numeric columns
    - Multi-select checkboxes for column selection
    - Layout type selector (Vertical Stack or Custom Grid)
    - Selection preview
    - Validation (at least one column required)
    """

    def __init__(self, parent, df: pd.DataFrame):
        """
        Initialize column selector dialog.

        Args:
            parent: Parent Tkinter window
            df: Loaded DataFrame with data to plot
        """
        self.parent = parent
        self.df = df
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Select Columns to Plot")
        self.dialog.geometry("500x500")

        # Center dialog on parent
        self._center_on_parent()

        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # State
        self.result = None  # Selection config if successful
        self.column_vars = {}  # Dict of column_name -> BooleanVar
        self.layout_var = tk.StringVar(value="vertical_stack")

        # Get numeric columns
        self.numeric_columns = self._get_numeric_columns()

        # Build UI
        self._build_ui()

        # Set default selections (all columns selected)
        self._select_all_columns()

        # Update preview
        self._update_preview()

    def _center_on_parent(self):
        """Center dialog on parent window."""
        self.dialog.update_idletasks()

        # Get parent position and size
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        # Get dialog size
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()

        # Calculate center position
        x = parent_x + (parent_width // 2) - (dialog_width // 2)
        y = parent_y + (parent_height // 2) - (dialog_height // 2)

        self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")

    def _get_numeric_columns(self) -> List[str]:
        """
        Extract numeric column names from DataFrame.

        Returns:
            List of numeric column names
        """
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        return numeric_cols

    def _build_ui(self):
        """Build dialog UI."""
        # Main container
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill='both', expand=True)

        # Column selection section
        column_frame = ttk.LabelFrame(
            main_frame,
            text=f"Available Columns ({len(self.numeric_columns)} numeric columns found)",
            padding="10"
        )
        column_frame.pack(fill='both', expand=True, pady=(0, 10))

        # Scrollable frame for checkboxes
        canvas = tk.Canvas(column_frame, height=150)
        scrollbar = ttk.Scrollbar(column_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create checkboxes for each numeric column
        for col in self.numeric_columns:
            var = tk.BooleanVar(value=False)
            self.column_vars[col] = var

            cb = ttk.Checkbutton(
                scrollable_frame,
                text=col,
                variable=var,
                command=self._on_column_toggle
            )
            cb.pack(anchor='w', pady=2)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Select/Deselect all buttons
        button_frame = ttk.Frame(column_frame)
        button_frame.pack(fill='x', pady=(5, 0))

        select_all_btn = tk.Button(
            button_frame,
            text="Select All",
            command=self._select_all_columns
        )
        select_all_btn.pack(side='left', padx=(0, 5))

        deselect_all_btn = tk.Button(
            button_frame,
            text="Deselect All",
            command=self._deselect_all_columns
        )
        deselect_all_btn.pack(side='left')

        # Layout selection section
        layout_frame = ttk.LabelFrame(main_frame, text="Layout", padding="10")
        layout_frame.pack(fill='x', pady=(0, 10))

        vertical_radio = ttk.Radiobutton(
            layout_frame,
            text="Vertical Stack (recommended)",
            variable=self.layout_var,
            value="vertical_stack",
            command=self._update_preview
        )
        vertical_radio.pack(anchor='w', pady=2)

        custom_radio = ttk.Radiobutton(
            layout_frame,
            text="Custom Grid (advanced -- design your own panel arrangement)",
            variable=self.layout_var,
            value="custom_grid",
            command=self._update_preview
        )
        custom_radio.pack(anchor='w', pady=2)

        # Preview section
        preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding="10")
        preview_frame.pack(fill='x', pady=(0, 10))

        self.preview_label = ttk.Label(preview_frame, text="", foreground='gray')
        self.preview_label.pack(anchor='w')

        # Buttons
        button_container = ttk.Frame(main_frame)
        button_container.pack(fill='x')

        # Right-aligned buttons
        btn_frame = ttk.Frame(button_container)
        btn_frame.pack(side='right')

        cancel_btn = tk.Button(btn_frame, text="Cancel", command=self._on_cancel)
        cancel_btn.pack(side='left', padx=(0, 5))

        self.continue_btn = tk.Button(
            btn_frame,
            text="Continue",
            command=self._on_continue,
            state='disabled'
        )
        self.continue_btn.pack(side='left')

    def _select_all_columns(self):
        """Select all columns."""
        for var in self.column_vars.values():
            var.set(True)
        self._update_preview()

    def _deselect_all_columns(self):
        """Deselect all columns."""
        for var in self.column_vars.values():
            var.set(False)
        self._update_preview()

    def _on_column_toggle(self):
        """Handle column checkbox toggle."""
        self._update_preview()

    def _update_preview(self):
        """Update preview and validate selection."""
        selected = self._get_selected_columns()
        is_valid, error_msg = self._validate_selection()

        if is_valid:
            layout_name = "Vertical Stack" if self.layout_var.get() == "vertical_stack" else "Custom Grid"
            self.preview_label.config(
                text=f"{len(selected)} column(s) selected: {', '.join(selected)}\nLayout: {layout_name}",
                foreground='black'
            )
            self.continue_btn.config(state='normal')
        else:
            self.preview_label.config(
                text=error_msg,
                foreground='red'
            )
            self.continue_btn.config(state='disabled')

    def _get_selected_columns(self) -> List[str]:
        """
        Get list of selected column names.

        Returns:
            List of selected column names
        """
        selected = []
        for col, var in self.column_vars.items():
            if var.get():
                selected.append(col)
        return selected

    def _validate_selection(self) -> tuple[bool, str]:
        """
        Validate column selection.

        Requirements:
        - At least one column must be selected

        Returns:
            Tuple of (is_valid, error_message)
        """
        selected = self._get_selected_columns()

        if len(selected) == 0:
            return False, "Error: Please select at least one column to plot."

        return True, ""

    def run(self) -> Optional[dict]:
        """
        Show dialog and return selection configuration.

        Returns:
            Dict with 'columns' and 'layout_type' keys if successful,
            None if cancelled
        """
        # Wait for dialog to close
        self.dialog.wait_window()
        return self.result

    def _on_continue(self):
        """Handle continue button click."""
        selected = self._get_selected_columns()
        is_valid, error_msg = self._validate_selection()

        if not is_valid:
            messagebox.showerror(
                "Invalid Selection",
                error_msg,
                parent=self.dialog
            )
            return

        # Success - store result and close
        self.result = {
            'columns': selected,
            'layout_type': self.layout_var.get()
        }
        self.dialog.destroy()

    def _on_cancel(self):
        """Handle cancel button click."""
        self.result = None
        self.dialog.destroy()
