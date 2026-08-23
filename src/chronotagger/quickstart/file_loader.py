"""
File loading dialog for ChronoTagger quick-start wizard.

Provides a GUI for selecting and loading CSV/Parquet files with
automatic time column detection and data validation.
"""

import logging
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk
from typing import Optional
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class FileLoaderDialog:
    """
    Dialog for loading data files (CSV, Parquet).

    Features:
    - File browser for selecting CSV/Parquet files
    - Auto-detection of time column
    - Manual time column selection
    - Data preview (first 10 rows)
    - Validation (DatetimeIndex, numeric columns, sorted)
    - Error handling with helpful messages
    """

    def __init__(self, parent):
        """
        Initialize file loader dialog.

        Args:
            parent: Parent Tkinter window
        """
        self.parent = parent
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Load Data File")
        self.dialog.geometry("700x600")

        # Center dialog on parent
        self._center_on_parent()

        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # State
        self.result = None  # DataFrame if successful
        self.current_file = None
        self.loaded_df = None
        self.time_column_var = tk.StringVar(value="Auto-detect")

        # Build UI
        self._build_ui()

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

    def _build_ui(self):
        """Build dialog UI."""
        # Main container
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill='both', expand=True)

        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.pack(fill='x', pady=(0, 10))

        # File path entry and browse button
        file_input_frame = ttk.Frame(file_frame)
        file_input_frame.pack(fill='x')

        ttk.Label(file_input_frame, text="File:").pack(side='left', padx=(0, 5))

        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_input_frame, textvariable=self.file_path_var, state='readonly')
        file_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))

        browse_btn = tk.Button(file_input_frame, text="Browse...", command=self._browse_file)
        browse_btn.pack(side='left')

        # Time column selection
        time_col_frame = ttk.Frame(file_frame)
        time_col_frame.pack(fill='x', pady=(10, 0))

        ttk.Label(time_col_frame, text="Time Column:").pack(side='left', padx=(0, 5))

        self.time_col_combo = ttk.Combobox(
            time_col_frame,
            textvariable=self.time_column_var,
            state='readonly',
            width=20
        )
        self.time_col_combo.pack(side='left')
        self.time_col_combo.bind('<<ComboboxSelected>>', self._on_time_column_changed)

        # Data preview section
        preview_frame = ttk.LabelFrame(main_frame, text="Data Preview", padding="10")
        preview_frame.pack(fill='both', expand=True, pady=(0, 10))

        # Treeview with scrollbars
        tree_frame = ttk.Frame(preview_frame)
        tree_frame.pack(fill='both', expand=True)

        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        vsb.pack(side='right', fill='y')

        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        hsb.pack(side='bottom', fill='x')

        # Configure Treeview style for better visibility on macOS
        style = ttk.Style()
        style.configure("Treeview.Heading", background="#e0e0e0", foreground="black", relief="raised")

        # Treeview
        self.tree = ttk.Treeview(
            tree_frame,
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
            height=10
        )
        self.tree.pack(fill='both', expand=True)

        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)

        # Status section
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill='x', pady=(0, 10))

        self.status_label = ttk.Label(status_frame, text="No file loaded", foreground='gray')
        self.status_label.pack(side='left')

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x')

        # Right-aligned buttons
        btn_container = ttk.Frame(button_frame)
        btn_container.pack(side='right')

        cancel_btn = tk.Button(btn_container, text="Cancel", command=self._on_cancel)
        cancel_btn.pack(side='left', padx=(0, 5))

        self.continue_btn = tk.Button(btn_container, text="Continue", command=self._on_continue, state='disabled')
        self.continue_btn.pack(side='left')

    def run(self) -> Optional[pd.DataFrame]:
        """
        Show dialog and return loaded DataFrame.

        Returns:
            DataFrame if successful, None if cancelled
        """
        # Wait for dialog to close
        self.dialog.wait_window()
        return self.result

    def _browse_file(self):
        """Open file dialog for selecting CSV or Parquet file."""
        file_path = filedialog.askopenfilename(
            parent=self.dialog,
            title="Select Data File",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Parquet files", "*.parquet"),
                ("All supported files", "*.csv *.parquet"),
            ]
        )

        if file_path:
            self._load_and_preview(file_path)

    def _load_and_preview(self, file_path: str):
        """Load file and show preview."""
        try:
            # Load file
            df = self._load_file(file_path)

            # Store loaded data
            self.current_file = file_path
            self.loaded_df = df
            self.file_path_var.set(file_path)

            # Auto-detect time column
            time_col = self._auto_detect_time_column(df)

            # Update time column dropdown
            self._update_time_column_options(df, time_col)

            # Show preview
            self._show_preview(df)

            # Update status
            self.status_label.config(
                text=f"Rows: {len(df)}, Columns: {len(df.columns)}",
                foreground='black'
            )

            # Enable continue button if data is valid
            df_with_time = self._apply_time_column(df, time_col) if time_col else df
            is_valid, error_msg = self._validate_data(df_with_time)
            self.continue_btn.config(state='normal' if is_valid else 'disabled')

        except Exception as e:
            self._handle_load_error(e)

    def _load_file(self, file_path: str) -> pd.DataFrame:
        """
        Load file based on extension.

        Args:
            file_path: Path to CSV or Parquet file

        Returns:
            Loaded DataFrame

        Raises:
            ValueError: If file type not supported
            Exception: If file cannot be loaded
        """
        path = Path(file_path)

        if path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif path.suffix.lower() == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    def _auto_detect_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Auto-detect time column.

        Checks:
        1. If DatetimeIndex already exists
        2. Common column names (time, timestamp, datetime, date)
        3. Columns with datetime dtype

        Args:
            df: DataFrame to analyze

        Returns:
            Name of detected time column, or None
        """
        # Check if already has DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            return df.index.name if df.index.name else "index"

        # Check common column names
        common_names = ['time', 'timestamp', 'datetime', 'date', 'dt']
        for name in common_names:
            if name in df.columns:
                # Verify it can be converted to datetime
                try:
                    pd.to_datetime(df[name])
                    return name
                except Exception:
                    continue

        # Check for datetime dtype columns (guarded: this probe was the
        # ONLY unprotected statement in the method -- Pack 4 G1 3.6)
        for col in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    return col
            except Exception:
                continue

        # Last resort: a column whose VALUES coerce to datetimes -- but
        # NEVER a numeric column (R8 as amended). Any numeric column
        # coerces (read as ns-since-1970), so column ORDER used to pick a
        # magnetometer trace as the time axis with validation passing
        # (A2/A3, execute-proven) -- and monotonicity is no gate: ramps,
        # counters and L-shells are monotonic (verifier-proven on the
        # gather's own linear-ramp Bx_nT). String/object columns still
        # coerce, even unsorted: the downstream validator owns the
        # "must be sorted" message, and refusing them here handed the
        # win to a numeric column (verifier B4).
        skipped_numeric = 0
        for col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    skipped_numeric += 1
                    continue
                # Try to parse first few values
                pd.to_datetime(df[col].head())
                return col
            except Exception:
                continue

        if skipped_numeric:
            logger.warning(
                "no time column detected; %d numeric column(s) were not "
                "considered (numeric values are never coerced to "
                "timestamps)", skipped_numeric)
        return None

    def _update_time_column_options(self, df: pd.DataFrame, auto_detected: Optional[str]):
        """Update time column dropdown options."""
        options = ["Auto-detect"]

        # Add all columns as options
        options.extend(df.columns.tolist())

        # If index has a name and is datetime, add it
        if isinstance(df.index, pd.DatetimeIndex) and df.index.name:
            options.insert(1, f"index ({df.index.name})")

        self.time_col_combo['values'] = options

        # Set default selection
        if auto_detected:
            if auto_detected == "index" and df.index.name:
                self.time_column_var.set(f"index ({df.index.name})")
            elif auto_detected in df.columns:
                self.time_column_var.set(auto_detected)
            else:
                self.time_column_var.set("Auto-detect")
        else:
            self.time_column_var.set("Auto-detect")

    def _on_time_column_changed(self, event=None):
        """Handle time column selection change."""
        if self.loaded_df is None:
            return

        selected = self.time_column_var.get()

        if selected == "Auto-detect":
            time_col = self._auto_detect_time_column(self.loaded_df)
        else:
            time_col = selected.replace("index (", "").replace(")", "") if "index" in selected else selected

        # Validate with new time column
        df_with_time = self._apply_time_column(self.loaded_df, time_col) if time_col else self.loaded_df
        is_valid, error_msg = self._validate_data(df_with_time)

        # Update continue button
        self.continue_btn.config(state='normal' if is_valid else 'disabled')

        # Update status
        if not is_valid:
            self.status_label.config(text=error_msg, foreground='red')
        else:
            self.status_label.config(
                text=f"Rows: {len(df_with_time)}, Columns: {len(df_with_time.columns)}",
                foreground='black'
            )

    def _apply_time_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Apply time column as index.

        Args:
            df: DataFrame
            column: Column name to use as time index

        Returns:
            DataFrame with time column as DatetimeIndex
        """
        df_copy = df.copy()

        # If column is "index", just ensure it's datetime
        if column == "index" or column.startswith("index ("):
            if not isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy.index = pd.to_datetime(df_copy.index)
            return df_copy

        # Convert column to datetime and set as index
        if column in df_copy.columns:
            df_copy[column] = pd.to_datetime(df_copy[column])
            df_copy = df_copy.set_index(column)

        return df_copy

    def _show_preview(self, df: pd.DataFrame):
        """
        Show first 10 rows in Treeview.

        Args:
            df: DataFrame to preview
        """
        # Clear existing data
        self.tree.delete(*self.tree.get_children())

        # Get first 10 rows
        preview_df = df.head(10)

        # Setup columns (include index)
        columns = ['index'] + list(preview_df.columns)
        self.tree['columns'] = columns
        self.tree['show'] = 'headings'

        # Configure column headers
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor='w')

        # Add data rows
        for idx, row in preview_df.iterrows():
            values = [str(idx)] + [str(val) for val in row]
            self.tree.insert('', 'end', values=values)

    def _validate_data(self, df: pd.DataFrame) -> tuple[bool, str]:
        """
        Validate DataFrame for ChronoTagger.

        Requirements:
        - Has DatetimeIndex
        - At least one numeric column
        - At least 2 rows
        - Index sorted ascending
        - No NaN in index

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check has DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            return False, "Error: No valid time column. Please select a time column."

        # Check for NaN in index
        if df.index.isna().any():
            return False, "Error: Time index contains missing values (NaN)."

        # Check at least 2 rows
        if len(df) < 2:
            return False, "Error: Data must have at least 2 rows."

        # Check index sorted
        if not df.index.is_monotonic_increasing:
            return False, "Error: Time index must be sorted in ascending order."

        # Check at least one numeric column
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return False, "Error: No numeric columns found. Data must have at least one numeric column."

        return True, ""

    def _handle_load_error(self, error: Exception):
        """
        Show error dialog with helpful suggestions.

        Args:
            error: Exception that occurred
        """
        # The dialog is transient; the traceback is not (Pack 4).
        logger.error("file load failed", exc_info=error)
        error_msg = str(error)

        # Provide helpful suggestions based on error type
        if "CSV" in error_msg or "Parquet" in error_msg:
            suggestion = "\n\nSuggestions:\n- Verify the file is not corrupted\n- Check file permissions\n- Ensure file is in correct format"
        elif "Unsupported file type" in error_msg:
            suggestion = "\n\nOnly CSV and Parquet files are supported."
        else:
            suggestion = "\n\nPlease check the file and try again."

        messagebox.showerror(
            "Error Loading File",
            f"Failed to load file:\n\n{error_msg}{suggestion}",
            parent=self.dialog
        )

        # Reset state
        self.file_path_var.set("")
        self.current_file = None
        self.loaded_df = None
        self.continue_btn.config(state='disabled')
        self.status_label.config(text="No file loaded", foreground='gray')
        self.tree.delete(*self.tree.get_children())

    def _on_continue(self):
        """Handle continue button click."""
        if self.loaded_df is None:
            return

        # Get selected time column
        selected = self.time_column_var.get()

        if selected == "Auto-detect":
            time_col = self._auto_detect_time_column(self.loaded_df)
        else:
            time_col = selected.replace("index (", "").replace(")", "") if "index" in selected else selected

        # Apply time column
        if time_col:
            df_result = self._apply_time_column(self.loaded_df, time_col)
        else:
            df_result = self.loaded_df

        # Final validation
        is_valid, error_msg = self._validate_data(df_result)

        if not is_valid:
            messagebox.showerror(
                "Invalid Data",
                error_msg,
                parent=self.dialog
            )
            return

        # Success - store result and close
        self.result = df_result
        self.dialog.destroy()

    def _on_cancel(self):
        """Handle cancel button click."""
        self.result = None
        self.dialog.destroy()
