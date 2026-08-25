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

# The emitter's spelling of the accepted units, imported rather than
# retyped: the loader and the driver it writes must agree about what
# "us" means (Pack 8 R11/R12).
from chronotagger.quickstart.driver_export import EPOCH_UNITS

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
        # Pack 8 R11.  pd.to_datetime reads a bare INTEGER column as
        # NANOSECONDS, so a microsecond epoch validated clean and landed
        # in 1970 -- a 3h59m dataset became 14.3 seconds wide and the
        # status line went black (measured end to end, csv and parquet).
        # The unit is now asked for, and asked for BLOCKINGLY.
        self.time_unit_var = tk.StringVar(value="")
        self.time_unit_combo = None
        self._unit_column = None
        # Pack 8 R14: set when the frame handed on had to be sorted.
        self._sorted_at_load = False
        # What the wizard needs to write a driver that loads this same
        # file the same way (Pack 8 R6).  Filled by _on_continue.
        self.time_column = None
        self.time_is_epoch = False
        self.time_unit = None

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

        # Epoch unit -- enabled only for an INTEGER time column, where it
        # is REQUIRED (Pack 8 R11).  It sits beside the column dropdown
        # because it is part of the same answer: "this column, read this
        # way".  Pack 6 F8's numeric gate makes auto-detect refuse an
        # integer column, so the manual dropdown is the only route to one
        # -- which is exactly the route that was unguarded.
        ttk.Label(time_col_frame, text="Epoch unit:").pack(side='left', padx=(15, 5))

        self.time_unit_combo = ttk.Combobox(
            time_col_frame,
            textvariable=self.time_unit_var,
            state='disabled',
            values=list(EPOCH_UNITS),
            width=6
        )
        self.time_unit_combo.pack(side='left')
        self.time_unit_combo.bind('<<ComboboxSelected>>', self._on_time_column_changed)

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
                ("Gzipped CSV files", "*.csv.gz"),
                ("Parquet files", "*.parquet"),
                ("All supported files", "*.csv *.csv.gz *.parquet"),
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

            # Resolve the frame the labeler would actually be handed
            # BEFORE writing the status line: R14 sorts an unsorted index
            # here instead of refusing it, and the status line is where
            # the user is told that it happened.
            df_with_time, is_valid, error_msg = self._resolve_frame(df, time_col)
            self._set_status(df_with_time, is_valid, error_msg)
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
        suffixes = [s.lower() for s in path.suffixes]

        # Pack 8 R12.  `Path.suffix` is the LAST suffix only, so a
        # `.csv.gz` read as `.gz`, fell into the else branch, and told
        # the user "Only CSV and Parquet files are supported" about a
        # file that IS a csv -- one `pd.read_csv` opens natively, by
        # decompressing on the extension. `.suffixes` is what knows the
        # difference. ONLY `.gz`: `.bz2` / `.zip` / `.xz` are not offered
        # by the picker and are not claimed here.
        if suffixes[-2:] == ['.csv', '.gz']:
            return pd.read_csv(file_path)
        elif suffixes[-1:] == ['.csv']:
            return pd.read_csv(file_path)
        elif suffixes[-1:] == ['.parquet']:
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

        # Check common column names.
        #
        # Pack 6 F8: the numeric gate Pack 5's amended R8 put on the
        # last-resort VALUE probe belongs here too. This loop runs FIRST
        # and had no gate at all, so a float column merely NAMED 'time'
        # won outright: pd.to_datetime read it as nanoseconds since 1970,
        # the whole dataset collapsed into a single 1970 instant, and
        # _validate_data passed. Measured on
        # ['time' float64, 'Bx_nT' float64, 'epoch' datetime64]:
        # detected 'time', index 1970-01-01 00:00:00 through
        # 1970-01-01 00:00:00.000005970, ok=True. A numeric column is
        # never a time axis -- not by VALUE, and now not by NAME either.
        skipped_numeric = 0
        common_names = ['time', 'timestamp', 'datetime', 'date', 'dt']
        for name in common_names:
            if name in df.columns:
                # Verify it can be converted to datetime
                try:
                    if pd.api.types.is_numeric_dtype(df[name]):
                        skipped_numeric += 1
                        continue
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
        # (Pack 6 F8: skipped_numeric is initialised before the NAME loop
        # above now, so the warning below reports refusals from BOTH.)
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

        self._sync_time_unit_control(df, auto_detected)

    def _on_time_column_changed(self, event=None):
        """Handle time column selection change."""
        if self.loaded_df is None:
            return

        selected = self.time_column_var.get()

        if selected == "Auto-detect":
            time_col = self._auto_detect_time_column(self.loaded_df)
        else:
            time_col = selected.replace("index (", "").replace(")", "") if "index" in selected else selected

        # The unit control follows the column: enabled and preselected
        # for an integer epoch, disabled and blank otherwise (R11).  This
        # handler is bound to BOTH dropdowns, so re-entry after the user
        # picks a unit must not overwrite that choice -- see
        # _sync_time_unit_control.
        self._sync_time_unit_control(self.loaded_df, time_col)

        df_with_time, is_valid, error_msg = self._resolve_frame(
            self.loaded_df, time_col)

        # Update continue button
        self.continue_btn.config(state='normal' if is_valid else 'disabled')

        # Update status
        self._set_status(df_with_time, is_valid, error_msg)

    def _is_epoch_column(self, df: pd.DataFrame, column) -> bool:
        """Is `column` a NUMERIC epoch, i.e. one that needs a unit?

        NUMERIC dtype, integers and floats alike, because both are read
        as NANOSECONDS by a bare pd.to_datetime and both therefore land
        in 1970 unlabelled.  Measured on a float64 seconds column: index
        1970-01-01 00:00:01.614556800 .. .614559190, span 2.39 us for a
        dataset 39m50s wide, Continue enabled and the status line black.
        Float seconds is the shape pyspedas and CDF time variables
        arrive in.  BOOL is excluded explicitly: pandas reports
        is_numeric_dtype(bool) as True.  A datetime-typed or string
        column still names its own units and is untouched.
        """
        if df is None or not isinstance(column, str) or not column:
            return False
        if column == "index" or column.startswith("index ("):
            return False
        if column not in df.columns:
            return False
        try:
            return bool(pd.api.types.is_numeric_dtype(df[column])
                        and not pd.api.types.is_bool_dtype(df[column]))
        except Exception:
            return False

    @staticmethod
    def _guess_epoch_unit(series) -> Optional[str]:
        """Preselect a unit from the MEDIAN MAGNITUDE of the column.

        A present-day timestamp is ~1.7e9 in seconds, ~1.7e12 in
        milliseconds, ~1.7e15 in microseconds and ~1.7e18 in
        nanoseconds, so three decades of magnitude separate the four.
        The boundaries sit between those decades; a guess is a
        preselection the user can override, never a silent decision.

        The MEDIAN, not the first row (Pack 8 F12).  Measured: a column
        of 1000 NANOSECOND values whose first row is a 0 fill value
        preselected "s" -- three decades wrong -- and Continue lights up
        the moment any unit is present, so nothing made the user look.
        A fill value need not be zero (-1, -9999 and 1e31 are all in the
        wild) and need not be first, which is why this is a median over
        the column rather than "the first non-zero value" or a head
        sample.  `to_numeric(errors="coerce").dropna()` has already
        dropped the NaN, so the median is over real values only, and
        taking the MAGNITUDE first keeps a pre-1970 negative epoch
        answering the same as its positive twin (measured: -1.5e9 -> s,
        -1.5e18 -> ns, an all-zero column -> s).
        """
        try:
            values = pd.to_numeric(series, errors="coerce").dropna()
        except Exception:
            return None
        if len(values) == 0:
            return None
        try:
            magnitude = abs(float(values.abs().median()))
        except Exception:
            return None
        if magnitude < 1e11:
            return "s"
        if magnitude < 1e14:
            return "ms"
        if magnitude < 1e17:
            return "us"
        return "ns"

    def _sync_time_unit_control(self, df, column) -> None:
        """Enable + preselect the unit dropdown for an integer column.

        The preselection fires only when the COLUMN CHANGED, so a unit
        the user picked by hand survives the `<<ComboboxSelected>>`
        re-entry this method is called from -- and so does a unit they
        cleared, which is what makes the blocking branch of R11
        reachable rather than a guard nothing can ever trip.
        """
        if self.time_unit_combo is None:
            return
        if not self._is_epoch_column(df, column):
            self.time_unit_var.set("")
            self._unit_column = None
            self.time_unit_combo.config(state='disabled')
            return
        if self._unit_column != column:
            self.time_unit_var.set(self._guess_epoch_unit(df[column]) or "")
        self._unit_column = column
        self.time_unit_combo.config(state='readonly')

    def _selected_time_unit(self, df, column) -> Optional[str]:
        """The unit to convert `column` with, or None for "no unit"."""
        if not self._is_epoch_column(df, column):
            return None
        unit = self.time_unit_var.get()
        return unit if unit in EPOCH_UNITS else None

    def _validate_time_selection(self, df, column) -> tuple[bool, str]:
        """The BLOCKING half of R11, ahead of the frame validation.

        An integer column with no unit is refused outright rather than
        converted on a guess: a bare conversion reads integers as
        nanoseconds, which is right for one unit in four and moves the
        dataset to 1970 for the other three -- silently, with a valid
        DatetimeIndex and a green status line.
        """
        if not self._is_epoch_column(df, column):
            return True, ""
        if self.time_unit_var.get() in EPOCH_UNITS:
            return True, ""
        return False, (
            f"Error: '{column}' holds numbers, not dates. Choose its "
            f"epoch unit (s / ms / us / ns) -- read as the wrong unit, "
            f"the data lands in 1970 without complaining."
        )

    def _resolve_frame(self, df, column):
        """(frame, is_valid, message) for one (df, time column) choice.

        The single place the three screens agree on what the labeler
        would be handed: the unit gate (R11), the conversion, the sort
        (R14), then the frame validation -- in that order.  A conversion
        that RAISES is turned into the same (frame, False, message) shape
        the two gates already return (Pack 8 F13), so every failure on
        this screen reaches the user through one status line.
        """
        ok, msg = self._validate_time_selection(df, column)
        if not ok:
            return df, False, msg
        if column:
            try:
                frame = self._apply_time_column(
                    df, column, self._selected_time_unit(df, column))
            except Exception as exc:
                # A column of category strings is one misclick away in
                # the dropdown, and pd.to_datetime raises rather than
                # returning NaT.  Unguarded, that exception escapes a
                # <<ComboboxSelected>> callback and a Button command:
                # Tkinter prints a traceback nobody sees and the screen
                # keeps a stale black status line with Continue still
                # enabled.  The traceback is not lost -- Pack 4 R6c.
                logger.warning("cannot read %r as a time column",
                               column, exc_info=exc)
                return df, False, (
                    f"Error: '{column}' cannot be read as a time column "
                    f"({type(exc).__name__}). Choose a different column.")
        else:
            self._sorted_at_load = False
            frame = df
        ok, msg = self._validate_data(frame)
        return frame, ok, msg

    def _set_status(self, df, is_valid: bool, error_msg: str) -> None:
        """One status line, and it tells the truth about the sort (R14)."""
        if not is_valid:
            self.status_label.config(text=error_msg, foreground='red')
            return
        note = " -- sorted by time" if self._sorted_at_load else ""
        self.status_label.config(
            text=f"Rows: {len(df)}, Columns: {len(df.columns)}{note}",
            foreground='black'
        )

    def _sort_if_needed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pack 8 R14: SORT an unsorted time index, do not refuse it.

        Three sites had three policies on the same frame: this screen
        REFUSED, `TimeIntervalLabeler.__init__` sorts with a warning
        (Pack 6 R9), and a Pack-7 emitted driver sorts unconditionally
        and silently.  So the wizard blocked the user from generating the
        very driver that handles their data correctly.  Sorting here is
        the behaviour two of the three paths already had; the status line
        is the visible half, because silently reordering someone's frame
        is its own surprise.
        """
        if (isinstance(df.index, pd.DatetimeIndex)
                and not df.index.is_monotonic_increasing):
            self._sorted_at_load = True
            logger.warning(
                "time index is not monotonically increasing (%d rows); "
                "sorting by index. This also changes the dataset "
                "fingerprint, so an autosave written from the unsorted "
                "frame will not be offered for recovery.", len(df.index))
            return df.sort_index()
        return df

    def _apply_time_column(self, df: pd.DataFrame, column: str,
                           unit: Optional[str] = None) -> pd.DataFrame:
        """
        Apply time column as index.

        Args:
            df: DataFrame
            column: Column name to use as time index
            unit: Epoch unit ("s"/"ms"/"us"/"ns") for an INTEGER column,
                or None when the column names its own units.  A bare
                `pd.to_datetime` on integers means NANOSECONDS, which is
                a decision, not a default (Pack 8 R11).

        Returns:
            DataFrame with time column as DatetimeIndex, sorted (R14)
        """
        df_copy = df.copy()
        self._sorted_at_load = False

        # If column is "index", just ensure it's datetime
        if column == "index" or column.startswith("index ("):
            if not isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy.index = pd.to_datetime(df_copy.index)
            return self._sort_if_needed(df_copy)

        # Convert column to datetime and set as index
        if column in df_copy.columns:
            if unit is None:
                df_copy[column] = pd.to_datetime(df_copy[column])
            else:
                df_copy[column] = pd.to_datetime(df_copy[column], unit=unit)
            df_copy = df_copy.set_index(column)

        return self._sort_if_needed(df_copy)

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
        - No NaN in index

        NOT "index sorted ascending" any more (Pack 8 R14): an unsorted
        frame is SORTED by _apply_time_column and reported on the status
        line, because the labeler constructor and the emitted driver both
        sort it too and refusing here blocked the user from generating
        the driver that handles their data correctly.

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

        # Apply the time column, through the same gate the live status
        # line uses, so Continue can never accept what the screen was
        # showing as an error.
        df_result, is_valid, error_msg = self._resolve_frame(
            self.loaded_df, time_col)

        if not is_valid:
            messagebox.showerror(
                "Invalid Data",
                error_msg,
                parent=self.dialog
            )
            return

        # Record HOW this frame was built, so the wizard can write a
        # driver that loads the same file the same way (Pack 8 R6/R11).
        # A time value taken from the INDEX has no column to convert, so
        # the driver is told there is none.
        self.time_column = (
            time_col if time_col and time_col in self.loaded_df.columns
            else None)
        self.time_is_epoch = self._is_epoch_column(self.loaded_df,
                                                   self.time_column)
        self.time_unit = self._selected_time_unit(self.loaded_df,
                                                  self.time_column)

        # Success - store result and close
        self.result = df_result
        self.dialog.destroy()

    def _on_cancel(self):
        """Handle cancel button click."""
        self.result = None
        self.dialog.destroy()
