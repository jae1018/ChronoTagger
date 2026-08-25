"""
Quick-Start Wizard for ChronoTagger.

Provides a GUI-based workflow for loading data and configuring plots
without requiring users to write Python code.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
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
        # Pack 8 R13. Cancel used to call sys.exit(0), which propagated
        # out of run() and killed the host interpreter: fine from a
        # terminal, fatal in a Jupyter kernel, an IDE run, or any script
        # that calls run() and expects to continue -- and with exit code
        # 0, so a wrapping shell could not tell "cancelled" from "done".
        self.cancelled: bool = False
        # Configuration produced by the tab planner -- a list of dicts,
        # one per labeler tab. Each entry has 'title', 'columns',
        # 'layout_type', and (for custom_grid) 'layout_spec' +
        # 'plot_config'. See chronotagger.quickstart.tab_planner.
        self.tabs_config: Optional[list] = None
        # Filled by _launch_labeler: one {'title', 'layout_spec',
        # 'plot_config'} per pane, i.e. everything a driver file needs
        # about the FIGURE. Kept separate from tabs_config because that
        # one holds the user's raw answers, not the resolved spec.
        self.pane_specs: list = []
        # Pack 8 R8: the label schema, asked ONCE between the tab planner
        # and the launch, and handed to BOTH the live labeler and any
        # driver file the user saves. None means "not asked yet" -- the
        # guard that stops the launch-failure retry re-asking.
        self.classes: Optional[list] = None
        self.class_colors: Optional[dict] = None
        # Where the data came from. source_name is the PORTABLE STEM,
        # the same spelling the emitter writes into SOURCE_NAME (Pack 8
        # R4): _check_autosave compares source_name when two datasets
        # share a fingerprint, and the wizard used to store the full path
        # while a driver from the same session stored the stem -- two
        # identities for one dataset. source_path keeps the full path,
        # which only DATA_PATH and the autosave folder need.
        self.source_name: Optional[str] = None
        self.source_path: Optional[str] = None
        # How the loader built the DatetimeIndex, so a driver written
        # from this session loads the same file the same way (R6/R11).
        self.time_column: Optional[str] = None
        self.time_is_epoch: bool = False
        self.time_unit: Optional[str] = None

    def run(self):
        """
        Run the quick-start wizard.  Returns None.

        This is the main entry point called by launcher.py.

        Note the shape, because Pack 8 R13 depends on it: the ENTIRE
        wizard flow runs at _show_file_loader() BELOW, before
        mainloop(). Every screen blocks on wait_window(), which runs a
        nested Tcl event loop, so by the time mainloop() is reached the
        user has already finished or cancelled. A cancel therefore only
        has to unwind and skip the loop -- it never had to kill the
        process to get out.
        """
        from chronotagger.labeler.mixins.view_build.window import _new_tk_root

        # Pack 8 R16: Pack 6 R10's bounded retry, on the wizard's own
        # root. tk.Tk() raised a transient TclError in 89% of full-suite
        # runs on this machine; the retry recovers inside two attempts.
        # Every other root in the package was already protected.
        self.root = _new_tk_root()
        self.root.title("ChronoTagger Quick Start")
        self.root.geometry("700x600")  # Larger for file dialog

        # Center window on screen
        self._center_window()

        # Start the wizard flow
        self._show_file_loader()

        if self.cancelled:
            # The root is already destroyed; entering mainloop() on it
            # would be a no-op at best. Return cleanly instead (R13).
            return None

        # Start Tkinter main loop
        self.root.mainloop()
        return None

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
        #
        # Pack 8 R4: source_name is the portable STEM, produced by the
        # EMITTER's own rule, so the live session and a driver generated
        # from it carry one identity rather than two spellings of one
        # dataset. source_path keeps the full path for DATA_PATH and the
        # autosave folder.
        from chronotagger.quickstart.driver_export import portable_stem

        self.df = df
        self.source_path = getattr(file_dialog, 'current_file', None)
        self.source_name = (portable_stem(self.source_path)
                            if self.source_path else None)
        # How the loader read the time axis -- the driver needs all three
        # to reproduce the same DatetimeIndex (Pack 8 R6/R11).
        self.time_column = getattr(file_dialog, 'time_column', None)
        self.time_is_epoch = bool(getattr(file_dialog, 'time_is_epoch', False))
        self.time_unit = getattr(file_dialog, 'time_unit', None)

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

        # Pack 8 R5: tab planner -> CLASSES -> launch. Asked ONCE: a
        # failed launch recurses back into THIS method (see
        # _launch_labeler's except branch), and re-asking a question the
        # user already answered is how a retry loop becomes a trap.
        if self.classes is None:
            self._show_classes()

        self._launch_labeler()

    def _show_classes(self):
        """Ask for the label schema -- names, colours, order (R3/R8).

        REUSES the labeler's own Label Manager rather than growing a
        second schema editor: it already does add / rename / recolour /
        move / delete-with-reassign, it returns exactly the two fields
        both consumers need, and it constructs with no labeler behind it.
        Two arguments differ from the live call, both deliberately:

          * `reserved=frozenset()`. UNKNOWN is the CONSTRUCTOR's default
            first class, not a law about what a user may call theirs --
            and a wizard schema of sheet / lobe / sheath is unreachable
            while it is reserved. The LIVE Label Manager keeps its
            reservation untouched, because there intervals already carry
            the name.
          * `usage_counts={}`. Nothing has been labelled yet.

        Cancelling keeps the defaults, which is what "I did not want to
        change the schema" means. Without this screen a wizard session
        produced a training set whose classes were UNKNOWN / label_1 /
        label_2 -- a faithful mapping carrying no physics.
        """
        from chronotagger.labeler.dialogs.label_manager import (
            LabelManagerDialog,
        )
        from chronotagger.quickstart.driver_export import (
            DEFAULT_CLASSES,
            DEFAULT_COLORS,
        )

        classes = list(DEFAULT_CLASSES)
        colors = {name: DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                  for i, name in enumerate(classes)}

        dialog = LabelManagerDialog(
            parent=self.root,
            classes=classes,
            class_colors=colors,
            usage_counts={},
            reserved=frozenset(),
        )
        self.root.wait_window(dialog)

        if dialog.result is not None and dialog.result.classes:
            classes = list(dialog.result.classes)
            colors = dict(dialog.result.class_colors)

        self.classes = classes
        self.class_colors = colors

    def _autosave_folder(self) -> str:
        """`<data file's directory>/chronotagger_autosave` (Pack 8 R9).

        The constructor default is "." and the wizard never overrode it,
        so a session dropped its fingerprinted autosave JSON -- and
        chronotagger.log, which Pack 4 put beside it -- into whatever
        directory the app happened to be launched from. Measured: the log
        was already written before any screen could have asked.
        """
        if self.source_path:
            return os.path.join(
                os.path.dirname(os.path.abspath(self.source_path)),
                "chronotagger_autosave")
        return os.path.join(".", "chronotagger_autosave")

    def _data_format(self) -> str:
        """"csv", "csv.gz" or "parquet", from the data file's name."""
        suffixes = [s.lower() for s in Path(self.source_path or "").suffixes]
        if suffixes[-2:] == [".csv", ".gz"]:
            return "csv.gz"
        if suffixes[-1:] == [".parquet"]:
            return "parquet"
        return "csv"

    def _offer_save_as(self, window, step) -> Optional[str]:
        """Offer to write a driver file for THIS session (R6/R7).

        The DIALOG COMES FIRST and the basename it returns is passed to
        the emitter as `file_name`, so the generated header's
        `python <name>` line and the file on disk cannot disagree --
        which they would the moment the user renamed in the dialog.

        `overwrite=True` is honest here and only here: the native
        save-as dialog has already asked about an existing file, and
        `write_driver` refuses to replace one silently otherwise.
        `confirmoverwrite=True` is passed EXPLICITLY rather than left to
        tkinter (Pack 8, V1 FOLD 7): `tk_getSaveFile`'s own default is
        true, but it is a platform default, and it is the only thing
        standing between `overwrite=True` here and silently replacing a
        driver the user has hand-edited.  W8 says "never silent
        overwrite", so the pack says it rather than inheriting it.

        DECLINING WRITES NOTHING -- no fallback file, no temp copy (R7).
        The session launches either way.

        A failure to WRITE is reported and swallowed: the user asked for
        a labeling session, and losing it because a driver file could not
        be saved would be the tail wagging the dog.
        """
        import logging

        from chronotagger.quickstart.driver_export import (
            generate_driver,
            write_driver,
        )

        if not self.source_path or not self.pane_specs:
            return None

        default_name = "drive_%s.py" % (self.source_name or "chronotagger")
        chosen = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save a driver file for this session?",
            defaultextension=".py",
            initialfile=default_name,
            initialdir=os.path.dirname(os.path.abspath(self.source_path)),
            confirmoverwrite=True,
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
        )
        if not chosen:
            return None

        try:
            text = generate_driver(
                self.pane_specs,
                data_path=self.source_path,
                fmt=self._data_format(),
                time_column=self.time_column,
                time_is_epoch=self.time_is_epoch,
                time_unit=self.time_unit,
                classes=self.classes,
                colors=self.class_colors,
                window=window,
                step=step,
                # Pack 8 A8-2: the driver DERIVES its autosave folder
                # from DATA_PATH at runtime, so this session and the file
                # it is writing share one autosave lineage -- measured
                # before the amendment, session and driver wrote to
                # different folders, so the wizard's own autosave was not
                # offered when the wizard's own driver was launched from
                # anywhere but the data directory. Nothing
                # machine-specific is added to the text; DATA_PATH is
                # already absolute and already there.
                autosave_beside_data=True,
                source_name=self.source_name,
                file_name=os.path.basename(chosen),
            )
            write_driver(text, chosen, overwrite=True)
        except Exception as exc:
            logging.getLogger("chronotagger.quickstart.wizard").exception(
                "driver export failed")
            messagebox.showerror(
                "Could Not Write Driver",
                f"The labeling session will still open.\n\n"
                f"Writing {chosen} failed:\n\n{exc}",
                parent=self.root,
            )
            return None
        return chosen

    def _build_tab_plot(self, tab: dict):
        """
        Build (layout_spec, plot_config, plot_fn) for one tab config dict.

        Both layout types now end at the SAME generator (Pack 7 W1).
        'vertical_stack' is a preset that produces a designer-shaped
        (layout_spec, plot_config) pair; 'custom_grid' reuses the pair
        the user designed in the planner. Either way the runnable
        plot_fn comes from `plot_generator.generate_plot_fn`, so the
        live figure and a driver file emitted from the same state are
        the same figure.

        The pair is returned alongside the plot_fn because the driver
        emitter needs the plot_config, and a closure cannot be read
        back out of a plot_fn.
        """
        from chronotagger.labeler.utils.plot_generator import (
            generate_plot_fn,
            normalize_time_columns,
            validate_plot_inputs,
            vertical_stack_config,
        )

        columns = tab["columns"]
        layout_type = tab["layout_type"]

        if layout_type == "vertical_stack":
            validate_plot_inputs(self.df, columns)
            layout_spec, plot_config = vertical_stack_config(columns)
        elif layout_type == "custom_grid":
            layout_spec = tab["layout_spec"]
            plot_config = tab["plot_config"]
            # Only the designed layout needs coercing: the preset builds
            # every time area in column 0 already, and normalizing it
            # would write colspan=1 onto areas that deliberately omit it.
            normalize_time_columns(layout_spec)
        else:
            raise ValueError(
                f"Unknown layout type: {layout_type!r}. "
                f"Expected 'vertical_stack' or 'custom_grid'."
            )

        plot_fn = generate_plot_fn(plot_config)
        return plot_fn, layout_spec, plot_config

    def _launch_labeler(self):
        """Launch TimeIntervalLabeler with the configured tabs."""
        from chronotagger import TimeIntervalLabeler

        try:
            # Build (plot_fn, layout_spec) for each tab. The plot_config
            # is kept beside them on self.pane_specs so a later screen
            # can hand the pair to the driver emitter -- the labeler
            # itself never sees it (Pack 7 W1/W5).
            pane_configs = []
            self.pane_specs = []
            for tab in self.tabs_config:
                plot_fn, layout_spec, plot_config = self._build_tab_plot(tab)
                pane_configs.append({
                    "title": tab["title"],
                    "plot_fn": plot_fn,
                    "layout_spec": layout_spec,
                })
                self.pane_specs.append({
                    "title": tab["title"],
                    "layout_spec": layout_spec,
                    "plot_config": plot_config,
                })

            # Calculate a reasonable default window (10% of data range)
            time_range = self.df.index[-1] - self.df.index[0]
            default_window = time_range * 0.1

            # Insurance for multi-million-row frames (Pack 5 R5). The 10%
            # rule scales by TIME RANGE, not row count, so on the real
            # files it opens 304,119 points (3.0M-row peif) and 1,033,278
            # (13.6M-row spinres) in the first frame. Cap the FIRST window
            # at ~200k samples by asking the index where sample 200,000
            # sits. Honest limit, also measured: at the 147k-point default
            # scale a cap buys ~14% -- this is a bound on the worst case,
            # not the fix for "feels slow on first open" (pack5_g2 7/S6).
            first_frame_cap = 200_000
            if len(self.df.index) > first_frame_cap:
                try:
                    span_cap = self.df.index[first_frame_cap] - self.df.index[0]
                    if pd.Timedelta(0) < span_cap < default_window:
                        default_window = span_cap
                except Exception:
                    pass

            # Pack 8 R10: the constructor's own 30/15 ratio, applied to
            # the window the wizard just computed. app.py's 15-minute
            # default step survived untouched against a 10%-of-range
            # window, so on a ten-minute file one "next window" skipped
            # fifteen times the visible span (measured: window 59.9 s,
            # step 15 min).
            #
            # `or` is the floor, not a style choice: a frame whose
            # timestamps are all identical passes _validate_data (>= 2
            # rows, no NaT, non-strictly monotonic), so time_range * 0.1
            # is Timedelta(0) and window / 2 is Timedelta(0) too --
            # "next window" would advance by nothing. pd.Timedelta(0) is
            # falsy; any real step is truthy and passes through
            # untouched.
            default_step = default_window / 2 or pd.Timedelta("15min")

            # Pack 8 R9: autosaves and the forensic log go beside the
            # DATA, not into the process CWD.
            autosave_folder = self._autosave_folder()

            # Pack 8 R5/R6: save-as sits here -- after pane_specs, the
            # window and the step exist, and before the labeler is built
            # -- so the driver states the session that is about to open.
            # Declining writes nothing and launches anyway (R7).
            self._offer_save_as(default_window, default_step)

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
                    classes=self.classes,
                    class_colors=self.class_colors,
                    window=default_window,
                    step=default_step,
                    autosave_folder=autosave_folder,
                    source_name=self.source_name,
                    parent=self.root,
                )
            else:
                # Multi-pane API
                labeler = TimeIntervalLabeler(
                    df=self.df,
                    panes=pane_configs,
                    classes=self.classes,
                    class_colors=self.class_colors,
                    window=default_window,
                    step=default_step,
                    autosave_folder=autosave_folder,
                    source_name=self.source_name,
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
        """Handle cancellation -- WITHOUT killing the interpreter (R13).

        `sys.exit(0)` used to run here. It is not in a Tk callback -- the
        cancel button merely destroys its dialog, wait_window() returns,
        and this runs as a plain call on the pre-mainloop stack -- so the
        SystemExit propagated through _show_file_loader, run(), and
        launcher.py's `except Exception` (SystemExit is a BaseException)
        straight out of the process. Setting a flag lets run() return.
        """
        result = messagebox.askyesno(
            "Exit Wizard",
            "Exit the wizard?",
            parent=self.root
        )
        if result:
            self.cancelled = True
            self.root.destroy()


def run():
    """
    Run the quick-start wizard.

    This function is called by launcher.py.  Returns None, including
    when the user cancels -- which no longer terminates the interpreter
    (Pack 8 R13), so an embedder that calls this and carries on now
    survives a cancel.
    """
    wizard = QuickStartWizard()
    return wizard.run()


if __name__ == "__main__":
    # Allow running wizard directly for testing
    run()
