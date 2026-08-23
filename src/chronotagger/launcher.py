#!/usr/bin/env python3
"""
ChronoTagger application launcher.

Provides entry point for both command-line and bundled app usage.
"""
import logging
import sys
import matplotlib
matplotlib.use('TkAgg')  # Force Tkinter-compatible backend for better performance


def main():
    """Launch ChronoTagger application."""
    try:
        # ONLY the import is guarded (Pack 4 R10): a failed import of the
        # wizard package is the one case where "wizard not available" is
        # true. Any ImportError from inside the RUNNING app used to be
        # reported with this message plus a use-the-API advert -- an
        # actively misleading answer to a missing optional dependency.
        from chronotagger.quickstart import wizard
    except ImportError as e:
        logging.getLogger("chronotagger.launcher").exception(
            "quick-start wizard import failed")
        print(f"Error: Quick-start wizard not available: {e}")
        import traceback
        traceback.print_exc()
        print("\nTo use ChronoTagger programmatically:")
        print("  from chronotagger import TimeIntervalLabeler")
        print("  app = TimeIntervalLabeler(df, plot_fn, labels)")
        print("  app.run()")
        sys.exit(1)

    try:
        wizard.run()
    except Exception as e:
        # Anything from inside the running app -- ImportError included --
        # is reported honestly, with a traceback, on both channels.
        logging.getLogger("chronotagger.launcher").exception(
            "ChronoTagger crashed")
        print(f"Error launching ChronoTagger: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
