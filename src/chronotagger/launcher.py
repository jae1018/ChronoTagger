#!/usr/bin/env python3
"""
ChronoTagger application launcher.

Provides entry point for both command-line and bundled app usage.
"""
import sys
import matplotlib
matplotlib.use('TkAgg')  # Force Tkinter-compatible backend for better performance


def main():
    """Launch ChronoTagger application."""
    try:
        # Try to import and run the quick-start wizard
        from chronotagger.quickstart import wizard
        wizard.run()
    except ImportError as e:
        # Quick-start not available (shouldn't happen, but defensive)
        print(f"Error: Quick-start wizard not available: {e}")
        print("\nTo use ChronoTagger programmatically:")
        print("  from chronotagger import TimeIntervalLabeler")
        print("  app = TimeIntervalLabeler(df, plot_fn, labels)")
        print("  app.run()")
        sys.exit(1)
    except Exception as e:
        # Unexpected error
        print(f"Error launching ChronoTagger: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
