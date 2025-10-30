"""
Example Usage of Time Interval Labeler

This script demonstrates how to use the TimeIntervalLabeler tool
with synthetic time-series data.
"""

import pandas as pd
import numpy as np
from labeler import TimeIntervalLabeler


def my_plot_function(axs, df, t0, t1):
    """
    User-defined plotting function.
    
    This function receives a dictionary of matplotlib axes and draws
    the desired panels for the time range [t0, t1).
    
    Parameters
    ----------
    axs : dict
        Dictionary of matplotlib axes. Keys are 'panel1', 'panel2', etc.
    df : pd.DataFrame
        Dataframe slice for the current time window (already filtered).
    t0 : pd.Timestamp
        Window start time.
    t1 : pd.Timestamp
        Window end time.
    """
    # Panel 1: Density (log scale)
    axs["panel1"].plot(df.index, df["log10n"], color='blue', linewidth=1)
    axs["panel1"].set_ylabel("log₁₀(n) [cm⁻³]", fontsize=10)
    axs["panel1"].grid(True, alpha=0.3)
    axs["panel1"].set_title(f"Plasma Density: {t0.strftime('%Y-%m-%d %H:%M')} to {t1.strftime('%H:%M')}")
    
    # Panel 2: Magnetic field components
    axs["panel2"].plot(df.index, df["BX"], label="Bₓ", color='red', linewidth=1)
    axs["panel2"].plot(df.index, df["BY"], label="Bᵧ", color='green', linewidth=1)
    axs["panel2"].plot(df.index, df["BZ"], label="Bᵨ", color='blue', linewidth=1)
    axs["panel2"].set_ylabel("B [nT]", fontsize=10)
    axs["panel2"].legend(loc='upper right', framealpha=0.5, fontsize=8)
    axs["panel2"].grid(True, alpha=0.3)
    axs["panel2"].axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)


def generate_synthetic_data():
    """
    Generate synthetic magnetosphere-like time-series data.
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with DatetimeIndex and columns: log10n, BX, BY, BZ
    """
    # Create irregular time index (simulating realistic spacecraft data)
    # Base: ~37 second cadence with some gaps
    base_times = pd.date_range("2015-01-03", "2015-01-07", freq="37S")
    
    # Add some gaps to simulate data dropouts
    # Remove 10 random hour-long chunks
    np.random.seed(42)
    keep_mask = np.ones(len(base_times), dtype=bool)
    for _ in range(10):
        gap_start = np.random.randint(0, len(base_times) - 100)
        gap_length = np.random.randint(50, 100)  # ~30-60 minutes
        keep_mask[gap_start:gap_start + gap_length] = False
    
    times = base_times[keep_mask]
    
    # Create DataFrame
    df = pd.DataFrame(index=times)
    
    # Generate synthetic data with different "regions"
    # Simulate plasma sheet, lobe, magnetosheath transitions
    
    # Density: varies by region
    # Low in lobes (~0.01), medium in plasma sheet (~0.3), high in magnetosheath (~10)
    t_numeric = np.arange(len(df))
    base_density = 0.5 + 0.3 * np.sin(2 * np.pi * t_numeric / 3000)
    
    # Add region-like variations
    region_signal = 2 * np.sin(2 * np.pi * t_numeric / 8000)
    log10n = base_density + region_signal + 0.1 * np.random.randn(len(df))
    df["log10n"] = log10n
    
    # Magnetic field: typical magnetosphere behavior
    # BX: mostly positive in tail
    df["BX"] = 10 + 5 * np.sin(2 * np.pi * t_numeric / 5000) + 2 * np.random.randn(len(df))
    
    # BY: varies
    df["BY"] = 3 * np.cos(2 * np.pi * t_numeric / 7000) + 1.5 * np.random.randn(len(df))
    
    # BZ: important for region identification
    # Negative in plasma sheet, more variable in magnetosheath
    df["BZ"] = -5 + 8 * np.sin(2 * np.pi * t_numeric / 4000) + 2 * np.random.randn(len(df))
    
    return df


def main():
    """
    Main function to demonstrate the Time Interval Labeler.
    """
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    print(f"Created dataframe with {len(df)} samples")
    print(f"Time range: {df.index[0]} to {df.index[-1]}")
    print(f"Columns: {list(df.columns)}")
    
    # Define label classes (customize these for your science case)
    classes = [
        "PlasmaSheet",
        "Lobe", 
        "Magnetosheath",
        "SolarWind",
        "UNKNOWN"
    ]
    
    # Define colors for each class (optional, will use defaults if not provided)
    colors = {
        "PlasmaSheet": "#4e79a7",    # Blue
        "Lobe": "#f28e2b",           # Orange
        "Magnetosheath": "#e15759",  # Red
        "SolarWind": "#76b7b2",      # Teal
        "UNKNOWN": "#bab0ac"         # Gray
    }
    
    print("\nLaunching Time Interval Labeler...")
    print("\nKeyboard shortcuts:")
    print("  1-5: Select label class")
    print("  n/p: Next/Previous window")
    print("  u: Select UNKNOWN class")
    print("  s: Save session")
    print("  e: Export intervals")
    print("  Backspace: Undo")
    print("  Shift+Backspace: Redo")
    print("\nWorkflow:")
    print("  1. Drag on the plot to select a time range")
    print("  2. Choose a class (radio buttons or number keys)")
    print("  3. Click 'Add Interval'")
    print("  4. Click on colored bars in the strip to select/edit them")
    print("  5. Use 'Assign Remainder → UNKNOWN' to label unlabeled gaps")
    
    # Create and run the labeler
    labeler = TimeIntervalLabeler(
        df=df,
        plot_fn=my_plot_function,
        classes=classes,
        class_colors=colors,
        window=pd.Timedelta("60min"),  # 1-hour windows
        step=pd.Timedelta("30min"),    # 30-minute steps
        autosave_path="session_autosave.json",  # Auto-save on each change
    )
    
    # Start the GUI
    labeler.run()
    
    print("\nLabeler closed. Have a great day!")


if __name__ == "__main__":
    main()
