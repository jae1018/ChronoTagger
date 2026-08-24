# examples/mixed_layout.py
"""
Example demonstrating mixed time-series and position plots.

This shows how to use role="not-time" for position/phase-space plots
where box selection maps point order back to timestamps.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from chronotagger.labeler import TimeIntervalLabeler

def make_df():
    """Create synthetic spacecraft trajectory data."""
    idx = pd.date_range("2015-01-03 00:00:00", periods=720, freq="5s")
    n = len(idx)
    t = np.linspace(0, 6*np.pi, n)
    
    # Time series measurements
    log10n = 0.3 + 0.1*np.sin(t) + 0.02*np.random.randn(n)
    bx = 8*np.sin(t) + 0.8*np.random.randn(n)
    by = 6*np.cos(t) + 0.6*np.random.randn(n)
    bz = -2 + 2*np.sin(0.7*t) + 0.5*np.random.randn(n)
    
    # Position coordinates (X-Y-Z in some reference frame)
    # These represent the spacecraft position at each time
    x_gse = 5*np.cos(t*0.8) + 2.0
    y_gse = 3*np.sin(t*0.8)
    z_gse = 1.5*np.sin(t*1.2)
    
    df = pd.DataFrame({
        "log10n": log10n,
        "BX": bx, "BY": by, "BZ": bz,
        "X_GSE": x_gse, "Y_GSE": y_gse, "Z_GSE": z_gse,
    }, index=idx)
    
    return df

layout_spec = {
    # Row 3 is the role="labels" strip. It became mandatory in 584f705
    # (2025-11-04, "Added labels plot as mandatory to layout_builder");
    # this example was last touched before that and had been raising
    # ValueError at build time ever since. It sits in column 0 so it lines
    # up with the time panels it annotates.
    "nrows": 4, "ncols": 2,
    "height_ratios": [1.0, 1.0, 1.0, 0.4],
    "width_ratios": [2.0, 1.5],
    "hspace": 0.15, "wspace": 0.12,
    "areas": [
        # Left column: time-series plots (role="time")
        {"key": "n",      "row": 0, "col": 0, "role": "time"},
        {"key": "b",      "row": 1, "col": 0, "role": "time"},
        {"key": "bmag",   "row": 2, "col": 0, "role": "time"},

        # Right column: position plots (role="not-time")
        # Box selection on these axes will map point order -> timestamps
        {"key": "pos_xy", "row": 0, "col": 1, "role": "not-time"},
        {"key": "pos_xz", "row": 1, "col": 1, "role": "not-time"},
        {"key": "pos_yz", "row": 2, "col": 1, "role": "not-time"},

        # Labels strip, under the time column
        {"key": "labels", "row": 3, "col": 0, "role": "labels"},
    ],
}

def plot_fn(axs, df, t0, t1):
    """
    Plot function that creates both time-series and position plots.
    
    IMPORTANT for role="not-time" axes:
    - Plot data in the ORDER of df.index (don't sort by X/Y!)
    - The Nth point in the scatter corresponds to df.index[N]
    - This allows box selection to map selected points back to time
    """
    # === Time-series plots (left column) ===
    ax = axs["n"]
    ax.plot(df.index, df["log10n"], 'o-', markersize=2)
    ax.set_ylabel("log10(n) [cm$^{-3}$]")
    ax.grid(alpha=0.3)
    
    ax = axs["b"]
    ax.plot(df.index, df["BX"], label="B$_X$", linewidth=1)
    ax.plot(df.index, df["BY"], label="B$_Y$", linewidth=1)
    ax.plot(df.index, df["BZ"], label="B$_Z$", linewidth=1)
    ax.set_ylabel("B [nT]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    
    ax = axs["bmag"]
    bmag = np.sqrt(df["BX"]**2 + df["BY"]**2 + df["BZ"]**2)
    ax.plot(df.index, bmag, 'o-', markersize=2, color='purple')
    ax.set_ylabel("|B| [nT]")
    ax.grid(alpha=0.3)
    
    # === Position plots (right column) ===
    # CRITICAL: Use df.index order for positions!
    # Don't sort by X/Y coordinates - the temporal order must be preserved
    
    # X-Y projection
    ax = axs["pos_xy"]
    ax.scatter(df["X_GSE"], df["Y_GSE"], s=10, c='blue', alpha=0.6)
    ax.set_xlabel("X (GSE - RE)")
    ax.set_ylabel("Y (GSE - RE)")
    ax.set_title("X-Y Position", fontsize=9)
    ax.grid(alpha=0.3)
    ax.axis('equal')
    
    # X-Z projection
    ax = axs["pos_xz"]
    ax.scatter(df["X_GSE"], df["Z_GSE"], s=10, c='green', alpha=0.6)
    ax.set_xlabel("X (GSE - RE)")
    ax.set_ylabel("Z (GSE - RE)")
    ax.set_title("X-Z Position", fontsize=9)
    ax.grid(alpha=0.3)
    ax.axis('equal')
    
    # Y-Z projection
    ax = axs["pos_yz"]
    ax.scatter(df["Y_GSE"], df["Z_GSE"], s=10, c='red', alpha=0.6)
    ax.set_xlabel("Y (GSE - RE)")
    ax.set_ylabel("Z (GSE - RE)")
    ax.set_title("Y-Z Position", fontsize=9)
    ax.grid(alpha=0.3)
    ax.axis('equal')

if __name__ == "__main__":
    print("=== ChronoTagger: Mixed Time-Series + Position Plots ===")
    print()
    print("FEATURES:")
    print("  • Left column: Time-series plots (drag box to select by time + value)")
    print("  • Right column: Position plots (drag box to select by position)")
    print()
    print("HOW TO USE:")
    print("  1. Drag a box on any LEFT panel -> selects points in time-value space")
    print("  2. Drag a box on any RIGHT panel -> selects points in position space")
    print("  3. Both methods create time intervals you can label!")
    print()
    print("TIP: Try selecting a specific region in Y-Z space to label trajectory segments")
    print("=" * 60)
    print()
    
    df = make_df()
    app = TimeIntervalLabeler(
        df=df,
        plot_fn=plot_fn,
        layout_spec=layout_spec,
        window=pd.Timedelta("30min"),
        step=pd.Timedelta("15min"),
        classes=["PlasmaSheet", "Lobe", "Magnetosheath", "SolarWind", "UNKNOWN"],
    )
    app.run()
