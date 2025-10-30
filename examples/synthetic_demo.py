"""
Synthetic demo for ChronoTagger.

Run:
    python examples/synthetic_demo.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from chronotagger import TimeIntervalLabeler


def my_plot_function(axs, df: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> None:
    """
    Example user-defined plot function.

    Parameters
    ----------
    axs : dict[str, matplotlib.axes.Axes]
        Dictionary of axes. Provided keys: 'panel1', 'panel2'.
    df : pd.DataFrame
        Windowed data in [t0, t1].
    t0, t1 : pd.Timestamp
        Window start/end.
    """
    # Panel 1: log10 density
    axs["panel1"].plot(df.index, df["log10n"], linewidth=1)
    axs["panel1"].set_ylabel("log10(n) [cm^-3]", fontsize=10)
    axs["panel1"].grid(True, alpha=0.3)
    axs["panel1"].set_title(f"Plasma Density: {t0.strftime('%Y-%m-%d %H:%M')} → {t1.strftime('%H:%M')}")

    # Panel 2: magnetic field components
    axs["panel2"].plot(df.index, df["BX"], label="B_x", linewidth=1)
    axs["panel2"].plot(df.index, df["BY"], label="B_y", linewidth=1)
    axs["panel2"].plot(df.index, df["BZ"], label="B_z", linewidth=1)
    axs["panel2"].set_ylabel("B [nT]", fontsize=10)
    axs["panel2"].legend(loc="upper right", framealpha=0.5, fontsize=8)
    axs["panel2"].grid(True, alpha=0.3)
    axs["panel2"].axhline(y=0, linestyle="--", linewidth=0.5, alpha=0.5)


def generate_synthetic_data() -> pd.DataFrame:
    """
    Generate synthetic, slightly irregular time-series data resembling spacecraft data.

    Returns
    -------
    pd.DataFrame with DatetimeIndex and columns: log10n, BX, BY, BZ
    """
    # ~37-second cadence with randomized gaps
    base_times = pd.date_range("2015-01-03", "2015-01-07", freq="37s")
    rng = np.random.default_rng(42)
    keep = np.ones(len(base_times), dtype=bool)
    for _ in range(10):
        start = rng.integers(0, len(base_times) - 100)
        length = rng.integers(50, 100)
        keep[start : start + length] = False
    times = base_times[keep]

    df = pd.DataFrame(index=times)

    t = np.arange(len(df))
    base_density = 0.5 + 0.3 * np.sin(2 * np.pi * t / 3000)
    region_signal = 2 * np.sin(2 * np.pi * t / 8000)
    df["log10n"] = base_density + region_signal + 0.1 * rng.normal(size=len(df))

    df["BX"] = 10 + 5 * np.sin(2 * np.pi * t / 5000) + 2 * rng.normal(size=len(df))
    df["BY"] = 3 * np.cos(2 * np.pi * t / 7000) + 1.5 * rng.normal(size=len(df))
    df["BZ"] = -5 + 8 * np.sin(2 * np.pi * t / 4000) + 2 * rng.normal(size=len(df))

    return df


def main() -> None:
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    print(f"Created dataframe with {len(df)} samples")
    print(f"Time range: {df.index[0]} → {df.index[-1]}")
    print(f"Columns: {list(df.columns)}")

    classes = ["PlasmaSheet", "Lobe", "Magnetosheath", "SolarWind", "UNKNOWN"]
    colors = {
        "PlasmaSheet": "#4e79a7",
        "Lobe": "#f28e2b",
        "Magnetosheath": "#e15759",
        "SolarWind": "#76b7b2",
        "UNKNOWN": "#bab0ac",
    }

    print("\nLaunching ChronoTagger...")
    print("Shortcuts: 1-9 class | n/p/←/→ nav | a add | d delete | Ctrl+S save | Ctrl+E export | "
          "Ctrl+Z or Backspace undo | Ctrl+Y or Shift+Backspace redo")

    labeler = TimeIntervalLabeler(
        df=df,
        plot_fn=my_plot_function,
        classes=classes,
        class_colors=colors,
        window=pd.Timedelta("60min"),
        step=pd.Timedelta("30min"),
        autosave_path="session_autosave.json",
    )
    labeler.run()
    print("Closed. Bye!")


if __name__ == "__main__":
    main()
