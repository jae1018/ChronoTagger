"""
Multi-pane labeler example with an ion-flux spectrogram.

Demonstrates a richer ChronoTagger setup than the bundled wizard:
    * Two panes (`First Set`, `Second Set`) labelled in lockstep.
    * Pane 1 has a 32-channel ion energy-flux spectrogram (pcolormesh)
      alongside time-series and two cross-plot ("not-time") panels for
      spacecraft position in the GSE and SSE (Moon-centered) frames.
    * Pane 2 has a complementary set of derived quantities + the same
      cross-plots, so labels created in either tab stay consistent.
    * Four placeholder classes (``label_1`` ... ``label_4``) -- swap
      these for whatever your labelling task actually needs.

This script is the labeler-driver half of a real ARTEMIS / cislunar
plasma-classification workflow.  The DataFrame and layout convention
shown here -- column-keyed `axs` dict, `layout_spec` with
`role: time / not-time / labels` and `x_col` / `y_col` mappings for
cross-plot panels -- is the same surface the bundled quick-start wizard
auto-generates, so this example also doubles as a "what does the
generated layout look like in long form" reference.

Required dataframe columns
--------------------------
Time-series:
    n, P, SCPot, BX, BY, BZ, T, Beta, VX, VY, VZ
Position (Earth-centered, R_E):  X, Y, Z
Position (Moon-centered, R_L):   Xm, Ym, Zm
Spectrogram (32 ion channels):   C0, C1, ..., C31

Optional:
    ``geospacefronts`` -- if installed, the bow-shock and magnetopause
    boundaries from Chao (2002) and Shue (1998) are overlaid on the GSE
    cross-plot.  Skipped silently otherwise.

Run
---
The script looks for the parquet file at the path in the
``CHRONOTAGGER_EXAMPLE_DATA`` environment variable, or at
``examples/data/thb_peif_common_eflux.parquet`` relative to this file
if the variable is unset.  The data is NOT shipped with the repository.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Optional bow-shock / magnetopause overlay.  Install `geospacefronts`
# to enable; the example still runs without it.
try:
    from geospacefronts import ChaoBowShock, ShueMagnetopause
    _HAVE_GEOSPACEFRONTS = True
except ImportError:
    _HAVE_GEOSPACEFRONTS = False
    warnings.warn(
        "geospacefronts not installed -- bow-shock and magnetopause "
        "overlays disabled.  pip install geospacefronts to enable.",
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _add_bs_and_mp_to_ax(ax, x_min: float = -30.0) -> None:
    """Overlay the Chao bow shock and Shue magnetopause on a GSE x-y plot."""
    if not _HAVE_GEOSPACEFRONTS:
        return
    theta = np.linspace(0, 170, 241)
    mp = ShueMagnetopause(bz=-5.0, dp=2.0)
    bs = ChaoBowShock(bz=0.2, dp=2.0, mgs=6.0, beta=1.0)
    ax.plot(*mp.xy(theta, x_min=x_min).T, ls="solid", c="black")
    ax.plot(*bs.xy(theta, x_min=x_min).T, ls="dashed", c="black")


def _draw_position_cross_plots(axs, df: pd.DataFrame) -> None:
    """Draw the two shared cross-plot panels (Earth-frame and Moon-frame)."""
    scatter_kwargs = dict(s=1, marker=".")

    # GSE (Earth-centered)
    axs["xy_gse"].scatter(df["X"], df["Y"], **scatter_kwargs)
    xmin_gse = axs["xy_gse"].get_xlim()[0]
    _add_bs_and_mp_to_ax(axs["xy_gse"], x_min=xmin_gse - 10)
    axs["xy_gse"].set_xlabel("X (GSE - R_E)")
    axs["xy_gse"].set_ylabel("Y (GSE - R_E)")
    axs["xy_gse"].add_artist(patches.Circle((0, 0), 1, fill=False, edgecolor="black"))

    # SSE (Moon-centered)
    axs["xy_sse"].scatter(df["Xm"], df["Ym"], **scatter_kwargs)
    axs["xy_sse"].set_xlabel("X (SSE - R_L)")
    axs["xy_sse"].set_ylabel("Y (SSE - R_L)")
    xmin_sse = axs["xy_sse"].get_xlim()[0]
    if xmin_sse < 0:
        # Lunar wake boundary
        axs["xy_sse"].hlines(y=1, xmin=xmin_sse, xmax=0, ls="dashed", colors="black", lw=1.5)
        axs["xy_sse"].hlines(y=-1, xmin=xmin_sse, xmax=0, ls="dashed", colors="black", lw=1.5)
    axs["xy_sse"].add_artist(patches.Circle((0, 0), 1, fill=False, edgecolor="black"))


def first_pane_plot_fn(axs, df: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> None:
    """Pane 1: density, ion spectrogram, SCPot, pressure, B-field, positions."""
    ms = 1
    marker = "x"

    axs["n"].plot(df.index, np.log10(df["n"]), marker=marker, ms=ms)
    axs["n"].set_ylabel("log10 n")

    # 32-channel ion energy-flux spectrogram
    channel_cols = [f"C{i}" for i in range(32)]
    Z = df[channel_cols].to_numpy(dtype=float).T  # (channels, time)
    axs["spectrogram"].pcolormesh(
        df.index, np.arange(32), np.log10(Z + 1),
        shading="auto", cmap="viridis", vmin=0, vmax=9,
    )
    axs["spectrogram"].set_ylabel("E (keV)")

    axs["scpot"].plot(df.index, df["SCPot"], marker=marker, ms=ms)
    axs["scpot"].set_ylabel("SCPot (V)")

    axs["p"].plot(df.index, np.log10(df["P"]), marker=marker, ms=ms)
    axs["p"].set_ylabel("log10 P")

    for B_label in ("BX", "BY", "BZ"):
        axs["b"].plot(df.index, df[B_label], label=B_label, marker=marker, ms=ms)
    axs["b"].set_ylabel("B (nT)")
    axs["b"].legend(framealpha=0.25, loc="center right")

    _draw_position_cross_plots(axs, df)


def second_pane_plot_fn(axs, df: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> None:
    """Pane 2: beta, temperature, velocity, plus the same position cross-plots."""
    ms = 5
    marker = "x"

    axs["Beta"].plot(df.index, np.log10(df["Beta"]), marker=marker, ms=ms)
    axs["Beta"].set_ylabel("log10 beta")

    axs["T"].plot(df.index, np.log10(df["T"]), marker=marker, ms=ms)
    axs["T"].set_ylabel("log10 T (eV)")

    for V_label in ("VX", "VY", "VZ"):
        axs["veloc"].plot(df.index, df[V_label], label=V_label, marker=marker, ms=ms)
    axs["veloc"].set_ylabel("veloc (km/s)")
    axs["veloc"].legend(framealpha=0.25, loc="center right")

    _draw_position_cross_plots(axs, df)


# ---------------------------------------------------------------------------
# Data loading + filtering
# ---------------------------------------------------------------------------

R_EARTH_KM = 6371.0
R_MOON_KM = 1737.0


def _convert_position_units(df: pd.DataFrame) -> pd.DataFrame:
    """km -> R_E for GSE, km -> R_L for SSE."""
    df[["X", "Y", "Z"]] = df[["X", "Y", "Z"]] / R_EARTH_KM
    df[["Xm", "Ym", "Zm"]] = df[["Xm", "Ym", "Zm"]] / R_MOON_KM
    return df


def _filter_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Drop obvious spacecraft-charging / saturation outliers."""
    n_before = len(df)
    keep = (
        (df["SCPot"] >= 0)
        & (df["SCPot"] < 120)
        & (df["n"] > 0)
        & (df["n"] < 100)
    )
    df = df[keep]
    print(f"filter_basic: kept {len(df)}/{n_before} rows")
    return df


def _resolve_data_path() -> Path:
    """Look up the parquet path from env var, with a sensible default."""
    env = os.environ.get("CHRONOTAGGER_EXAMPLE_DATA")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).parent / "data" / "thb_peif_common_eflux.parquet"


def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.is_file():
        sys.exit(
            f"Spectrogram example data not found at:\n  {path}\n\n"
            "This dataset is not shipped with the repository.  Set the\n"
            "CHRONOTAGGER_EXAMPLE_DATA environment variable to point at\n"
            "your own parquet file with the required columns (see the\n"
            "module docstring), or drop a file at examples/data/."
        )
    df = pd.read_parquet(path)
    df = _convert_position_units(df)
    df = _filter_basic(df)
    return df


# ---------------------------------------------------------------------------
# Layout specs (matching the per-pane plot_fn keys above)
# ---------------------------------------------------------------------------

LAYOUT_FIRST = {
    "nrows": 6,
    "ncols": 2,
    "hspace": 0.05,
    "areas": [
        {"key": "n",           "row": 0, "col": 0, "role": "time"},
        {"key": "spectrogram", "row": 1, "col": 0, "role": "time"},
        {"key": "scpot",       "row": 2, "col": 0, "role": "time"},
        {"key": "p",           "row": 3, "col": 0, "role": "time"},
        {"key": "b",           "row": 4, "col": 0, "role": "time"},
        {"key": "labels",      "row": 5, "col": 0, "role": "labels"},
        # Cross-plot panels (column 1) reuse columns from the dataframe
        # via x_col / y_col mappings -- this is what powers the
        # box-select-on-cross-plot -> highlight-on-time-series flow.
        {"key": "xy_gse", "row": 0, "col": 1, "role": "not-time", "rowspan": 2,
         "x_col": "X", "y_col": "Y"},
        {"key": "xy_sse", "row": 2, "col": 1, "role": "not-time", "rowspan": 2,
         "x_col": "Xm", "y_col": "Ym"},
    ],
}

LAYOUT_SECOND = {
    "nrows": 4,
    "ncols": 2,
    "hspace": 0.05,
    "areas": [
        {"key": "Beta",   "row": 0, "col": 0, "role": "time"},
        {"key": "T",      "row": 1, "col": 0, "role": "time"},
        {"key": "veloc",  "row": 2, "col": 0, "role": "time"},
        {"key": "labels", "row": 3, "col": 0, "role": "labels"},
        {"key": "xy_gse", "row": 0, "col": 1, "role": "not-time", "rowspan": 2,
         "x_col": "X", "y_col": "Y"},
        {"key": "xy_sse", "row": 2, "col": 1, "role": "not-time", "rowspan": 2,
         "x_col": "Xm", "y_col": "Ym"},
    ],
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(data_path: Optional[Path] = None) -> None:
    from chronotagger.labeler import TimeIntervalLabeler

    if data_path is None:
        data_path = _resolve_data_path()
    df = _load_dataframe(data_path)

    panes = [
        {"title": "First Set",  "plot_fn": first_pane_plot_fn,  "layout_spec": LAYOUT_FIRST},
        {"title": "Second Set", "plot_fn": second_pane_plot_fn, "layout_spec": LAYOUT_SECOND},
    ]

    autosave_folder = Path(__file__).parent / "autosave"
    autosave_folder.mkdir(exist_ok=True)

    labeler = TimeIntervalLabeler(
        df=df,
        panes=panes,
        classes=["UNKNOWN", "label_1", "label_2", "label_3", "label_4"],
        window=pd.Timedelta("4h"),
        step=pd.Timedelta("30min"),
        autosave_folder=str(autosave_folder),
    )
    labeler.run()


if __name__ == "__main__":
    main()
