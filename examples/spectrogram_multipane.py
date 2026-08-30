"""
Multi-pane labeler example with an ion-flux spectrogram.

Demonstrates a richer ChronoTagger setup than the bundled wizard:
    * Two panes (`First Set`, `Second Set`) labelled in lockstep.
    * Pane 1 has a 32-channel ion energy-flux spectrogram -- drawn with
      the Pack 8.5 image helpers, on a real energy axis, with a colorbar
      that survives the redraw loop -- alongside time-series and two
      cross-plot ("not-time") panels for spacecraft position in the GSE
      and SSE (Moon-centered) frames.
    * Pane 1's layout also reserves an empty gridspec column as a
      COLORBAR GUTTER (`ncols` + `width_ratios`, no area in column 1),
      which is what makes the eflux bar span its own panel instead of
      the whole stack.
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
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedFormatter, FixedLocator

from chronotagger.labeler.utils import attach_colorbar, draw_spectrogram

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
# Spectrogram conventions
# ---------------------------------------------------------------------------

#: Channel centres for C0..C31, in eV.  These are the `plasma_E`
#: coordinate of an ARTEMIS/THEMIS-B survey product (thb, 2011-08-14):
#: a UNIFORM LOG GRID the product was regridded onto (1 eV .. 30 keV,
#: geometric to 5e-8), NOT the ESA instrument's native sweep, whose
#: reduced-mode ion range is narrower and not exactly geometric.  It is
#: here because it is the right SHAPE for a demo energy axis -- do not
#: cite it as an instrument table.  Thinned from its 56
#: native channels to the 32 this example carries by
#: ``round(linspace(0, 55, 32))``, so the full 1 eV .. 30 keV range is
#: spanned.  If your own file ships its energies, drop them in here --
#: they are what puts the image at the right height on the axis.
CHANNEL_ENERGIES_EV = (
    1.0, 1.4548, 2.1165, 2.5528, 3.7138, 5.4028, 7.8600, 9.4804,
    13.7921, 20.0648, 29.1904, 42.4663, 51.2208, 74.5162, 108.4064,
    157.7101, 190.2224, 276.7363, 402.5970, 585.6996, 706.4431,
    1027.7360, 1495.1542, 2175.1560, 3164.4250, 3816.7788, 5552.6641,
    8078.0366, 11751.9580, 14174.6533, 20621.3379, 30000.0,
)

#: The house energy-flux colour convention: LogNorm from 1e3 to 1e8 with
#: `jet`.  Everything about a spectrogram people argue over is in these
#: two lines, so they are named rather than buried in the call.
EFLUX_NORM = LogNorm(vmin=1e3, vmax=1e8)
EFLUX_CMAP = "jet"

#: Decades to tick the energy axis at.  The axis carries log10(E) on a
#: LINEAR scale ON PURPOSE: `imshow` maps its extent through the axes
#: transform by the corners only, so a genuine log scale would stretch
#: the image linearly and put every channel at the wrong height.  Ticking
#: log10 values with energy labels gives the reader a log energy axis and
#: keeps the picture correct.
ENERGY_TICKS_EV = (1e0, 1e1, 1e2, 1e3, 1e4)


def _format_energy_axis(ax) -> None:
    """Label a log10(E) axis with the energies themselves."""
    ax.yaxis.set_major_locator(
        FixedLocator([np.log10(e) for e in ENERGY_TICKS_EV]))
    ax.yaxis.set_major_formatter(
        FixedFormatter(["%g" % e for e in ENERGY_TICKS_EV]))
    ax.set_ylabel("E (eV)")


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

    # 32-channel ion energy-flux spectrogram.
    #
    # draw_spectrogram rebins the window onto the panel's pixel columns
    # (default aggregator "max" -- a burst narrower than a pixel is drawn
    # at its true magnitude, at the cost of about +0.30 dex of overall
    # brightness; pass aggregator="logmean" for an unbiased level) and
    # draws the result with imshow, which is exact on the uniform grid
    # the rebin manufactures and costs a few ms regardless of how wide
    # the window is.
    channel_cols = [f"C{i}" for i in range(32)]
    Z = df[channel_cols].to_numpy(dtype=float).T  # (channels, time)
    ax_spec = axs["spectrogram"]
    im = draw_spectrogram(
        ax_spec, df.index, Z,
        y_centers=np.log10(CHANNEL_ENERGIES_EV),
        norm=EFLUX_NORM, cmap=EFLUX_CMAP,
    )
    _format_energy_axis(ax_spec)
    # Idempotent: creates the bar on the first redraw and re-points it on
    # every later one.  LAYOUT_FIRST reserves an empty gridspec column
    # beside this panel, so the bar is built there and spans exactly this
    # panel -- not the whole stack.  _update_plot's sweep keeps it pointed
    # at the live image without any per-frame code here.
    attach_colorbar(ax_spec, im, label="ion eflux (eV/cm^2-s-sr-eV)")

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
    # THREE columns, and the middle one is empty on purpose: it is the
    # COLORBAR GUTTER.  `attach_colorbar` looks for the first gridspec
    # column to the right of the panel it belongs to that no area
    # occupies, and builds the bar there as a real subplot spanning
    # exactly that panel's rows -- so the bar marks the spectrogram and
    # nothing else (measured bar height / panel height = 1.000, stable
    # across redraws, a pan and a window resize).  Without a gutter the
    # helper falls back to taking width off the shared-x group, which
    # keeps the x axes aligned but lays every bar out against the whole
    # column of panels: 2.1x its own panel's height with one
    # spectrogram, 4.7x with three.
    #
    # The width is now STATED here rather than taken from the panels, and
    # one gutter column pays for any number of bars.  0.07 of a content
    # column is the value that reproduces the bar width this example used
    # to get from the steal (0.0279 of the figure against 0.0281),
    # measured; wider or narrower is a taste knob and costs the panels
    # about 1 % of their width per 0.05 of ratio.
    "ncols": 3,
    "width_ratios": [1.0, 0.07, 1.0],
    "hspace": 0.05,
    "areas": [
        {"key": "n",           "row": 0, "col": 0, "role": "time"},
        {"key": "spectrogram", "row": 1, "col": 0, "role": "time"},
        {"key": "scpot",       "row": 2, "col": 0, "role": "time"},
        {"key": "p",           "row": 3, "col": 0, "role": "time"},
        {"key": "b",           "row": 4, "col": 0, "role": "time"},
        {"key": "labels",      "row": 5, "col": 0, "role": "labels"},
        # Cross-plot panels (column 2 -- column 1 is the gutter) reuse
        # columns from the dataframe via x_col / y_col mappings -- this is
        # what powers the box-select-on-cross-plot ->
        # highlight-on-time-series flow.
        {"key": "xy_gse", "row": 0, "col": 2, "role": "not-time", "rowspan": 2,
         "x_col": "X", "y_col": "Y"},
        {"key": "xy_sse", "row": 2, "col": 2, "role": "not-time", "rowspan": 2,
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

def main(data_path: Optional[Path] = None, run: bool = True):
    """Build the two-pane labeler and (by default) run it.

    ``run=False`` returns the constructed labeler without entering the
    Tk main loop, which is how the smoke test drives this file headlessly
    -- an example nobody can execute is an example nobody can trust.
    """
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
    if run:
        labeler.run()
    return labeler


if __name__ == "__main__":
    main()
