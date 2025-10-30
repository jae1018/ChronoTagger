# Always use a headless backend for tests
import matplotlib
matplotlib.use("Agg")

import pytest
import pandas as pd
import numpy as np
import tkinter as tk


@pytest.fixture
def df_hour():
    """120 samples every 30s."""
    idx = pd.date_range("2015-01-03 00:00:00", periods=120, freq="30s")
    return pd.DataFrame(
        {
            "log10n": np.linspace(0.5, 2.0, len(idx)),
            "BX": np.sin(np.linspace(0, 30, len(idx))) * 10,
            "BY": np.cos(np.linspace(0, 30, len(idx))) * 5,
            "BZ": np.linspace(-7, 3, len(idx)),
        },
        index=idx,
    )


@pytest.fixture
def plot_fn():
    def fn(axs, df, t0, t1):
        axs["panel1"].plot(df.index, df["log10n"])
        axs["panel1"].set_ylabel("log10(n) [cm^-3]")
        axs["panel2"].plot(df.index, df["BX"], label="B_x")
        axs["panel2"].plot(df.index, df["BY"], label="B_y")
        axs["panel2"].plot(df.index, df["BZ"], label="B_z")
        axs["panel2"].legend(loc="upper right")
    return fn


@pytest.fixture
def labeler(df_hour, plot_fn):
    """Create a labeler with GUI built but window withdrawn (no popup)."""
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(df=df_hour, plot_fn=plot_fn, window=pd.Timedelta("30min"))
    lbl._build_gui()           # build axes & widgets
    lbl.root.withdraw()        # keep window hidden on CI/Windows
    yield lbl
    lbl.root.destroy()
