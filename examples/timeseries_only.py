# examples/timeseries_only.py
import numpy as np
import pandas as pd
from chronotagger.labeler import TimeIntervalLabeler

def make_df():
    idx = pd.date_range("2015-01-03 00:00:00", periods=720, freq="5s")
    t = np.linspace(0, 6*np.pi, len(idx))
    df = pd.DataFrame(
        {
            "log10n": 0.3 + 0.1*np.sin(t) + 0.02*np.random.randn(len(idx)),
            "BX": 8*np.sin(t) + 0.8*np.random.randn(len(idx)),
            "BY": 6*np.cos(t*0.9),
            "BZ": -3 + 2*np.sin(t*0.7),
        },
        index=idx,
    )
    return df

def plot_fn(axs, df, t0, t1):
    # Axes keys match layout_spec["areas"][*]["key"]
    ax_top = axs["top"]
    ax_bottom = axs["bottom"]

    ax_top.plot(df.index, df["log10n"])
    ax_top.set_ylabel("log10 n")

    ax_bottom.plot(df.index, df["BX"], label="B_x")
    ax_bottom.plot(df.index, df["BY"], label="B_y")
    ax_bottom.plot(df.index, df["BZ"], label="B_z")
    ax_bottom.set_ylabel("B [nT]")
    ax_bottom.legend(loc="upper right")

if __name__ == "__main__":
    df = make_df()

    # Grid-only layout: at least one role='time' axis in column 0.
    layout_spec = {
        "nrows": 2,
        "ncols": 1,
        #"height_ratios": [3.0, 3.0],  # optional
        "areas": [
            {"key": "top",    "row": 0, "col": 0, "role": "time"},
            {"key": "bottom", "row": 1, "col": 0, "role": "time"},
        ],
    }

    app = TimeIntervalLabeler(
        df=df,
        plot_fn=plot_fn,
        layout_spec=layout_spec,       # <= grid-only (no n_panels)
        window=pd.Timedelta("30min"),
        step=pd.Timedelta("15min"),
    )
    app.run()
