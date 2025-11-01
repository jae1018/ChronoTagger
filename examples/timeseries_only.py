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
    axs["panel1"].plot(df.index, df["log10n"])
    axs["panel1"].set_ylabel("log10 n")
    axs["panel2"].plot(df.index, df["BX"], label="B_x")
    axs["panel2"].plot(df.index, df["BY"], label="B_y")
    axs["panel2"].plot(df.index, df["BZ"], label="B_z")
    axs["panel2"].set_ylabel("B [nT]")
    axs["panel2"].legend(loc="upper right")

if __name__ == "__main__":
    df = make_df()
    app = TimeIntervalLabeler(
        df=df,
        plot_fn=plot_fn,
        n_panels=2,  # legacy simple mode
        window=pd.Timedelta("30min"),
        step=pd.Timedelta("15min"),
    )
    app.run()
