# examples/mixed_layout.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from chronotagger.labeler import TimeIntervalLabeler
from chronotagger.labeler.utils.colorbar import ensure_lane_colorbar

def make_df():
    idx = pd.date_range("2015-01-03 00:00:00", periods=720, freq="5s")
    t = np.linspace(0, 6*np.pi, len(idx))
    # time series
    log10n = 0.3 + 0.1*np.sin(t) + 0.02*np.random.randn(len(idx))
    bx = 8*np.sin(t)
    by = 6*np.cos(t)
    bz = -2 + 2*np.sin(0.7*t)
    # simple “spectrogram-like” data: E bins x time
    e_bins = np.geomspace(1e-1, 1e2, 48)
    spec = np.exp(-(np.log10(e_bins)[:, None]-0.6*np.sin(0.3*t)[None,:])**2/0.3) * (0.6+0.4*np.random.rand(len(e_bins), len(idx)))
    df = pd.DataFrame({"log10n":log10n, "BX":bx, "BY":by, "BZ":bz}, index=idx)
    df.attrs["E"] = e_bins
    df.attrs["SPEC"] = spec
    # fake XY cloud
    theta = np.linspace(0, 2*np.pi, 1200)
    x = 6*np.cos(theta) + 1.2*np.random.randn(theta.size)
    y = 3*np.sin(theta) + 1.2*np.random.randn(theta.size)
    df.attrs["XY1"] = (x,y)
    df.attrs["XY2"] = (x*0.8+2.0, -y*0.8)
    return df

layout_spec = {
    "nrows": 4, "ncols": 2,
    #"height_ratios": [1.2, 1.6, 1.2, 1.2],
    #"width_ratios": [2.2, 1.6],
    #"hspace": 0.20, "wspace": 0.10,
    "areas": [
        {"key":"n",     "row":0, "col":0, "role":"time"},
        {"key":"b",     "row":1, "col":0, "role":"time"},
        {"key":"spec",  "row":2, "col":0, "role":"time"},
        # right column = XY
        {"key":"xy_top",    "row":0, "col":1, "role":"xy", "rowspan":2},
        {"key":"xy_bottom", "row":2, "col":1, "role":"xy", "rowspan":2},
    ],
    # Reserve a gutter on the right of the time lane for colorbars
    #"time_lane_cbar_gutter": {"col": 0, "size": "7%", "pad": "2%"},
}

def plot_fn(axs, df, t0, t1):
    # time panels
    axs["n"].plot(df.index, df["log10n"]); axs["n"].set_ylabel("log10 n")
    axs["b"].plot(df.index, df["BX"], label="B_x")
    axs["b"].plot(df.index, df["BY"], label="B_y")
    axs["b"].plot(df.index, df["BZ"], label="B_z")
    axs["b"].set_ylabel("B [nT]"); axs["b"].legend(loc="upper right")

    # spectrogram in time lane + colorbar in the lane gutter
    e = df.attrs["E"]; S = df.attrs["SPEC"]
    # clip to window for plotting speed
    sub = df.loc[t0:t1]
    j0, j1 = df.index.get_indexer([sub.index[0], sub.index[-1]], method="nearest")
    im = axs["spec"].pcolormesh(df.index[j0:j1+1], e, S[:, j0:j1+1], shading="auto")
    axs["spec"].set_yscale("log"); axs["spec"].set_ylabel("E (keV)")
    #cb = ensure_lane_colorbar(axs["spec"], im, label="flux", tick_params={"labelsize":8}, width_frac=0.6)

    # XY panels
    x1,y1 = df.attrs["XY1"]; x2,y2 = df.attrs["XY2"]
    axs["xy_top"].scatter(x1, y1, s=6, alpha=0.6)
    axs["xy_top"].set_xlabel("Y (R_E)"); axs["xy_top"].set_ylabel("Z (R_E)")
    axs["xy_bottom"].scatter(x2, y2, s=6, alpha=0.6)
    axs["xy_bottom"].set_xlabel("Y (R_E)"); axs["xy_bottom"].set_ylabel("Z (R_E)")

if __name__ == "__main__":
    df = make_df()
    app = TimeIntervalLabeler(
        df=df,
        plot_fn=plot_fn,
        layout_spec=layout_spec,     # grid mode
        window=pd.Timedelta("30min"),
        step=pd.Timedelta("15min"),
    )
    app.run()
