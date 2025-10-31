# src/chronotagger/labeler/utils/colorbar.py
from __future__ import annotations
from typing import Optional
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def ensure_lane_colorbar(owner_ax, mappable, *, label=None, tick_params=None, width_frac=0.5):
    """
    Create/replace a colorbar for `mappable` inside the time-lane gutter.

    width_frac ∈ (0,1]: fraction of the reserved gutter width to use for the bar.
    (The rest is left as breathing room for ticklabels/units.)
    """
    import matplotlib.pyplot as plt
    fig = owner_ax.figure
    lane_box = getattr(fig, "_chrono_lane_gutter", None)

    # clear previous
    for attr in ("_lane_cbar", "_lane_cbar_ax"):
        old = getattr(owner_ax, attr, None)
        if old is not None:
            try:
                old.remove()
            except Exception:
                pass
            setattr(owner_ax, attr, None)

    if not lane_box:
        # fallback if no gutter (shouldn't happen in lane mode)
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cax = inset_axes(owner_ax, width="3%", height="90%", loc="center right", borderpad=0.8)
        cb = fig.colorbar(mappable, cax=cax)
        if label:
            cb.set_label(label)
        if tick_params:
            cb.ax.tick_params(**tick_params)
        owner_ax._lane_cbar, owner_ax._lane_cbar_ax = cb, cax
        return cb

    # figure coords for gutter
    gx = float(lane_box["x"])
    gw = float(lane_box["w"])
    inner = 0.002  # tiny padding inside gutter
    gx += inner
    gw = max(0.005, gw - 2*inner)

    # thickness control: use only a fraction of gutter width, right-aligned
    width_frac = max(0.1, min(1.0, float(width_frac)))
    bar_w = gw * width_frac
    x_left = gx + (gw - bar_w)  # right align bar, leaves room for ticks/label to the left

    ap = owner_ax.get_position()
    y, h = ap.y0, ap.height
    cax = fig.add_axes([x_left, y, bar_w, h])

    cb = fig.colorbar(mappable, cax=cax)
    if label:
        cb.set_label(label)
    if tick_params:
        cb.ax.tick_params(**tick_params)

    owner_ax._lane_cbar = cb
    owner_ax._lane_cbar_ax = cax

    # keep aligned with owner on resize/redraw
    def _sync(_evt):
        ap2 = owner_ax.get_position()
        cax.set_position([x_left, ap2.y0, bar_w, ap2.height])

    if getattr(owner_ax, "_lane_cbar_sync_cid", None) is None:
        owner_ax._lane_cbar_sync_cid = fig.canvas.mpl_connect("draw_event", _sync)

    return cb