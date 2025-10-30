"""
Time-axis formatting utilities (Matplotlib).
"""

from __future__ import annotations
import matplotlib.dates as mdates


def apply_time_axis_format(ax) -> None:
    """
    Ensure the x-axis is treated and formatted as dates (no scientific offsets).

    Uses AutoDateLocator + ConciseDateFormatter (adapts nicely while panning/zooming).
    IMPORTANT: do NOT call ticklabel_format(...); that installs ScalarFormatter and
    destroys the date formatting.
    """
    if ax is None:
        return
    ax.xaxis_date()
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
