"""Pack M2.7 -- AN 'Export Intervals...' BUTTON, WITH THE GATE ITS TWIN
HAS.

The intervals file -- header `start,end,track,label,notes`, one row per
interval, EVERY lane -- has existed since Pack M1 and was reachable only
by Ctrl+E. The I/O grid is 2x2 and its fourth cell was already there,
marked "intentionally left empty". The button goes in it.

And the path behind it had no orphan gate. `export_intervals` (the
programmatic twin) calls `refuse_export_orphans`; `_export_intervals` did
not, so the new button would have been the one export control in the
window with no orphan check behind it, sitting beside "Export Labels..."
which has one. Same gate, GUI channel: a message instead of a traceback,
and no file.

WHAT THE BUTTON DOES NOT DO, said out loud: it ignores every lane
control. That is the contract -- the `track` column exists precisely so
one file can carry every lane -- and it is what the tooltip says.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tkinter as tk
from tkinter import ttk

from chronotagger.core.models import Interval

DAY = "2015-01-03 "

HUMAN = {"id": "human", "name": "Human", "classes": ["sw", "msh", "UNKNOWN"],
         "class_colors": {"sw": "#4e79a7", "msh": "#f28e2b",
                          "UNKNOWN": "#7f7f7f"}, "order": 0}
MODEL = {"id": "model", "name": "Model", "classes": ["0", "1"],
         "class_colors": {"0": "#111111", "1": "#222222"}, "order": 1}

LAYOUT = {
    "nrows": 2, "ncols": 1,
    "areas": [
        {"key": "panel1", "row": 0, "col": 0, "role": "time"},
        {"key": "labels", "row": 1, "col": 0, "role": "labels"},
    ],
}

BUTTON = "Export Intervals..."


def ts(hhmmss):
    return pd.Timestamp(DAY + hhmmss)


def frame():
    idx = pd.date_range(DAY + "00:00:00", periods=120, freq="30s")
    return pd.DataFrame({"a": np.linspace(0.0, 1.0, len(idx))}, index=idx)


def plot_fn(axs, df, t0, t1):
    axs["panel1"].plot(df.index, df["a"])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def dialogs(monkeypatch):
    import tkinter.messagebox as mb
    calls = []
    for kind in ("showinfo", "showwarning", "showerror", "askyesno",
                 "askyesnocancel", "askokcancel"):
        monkeypatch.setattr(
            mb, kind,
            lambda *a, _k=kind, **kw: calls.append((_k, a[0] if a else ""))
            or True)
    return calls


@pytest.fixture
def app(tmp_path):
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(
        df=frame(), plot_fn=plot_fn, layout_spec=dict(LAYOUT),
        window=pd.Timedelta("60min"), autosave_folder=str(tmp_path),
        tracks=[dict(HUMAN), dict(MODEL)])
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    lbl.intervals[:] = [
        Interval(ts("00:05:00"), ts("00:08:00"), "0", None, track="model"),
        Interval(ts("00:20:00"), ts("00:25:00"), "sw", None, track="human")]
    yield lbl
    lbl.root.destroy()


def widgets(w, out):
    for ch in w.winfo_children():
        out.append(ch)
        widgets(ch, out)
    return out


def io_group(app):
    for w in widgets(app.root, []):
        try:
            if isinstance(w, ttk.LabelFrame) and w.cget("text") == "I/O":
                return w
        except Exception:
            continue
    raise AssertionError("no I/O group in the window")


def buttons_of(w):
    out = []
    for ch in widgets(w, []):
        if isinstance(ch, ttk.Button) or getattr(ch, "winfo_class",
                                                 lambda: "")() == "TButton":
            try:
                out.append((str(ch.cget("text")), ch))
            except Exception:
                pass
    return out


# ================================================ 1. the button is there


def test_the_button_is_in_the_IO_group(app):
    texts = [t for t, _ in buttons_of(io_group(app))]
    assert BUTTON in texts
    assert "Export Labels..." in texts
    assert "Save Session" in texts and "Load Session" in texts
    assert len(texts) == 4, texts


def test_the_button_carries_its_tooltip(app):
    btn = [b for t, b in buttons_of(io_group(app)) if t == BUTTON][0]
    assert getattr(btn, "tooltip_text", None) == \
        "Write one row per interval, every lane (Ctrl+E)"


def test_pressing_it_runs_the_intervals_export_and_nothing_else(
        app, monkeypatch, dialogs, tmp_path):
    import tkinter.filedialog as fd
    asked = []
    monkeypatch.setattr(fd, "asksaveasfilename",
                        lambda *a, **k: asked.append("ask") or "")
    btn = [b for t, b in buttons_of(io_group(app)) if t == BUTTON][0]
    btn.invoke()
    assert asked == ["ask"], "exactly one dialog"
    assert dialogs == [], "a cancelled save is silent, and writes nothing"
    assert list(tmp_path.glob("*.csv")) == []
    assert [w for w in app.root.winfo_children()
            if isinstance(w, tk.Toplevel)] == []


def test_the_button_writes_every_lane(app, monkeypatch, dialogs, tmp_path):
    import tkinter.filedialog as fd
    out = tmp_path / "ivs.csv"
    monkeypatch.setattr(fd, "asksaveasfilename", lambda *a, **k: str(out))
    btn = [b for t, b in buttons_of(io_group(app)) if t == BUTTON][0]
    btn.invoke()
    written = pd.read_csv(out)
    assert list(written.columns) == ["start", "end", "track", "label",
                                     "notes"]
    assert sorted(written["track"]) == ["human", "model"]
    assert app.status_var.get() == "Exported to %s" % (out,)


# ====================================== 2. the gate the twin already had


def test_the_orphan_gate_refuses_exactly_where_the_twin_refuses(
        app, monkeypatch, dialogs, tmp_path):
    app.intervals[:] = [
        Interval(ts("00:05:00"), ts("00:08:00"), "GHOST", None,
                 track="human")]

    # the programmatic twin: a raise, and no file
    twin = tmp_path / "twin.csv"
    with pytest.raises(ValueError) as exc:
        app.export_intervals(str(twin))
    assert "GHOST" in str(exc.value)
    assert not twin.exists()

    # the GUI path: the same refusal, as a message, and it never even
    # opens the save dialog
    import tkinter.filedialog as fd
    asked = []
    monkeypatch.setattr(
        fd, "asksaveasfilename",
        lambda *a, **k: asked.append("ask") or str(tmp_path / "gui.csv"))
    dialogs[:] = []
    app._export_intervals()

    assert asked == []
    assert [d[0] for d in dialogs] == ["showerror"]
    assert dialogs[0][1] == "Export Blocked"
    assert not (tmp_path / "gui.csv").exists()
    assert app.status_var.get().startswith("Intervals export refused")


def test_a_clean_session_is_not_refused(app, monkeypatch, dialogs, tmp_path):
    """The control: the gate only fires on an orphan."""
    import tkinter.filedialog as fd
    out = tmp_path / "clean.csv"
    monkeypatch.setattr(fd, "asksaveasfilename", lambda *a, **k: str(out))
    app._export_intervals()
    assert [d[0] for d in dialogs] == ["showinfo"], "Export Complete"
    assert out.exists()
