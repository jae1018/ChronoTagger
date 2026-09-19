"""Pack M2.7 -- THE ONE NON-ASCII CHARACTER IN A MESSAGE THE USER READS.

`src/chronotagger/labeler/mixins/intervals/crud.py` line 70 said

    Select a time range first (drag or click<U+00D7>2).

with a real U+00D7 multiplication sign -- two UTF-8 bytes, C3 97, in a
warning box anyone who presses Add with nothing selected reads. Pack
M2.6 deferred it (its DR8) for a mechanical reason: the house applier
asserts the pack document is pure ASCII, and a replacement can only
change bytes inside its own anchor, so no anchor may contain the byte
pair and no operation can delete it.

Pack M2.7 takes it as OP 612, the pack's one scripted binary operation.
THIS MODULE IS WHAT MAKES THAT OPERATION VISIBLE: without it, the pack
would add a change that nothing in the suite would notice being undone.
It is the LAST module the pack adds, and it goes RED on any tree where
OP 612 has not been run.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

DAY = "2015-01-03 "

HUMAN = {"id": "human", "name": "Human", "classes": ["sw", "msh", "UNKNOWN"],
         "class_colors": {"sw": "#4e79a7", "msh": "#f28e2b",
                          "UNKNOWN": "#7f7f7f"}, "order": 0}

LAYOUT = {
    "nrows": 2, "ncols": 1,
    "areas": [
        {"key": "panel1", "row": 0, "col": 0, "role": "time"},
        {"key": "labels", "row": 1, "col": 0, "role": "labels"},
    ],
}

WARNING = "Select a time range first (drag or click x2)."


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
def app(tmp_path):
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(
        df=frame(), plot_fn=plot_fn, layout_spec=dict(LAYOUT),
        window=pd.Timedelta("60min"), autosave_folder=str(tmp_path),
        tracks=[dict(HUMAN)])
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    yield lbl
    lbl.root.destroy()


def test_the_No_Selection_warning_is_pure_ASCII(app, monkeypatch):
    """The box the user actually reads, through the path he takes: press
    Add with nothing selected."""
    import tkinter.messagebox as mb
    said = []
    monkeypatch.setattr(mb, "showwarning",
                        lambda *a, **k: said.append(a) or True)
    app.current_selection = None
    app._commit_spans.clear()
    app._add_interval()

    assert len(said) == 1, said
    title, body = said[0][0], said[0][1]
    assert title == "No Selection"
    assert body == WARNING
    body.encode("ascii")          # raises if a non-ASCII byte came back


def test_the_source_file_carries_no_multiplication_sign_at_all():
    """crud.py held exactly ONE occurrence of C3 97 and OP 612 removes
    it. Its nine other non-ASCII lines are out of scope and use other
    characters, so the whole file is clean of this one."""
    import os
    import chronotagger.labeler.mixins.intervals.crud as crud
    raw = open(os.path.abspath(crud.__file__), "rb").read()
    assert raw.count(b"\xc3\x97") == 0
    assert b"click x2" in raw


def test_the_operation_is_idempotent_by_REFUSAL():
    """OP 612 counts the byte sequence and refuses unless it finds
    exactly one. Running it again on an applied tree must change
    nothing, which is what this pin holds from the file's side: the
    sequence it looks for is gone, so a second run has nothing to do."""
    import os
    import chronotagger.labeler.mixins.intervals.crud as crud
    raw = open(os.path.abspath(crud.__file__), "rb").read()
    assert raw.count(b"click\xc3\x972") == 0
    assert raw.count(b"click x2") == 1
