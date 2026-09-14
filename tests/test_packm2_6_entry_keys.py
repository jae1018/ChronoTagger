"""Pack M2.6 -- Ctrl+D / Ctrl+H / Ctrl+K STOP EDITING THE TIME BOXES.

THE DEFECT. The Start and End boxes are ttk Entries, and their bindtags
are (the widget, 'TEntry', '.', 'all'). The TEntry CLASS binds three
Control keys to editing scripts -- measured on this machine
(edit_pack/evidence/scratch/m2_orch/logs/probe_s3_ctrl_d_entry.txt):

    <Control-Key-d>   ttk::entry::Delete %W      delete the character
    <Control-Key-h>   ttk::entry::Backspace %W   delete the one before it
    <Control-Key-k>   %W delete insert end       delete to end of line

None of the three ends in `break`, so after editing the text the event
carried on to the root's <Key> handler as well. Ctrl+H in the Start box
therefore did BOTH: "2020-09-02 07:00:00" became "2020-09-02 7:00:00"
AND the active lane was hidden. Ctrl+D deleting a character is what
computer-use session 3 reported as "the Start box reverted to 00:00:00".

Pack M2's comment claimed "no Tk-level bind on root collides". True, and
the wrong question: the collision is a CLASS bind, not a root bind.

THE FIX. The three sequences are rebound app-wide, on the TEntry class,
to a Python no-op that returns None -- not "break". The text is never
edited, and because the callback does not break, the event still reaches
the root handler, so Ctrl+H goes on hiding the lane from inside a text
box, exactly once, which is what the Control-defined lane keys are for.

HOW THE KEYSTROKE IS DELIVERED HERE. `event_generate` delivers NOTHING
on a withdrawn window, so a pin that used it would be vacuous. These
pins evaluate the Tk CLASS SCRIPT itself through `root.tk.eval`, with
every % substitution filled in -- which is how
`probe_s3_ctrl_d_entry.py` measured the defect in the first place. The
last pin in the first section is the CONTROL: it evaluates the raw
`ttk::entry::Backspace` script by hand and asserts the box really does
change, so "unchanged" above means the override worked and not that the
delivery route is dead.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tkinter as tk

DAY = "2015-01-03 "

TEXT = "2020-09-02 07:00:00"
CURSOR = 12

SEQS = ("<Control-Key-d>", "<Control-Key-h>", "<Control-Key-k>")

# Every % substitution a Tk binding script can carry, with a value that
# tkinter's own dispatcher can parse back into an Event.
SUBS = (("%#", "0"), ("%b", "0"), ("%f", "0"), ("%h", "0"), ("%k", "0"),
        ("%s", "0"), ("%t", "0"), ("%w", "0"), ("%x", "0"), ("%y", "0"),
        ("%A", "x"), ("%E", "0"), ("%K", "h"), ("%N", "0"), ("%T", "2"),
        ("%X", "0"), ("%Y", "0"), ("%D", "0"))

TWO = [
    {"id": "human", "name": "Human", "classes": ["sw", "UNKNOWN"],
     "class_colors": {"sw": "#4e79a7", "UNKNOWN": "#7f7f7f"}, "order": 0},
    {"id": "model", "name": "Model", "classes": ["0", "1"],
     "class_colors": {"0": "#111111", "1": "#222222"}, "order": 1},
]

LAYOUT = {
    "nrows": 2, "ncols": 1,
    "areas": [
        {"key": "panel1", "row": 0, "col": 0, "role": "time"},
        {"key": "labels", "row": 1, "col": 0, "role": "labels"},
    ],
}


class Key(object):
    """A synthetic key event. state 0x4 is the Control modifier."""

    def __init__(self, keysym, state=0):
        self.keysym = keysym
        self.state = state
        self.char = ""


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
        tracks=[dict(t) for t in TWO])
    lbl._build_gui()
    lbl._update_plot()
    lbl.root.withdraw()
    yield lbl
    lbl.root.destroy()


def set_box(app):
    box = app.start_time_entry
    box.delete(0, "end")
    box.insert(0, TEXT)
    box.icursor(CURSOR)
    app.root.update_idletasks()
    return box


def fire_class_script(app, seq):
    """Run the TEntry class binding for `seq` against the Start box."""
    script = app.root.bind_class("TEntry", seq)
    assert script, seq
    script = script.replace("%W", str(app.start_time_entry))
    for token, value in SUBS:
        script = script.replace(token, value)
    app.root.tk.eval(script)


# ============================================== 1. the class bindings


@pytest.mark.parametrize("seq", SEQS)
def test_the_class_binding_is_no_longer_tks_editing_script(app, seq):
    script = app.root.bind_class("TEntry", seq)
    assert script, "the sequence must still be bound -- to OUR no-op"
    assert "ttk::entry::" not in script, script
    assert "delete" not in script, script


@pytest.mark.parametrize("seq", SEQS)
def test_firing_the_class_binding_leaves_the_text_byte_identical(app, seq):
    box = set_box(app)
    fire_class_script(app, seq)
    app.root.update_idletasks()
    assert box.get() == TEXT, seq


def test_the_delivery_route_really_can_edit_the_box(app):
    """THE CONTROL. Without this the pins above would pass on a route
    that does nothing at all. This is the exact script Tk had bound to
    Ctrl+H, run by hand."""
    box = set_box(app)
    app.root.tk.eval("ttk::entry::Backspace %s" % (box,))
    app.root.update_idletasks()
    assert box.get() == "2020-09-02 7:00:00", \
        "the raw ttk script still deletes -- so the pins above mean it"


# ============================================== 2. the app's own half


def test_ctrl_h_still_hides_the_active_lane_exactly_once(app):
    before = [(t.id, t.visible) for t in app.tracks]
    assert before == [("human", True), ("model", True)]
    app._on_key_press(Key("h", 0x4))
    after = [(t.id, t.visible) for t in app.tracks]
    assert after == [("human", False), ("model", True)]
    assert app.active_track_id == "model", "hiding moves the active lane"
    assert "hidden" in app.status_var.get()


def test_ctrl_h_through_the_class_script_and_the_handler_is_ONE_hide(app):
    """The live keystroke is both halves in a row: the class binding
    first (now a no-op), then the root handler. The box keeps its text
    and the lane is hidden once, not twice."""
    box = set_box(app)
    fire_class_script(app, "<Control-Key-h>")
    app._on_key_press(Key("h", 0x4))
    app.root.update_idletasks()
    assert box.get() == TEXT
    assert [t.visible for t in app.tracks] == [False, True]


def test_ctrl_d_and_ctrl_k_do_nothing_at_all(app):
    """Pack M2.5's `plain` guard already made these inert while Control
    is held; PART E takes away the other half, the text edit."""
    box = set_box(app)
    app.intervals[:] = []
    before = [(t.id, t.visible, t.locked) for t in app.tracks]
    for seq, keysym in (("<Control-Key-d>", "d"), ("<Control-Key-k>", "k")):
        fire_class_script(app, seq)
        app._on_key_press(Key(keysym, 0x4))
    app.root.update_idletasks()
    assert box.get() == TEXT
    assert app.intervals == []
    assert [(t.id, t.visible, t.locked) for t in app.tracks] == before


def test_ctrl_a_selects_all_and_is_untouched(app):
    script = app.root.bind_class("TEntry", "<<SelectAll>>")
    assert "selection range 0 end" in script, script
    box = set_box(app)
    app.root.tk.eval(script.replace("%W", str(box)))
    app.root.update_idletasks()
    assert box.get() == TEXT
    assert box.selection_present()


def test_the_override_returns_None_and_not_break(app):
    """THE CONTRACT, in one line. "break" would stop Tk walking the
    bindtags, so Ctrl+H would edit nothing AND no longer hide the lane --
    one defect traded for another. None keeps the app's own half alive."""
    assert app._entry_class_key_noop(None) is None
    assert app._entry_class_key_noop(Key("h", 0x4)) is None


def test_the_override_is_armed_by_the_build_not_by_the_test(app):
    """`_build_gui` is what calls `_neutralize_entry_class_keys`, so a
    freshly built window is already safe -- nothing else has to remember."""
    assert hasattr(app, "_neutralize_entry_class_keys")
    for seq in app.ENTRY_CLASS_KEYS:
        assert "ttk::entry::" not in app.root.bind_class("TEntry", seq)
