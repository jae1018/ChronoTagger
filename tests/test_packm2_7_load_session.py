"""Pack M2.7 -- LOAD SESSION CLOSES THE WAY A LANE SWITCH DOES.

Four defects, all of them in the last twenty lines of `_load_session`,
all of them measured on the feel-test session (probe_s4_save_load).

  (a) A STAGED RULE SURVIVED THE LOAD. `_commit_spans` and the yellow in
      `current_spans` were carved against the session being replaced, so
      one press of Add after a load committed the PREVIOUS session's
      geometry onto the loaded one -- and the Replace branch wrote three
      UNKNOWN intervals over the loaded lane and destroyed its interval.
      This is the defect class Pack M2.6 closed for a lane SWITCH.
  (b) A SESSION FILE HOLDING AN INTERVAL ON A LANE ITS OWN TABLE LACKS
      raised a bare ValueError -- a traceback in the terminal -- while
      the RECOVERY path caught the identical error and made it a message.
  (c) A SAVED active_track THAT NAMES NO LANE fell back to `tracks[0]`,
      which can be a HIDDEN row: a state the GUI itself refuses to reach.
      It now resolves the way Pack M2.6's `resolve_active_row` does, to
      the first VISIBLE lane, and it says so.
  (d) THE LOAD WROTE NO AUTOSAVE and set modified=False, so a prompt-free
      close right afterwards left an autosave describing the DISCARDED
      pre-load state -- offered at the next launch with Recover as the
      focused default button.

And ONE SENTENCE carries all of it, because `_load_session`'s closing
`status_var.set` is the last write on the path and erases anything set
before it.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

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


class _Var(object):
    """The status bar of a GUI-free host."""

    def __init__(self):
        self._v = ""

    def set(self, v):
        self._v = str(v)

    def get(self):
        return self._v


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
    """A GLOBAL messagebox stub that RECORDS (kind, title)."""
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
    yield lbl
    lbl.root.destroy()


def headless(tmp_path):
    """A labeler with NO GUI: `root` is None, which is the channel
    app.load(path) and tests/test_persistence_safety.py drive."""
    from chronotagger.labeler import TimeIntervalLabeler
    lbl = TimeIntervalLabeler(
        df=frame(), plot_fn=plot_fn, layout_spec=dict(LAYOUT),
        window=pd.Timedelta("60min"), autosave_folder=str(tmp_path),
        tracks=[dict(HUMAN), dict(MODEL)])
    lbl.status_var = _Var()
    lbl._update_plot = lambda: None
    assert lbl.root is None
    return lbl


def stage_a_rule(app):
    from chronotagger.labeler.dialogs.label_by_rule import (
        LabelByRuleResult, RuleCondition)
    app._rule_preview_apply(LabelByRuleResult(
        conditions=[RuleCondition(column="a", op=">=", value=0.0)],
        combine_mode="AND", nan_as_true=False, overlap_policy="skip",
        scope="window"))


def patch_session(path, **changes):
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


# ================================= (a) a staged rule does not survive


def test_a_load_drops_a_staged_rule_and_says_so_in_ONE_sentence(app,
                                                                tmp_path):
    sess = tmp_path / "s.json"
    app.intervals[:] = [
        Interval(ts("00:20:00"), ts("00:25:00"), "sw", None, track="human")]
    assert app._save_session(str(sess)) is True

    stage_a_rule(app)
    assert list(app._commit_spans), "something really is staged"

    app._load_session(str(sess))
    assert list(app._commit_spans) == []
    assert list(app.current_spans) == []
    assert app.status_var.get() == (
        "Loaded from %s -- rule preview cleared" % (sess,))


def test_one_Add_after_a_load_commits_nothing_from_the_old_session(
        app, tmp_path, dialogs):
    sess = tmp_path / "s.json"
    app.intervals[:] = [
        Interval(ts("00:20:00"), ts("00:25:00"), "sw", None, track="human")]
    app._save_session(str(sess))

    stage_a_rule(app)
    app._load_session(str(sess))

    before = [(iv.start, iv.end, iv.label, iv.track) for iv in app.intervals]
    app._add_interval()
    assert [(iv.start, iv.end, iv.label, iv.track)
            for iv in app.intervals] == before
    assert [d[0] for d in dialogs] == ["showwarning"], \
        "there is nothing to add, and it says so"


def test_a_plain_load_still_says_exactly_one_thing(app, tmp_path):
    """The control. Nothing staged, a real active lane, no stray track."""
    sess = tmp_path / "s.json"
    app._save_session(str(sess))
    app._load_session(str(sess))
    assert app.status_var.get() == "Loaded from %s" % (sess,)


# ===================== (b) a refused load is a message, not a traceback


def _stray_track_session(lbl, path):
    lbl.intervals[:] = [
        Interval(ts("00:05:00"), ts("00:08:00"), "0", None, track="model")]
    lbl._save_session(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["tracks"] = [t for t in data["tracks"] if t["id"] != "model"]
    data["active_track"] = "human"
    path.write_text(json.dumps(data), encoding="utf-8")


def test_a_session_on_a_missing_lane_is_refused_with_a_message(
        app, tmp_path, dialogs):
    bad = tmp_path / "bad.json"
    _stray_track_session(app, bad)

    live = [Interval(ts("00:20:00"), ts("00:25:00"), "sw", None,
                     track="human")]
    app.intervals[:] = list(live)
    app.status_var.set("quiet")
    dialogs[:] = []

    app._load_session(str(bad))

    assert [d for d in dialogs] == [("showerror", "Load Failed")]
    assert [(iv.start, iv.end, iv.label, iv.track) for iv in app.intervals] \
        == [(iv.start, iv.end, iv.label, iv.track) for iv in live], \
        "live state is untouched -- nothing had been published"
    assert app.status_var.get().startswith("Refused %s --" % (bad,))


def test_the_head_less_channel_still_RAISES(tmp_path):
    """app.load(path) and the GUI-free hosts in the suite must keep the
    exception: a modal would hang a display-less script forever."""
    host = headless(tmp_path)
    bad = tmp_path / "bad_headless.json"
    _stray_track_session(host, bad)

    live = [Interval(ts("00:20:00"), ts("00:25:00"), "sw", None,
                     track="human")]
    host.intervals[:] = list(live)
    with pytest.raises(ValueError) as exc:
        host.load(str(bad))
    assert "not in its track table" in str(exc.value)
    assert len(host.intervals) == 1 and host.intervals[0].track == "human"


# ============== (c) a saved active lane that is gone resolves visibly


def test_a_bogus_active_lane_resolves_to_the_first_VISIBLE_lane(app,
                                                                tmp_path):
    sess = tmp_path / "s.json"
    app._save_session(str(sess))
    data = json.loads(sess.read_text(encoding="utf-8"))
    data["active_track"] = "ghost"
    # Hide the FIRST row, so tracks[0] -- the old fallback -- would have
    # installed an active lane the strip does not paint.
    data["tracks"][0]["visible"] = False
    sess.write_text(json.dumps(data), encoding="utf-8")

    app._load_session(str(sess))

    assert app._active_track_id == "model"
    assert app.track_by_id("model").visible is True
    assert app.status_var.get() == (
        "Loaded from %s -- lane 'ghost' no longer exists -- active lane "
        "is now 'Model'" % (sess,))


def test_a_real_saved_active_lane_is_installed_unchanged(app, tmp_path):
    """The control: a file that names a lane it holds still gets it."""
    sess = tmp_path / "s.json"
    app._set_active_track("model", announce=False, repaint=False)
    app._save_session(str(sess))
    app._set_active_track("human", announce=False, repaint=False)
    app._load_session(str(sess))
    assert app._active_track_id == "model"
    assert app.status_var.get() == "Loaded from %s" % (sess,)


# ===================== (d) the autosave describes the loaded session


def test_the_load_writes_an_autosave_over_the_discarded_state(app,
                                                              tmp_path):
    sess = tmp_path / "s.json"
    app.intervals[:] = [
        Interval(ts("00:20:00"), ts("00:25:00"), "sw", None, track="human")]
    app._save_session(str(sess))

    # drift away from the saved session, and autosave THAT
    app.intervals.append(
        Interval(ts("00:40:00"), ts("00:45:00"), "msh", None, track="human"))
    app._save_autosave()
    assert len(json.loads(app.autosave_file.read_text(
        encoding="utf-8"))["intervals"]) == 2

    app._load_session(str(sess))

    assert len(app.intervals) == 1
    payload = json.loads(app.autosave_file.read_text(encoding="utf-8"))
    assert len(payload["intervals"]) == 1, \
        "the autosave describes the session on screen, not the one the " \
        "user loaded away from"
    assert payload["active_track"] == "human"


def test_a_refused_load_writes_no_autosave(app, tmp_path, dialogs):
    """Fail closed on the disk too: the refusal path returns before the
    publish, so the autosave still describes the live session."""
    bad = tmp_path / "bad.json"
    _stray_track_session(app, bad)
    app.intervals[:] = [
        Interval(ts("00:20:00"), ts("00:25:00"), "sw", None, track="human")]
    app._save_autosave()
    before = app.autosave_file.read_bytes()
    app._load_session(str(bad))
    assert app.autosave_file.read_bytes() == before
