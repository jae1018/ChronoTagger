"""Pack 6 PART A (R10) -- the bounded retry around tk.Tk().

Severable: this module pins PART A and nothing else, so PART A can land,
be graded and be trusted before any PART B evidence is collected.

No tk.Tk() is constructed here.  tk.Tk is replaced by a fake that raises
tkinter.TclError a chosen number of times, which is the only way to
exercise a fault that fires ~89% of the time on one machine and 0% on
another.  There is no timing assertion anywhere in this file: the backoff
is pinned by the RECORDED sleep arguments, not by a wall clock.
"""
from __future__ import annotations

import inspect
import logging

import pytest
import tkinter as tk

from chronotagger.labeler.mixins.view_build import window as window_mod

LOGGER_NAME = "chronotagger.labeler.mixins.view_build.window"


class _FakeRoot:
    """Stand-in for the object tk.Tk() would have returned."""


@pytest.fixture
def tk_probe(monkeypatch):
    """Replace tk.Tk and the module's sleep; hand back a recorder."""
    state = {"calls": 0, "sleeps": []}

    monkeypatch.setattr(window_mod.time, "sleep",
                        lambda s: state["sleeps"].append(s))
    monkeypatch.setattr(window_mod, "tk_root_retry_recoveries", 0,
                        raising=False)

    def arm(fail_times, exc=None):
        exc_factory = exc or (lambda i: tk.TclError(
            "Can't find a usable init.tcl (synthetic #%d)" % i))

        def fake_tk():
            state["calls"] += 1
            if state["calls"] <= fail_times:
                raise exc_factory(state["calls"])
            return _FakeRoot()

        monkeypatch.setattr(window_mod.tk, "Tk", fake_tk)
        return state

    state["arm"] = arm
    return state


def test_a_transient_tclerror_is_retried_and_the_recovery_is_counted(
        tk_probe, caplog):
    """Two failures then a success: a root comes back, and the session
    counter records exactly one recovery."""
    tk_probe["arm"](fail_times=2)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        root = window_mod._new_tk_root()

    assert isinstance(root, _FakeRoot)
    assert tk_probe["calls"] == 3
    assert window_mod.tk_root_retry_recoveries == 1

    warnings = [r for r in caplog.records
                if r.name == LOGGER_NAME and r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "TclError" in warnings[0].getMessage()


def test_the_backoff_is_bounded_and_grows(tk_probe):
    """The recorded sleeps are the declared backoff, in order -- pinned by
    the arguments, never by elapsed time."""
    tk_probe["arm"](fail_times=2)
    window_mod._new_tk_root()

    assert tk_probe["sleeps"] == list(window_mod._TK_ROOT_BACKOFF_S[:2])
    assert len(window_mod._TK_ROOT_BACKOFF_S) == window_mod._TK_ROOT_ATTEMPTS - 1


def test_a_clean_first_attempt_neither_sleeps_nor_logs(tk_probe, caplog):
    """The happy path must be byte-for-byte the old behaviour: one call,
    no sleep, no log record, counter untouched."""
    tk_probe["arm"](fail_times=0)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        root = window_mod._new_tk_root()

    assert isinstance(root, _FakeRoot)
    assert tk_probe["calls"] == 1
    assert tk_probe["sleeps"] == []
    assert window_mod.tk_root_retry_recoveries == 0
    assert [r for r in caplog.records if r.name == LOGGER_NAME] == []


def test_six_consecutive_failures_propagate_the_last_tclerror(tk_probe):
    """A genuinely broken Tk install still fails loudly.  Six failures is
    one more than the bound, so every attempt is spent and the LAST error
    is what the caller sees -- not a wrapped or swallowed one."""
    tk_probe["arm"](fail_times=6)

    with pytest.raises(tk.TclError) as excinfo:
        window_mod._new_tk_root()

    assert tk_probe["calls"] == window_mod._TK_ROOT_ATTEMPTS
    assert "synthetic #%d" % window_mod._TK_ROOT_ATTEMPTS in str(excinfo.value)
    assert window_mod.tk_root_retry_recoveries == 0
    assert len(tk_probe["sleeps"]) == window_mod._TK_ROOT_ATTEMPTS - 1


def test_a_non_tclerror_is_not_retried(tk_probe):
    """The retry is scoped to TclError.  Anything else -- a MemoryError, a
    KeyboardInterrupt, a typo in the caller -- propagates on the first
    raise with no sleep and no second attempt."""
    tk_probe["arm"](fail_times=99,
                    exc=lambda i: RuntimeError("not a Tcl fault"))

    with pytest.raises(RuntimeError):
        window_mod._new_tk_root()

    assert tk_probe["calls"] == 1
    assert tk_probe["sleeps"] == []


def test_build_gui_creates_its_root_through_the_retry():
    """Structural pin for the call site.  Building a real GUI here would
    re-introduce the very flake this pack removes, so the assertion reads
    the source of _build_gui instead: the bare tk.Tk() must be gone and
    the helper must be what is called."""
    src = inspect.getsource(window_mod.WindowMixin._build_gui)
    assert "self.root = _new_tk_root()" in src
    # The bare construction is gone.  (The method's own comment block
    # still says the words "nested tk.Tk()", which is why this asserts on
    # the ASSIGNMENT and not on the bare call text.)
    assert "self.root = tk.Tk()" not in src
    # The parent branch is untouched: a wizard-launched labeler still
    # mounts as a Toplevel and never reaches the retry at all.
    assert "tk.Toplevel(parent)" in src
