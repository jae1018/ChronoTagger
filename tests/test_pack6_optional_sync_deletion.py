"""Pack 6 PART C (OPTIONAL) -- the four zero-caller PaneSyncManager hooks.

SEVERABLE WITH EDIT 248. If J.E. declines PART C at the read gate, delete
this file and strike EDIT 248; nothing else in the pack refers to either.

PART C is a verifier-driven scope addition, not part of the census's
proven delete set. Its whole warrant is a caller census: four of the five
PaneSyncManager methods have ZERO call sites anywhere in src/ or tests/,
so removing them costs no call-site edits and touches no other file. This
module pins that warrant from both ends -- the four names are gone, AND
the one method that IS called is still there and still callable.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import chronotagger
from chronotagger.labeler.sync import PaneSyncManager

SRC_ROOT = Path(chronotagger.__file__).resolve().parent


def _all_src_files():
    return [p for p in SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts]


def _code_only(path: Path) -> str:
    """CODE tokens only -- comments AND string literals dropped.

    Reserved for the two assertions below that are about token SHAPE
    (`self . sync_manager . sync_intervals_changed ( )`). It is BLIND to
    any name that lives inside a string; use `_code_and_literals` for
    NAME-absence. See test_pack6_cleanup.py for the measured reason.
    """
    import tokenize

    out = []
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    return " ".join(out)


def _strip_docstrings(node):
    body = getattr(node, "body", None)
    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                         ast.AsyncFunctionDef)) and body:
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    for child in ast.iter_child_nodes(node):
        _strip_docstrings(child)


def _code_and_literals(path: Path) -> str:
    """Code INCLUDING string literals, with comments and docstrings gone.

    The non-blind detector. EDIT 248's AFTER names all four deleted hooks
    in its module docstring, which is exactly why docstrings have to be
    stripped and string LITERALS must not be.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    _strip_docstrings(tree)
    return ast.unparse(tree)


@pytest.mark.parametrize("name", [
    "sync_time_window",
    "sync_labels_changed",
    "sync_selection_changed",
    "mark_all_dirty",
])
def test_the_zero_caller_sync_hooks_are_gone(name):
    assert not hasattr(PaneSyncManager, name)


@pytest.mark.parametrize("name", [
    "sync_time_window",
    "sync_labels_changed",
    "sync_selection_changed",
    "mark_all_dirty",
])
def test_the_deleted_hooks_appear_nowhere_in_src_code(name):
    """The census that justifies PART C, re-run as an assertion: if any of
    these four ever acquires a caller -- in code OR through a string, e.g.
    `getattr(mgr, "mark_all_dirty")` -- this test is the thing that says so
    before the deletion ships."""
    hits = [str(p.relative_to(SRC_ROOT))
            for p in _all_src_files()
            if name in _code_and_literals(p)]
    assert hits == []


def test_the_hook_that_IS_called_survives_and_is_callable():
    """sync_intervals_changed has 8 live call sites and must not go with
    the others. It is a declared no-op, so calling it must be harmless."""
    assert hasattr(PaneSyncManager, "sync_intervals_changed")

    mgr = PaneSyncManager(object())
    assert mgr.sync_intervals_changed() is None      # no-op, and no raise


def test_the_surviving_hook_still_has_its_eight_call_sites():
    """PART C must not have taken a live call site with it."""
    hits = 0
    for p in _all_src_files():
        hits += _code_only(p).count(
            "self . sync_manager . sync_intervals_changed ( )")
    assert hits == 8


def test_the_module_no_longer_imports_what_only_the_deleted_hooks_used():
    """`import pandas as pd` in sync.py's TYPE_CHECKING block existed only
    for sync_time_window's two pd.Timestamp annotations."""
    code = _code_only(SRC_ROOT / "labeler" / "sync.py")
    assert "pandas" not in code
    # The one annotation that IS still needed survives.
    assert "TimeIntervalLabeler" in code


def test_the_class_is_now_one_hook_plus_its_constructor():
    methods = [n for n, _ in inspect.getmembers(
        PaneSyncManager, predicate=inspect.isfunction)
        if not n.startswith("__")]
    assert sorted(methods) == ["sync_intervals_changed"]
