"""
Pack 2 GUI-backed test: the recovery dialog stays usable in the
identity-mismatch case (fold BLOCKER: the v1 draft starved all four
buttons off-screen in exactly that branch).
"""

import tkinter as tk


def _find_buttons(widget, found):
    for child in widget.winfo_children():
        if isinstance(child, tk.Button):
            found[child.cget("text")] = child
        _find_buttons(child, found)


def test_recovery_dialog_mismatch_stays_usable(labeler, monkeypatch):
    payload = {
        "intervals": [],
        "metadata": {
            "autosave_timestamp": "2026-08-22 12:00:00",
            "coverage_percent": 41.5,
            "total_intervals": 12,
            "fingerprint": "abc123def456",
            "source_name": "tha_2024-03-01.parquet",
            "time_range": {"start": "2024-03-01T00:00:00",
                           "end": "2024-03-02T00:00:00"},
        },
        "label_stats": {
            f"label_{i}": {"count": 2, "duration_hours": 1.0}
            for i in range(6)
        },
        "_identity": {
            "mismatch": True,
            "lines": [
                "WARNING: saved columns differ from current data",
                "  only in autosave: n_ion",
                "  only in current:  n_elec",
                "WARNING: autosave came from a different source file",
                "  autosave: tha_2024-03-01.parquet",
                "  current:  thd_2024-03-01.parquet",
            ],
        },
        "_loaded_path": ("C:/some/very/long/path/to/the/autosave/file/"
                         "chronotagger_autosave_abc123def456.json"),
    }

    seen = {}

    def fake_wait(dialog_widget):
        dialog_widget.update_idletasks()
        dialog_widget.update()
        buttons = {}
        _find_buttons(dialog_widget, buttons)
        h = dialog_widget.winfo_height()
        for text, btn in buttons.items():
            y = btn.winfo_rooty() - dialog_widget.winfo_rooty()
            seen[text] = {
                "mapped": bool(btn.winfo_ismapped()),
                "height": btn.winfo_height(),
                "inside": 0 <= y < h,
            }
        seen["_focus"] = str(dialog_widget.focus_get())
        seen["_fresh"] = str(buttons.get("Start Fresh"))
        seen["_protocol"] = dialog_widget.protocol("WM_DELETE_WINDOW")
        dialog_widget.destroy()

    monkeypatch.setattr(tk.Toplevel, "wait_window", fake_wait)

    # The fixture withdraws the root; a transient Toplevel of a
    # withdrawn master is itself withdrawn (geometry 1x1, nothing
    # mapped), so deiconify for the measurement (recheck M2).
    labeler.root.deiconify()
    try:
        choice = labeler._show_recovery_dialog(payload)
    finally:
        labeler.root.withdraw()

    # All four buttons visible and usable, even with the full mismatch
    # block plus six label rows on screen.
    for text in ("Recover Session", "Start Fresh",
                 "Save & Start Fresh", "Exit ChronoTagger"):
        assert text in seen, f"button {text!r} not found"
        assert seen[text]["mapped"] and seen[text]["height"] > 5, text
        assert seen[text]["inside"], f"{text} rendered outside the dialog"

    # Mismatch flips the safe default to Start Fresh (skip the check if
    # the window manager declined to assign focus at all).
    if seen["_focus"] not in ("None", ""):
        assert seen["_focus"] == seen["_fresh"]

    # X-close is wired to a registered protocol handler.
    assert seen["_protocol"]

    # The inspector destroyed the dialog without pressing a button, so
    # no choice was recorded -- the assertions above are the substance.
    assert choice is None
