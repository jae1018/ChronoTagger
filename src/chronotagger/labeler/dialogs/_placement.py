# src/chronotagger/labeler/dialogs/_placement.py
"""Where a dialog opens: over the middle of the MAIN WINDOW. Pack M3.0.

Not one modal in `labeler/dialogs/` placed itself -- no geometry call, no
parent read -- so the window manager put it wherever it cascades, which
on Windows is the primary monitor and not a position derived from the
app. The two boxes that DID place themselves, the recovery box
(`mixins/io_export.py`) and Select Component
(`mixins/events/selection.py`), centred on the SCREEN.

MEASURED: the recovery box calls `transient(parent)` between
`geometry("640x700")` and `update_idletasks()`, and on an unmapped
Toplevel `transient` throws the requested size away -- `winfo_width()`
answers 1, so the old arithmetic centred a 1x1 box and put a 640x700
window's CORNER at the middle of the primary display, with its bottom
off the screen. Callers therefore pass a known size in rather than ask
an unmapped window to measure itself, and the centre is the MAIN
WINDOW's, not the primary display's.

Five ad-hoc Toplevels inside the mixins already do the right thing, with
the same twelve hand-copied lines each (`intervals/crud.py` twice,
`intervals/gaps.py`, `help.py`, and the export box in `io_export.py`).
This is that block, once, plus the clamp they do not have. Those five are
deliberately left alone.

The size comes from `winfo_reqwidth` / `winfo_reqheight`, which are valid
BEFORE a window is mapped. A caller whose content is built after it asks
-- the recovery box -- passes its own fixed size in instead.
"""
from __future__ import annotations

from typing import Optional, Tuple


def centered_geometry(parent_x, parent_y, parent_w, parent_h,
                      dialog_w, dialog_h, screen_w, screen_h
                      ) -> Tuple[int, int]:
    """The top-left corner of a dialog centred over its parent.

    Pure numbers and no widgets, so the rule can be pinned without a
    screen.

    CLAMPED to the screen -- but only when the parent's own centre is ON
    the screen being described. `winfo_screenwidth` reports the PRIMARY
    display while window coordinates span the whole virtual desktop, so
    a main window on a second monitor has a centre outside that
    rectangle, and clamping there would drag the dialog back onto the
    primary monitor, which is the exact defect this module exists to
    remove. A dialog larger than the screen pins to the top-left corner.
    """
    px, py = int(parent_x), int(parent_y)
    pw, ph = int(parent_w), int(parent_h)
    dw, dh = int(dialog_w), int(dialog_h)
    sw, sh = int(screen_w), int(screen_h)
    x = px + (pw - dw) // 2
    y = py + (ph - dh) // 2
    cx, cy = px + pw // 2, py + ph // 2
    if sw > 0 and sh > 0 and 0 <= cx < sw and 0 <= cy < sh:
        x = max(0, min(x, sw - dw))
        y = max(0, min(y, sh - dh))
    return x, y


def center_on_parent(dialog, parent, width=None, height=None
                     ) -> Optional[str]:
    """Put `dialog` over the middle of `parent`; returns the "+X+Y" set.

    Never raises. A dialog that cannot be measured keeps the position it
    had, which is exactly what every dialog that does not call this
    already does.
    """
    if dialog is None or parent is None:
        return None
    try:
        dialog.update_idletasks()
    except Exception:
        pass
    try:
        dw = int(width) if width else int(dialog.winfo_reqwidth())
        dh = int(height) if height else int(dialog.winfo_reqheight())
        x, y = centered_geometry(parent.winfo_rootx(), parent.winfo_rooty(),
                                 parent.winfo_width(), parent.winfo_height(),
                                 dw, dh, dialog.winfo_screenwidth(),
                                 dialog.winfo_screenheight())
    except Exception:
        return None
    spec = "+%d+%d" % (x, y)
    try:
        dialog.geometry(spec)
    except Exception:
        return None
    return spec
