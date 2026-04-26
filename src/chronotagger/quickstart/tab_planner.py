"""
Tab planner dialog for ChronoTagger quick-start wizard.

Replaces the older single-tab column selector. Lets the user configure
1+ labeler tabs in a single dialog using a ttk.Notebook (each Notebook
page is one eventual labeler tab). Defaults to one tab; the user can
add more via "+ Add Tab" up to MAX_TABS, or remove any tab beyond the
first via the per-tab "Remove tab" button.

Each tab carries:
  - title (free-form string; defaults to "Tab N")
  - column subset (multi-select from the DataFrame's numeric columns)
  - layout type ('vertical_stack' or 'custom_grid')
  - if 'custom_grid' AND the user pressed 'Design Layout...': a
    layout_spec + plot_config produced by the existing
    chronotagger.labeler.utils.layout_builder.build_layout() designer

The dialog returns:
    {'tabs': [
        {'title': str,
         'columns': List[str],
         'layout_type': 'vertical_stack' | 'custom_grid',
         'layout_spec': dict,    # only present for custom_grid
         'plot_config': dict},   # only present for custom_grid
        ...
    ]}
or None if the user cancelled.

Layout-design flow: clicking the "Custom Grid" radio enables a "Design
Layout..." button. Clicking that button launches the layout designer
filtered to the tab's currently-selected columns. The resulting
layout_spec / plot_config are stored on the tab; the button label
flips to "Re-design Layout (designed)" so the user can revise it.
Cancelling the designer leaves the tab in whatever state it was in
(the radio doesn't auto-revert).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional

import pandas as pd


class _TabState:
    """
    Per-tab state container. One instance lives on each Notebook page.

    Holds Tk variable references (so the entries / checkboxes / radios
    in the UI write into the tab's state) plus the optional designed
    custom-grid output.
    """

    def __init__(self, title: str):
        self.title_var = tk.StringVar(value=title)
        # column_vars filled in _build_tab_ui (one BooleanVar per column)
        self.column_vars: Dict[str, tk.BooleanVar] = {}
        self.layout_var = tk.StringVar(value="vertical_stack")
        # Filled when the user designs a custom grid for this tab
        self.layout_spec: Optional[Dict[str, Any]] = None
        self.plot_config: Optional[Dict[str, Any]] = None
        # Widget refs (set during _build_tab_ui)
        self.page_frame: Optional[ttk.Frame] = None
        self.design_btn: Optional[tk.Button] = None
        self.remove_btn: Optional[tk.Button] = None


class TabPlannerDialog:
    """
    Multi-tab planner dialog. Always uses a ttk.Notebook; defaults to
    one tab so single-pane users see a familiar one-page form.
    """

    MAX_TABS = 6

    def __init__(self, parent: tk.Tk, df: pd.DataFrame):
        self.parent = parent
        self.df = df
        self.numeric_columns: List[str] = (
            df.select_dtypes(include=["number"]).columns.tolist()
        )

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Configure Labeler Tabs")
        self.dialog.geometry("620x680")
        self._center_on_parent()
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # State
        self.tabs: List[_TabState] = []
        self.result: Optional[Dict[str, Any]] = None

        # Widgets
        self.notebook: Optional[ttk.Notebook] = None
        self.add_tab_btn: Optional[tk.Button] = None
        self.tab_count_label: Optional[ttk.Label] = None
        self.continue_btn: Optional[tk.Button] = None

        self._build_ui()
        # Always start with one tab
        self._add_tab()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _center_on_parent(self):
        self.dialog.update_idletasks()
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_w = self.parent.winfo_width()
        parent_h = self.parent.winfo_height()
        dialog_w = self.dialog.winfo_width()
        dialog_h = self.dialog.winfo_height()
        x = parent_x + (parent_w // 2) - (dialog_w // 2)
        y = parent_y + (parent_h // 2) - (dialog_h // 2)
        self.dialog.geometry(f"{dialog_w}x{dialog_h}+{x}+{y}")

    def _build_ui(self):
        # Top bar: Add Tab button + tab count
        top = ttk.Frame(self.dialog, padding="10")
        top.pack(fill="x")

        self.add_tab_btn = tk.Button(
            top, text="+ Add Tab", command=self._on_add_tab
        )
        self.add_tab_btn.pack(side="left")

        self.tab_count_label = ttk.Label(top, text="", foreground="gray")
        self.tab_count_label.pack(side="left", padx=(10, 0))

        # Notebook (one page per labeler tab)
        self.notebook = ttk.Notebook(self.dialog)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Bottom buttons
        btn_row = ttk.Frame(self.dialog, padding="10")
        btn_row.pack(fill="x")

        self.continue_btn = tk.Button(
            btn_row, text="Continue", command=self._on_continue
        )
        self.continue_btn.pack(side="right")

        cancel_btn = tk.Button(btn_row, text="Cancel", command=self._on_cancel)
        cancel_btn.pack(side="right", padx=(0, 5))

    # ------------------------------------------------------------------
    # Tab management
    # ------------------------------------------------------------------

    def _on_add_tab(self):
        if len(self.tabs) >= self.MAX_TABS:
            messagebox.showinfo(
                "Maximum tabs reached",
                f"Up to {self.MAX_TABS} tabs are supported.",
                parent=self.dialog,
            )
            return
        self._add_tab()

    def _add_tab(self):
        idx = len(self.tabs)
        tab = _TabState(title=f"Tab {idx + 1}")
        self.tabs.append(tab)

        page = ttk.Frame(self.notebook, padding="10")
        tab.page_frame = page
        self._build_tab_ui(page, tab)

        self.notebook.add(page, text=tab.title_var.get())
        # Live-update the Notebook tab text when the title field changes
        tab.title_var.trace_add(
            "write",
            lambda *_, t=tab: self._on_title_changed(t),
        )

        # Show the new tab and update affordances
        self.notebook.select(page)
        self._refresh_affordances()

    def _on_title_changed(self, tab: _TabState):
        if tab.page_frame is None:
            return
        try:
            page_id = self.notebook.index(tab.page_frame)
        except tk.TclError:
            return  # page may be in the middle of being destroyed
        new_title = tab.title_var.get().strip() or "Tab"
        self.notebook.tab(page_id, text=new_title)

    def _on_remove_tab(self, tab: _TabState):
        if len(self.tabs) <= 1:
            return  # remove buttons should be hidden in this case
        if tab.page_frame is None:
            return
        try:
            page_id = self.notebook.index(tab.page_frame)
        except tk.TclError:
            return
        self.notebook.forget(page_id)
        self.tabs.remove(tab)
        self._refresh_affordances()

    def _refresh_affordances(self):
        """Update the tab count label, the Add Tab button enabled state,
        and the per-tab Remove buttons (hidden when only one tab exists).
        """
        n = len(self.tabs)

        if self.tab_count_label is not None:
            self.tab_count_label.config(text=f"{n} of {self.MAX_TABS} tabs")

        if self.add_tab_btn is not None:
            self.add_tab_btn.config(
                state="normal" if n < self.MAX_TABS else "disabled"
            )

        for tab in self.tabs:
            if tab.remove_btn is None:
                continue
            if n >= 2:
                # Show the button if it's not currently visible
                tab.remove_btn.pack(side="right")
            else:
                tab.remove_btn.pack_forget()

    # ------------------------------------------------------------------
    # Per-tab UI
    # ------------------------------------------------------------------

    def _build_tab_ui(self, page: ttk.Frame, tab: _TabState):
        # Title row + remove button on the right
        title_row = ttk.Frame(page)
        title_row.pack(fill="x", pady=(0, 10))

        ttk.Label(title_row, text="Title:").pack(side="left")
        ttk.Entry(
            title_row, textvariable=tab.title_var, width=40
        ).pack(side="left", padx=(5, 0))

        tab.remove_btn = tk.Button(
            title_row,
            text="x Remove tab",
            command=lambda t=tab: self._on_remove_tab(t),
        )
        # Visibility is managed by _refresh_affordances

        # Column selection (scrollable list of checkboxes)
        col_frame = ttk.LabelFrame(
            page, text="Columns on this tab", padding="5"
        )
        col_frame.pack(fill="both", expand=True, pady=(0, 10))

        canvas = tk.Canvas(col_frame, height=180)
        scrollbar = ttk.Scrollbar(
            col_frame, orient="vertical", command=canvas.yview
        )
        scrollable = ttk.Frame(canvas)
        scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        for col in self.numeric_columns:
            var = tk.BooleanVar(value=True)
            tab.column_vars[col] = var
            cb = ttk.Checkbutton(scrollable, text=col, variable=var)
            cb.pack(anchor="w", pady=1)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Select / Deselect all
        col_btns = ttk.Frame(col_frame)
        col_btns.pack(fill="x", pady=(5, 0))
        tk.Button(
            col_btns,
            text="Select All",
            command=lambda t=tab: self._select_all(t),
        ).pack(side="left", padx=(0, 5))
        tk.Button(
            col_btns,
            text="Deselect All",
            command=lambda t=tab: self._deselect_all(t),
        ).pack(side="left")

        # Layout choice
        layout_frame = ttk.LabelFrame(
            page, text="Layout for this tab", padding="5"
        )
        layout_frame.pack(fill="x")

        ttk.Radiobutton(
            layout_frame,
            text="Vertical Stack (recommended)",
            variable=tab.layout_var,
            value="vertical_stack",
            command=lambda t=tab: self._on_layout_changed(t),
        ).pack(anchor="w", pady=2)

        cg_row = ttk.Frame(layout_frame)
        cg_row.pack(fill="x", pady=2)

        ttk.Radiobutton(
            cg_row,
            text="Custom Grid",
            variable=tab.layout_var,
            value="custom_grid",
            command=lambda t=tab: self._on_layout_changed(t),
        ).pack(side="left")

        tab.design_btn = tk.Button(
            cg_row,
            text="Design Layout...",
            command=lambda t=tab: self._on_design_layout(t),
            state="disabled",
        )
        tab.design_btn.pack(side="left", padx=(15, 0))

    # ------------------------------------------------------------------
    # Per-tab events
    # ------------------------------------------------------------------

    def _select_all(self, tab: _TabState):
        for var in tab.column_vars.values():
            var.set(True)

    def _deselect_all(self, tab: _TabState):
        for var in tab.column_vars.values():
            var.set(False)

    def _on_layout_changed(self, tab: _TabState):
        """Enable/disable the Design Layout button to match the radio."""
        if tab.design_btn is None:
            return
        if tab.layout_var.get() == "custom_grid":
            tab.design_btn.config(state="normal")
        else:
            tab.design_btn.config(state="disabled")
        self._refresh_design_btn_label(tab)

    def _refresh_design_btn_label(self, tab: _TabState):
        if tab.design_btn is None:
            return
        if tab.layout_spec is not None:
            tab.design_btn.config(text="Re-design Layout (designed)")
        else:
            tab.design_btn.config(text="Design Layout...")

    def _on_design_layout(self, tab: _TabState):
        """Launch the layout designer for this tab's column subset."""
        selected = [c for c, v in tab.column_vars.items() if v.get()]
        if not selected:
            messagebox.showwarning(
                "Select columns first",
                "Please select at least one column for this tab "
                "before designing the layout.",
                parent=self.dialog,
            )
            return

        # Pass only this tab's columns to the designer so its dropdowns
        # are scoped to what the tab will actually plot.
        from chronotagger.labeler.utils.layout_builder import build_layout

        df_subset = self.df[selected]
        layout_spec, plot_config = build_layout(
            df_subset, parent=self.dialog
        )

        if layout_spec is None:
            return  # User cancelled the designer; tab state unchanged

        tab.layout_spec = layout_spec
        tab.plot_config = plot_config
        self._refresh_design_btn_label(tab)

    # ------------------------------------------------------------------
    # Validation + finish
    # ------------------------------------------------------------------

    def _validate(self) -> tuple[bool, str]:
        if not self.tabs:
            return False, "No tabs configured."
        for i, tab in enumerate(self.tabs, start=1):
            title = tab.title_var.get().strip() or f"Tab {i}"
            selected = [c for c, v in tab.column_vars.items() if v.get()]
            if not selected:
                return (
                    False,
                    f"'{title}' has no columns selected. "
                    f"Please pick at least one column.",
                )
            if tab.layout_var.get() == "custom_grid":
                if tab.layout_spec is None:
                    return (
                        False,
                        f"'{title}' has Custom Grid selected but no layout "
                        f"has been designed yet. Click 'Design Layout...' "
                        f"on that tab.",
                    )
                # The layout was designed against a previous column
                # selection.  Make sure every column the layout still
                # references is still selected -- otherwise the labeler
                # would render "Column X not found" panels.
                referenced = self._columns_referenced_by_plot_config(
                    tab.plot_config or {}
                )
                missing = sorted(referenced - set(selected))
                if missing:
                    return (
                        False,
                        f"'{title}' has a custom layout that references "
                        f"columns no longer selected: {', '.join(missing)}. "
                        f"Re-design the layout (click 'Design Layout...') "
                        f"or restore the missing columns.",
                    )
        return True, ""

    @staticmethod
    def _columns_referenced_by_plot_config(plot_config: Dict[str, Any]) -> set:
        """Collect every dataframe column name a custom plot_config uses."""
        cols: set = set()
        for panel in plot_config.values():
            if not isinstance(panel, dict):
                continue
            for key in ("x_column", "y_column"):
                col = panel.get(key)
                if isinstance(col, str) and col:
                    cols.add(col)
        return cols

    def _on_continue(self):
        ok, msg = self._validate()
        if not ok:
            messagebox.showerror("Cannot continue", msg, parent=self.dialog)
            return

        result_tabs: List[Dict[str, Any]] = []
        for i, tab in enumerate(self.tabs, start=1):
            entry: Dict[str, Any] = {
                "title": tab.title_var.get().strip() or f"Tab {i}",
                "columns": [c for c, v in tab.column_vars.items() if v.get()],
                "layout_type": tab.layout_var.get(),
            }
            if (
                tab.layout_var.get() == "custom_grid"
                and tab.layout_spec is not None
            ):
                entry["layout_spec"] = tab.layout_spec
                entry["plot_config"] = tab.plot_config
            result_tabs.append(entry)

        self.result = {"tabs": result_tabs}
        self.dialog.destroy()

    def _on_cancel(self):
        self.result = None
        self.dialog.destroy()

    def run(self) -> Optional[Dict[str, Any]]:
        self.dialog.wait_window()
        return self.result
