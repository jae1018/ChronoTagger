from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from ..dialogs.label_by_rule import LabelByRuleDialog, LabelByRuleResult


class RulesMixin:
    """
    'Label by Rule' flow:
      1) Open dialog → preview computes runs over chosen scope.
      2) Preview draws only spans intersecting the *current window*.
      3) Commit list (_commit_spans) keeps *all* scope spans (half-open).
      4) 'Add Label' applies with chosen overlap policy.
    """

    # ---------- Public entrypoint ----------
    def _open_label_by_rule_dialog(self) -> None:
        numeric_cols = self._rule_numeric_columns()
        dlg = LabelByRuleDialog(
            parent=self.root,                     # type: ignore[arg-type]
            numeric_columns=numeric_cols,
            on_preview=self._rule_preview_apply,
            on_clear_preview=self._rule_preview_clear,
        )
        self.root.wait_window(dlg)               # type: ignore[union-attr]
        if dlg.result is None:
            return
        # Remember selected overlap policy; _commit_spans already prepared by preview
        self._overlap_policy = dlg.result.overlap_policy or "skip"

    # ---------- Preview plumbing ----------
    def _rule_preview_apply(self, res: LabelByRuleResult) -> tuple[int, int]:
        df_scope = self._resolve_scope_df(res)
        if res.column not in df_scope.columns:
            raise ValueError(f"Column '{res.column}' not found in the chosen scope.")

        mask = self._rule_eval_mask(df_scope, res.column, res.op, res.value, res.nan_as_true)
        points = int(mask.sum())
        runs = self._mask_to_runs(mask)

        # Build full-scope preview+commit
        preview_full, commit_full = self._runs_to_preview_and_commit(df_scope.index, runs)

        # For drawing, clip preview to current visible window only (avoid painting the whole dataset)
        preview_for_window = self._clip_spans_to_window(preview_full, self.t0, self.t1)

        # Drive UI state
        self.current_selection = None
        self.current_spans = preview_for_window
        self._commit_spans = commit_full    # full scope; used by Add Label
        self._overlap_policy = res.overlap_policy or "skip"

        if self.status_var is not None:
            self.status_var.set(
                f"Rule preview: {points} points → {len(commit_full)} spans (scope={res.scope}, policy={self._overlap_policy})"
            )

        self._update_plot()
        return points, len(commit_full)

    def _rule_preview_clear(self) -> None:
        self.current_selection = None
        self.current_spans = []
        self._commit_spans = []
        if self.status_var is not None:
            self.status_var.set("Preview cleared")
        self._update_plot()

    # ---------- Scope helpers ----------
    def _resolve_scope_df(self, res: LabelByRuleResult) -> pd.DataFrame:
        """Return the DataFrame slice for the chosen scope; clamps to dataset bounds."""
        if res.scope == "window":
            try:
                return self.df.loc[self.t0:self.t1]
            except Exception:
                return self.df

        if res.scope == "dataset":
            return self.df

        # custom
        start = self.data_start
        end = self.data_end
        try:
            if res.custom_start:
                start = pd.to_datetime(res.custom_start)
            if res.custom_end:
                end = pd.to_datetime(res.custom_end)
        except Exception as e:
            raise ValueError(f"Could not parse custom range: {e}")

        # clamp + order
        if start > end:
            start, end = end, start
        start = max(start, self.data_start)
        end = min(end, self.data_end)

        if start >= end:
            # Fallback to empty slice semantics; downstream code will yield 0 runs
            return self.df.iloc[0:0]
        return self.df.loc[start:end]

    # ---------- Pure helpers ----------
    def _rule_numeric_columns(self) -> List[str]:
        try:
            import numpy as _np
            return list(self.df.select_dtypes(include=[_np.number]).columns)
        except Exception:
            return []

    def _rule_eval_mask(self, sub_df: pd.DataFrame, column: str, op: str, value: float, nan_as_true: bool) -> np.ndarray:
        s = sub_df[column].astype(float)

        if op == "<":
            base = s < value
        elif op == "<=":
            base = s <= value
        elif op == "==":
            base = s == value
        elif op == ">=":
            base = s >= value
        elif op == ">":
            base = s > value
        elif op == "!=":
            base = s != value
        else:
            raise ValueError(f"Unsupported operator: {op}")

        base = base.fillna(False)
        if nan_as_true:
            base = base | s.isna()

        return base.to_numpy(dtype=bool, copy=False)

    def _mask_to_runs(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Boolean mask → inclusive index runs [(i0, i1), ...] where True and contiguous."""
        runs: List[Tuple[int, int]] = []
        if mask.size == 0:
            return runs
        n = mask.size
        i = 0
        while i < n:
            if not mask[i]:
                i += 1
                continue
            i0 = i
            i += 1
            while i < n and mask[i]:
                i += 1
            runs.append((i0, i - 1))
        return runs

    def _runs_to_preview_and_commit(
        self, idx: pd.DatetimeIndex, runs: List[Tuple[int, int]]
    ) -> tuple[List[tuple[pd.Timestamp, pd.Timestamp]], List[tuple[pd.Timestamp, pd.Timestamp]]]:
        """
        preview spans end AT last included sample; commit spans are half-open [start, next(last)].
        """
        preview_spans: List[tuple[pd.Timestamp, pd.Timestamp]] = []
        for i0, i1 in runs:
            preview_spans.append((pd.Timestamp(idx[i0]), pd.Timestamp(idx[i1])))
        commit_spans = self._runs_to_half_open_intervals(idx, runs)  # from EventsMixin
        return preview_spans, commit_spans

    def _clip_spans_to_window(
        self,
        spans: List[tuple[pd.Timestamp, pd.Timestamp]],
        t0: pd.Timestamp,
        t1: pd.Timestamp
    ) -> List[tuple[pd.Timestamp, pd.Timestamp]]:
        """Return spans intersecting [t0,t1], clipped to the window."""
        out: List[tuple[pd.Timestamp, pd.Timestamp]] = []
        for s, e in spans:
            if e <= t0 or s >= t1:
                continue
            out.append((max(s, t0), min(e, t1)))
        return out