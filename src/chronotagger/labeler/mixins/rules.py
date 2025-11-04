"""
Rule-based labeling with support for multiple conditions combined with AND/OR logic.

Key Features:
- Single condition (backward compatible)
- Multiple conditions with AND/OR combination
- Preview shows results before committing
- Respects overlap policies
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from ..dialogs.label_by_rule import LabelByRuleDialog, LabelByRuleResult, RuleCondition


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
        """Open the Label by Rule dialog and handle the result."""
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
        """
        Apply rule preview with multiple conditions.
        
        Args:
            res: Complete rule specification with one or more conditions
            
        Returns:
            Tuple of (num_points_selected, num_spans_created)
        """
        df_scope = self._resolve_scope_df(res)
        
        # Validate all columns exist
        for cond in res.conditions:
            if cond.column not in df_scope.columns:
                raise ValueError(f"Column '{cond.column}' not found in the chosen scope.")

        # Evaluate all conditions and combine with AND/OR
        mask = self._rule_eval_combined_mask(
            df_scope, 
            res.conditions, 
            res.combine_mode, 
            res.nan_as_true
        )
        
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
            mode_str = res.combine_mode if len(res.conditions) > 1 else ""
            cond_str = f"{len(res.conditions)} condition{'s' if len(res.conditions) > 1 else ''}"
            if mode_str:
                cond_str = f"{cond_str} ({mode_str})"
            
            self.status_var.set(
                f"Rule preview: {points} points → {len(commit_full)} spans "
                f"[{cond_str}, scope={res.scope}, policy={self._overlap_policy}]"
            )

        self._update_plot()
        return points, len(commit_full)

    def _rule_preview_clear(self) -> None:
        """Clear the rule preview from the UI."""
        self.current_selection = None
        self.current_spans = []
        self._commit_spans = []
        if self.status_var is not None:
            self.status_var.set("Preview cleared")
        self._update_plot()

    # ---------- Scope helpers ----------
    def _resolve_scope_df(self, res: LabelByRuleResult) -> pd.DataFrame:
        """
        Return the DataFrame slice for the chosen scope; clamps to dataset bounds.
        
        Args:
            res: Rule result containing scope specification
            
        Returns:
            DataFrame slice for the chosen scope
        """
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

    # ---------- Core evaluation logic ----------
    def _rule_eval_combined_mask(
        self, 
        sub_df: pd.DataFrame, 
        conditions: List[RuleCondition], 
        combine_mode: str,
        nan_as_true: bool
    ) -> np.ndarray:
        """
        Evaluate multiple conditions and combine them with AND or OR logic.
        
        Args:
            sub_df: DataFrame to evaluate conditions on
            conditions: List of conditions to evaluate
            combine_mode: "AND" (all must be true) or "OR" (any must be true)
            nan_as_true: Whether NaN values should be treated as True
            
        Returns:
            Boolean mask array where True means the combined condition is satisfied
            
        Examples:
            # X < 0 AND Y < 0
            conditions = [
                RuleCondition(column="X", op="<", value=0),
                RuleCondition(column="Y", op="<", value=0)
            ]
            mask = _rule_eval_combined_mask(df, conditions, "AND", False)
            
            # BX > 10 OR BY > 10
            conditions = [
                RuleCondition(column="BX", op=">", value=10),
                RuleCondition(column="BY", op=">", value=10)
            ]
            mask = _rule_eval_combined_mask(df, conditions, "OR", False)
        """
        if not conditions:
            return np.zeros(len(sub_df), dtype=bool)
        
        # Evaluate first condition
        first_mask = self._rule_eval_single_mask(
            sub_df, 
            conditions[0].column, 
            conditions[0].op, 
            conditions[0].value, 
            nan_as_true
        )
        
        # If only one condition, return it
        if len(conditions) == 1:
            return first_mask
        
        # Combine with remaining conditions
        if combine_mode == "AND":
            # Start with first condition, AND with all others
            combined = first_mask
            for cond in conditions[1:]:
                mask = self._rule_eval_single_mask(
                    sub_df, cond.column, cond.op, cond.value, nan_as_true
                )
                combined = combined & mask
            return combined
        
        elif combine_mode == "OR":
            # Start with first condition, OR with all others
            combined = first_mask
            for cond in conditions[1:]:
                mask = self._rule_eval_single_mask(
                    sub_df, cond.column, cond.op, cond.value, nan_as_true
                )
                combined = combined | mask
            return combined
        
        else:
            raise ValueError(f"Unknown combine_mode: {combine_mode}. Must be 'AND' or 'OR'.")

    def _rule_eval_single_mask(
        self, 
        sub_df: pd.DataFrame, 
        column: str, 
        op: str, 
        value: float, 
        nan_as_true: bool
    ) -> np.ndarray:
        """
        Evaluate a single condition on a DataFrame column.
        
        Args:
            sub_df: DataFrame to evaluate condition on
            column: Column name to evaluate
            op: Comparison operator ("<", "<=", "==", ">=", ">", "!=")
            value: Value to compare against
            nan_as_true: Whether NaN values should be treated as True
            
        Returns:
            Boolean mask array where True means the condition is satisfied
        """
        s = sub_df[column].astype(float)

        # Apply operator
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

        # Handle NaNs
        base = base.fillna(False)
        if nan_as_true:
            base = base | s.isna()

        return base.to_numpy(dtype=bool, copy=False)

    # ---------- Pure helpers ----------
    def _rule_numeric_columns(self) -> List[str]:
        """
        Get list of numeric column names from the dataframe.
        
        Returns:
            List of column names that contain numeric data
        """
        try:
            import numpy as _np
            return list(self.df.select_dtypes(include=[_np.number]).columns)
        except Exception:
            return []

    def _mask_to_runs(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Convert boolean mask to inclusive index runs.
        
        Args:
            mask: Boolean array where True indicates selected samples
            
        Returns:
            List of (start_idx, end_idx) tuples for contiguous True regions
            Both indices are inclusive.
            
        Example:
            mask = [False, True, True, False, True, False]
            returns [(1, 2), (4, 4)]
        """
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
        self, 
        idx: pd.DatetimeIndex, 
        runs: List[Tuple[int, int]]
    ) -> tuple[List[tuple[pd.Timestamp, pd.Timestamp]], List[tuple[pd.Timestamp, pd.Timestamp]]]:
        """
        Convert index runs to preview and commit timestamp spans.
        
        Preview spans end AT the last included sample (for display).
        Commit spans are half-open [start, next(last)) (for actual labeling).
        
        Args:
            idx: DatetimeIndex to extract timestamps from
            runs: List of inclusive (start_idx, end_idx) tuples
            
        Returns:
            Tuple of (preview_spans, commit_spans) where each is a list of
            (start_timestamp, end_timestamp) tuples
        """
        preview_spans: List[tuple[pd.Timestamp, pd.Timestamp]] = []
        for i0, i1 in runs:
            preview_spans.append((pd.Timestamp(idx[i0]), pd.Timestamp(idx[i1])))
        
        # Reuse the half-open interval logic from EventsMixin
        commit_spans = self._runs_to_half_open_intervals(idx, runs)
        
        return preview_spans, commit_spans

    def _clip_spans_to_window(
        self,
        spans: List[tuple[pd.Timestamp, pd.Timestamp]],
        t0: pd.Timestamp,
        t1: pd.Timestamp
    ) -> List[tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Clip spans to the visible time window.
        
        Args:
            spans: List of (start, end) timestamp tuples
            t0: Window start time
            t1: Window end time
            
        Returns:
            List of spans that intersect [t0, t1], clipped to window bounds
        """
        out: List[tuple[pd.Timestamp, pd.Timestamp]] = []
        for s, e in spans:
            # Skip spans completely outside window
            if e <= t0 or s >= t1:
                continue
            # Clip to window bounds
            out.append((max(s, t0), min(e, t1)))
        return out
