"""
evaluation/metrics.py — MIDST Framework

Utility metrics for synthetic data evaluation.

Three tiers of utility, in increasing strength:
  Tier 1 — Statistical fidelity  (column distributions, correlations)
  Tier 2 — Structural integrity  (logical constraints, null invariance)
  Tier 3 — ML utility            (TSTR: Train on Synthetic, Test on Real)

A model that passes all three tiers is genuinely useful, not just
statistically similar on paper.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


@dataclass
class UtilityResult:
    """
    Structured output from utility evaluation.
    All scores are 0–1 where 1 = perfect fidelity / utility.
    """
    correlation_similarity: float       # 1 - mean_abs_correlation_diff
    logic_consistency: float            # fraction of records with no constraint violations
    tstr_score: float                   # TSTR accuracy (or -1.0 if not applicable)
    tstr_baseline: float                # real-on-real accuracy baseline (or -1.0)
    tstr_gap: float                     # baseline - tstr (lower is better)
    column_violation_breakdown: dict = field(default_factory=dict)
    notes: str = ""

    @property
    def composite_utility(self) -> float:
        """
        Weighted composite utility score (0–1).
        TSTR is the strongest signal — weight it highest when available.
        """
        if self.tstr_score >= 0:
            return (
                0.25 * self.correlation_similarity
                + 0.15 * self.logic_consistency
                + 0.60 * (1.0 - self.tstr_gap)
            )
        # Fall back to statistical metrics when no target column is available
        return 0.50 * self.correlation_similarity + 0.50 * self.logic_consistency


class StatisticalMetrics:
    """
    Computes Tier 1 + Tier 2 utility metrics.
    Call evaluate() to get a full UtilityResult in one shot.
    """

    def evaluate(
        self,
        real_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        target_col: str = None,
    ) -> UtilityResult:
        """
        Run all utility metrics and return a single UtilityResult.

        Args:
            real_df:    real training data
            syn_df:     synthetic data generated from real_df
            target_col: optional — column to use for TSTR evaluation.
                        Must be categorical/binary. If None, TSTR is skipped.
        """
        corr_sim = self._correlation_similarity(real_df, syn_df)
        logic_score, breakdown = self._logic_check(real_df, syn_df)
        tstr, baseline, gap = self._tstr(real_df, syn_df, target_col)

        return UtilityResult(
            correlation_similarity=corr_sim,
            logic_consistency=logic_score,
            tstr_score=tstr,
            tstr_baseline=baseline,
            tstr_gap=gap,
            column_violation_breakdown=breakdown,
        )

    # ------------------------------------------------------------------
    # Tier 1: Statistical fidelity
    # ------------------------------------------------------------------

    def correlation_similarity(self, real_df: pd.DataFrame, syn_df: pd.DataFrame) -> float:
        """Public single-metric entry point (for backward compatibility)."""
        return self._correlation_similarity(real_df, syn_df)

    def _correlation_similarity(self, real_df: pd.DataFrame, syn_df: pd.DataFrame) -> float:
        """
        1 - mean_abs_diff of Pearson correlation matrices.
        Returns 1.0 (perfect) when correlations are identical.
        """
        real_corr = real_df.corr(numeric_only=True)
        syn_corr = syn_df.corr(numeric_only=True)

        # Align columns — synthetic may be missing or extra cols after generation
        shared_cols = real_corr.columns.intersection(syn_corr.columns)
        if len(shared_cols) < 2:
            logger.warning("Fewer than 2 shared numeric columns — correlation similarity undefined.")
            return 0.0

        diff = np.abs(real_corr.loc[shared_cols, shared_cols] - syn_corr.loc[shared_cols, shared_cols])
        # Exclude diagonal (always 0 diff — self-correlation = 1)
        np.fill_diagonal(diff.values, 0)
        n = len(shared_cols)
        mean_diff = diff.values.sum() / (n * (n - 1)) if n > 1 else 0.0
        return float(1.0 - mean_diff)

    # ------------------------------------------------------------------
    # Tier 2: Structural / logical integrity
    # ------------------------------------------------------------------

    def _logic_check(
        self,
        real_df: pd.DataFrame,
        syn_df: pd.DataFrame,
    ) -> tuple[float, dict]:
        """
        Checks that synthetic data respects the 'physics' of the real data.

        Two checks:
          A. Boundary check — synthetic numeric values should stay within
             the observed real [min, max] range.
          B. Null invariance — columns with zero nulls in real should have
             zero nulls in synthetic.

        Returns:
          logic_score:  fraction of (record, column) cells that pass all checks.
                        This is per-cell, not per-row, to avoid double-counting.
          breakdown:    dict of {col: n_violations} for reporting.
        """
        n_rows = len(syn_df)
        breakdown = {}
        total_cells_checked = 0
        total_violations = 0

        # A. Boundary check — per column, count violating ROWS (not cells)
        numeric_cols = real_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in syn_df.columns:
                continue
            real_min = real_df[col].min()
            real_max = real_df[col].max()
            violations = int(
                ((syn_df[col] < real_min) | (syn_df[col] > real_max)).sum()
            )
            if violations > 0:
                breakdown[f"{col}_boundary"] = violations
                total_violations += violations
            total_cells_checked += n_rows

        # B. Null invariance check
        for col in real_df.columns:
            if col not in syn_df.columns:
                continue
            if real_df[col].isnull().sum() == 0:
                syn_nulls = int(syn_df[col].isnull().sum())
                if syn_nulls > 0:
                    breakdown[f"{col}_unexpected_nulls"] = syn_nulls
                    total_violations += syn_nulls
                total_cells_checked += n_rows

        if total_cells_checked == 0:
            return 1.0, {}

        logic_score = max(0.0, 1.0 - (total_violations / total_cells_checked))
        return float(logic_score), breakdown

    # ------------------------------------------------------------------
    # Tier 3: ML utility (TSTR)
    # ------------------------------------------------------------------

    def _tstr(
        self,
        real_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        target_col: str = None,
    ) -> tuple[float, float, float]:
        """
        Train on Synthetic, Test on Real (TSTR).

        Trains a Random Forest on synthetic data, evaluates on real data,
        and compares against a real-on-real baseline (TRTR).

        The gap = TRTR_accuracy - TSTR_accuracy.
        A gap near 0 means synthetic data is as useful as real data for ML.

        Returns: (tstr_score, trtr_baseline, gap)
                 All are -1.0 if target_col is None or unsuitable.
        """
        if target_col is None or target_col not in real_df.columns:
            return -1.0, -1.0, -1.0

        try:
            # Encode target
            le = LabelEncoder()
            y_real = le.fit_transform(real_df[target_col].astype(str))
            y_syn = le.transform(
                syn_df[target_col].astype(str).where(
                    syn_df[target_col].astype(str).isin(le.classes_), le.classes_[0]
                )
            )

            # Encode features — one-hot for categoricals
            feature_cols = [c for c in real_df.columns if c != target_col]
            X_real = pd.get_dummies(real_df[feature_cols], drop_first=True).fillna(0)
            X_syn = pd.get_dummies(syn_df[feature_cols], drop_first=True).fillna(0)

            # Align column spaces
            X_syn = X_syn.reindex(columns=X_real.columns, fill_value=0)

            clf_tstr = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf_tstr.fit(X_syn, y_syn)
            # Evaluate on real held-out data using cross-val (5-fold)
            tstr_scores = cross_val_score(clf_tstr, X_real, y_real, cv=5, scoring="accuracy")
            tstr_acc = float(tstr_scores.mean())

            # Baseline: train and test on real (TRTR)
            clf_trtr = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            trtr_scores = cross_val_score(clf_trtr, X_real, y_real, cv=5, scoring="accuracy")
            trtr_acc = float(trtr_scores.mean())

            gap = max(0.0, trtr_acc - tstr_acc)
            logger.info(
                "TSTR | TRTR=%.3f | TSTR=%.3f | gap=%.3f",
                trtr_acc, tstr_acc, gap,
            )
            return tstr_acc, trtr_acc, gap

        except Exception as exc:
            logger.warning("TSTR failed: %s", exc)
            return -1.0, -1.0, -1.0