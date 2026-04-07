"""
evaluation/attacks.py — MIDST Framework

Privacy attack simulations. Every attack returns an AttackResult dataclass
so downstream scoring has a consistent schema to work with.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class AttackResult:
    """
    Unified output schema for every attack.
    Downstream scoring only reads this — never raw attack internals.
    """
    attack_name: str
    risk_score: float               # 0.0–1.0, higher = more privacy risk
    ci_lower: float                 # confidence interval lower bound
    ci_upper: float                 # confidence interval upper bound
    n_attacks_run: int
    passed_threshold: bool          # filled by selector, not here
    notes: str = ""


class PrivacyAttacks:
    """
    Runs three complementary privacy attacks:
      1. Singling-out  (Anonymeter) — can a single record be uniquely identified?
      2. Linkability   (Anonymeter) — can partial attributes re-link to an individual?
      3. CMLA          (custom)     — do synthetic cluster medoids reveal real record structure?

    Args:
        real:       training split of the real dataset
        syn:        synthetic dataset generated from `real`
        control:    held-out real data NOT used for training (Anonymeter baseline)
        n_attacks:  number of attack queries (capped to dataset size automatically)
    """

    def __init__(
        self,
        real: pd.DataFrame,
        syn: pd.DataFrame,
        control: pd.DataFrame,
        n_attacks: int = 500,
    ):
        self.real = real.reset_index(drop=True)
        self.syn = syn.reset_index(drop=True)
        self.control = control.reset_index(drop=True)
        # Cap to the smallest dataset to avoid Anonymeter size errors
        self.n_attacks = min(n_attacks, len(real), len(control), len(syn))
        logger.info("PrivacyAttacks initialised | n_attacks=%d", self.n_attacks)

    # ------------------------------------------------------------------
    # 1. Singling-out attack
    # ------------------------------------------------------------------

    def singling_out(self) -> AttackResult:
        """
        Tests whether an attacker can craft a query that uniquely identifies
        a real individual using only the synthetic dataset.
        """
        logger.info("Running Singling-Out attack ...")
        try:
            from anonymeter.evaluators import SinglingOutEvaluator
            evaluator = SinglingOutEvaluator(
                ori=self.real,
                syn=self.syn,
                control=self.control,
                n_attacks=self.n_attacks,
            )
            evaluator.evaluate(mode="univariate")
            risk = evaluator.risk()
            res = evaluator.results()

            return AttackResult(
                attack_name="SinglingOut",
                risk_score=float(risk.value),
                ci_lower=float(risk.ci[0]) if hasattr(risk, "ci") else 0.0,
                ci_upper=float(risk.ci[1]) if hasattr(risk, "ci") else 1.0,
                n_attacks_run=self.n_attacks,
                passed_threshold=False,  # filled by selector
                notes=f"n_success={getattr(res, 'n_success', 'N/A')}",
            )
        except Exception as exc:
            logger.error("Singling-Out attack failed: %s", exc)
            return self._failed_result("SinglingOut", str(exc))

    # ------------------------------------------------------------------
    # 2. Linkability attack
    # ------------------------------------------------------------------

    def linkability(self, n_aux_cols: Optional[int] = None) -> AttackResult:
        """
        Tests whether an attacker who knows a subset of a person's attributes
        can link them across the real and synthetic datasets.

        The column set is split into two disjoint halves.
        Passing ALL columns (as many implementations do) defeats the attack —
        the attacker already has the full record, so linkability is trivially 1.

        Args:
            n_aux_cols: how many columns to give the attacker.
                        Defaults to half of all columns.
        """
        logger.info("Running Linkability attack ...")
        try:
            from anonymeter.evaluators import LinkabilityEvaluator

            cols = list(self.real.columns)
            n = n_aux_cols or max(1, len(cols) // 2)

            # Two disjoint halves — attacker sees the first half,
            # the second half is what we're trying to protect
            aux_cols = [cols[:n], cols[n:]]

            evaluator = LinkabilityEvaluator(
                ori=self.real,
                syn=self.syn,
                control=self.control,
                aux_cols=aux_cols,
                n_attacks=self.n_attacks,
            )
            evaluator.evaluate(n_jobs=-1)
            risk = evaluator.risk()
            res = evaluator.results()

            return AttackResult(
                attack_name="Linkability",
                risk_score=float(risk.value),
                ci_lower=float(risk.ci[0]) if hasattr(risk, "ci") else 0.0,
                ci_upper=float(risk.ci[1]) if hasattr(risk, "ci") else 1.0,
                n_attacks_run=self.n_attacks,
                passed_threshold=False,
                notes=f"aux_cols={n}/{len(cols)} | n_success={getattr(res, 'n_success', 'N/A')}",
            )
        except Exception as exc:
            logger.error("Linkability attack failed: %s", exc)
            return self._failed_result("Linkability", str(exc))

    # ------------------------------------------------------------------
    # 3. CMLA — Cluster-Medoid Leakage Attack (custom)
    # ------------------------------------------------------------------

    def cmla_leakage(
        self,
        n_clusters: int = 10,
        distance_percentile: float = 5.0,
    ) -> AttackResult:
        """
        Custom structural leakage attack.

        Concept: cluster the REAL data into neighbourhoods, find the true
        medoid of each neighbourhood (the most central real record), then
        measure how close synthetic points come to those medoids.

        If synthetic points land very close to real medoids, the generator
        has memorised the structural 'skeleton' of the real data — a form
        of leakage that Singling-Out and Linkability don't catch.

        Key fix vs. naive implementations:
          - Data is z-score normalised before distance calculation.
            Without this, high-variance columns (e.g. capital-gain 0–99k)
            dominate the distance metric and the attack is meaningless.
          - The leakage threshold is data-adaptive: a record is "too close"
            if its distance falls below the `distance_percentile`-th percentile
            of all pairwise real-to-real distances. This replaces the broken
            hardcoded 0.05 threshold.

        Args:
            n_clusters:           number of real-data neighbourhoods
            distance_percentile:  bottom X% of real-real distances = "too close"
        """
        logger.info("Running CMLA attack ...")
        try:
            # --- Prep: numeric only, z-score normalise ---
            real_num = self.real.select_dtypes(include=[np.number]).copy()
            syn_num = self.syn[real_num.columns].copy()

            if real_num.shape[1] == 0:
                return self._failed_result("CMLA", "No numeric columns found.")

            scaler = StandardScaler()
            real_scaled = scaler.fit_transform(real_num.fillna(0))
            syn_scaled = scaler.transform(syn_num.fillna(0))

            # --- Step 1: cluster REAL data ---
            k = min(n_clusters, len(real_scaled) - 1)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(real_scaled)

            # --- Step 2: find TRUE medoid of each real cluster ---
            # (the actual real data point closest to the centroid)
            real_medoids = []
            for cluster_id in range(k):
                mask = kmeans.labels_ == cluster_id
                cluster_pts = real_scaled[mask]
                if len(cluster_pts) == 0:
                    continue
                centroid = cluster_pts.mean(axis=0).reshape(1, -1)
                closest_idx, _ = pairwise_distances_argmin_min(centroid, cluster_pts)
                real_medoids.append(cluster_pts[closest_idx[0]])

            real_medoids = np.array(real_medoids)

            # --- Step 3: adaptive threshold ---
            # Sample real-real pairwise distances to calibrate "too close"
            sample_size = min(500, len(real_scaled))
            sample_idx = np.random.choice(len(real_scaled), sample_size, replace=False)
            real_sample = real_scaled[sample_idx]
            _, real_real_dists = pairwise_distances_argmin_min(real_sample, real_scaled)
            threshold = np.percentile(real_real_dists, distance_percentile)

            # --- Step 4: measure how close synthetic points are to real medoids ---
            _, syn_to_medoid_dists = pairwise_distances_argmin_min(syn_scaled, real_medoids)
            n_leaking = int(np.sum(syn_to_medoid_dists < threshold))
            asr = n_leaking / len(syn_scaled)

            logger.info(
                "CMLA | threshold=%.4f (p%.0f of real-real dists) | "
                "leaking=%d/%d (ASR=%.3f)",
                threshold, distance_percentile, n_leaking, len(syn_scaled), asr,
            )

            return AttackResult(
                attack_name="CMLA",
                risk_score=float(asr),
                ci_lower=max(0.0, float(asr) - 0.02),   # approximate ±2% for ASR
                ci_upper=min(1.0, float(asr) + 0.02),
                n_attacks_run=len(syn_scaled),
                passed_threshold=False,
                notes=(
                    f"k={k} | threshold={threshold:.4f} "
                    f"(p{distance_percentile:.0f} real-real) | "
                    f"leaking={n_leaking}/{len(syn_scaled)}"
                ),
            )

        except Exception as exc:
            logger.error("CMLA attack failed: %s", exc)
            return self._failed_result("CMLA", str(exc))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _failed_result(name: str, reason: str) -> AttackResult:
        """Return a max-risk result when an attack errors out — fail safe."""
        logger.warning("Attack '%s' failed — defaulting to risk=1.0. Reason: %s", name, reason)
        return AttackResult(
            attack_name=name,
            risk_score=1.0,
            ci_lower=1.0,
            ci_upper=1.0,
            n_attacks_run=0,
            passed_threshold=False,
            notes=f"FAILED: {reason}",
        )