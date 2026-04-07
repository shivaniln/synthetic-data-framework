"""
main.py — MIDST Framework controller

Orchestrates the full pipeline:
  1. Load & clean data
  2. Train all configured generators
  3. Run privacy attacks + utility metrics on each synthetic dataset
  4. Rank models via Pareto scoring and export results

All tuneable values live in RUN_CONFIG below — nothing is hardcoded in logic.
"""

import logging
import os
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import DataLoader, LoaderConfig
from src.models.base_generator import build_generator
from src.evaluation.attacks import PrivacyAttacks
from src.evaluation.metrics import StatisticalMetrics
from src.utils.visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("midst.main")


# =============================================================================
# RUN CONFIGURATION
# Move this to a YAML file (e.g. config.yaml) in production.
# =============================================================================

RUN_CONFIG = {
    # Data
    "max_rows": 2000,               # subsample cap (None = no cap)
    "train_ratio": 0.5,             # fraction used for training; remainder = control
    "random_seed": 42,

    # Models to benchmark and their hyperparameters
    "models": {
        "copula":  {},
        "ctgan":   {"epochs": 300, "batch_size": 500},
        "tvae":    {"epochs": 300, "batch_size": 500},
        # Uncomment when smartnoise-synth is installed:
        # "privbayes": {"epsilon": 1.0},
        # "pategan":   {"epsilon": 1.0},
    },

    # Privacy thresholds (all three attacks must pass)
    "privacy_thresholds": {
        "singling_out":  0.10,      # max acceptable risk (0–1)
        "linkability":   0.10,
        "cmla":          0.10,
    },

    # Utility thresholds
    "utility_thresholds": {
        "logic_consistency":      0.70,
        "correlation_similarity": 0.70,
        # tstr_gap is optional — only checked if target_col is set
        "tstr_gap_max":           0.10,
    },

    # Scoring weights for Pareto composite score
    # privacy_score = 1 - max(risk_scores)
    # utility_score = UtilityResult.composite_utility
    "score_weights": {
        "privacy": 0.50,
        "utility": 0.50,
    },

    # TSTR target column (set to None to skip TSTR)
    "target_col": None,             # e.g. "income" for Adult Census

    # Attack config
    "n_attacks": 500,               # reduce to 50–100 for quick demo runs

    # Output
    "output_dir": "data/output",
}


# =============================================================================
# Pipeline
# =============================================================================

def run_framework(filename: str, config: dict = RUN_CONFIG) -> None:
    t_start = time.perf_counter()
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load & clean
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("MIDST Framework — starting run")
    logger.info("=" * 60)

    input_path = Path("data") / "input" / filename
    loader = DataLoader(str(input_path), config=LoaderConfig())
    result = loader.load_and_clean()
    real_df = result.df

    logger.info(
        "Loaded: %s | shape %s → %s | dropped cols: %s",
        filename, result.original_shape, result.cleaned_shape,
        result.dropped_columns or "none",
    )

    # Optional subsampling
    max_rows = config.get("max_rows")
    if max_rows and len(real_df) > max_rows:
        logger.warning("Subsampling from %d to %d rows.", len(real_df), max_rows)
        real_df = real_df.sample(n=max_rows, random_state=config["random_seed"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Train / control split — SHUFFLE FIRST
    # ------------------------------------------------------------------
    real_df = real_df.sample(frac=1, random_state=config["random_seed"]).reset_index(drop=True)
    split = int(len(real_df) * config["train_ratio"])
    train_df = real_df.iloc[:split].reset_index(drop=True)
    control_df = real_df.iloc[split:].reset_index(drop=True)
    logger.info("Split: train=%d rows | control=%d rows", len(train_df), len(control_df))

    # ------------------------------------------------------------------
    # 3. Benchmarking loop
    # ------------------------------------------------------------------
    metrics_engine = StatisticalMetrics()
    visualizer = Visualizer()
    records = []
    synthetic_datasets = {}

    for model_name, model_kwargs in config["models"].items():
        logger.info("-" * 50)
        logger.info("Benchmarking: %s", model_name.upper())

        try:
            gen = build_generator(model_name, result.metadata, **model_kwargs)
        except ImportError as exc:
            logger.warning("Skipping %s — dependency not installed: %s", model_name, exc)
            continue

        # Train
        try:
            gen.fit(train_df)
        except Exception as exc:
            logger.error("Training failed for %s: %s", model_name, exc)
            records.append(_failed_record(model_name, f"Training failed: {exc}"))
            continue

        # Sample
        try:
            gen_result = gen.sample(len(train_df))
            syn_df = gen_result.synthetic_df
            synthetic_datasets[model_name] = syn_df
        except Exception as exc:
            logger.error("Sampling failed for %s: %s", model_name, exc)
            records.append(_failed_record(model_name, f"Sampling failed: {exc}"))
            continue

        # Privacy attacks
        attacker = PrivacyAttacks(
            real=train_df,
            syn=syn_df,
            control=control_df,
            n_attacks=config["n_attacks"],
        )
        so_result   = attacker.singling_out()
        link_result = attacker.linkability()
        cmla_result = attacker.cmla_leakage()

        # Utility metrics
        utility = metrics_engine.evaluate(
            real_df=train_df,
            syn_df=syn_df,
            target_col=config.get("target_col"),
        )

        # Threshold checks
        pt = config["privacy_thresholds"]
        ut = config["utility_thresholds"]

        privacy_pass = (
            so_result.risk_score   <= pt["singling_out"]
            and link_result.risk_score <= pt["linkability"]
            and cmla_result.risk_score <= pt["cmla"]
        )
        utility_pass = (
            utility.logic_consistency      >= ut["logic_consistency"]
            and utility.correlation_similarity >= ut["correlation_similarity"]
        )
        if config.get("target_col") and utility.tstr_gap >= 0:
            utility_pass = utility_pass and (utility.tstr_gap <= ut["tstr_gap_max"])

        overall_pass = privacy_pass and utility_pass

        # Composite score for ranking
        w = config["score_weights"]
        privacy_score = 1.0 - max(
            so_result.risk_score, link_result.risk_score, cmla_result.risk_score
        )
        composite_score = (
            w["privacy"] * privacy_score
            + w["utility"] * utility.composite_utility
        )

        record = {
            "model":                  model_name,
            # Privacy
            "singling_out_risk":      round(so_result.risk_score, 4),
            "singling_out_ci":        f"[{so_result.ci_lower:.3f}, {so_result.ci_upper:.3f}]",
            "linkability_risk":       round(link_result.risk_score, 4),
            "linkability_ci":         f"[{link_result.ci_lower:.3f}, {link_result.ci_upper:.3f}]",
            "cmla_risk":              round(cmla_result.risk_score, 4),
            # Utility
            "correlation_similarity": round(utility.correlation_similarity, 4),
            "logic_consistency":      round(utility.logic_consistency, 4),
            "tstr_score":             round(utility.tstr_score, 4),
            "tstr_baseline":          round(utility.tstr_baseline, 4),
            "tstr_gap":               round(utility.tstr_gap, 4),
            "composite_utility":      round(utility.composite_utility, 4),
            # Scoring
            "privacy_score":          round(privacy_score, 4),
            "composite_score":        round(composite_score, 4),
            "privacy_pass":           privacy_pass,
            "utility_pass":           utility_pass,
            "overall_pass":           overall_pass,
            # Config
            "model_config":           json.dumps(model_kwargs),
            "violation_breakdown":    json.dumps(utility.column_violation_breakdown),
        }
        records.append(record)

        logger.info(
            "%s | SO=%.3f | Link=%.3f | CMLA=%.3f | CorSim=%.3f | "
            "Logic=%.3f | Composite=%.3f | PASS=%s",
            model_name.upper(),
            so_result.risk_score, link_result.risk_score, cmla_result.risk_score,
            utility.correlation_similarity, utility.logic_consistency,
            composite_score, overall_pass,
        )

    # ------------------------------------------------------------------
    # 4. Ranking & selection
    # ------------------------------------------------------------------
    if not records:
        logger.error("No models completed benchmarking. Aborting.")
        return

    report_df = pd.DataFrame(records).sort_values("composite_score", ascending=False)

    logger.info("=" * 60)
    logger.info("FINAL AUDIT REPORT")
    logger.info("=" * 60)
    print(report_df[["model", "singling_out_risk", "linkability_risk", "cmla_risk",
                      "correlation_similarity", "logic_consistency",
                      "composite_score", "overall_pass"]].to_string(index=False))

    # Best model = highest composite score among passing models
    passing = report_df[report_df["overall_pass"] == True]

    if not passing.empty:
        best_row = passing.iloc[0]
        best_name = best_row["model"]
        logger.info("RECOMMENDED: %s (composite=%.4f)", best_name, best_row["composite_score"])
    else:
        # Pareto fallback — best model even if no thresholds passed
        best_row = report_df.iloc[0]
        best_name = best_row["model"]
        logger.warning(
            "NO model passed all thresholds. "
            "Best available: %s (composite=%.4f). "
            "Review thresholds or improve model quality before deployment.",
            best_name, best_row["composite_score"],
        )

    # ------------------------------------------------------------------
    # 5. Export
    # ------------------------------------------------------------------
    report_df["is_recommended"] = report_df["model"] == best_name
    report_path = output_dir / "final_audit_report.csv"
    report_df.to_csv(report_path, index=False)
    logger.info("Audit report saved: %s", report_path)

    if best_name in synthetic_datasets:
        syn_path = output_dir / f"{best_name}_best_synthetic.csv"
        synthetic_datasets[best_name].to_csv(syn_path, index=False)
        logger.info("Best synthetic dataset saved: %s", syn_path)
        visualizer.plot_winner_comparison(train_df, synthetic_datasets[best_name], best_name)

    # Machine-readable JSON summary
    summary = {
        "recommended_model": best_name,
        "threshold_passed": bool(not passing.empty),
        "run_config": {k: v for k, v in config.items() if k != "models"},
        "results": report_df.to_dict(orient="records"),
        "total_runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    json_path = output_dir / "audit_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("JSON summary saved: %s", json_path)
    logger.info("Total runtime: %.1fs", time.perf_counter() - t_start)


# =============================================================================
# Helpers
# =============================================================================

def _failed_record(model_name: str, reason: str) -> dict:
    """Placeholder row for a model that errored out during benchmarking."""
    return {
        "model": model_name,
        "singling_out_risk": 1.0, "singling_out_ci": "N/A",
        "linkability_risk": 1.0, "linkability_ci": "N/A",
        "cmla_risk": 1.0,
        "correlation_similarity": 0.0, "logic_consistency": 0.0,
        "tstr_score": -1.0, "tstr_baseline": -1.0, "tstr_gap": -1.0,
        "composite_utility": 0.0, "privacy_score": 0.0, "composite_score": 0.0,
        "privacy_pass": False, "utility_pass": False, "overall_pass": False,
        "model_config": "{}", "violation_breakdown": "{}",
        "notes": reason,
    }


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    run_framework("test.csv")