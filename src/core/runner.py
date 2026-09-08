"""
Shared audit runners for MIDST.

The existing tabular audit remains in app.py.
This module contains the Phase 2 time-series audit runner.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from src.core.registry import get_modality


LogFunction = Callable[[str, str], None]


class TimeSeriesAuditRunner:
    """
    Run generation, privacy evaluation, utility evaluation,
    ranking, and export for time-series datasets.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        emit: LogFunction,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.emit = emit

    def run(self, config: dict) -> dict:
        """Run a complete time-series audit and return its summary."""
        started_at = time.perf_counter()

        filename = config["filename"]
        file_path = self.input_dir / filename

        modality = get_modality("time_series")

        self.emit(
            "info",
            f"Loading time-series dataset: {filename} ...",
        )

        loader = modality.create_loader(
            str(file_path),
            timestamp_column=config.get("timestamp_column"),
        )

        load_result = loader.load_and_prepare()
        full_df = load_result.df

        self.emit(
            "info",
            f"Detected timestamp column: "
            f"{load_result.timestamp_column}",
        )

        self.emit(
            "info",
            "Detected value columns: "
            + ", ".join(load_result.value_columns),
        )

        max_rows = int(config.get("max_rows", len(full_df)))

        if len(full_df) > max_rows:
            # Preserve chronology by taking recent observations.
            full_df = full_df.iloc[-max_rows:].reset_index(drop=True)

            self.emit(
                "warn",
                f"Using the latest {max_rows} rows "
                "to preserve time order.",
            )

        train_ratio = float(config.get("train_ratio", 0.5))
        split_index = int(len(full_df) * train_ratio)

        train_df = full_df.iloc[:split_index].reset_index(drop=True)
        control_df = full_df.iloc[split_index:].reset_index(drop=True)

        if len(train_df) < 24 or len(control_df) < 12:
            raise ValueError(
                "The selected dataset is too small for time-series "
                "auditing. Use at least 48 valid time-ordered rows."
            )

        self.emit(
            "info",
            f"Chronological split: {len(train_df)} training | "
            f"{len(control_df)} control rows",
        )

        models_config = config.get(
            "models",
            {
                "block_bootstrap": {
                    "block_size": 24,
                    "noise_scale": 0.03,
                    "window_size": 12,
                    "max_queries": 100,
                    "random_seed": 42,
                },
                "fourier_surrogate": {
                    "harmonics": 24,
                    "noise_scale": 0.03,
                    "window_size": 12,
                    "max_queries": 100,
                    "random_seed": 42,
                },
            },
        )

        privacy_limit = float(
            config.get("privacy_thresholds", {}).get(
                "sequence_similarity",
                0.10,
            )
        )

        utility_limit = float(
            config.get("utility_thresholds", {}).get(
                "composite_utility",
                0.70,
            )
        )

        score_weights = config.get(
            "score_weights",
            {
                "privacy": 0.5,
                "utility": 0.5,
            },
        )

        privacy_weight = float(
            score_weights.get("privacy", 0.5)
        )

        utility_weight = float(
            score_weights.get("utility", 0.5)
        )

        records = []
        synthetic_datasets = {}

        for model_name, model_config in models_config.items():
            display_name = model_name.replace("_", " ").upper()

            self.emit("info", f"--- {display_name} ---")

            try:
                generator = modality.create_generator(
                    model_name=model_name,
                    timestamp_column=load_result.timestamp_column,
                    value_columns=load_result.value_columns,
                    **model_config,
                )

                self.emit(
                    "info",
                    f"Training {display_name} ...",
                )

                generator.fit(train_df)

                self.emit(
                    "info",
                    f"Generating {len(train_df)} synthetic "
                    "time-series rows ...",
                )

                generation_result = generator.sample(len(train_df))
                synthetic_df = generation_result.synthetic_df

                synthetic_datasets[model_name] = synthetic_df

            except Exception as exc:
                self.emit(
                    "error",
                    f"{display_name} generation failed: {exc}",
                )
                continue

            try:
                self.emit(
                    "info",
                    "Evaluating time-series privacy risk ...",
                )

                privacy_evaluator = modality.create_privacy_evaluator(
                    control_df=control_df,
                    synthetic_df=synthetic_df,
                    value_columns=load_result.value_columns,
                    window_size=int(
                        model_config.get("window_size", 12)
                    ),
                    max_queries=int(
                        model_config.get("max_queries", 100)
                    ),
                    random_seed=int(
                        model_config.get("random_seed", 42)
                    ),
                )

                privacy_result = (
                    privacy_evaluator.sequence_similarity_risk()
                )

                self.emit(
                    "info",
                    "  Sequence similarity risk: "
                    f"{privacy_result.risk_score:.1%}",
                )

                self.emit(
                    "info",
                    "Evaluating time-series utility ...",
                )

                metrics_engine = (
                    modality.create_utility_evaluator()
                )

                utility_result = metrics_engine.evaluate(
                    train_df,
                    synthetic_df,
                    load_result.value_columns,
                )

                self.emit(
                    "info",
                    f"  Trend similarity: "
                    f"{utility_result.trend_similarity:.1%} | "
                    f"Autocorrelation: "
                    f"{utility_result.autocorrelation_similarity:.1%} | "
                    f"Distribution: "
                    f"{utility_result.distribution_similarity:.1%}",
                )

            except Exception as exc:
                self.emit(
                    "error",
                    f"{display_name} evaluation failed: {exc}",
                )
                continue

            privacy_pass = (
                privacy_result.risk_score <= privacy_limit
            )

            utility_pass = (
                utility_result.composite_utility >= utility_limit
            )

            overall_pass = privacy_pass and utility_pass

            privacy_score = 1.0 - privacy_result.risk_score

            composite_score = (
                privacy_weight * privacy_score
                + utility_weight * utility_result.composite_utility
            )

            status = "PASSED" if overall_pass else "FAILED"

            self.emit(
                "info" if overall_pass else "warn",
                f"{display_name} -> {status} | "
                f"score={composite_score:.3f}",
            )

            records.append(
                {
                    "model": model_name,
                    "sequence_similarity_risk": round(
                        privacy_result.risk_score,
                        4,
                    ),
                    "trend_similarity": round(
                        utility_result.trend_similarity,
                        4,
                    ),
                    "autocorrelation_similarity": round(
                        utility_result.autocorrelation_similarity,
                        4,
                    ),
                    "distribution_similarity": round(
                        utility_result.distribution_similarity,
                        4,
                    ),
                    "composite_utility": round(
                        utility_result.composite_utility,
                        4,
                    ),
                    "privacy_score": round(
                        privacy_score,
                        4,
                    ),
                    "composite_score": round(
                        composite_score,
                        4,
                    ),
                    "privacy_pass": privacy_pass,
                    "utility_pass": utility_pass,
                    "overall_pass": overall_pass,
                    "privacy_notes": privacy_result.notes,
                    "model_config": json.dumps(model_config),
                }
            )

        if not records:
            raise ValueError(
                "No selected time-series generator completed successfully."
            )

        report_df = pd.DataFrame(records).sort_values(
            "composite_score",
            ascending=False,
        )

        passing_df = report_df[
            report_df["overall_pass"]
        ]

        if not passing_df.empty:
            best_model = passing_df.iloc[0]["model"]

            self.emit(
                "info",
                f"Recommended: {best_model.replace('_', ' ').upper()}",
            )

        else:
            best_model = report_df.iloc[0]["model"]

            self.emit(
                "warn",
                "No model passed all thresholds. "
                "Best available: "
                f"{best_model.replace('_', ' ').upper()}.",
            )

        report_df["is_recommended"] = (
            report_df["model"] == best_model
        )

        report_path = (
            self.output_dir / "time_series_audit_report.csv"
        )

        report_df.to_csv(report_path, index=False)

        self.emit(
            "info",
            "Saved: time_series_audit_report.csv",
        )

        for model_name, synthetic_df in synthetic_datasets.items():
            output_path = (
                self.output_dir
                / f"{model_name}_time_series_synthetic.csv"
            )

            synthetic_df.to_csv(output_path, index=False)

            self.emit(
                "info",
                f"Saved: {output_path.name}",
            )

        elapsed = round(
            time.perf_counter() - started_at,
            2,
        )

        summary = {
            "modality": "time_series",
            "recommended_model": best_model,
            "threshold_passed": bool(not passing_df.empty),
            "total_runtime_seconds": elapsed,
            "dataset_details": {
                "timestamp_column": load_result.timestamp_column,
                "value_columns": load_result.value_columns,
                "original_shape": load_result.original_shape,
                "cleaned_shape": load_result.cleaned_shape,
                "training_rows": len(train_df),
                "control_rows": len(control_df),
            },
            "results": report_df.to_dict(orient="records"),
            "run_config": config,
        }

        summary_path = (
            self.output_dir / "time_series_audit_summary.json"
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as summary_file:
            json.dump(
                summary,
                summary_file,
                indent=2,
                default=str,
            )

        self.emit(
            "info",
            f"Time-series audit complete in {elapsed}s",
        )

        return summary

class ImageAuditRunner:
    """
    Compare an original image folder against a synthetic image folder.

    This runner audits already generated images. It does not train or
    run an image-generation model in the current Phase 2 version.
    """

    def __init__(
        self,
        output_dir: Path,
        emit: LogFunction,
    ):
        self.output_dir = output_dir
        self.emit = emit

    def run(self, config: dict) -> dict:
        """Run a complete local image privacy and utility audit."""
        started_at = time.perf_counter()

        original_folder = config.get("original_folder")
        synthetic_folder = config.get("synthetic_folder")

        if not original_folder or not synthetic_folder:
            raise ValueError(
                "Image auditing requires both original_folder and "
                "synthetic_folder paths."
            )

        max_images = config.get("max_images")

        if max_images is not None:
            max_images = int(max_images)

        hash_distance_limit = int(
            config.get("hash_distance_limit", 6)
        )

        privacy_limit = float(
            config.get("privacy_thresholds", {}).get(
                "near_duplicate",
                0.10,
            )
        )

        utility_limit = float(
            config.get("utility_thresholds", {}).get(
                "composite_utility",
                0.70,
            )
        )

        score_weights = config.get(
            "score_weights",
            {
                "privacy": 0.5,
                "utility": 0.5,
            },
        )

        privacy_weight = float(
            score_weights.get("privacy", 0.5)
        )

        utility_weight = float(
            score_weights.get("utility", 0.5)
        )

        modality = get_modality("image")

        self.emit(
            "info",
            "Loading original image dataset ...",
        )

        original_loader = modality.create_loader(
            folder_path=original_folder,
            max_images=max_images,
        )

        original_dataset = original_loader.load()

        self.emit(
            "info",
            f"Loaded {len(original_dataset.paths)} original images.",
        )

        self.emit(
            "info",
            "Loading synthetic image dataset ...",
        )

        synthetic_loader = modality.create_loader(
            folder_path=synthetic_folder,
            max_images=max_images,
        )

        synthetic_dataset = synthetic_loader.load()

        self.emit(
            "info",
            f"Loaded {len(synthetic_dataset.paths)} synthetic images.",
        )

        self.emit(
            "info",
            "Evaluating image utility ...",
        )

        utility_evaluator = (
            modality.create_utility_evaluator()
        )

        utility_result = utility_evaluator.evaluate(
            original_dataset,
            synthetic_dataset,
        )

        self.emit(
            "info",
            f"  Colour similarity: "
            f"{utility_result.colour_similarity:.1%} | "
            f"Visual similarity: "
            f"{utility_result.visual_similarity:.1%} | "
            f"Structure similarity: "
            f"{utility_result.structure_similarity:.1%}",
        )

        self.emit(
            "info",
            "Evaluating near-duplicate privacy risk ...",
        )

        privacy_evaluator = modality.create_privacy_evaluator(
            original=original_dataset,
            synthetic=synthetic_dataset,
            hash_distance_limit=hash_distance_limit,
        )

        privacy_result = privacy_evaluator.near_duplicate_risk()

        self.emit(
            "info",
            f"  Near-duplicate risk: "
            f"{privacy_result.near_duplicate_risk:.1%} | "
            f"Matches: {privacy_result.duplicate_count}",
        )

        privacy_pass = (
            privacy_result.near_duplicate_risk <= privacy_limit
        )

        utility_pass = (
            utility_result.composite_utility >= utility_limit
        )

        overall_pass = privacy_pass and utility_pass

        privacy_score = (
            1.0 - privacy_result.near_duplicate_risk
        )

        composite_score = (
            privacy_weight * privacy_score
            + utility_weight * utility_result.composite_utility
        )

        status = "PASSED" if overall_pass else "FAILED"

        self.emit(
            "info" if overall_pass else "warn",
            f"IMAGE AUDIT -> {status} | "
            f"score={composite_score:.3f}",
        )

        result_record = {
            "model": "image_audit",
            "near_duplicate_risk": round(
                privacy_result.near_duplicate_risk,
                4,
            ),
            "duplicate_count": privacy_result.duplicate_count,
            "colour_similarity": round(
                utility_result.colour_similarity,
                4,
            ),
            "visual_similarity": round(
                utility_result.visual_similarity,
                4,
            ),
            "structure_similarity": round(
                utility_result.structure_similarity,
                4,
            ),
            "composite_utility": round(
                utility_result.composite_utility,
                4,
            ),
            "privacy_score": round(
                privacy_score,
                4,
            ),
            "composite_score": round(
                composite_score,
                4,
            ),
            "privacy_pass": privacy_pass,
            "utility_pass": utility_pass,
            "overall_pass": overall_pass,
            "privacy_notes": privacy_result.notes,
        }

        report_df = pd.DataFrame([result_record])

        report_path = self.output_dir / "image_audit_report.csv"

        report_df.to_csv(report_path, index=False)

        elapsed = round(
            time.perf_counter() - started_at,
            2,
        )

        summary = {
            "modality": "image",
            "recommended_model": "image_audit",
            "threshold_passed": overall_pass,
            "total_runtime_seconds": elapsed,
            "dataset_details": {
                "original_images": len(original_dataset.paths),
                "synthetic_images": len(synthetic_dataset.paths),
                "max_images_used": max_images,
            },
            "results": [result_record],
            "run_config": config,
        }

        summary_path = (
            self.output_dir / "image_audit_summary.json"
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as summary_file:
            json.dump(
                summary,
                summary_file,
                indent=2,
                default=str,
            )

        self.emit(
            "info",
            "Saved: image_audit_report.csv",
        )

        self.emit(
            "info",
            f"Image audit complete in {elapsed}s",
        )

        return summary