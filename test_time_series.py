"""
Standalone test for the MIDST time-series audit pipeline.

Run:
    python test_time_series.py

It creates a small demo electricity-demand dataset if one does not
already exist, then runs the time-series audit without using Flask.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.runner import TimeSeriesAuditRunner


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

DEMO_FILE = INPUT_DIR / "time_series_demo.csv"


def emit(level: str, message: str) -> None:
    """Print runner logs in a readable terminal format."""
    print(f"[{level.upper()}] {message}")


def create_demo_dataset() -> None:
    """Create a timestamped electricity-demand series for testing."""
    if DEMO_FILE.exists():
        print(f"Using existing demo dataset: {DEMO_FILE.name}")
        return

    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    rows = 240

    timestamps = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=rows,
        freq="h",
    )

    index = np.arange(rows)

    daily_pattern = 20 * np.sin(2 * np.pi * index / 24)
    weekly_pattern = 10 * np.sin(2 * np.pi * index / (24 * 7))
    baseline = 150

    electricity_demand = (
        baseline
        + daily_pattern
        + weekly_pattern
        + rng.normal(0, 4, rows)
    )

    temperature = (
        28
        + 4 * np.sin(2 * np.pi * index / 24)
        + rng.normal(0, 1, rows)
    )

    demo_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "electricity_demand": electricity_demand.round(2),
            "temperature": temperature.round(2),
        }
    )

    demo_df.to_csv(DEMO_FILE, index=False)

    print(f"Created demo dataset: {DEMO_FILE.name}")


def main() -> None:
    create_demo_dataset()

    runner = TimeSeriesAuditRunner(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        emit=emit,
    )

    config = {
        "filename": DEMO_FILE.name,
        "timestamp_column": "timestamp",
        "max_rows": 240,
        "train_ratio": 0.5,
        "models": {
            "block_bootstrap": {
                "block_size": 24,
                "noise_scale": 0.03,
                "window_size": 12,
                "max_queries": 100,
                "random_seed": 42,
            }
        },
        "privacy_thresholds": {
            "sequence_similarity": 0.10,
        },
        "utility_thresholds": {
            "composite_utility": 0.70,
        },
        "score_weights": {
            "privacy": 0.5,
            "utility": 0.5,
        },
    }

    summary = runner.run(config)

    print("\n" + "=" * 55)
    print("TIME-SERIES AUDIT COMPLETE")
    print("=" * 55)
    print(f"Recommended model: {summary['recommended_model']}")
    print(f"Passed thresholds: {summary['threshold_passed']}")
    print(f"Runtime: {summary['total_runtime_seconds']} seconds")

    print("\nResults:")

    for result in summary["results"]:
        print(
            f"- {result['model']}: "
            f"risk={result['sequence_similarity_risk']:.1%}, "
            f"utility={result['composite_utility']:.1%}, "
            f"score={result['composite_score']:.3f}"
        )


if __name__ == "__main__":
    main()