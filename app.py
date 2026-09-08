"""
app.py - MIDST Framework Flask Backend

Supports:
1. Tabular CSV upload and audit
2. Time-series CSV upload and audit
3. Image-folder audit through two ZIP uploads
4. Live Server-Sent Event logs
5. Local result and output-file downloads
"""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
import traceback
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from src.core.image_api import register_image_routes
from src.core.runner import TimeSeriesAuditRunner
from src.data_loader import DataLoader, LoaderConfig
from src.evaluation.attacks import PrivacyAttacks
from src.evaluation.metrics import StatisticalMetrics
from src.models.base_generator import GENERATOR_REGISTRY, build_generator
from src.utils.visualizer import Visualizer


# ---------------------------------------------------------------------------
# JSON safety
# ---------------------------------------------------------------------------

class SafeEncoder(json.JSONEncoder):
    """Convert NumPy values and NaN/Infinity values into valid JSON."""

    def iterencode(self, obj, _one_shot=False):
        return super().iterencode(self.clean(obj), _one_shot)

    def clean(self, obj):
        if isinstance(obj, dict):
            return {key: self.clean(value) for key, value in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [self.clean(value) for value in obj]

        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj

        # Handles NumPy scalar values such as np.float64 and np.int64.
        if hasattr(obj, "item"):
            try:
                return self.clean(obj.item())
            except Exception:
                return str(obj)

        return obj


def safe_dumps(obj: dict) -> str:
    """Return a JSON-safe representation of a result object."""
    return json.dumps(obj, cls=SafeEncoder)


# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="frontend", static_url_path="")

INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

run_state = {
    "running": False,
    "last_result": None,
}

_subscribers: list[queue.Queue] = []
_subscribers_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Logging and SSE helpers
# ---------------------------------------------------------------------------

def emit(level: str, message: str) -> None:
    """Broadcast one log event to all connected browser clients."""
    item = {
        "level": level,
        "message": message,
        "ts": time.strftime("%H:%M:%S"),
    }

    with _subscribers_lock:
        for subscriber in list(_subscribers):
            subscriber.put(item)

    logger_level = {
        "info": logging.INFO,
        "warn": logging.WARNING,
        "error": logging.ERROR,
        "done": logging.INFO,
    }.get(level, logging.INFO)

    logging.getLogger("midst.app").log(logger_level, message)


def add_subscriber() -> queue.Queue:
    subscriber: queue.Queue = queue.Queue()

    with _subscribers_lock:
        _subscribers.append(subscriber)

    return subscriber


def remove_subscriber(subscriber: queue.Queue) -> None:
    with _subscribers_lock:
        try:
            _subscribers.remove(subscriber)
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Register image-specific API routes
# ---------------------------------------------------------------------------

register_image_routes(
    app=app,
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    run_state=run_state,
    emit=emit,
    safe_dumps=safe_dumps,
)


# ---------------------------------------------------------------------------
# General routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/status")
def get_status():
    return jsonify({"running": run_state["running"]})


@app.route("/api/logs")
def stream_logs():
    """Send live audit logs to the frontend through Server-Sent Events."""
    subscriber = add_subscriber()

    def generate():
        try:
            yield "retry: 1000\n\n"

            while True:
                try:
                    item = subscriber.get(timeout=30)
                    yield f"data: {json.dumps(item)}\n\n"

                    if item.get("level") == "done":
                        return

                except queue.Empty:
                    yield ": keepalive\n\n"

        finally:
            remove_subscriber(subscriber)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# CSV upload route for tabular and time-series data
# ---------------------------------------------------------------------------

def allowed_csv(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "csv"


@app.route("/api/upload", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file was uploaded."}), 400

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        return jsonify({"error": "Choose a CSV file first."}), 400

    if not allowed_csv(uploaded_file.filename):
        return jsonify({"error": "Only CSV files are supported here."}), 400

    filename = secure_filename(uploaded_file.filename)
    save_path = INPUT_DIR / filename
    uploaded_file.save(save_path)

    try:
        import pandas as pd

        dataframe = pd.read_csv(save_path)
        preview = dataframe.head(5).to_dict(orient="records")

        return jsonify({
            "filename": filename,
            "rows": int(len(dataframe)),
            "columns": list(dataframe.columns),
            "preview": preview,
        })

    except Exception as exc:
        save_path.unlink(missing_ok=True)
        return jsonify({
            "error": f"Could not read the CSV file: {exc}"
        }), 400


# ---------------------------------------------------------------------------
# Main run route for tabular and time-series data
# ---------------------------------------------------------------------------

@app.route("/api/run", methods=["POST"])
def run_audit():
    if run_state["running"]:
        return jsonify({"error": "An audit is already running."}), 409

    config = request.get_json(force=True) or {}

    if not config.get("filename"):
        return jsonify({"error": "Upload a CSV dataset before starting the audit."}), 400

    run_state["running"] = True
    run_state["last_result"] = None

    threading.Thread(
        target=run_pipeline,
        args=(config,),
        daemon=True,
    ).start()

    return jsonify({"status": "started"})


def run_pipeline(config: dict) -> None:
    """Choose the correct local audit pipeline for the requested modality."""
    modality = str(config.get("modality", "tabular")).lower().replace("-", "_")

    try:
        if modality in {"time_series", "timeseries"}:
            run_time_series_pipeline(config)
        else:
            run_tabular_pipeline(config)

    except Exception:
        emit("error", f"Unexpected error:\n{traceback.format_exc()}")

    finally:
        run_state["running"] = False


# ---------------------------------------------------------------------------
# Time-series pipeline
# ---------------------------------------------------------------------------

def run_time_series_pipeline(config: dict) -> None:
    """Run the registered time-series models and evaluations."""
    emit("info", "Starting local time-series audit ...")

    file_path = INPUT_DIR / config["filename"]

    if not file_path.exists():
        emit("error", f"Dataset not found: {config['filename']}")
        return

    runner = TimeSeriesAuditRunner(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        emit=emit,
    )

    time_series_config = {
        **config,
        "file_path": str(file_path),
    }

    summary = runner.run(time_series_config)
    run_state["last_result"] = summary

    summary_path = OUTPUT_DIR / "time_series_audit_summary.json"
    summary_path.write_text(
        safe_dumps(summary) + "\n",
        encoding="utf-8",
    )

    emit(
        "info",
        f"Time-series audit complete in "
        f"{summary.get('total_runtime_seconds', 0):.2f}s",
    )
    emit("done", safe_dumps(summary))


# ---------------------------------------------------------------------------
# Tabular audit pipeline
# ---------------------------------------------------------------------------

def run_tabular_pipeline(config: dict) -> None:
    """Run the original MIDST synthetic tabular-data audit pipeline."""
    import pandas as pd

    start_time = time.perf_counter()
    records = []
    synthetic_datasets = {}

    filename = config["filename"]
    source_path = INPUT_DIR / filename

    if not source_path.exists():
        emit("error", f"Dataset not found: {filename}")
        return

    emit("info", f"Loading and cleaning: {filename} ...")

    loader = DataLoader(
        str(source_path),
        config=LoaderConfig(),
    )

    loaded_data = loader.load_and_clean()
    real_dataframe = loaded_data.df

    emit(
        "info",
        f"Shape: {loaded_data.original_shape} -> {loaded_data.cleaned_shape} | "
        f"Dropped: {loaded_data.dropped_columns or 'none'}",
    )

    max_rows = int(config.get("max_rows", 2000))
    random_seed = int(config.get("random_seed", 42))

    if len(real_dataframe) > max_rows:
        emit(
            "warn",
            f"Subsampling {len(real_dataframe)} rows -> {max_rows}",
        )

        real_dataframe = real_dataframe.sample(
            n=max_rows,
            random_state=random_seed,
        ).reset_index(drop=True)

    real_dataframe = real_dataframe.sample(
        frac=1,
        random_state=random_seed,
    ).reset_index(drop=True)

    train_ratio = float(config.get("train_ratio", 0.5))
    split_index = int(len(real_dataframe) * train_ratio)

    train_dataframe = real_dataframe.iloc[:split_index].reset_index(drop=True)
    control_dataframe = real_dataframe.iloc[split_index:].reset_index(drop=True)

    emit(
        "info",
        f"Split: {len(train_dataframe)} training | "
        f"{len(control_dataframe)} control rows",
    )

    metrics_engine = StatisticalMetrics()

    selected_models = config.get("models", {})
    if not selected_models:
        emit("error", "No tabular models were selected.")
        return

    privacy_thresholds = config.get(
        "privacy_thresholds",
        {
            "singling_out": 0.10,
            "linkability": 0.10,
            "cmla": 0.10,
        },
    )

    utility_thresholds = config.get(
        "utility_thresholds",
        {
            "logic_consistency": 0.70,
            "correlation_similarity": 0.70,
        },
    )

    score_weights = config.get(
        "score_weights",
        {
            "privacy": 0.50,
            "utility": 0.50,
        },
    )

    target_column = config.get("target_col") or None
    number_of_attacks = int(config.get("n_attacks", 500))

    for model_name, model_config in selected_models.items():
        emit("info", f"--- {model_name.upper()} ---")

        try:
            generator = build_generator(
                model_name,
                loaded_data.metadata,
                **model_config,
            )

        except ImportError as exc:
            emit("warn", f"Skipping {model_name}: {exc}")
            records.append(failed_record(model_name, f"Not installed: {exc}"))
            continue

        except ValueError as exc:
            emit("error", f"Unknown model '{model_name}': {exc}")
            records.append(failed_record(model_name, str(exc)))
            continue

        try:
            emit("info", f"Training {model_name} ...")
            generator.fit(train_dataframe)
            emit("info", "Training complete.")

        except Exception as exc:
            emit("error", f"Training failed for {model_name}: {exc}")
            records.append(failed_record(model_name, f"Training failed: {exc}"))
            continue

        try:
            emit(
                "info",
                f"Generating {len(train_dataframe)} synthetic rows ...",
            )

            generation_result = generator.sample(len(train_dataframe))
            synthetic_dataframe = generation_result.synthetic_df
            synthetic_datasets[model_name] = synthetic_dataframe

        except Exception as exc:
            emit("error", f"Sampling failed for {model_name}: {exc}")
            records.append(failed_record(model_name, f"Sampling failed: {exc}"))
            continue

        try:
            emit("info", "Running privacy attacks ...")

            attacker = PrivacyAttacks(
                train_dataframe,
                synthetic_dataframe,
                control_dataframe,
                n_attacks=number_of_attacks,
            )

            singling_out = attacker.singling_out()
            linkability = attacker.linkability()
            cmla = attacker.cmla_leakage()

            emit(
                "info",
                f"  Singling-out: {singling_out.risk_score:.1%} "
                f"[{singling_out.ci_lower:.1%}-{singling_out.ci_upper:.1%}]",
            )

            emit(
                "info",
                f"  Linkability: {linkability.risk_score:.1%} "
                f"[{linkability.ci_lower:.1%}-{linkability.ci_upper:.1%}]",
            )

            emit("info", f"  CMLA: {cmla.risk_score:.1%}")

        except Exception as exc:
            emit("error", f"Privacy evaluation failed for {model_name}: {exc}")
            records.append(failed_record(model_name, f"Privacy evaluation failed: {exc}"))
            continue

        try:
            emit("info", "Computing utility metrics ...")

            utility = metrics_engine.evaluate(
                train_dataframe,
                synthetic_dataframe,
                target_col=target_column,
            )

            emit(
                "info",
                f"  Correlation similarity: {utility.correlation_similarity:.1%} | "
                f"Logic consistency: {utility.logic_consistency:.1%}",
            )

        except Exception as exc:
            emit("error", f"Utility evaluation failed for {model_name}: {exc}")
            records.append(failed_record(model_name, f"Utility evaluation failed: {exc}"))
            continue

        privacy_pass = (
            singling_out.risk_score <= privacy_thresholds["singling_out"]
            and linkability.risk_score <= privacy_thresholds["linkability"]
            and cmla.risk_score <= privacy_thresholds["cmla"]
        )

        utility_pass = (
            utility.logic_consistency >= utility_thresholds["logic_consistency"]
            and utility.correlation_similarity
            >= utility_thresholds["correlation_similarity"]
        )

        overall_pass = privacy_pass and utility_pass

        privacy_score = 1.0 - max(
            singling_out.risk_score,
            linkability.risk_score,
            cmla.risk_score,
        )

        composite_score = (
            score_weights["privacy"] * privacy_score
            + score_weights["utility"] * utility.composite_utility
        )

        emit(
            "info" if overall_pass else "warn",
            f"{model_name.upper()} -> "
            f"{'PASSED' if overall_pass else 'FAILED'} | "
            f"score={composite_score:.3f}",
        )

        records.append({
            "model": model_name,
            "singling_out_risk": round(singling_out.risk_score, 4),
            "singling_out_ci": (
                f"[{singling_out.ci_lower:.3f}, {singling_out.ci_upper:.3f}]"
            ),
            "linkability_risk": round(linkability.risk_score, 4),
            "linkability_ci": (
                f"[{linkability.ci_lower:.3f}, {linkability.ci_upper:.3f}]"
            ),
            "cmla_risk": round(cmla.risk_score, 4),
            "correlation_similarity": round(utility.correlation_similarity, 4),
            "logic_consistency": round(utility.logic_consistency, 4),
            "tstr_score": round(utility.tstr_score, 4),
            "tstr_baseline": round(utility.tstr_baseline, 4),
            "tstr_gap": round(utility.tstr_gap, 4),
            "composite_utility": round(utility.composite_utility, 4),
            "privacy_score": round(privacy_score, 4),
            "composite_score": round(composite_score, 4),
            "privacy_pass": privacy_pass,
            "utility_pass": utility_pass,
            "overall_pass": overall_pass,
            "model_config": json.dumps(model_config),
            "violation_breakdown": json.dumps(
                utility.column_violation_breakdown
            ),
            "cmla_notes": cmla.notes,
        })

    if not records:
        emit("error", "No tabular model completed successfully.")
        return

    report_dataframe = pd.DataFrame(records).sort_values(
        "composite_score",
        ascending=False,
    )

    successful_models = report_dataframe

    passing_models = report_dataframe[report_dataframe["overall_pass"]]

    if not passing_models.empty:
        best_model = passing_models.iloc[0]["model"]
        threshold_passed = True
        emit("info", f"Recommended model: {best_model.upper()}")

    else:
        best_model = successful_models.iloc[0]["model"]
        threshold_passed = False
        emit(
            "warn",
            f"No model passed all thresholds. "
            f"Best available: {best_model.upper()}",
        )

    report_dataframe["is_recommended"] = (
        report_dataframe["model"] == best_model
    )

    report_path = OUTPUT_DIR / "final_audit_report.csv"
    report_dataframe.to_csv(report_path, index=False)
    emit("info", "Saved: final_audit_report.csv")

    if best_model in synthetic_datasets:
        synthetic_path = OUTPUT_DIR / f"{best_model}_best_synthetic.csv"

        synthetic_datasets[best_model].to_csv(
            synthetic_path,
            index=False,
        )

        emit("info", f"Saved: {synthetic_path.name}")

        try:
            Visualizer().plot_winner_comparison(
                train_dataframe,
                synthetic_datasets[best_model],
                best_model,
            )
            emit("info", "Saved: winner_comparison.png")

        except Exception:
            emit("warn", "Comparison plot was skipped.")

    elapsed_seconds = round(time.perf_counter() - start_time, 2)

    summary = {
        "modality": "tabular",
        "recommended_model": best_model,
        "threshold_passed": threshold_passed,
        "total_runtime_seconds": elapsed_seconds,
        "results": report_dataframe.to_dict(orient="records"),
        "run_config": {
            key: value
            for key, value in config.items()
            if key != "models"
        },
    }

    summary_path = OUTPUT_DIR / "audit_summary.json"
    summary_path.write_text(
        safe_dumps(summary) + "\n",
        encoding="utf-8",
    )

    run_state["last_result"] = summary

    emit("info", f"Tabular audit complete in {elapsed_seconds:.2f}s")
    emit("done", safe_dumps(summary))


def failed_record(model_name: str, reason: str) -> dict:
    """Keep failed or skipped models visible without treating them as evaluated."""
    return {
        "model": model_name,
        "singling_out_risk": None,
        "singling_out_ci": "N/A",
        "linkability_risk": None,
        "linkability_ci": "N/A",
        "cmla_risk": None,
        "correlation_similarity": None,
        "logic_consistency": None,
        "tstr_score": -1.0,
        "tstr_baseline": -1.0,
        "tstr_gap": -1.0,
        "composite_utility": 0.0,
        "privacy_score": 0.0,
        "composite_score": 0.0,
        "privacy_pass": False,
        "utility_pass": False,
        "overall_pass": False,
        "model_config": "{}",
        "violation_breakdown": "{}",
        "cmla_notes": "",
        "notes": reason,
        "skipped": True,
    }


# ---------------------------------------------------------------------------
# Results, downloads, and model availability
# ---------------------------------------------------------------------------

@app.route("/api/results")
def get_results():
    """Return the most recent in-memory result if available."""
    if run_state["last_result"] is not None:
        return app.response_class(
            safe_dumps(run_state["last_result"]),
            mimetype="application/json",
        )

    possible_summary_files = [
        OUTPUT_DIR / "audit_summary.json",
        OUTPUT_DIR / "time_series_audit_summary.json",
        OUTPUT_DIR / "image_audit_summary.json",
    ]

    for summary_file in possible_summary_files:
        if summary_file.exists():
            return app.response_class(
                summary_file.read_text(encoding="utf-8"),
                mimetype="application/json",
            )

    return jsonify({"error": "No audit results are available yet."}), 404


@app.route("/api/download/<filename>")
def download_file(filename: str):
    safe_filename = secure_filename(filename)
    file_path = OUTPUT_DIR / safe_filename

    if not file_path.exists():
        return jsonify({"error": f"File not found: {safe_filename}"}), 404

    return send_from_directory(
        str(OUTPUT_DIR.resolve()),
        safe_filename,
        as_attachment=True,
    )


@app.route("/api/models")
def list_models():
    """Report the tabular generators currently supported by MIDST."""
    status = {}

    for model_name in GENERATOR_REGISTRY:
        status[model_name] = "available"

    return jsonify(status)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "=" * 56)
    print("MIDST - open http://localhost:5000")
    print("=" * 56 + "\n")

    app.run(
        debug=False,
        threaded=True,
        port=5000,
    )
