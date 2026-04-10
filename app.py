"""
app.py — MIDST Framework Flask Backend
--------------------------------------
Exposes the pipeline as a REST API so the frontend can:
  1. Upload a CSV file
  2. Trigger an audit run with chosen config
  3. Stream live progress logs back to the browser via SSE
  4. Serve the final JSON results when done
  5. Download output files (synthetic CSV, report)

Run with:
    python app.py
Then open http://localhost:5000 in your browser.
"""

import json
import logging
import math
import os
import queue
import threading
import time
import traceback
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename


# ─── JSON safety ──────────────────────────────────────────────────────────────
# Python's json module writes float('nan') as the bare token NaN which is not
# valid JSON. This encoder replaces NaN/Inf with null at the serialisation layer,
# so it is impossible for bad tokens to appear regardless of input source.

class _SafeEncoder(json.JSONEncoder):
    def iterencode(self, o, _one_shot=False):
        # Pre-process the entire object tree before encoding
        return super().iterencode(self._clean(o), _one_shot)

    def _clean(self, obj):
        if isinstance(obj, dict):
            return {k: self._clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._clean(v) for v in obj]
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        # numpy scalars
        if hasattr(obj, "item"):
            return self._clean(obj.item())
        return obj


def safe_dumps(obj) -> str:
    """json.dumps that converts NaN/Inf/numpy scalars to JSON-safe values."""
    return json.dumps(obj, cls=_SafeEncoder)

from src.data_loader import DataLoader, LoaderConfig
from src.models.base_generator import build_generator, GENERATOR_REGISTRY
from src.evaluation.attacks import PrivacyAttacks
from src.evaluation.metrics import StatisticalMetrics
from src.utils.visualizer import Visualizer

app = Flask(__name__, static_folder="frontend", static_url_path="")

INPUT_DIR  = Path("data/input")
OUTPUT_DIR = Path("data/output")
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Per-connection queues — each SSE client gets its own queue so reconnects
# don't steal messages from the active connection.
_subscribers: list = []
_subscribers_lock = threading.Lock()
run_state = {"running": False, "last_result": None}


def _new_subscriber() -> queue.Queue:
    q: queue.Queue = queue.Queue()
    with _subscribers_lock:
        _subscribers.append(q)
    return q


def _remove_subscriber(q: queue.Queue) -> None:
    with _subscribers_lock:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "csv"


def emit(level: str, message: str) -> None:
    item = {"level": level, "message": message, "ts": time.strftime("%H:%M:%S")}
    with _subscribers_lock:
        for q in list(_subscribers):
            q.put(item)
    logging.getLogger("midst.app").log(
        {"info": logging.INFO, "warn": logging.WARNING, "error": logging.ERROR}.get(level, logging.INFO),
        message,
    )


# ─── Static ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


# ─── Upload ───────────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file in request"}), 400
    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Only .csv files are accepted"}), 400

    filename  = secure_filename(file.filename)
    save_path = INPUT_DIR / filename
    file.save(save_path)

    try:
        import pandas as pd
        full = pd.read_csv(save_path)
        preview = pd.read_csv(save_path, nrows=5)
        return jsonify({
            "filename": filename,
            "rows":     len(full),
            "columns":  list(full.columns),
            "preview":  preview.to_dict(orient="records"),
        })
    except Exception as exc:
        return jsonify({"error": f"Could not read CSV: {exc}"}), 400


# ─── Run ──────────────────────────────────────────────────────────────────────

@app.route("/api/run", methods=["POST"])
def run_audit():
    if run_state["running"]:
        return jsonify({"error": "An audit is already running."}), 409

    body = request.get_json(force=True)
    if not body or "filename" not in body:
        return jsonify({"error": "Missing 'filename' in body"}), 400

    # New per-connection queues are created fresh per SSE connection — no clearing needed.

    run_state["running"] = True
    run_state["last_result"] = None
    threading.Thread(target=_run_pipeline, args=(body,), daemon=True).start()
    return jsonify({"status": "started"})


def _run_pipeline(config: dict) -> None:
    import pandas as pd

    t_start = time.perf_counter()
    records = []
    synthetic_datasets = {}

    try:
        # 1. Load & clean
        filename   = config["filename"]
        emit("info", f"Loading and cleaning: {filename} …")
        loader = DataLoader(str(INPUT_DIR / filename), config=LoaderConfig())
        result = loader.load_and_clean()
        real_df = result.df
        emit("info",
             f"Shape: {result.original_shape} → {result.cleaned_shape} | "
             f"Dropped: {result.dropped_columns or 'none'}")

        # 2. Subsample
        max_rows = int(config.get("max_rows", 2000))
        seed     = int(config.get("random_seed", 42))
        if len(real_df) > max_rows:
            emit("warn", f"Subsampling {len(real_df)} rows → {max_rows}")
            real_df = real_df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

        # 3. Shuffle + split
        real_df     = real_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        train_ratio = float(config.get("train_ratio", 0.5))
        split       = int(len(real_df) * train_ratio)
        train_df    = real_df.iloc[:split].reset_index(drop=True)
        control_df  = real_df.iloc[split:].reset_index(drop=True)
        emit("info", f"Split: {len(train_df)} training · {len(control_df)} control rows")

        # 4. Per-model loop
        metrics_engine = StatisticalMetrics()
        models_cfg  = config.get("models", {})
        pt          = config.get("privacy_thresholds", {"singling_out": 0.1, "linkability": 0.1, "cmla": 0.1})
        ut          = config.get("utility_thresholds", {"logic_consistency": 0.7, "correlation_similarity": 0.7, "tstr_gap_max": 0.1})
        sw          = config.get("score_weights",       {"privacy": 0.5, "utility": 0.5})
        target_col  = config.get("target_col") or None
        n_attacks   = int(config.get("n_attacks", 500))

        for model_name, model_kwargs in models_cfg.items():
            emit("info", f"━━━ {model_name.upper()} ━━━")

            try:
                gen = build_generator(model_name, result.metadata, **model_kwargs)
            except ImportError as exc:
                emit("warn", f"Skipping {model_name} — not installed: {exc}")
                records.append(_failed_record(model_name, f"Not installed: {exc}"))
                continue
            except ValueError as exc:
                emit("error", f"Unknown model '{model_name}': {exc}")
                records.append(_failed_record(model_name, str(exc)))
                continue

            emit("info", f"Training {model_name} …")
            try:
                gen.fit(train_df)
                emit("info", "Training done.")
            except Exception as exc:
                emit("error", f"Training failed: {exc}")
                records.append(_failed_record(model_name, f"Training failed: {exc}"))
                continue

            emit("info", f"Generating {len(train_df)} synthetic rows …")
            try:
                gen_result = gen.sample(len(train_df))
                syn_df     = gen_result.synthetic_df
                synthetic_datasets[model_name] = syn_df
            except Exception as exc:
                emit("error", f"Sampling failed: {exc}")
                records.append(_failed_record(model_name, f"Sampling failed: {exc}"))
                continue

            emit("info", "Running privacy attacks …")
            attacker    = PrivacyAttacks(train_df, syn_df, control_df, n_attacks=n_attacks)
            so_result   = attacker.singling_out()
            emit("info", f"  Singling-out: {so_result.risk_score:.1%}  [{so_result.ci_lower:.1%}–{so_result.ci_upper:.1%}]")
            link_result = attacker.linkability()
            emit("info", f"  Linkability:  {link_result.risk_score:.1%}  [{link_result.ci_lower:.1%}–{link_result.ci_upper:.1%}]")
            cmla_result = attacker.cmla_leakage()
            emit("info", f"  CMLA:         {cmla_result.risk_score:.1%}")

            emit("info", "Computing utility metrics …")
            utility = metrics_engine.evaluate(train_df, syn_df, target_col=target_col)
            emit("info",
                 f"  Corr sim: {utility.correlation_similarity:.1%} | "
                 f"Logic: {utility.logic_consistency:.1%}" +
                 (f" | TSTR gap: {utility.tstr_gap:.1%}" if utility.tstr_gap >= 0 else ""))

            privacy_pass = (
                so_result.risk_score   <= pt["singling_out"]
                and link_result.risk_score <= pt["linkability"]
                and cmla_result.risk_score <= pt["cmla"]
            )
            utility_pass = (
                utility.logic_consistency      >= ut["logic_consistency"]
                and utility.correlation_similarity >= ut["correlation_similarity"]
            )
            if target_col and utility.tstr_gap >= 0:
                utility_pass = utility_pass and (utility.tstr_gap <= ut["tstr_gap_max"])

            overall_pass  = privacy_pass and utility_pass
            privacy_score = 1.0 - max(so_result.risk_score, link_result.risk_score, cmla_result.risk_score)
            composite     = sw["privacy"] * privacy_score + sw["utility"] * utility.composite_utility

            emit("info" if overall_pass else "warn",
                 f"{model_name.upper()} → {'PASSED ✓' if overall_pass else 'FAILED ✗'} | score={composite:.3f}")

            records.append({
                "model":                  model_name,
                "singling_out_risk":      round(so_result.risk_score, 4),
                "singling_out_ci":        f"[{so_result.ci_lower:.3f}, {so_result.ci_upper:.3f}]",
                "linkability_risk":       round(link_result.risk_score, 4),
                "linkability_ci":         f"[{link_result.ci_lower:.3f}, {link_result.ci_upper:.3f}]",
                "cmla_risk":              round(cmla_result.risk_score, 4),
                "correlation_similarity": round(utility.correlation_similarity, 4),
                "logic_consistency":      round(utility.logic_consistency, 4),
                "tstr_score":             round(utility.tstr_score, 4),
                "tstr_baseline":          round(utility.tstr_baseline, 4),
                "tstr_gap":               round(utility.tstr_gap, 4),
                "composite_utility":      round(utility.composite_utility, 4),
                "privacy_score":          round(privacy_score, 4),
                "composite_score":        round(composite, 4),
                "privacy_pass":           privacy_pass,
                "utility_pass":           utility_pass,
                "overall_pass":           overall_pass,
                "model_config":           json.dumps(model_kwargs),
                "violation_breakdown":    json.dumps(utility.column_violation_breakdown),
                "cmla_notes":             cmla_result.notes,
            })

        # 5. Rank
        if not records:
            emit("error", "No models completed. Check your config.")
            return

        report_df = pd.DataFrame(records).sort_values("composite_score", ascending=False)
        passing   = report_df[report_df["overall_pass"]]

        if not passing.empty:
            best_name = passing.iloc[0]["model"]
            emit("info", f"✦ RECOMMENDED: {best_name.upper()}")
        else:
            best_name = report_df.iloc[0]["model"]
            emit("warn", f"No model passed thresholds. Best available: {best_name.upper()}. Review before deploying.")

        # 6. Export
        report_df["is_recommended"] = report_df["model"] == best_name
        report_df.to_csv(OUTPUT_DIR / "final_audit_report.csv", index=False)
        emit("info", "Saved: final_audit_report.csv")

        if best_name in synthetic_datasets:
            syn_path = OUTPUT_DIR / f"{best_name}_best_synthetic.csv"
            synthetic_datasets[best_name].to_csv(syn_path, index=False)
            emit("info", f"Saved: {best_name}_best_synthetic.csv")
            try:
                Visualizer().plot_winner_comparison(train_df, synthetic_datasets[best_name], best_name)
                emit("info", "Saved: winner_comparison.png")
            except Exception:
                emit("warn", "Plot skipped (matplotlib issue).")

        elapsed = round(time.perf_counter() - t_start, 1)

        # Build the summary. safe_dumps() / _SafeEncoder handles NaN, Inf,
        # numpy bools, and numpy scalars at the serialisation layer — nothing
        # can produce invalid JSON tokens regardless of what pandas returns.
        summary = {
            "recommended_model":     best_name,
            "threshold_passed":      bool(not passing.empty),
            "total_runtime_seconds": elapsed,
            "results":               report_df.to_dict(orient="records"),
            "run_config":            {k: v for k, v in config.items() if k != "models"},
        }
        # Write to disk using the safe encoder
        with open(OUTPUT_DIR / "audit_summary.json", "w") as f:
            f.write(safe_dumps(summary))
            f.write("\n")

        run_state["last_result"] = summary
        emit("info", f"Audit complete in {elapsed}s")
        # safe_dumps() guarantees no NaN/Inf — JSON.parse() will always succeed.
        emit("done", safe_dumps(summary))

    except Exception:
        emit("error", f"Unexpected error:\n{traceback.format_exc()}")
    finally:
        run_state["running"] = False


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _failed_record(model_name: str, reason: str) -> dict:
    """Placeholder row inserted when a model errors out, so the run continues."""
    return {
        "model":                  model_name,
        "singling_out_risk":      1.0, "singling_out_ci": "N/A",
        "linkability_risk":       1.0, "linkability_ci":  "N/A",
        "cmla_risk":              1.0,
        "correlation_similarity": 0.0, "logic_consistency":  0.0,
        "tstr_score":  -1.0, "tstr_baseline": -1.0, "tstr_gap": -1.0,
        "composite_utility": 0.0, "privacy_score": 0.0, "composite_score": 0.0,
        "privacy_pass": False, "utility_pass": False, "overall_pass": False,
        "model_config": "{}", "violation_breakdown": "{}", "cmla_notes": "",
        "notes": reason,
    }


# ─── SSE log stream ───────────────────────────────────────────────────────────

@app.route("/api/logs")
def stream_logs():
    """
    Each browser connection gets its own queue.
    emit() broadcasts to every active queue simultaneously,
    so reconnects or multiple tabs never steal each other's messages.
    """
    my_queue = _new_subscriber()

    def generate():
        try:
            yield "retry: 1000\n\n"
            while True:
                try:
                    item = my_queue.get(timeout=30)
                    yield f"data: {json.dumps(item)}\n\n"
                    # Close the generator after sending "done" so the HTTP response ends cleanly.
                    if item.get("level") == "done":
                        return
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            _remove_subscriber(my_queue)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─── Results & downloads ──────────────────────────────────────────────────────

@app.route("/api/results")
def get_results():
    path = OUTPUT_DIR / "audit_summary.json"
    if not path.exists():
        return jsonify({"error": "No results yet."}), 404
    # Read the raw text and replace any bare NaN/Infinity tokens that slipped
    # through — these are invalid JSON but valid Python float repr.
    raw = path.read_text(encoding="utf-8")
    raw = raw.replace(": NaN",  ": null")              .replace(":NaN",   ":null")              .replace(": Infinity",  ": null")              .replace(": -Infinity", ": null")
    try:
        return app.response_class(raw, mimetype="application/json")
    except Exception:
        return jsonify({"error": "Could not parse audit_summary.json"}), 500


@app.route("/api/status")
def get_status():
    return jsonify({"running": run_state["running"]})


@app.route("/api/download/<filename>")
def download_file(filename: str):
    safe = secure_filename(filename)
    if not (OUTPUT_DIR / safe).exists():
        return jsonify({"error": f"Not found: {safe}"}), 404
    return send_from_directory(str(OUTPUT_DIR.resolve()), safe, as_attachment=True)


@app.route("/api/models")
def list_models():
    status = {}
    for name in GENERATOR_REGISTRY:
        try:
            if name in ("privbayes", "pategan"):
                import snsynth  # noqa: F401
            status[name] = "available"
        except ImportError:
            status[name] = "not_installed"
    return jsonify(status)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%H:%M:%S")
    print("\n" + "═" * 50)
    print("  MIDST  —  open http://localhost:5000")
    print("═" * 50 + "\n")
    app.run(debug=False, threaded=True, port=5000)