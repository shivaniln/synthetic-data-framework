"""
Image upload and audit API routes for MIDST.

Users upload two ZIP files:
1. Original image dataset
2. Synthetic image dataset

Both files are extracted locally and passed to ImageAuditRunner.
"""

from __future__ import annotations

import shutil
import threading
import uuid
import zipfile
import base64
import mimetypes
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from flask import jsonify, request

from src.core.runner import ImageAuditRunner
from src.modalities.image_generator import generate_pca_images


ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _zip_image_folder(folder: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for image_path in folder.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES:
                archive.write(image_path, image_path.relative_to(folder))
    return destination.name


def _safe_extract_images(zip_file: Any, destination: Path) -> int:
    """Extract only supported image files and block ZIP path traversal."""
    destination.mkdir(parents=True, exist_ok=True)
    extracted_count = 0

    try:
        with zipfile.ZipFile(zip_file.stream) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue

                member_path = PurePosixPath(member.filename)

                # Reject absolute paths and attempts such as ../../file.png
                if member_path.is_absolute() or ".." in member_path.parts:
                    continue

                if member_path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
                    continue

                target = (destination / Path(*member_path.parts)).resolve()
                destination_root = destination.resolve()

                if destination_root not in target.parents and target != destination_root:
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)

                with archive.open(member) as source, open(target, "wb") as output:
                    shutil.copyfileobj(source, output)

                extracted_count += 1

    except zipfile.BadZipFile as exc:
        raise ValueError("The uploaded file is not a valid ZIP archive.") from exc

    if extracted_count == 0:
        raise ValueError(
            "No supported images were found. ZIP files may contain "
            "JPG, JPEG, PNG, BMP, or WEBP images."
        )

    return extracted_count


def register_image_routes(
    app: Any,
    input_dir: Path,
    output_dir: Path,
    run_state: dict,
    emit: Callable[[str, str], None],
    safe_dumps: Callable[[dict], str],
) -> None:
    """Attach image-specific upload and audit endpoints to the Flask app."""

    image_upload_dir = input_dir / "image_audits"
    image_upload_dir.mkdir(parents=True, exist_ok=True)

    @app.route("/api/upload-images", methods=["POST"])
    def upload_images():
        original_zip = request.files.get("original_zip")
        synthetic_zip = request.files.get("synthetic_zip")

        if not original_zip or not synthetic_zip:
            return jsonify({
                "error": "Upload both the original-images ZIP and synthetic-images ZIP."
            }), 400

        if not original_zip.filename.lower().endswith(".zip"):
            return jsonify({"error": "Original dataset must be a ZIP file."}), 400

        if not synthetic_zip.filename.lower().endswith(".zip"):
            return jsonify({"error": "Synthetic dataset must be a ZIP file."}), 400

        audit_id = uuid.uuid4().hex[:12]
        audit_dir = image_upload_dir / audit_id
        original_dir = audit_dir / "original"
        synthetic_dir = audit_dir / "synthetic"

        try:
            original_count = _safe_extract_images(original_zip, original_dir)
            synthetic_count = _safe_extract_images(synthetic_zip, synthetic_dir)
        except ValueError as exc:
            shutil.rmtree(audit_dir, ignore_errors=True)
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            shutil.rmtree(audit_dir, ignore_errors=True)
            return jsonify({"error": f"Could not process image ZIP files: {exc}"}), 500

        return jsonify({
            "audit_id": audit_id,
            "original_folder": str(original_dir),
            "synthetic_folder": str(synthetic_dir),
            "original_images": original_count,
            "synthetic_images": synthetic_count,
        })

    @app.route("/api/generate-image-audit", methods=["POST"])
    def generate_image_audit():
        """Extract one original ZIP, generate synthetic images, and prepare an audit."""
        original_zip = request.files.get("original_zip")

        if not original_zip or not original_zip.filename.lower().endswith(".zip"):
            return jsonify({"error": "Upload an original-images ZIP file first."}), 400

        audit_id = uuid.uuid4().hex[:12]
        audit_dir = image_upload_dir / audit_id
        original_dir = audit_dir / "original"
        synthetic_dir = audit_dir / "synthetic_generated"

        try:
            original_count = _safe_extract_images(original_zip, original_dir)
            generation = generate_pca_images(
                original_dir,
                synthetic_dir,
                number_of_images=min(max(original_count, 50), 200),
            )
            generated_zip = _zip_image_folder(
                synthetic_dir,
                output_dir / f"pca_synthetic_images_{audit_id}.zip",
            )
        except Exception as exc:
            shutil.rmtree(audit_dir, ignore_errors=True)
            return jsonify({"error": f"Could not generate image dataset: {exc}"}), 400

        return jsonify({
            "audit_id": audit_id,
            "original_folder": str(original_dir),
            "synthetic_folder": str(synthetic_dir),
            "original_images": original_count,
            "synthetic_images": generation["synthetic_images"],
            "explained_variance": generation["explained_variance"],
            "generated_zip": generated_zip,
            "generated_by": "local_pca",
        })

    @app.route("/api/image-previews", methods=["POST"])
    def image_previews():
        body = request.get_json(force=True) or {}
        preview_items = []

        for kind in ("original", "synthetic"):
            folder_value = body.get(f"{kind}_folder")
            if not folder_value:
                continue

            folder = Path(folder_value).resolve()
            if not folder.exists() or not folder.is_dir():
                continue

            paths = [
                path for path in sorted(folder.rglob("*"))
                if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES
            ][:4]

            for path in paths:
                mime = mimetypes.guess_type(path.name)[0] or "image/png"
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                preview_items.append({
                    "kind": kind,
                    "name": path.name,
                    "data_url": f"data:{mime};base64,{encoded}",
                })

        return jsonify({"previews": preview_items})

    @app.route("/api/run-image-audit", methods=["POST"])
    def run_image_audit():
        if run_state["running"]:
            return jsonify({"error": "An audit is already running."}), 409

        body = request.get_json(force=True) or {}

        if not body.get("original_folder") or not body.get("synthetic_folder"):
            return jsonify({
                "error": "Upload the original and synthetic image ZIP files first."
            }), 400

        run_state["running"] = True
        run_state["last_result"] = None

        def image_pipeline() -> None:
            try:
                emit("info", "Starting local image privacy and utility audit ...")

                runner = ImageAuditRunner(
                    output_dir=output_dir,
                    emit=emit,
                )
                summary = runner.run(body)

                run_state["last_result"] = summary

                summary_path = output_dir / "image_audit_summary.json"
                summary_path.write_text(
                    safe_dumps(summary) + "\n",
                    encoding="utf-8",
                )

                emit("done", safe_dumps(summary))

            except Exception as exc:
                emit("error", f"Image audit failed: {exc}")
            finally:
                run_state["running"] = False

        threading.Thread(target=image_pipeline, daemon=True).start()
        return jsonify({"status": "started"})
