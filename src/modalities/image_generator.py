"""Lightweight local image generator used by the MIDST image workflow."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = (64, 64)


def generate_pca_images(
    source_dir: str | Path,
    output_dir: str | Path,
    number_of_images: int = 200,
    components: int = 12,
    latent_scale: float = 2.5,
    pixel_noise: float = 0.025,
    seed: int = 123,
) -> dict:
    """Learn a PCA image representation and sample new local images."""
    source = Path(source_dir)
    output = Path(output_dir)
    paths = sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    if not paths:
        raise ValueError("No supported images were found in the original dataset.")

    rows = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("L").resize(
                IMAGE_SIZE,
                Image.Resampling.LANCZOS,
            )
            rows.append((np.asarray(image, dtype=np.float32) / 255.0).flatten())

    matrix = np.stack(rows)
    component_count = min(components, matrix.shape[0] - 1, matrix.shape[1])

    if component_count < 2:
        raise ValueError("At least two original images are required for generation.")

    pca = PCA(
        n_components=component_count,
        svd_solver="randomized",
        random_state=seed,
    )
    latent = pca.fit_transform(matrix)

    rng = np.random.default_rng(seed)
    latent_std = np.maximum(latent.std(axis=0), 0.01)
    sampled = rng.normal(
        loc=latent.mean(axis=0),
        scale=latent_std * latent_scale,
        size=(number_of_images, component_count),
    )

    generated = np.clip(
        pca.inverse_transform(sampled)
        + rng.normal(0, pixel_noise, size=(number_of_images, matrix.shape[1])),
        0,
        1,
    )

    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(generated):
        pixels = (row.reshape(IMAGE_SIZE[1], IMAGE_SIZE[0]) * 255).astype(np.uint8)
        Image.fromarray(pixels, mode="L").save(output / f"pca_synthetic_{index:03d}.png")

    return {
        "original_images": len(paths),
        "synthetic_images": number_of_images,
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "output_folder": str(output),
    }
