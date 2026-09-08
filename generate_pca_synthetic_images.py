"""
Generate a new synthetic image set using PCA latent-space sampling.

This is a lightweight statistical image generator. It learns broad visual
patterns from an image folder, samples new latent vectors, and reconstructs
new images. It is not a GAN or diffusion model, but it creates new images
rather than lightly modifying original files.
"""

from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


SOURCE_DIR = Path("data/input/digits_original")
OUTPUT_DIR = Path("data/output/pca_synthetic_digits")

IMAGE_SIZE = (64, 64)
NUMBER_OF_IMAGES = 200
PCA_COMPONENTS = 20
RANDOM_SEED = 42

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_images(folder: Path) -> np.ndarray:
    image_paths = sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
    )

    if not image_paths:
        raise ValueError(f"No supported images found in: {folder}")

    rows = []

    for image_path in image_paths:
        with Image.open(image_path) as image:
            prepared = image.convert("L").resize(
                IMAGE_SIZE,
                Image.Resampling.LANCZOS,
            )

            pixels = np.asarray(prepared, dtype=np.float32) / 255.0
            rows.append(pixels.flatten())

    return np.stack(rows)


def save_images(matrix: np.ndarray, folder: Path) -> None:
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(matrix):
        pixels = np.clip(row, 0.0, 1.0)
        pixels = (pixels.reshape(IMAGE_SIZE[1], IMAGE_SIZE[0]) * 255).astype(np.uint8)

        image = Image.fromarray(pixels, mode="L")
        image.save(folder / f"pca_synthetic_{index:03d}.png")


def main() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Original image folder not found: {SOURCE_DIR}\n"
            "Run prepare_digits_image_audit.py first."
        )

    print(f"Loading original images from: {SOURCE_DIR}")
    original_matrix = load_images(SOURCE_DIR)

    component_count = min(
        PCA_COMPONENTS,
        original_matrix.shape[0] - 1,
        original_matrix.shape[1],
    )

    print(
        f"Training PCA image generator on {original_matrix.shape[0]} images "
        f"with {component_count} components ..."
    )

    pca = PCA(
        n_components=component_count,
        svd_solver="randomized",
        random_state=RANDOM_SEED,
    )

    latent_vectors = pca.fit_transform(original_matrix)

    generator = np.random.default_rng(RANDOM_SEED)

    latent_mean = latent_vectors.mean(axis=0)
    latent_std = latent_vectors.std(axis=0)

    # Prevent components with almost no variation from becoming unstable.
    latent_std = np.maximum(latent_std, 0.01)

    sampled_latent_vectors = generator.normal(
        loc=latent_mean,
        scale=latent_std,
        size=(NUMBER_OF_IMAGES, component_count),
    )

    synthetic_matrix = pca.inverse_transform(sampled_latent_vectors)

    save_images(synthetic_matrix, OUTPUT_DIR)

    explained_variance = pca.explained_variance_ratio_.sum()

    print("\nPCA SYNTHETIC IMAGE GENERATION COMPLETE")
    print(f"Original images used: {original_matrix.shape[0]}")
    print(f"Synthetic images created: {NUMBER_OF_IMAGES}")
    print(f"Explained variance: {explained_variance:.1%}")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()