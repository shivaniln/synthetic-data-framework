from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


SOURCE_DIR = Path("data/input/digits_original")
OUTPUT_DIR = Path("data/output/diverse_pca_digits")

IMAGE_SIZE = (64, 64)
NUMBER_OF_IMAGES = 200
PCA_COMPONENTS = 12
LATENT_SCALE = 2.5
PIXEL_NOISE = 0.025
RANDOM_SEED = 123


def load_images(folder):
    image_paths = sorted(folder.glob("*.png"))

    if not image_paths:
        raise ValueError(f"No PNG images found in {folder}")

    rows = []

    for path in image_paths:
        with Image.open(path) as image:
            image = image.convert("L").resize(
                IMAGE_SIZE,
                Image.Resampling.LANCZOS,
            )

            pixels = np.asarray(image, dtype=np.float32) / 255.0
            rows.append(pixels.flatten())

    return np.stack(rows)


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {SOURCE_DIR}")

    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original_matrix = load_images(SOURCE_DIR)

    components = min(
        PCA_COMPONENTS,
        original_matrix.shape[0] - 1,
        original_matrix.shape[1],
    )

    print(f"Training diverse PCA generator with {components} components...")

    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        random_state=RANDOM_SEED,
    )

    latent = pca.fit_transform(original_matrix)

    rng = np.random.default_rng(RANDOM_SEED)

    latent_mean = latent.mean(axis=0)
    latent_std = np.maximum(latent.std(axis=0), 0.01)

    sampled_latent = rng.normal(
        loc=latent_mean,
        scale=latent_std * LATENT_SCALE,
        size=(NUMBER_OF_IMAGES, components),
    )

    generated = pca.inverse_transform(sampled_latent)

    noise = rng.normal(
        0,
        PIXEL_NOISE,
        size=generated.shape,
    )

    generated = np.clip(generated + noise, 0, 1)

    for index, row in enumerate(generated):
        pixels = (row.reshape(IMAGE_SIZE[1], IMAGE_SIZE[0]) * 255).astype(np.uint8)
        image = Image.fromarray(pixels, mode="L")
        image.save(OUTPUT_DIR / f"diverse_pca_{index:03d}.png")

    print("\nDIVERSE PCA IMAGE GENERATION COMPLETE")
    print(f"Images created: {NUMBER_OF_IMAGES}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.1%}")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()