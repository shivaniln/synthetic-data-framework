"""
Prepare a real image benchmark for the MIDST image-audit workflow.

The original folder contains genuine handwritten-digit images from scikit-learn.
The comparison folder contains lightly transformed versions. It is intentionally
expected to have a high near-duplicate risk, which validates the privacy audit.
"""

from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.datasets import load_digits


ORIGINAL_DIR = Path("data/input/digits_original")
SYNTHETIC_DIR = Path("data/input/digits_transformed")


def reset_folder(folder: Path) -> None:
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)


def main() -> None:
    digits = load_digits()

    reset_folder(ORIGINAL_DIR)
    reset_folder(SYNTHETIC_DIR)

    # Use 200 real handwritten digits: 20 examples for each digit class.
    selected_indices = []

    for digit_label in range(10):
        matching = np.where(digits.target == digit_label)[0][:20]
        selected_indices.extend(matching.tolist())

    for output_index, source_index in enumerate(selected_indices):
        pixels = digits.images[source_index]

        # Convert sklearn's 8x8 values from 0-16 into normal 0-255 pixels.
        image_array = (pixels / 16.0 * 255).astype(np.uint8)
        original_image = Image.fromarray(image_array, mode="L").resize(
            (128, 128),
            Image.Resampling.NEAREST,
        )

        original_path = ORIGINAL_DIR / f"digit_{output_index:03d}.png"
        original_image.save(original_path)

        # These are intentionally small transformations, not a true generator.
        # The resulting high duplicate risk is expected and validates the audit.
        transformed_image = original_image.filter(ImageFilter.GaussianBlur(radius=0.7))
        transformed_image = ImageEnhance.Contrast(transformed_image).enhance(1.15)

        transformed_path = SYNTHETIC_DIR / f"synthetic_digit_{output_index:03d}.png"
        transformed_image.save(transformed_path)

    print("Created real-image benchmark folders:")
    print(f"  Original images: {ORIGINAL_DIR}")
    print(f"  Comparison images: {SYNTHETIC_DIR}")
    print(f"  Total images in each folder: {len(selected_indices)}")


if __name__ == "__main__":
    main()