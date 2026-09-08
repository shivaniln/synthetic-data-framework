"""
Standalone test for the MIDST image-audit pipeline.

Run:
    python test_image_audit.py

It creates two local demo folders:
- original_demo_images
- synthetic_demo_images

The synthetic images are modified versions of the originals, allowing
the image privacy and utility audit to be tested without a generator.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.core.runner import ImageAuditRunner


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

ORIGINAL_DIR = (
    PROJECT_ROOT / "data" / "input" / "original_demo_images"
)

SYNTHETIC_DIR = (
    PROJECT_ROOT / "data" / "input" / "synthetic_demo_images"
)


def emit(level: str, message: str) -> None:
    """Print runner logs in the terminal."""
    print(f"[{level.upper()}] {message}")


def create_demo_image_folders() -> None:
    """Create a small original/synthetic image pair dataset."""
    if list(ORIGINAL_DIR.glob("*.png")) and list(
        SYNTHETIC_DIR.glob("*.png")
    ):
        print("Using existing image demo folders.")
        return

    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    for index in range(30):
        background = (
            int(rng.integers(20, 100)),
            int(rng.integers(20, 100)),
            int(rng.integers(20, 100)),
        )

        original = Image.new(
            "RGB",
            (128, 128),
            background,
        )

        original_draw = ImageDraw.Draw(original)

        circle_x = int(rng.integers(20, 70))
        circle_y = int(rng.integers(20, 70))
        radius = int(rng.integers(18, 35))

        circle_colour = (
            int(rng.integers(120, 255)),
            int(rng.integers(80, 230)),
            int(rng.integers(80, 230)),
        )

        rectangle_colour = (
            int(rng.integers(40, 180)),
            int(rng.integers(80, 220)),
            int(rng.integers(120, 255)),
        )

        original_draw.ellipse(
            (
                circle_x,
                circle_y,
                circle_x + radius,
                circle_y + radius,
            ),
            fill=circle_colour,
        )

        original_draw.rectangle(
            (
                18,
                88,
                110,
                108,
            ),
            fill=rectangle_colour,
        )

        original.save(
            ORIGINAL_DIR / f"original_{index:03d}.png"
        )

        # Create a related but altered synthetic image.
        synthetic = Image.new(
            "RGB",
            (128, 128),
            tuple(
                max(0, min(255, value + int(rng.integers(-18, 19))))
                for value in background
            ),
        )

        synthetic_draw = ImageDraw.Draw(synthetic)

        synthetic_draw.ellipse(
            (
                circle_x + int(rng.integers(-10, 11)),
                circle_y + int(rng.integers(-10, 11)),
                circle_x + radius + int(rng.integers(-10, 11)),
                circle_y + radius + int(rng.integers(-10, 11)),
            ),
            fill=tuple(
                max(0, min(255, value + int(rng.integers(-22, 23))))
                for value in circle_colour
            ),
        )

        synthetic_draw.rectangle(
            (
                18,
                88,
                110,
                108,
            ),
            fill=tuple(
                max(0, min(255, value + int(rng.integers(-22, 23))))
                for value in rectangle_colour
            ),
        )

        synthetic.save(
            SYNTHETIC_DIR / f"synthetic_{index:03d}.png"
        )

    print("Created original and synthetic demo image folders.")


def main() -> None:
    create_demo_image_folders()

    runner = ImageAuditRunner(
        output_dir=OUTPUT_DIR,
        emit=emit,
    )

    config = {
        "original_folder": str(ORIGINAL_DIR),
        "synthetic_folder": str(SYNTHETIC_DIR),
        "max_images": 30,
        "hash_distance_limit": 6,
        "privacy_thresholds": {
            "near_duplicate": 0.10,
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

    result = summary["results"][0]

    print("\n" + "=" * 55)
    print("IMAGE AUDIT COMPLETE")
    print("=" * 55)
    print(
        "Original images:",
        summary["dataset_details"]["original_images"],
    )
    print(
        "Synthetic images:",
        summary["dataset_details"]["synthetic_images"],
    )
    print(
        "Near-duplicate risk:",
        f"{result['near_duplicate_risk']:.1%}",
    )
    print(
        "Composite utility:",
        f"{result['composite_utility']:.1%}",
    )
    print(
        "Composite score:",
        f"{result['composite_score']:.3f}",
    )
    print(
        "Passed thresholds:",
        summary["threshold_passed"],
    )


if __name__ == "__main__":
    main()