"""
Image modality for MIDST Phase 2.

This first image module audits an original image dataset against an
already generated synthetic image dataset. It does not generate images.

The audit measures:
- Colour-distribution similarity
- Visual feature similarity
- Near-duplicate privacy risk
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


@dataclass
class ImageDataset:
    """Loaded image paths and extracted numerical features."""

    paths: list[Path]
    features: np.ndarray
    histograms: np.ndarray
    hashes: list[int]


@dataclass
class ImageUtilityResult:
    """Image quality and similarity measures."""

    colour_similarity: float
    visual_similarity: float
    structure_similarity: float
    composite_utility: float


@dataclass
class ImagePrivacyResult:
    """Risk result based on near-duplicate image matches."""

    near_duplicate_risk: float
    duplicate_count: int
    notes: str


class ImageFolderLoader:
    """Load a local folder of images and create audit features."""

    def __init__(
        self,
        folder_path: str,
        max_images: int | None = None,
    ):
        self.folder_path = Path(folder_path)
        self.max_images = max_images

    def load(self) -> ImageDataset:
        """Load images from a folder and extract their visual features."""
        if not self.folder_path.exists():
            raise ValueError(
                f"Image folder was not found: {self.folder_path}"
            )

        if not self.folder_path.is_dir():
            raise ValueError(
                "Image input must be a folder containing image files."
            )

        image_paths = sorted(
            path
            for path in self.folder_path.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )

        if self.max_images:
            image_paths = image_paths[:self.max_images]

        if not image_paths:
            raise ValueError(
                "No supported image files were found. "
                "Use PNG, JPG, JPEG, BMP, or WEBP files."
            )

        features = []
        histograms = []
        hashes = []
        valid_paths = []

        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")

                feature, histogram, image_hash = (
                    self._extract_features(image)
                )

                features.append(feature)
                histograms.append(histogram)
                hashes.append(image_hash)
                valid_paths.append(path)

            except Exception:
                # Ignore unreadable files without stopping the full audit.
                continue

        if not valid_paths:
            raise ValueError(
                "The folder did not contain readable image files."
            )

        return ImageDataset(
            paths=valid_paths,
            features=np.vstack(features),
            histograms=np.vstack(histograms),
            hashes=hashes,
        )

    def _extract_features(
        self,
        image: Image.Image,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Create compact colour, structure, and perceptual-hash features."""
        thumbnail = image.resize((32, 32))
        array = np.asarray(thumbnail, dtype=np.float32) / 255.0

        histogram_parts = []

        for channel in range(3):
            histogram, _ = np.histogram(
                array[:, :, channel],
                bins=16,
                range=(0.0, 1.0),
                density=True,
            )

            histogram_parts.append(histogram)

        histogram_feature = np.concatenate(histogram_parts)
        histogram_feature = histogram_feature / (
            np.linalg.norm(histogram_feature) + 1e-8
        )

        mean_rgb = array.mean(axis=(0, 1))
        std_rgb = array.std(axis=(0, 1))

        grayscale = (
            0.299 * array[:, :, 0]
            + 0.587 * array[:, :, 1]
            + 0.114 * array[:, :, 2]
        )

        horizontal_edges = np.abs(np.diff(grayscale, axis=1)).mean()
        vertical_edges = np.abs(np.diff(grayscale, axis=0)).mean()

        low_resolution = image.resize((8, 8))
        low_array = np.asarray(
            low_resolution,
            dtype=np.float32,
        ).flatten() / 255.0

        feature = np.concatenate(
            [
                mean_rgb,
                std_rgb,
                np.array(
                    [
                        horizontal_edges,
                        vertical_edges,
                    ]
                ),
                histogram_feature,
                low_array,
            ]
        )

        feature = feature / (
            np.linalg.norm(feature) + 1e-8
        )

        image_hash = self._average_hash(image)

        return feature, histogram_feature, image_hash

    @staticmethod
    def _average_hash(image: Image.Image) -> int:
        """Create a 64-bit perceptual hash for duplicate checking."""
        grayscale = image.convert("L").resize((8, 8))
        pixels = np.asarray(grayscale, dtype=np.float32)

        average = pixels.mean()
        bits = (pixels >= average).flatten()

        value = 0

        for bit in bits:
            value = (value << 1) | int(bit)

        return value


class ImageMetrics:
    """Evaluate visual similarity between original and synthetic images."""

    def evaluate(
        self,
        original: ImageDataset,
        synthetic: ImageDataset,
    ) -> ImageUtilityResult:
        """Calculate colour, visual, and structure similarity scores."""
        colour_similarity = self._centroid_similarity(
            original.histograms,
            synthetic.histograms,
        )

        visual_similarity = self._centroid_similarity(
            original.features,
            synthetic.features,
        )

        original_structure = original.features[:, :8]
        synthetic_structure = synthetic.features[:, :8]

        structure_similarity = self._centroid_similarity(
            original_structure,
            synthetic_structure,
        )

        composite_utility = float(
            np.mean(
                [
                    colour_similarity,
                    visual_similarity,
                    structure_similarity,
                ]
            )
        )

        return ImageUtilityResult(
            colour_similarity=colour_similarity,
            visual_similarity=visual_similarity,
            structure_similarity=structure_similarity,
            composite_utility=composite_utility,
        )

    @staticmethod
    def _centroid_similarity(
        original_features: np.ndarray,
        synthetic_features: np.ndarray,
    ) -> float:
        """Compare average feature profiles using cosine similarity."""
        original_centroid = original_features.mean(axis=0)
        synthetic_centroid = synthetic_features.mean(axis=0)

        numerator = float(
            np.dot(
                original_centroid,
                synthetic_centroid,
            )
        )

        denominator = float(
            np.linalg.norm(original_centroid)
            * np.linalg.norm(synthetic_centroid)
        )

        if denominator < 1e-8:
            return 0.0

        similarity = numerator / denominator

        return float(np.clip(similarity, 0.0, 1.0))


class ImagePrivacyEvaluator:
    """
    Estimate privacy risk from near-duplicate synthetic images.

    A synthetic image is considered risky when its perceptual hash is
    very close to an original image hash.
    """

    def __init__(
        self,
        original: ImageDataset,
        synthetic: ImageDataset,
        hash_distance_limit: int = 6,
    ):
        self.original = original
        self.synthetic = synthetic
        self.hash_distance_limit = hash_distance_limit

    def near_duplicate_risk(self) -> ImagePrivacyResult:
        """Calculate the rate of near-duplicate synthetic images."""
        duplicate_count = 0

        for synthetic_hash in self.synthetic.hashes:
            nearest_distance = min(
                self._hamming_distance(
                    synthetic_hash,
                    original_hash,
                )
                for original_hash in self.original.hashes
            )

            if nearest_distance <= self.hash_distance_limit:
                duplicate_count += 1

        risk_score = duplicate_count / len(self.synthetic.hashes)

        return ImagePrivacyResult(
            near_duplicate_risk=float(risk_score),
            duplicate_count=duplicate_count,
            notes=(
                "Risk is the proportion of synthetic images whose "
                "perceptual hash is very close to an original image."
            ),
        )

    @staticmethod
    def _hamming_distance(
        first_hash: int,
        second_hash: int,
    ) -> int:
        """Count different bits between two perceptual hashes."""
        return (first_hash ^ second_hash).bit_count()


class ImageModality:
    """Factory interface for local image-audit components."""

    name = "image"
    accepted_extensions = SUPPORTED_IMAGE_EXTENSIONS

    def create_loader(
        self,
        folder_path: str,
        max_images: int | None = None,
    ) -> ImageFolderLoader:
        return ImageFolderLoader(
            folder_path=folder_path,
            max_images=max_images,
        )

    def create_utility_evaluator(self) -> ImageMetrics:
        return ImageMetrics()

    def create_privacy_evaluator(
        self,
        original: ImageDataset,
        synthetic: ImageDataset,
        hash_distance_limit: int = 6,
    ) -> ImagePrivacyEvaluator:
        return ImagePrivacyEvaluator(
            original=original,
            synthetic=synthetic,
            hash_distance_limit=hash_distance_limit,
        )