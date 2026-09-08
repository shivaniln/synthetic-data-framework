"""Registry for data modalities supported by MIDST."""

from src.modalities.image import ImageModality
from src.modalities.tabular import TabularModality
from src.modalities.time_series import TimeSeriesModality


MODALITY_REGISTRY = {
    "tabular": TabularModality,
    "time_series": TimeSeriesModality,
    "image": ImageModality,
}


def get_modality(name: str):
    """Return an instance of the requested modality handler."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")

    if key not in MODALITY_REGISTRY:
        supported = ", ".join(MODALITY_REGISTRY.keys())

        raise ValueError(
            f"Unsupported modality '{name}'. "
            f"Supported modalities: {supported}."
        )

    return MODALITY_REGISTRY[key]()