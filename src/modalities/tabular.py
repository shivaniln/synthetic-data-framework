"""
Tabular modality adapter.

This file connects the existing tabular-data components to the
new multi-modality project structure. It does not replace the
current tabular implementation yet.
"""

from src.data_loader import DataLoader, LoaderConfig
from src.models.base_generator import build_generator, GENERATOR_REGISTRY
from src.evaluation.attacks import PrivacyAttacks
from src.evaluation.metrics import StatisticalMetrics


class TabularModality:
    """Provides the currently supported CSV/tabular audit components."""

    name = "tabular"
    accepted_extensions = {".csv"}

    def create_loader(self, file_path: str) -> DataLoader:
        return DataLoader(file_path, config=LoaderConfig())

    def create_generator(self, model_name: str, metadata, **kwargs):
        return build_generator(model_name, metadata, **kwargs)

    def create_privacy_evaluator(self, train_df, synthetic_df, control_df, n_attacks: int):
        return PrivacyAttacks(
            train_df,
            synthetic_df,
            control_df,
            n_attacks=n_attacks,
        )

    def create_utility_evaluator(self) -> StatisticalMetrics:
        return StatisticalMetrics()

    def available_models(self) -> list[str]:
        return list(GENERATOR_REGISTRY.keys())