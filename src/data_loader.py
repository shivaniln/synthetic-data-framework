"""
data_loader.py — MIDST Framework
Handles CSV ingestion, cleaning, and SDV metadata extraction.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sdv.metadata import Metadata

logger = logging.getLogger(__name__)


class DataLoaderError(Exception):
    """Raised when data loading or validation fails."""
    pass


@dataclass
class LoaderConfig:
    """
    All tuneable parameters in one place.
    Pass this from your YAML config — never hardcode values in logic.
    """
    high_cardinality_threshold: float = 0.5   # drop col if nunique/nrows > this
    min_rows_after_cleaning: int = 50          # raise if fewer rows survive
    drop_id_patterns: list = field(default_factory=lambda: [
        "id", "_id", "uuid", "index", "row_num"
    ])
    metadata_overrides: dict = field(default_factory=dict)  # col -> sdv_type


@dataclass
class LoaderResult:
    """Structured output — downstream modules consume this, not raw tuples."""
    df: pd.DataFrame
    metadata: Metadata
    original_shape: tuple
    cleaned_shape: tuple
    dropped_columns: list
    imputation_log: dict  # col -> strategy used


class DataLoader:
    """
    Loads, validates, and cleans a tabular CSV for synthetic data generation.

    Usage:
        loader = DataLoader("data/input/adult.csv", config=LoaderConfig())
        result = loader.load_and_clean()
        df, metadata = result.df, result.metadata
    """

    def __init__(self, file_path: str, config: Optional[LoaderConfig] = None):
        self.file_path = Path(file_path)
        self.config = config or LoaderConfig()
        self._result: Optional[LoaderResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_and_clean(self) -> LoaderResult:
        """Full pipeline: load → validate → clean → extract metadata."""
        df = self._load_csv()
        original_shape = df.shape

        df, dropped_columns = self._drop_useless_columns(df)
        df, imputation_log = self._impute(df)
        self._validate_post_clean(df)

        metadata = self._build_metadata(df)

        self._result = LoaderResult(
            df=df,
            metadata=metadata,
            original_shape=original_shape,
            cleaned_shape=df.shape,
            dropped_columns=dropped_columns,
            imputation_log=imputation_log,
        )

        logger.info(
            "Loaded %s | %d→%d rows | %d→%d cols | dropped: %s",
            self.file_path.name,
            original_shape[0], df.shape[0],
            original_shape[1], df.shape[1],
            dropped_columns or "none",
        )
        return self._result

    # ------------------------------------------------------------------
    # Private steps
    # ------------------------------------------------------------------

    def _load_csv(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise DataLoaderError(f"File not found: {self.file_path}")

        try:
            df = pd.read_csv(self.file_path)
        except Exception as exc:
            raise DataLoaderError(f"Could not parse CSV: {exc}") from exc

        if df.empty:
            raise DataLoaderError("CSV is empty.")

        # Drop rows/columns that are entirely null
        df = df.dropna(how="all", axis=0)
        df = df.dropna(how="all", axis=1)
        return df

    def _drop_useless_columns(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        """
        Drop columns that will hurt generation quality:
          - likely identifier columns (by name pattern)
          - high-cardinality categorical columns (unique ratio > threshold)
        """
        dropped = []

        for col in df.columns:
            col_lower = col.lower()

            # Drop by name pattern (IDs, row numbers, etc.)
            if any(pat in col_lower for pat in self.config.drop_id_patterns):
                dropped.append(col)
                logger.warning("Dropping likely ID column: '%s'", col)
                continue

            # Drop high-cardinality categoricals
            if self._is_categorical(df[col]):
                ratio = df[col].nunique() / len(df)
                if ratio > self.config.high_cardinality_threshold:
                    dropped.append(col)
                    logger.warning(
                        "Dropping high-cardinality column '%s' (%.0f%% unique values).",
                        col, ratio * 100,
                    )

        return df.drop(columns=dropped), dropped

    def _impute(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Impute missing values per column.
        Returns the cleaned DataFrame and a log of strategies used.
        """
        df = df.copy()
        imputation_log = {}

        for col in df.columns:
            null_count = df[col].isna().sum()
            if null_count == 0:
                continue

            if self._is_datetime(df[col]):
                # Forward-fill datetimes; flag in log
                df[col] = df[col].fillna(method="ffill").fillna(method="bfill")
                imputation_log[col] = f"ffill+bfill ({null_count} nulls)"

            elif self._is_categorical(df[col]):
                mode_vals = df[col].mode()
                fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
                df[col] = df[col].fillna(fill_val)
                # Normalize string whitespace (prevents GAN category duplication)
                df[col] = df[col].astype(str).str.strip()
                imputation_log[col] = f"mode='{fill_val}' ({null_count} nulls)"

            elif pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)
                imputation_log[col] = f"median={fill_val:.4g} ({null_count} nulls)"

            else:
                # Fallback: drop remaining nulls for unknown dtypes
                before = len(df)
                df = df.dropna(subset=[col])
                imputation_log[col] = f"rows_dropped ({before - len(df)} rows)"

        return df, imputation_log

    def _validate_post_clean(self, df: pd.DataFrame) -> None:
        """Raise early if cleaning left us with too little data."""
        if len(df) < self.config.min_rows_after_cleaning:
            raise DataLoaderError(
                f"Only {len(df)} rows survived cleaning "
                f"(minimum: {self.config.min_rows_after_cleaning}). "
                "Check your data or lower min_rows_after_cleaning in LoaderConfig."
            )
        if df.shape[1] < 2:
            raise DataLoaderError(
                "Fewer than 2 columns remain after cleaning. Cannot generate synthetic data."
            )

    def _build_metadata(self, df: pd.DataFrame) -> Metadata:
        """
        Auto-detect SDV metadata, then apply any manual overrides.

        Overrides example (pass via LoaderConfig):
            metadata_overrides = {"zip_code": "categorical", "age": "numerical"}
        """
        metadata = Metadata.detect_from_dataframe(
            data=df,
            table_name="input_table"
        )

        for col, sdv_type in self.config.metadata_overrides.items():
            if col in df.columns:
                metadata.update_column(
                    table_name="input_table",
                    column_name=col,
                    sdtype=sdv_type,
                )
                logger.info("Metadata override: '%s' → %s", col, sdv_type)
            else:
                logger.warning("Override column '%s' not found in DataFrame.", col)

        return metadata

    # ------------------------------------------------------------------
    # Dtype helpers — centralised so the logic is not scattered
    # ------------------------------------------------------------------

    @staticmethod
    def _is_categorical(series: pd.Series) -> bool:
        return (
            pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
            or series.dtype == object
        )

    @staticmethod
    def _is_datetime(series: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        # Try to parse if it looks like a date string
        if series.dtype == object and series.dropna().shape[0] > 0:
            try:
                pd.to_datetime(series.dropna().iloc[:5])
                return True
            except Exception:
                pass
        return False
