"""
Time-series modality for MIDST Phase 2.

Supports CSV datasets containing:
- one timestamp column
- one or more numerical measurement columns

Generators:
- Block Bootstrap: preserves short local sequence patterns
- Fourier Surrogate: preserves dominant periodic patterns
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TimeSeriesLoadResult:
    """Prepared time-series dataset and detected schema details."""

    df: pd.DataFrame
    timestamp_column: str
    value_columns: list[str]
    original_shape: tuple[int, int]
    cleaned_shape: tuple[int, int]


@dataclass
class TimeSeriesGenerationResult:
    """Synthetic output returned by a time-series generator."""

    synthetic_df: pd.DataFrame
    model_name: str
    config: dict


@dataclass
class TimeSeriesUtilityResult:
    """Utility scores for original and synthetic sequences."""

    trend_similarity: float
    autocorrelation_similarity: float
    distribution_similarity: float
    composite_utility: float


@dataclass
class TimeSeriesPrivacyResult:
    """Sequence-similarity privacy result."""

    risk_score: float
    threshold: float
    notes: str


class TimeSeriesLoader:
    """Load, validate, clean, and sort a timestamped CSV dataset."""

    def __init__(
        self,
        file_path: str,
        timestamp_column: str | None = None,
    ):
        self.file_path = Path(file_path)
        self.timestamp_column = timestamp_column

    def load_and_prepare(self) -> TimeSeriesLoadResult:
        """Read and prepare a CSV dataset for sequence generation."""
        df = pd.read_csv(self.file_path)
        original_shape = df.shape

        timestamp_column = self._detect_timestamp_column(df)

        df[timestamp_column] = pd.to_datetime(
            df[timestamp_column],
            errors="coerce",
        )

        df = df.dropna(subset=[timestamp_column]).copy()

        value_columns = [
            column
            for column in df.columns
            if column != timestamp_column
            and pd.api.types.is_numeric_dtype(df[column])
        ]

        if not value_columns:
            raise ValueError(
                "No numeric value columns were found. "
                "A time-series CSV needs a timestamp column and at least "
                "one numerical measurement column."
            )

        for column in value_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            df[column] = df[column].interpolate(
                limit_direction="both",
            )
            df[column] = df[column].fillna(df[column].median())

        df = (
            df[[timestamp_column, *value_columns]]
            .sort_values(timestamp_column)
            .drop_duplicates(subset=[timestamp_column])
            .reset_index(drop=True)
        )

        if len(df) < 48:
            raise ValueError(
                "The time-series dataset needs at least 48 valid rows."
            )

        return TimeSeriesLoadResult(
            df=df,
            timestamp_column=timestamp_column,
            value_columns=value_columns,
            original_shape=original_shape,
            cleaned_shape=df.shape,
        )

    def _detect_timestamp_column(self, df: pd.DataFrame) -> str:
        """Use a supplied timestamp column or detect a likely one."""
        if self.timestamp_column:
            if self.timestamp_column not in df.columns:
                raise ValueError(
                    f"Timestamp column '{self.timestamp_column}' was not found."
                )
            return self.timestamp_column

        preferred_names = {
            "timestamp",
            "time",
            "date",
            "datetime",
            "datetime_utc",
        }

        for column in df.columns:
            if column.strip().lower() in preferred_names:
                return column

        for column in df.columns:
            converted = pd.to_datetime(
                df[column],
                errors="coerce",
            )

            if converted.notna().mean() >= 0.8:
                return column

        raise ValueError(
            "Could not detect a timestamp column. "
            "Use a column named timestamp, time, date, or datetime."
        )


class BlockBootstrapGenerator:
    """
    Sequence-aware baseline generator.

    It samples contiguous blocks instead of individual rows to retain
    local dependencies such as short-term trends and autocorrelation.
    """

    name = "BlockBootstrap"

    def __init__(
        self,
        timestamp_column: str,
        value_columns: list[str],
        block_size: int = 24,
        noise_scale: float = 0.03,
        random_seed: int = 42,
    ):
        self.timestamp_column = timestamp_column
        self.value_columns = value_columns
        self.block_size = block_size
        self.noise_scale = noise_scale
        self.random_seed = random_seed
        self._train_df: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> None:
        """Store the cleaned training sequence."""
        if len(df) < self.block_size:
            raise ValueError(
                f"Training data needs at least {self.block_size} rows "
                "for the selected block size."
            )

        self._train_df = df.reset_index(drop=True).copy()

    def sample(self, n_rows: int) -> TimeSeriesGenerationResult:
        """Generate a synthetic sequence with the requested length."""
        if self._train_df is None:
            raise RuntimeError("fit() must be called before sample().")

        rng = np.random.default_rng(self.random_seed)
        blocks = []
        rows_collected = 0
        max_start = len(self._train_df) - self.block_size

        while rows_collected < n_rows:
            start = int(rng.integers(0, max_start + 1))

            block = self._train_df.iloc[
                start:start + self.block_size
            ].copy()

            blocks.append(block)
            rows_collected += len(block)

        synthetic_df = pd.concat(
            blocks,
            ignore_index=True,
        ).iloc[:n_rows].copy()

        synthetic_df[self.timestamp_column] = self._create_timestamps(
            n_rows
        )

        for column in self.value_columns:
            source_std = float(self._train_df[column].std())

            if np.isfinite(source_std) and source_std > 0:
                noise = rng.normal(
                    loc=0,
                    scale=source_std * self.noise_scale,
                    size=n_rows,
                )

                synthetic_df[column] = synthetic_df[column] + noise

            lower = self._train_df[column].quantile(0.001)
            upper = self._train_df[column].quantile(0.999)

            synthetic_df[column] = synthetic_df[column].clip(
                lower=lower,
                upper=upper,
            )

        return TimeSeriesGenerationResult(
            synthetic_df=synthetic_df,
            model_name=self.name,
            config={
                "block_size": self.block_size,
                "noise_scale": self.noise_scale,
                "random_seed": self.random_seed,
            },
        )

    def _create_timestamps(self, n_rows: int) -> pd.Series:
        """Create a regular timestamp sequence over the training range."""
        assert self._train_df is not None

        timestamps = pd.to_datetime(
            self._train_df[self.timestamp_column]
        )

        return _generate_regular_timestamps(
            timestamps,
            n_rows,
        )


class FourierSurrogateGenerator:
    """
    Frequency-based synthetic generator.

    It retains strong Fourier components that represent periodic
    behaviour, then shifts and perturbs the reconstructed sequence.
    """

    name = "FourierSurrogate"

    def __init__(
        self,
        timestamp_column: str,
        value_columns: list[str],
        harmonics: int = 24,
        noise_scale: float = 0.03,
        random_seed: int = 42,
    ):
        self.timestamp_column = timestamp_column
        self.value_columns = value_columns
        self.harmonics = harmonics
        self.noise_scale = noise_scale
        self.random_seed = random_seed
        self._train_df: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> None:
        """Store the cleaned training sequence."""
        if len(df) < 24:
            raise ValueError(
                "Fourier Surrogate needs at least 24 training rows."
            )

        self._train_df = df.reset_index(drop=True).copy()

    def sample(self, n_rows: int) -> TimeSeriesGenerationResult:
        """Generate synthetic values using dominant frequency patterns."""
        if self._train_df is None:
            raise RuntimeError("fit() must be called before sample().")

        rng = np.random.default_rng(self.random_seed)

        source = self._repeat_source_rows(n_rows)

        synthetic_df = pd.DataFrame()

        timestamps = pd.to_datetime(
            self._train_df[self.timestamp_column]
        )

        synthetic_df[self.timestamp_column] = _generate_regular_timestamps(
            timestamps,
            n_rows,
        )

        for column in self.value_columns:
            values = source[column].to_numpy(dtype=float)

            generated = self._generate_series(
                values,
                rng,
            )

            lower = self._train_df[column].quantile(0.001)
            upper = self._train_df[column].quantile(0.999)

            synthetic_df[column] = np.clip(
                generated,
                lower,
                upper,
            )

        return TimeSeriesGenerationResult(
            synthetic_df=synthetic_df,
            model_name=self.name,
            config={
                "harmonics": self.harmonics,
                "noise_scale": self.noise_scale,
                "random_seed": self.random_seed,
            },
        )

    def _repeat_source_rows(self, n_rows: int) -> pd.DataFrame:
        """Repeat source rows only when more output rows are requested."""
        assert self._train_df is not None

        if n_rows <= len(self._train_df):
            return self._train_df.iloc[:n_rows].copy()

        repeats = int(np.ceil(n_rows / len(self._train_df)))

        return pd.concat(
            [self._train_df] * repeats,
            ignore_index=True,
        ).iloc[:n_rows].copy()

    def _generate_series(
        self,
        values: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Reconstruct a sequence from its strongest frequency signals."""
        mean = float(np.mean(values))
        centered = values - mean

        spectrum = np.fft.rfft(centered)
        magnitudes = np.abs(spectrum)

        keep_count = min(
            self.harmonics,
            max(1, len(spectrum) - 1),
        )

        strongest_indices = np.argsort(magnitudes)[-keep_count:]

        filtered_spectrum = np.zeros_like(spectrum)
        filtered_spectrum[strongest_indices] = (
            spectrum[strongest_indices]
        )

        reconstructed = np.fft.irfft(
            filtered_spectrum,
            n=len(values),
        ) + mean

        # Prevent the output from retaining original values at the
        # same positions in time.
        shift = int(rng.integers(1, max(2, len(values))))
        reconstructed = np.roll(reconstructed, shift)

        source_std = float(np.std(values))

        if source_std > 0:
            reconstructed += rng.normal(
                loc=0,
                scale=source_std * self.noise_scale,
                size=len(values),
            )

        return reconstructed


class TimeSeriesMetrics:
    """Calculate time-series utility scores."""

    def evaluate(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        value_columns: list[str],
    ) -> TimeSeriesUtilityResult:
        """Compare trend, autocorrelation, and distributions."""
        real_values = real_df[value_columns].reset_index(drop=True)
        synthetic_values = synthetic_df[value_columns].reset_index(
            drop=True
        )

        common_length = min(
            len(real_values),
            len(synthetic_values),
        )

        real_values = real_values.iloc[:common_length]
        synthetic_values = synthetic_values.iloc[:common_length]

        trend_scores = []
        autocorrelation_scores = []
        distribution_scores = []

        for column in value_columns:
            real_series = real_values[column].to_numpy(dtype=float)
            synthetic_series = synthetic_values[column].to_numpy(
                dtype=float
            )

            trend_scores.append(
                self._trend_similarity(
                    real_series,
                    synthetic_series,
                )
            )

            autocorrelation_scores.append(
                self._autocorrelation_similarity(
                    real_series,
                    synthetic_series,
                )
            )

            distribution_scores.append(
                self._distribution_similarity(
                    real_series,
                    synthetic_series,
                )
            )

        trend_similarity = float(np.mean(trend_scores))

        autocorrelation_similarity = float(
            np.mean(autocorrelation_scores)
        )

        distribution_similarity = float(
            np.mean(distribution_scores)
        )

        composite_utility = float(
            np.mean(
                [
                    trend_similarity,
                    autocorrelation_similarity,
                    distribution_similarity,
                ]
            )
        )

        return TimeSeriesUtilityResult(
            trend_similarity=trend_similarity,
            autocorrelation_similarity=autocorrelation_similarity,
            distribution_similarity=distribution_similarity,
            composite_utility=composite_utility,
        )

    def _trend_similarity(
        self,
        real_series: np.ndarray,
        synthetic_series: np.ndarray,
    ) -> float:
        """Compare normalized rolling-average trends."""
        real_trend = self._normalize(
            self._rolling_mean(real_series)
        )

        synthetic_trend = self._normalize(
            self._rolling_mean(synthetic_series)
        )

        error = float(
            np.mean(
                np.abs(real_trend - synthetic_trend)
            )
        )

        return float(np.clip(1.0 - error, 0.0, 1.0))

    def _autocorrelation_similarity(
        self,
        real_series: np.ndarray,
        synthetic_series: np.ndarray,
        max_lag: int = 10,
    ) -> float:
        """Compare autocorrelation values across several lags."""
        max_lag = min(
            max_lag,
            len(real_series) // 4,
        )

        if max_lag < 1:
            return 0.0

        real_acf = [
            self._autocorrelation(real_series, lag)
            for lag in range(1, max_lag + 1)
        ]

        synthetic_acf = [
            self._autocorrelation(synthetic_series, lag)
            for lag in range(1, max_lag + 1)
        ]

        error = float(
            np.mean(
                np.abs(
                    np.array(real_acf) - np.array(synthetic_acf)
                )
            )
        )

        return float(np.clip(1.0 - error, 0.0, 1.0))

    def _distribution_similarity(
        self,
        real_series: np.ndarray,
        synthetic_series: np.ndarray,
    ) -> float:
        """Compare basic distribution statistics."""
        real_stats = np.array(
            [
                np.mean(real_series),
                np.std(real_series),
                np.quantile(real_series, 0.25),
                np.quantile(real_series, 0.50),
                np.quantile(real_series, 0.75),
            ]
        )

        synthetic_stats = np.array(
            [
                np.mean(synthetic_series),
                np.std(synthetic_series),
                np.quantile(synthetic_series, 0.25),
                np.quantile(synthetic_series, 0.50),
                np.quantile(synthetic_series, 0.75),
            ]
        )

        scale = np.maximum(np.abs(real_stats), 1e-8)

        relative_error = float(
            np.mean(
                np.abs(real_stats - synthetic_stats) / scale
            )
        )

        return float(
            np.clip(
                1.0 - relative_error,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _rolling_mean(values: np.ndarray) -> np.ndarray:
        window = max(
            3,
            min(12, len(values) // 10),
        )

        return (
            pd.Series(values)
            .rolling(
                window=window,
                min_periods=1,
                center=True,
            )
            .mean()
            .to_numpy()
        )

    @staticmethod
    def _autocorrelation(
        values: np.ndarray,
        lag: int,
    ) -> float:
        if len(values) <= lag:
            return 0.0

        left = values[:-lag]
        right = values[lag:]

        if np.std(left) == 0 or np.std(right) == 0:
            return 0.0

        return float(
            np.corrcoef(left, right)[0, 1]
        )

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        lower = np.min(values)
        upper = np.max(values)

        if upper - lower < 1e-8:
            return np.zeros_like(values)

        return (values - lower) / (upper - lower)


class TimeSeriesPrivacyEvaluator:
    """
    Estimate privacy risk from highly similar sequence windows.

    A low score means fewer synthetic windows are unusually close to
    held-out real control windows.
    """

    def __init__(
        self,
        control_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        value_columns: list[str],
        window_size: int = 12,
        max_queries: int = 100,
        random_seed: int = 42,
    ):
        self.control_df = control_df.reset_index(drop=True)
        self.synthetic_df = synthetic_df.reset_index(drop=True)
        self.value_columns = value_columns
        self.window_size = window_size
        self.max_queries = max_queries
        self.random_seed = random_seed

    def sequence_similarity_risk(self) -> TimeSeriesPrivacyResult:
        """Calculate the proportion of unusually close synthetic windows."""
        control_windows = self._build_windows(self.control_df)

        synthetic_windows = self._build_windows(
            self.synthetic_df
        )

        if not control_windows or not synthetic_windows:
            return TimeSeriesPrivacyResult(
                risk_score=0.0,
                threshold=0.0,
                notes="Not enough rows to create sequence windows.",
            )

        control_matrix = np.vstack(control_windows)

        synthetic_matrix = np.vstack(
            synthetic_windows[:self.max_queries]
        )

        control_matrix = self._standardize(control_matrix)

        synthetic_matrix = self._standardize(
            synthetic_matrix
        )

        baseline_distances = self._pairwise_nearest_distances(
            control_matrix
        )

        threshold = float(
            np.quantile(baseline_distances, 0.05)
        )

        close_count = 0

        for synthetic_window in synthetic_matrix:
            distances = np.linalg.norm(
                control_matrix - synthetic_window,
                axis=1,
            )

            if float(np.min(distances)) <= threshold:
                close_count += 1

        risk_score = close_count / len(synthetic_matrix)

        return TimeSeriesPrivacyResult(
            risk_score=float(risk_score),
            threshold=threshold,
            notes=(
                "Risk is the proportion of sampled synthetic sequence "
                "windows that are unusually close to a control-set window."
            ),
        )

    def _build_windows(
        self,
        df: pd.DataFrame,
    ) -> list[np.ndarray]:
        values = df[self.value_columns].to_numpy(dtype=float)

        if len(values) < self.window_size:
            return []

        windows = []

        for start in range(
            len(values) - self.window_size + 1
        ):
            window = values[
                start:start + self.window_size
            ].flatten()

            windows.append(window)

        return windows

    @staticmethod
    def _standardize(matrix: np.ndarray) -> np.ndarray:
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)

        std[std < 1e-8] = 1.0

        return (matrix - mean) / std

    @staticmethod
    def _pairwise_nearest_distances(
        matrix: np.ndarray,
    ) -> np.ndarray:
        distances = []

        for index, row in enumerate(matrix):
            others = np.delete(
                matrix,
                index,
                axis=0,
            )

            if len(others) == 0:
                distances.append(0.0)
                continue

            row_distances = np.linalg.norm(
                others - row,
                axis=1,
            )

            distances.append(
                float(np.min(row_distances))
            )

        return np.array(distances)


class TimeSeriesModality:
    """Factory interface for MIDST time-series components."""

    name = "time_series"
    accepted_extensions = {".csv"}

    def create_loader(
        self,
        file_path: str,
        timestamp_column: str | None = None,
    ) -> TimeSeriesLoader:
        return TimeSeriesLoader(
            file_path,
            timestamp_column=timestamp_column,
        )

    def create_generator(
        self,
        model_name: str,
        timestamp_column: str,
        value_columns: list[str],
        **kwargs,
    ):
        """Create the requested time-series generator."""
        key = model_name.strip().lower()

        if key == "block_bootstrap":
            return BlockBootstrapGenerator(
                timestamp_column=timestamp_column,
                value_columns=value_columns,
                block_size=int(
                    kwargs.get("block_size", 24)
                ),
                noise_scale=float(
                    kwargs.get("noise_scale", 0.03)
                ),
                random_seed=int(
                    kwargs.get("random_seed", 42)
                ),
            )

        if key == "fourier_surrogate":
            return FourierSurrogateGenerator(
                timestamp_column=timestamp_column,
                value_columns=value_columns,
                harmonics=int(
                    kwargs.get("harmonics", 24)
                ),
                noise_scale=float(
                    kwargs.get("noise_scale", 0.03)
                ),
                random_seed=int(
                    kwargs.get("random_seed", 42)
                ),
            )

        raise ValueError(
            f"Unsupported time-series generator: {model_name}"
        )

    def create_utility_evaluator(self) -> TimeSeriesMetrics:
        return TimeSeriesMetrics()

    def create_privacy_evaluator(
        self,
        control_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        value_columns: list[str],
        **kwargs,
    ) -> TimeSeriesPrivacyEvaluator:
        return TimeSeriesPrivacyEvaluator(
            control_df=control_df,
            synthetic_df=synthetic_df,
            value_columns=value_columns,
            window_size=int(
                kwargs.get("window_size", 12)
            ),
            max_queries=int(
                kwargs.get("max_queries", 100)
            ),
            random_seed=int(
                kwargs.get("random_seed", 42)
            ),
        )

    def available_models(self) -> list[str]:
        return [
            "block_bootstrap",
            "fourier_surrogate",
        ]


def _generate_regular_timestamps(
    timestamps: pd.Series,
    n_rows: int,
) -> pd.Series:
    """Create regular timestamps matching the source frequency."""
    timestamps = pd.Series(
        pd.to_datetime(timestamps)
    ).reset_index(drop=True)

    if len(timestamps) < 2:
        return pd.Series(
            [timestamps.iloc[0]] * n_rows
        )

    frequency = pd.infer_freq(timestamps)

    if frequency:
        return pd.Series(
            pd.date_range(
                start=timestamps.iloc[0],
                periods=n_rows,
                freq=frequency,
            )
        )

    deltas = timestamps.diff().dropna()
    median_delta = deltas.median()

    if pd.isna(median_delta) or median_delta <= pd.Timedelta(0):
        median_delta = pd.Timedelta(days=1)

    return pd.Series(
        [
            timestamps.iloc[0] + index * median_delta
            for index in range(n_rows)
        ]
    )