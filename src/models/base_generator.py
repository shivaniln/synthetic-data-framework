"""
models/base_generator.py — MIDST Framework

Abstract base class for all synthetic data generators.
Every model (Copula, CTGAN, TVAE, PrivBayes, PATE-GAN...) is a subclass.
The controller (main.py) only ever calls .fit() and .sample() —
it never needs to know which model it's running.
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from sdv.metadata import Metadata

logger = logging.getLogger(__name__)


@dataclass
class GeneratorResult:
    """
    Structured output from every generator.
    Downstream evaluation modules consume this — not raw DataFrames.
    """
    synthetic_df: pd.DataFrame
    model_name: str
    training_time_seconds: float
    n_rows: int
    config: dict = field(default_factory=dict)
    fit_exception: Optional[str] = None  # set if fit() failed but we want soft failure


class BaseGenerator(ABC):
    """
    Abstract interface all generators must implement.

    Subclasses override:
        _build_model()   — instantiate the SDV / custom model
        _fit(df)         — train on real data
        _sample(n)       — return n synthetic rows as a DataFrame

    Subclasses should NOT override fit() or sample() directly —
    those handle timing, logging, and error wrapping.
    """

    def __init__(self, metadata: Metadata, **kwargs):
        self.metadata = metadata
        self.kwargs = kwargs
        self._model = None
        self._is_fitted = False
        self._build_model()

    # ------------------------------------------------------------------
    # Public API (do not override)
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> None:
        """Train the model on real data. Wraps _fit with timing + logging."""
        logger.info("[%s] Starting training on %d rows × %d cols ...",
                    self.name, df.shape[0], df.shape[1])
        t0 = time.perf_counter()
        self._fit(df)
        self._is_fitted = True
        elapsed = time.perf_counter() - t0
        logger.info("[%s] Training complete in %.1fs", self.name, elapsed)

    def sample(self, n: int) -> GeneratorResult:
        """
        Generate n synthetic rows.
        Returns a GeneratorResult — never a bare DataFrame.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.name}.fit() must be called before sample()."
            )
        logger.info("[%s] Sampling %d rows ...", self.name, n)
        t0 = time.perf_counter()
        synthetic_df = self._sample(n)
        elapsed = time.perf_counter() - t0
        logger.info("[%s] Sampling done in %.2fs", self.name, elapsed)

        return GeneratorResult(
            synthetic_df=synthetic_df,
            model_name=self.name,
            training_time_seconds=elapsed,
            n_rows=n,
            config=self.get_config(),
        )

    # ------------------------------------------------------------------
    # Abstract interface (subclasses must implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_model(self) -> None:
        """Instantiate self._model using self.metadata and self.kwargs."""

    @abstractmethod
    def _fit(self, df: pd.DataFrame) -> None:
        """Train self._model on df."""

    @abstractmethod
    def _sample(self, n: int) -> pd.DataFrame:
        """Return n synthetic rows as a DataFrame."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model identifier, e.g. 'CTGAN'."""

    # ------------------------------------------------------------------
    # Optional override
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        """Return hyperparameters for logging/reporting. Override in subclasses."""
        return self.kwargs


# =============================================================================
# Concrete implementations
# =============================================================================

class CopulaGenerator(BaseGenerator):
    """Gaussian Copula — fast statistical baseline."""

    @property
    def name(self) -> str:
        return "GaussianCopula"

    def _build_model(self) -> None:
        from sdv.single_table import GaussianCopulaSynthesizer
        self._model = GaussianCopulaSynthesizer(self.metadata)

    def _fit(self, df: pd.DataFrame) -> None:
        self._model.fit(df)

    def _sample(self, n: int) -> pd.DataFrame:
        return self._model.sample(num_rows=n)


class CTGANGenerator(BaseGenerator):
    """CTGAN — GAN-based deep learning model."""

    @property
    def name(self) -> str:
        return "CTGAN"

    def _build_model(self) -> None:
        from sdv.single_table import CTGANSynthesizer
        epochs = self.kwargs.get("epochs", 300)
        batch_size = self.kwargs.get("batch_size", 500)
        self._model = CTGANSynthesizer(
            self.metadata,
            epochs=epochs,
            batch_size=batch_size,
        )

    def _fit(self, df: pd.DataFrame) -> None:
        self._model.fit(df)

    def _sample(self, n: int) -> pd.DataFrame:
        return self._model.sample(num_rows=n)

    def get_config(self) -> dict:
        return {
            "epochs": self.kwargs.get("epochs", 300),
            "batch_size": self.kwargs.get("batch_size", 500),
        }


class TVAEGenerator(BaseGenerator):
    """TVAE — Variational Autoencoder model."""

    @property
    def name(self) -> str:
        return "TVAE"

    def _build_model(self) -> None:
        from sdv.single_table import TVAESynthesizer
        epochs = self.kwargs.get("epochs", 300)
        batch_size = self.kwargs.get("batch_size", 500)
        self._model = TVAESynthesizer(
            self.metadata,
            epochs=epochs,
            batch_size=batch_size,
        )

    def _fit(self, df: pd.DataFrame) -> None:
        self._model.fit(df)

    def _sample(self, n: int) -> pd.DataFrame:
        return self._model.sample(num_rows=n)

    def get_config(self) -> dict:
        return {
            "epochs": self.kwargs.get("epochs", 300),
            "batch_size": self.kwargs.get("batch_size", 500),
        }


# =============================================================================
# Planned models — stubs so the interface is established before implementation
# =============================================================================

class PrivBayesGenerator(BaseGenerator):
    """
    PrivBayes — Differentially private Bayesian network.
    Uses the `smartnoise-synth` library.

    Install: pip install smartnoise-synth
    """

    @property
    def name(self) -> str:
        return "PrivBayes"

    def _build_model(self) -> None:
        try:
            from snsynth import Synthesizer
            epsilon = self.kwargs.get("epsilon", 1.0)
            self._model = Synthesizer.create("privbayes", epsilon=epsilon)
            self._epsilon = epsilon
        except ImportError:
            raise ImportError(
                "PrivBayes requires smartnoise-synth: pip install smartnoise-synth"
            )

    def _fit(self, df: pd.DataFrame) -> None:
        self._model.fit(df, preprocessor_eps=0.1)

    def _sample(self, n: int) -> pd.DataFrame:
        return self._model.sample(n)

    def get_config(self) -> dict:
        return {"epsilon": self.kwargs.get("epsilon", 1.0)}


class PATEGANGenerator(BaseGenerator):
    """
    PATE-GAN — Teacher-student GAN with formal DP guarantees.
    Uses the `smartnoise-synth` library.

    Install: pip install smartnoise-synth
    """

    @property
    def name(self) -> str:
        return "PATE-GAN"

    def _build_model(self) -> None:
        try:
            from snsynth import Synthesizer
            epsilon = self.kwargs.get("epsilon", 1.0)
            self._model = Synthesizer.create("pategan", epsilon=epsilon)
            self._epsilon = epsilon
        except ImportError:
            raise ImportError(
                "PATE-GAN requires smartnoise-synth: pip install smartnoise-synth"
            )

    def _fit(self, df: pd.DataFrame) -> None:
        self._model.fit(df)

    def _sample(self, n: int) -> pd.DataFrame:
        return self._model.sample(n)

    def get_config(self) -> dict:
        return {"epsilon": self.kwargs.get("epsilon", 1.0)}


# =============================================================================
# Registry — maps string name to class (used by config-driven main.py)
# =============================================================================

GENERATOR_REGISTRY: dict[str, type[BaseGenerator]] = {
    "copula":    CopulaGenerator,
    "ctgan":     CTGANGenerator,
    "tvae":      TVAEGenerator,
    "privbayes": PrivBayesGenerator,
    "pategan":   PATEGANGenerator,
}


def build_generator(name: str, metadata: Metadata, **kwargs) -> BaseGenerator:
    """
    Factory function — instantiate a generator by name string.

    Usage:
        gen = build_generator("ctgan", metadata, epochs=200)
        gen.fit(df)
        result = gen.sample(1000)
    """
    key = name.lower().replace("-", "").replace("_", "")
    if key not in GENERATOR_REGISTRY:
        raise ValueError(
            f"Unknown generator: '{name}'. "
            f"Available: {list(GENERATOR_REGISTRY.keys())}"
        )
    return GENERATOR_REGISTRY[key](metadata, **kwargs)