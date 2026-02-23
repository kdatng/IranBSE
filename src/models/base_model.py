"""Abstract base model interface for IranBSE commodity futures models.

Provides the foundational contracts that all forecasting models must implement,
ensuring consistent prediction outputs, parameter introspection, and lifecycle
management across statistical, machine-learning, and ensemble model families.

Typical usage::

    class MyModel(BaseModel):
        def fit(self, data: pl.DataFrame) -> None: ...
        def predict(self, horizon: int, n_scenarios: int = 1000) -> PredictionResult: ...
        def get_params(self) -> dict: ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from pydantic import BaseModel as PydanticBaseModel, Field


class ModelState(str, Enum):
    """Lifecycle states of a model instance."""

    INITIALIZED = "initialized"
    FITTED = "fitted"
    FAILED = "failed"


class ModelConfig(PydanticBaseModel):
    """Configuration container for model instantiation.

    Attributes:
        name: Human-readable identifier for the model.
        version: Semantic version string following ``MAJOR.MINOR.PATCH``.
        params: Arbitrary hyper-parameters forwarded to the underlying
            estimator.  Each model subclass documents its own expected keys.
    """

    name: str
    version: str = "0.1.0"
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}


class PredictionResult(PydanticBaseModel):
    """Structured container for multi-horizon, multi-scenario forecasts.

    Attributes:
        point_forecast: Expected value at each horizon step.
        lower_bounds: Mapping of confidence level (e.g. 0.05 for 5 %) to
            the corresponding lower quantile series.
        upper_bounds: Mapping of confidence level to upper quantile series.
        scenarios: Optional named scenario trajectories
            (e.g. ``"war_escalation"``).
        metadata: Auxiliary information such as model name, fit statistics,
            or regime probabilities.
    """

    point_forecast: list[float]
    lower_bounds: dict[float, list[float]]
    upper_bounds: dict[float, list[float]]
    scenarios: dict[str, list[float]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class BaseModel(ABC):
    """Abstract base for all IranBSE forecasting models.

    Subclasses **must** implement :pymethod:`fit`, :pymethod:`predict`, and
    :pymethod:`get_params`.  The base class provides common lifecycle
    bookkeeping (state tracking, input validation) so that downstream
    consumers can rely on a uniform interface.

    Args:
        config: A :class:`ModelConfig` instance carrying the model name,
            version, and hyper-parameter dict.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.state: ModelState = ModelState.INITIALIZED
        self._fitted_data_shape: tuple[int, int] | None = None
        logger.info(
            "Initialized model '{}' v{} with params: {}",
            config.name,
            config.version,
            config.params,
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, data: pl.DataFrame) -> None:
        """Fit the model on historical data.

        Args:
            data: A Polars DataFrame whose columns correspond to the
                commodity / macro series required by the concrete model.

        Raises:
            ValueError: If the data does not satisfy the model's minimum
                length or column requirements.
        """

    @abstractmethod
    def predict(
        self,
        horizon: int,
        n_scenarios: int = 1000,  # JUSTIFIED: 1000 Monte-Carlo paths balances speed vs. convergence for VaR/ES estimation
    ) -> PredictionResult:
        """Generate forecasts over *horizon* future steps.

        Args:
            horizon: Number of forward periods to forecast.
            n_scenarios: Number of Monte-Carlo simulation paths for
                distributional outputs.

        Returns:
            A :class:`PredictionResult` with point forecasts, confidence
            bands, and optional named scenarios.

        Raises:
            RuntimeError: If the model has not been fitted.
        """

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return the currently fitted parameters.

        Returns:
            Dictionary mapping parameter names to their estimated values.

        Raises:
            RuntimeError: If the model has not been fitted.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        """Guard that raises if the model has not been fitted yet."""
        if self.state != ModelState.FITTED:
            raise RuntimeError(
                f"Model '{self.config.name}' must be fitted before calling "
                f"this method (current state: {self.state.value})."
            )

    def _validate_data(
        self,
        data: pl.DataFrame,
        required_columns: list[str] | None = None,
        min_rows: int = 30,  # JUSTIFIED: 30 observations is the minimum for asymptotic normality in CLT-based estimators
    ) -> None:
        """Validate incoming DataFrame dimensions and column presence.

        Args:
            data: Input data to validate.
            required_columns: Column names that must exist in *data*.
            min_rows: Minimum number of rows required.

        Raises:
            ValueError: On missing columns or insufficient rows.
        """
        if data.height < min_rows:
            raise ValueError(
                f"Data has {data.height} rows but model requires at least "
                f"{min_rows}."
            )

        if required_columns:
            missing = set(required_columns) - set(data.columns)
            if missing:
                raise ValueError(
                    f"Data is missing required columns: {sorted(missing)}"
                )

    def _mark_fitted(self, data: pl.DataFrame) -> None:
        """Transition the model to the FITTED state.

        Args:
            data: The training data (used to record shape only).
        """
        self.state = ModelState.FITTED
        self._fitted_data_shape = (data.height, data.width)
        logger.info(
            "Model '{}' fitted on data with shape {}",
            self.config.name,
            self._fitted_data_shape,
        )

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} name={self.config.name!r} "
            f"state={self.state.value}>"
        )
