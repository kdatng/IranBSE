"""Gradient boosting models for commodity futures forecasting.

Implements a unified interface over XGBoost, LightGBM, and CatBoost
backends using the strategy pattern.  Supports quantile regression for
prediction intervals, SHAP-based feature importance, and walk-forward
validation for rigorous out-of-sample evaluation.

Example::

    config = ModelConfig(
        name="gb_oil",
        params={"backend": "xgboost", "n_estimators": 500, "learning_rate": 0.05},
    )
    model = GradientBoostModel(config)
    model.fit(train_df)
    result = model.predict(horizon=20)
    importances = model.feature_importance(train_df)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import polars as pl
from loguru import logger

from src.models.base_model import BaseModel, ModelConfig, PredictionResult

# ---------------------------------------------------------------------------
# Backend strategy hierarchy
# ---------------------------------------------------------------------------

BackendName = Literal["xgboost", "lightgbm", "catboost"]


class _BoostBackend(ABC):
    """Internal strategy interface wrapping a single gradient-boosting library."""

    @abstractmethod
    def build(
        self,
        params: dict[str, Any],
        objective: str,
        quantile: float | None,
    ) -> Any:
        """Construct the underlying estimator.

        Args:
            params: Hyper-parameters forwarded to the library constructor.
            objective: Loss function identifier (e.g. ``"reg:squarederror"``).
            quantile: If not *None*, build a quantile regression estimator
                targeting this quantile level.

        Returns:
            An unfitted estimator object.
        """

    @abstractmethod
    def fit_estimator(
        self,
        estimator: Any,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None,
    ) -> Any:
        """Fit the estimator in-place and return it.

        Args:
            estimator: Estimator built by :meth:`build`.
            X: Training features of shape ``(n, p)``.
            y: Training targets of shape ``(n,)``.
            eval_set: Optional ``(X_val, y_val)`` pair for early stopping.

        Returns:
            The fitted estimator.
        """

    @abstractmethod
    def predict_estimator(self, estimator: Any, X: np.ndarray) -> np.ndarray:
        """Generate predictions from a fitted estimator.

        Args:
            estimator: Fitted estimator.
            X: Feature matrix of shape ``(n, p)``.

        Returns:
            Predictions of shape ``(n,)``.
        """


class _XGBoostBackend(_BoostBackend):
    """XGBoost strategy."""

    def build(
        self,
        params: dict[str, Any],
        objective: str,
        quantile: float | None,
    ) -> Any:
        import xgboost as xgb

        xgb_params: dict[str, Any] = {
            "n_estimators": params.get("n_estimators", 500),
            "learning_rate": params.get("learning_rate", 0.05),
            "max_depth": params.get("max_depth", 6),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.8),
            "verbosity": 0,
            "n_jobs": params.get("n_jobs", -1),
            "random_state": params.get("random_state", 42),
        }
        if quantile is not None:
            xgb_params["objective"] = "reg:quantileerror"
            xgb_params["quantile_alpha"] = quantile
        else:
            xgb_params["objective"] = objective
        return xgb.XGBRegressor(**xgb_params)

    def fit_estimator(
        self,
        estimator: Any,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None,
    ) -> Any:
        fit_kwargs: dict[str, Any] = {"verbose": False}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
        estimator.fit(X, y, **fit_kwargs)
        return estimator

    def predict_estimator(self, estimator: Any, X: np.ndarray) -> np.ndarray:
        return estimator.predict(X)  # type: ignore[no-any-return]


class _LightGBMBackend(_BoostBackend):
    """LightGBM strategy."""

    def build(
        self,
        params: dict[str, Any],
        objective: str,
        quantile: float | None,
    ) -> Any:
        import lightgbm as lgb

        lgb_params: dict[str, Any] = {
            "n_estimators": params.get("n_estimators", 500),
            "learning_rate": params.get("learning_rate", 0.05),
            "max_depth": params.get("max_depth", -1),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.8),
            "verbosity": -1,
            "n_jobs": params.get("n_jobs", -1),
            "random_state": params.get("random_state", 42),
        }
        if quantile is not None:
            lgb_params["objective"] = "quantile"
            lgb_params["alpha"] = quantile
        else:
            lgb_params["objective"] = objective
        return lgb.LGBMRegressor(**lgb_params)

    def fit_estimator(
        self,
        estimator: Any,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None,
    ) -> Any:
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
        estimator.fit(X, y, **fit_kwargs)
        return estimator

    def predict_estimator(self, estimator: Any, X: np.ndarray) -> np.ndarray:
        return estimator.predict(X)  # type: ignore[no-any-return]


class _CatBoostBackend(_BoostBackend):
    """CatBoost strategy."""

    def build(
        self,
        params: dict[str, Any],
        objective: str,
        quantile: float | None,
    ) -> Any:
        from catboost import CatBoostRegressor

        cb_params: dict[str, Any] = {
            "iterations": params.get("n_estimators", 500),
            "learning_rate": params.get("learning_rate", 0.05),
            "depth": params.get("max_depth", 6),
            "subsample": params.get("subsample", 0.8),
            "verbose": 0,
            "random_seed": params.get("random_state", 42),
            "thread_count": params.get("n_jobs", -1),
        }
        if quantile is not None:
            cb_params["loss_function"] = f"Quantile:alpha={quantile}"
        else:
            cb_params["loss_function"] = "RMSE"
        return CatBoostRegressor(**cb_params)

    def fit_estimator(
        self,
        estimator: Any,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None,
    ) -> Any:
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
        estimator.fit(X, y, **fit_kwargs)
        return estimator

    def predict_estimator(self, estimator: Any, X: np.ndarray) -> np.ndarray:
        return estimator.predict(X)  # type: ignore[no-any-return]


_BACKENDS: dict[BackendName, type[_BoostBackend]] = {
    "xgboost": _XGBoostBackend,
    "lightgbm": _LightGBMBackend,
    "catboost": _CatBoostBackend,
}


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------


class GradientBoostModel(BaseModel):
    """Gradient boosting regressor with quantile prediction intervals.

    Wraps XGBoost, LightGBM, or CatBoost behind a common interface and adds
    SHAP-based feature importance and walk-forward validation.

    Config params:
        backend: One of ``"xgboost"``, ``"lightgbm"``, ``"catboost"``.
        n_estimators: Number of boosting rounds (default 500).
        learning_rate: Step-size shrinkage (default 0.05).
        max_depth: Maximum tree depth (default 6).
        quantiles: Sequence of quantile levels for prediction intervals
            (default ``(0.05, 0.50, 0.95)``).
        target_col: Name of the target column in the DataFrame.
        feature_cols: Explicit list of feature columns.  If ``None``,
            all numeric columns except *target_col* are used.
    """

    # Default quantile levels for 90 % prediction interval
    _DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.50, 0.95)

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        backend_name: BackendName = config.params.get("backend", "xgboost")
        if backend_name not in _BACKENDS:
            raise ValueError(
                f"Unknown backend '{backend_name}'. "
                f"Choose from {list(_BACKENDS.keys())}."
            )
        self._backend: _BoostBackend = _BACKENDS[backend_name]()
        self._backend_name: BackendName = backend_name

        self._quantiles: tuple[float, ...] = tuple(
            config.params.get("quantiles", self._DEFAULT_QUANTILES)
        )
        self._target_col: str = config.params.get("target_col", "target")
        self._feature_cols: list[str] | None = config.params.get("feature_cols")

        # Fitted artefacts
        self._point_estimator: Any | None = None
        self._quantile_estimators: dict[float, Any] = {}
        self._fitted_feature_cols: list[str] = []

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fit(self, data: pl.DataFrame) -> None:
        """Fit point and quantile estimators on *data*.

        Args:
            data: Training DataFrame with feature columns and a target column.

        Raises:
            ValueError: If the target column is missing or data is too short.
        """
        self._validate_data(data, required_columns=[self._target_col], min_rows=50)
        feature_cols = self._resolve_feature_cols(data)
        self._fitted_feature_cols = feature_cols

        X = data.select(feature_cols).to_numpy().astype(np.float64)
        y = data[self._target_col].to_numpy().astype(np.float64)

        logger.info(
            "Fitting {} point estimator on ({}, {}) matrix",
            self._backend_name,
            *X.shape,
        )
        self._point_estimator = self._backend.build(
            self.config.params, objective="reg:squarederror", quantile=None
        )
        self._point_estimator = self._backend.fit_estimator(
            self._point_estimator, X, y, eval_set=None
        )

        # Quantile estimators
        self._quantile_estimators.clear()
        for q in self._quantiles:
            logger.debug("Fitting quantile estimator q={:.2f}", q)
            est = self._backend.build(self.config.params, objective="", quantile=q)
            self._quantile_estimators[q] = self._backend.fit_estimator(
                est, X, y, eval_set=None
            )

        self._mark_fitted(data)

    def predict(
        self,
        horizon: int,
        n_scenarios: int = 1000,
    ) -> PredictionResult:
        """Produce point and quantile forecasts.

        For tree-based models the *horizon* is interpreted as the number of
        pre-constructed feature rows in the prediction DataFrame passed via
        :meth:`predict_from_frame`.  This thin wrapper returns a single-step
        placeholder; prefer :meth:`predict_from_frame` for multi-step use.

        Args:
            horizon: Number of forecast steps.
            n_scenarios: Unused (kept for interface compatibility).

        Returns:
            A :class:`PredictionResult` with quantile-based bounds.
        """
        self._require_fitted()
        # Single-step stub -- real usage goes through predict_from_frame
        dummy = np.zeros((horizon, len(self._fitted_feature_cols)))
        return self._predict_arrays(dummy)

    def predict_from_frame(self, data: pl.DataFrame) -> PredictionResult:
        """Generate forecasts from a feature DataFrame.

        This is the primary prediction entry-point for walk-forward
        evaluation where the caller constructs feature rows for each
        future step.

        Args:
            data: DataFrame containing the same feature columns as training.

        Returns:
            :class:`PredictionResult` with point forecast and quantile bounds.
        """
        self._require_fitted()
        X = data.select(self._fitted_feature_cols).to_numpy().astype(np.float64)
        return self._predict_arrays(X)

    def get_params(self) -> dict[str, Any]:
        """Return model hyper-parameters and fitted metadata.

        Returns:
            Dictionary with backend name, quantile levels, feature columns,
            and the underlying estimator parameters.
        """
        self._require_fitted()
        return {
            "backend": self._backend_name,
            "quantiles": list(self._quantiles),
            "feature_cols": self._fitted_feature_cols,
            "n_features": len(self._fitted_feature_cols),
            "config_params": dict(self.config.params),
        }

    # ------------------------------------------------------------------
    # SHAP feature importance
    # ------------------------------------------------------------------

    def feature_importance(
        self,
        data: pl.DataFrame,
        method: Literal["shap", "gain"] = "shap",
        max_samples: int = 500,
    ) -> pl.DataFrame:
        """Compute feature importance scores.

        Args:
            data: DataFrame used to compute SHAP values (a background
                sample is drawn if ``data.height > max_samples``).
            method: ``"shap"`` for SHAP TreeExplainer, ``"gain"`` for
                the built-in split-gain importance.
            max_samples: Maximum rows to use for SHAP computation.

        Returns:
            A Polars DataFrame with columns ``["feature", "importance"]``
            sorted descending by importance.
        """
        self._require_fitted()
        feature_cols = self._fitted_feature_cols
        X = data.select(feature_cols).to_numpy().astype(np.float64)

        if method == "gain":
            return self._gain_importance(feature_cols)

        # SHAP path
        import shap

        if X.shape[0] > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(X.shape[0], size=max_samples, replace=False)
            X = X[idx]

        explainer = shap.TreeExplainer(self._point_estimator)
        shap_values = explainer.shap_values(X)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)  # type: ignore[union-attr]

        return pl.DataFrame(
            {"feature": feature_cols, "importance": mean_abs_shap.tolist()}
        ).sort("importance", descending=True)

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------

    def walk_forward_validate(
        self,
        data: pl.DataFrame,
        initial_train_size: int,
        step_size: int = 1,
        refit_every: int = 1,
    ) -> pl.DataFrame:
        """Expanding-window walk-forward validation.

        Args:
            data: Full dataset ordered chronologically.
            initial_train_size: Number of rows in the first training window.
            step_size: Number of rows to advance the test window each step.
            refit_every: Refit the model every *refit_every* steps.

        Returns:
            DataFrame with columns ``["step", "actual", "predicted",
            "lower_05", "upper_95"]``.
        """
        self._validate_data(data, required_columns=[self._target_col], min_rows=50)
        feature_cols = self._resolve_feature_cols(data)

        records: list[dict[str, float]] = []
        n = data.height
        steps_since_fit = 0

        for start in range(initial_train_size, n, step_size):
            end = min(start + step_size, n)
            train = data.slice(0, start)
            test = data.slice(start, end - start)

            if steps_since_fit % refit_every == 0:
                self.fit(train)
            steps_since_fit += 1

            X_test = test.select(feature_cols).to_numpy().astype(np.float64)
            y_test = test[self._target_col].to_numpy().astype(np.float64)

            preds = self._backend.predict_estimator(self._point_estimator, X_test)
            lower = self._backend.predict_estimator(
                self._quantile_estimators[self._quantiles[0]], X_test
            )
            upper = self._backend.predict_estimator(
                self._quantile_estimators[self._quantiles[-1]], X_test
            )

            for i in range(len(y_test)):
                records.append(
                    {
                        "step": float(start + i),
                        "actual": float(y_test[i]),
                        "predicted": float(preds[i]),
                        "lower_05": float(lower[i]),
                        "upper_95": float(upper[i]),
                    }
                )

        logger.info("Walk-forward validation completed: {} steps", len(records))
        return pl.DataFrame(records)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_feature_cols(self, data: pl.DataFrame) -> list[str]:
        """Determine feature columns from data or config."""
        if self._feature_cols is not None:
            return self._feature_cols
        numeric_dtypes = {pl.Float32, pl.Float64, pl.Int32, pl.Int64}
        return [
            c
            for c in data.columns
            if c != self._target_col and data[c].dtype in numeric_dtypes
        ]

    def _predict_arrays(self, X: np.ndarray) -> PredictionResult:
        """Build a PredictionResult from raw numpy arrays."""
        point = self._backend.predict_estimator(self._point_estimator, X)

        lower_bounds: dict[float, list[float]] = {}
        upper_bounds: dict[float, list[float]] = {}

        sorted_q = sorted(self._quantiles)
        median_q = 0.50
        for q in sorted_q:
            qpred = self._backend.predict_estimator(
                self._quantile_estimators[q], X
            ).tolist()
            if q < median_q:
                lower_bounds[q] = qpred
            elif q > median_q:
                upper_bounds[q] = qpred
            # median quantile is used as point forecast backup

        return PredictionResult(
            point_forecast=point.tolist(),
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            metadata={
                "model": self.config.name,
                "backend": self._backend_name,
                "quantiles": list(self._quantiles),
            },
        )

    def _gain_importance(self, feature_cols: list[str]) -> pl.DataFrame:
        """Fall back to split-gain importance without SHAP."""
        try:
            imp = self._point_estimator.feature_importances_
        except AttributeError:
            imp = np.ones(len(feature_cols)) / len(feature_cols)
        return pl.DataFrame(
            {"feature": feature_cols, "importance": imp.tolist()}
        ).sort("importance", descending=True)
