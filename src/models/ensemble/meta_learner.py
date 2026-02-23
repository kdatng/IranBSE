"""Stacking / blending meta-learner ensemble for IranBSE commodity futures.

Combines predictions from heterogeneous base models using Ridge regression,
regime-conditional weighting, and optional Bayesian model averaging to produce
calibrated probabilistic forecasts that adapt to shifting market regimes
(e.g., contango vs. backwardation, geopolitical escalation vs. calm).

Typical usage::

    meta = MetaLearner(config, base_models=[garch_model, bvar_model, xgb_model])
    meta.fit(historical_data)
    result = meta.predict(horizon=20, n_scenarios=2000)
    weights = meta.get_model_weights()
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel as PydanticBaseModel, Field
from scipy import linalg

from src.models.base_model import BaseModel, ModelConfig, PredictionResult


class MarketRegime(str, Enum):
    """Discrete market regime labels inferred from observable features."""

    CALM = "calm"
    ELEVATED = "elevated"
    CRISIS = "crisis"


class RegimeWeights(PydanticBaseModel):
    """Per-regime meta-learner weight vectors.

    Attributes:
        regime: The market regime label.
        weights: Model weights for this regime, ordered by ``base_models``.
        intercept: Regime-specific intercept term.
    """

    regime: MarketRegime
    weights: list[float]
    intercept: float = 0.0

    model_config = {"frozen": True}


class MetaLearner(BaseModel):
    """Stacking / blending ensemble with regime-conditional weights.

    The meta-learner collects out-of-sample predictions from each base model,
    then fits a Ridge regression in each detected regime to learn optimal
    combination weights.  An optional Bayesian model averaging (BMA) mode
    replaces Ridge with posterior-probability weights derived from marginal
    likelihoods.

    Args:
        config: Standard ``ModelConfig`` for the ensemble.
        base_models: List of fitted :class:`BaseModel` instances whose
            predictions are stacked.
        alpha: Ridge regularisation strength.  Higher values shrink weights
            toward equal weighting. Defaults to 1.0 as a moderate
            regularisation that prevents overfitting while allowing
            meaningful differentiation between models.
        use_bma: If True, use Bayesian model averaging instead of Ridge.
        regime_column: Column name in training data that contains the regime
            label (or ``None`` to auto-detect from volatility quantiles).
    """

    def __init__(
        self,
        config: ModelConfig,
        base_models: list[BaseModel],
        alpha: float = 1.0,
        use_bma: bool = False,
        regime_column: str | None = None,
    ) -> None:
        super().__init__(config)
        if not base_models:
            raise ValueError("MetaLearner requires at least one base model.")
        self.base_models = base_models
        self.alpha = alpha
        self.use_bma = use_bma
        self.regime_column = regime_column

        # Fitted artefacts
        self._regime_weights: dict[MarketRegime, RegimeWeights] = {}
        self._global_weights: NDArray[np.float64] | None = None
        self._global_intercept: float = 0.0
        self._regime_thresholds: tuple[float, float] = (0.0, 0.0)
        self._bma_log_marginal: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, data: pl.DataFrame) -> None:
        """Fit meta-learner weights from base-model out-of-sample predictions.

        If ``regime_column`` is present in *data* (or auto-detected), separate
        Ridge regressions are fitted per regime.  Otherwise, a single global
        weight vector is estimated.

        Args:
            data: Historical DataFrame with a ``target`` column and columns
                for each base model prediction (named ``pred_{model.config.name}``).
                Optionally includes a regime indicator column.

        Raises:
            ValueError: If required columns are missing or data is too short.
        """
        pred_cols = [f"pred_{m.config.name}" for m in self.base_models]
        required = ["target"] + pred_cols
        self._validate_data(data, required_columns=required)

        target: NDArray[np.float64] = data["target"].to_numpy().astype(np.float64)
        preds: NDArray[np.float64] = (
            data.select(pred_cols).to_numpy().astype(np.float64)
        )

        # --- Determine regimes ------------------------------------------------
        regimes = self._resolve_regimes(data, target)

        if self.use_bma:
            self._fit_bma(preds, target, regimes)
        else:
            self._fit_ridge(preds, target, regimes)

        self._mark_fitted(data)
        logger.info(
            "MetaLearner fitted with {} base models across {} regimes",
            len(self.base_models),
            len(self._regime_weights) or 1,
        )

    def predict(
        self,
        horizon: int,
        n_scenarios: int = 1000,
    ) -> PredictionResult:
        """Generate blended predictions from base models.

        Each base model produces ``n_scenarios`` forward paths.  The
        meta-learner applies regime-conditional weights to blend these
        paths into a combined forecast distribution.

        Args:
            horizon: Number of forward time steps.
            n_scenarios: Monte-Carlo paths per base model.

        Returns:
            PredictionResult with blended point forecast, confidence bands,
            and scenario paths.

        Raises:
            RuntimeError: If the meta-learner has not been fitted.
        """
        self._require_fitted()

        base_results: list[PredictionResult] = []
        for model in self.base_models:
            base_results.append(model.predict(horizon, n_scenarios))

        # Stack point forecasts: shape (n_models, horizon)
        point_stack = np.array(
            [r.point_forecast for r in base_results], dtype=np.float64
        )

        # Stack scenario matrices: shape (n_models, n_scenarios, horizon)
        scenario_stack = np.zeros(
            (len(self.base_models), n_scenarios, horizon), dtype=np.float64
        )
        for i, result in enumerate(base_results):
            if result.scenarios:
                for j, key in enumerate(sorted(result.scenarios)):
                    if j < n_scenarios:
                        scenario_stack[i, j, :] = result.scenarios[key][:horizon]

        # Apply weights (use global if no regime info at prediction time)
        weights = self._get_active_weights()
        blended_point = (weights @ point_stack + self._global_intercept).tolist()

        # Blend scenarios
        blended_scenarios = np.tensordot(weights, scenario_stack, axes=([0], [0]))
        blended_scenarios += self._global_intercept

        # Compute quantile bands from blended scenarios
        quantiles_lower = {
            0.05: np.quantile(blended_scenarios, 0.05, axis=0).tolist(),
            0.10: np.quantile(blended_scenarios, 0.10, axis=0).tolist(),
            0.25: np.quantile(blended_scenarios, 0.25, axis=0).tolist(),
        }
        quantiles_upper = {
            0.75: np.quantile(blended_scenarios, 0.75, axis=0).tolist(),
            0.90: np.quantile(blended_scenarios, 0.90, axis=0).tolist(),
            0.95: np.quantile(blended_scenarios, 0.95, axis=0).tolist(),
        }

        return PredictionResult(
            point_forecast=blended_point,
            lower_bounds=quantiles_lower,
            upper_bounds=quantiles_upper,
            scenarios={
                f"path_{i}": blended_scenarios[i].tolist()
                for i in range(min(n_scenarios, blended_scenarios.shape[0]))
            },
            metadata={
                "model": self.config.name,
                "weights": weights.tolist(),
                "regime_weights": {
                    k.value: v.weights for k, v in self._regime_weights.items()
                },
                "use_bma": self.use_bma,
            },
        )

    def get_params(self) -> dict[str, Any]:
        """Return fitted meta-learner parameters.

        Returns:
            Dictionary containing global weights, per-regime weights,
            BMA marginal likelihoods (if applicable), and regularisation alpha.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._require_fitted()
        params: dict[str, Any] = {
            "alpha": self.alpha,
            "use_bma": self.use_bma,
            "n_base_models": len(self.base_models),
            "global_weights": (
                self._global_weights.tolist()
                if self._global_weights is not None
                else None
            ),
            "global_intercept": self._global_intercept,
        }
        if self._regime_weights:
            params["regime_weights"] = {
                k.value: {"weights": v.weights, "intercept": v.intercept}
                for k, v in self._regime_weights.items()
            }
        if self._bma_log_marginal is not None:
            params["bma_log_marginal_likelihoods"] = self._bma_log_marginal.tolist()
        return params

    def get_model_weights(self) -> dict[str, float]:
        """Return a human-readable mapping of model name to global weight.

        Returns:
            Dictionary mapping each base model's name to its blending weight.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._require_fitted()
        weights = self._get_active_weights()
        return {
            m.config.name: float(w)
            for m, w in zip(self.base_models, weights, strict=True)
        }

    def regime_weights(self, regime: MarketRegime) -> RegimeWeights:
        """Return the weight vector for a specific regime.

        Args:
            regime: The market regime to query.

        Returns:
            ``RegimeWeights`` for the requested regime.

        Raises:
            RuntimeError: If the model has not been fitted.
            KeyError: If the requested regime was not observed during training.
        """
        self._require_fitted()
        if regime not in self._regime_weights:
            available = [r.value for r in self._regime_weights]
            raise KeyError(
                f"Regime '{regime.value}' not found. "
                f"Available: {available}"
            )
        return self._regime_weights[regime]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_regimes(
        self,
        data: pl.DataFrame,
        target: NDArray[np.float64],
    ) -> NDArray[np.int32]:
        """Map each observation to a regime index (0=calm, 1=elevated, 2=crisis).

        If ``regime_column`` exists in *data*, use it directly; otherwise
        partition by rolling volatility terciles.

        Args:
            data: Training DataFrame.
            target: Target array (used for volatility-based auto-detection).

        Returns:
            Integer array of regime indices, same length as *target*.
        """
        if self.regime_column and self.regime_column in data.columns:
            mapping = {r.value: i for i, r in enumerate(MarketRegime)}
            return (
                data[self.regime_column]
                .map_elements(lambda x: mapping.get(x, 0), return_dtype=pl.Int32)
                .to_numpy()
                .astype(np.int32)
            )

        # Auto-detect from rolling volatility (21-day window ~ 1 trading month)
        window = min(21, len(target) // 3)  # 21-day window is standard for monthly vol
        rolling_vol = np.array(
            [
                np.std(target[max(0, i - window) : i + 1])
                for i in range(len(target))
            ]
        )
        q33, q66 = np.quantile(rolling_vol, [0.333, 0.667])
        self._regime_thresholds = (float(q33), float(q66))

        regimes = np.zeros(len(target), dtype=np.int32)
        regimes[rolling_vol > q33] = 1
        regimes[rolling_vol > q66] = 2

        logger.debug(
            "Auto-detected regime thresholds: calm<{:.4f}, elevated<{:.4f}, crisis>={:.4f}",
            q33,
            q66,
            q66,
        )
        return regimes

    def _fit_ridge(
        self,
        preds: NDArray[np.float64],
        target: NDArray[np.float64],
        regimes: NDArray[np.int32],
    ) -> None:
        """Fit Ridge regression meta-learner per regime.

        Uses the closed-form solution (X'X + alpha*I)^-1 X'y for numerical
        stability and speed.

        Args:
            preds: Base model predictions, shape ``(n_obs, n_models)``.
            target: Realised target values, shape ``(n_obs,)``.
            regimes: Regime index per observation.
        """
        n_models = preds.shape[1]

        # Global fit (fallback)
        X_aug = np.column_stack([preds, np.ones(len(target))])
        A = X_aug.T @ X_aug + self.alpha * np.eye(n_models + 1)
        b = X_aug.T @ target
        solution = linalg.solve(A, b, assume_a="pos")
        self._global_weights = solution[:n_models]
        self._global_intercept = float(solution[n_models])

        # Per-regime fit
        for regime_idx, regime_label in enumerate(MarketRegime):
            mask = regimes == regime_idx
            if mask.sum() < max(n_models + 1, 10):
                logger.warning(
                    "Regime '{}' has only {} observations; using global weights",
                    regime_label.value,
                    int(mask.sum()),
                )
                self._regime_weights[regime_label] = RegimeWeights(
                    regime=regime_label,
                    weights=self._global_weights.tolist(),
                    intercept=self._global_intercept,
                )
                continue

            X_r = np.column_stack([preds[mask], np.ones(int(mask.sum()))])
            A_r = X_r.T @ X_r + self.alpha * np.eye(n_models + 1)
            b_r = X_r.T @ target[mask]
            sol_r = linalg.solve(A_r, b_r, assume_a="pos")

            self._regime_weights[regime_label] = RegimeWeights(
                regime=regime_label,
                weights=sol_r[:n_models].tolist(),
                intercept=float(sol_r[n_models]),
            )

    def _fit_bma(
        self,
        preds: NDArray[np.float64],
        target: NDArray[np.float64],
        regimes: NDArray[np.int32],
    ) -> None:
        """Fit Bayesian Model Averaging weights.

        Computes approximate marginal likelihoods under a Gaussian likelihood
        for each base model, then converts to posterior model probabilities
        using BIC approximation.

        Args:
            preds: Base model predictions, shape ``(n_obs, n_models)``.
            target: Realised target values, shape ``(n_obs,)``.
            regimes: Regime index per observation (used for regime-conditional BMA).
        """
        n_obs, n_models = preds.shape
        log_marginals = np.zeros(n_models, dtype=np.float64)

        for j in range(n_models):
            residuals = target - preds[:, j]
            sse = float(np.sum(residuals**2))
            # BIC-based marginal likelihood approximation:
            # -0.5 * n * ln(SSE/n) - 0.5 * k * ln(n)
            sigma2 = sse / n_obs
            log_marginals[j] = -0.5 * n_obs * np.log(max(sigma2, 1e-12)) - 0.5 * np.log(n_obs)

        self._bma_log_marginal = log_marginals

        # Convert to normalised probabilities via log-sum-exp
        max_lm = np.max(log_marginals)
        log_probs = log_marginals - max_lm
        probs = np.exp(log_probs)
        probs /= probs.sum()

        self._global_weights = probs
        self._global_intercept = 0.0

        # Per-regime BMA
        for regime_idx, regime_label in enumerate(MarketRegime):
            mask = regimes == regime_idx
            if mask.sum() < 10:
                self._regime_weights[regime_label] = RegimeWeights(
                    regime=regime_label,
                    weights=probs.tolist(),
                    intercept=0.0,
                )
                continue

            regime_log_m = np.zeros(n_models, dtype=np.float64)
            for j in range(n_models):
                residuals = target[mask] - preds[mask, j]
                sse = float(np.sum(residuals**2))
                n_r = int(mask.sum())
                sigma2 = sse / n_r
                regime_log_m[j] = (
                    -0.5 * n_r * np.log(max(sigma2, 1e-12))
                    - 0.5 * np.log(n_r)
                )

            max_r = np.max(regime_log_m)
            r_probs = np.exp(regime_log_m - max_r)
            r_probs /= r_probs.sum()

            self._regime_weights[regime_label] = RegimeWeights(
                regime=regime_label,
                weights=r_probs.tolist(),
                intercept=0.0,
            )

        logger.info(
            "BMA posterior weights: {}",
            {m.config.name: f"{w:.4f}" for m, w in zip(self.base_models, probs)},
        )

    def _get_active_weights(self) -> NDArray[np.float64]:
        """Return the currently active global weight vector.

        Returns:
            Weight array of shape ``(n_models,)``.
        """
        if self._global_weights is not None:
            return self._global_weights
        # Fallback: equal weighting
        n = len(self.base_models)
        return np.ones(n, dtype=np.float64) / n
