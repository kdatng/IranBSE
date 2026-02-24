"""Conformal prediction for distribution-free prediction intervals.

Provides finite-sample coverage guarantees without distributional assumptions.
Implements both split conformal prediction and adaptive conformal inference
(ACI) for time series, which adjusts the miscoverage level online to
maintain long-run coverage under temporal dependence and distribution shift.

References:
    - Vovk, Gammerman, Shafer (2005). Algorithmic Learning in a Random World.
    - Gibbs & Candes (2021). Adaptive Conformal Inference Under Distribution
      Shift.
    - Zaffran et al. (2022). Adaptive Conformal Predictions for Time Series.

Example::

    config = ModelConfig(
        name="conformal_oil",
        params={
            "target_col": "oil_close",
            "coverage": 0.90,
            "adaptation_rate": 0.05,
        },
    )
    predictor = ConformalPredictor(config)
    predictor.calibrate(cal_df, point_model)
    intervals = predictor.predict_interval(test_features, coverage=0.90)
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import polars as pl
from loguru import logger

from src.models.base_model import BaseModel, ModelConfig, PredictionResult


# ---------------------------------------------------------------------------
# Protocol for wrapped point models
# ---------------------------------------------------------------------------


class PointPredictor(Protocol):
    """Protocol for any model that produces point predictions from features."""

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        """Predict from a numpy feature matrix.

        Args:
            X: Feature matrix of shape ``(n, p)``.

        Returns:
            Predictions of shape ``(n,)``.
        """
        ...


# ---------------------------------------------------------------------------
# Non-conformity score functions
# ---------------------------------------------------------------------------


class _AbsoluteResidualScore:
    """Non-conformity score: |y - y_hat|."""

    @staticmethod
    def score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute absolute residual scores.

        Args:
            y_true: Observed values.
            y_pred: Predicted values.

        Returns:
            Non-conformity scores.
        """
        return np.abs(y_true - y_pred)

    @staticmethod
    def interval(
        y_pred: np.ndarray, quantile_score: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct symmetric interval from score quantile.

        Args:
            y_pred: Point predictions.
            quantile_score: Calibrated score threshold.

        Returns:
            (lower, upper) bounds.
        """
        return y_pred - quantile_score, y_pred + quantile_score


class _SignedResidualScore:
    """Non-conformity score preserving asymmetry: separate upper/lower quantiles."""

    @staticmethod
    def score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute signed residuals.

        Args:
            y_true: Observed values.
            y_pred: Predicted values.

        Returns:
            Signed residuals (y_true - y_pred).
        """
        return y_true - y_pred

    @staticmethod
    def interval(
        y_pred: np.ndarray,
        lower_quantile: float,
        upper_quantile: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct asymmetric interval from signed residual quantiles.

        Args:
            y_pred: Point predictions.
            lower_quantile: Lower residual quantile (typically negative).
            upper_quantile: Upper residual quantile (typically positive).

        Returns:
            (lower, upper) bounds.
        """
        return y_pred + lower_quantile, y_pred + upper_quantile


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------


class ConformalPredictor(BaseModel):
    """Distribution-free conformal prediction intervals.

    Wraps any point predictor and produces prediction intervals with
    finite-sample coverage guarantees.  For time series, the adaptive
    conformal inference (ACI) variant adjusts the effective coverage
    online to counteract distribution shift.

    Config params:
        target_col: Target column name (default ``"target"``).
        feature_cols: Feature column list (auto-detected if ``None``).
        coverage: Desired marginal coverage (default 0.90).
        score_type: ``"absolute"`` or ``"signed"`` (default ``"absolute"``).
        adaptation_rate: ACI step-size gamma for online adjustment
            (default 0.05).  Set to 0.0 to disable adaptation.
        symmetry: If ``True``, use symmetric intervals (default ``True``).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._target_col: str = config.params.get("target_col", "target")
        self._feature_cols: list[str] | None = config.params.get("feature_cols")
        self._coverage: float = config.params.get("coverage", 0.90)
        self._score_type: str = config.params.get("score_type", "absolute")
        self._gamma: float = config.params.get("adaptation_rate", 0.05)
        self._symmetric: bool = config.params.get("symmetry", True)

        # Calibration artefacts
        self._cal_scores: np.ndarray | None = None
        self._quantile_score: float | None = None
        self._lower_quantile: float | None = None
        self._upper_quantile: float | None = None
        self._point_model: PointPredictor | None = None
        self._fitted_feature_cols: list[str] = []

        # ACI online state
        self._alpha_t: float = 1.0 - self._coverage

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        data: pl.DataFrame,
        point_model: PointPredictor,
        coverage: float | None = None,
    ) -> None:
        """Calibrate the conformal predictor on held-out calibration data.

        Args:
            data: Calibration DataFrame with features and target.
            point_model: Any model implementing :class:`PointPredictor`.
            coverage: Override the configured coverage level.
        """
        coverage = coverage or self._coverage
        alpha = 1.0 - coverage

        self._point_model = point_model
        self._fitted_feature_cols = self._resolve_feature_cols(data)

        X_cal = data.select(self._fitted_feature_cols).to_numpy().astype(np.float64)
        y_cal = data[self._target_col].to_numpy().astype(np.float64)
        y_hat = point_model.predict_array(X_cal)

        if self._score_type == "absolute":
            scorer = _AbsoluteResidualScore()
            self._cal_scores = scorer.score(y_cal, y_hat)
            # Finite-sample corrected quantile: ceil((n+1)(1-alpha)) / n
            n = len(self._cal_scores)
            q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
            self._quantile_score = float(np.quantile(self._cal_scores, q_level))
            logger.info(
                "Calibrated absolute score quantile: {:.4f} at coverage {:.2%}",
                self._quantile_score,
                coverage,
            )
        else:
            scorer_signed = _SignedResidualScore()
            residuals = scorer_signed.score(y_cal, y_hat)
            n = len(residuals)
            q_lo = max(np.floor((n + 1) * (alpha / 2)) / n, 0.0)
            q_hi = min(np.ceil((n + 1) * (1 - alpha / 2)) / n, 1.0)
            self._lower_quantile = float(np.quantile(residuals, q_lo))
            self._upper_quantile = float(np.quantile(residuals, q_hi))
            self._cal_scores = residuals
            logger.info(
                "Calibrated signed residual quantiles: [{:.4f}, {:.4f}]",
                self._lower_quantile,
                self._upper_quantile,
            )

        self._alpha_t = alpha
        # Use _mark_fitted to set state
        self.state = __import__("src.models.base_model", fromlist=["ModelState"]).ModelState.FITTED
        self._fitted_data_shape = (data.height, data.width)
        logger.info("Conformal predictor calibrated on {} samples", data.height)

    def fit(self, data: pl.DataFrame) -> None:
        """Fit is a no-op; use :meth:`calibrate` instead.

        This method exists only to satisfy the BaseModel interface.
        It raises an informative error directing callers to ``calibrate``.

        Args:
            data: Unused.

        Raises:
            RuntimeError: Always -- callers should use ``calibrate``.
        """
        raise RuntimeError(
            "ConformalPredictor does not use fit(). "
            "Call calibrate(data, point_model) instead."
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        horizon: int,
        n_scenarios: int = 1000,
    ) -> PredictionResult:
        """Return a stub PredictionResult.

        For production use, call :meth:`predict_interval` directly.

        Args:
            horizon: Number of steps.
            n_scenarios: Unused.

        Returns:
            PredictionResult with empty forecasts.
        """
        self._require_fitted()
        return PredictionResult(
            point_forecast=[0.0] * horizon,
            lower_bounds={},
            upper_bounds={},
            metadata={"note": "Use predict_interval() for conformal intervals."},
        )

    def predict_interval(
        self,
        data: pl.DataFrame,
        coverage: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Produce prediction intervals with finite-sample coverage.

        Args:
            data: Feature DataFrame for prediction.
            coverage: Optional override of the coverage level.

        Returns:
            Dictionary with keys ``"lower"``, ``"point"``, ``"upper"``,
            each mapping to an array of shape ``(n,)``.
        """
        self._require_fitted()
        assert self._point_model is not None

        X = data.select(self._fitted_feature_cols).to_numpy().astype(np.float64)
        y_hat = self._point_model.predict_array(X)

        if coverage is not None and coverage != self._coverage:
            # Recalculate quantiles for different coverage
            alpha = 1.0 - coverage
            assert self._cal_scores is not None
            n = len(self._cal_scores)
            if self._score_type == "absolute":
                q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
                q_score = float(np.quantile(self._cal_scores, q_level))
                lower, upper = _AbsoluteResidualScore.interval(y_hat, q_score)
            else:
                q_lo = max(np.floor((n + 1) * (alpha / 2)) / n, 0.0)
                q_hi = min(np.ceil((n + 1) * (1 - alpha / 2)) / n, 1.0)
                lo_q = float(np.quantile(self._cal_scores, q_lo))
                hi_q = float(np.quantile(self._cal_scores, q_hi))
                lower, upper = _SignedResidualScore.interval(y_hat, lo_q, hi_q)
        else:
            if self._score_type == "absolute":
                assert self._quantile_score is not None
                lower, upper = _AbsoluteResidualScore.interval(
                    y_hat, self._quantile_score
                )
            else:
                assert self._lower_quantile is not None
                assert self._upper_quantile is not None
                lower, upper = _SignedResidualScore.interval(
                    y_hat, self._lower_quantile, self._upper_quantile
                )

        return {"lower": lower, "point": y_hat, "upper": upper}

    # ------------------------------------------------------------------
    # Adaptive Conformal Inference (ACI)
    # ------------------------------------------------------------------

    def update_online(
        self, y_true: float, y_lower: float, y_upper: float
    ) -> float:
        """Update the ACI miscoverage level after observing a new data point.

        If the observation falls outside the interval, the effective
        alpha is decreased (wider intervals next time); if inside,
        alpha is increased (narrower intervals).

        Args:
            y_true: Observed value.
            y_lower: Predicted lower bound.
            y_upper: Predicted upper bound.

        Returns:
            Updated miscoverage rate alpha_t.
        """
        err_t = 1.0 if (y_true < y_lower or y_true > y_upper) else 0.0
        target_alpha = 1.0 - self._coverage
        # ACI update: alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)
        self._alpha_t = max(
            0.001, min(0.999, self._alpha_t + self._gamma * (target_alpha - err_t))
        )
        return self._alpha_t

    def get_adaptive_coverage(self) -> float:
        """Return the current adaptive coverage level.

        Returns:
            Effective coverage (1 - alpha_t).
        """
        return 1.0 - self._alpha_t

    # ------------------------------------------------------------------
    # Coverage diagnostics
    # ------------------------------------------------------------------

    def coverage_check(
        self,
        data: pl.DataFrame,
        intervals: dict[str, np.ndarray] | None = None,
    ) -> dict[str, float]:
        """Empirically evaluate coverage on a test set.

        Args:
            data: Test DataFrame with target column and features.
            intervals: Pre-computed intervals.  If ``None``, they are
                computed from *data*.

        Returns:
            Dictionary with ``"empirical_coverage"``, ``"target_coverage"``,
            ``"mean_width"``, and ``"coverage_gap"``.
        """
        self._require_fitted()
        if intervals is None:
            intervals = self.predict_interval(data)

        y_true = data[self._target_col].to_numpy()
        lower = intervals["lower"]
        upper = intervals["upper"]

        covered = ((y_true >= lower) & (y_true <= upper)).astype(float)
        empirical = float(covered.mean())
        width = float((upper - lower).mean())

        result = {
            "empirical_coverage": empirical,
            "target_coverage": self._coverage,
            "coverage_gap": empirical - self._coverage,
            "mean_width": width,
            "n_test": len(y_true),
        }
        logger.info(
            "Coverage check: empirical={:.2%} target={:.2%} gap={:+.2%}",
            result["empirical_coverage"],
            result["target_coverage"],
            result["coverage_gap"],
        )
        return result

    def get_params(self) -> dict[str, Any]:
        """Return calibration configuration and diagnostics."""
        self._require_fitted()
        result: dict[str, Any] = {
            "coverage": self._coverage,
            "score_type": self._score_type,
            "adaptation_rate": self._gamma,
            "current_alpha": self._alpha_t,
        }
        if self._cal_scores is not None:
            result["n_calibration"] = len(self._cal_scores)
        if self._quantile_score is not None:
            result["quantile_score"] = self._quantile_score
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_feature_cols(self, data: pl.DataFrame) -> list[str]:
        """Determine feature columns."""
        if self._feature_cols is not None:
            return self._feature_cols
        numeric_dtypes = {pl.Float32, pl.Float64, pl.Int32, pl.Int64}
        return [
            c
            for c in data.columns
            if c != self._target_col and data[c].dtype in numeric_dtypes
        ]
