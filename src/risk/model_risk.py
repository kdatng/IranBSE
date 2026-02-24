"""Model risk analysis: disagreement, parameter sensitivity, and ensemble uncertainty.

Quantifies model risk -- the risk that model outputs are unreliable --
across the IranBSE model ensemble.  When models disagree strongly or
predictions are highly sensitive to parameter choices, the forecast
should be treated with lower confidence.

Typical usage::

    analyzer = ModelRiskAnalyzer(models=[garch, bvar, xgb])
    disagreement = analyzer.model_disagreement(horizon=20)
    sensitivity = analyzer.parameter_sensitivity(base_model, param_grid)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from numpy.typing import NDArray

from src.models.base_model import BaseModel, ModelConfig, PredictionResult


@dataclass(frozen=True)
class DisagreementReport:
    """Summary of model disagreement across the ensemble.

    Attributes:
        mean_spread: Average spread between highest and lowest model
            predictions at each horizon step.
        max_spread: Maximum spread across all horizon steps.
        coefficient_of_variation: CV of predictions across models
            (std / mean) at each horizon step.
        pairwise_correlations: Correlation matrix between model predictions.
        disagreement_trend: Whether disagreement is increasing over the
            forecast horizon (``"increasing"``, ``"decreasing"``,
            ``"stable"``).
        model_outliers: Models whose predictions deviate most from the
            ensemble median.
    """

    mean_spread: float
    max_spread: float
    coefficient_of_variation: list[float]
    pairwise_correlations: dict[str, dict[str, float]]
    disagreement_trend: str
    model_outliers: list[str]


@dataclass(frozen=True)
class SensitivityReport:
    """Parameter sensitivity analysis results.

    Attributes:
        parameter_name: Name of the varied parameter.
        parameter_values: The parameter values tested.
        prediction_range: Max minus min prediction at each horizon step
            across parameter values.
        elasticity: Approximate elasticity (% prediction change per %
            parameter change) at each horizon step.
        most_sensitive_horizon: Horizon step with highest sensitivity.
        stability_score: Overall stability (0 = very sensitive, 1 = stable).
    """

    parameter_name: str
    parameter_values: list[float]
    prediction_range: list[float]
    elasticity: list[float]
    most_sensitive_horizon: int
    stability_score: float


class ModelRiskAnalyzer:
    """Quantifies model risk across the IranBSE forecasting ensemble.

    Analyses three dimensions of model risk:
    1. **Disagreement**: How much do models diverge in their predictions?
    2. **Sensitivity**: How much do predictions change with parameters?
    3. **Uncertainty**: How wide is the ensemble prediction distribution?

    Args:
        models: List of fitted :class:`BaseModel` instances to analyse.
    """

    def __init__(self, models: list[BaseModel]) -> None:
        if not models:
            raise ValueError("ModelRiskAnalyzer requires at least one model.")
        self.models = models
        logger.info(
            "ModelRiskAnalyzer initialised with {} models: {}",
            len(models),
            [m.config.name for m in models],
        )

    def model_disagreement(
        self,
        horizon: int = 20,
        n_scenarios: int = 1000,
    ) -> DisagreementReport:
        """Measure prediction disagreement across the model ensemble.

        Args:
            horizon: Forecast horizon to analyse.
            n_scenarios: Monte-Carlo paths per model.

        Returns:
            DisagreementReport with spread, CV, correlations, and outlier
            identification.
        """
        # Collect point forecasts from all models
        predictions: dict[str, list[float]] = {}
        for model in self.models:
            try:
                result = model.predict(horizon, n_scenarios)
                predictions[model.config.name] = result.point_forecast
            except RuntimeError:
                logger.warning(
                    "Model '{}' not fitted; skipping", model.config.name
                )

        if len(predictions) < 2:
            raise ValueError(
                "Need at least 2 fitted models for disagreement analysis"
            )

        # Stack predictions: shape (n_models, horizon)
        names = list(predictions.keys())
        pred_matrix = np.array(
            [predictions[n] for n in names], dtype=np.float64
        )

        # Spread at each horizon step
        spreads = np.max(pred_matrix, axis=0) - np.min(pred_matrix, axis=0)
        mean_spread = float(np.mean(spreads))
        max_spread = float(np.max(spreads))

        # Coefficient of variation
        means = np.mean(pred_matrix, axis=0)
        stds = np.std(pred_matrix, axis=0)
        cv = np.where(
            np.abs(means) > 1e-10, stds / np.abs(means), 0.0
        ).tolist()

        # Pairwise correlations
        n_models = len(names)
        corr_dict: dict[str, dict[str, float]] = {}
        for i in range(n_models):
            corr_dict[names[i]] = {}
            for j in range(n_models):
                if np.std(pred_matrix[i]) > 1e-10 and np.std(pred_matrix[j]) > 1e-10:
                    corr_dict[names[i]][names[j]] = float(
                        np.corrcoef(pred_matrix[i], pred_matrix[j])[0, 1]
                    )
                else:
                    corr_dict[names[i]][names[j]] = 0.0

        # Disagreement trend
        if horizon > 5:
            first_half = float(np.mean(spreads[: horizon // 2]))
            second_half = float(np.mean(spreads[horizon // 2 :]))
            if second_half > first_half * 1.2:
                trend = "increasing"
            elif second_half < first_half * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Identify outlier models (>1.5 IQR from median)
        median_pred = np.median(pred_matrix, axis=0)
        deviations = np.mean(np.abs(pred_matrix - median_pred), axis=1)
        q1 = np.percentile(deviations, 25)
        q3 = np.percentile(deviations, 75)
        iqr = q3 - q1
        outlier_threshold = q3 + 1.5 * iqr
        outliers = [
            names[i]
            for i in range(n_models)
            if deviations[i] > outlier_threshold
        ]

        report = DisagreementReport(
            mean_spread=mean_spread,
            max_spread=max_spread,
            coefficient_of_variation=cv,
            pairwise_correlations=corr_dict,
            disagreement_trend=trend,
            model_outliers=outliers,
        )

        logger.info(
            "Model disagreement: mean_spread={:.4f}, max_spread={:.4f}, "
            "trend={}, outliers={}",
            mean_spread,
            max_spread,
            trend,
            outliers,
        )
        return report

    def parameter_sensitivity(
        self,
        model: BaseModel,
        param_name: str,
        param_values: list[float],
        data: pl.DataFrame,
        horizon: int = 20,
    ) -> SensitivityReport:
        """Analyse prediction sensitivity to a single parameter.

        Re-fits the model at each parameter value and records the effect
        on point forecasts.

        Args:
            model: The model to analyse.
            param_name: Name of the parameter to vary.
            param_values: List of parameter values to test.
            data: Training data for re-fitting.
            horizon: Forecast horizon.

        Returns:
            SensitivityReport with elasticities and stability score.
        """
        if len(param_values) < 2:
            raise ValueError("Need at least 2 parameter values")

        all_predictions: list[list[float]] = []
        original_params = model.config.params.copy()

        for val in param_values:
            # Update parameter and re-fit
            model.config.params[param_name] = val
            try:
                model.fit(data)
                result = model.predict(horizon)
                all_predictions.append(result.point_forecast)
            except Exception as exc:
                logger.warning(
                    "Sensitivity test failed for {}={}: {}",
                    param_name,
                    val,
                    exc,
                )
                all_predictions.append([np.nan] * horizon)

        # Restore original parameters
        model.config.params = original_params

        pred_matrix = np.array(all_predictions, dtype=np.float64)

        # Prediction range at each horizon step
        pred_range = (
            np.nanmax(pred_matrix, axis=0) - np.nanmin(pred_matrix, axis=0)
        ).tolist()

        # Elasticity: (delta_pred / delta_param) * (param / pred)
        elasticity = np.zeros(horizon, dtype=np.float64)
        if len(param_values) >= 2:
            for h in range(horizon):
                preds_h = pred_matrix[:, h]
                valid = ~np.isnan(preds_h)
                if valid.sum() >= 2:
                    p_vals = np.array(param_values)[valid]
                    p_preds = preds_h[valid]
                    if np.std(p_vals) > 1e-10 and np.abs(np.mean(p_preds)) > 1e-10:
                        # Numerical derivative
                        dp = np.diff(p_preds) / np.diff(p_vals)
                        mid_param = (p_vals[:-1] + p_vals[1:]) / 2
                        mid_pred = (p_preds[:-1] + p_preds[1:]) / 2
                        elast = dp * mid_param / np.where(
                            np.abs(mid_pred) > 1e-10, mid_pred, np.nan
                        )
                        elasticity[h] = float(np.nanmean(elast))

        most_sensitive = int(np.argmax(np.abs(elasticity)))
        max_range = float(np.nanmax(pred_range)) if pred_range else 0.0
        baseline_pred = float(np.nanmean(pred_matrix))
        stability = 1.0 - min(
            max_range / max(abs(baseline_pred), 1e-10), 1.0
        )

        report = SensitivityReport(
            parameter_name=param_name,
            parameter_values=param_values,
            prediction_range=pred_range,
            elasticity=elasticity.tolist(),
            most_sensitive_horizon=most_sensitive,
            stability_score=float(stability),
        )

        logger.info(
            "Sensitivity of '{}' to '{}': stability={:.3f}, "
            "most sensitive horizon={}",
            model.config.name,
            param_name,
            stability,
            most_sensitive,
        )
        return report

    def ensemble_uncertainty(
        self,
        horizon: int = 20,
        n_scenarios: int = 1000,
        confidence_levels: list[float] | None = None,
    ) -> dict[str, Any]:
        """Quantify overall ensemble prediction uncertainty.

        Pools all Monte Carlo scenarios from all models to build a
        super-ensemble distribution, then extracts uncertainty metrics.

        Args:
            horizon: Forecast horizon.
            n_scenarios: Scenarios per model.
            confidence_levels: Quantile levels for uncertainty bands.

        Returns:
            Dictionary with:
            - ``ensemble_mean``: Mean prediction path.
            - ``ensemble_std``: Standard deviation at each step.
            - ``confidence_bands``: Quantile bands.
            - ``entropy``: Prediction entropy (higher = more uncertain).
            - ``model_agreement_index``: 0 (no agreement) to 1 (perfect).
        """
        if confidence_levels is None:
            confidence_levels = [0.05, 0.25, 0.75, 0.95]

        all_scenarios: list[NDArray[np.float64]] = []

        for model in self.models:
            try:
                result = model.predict(horizon, n_scenarios)
                if result.scenarios:
                    for path in result.scenarios.values():
                        arr = np.array(path[:horizon], dtype=np.float64)
                        if len(arr) == horizon:
                            all_scenarios.append(arr)
            except RuntimeError:
                continue

        if not all_scenarios:
            raise RuntimeError("No models produced valid scenarios")

        scenario_matrix = np.array(all_scenarios, dtype=np.float64)
        ensemble_mean = np.mean(scenario_matrix, axis=0).tolist()
        ensemble_std = np.std(scenario_matrix, axis=0).tolist()

        bands: dict[float, list[float]] = {}
        for level in confidence_levels:
            bands[level] = np.quantile(
                scenario_matrix, level, axis=0
            ).tolist()

        # Prediction entropy: use histogram-based entropy at each step
        entropy_per_step = []
        for h in range(horizon):
            vals = scenario_matrix[:, h]
            vals = vals[~np.isnan(vals)]
            if len(vals) < 10:
                entropy_per_step.append(0.0)
                continue
            # Freedman-Diaconis bin width
            iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            bin_width = 2 * iqr * len(vals) ** (-1.0 / 3.0) if iqr > 0 else 1.0
            n_bins = max(int((vals.max() - vals.min()) / max(bin_width, 1e-10)), 5)
            hist, _ = np.histogram(vals, bins=n_bins, density=True)
            hist = hist[hist > 0]
            bin_w = (vals.max() - vals.min()) / n_bins
            probs = hist * bin_w
            probs = probs / probs.sum()
            entropy_per_step.append(float(-np.sum(probs * np.log(probs + 1e-12))))

        # Model agreement index: inverse of normalised inter-model variance
        point_forecasts = []
        for model in self.models:
            try:
                result = model.predict(horizon)
                point_forecasts.append(result.point_forecast)
            except RuntimeError:
                continue

        if len(point_forecasts) >= 2:
            pf_matrix = np.array(point_forecasts, dtype=np.float64)
            inter_model_var = float(np.mean(np.var(pf_matrix, axis=0)))
            total_var = float(np.mean(np.var(scenario_matrix, axis=0)))
            agreement = 1.0 - min(
                inter_model_var / max(total_var, 1e-10), 1.0
            )
        else:
            agreement = 1.0

        return {
            "ensemble_mean": ensemble_mean,
            "ensemble_std": ensemble_std,
            "confidence_bands": bands,
            "entropy": entropy_per_step,
            "mean_entropy": float(np.mean(entropy_per_step)),
            "model_agreement_index": agreement,
            "n_models_contributing": len(point_forecasts),
            "total_scenarios": scenario_matrix.shape[0],
        }

    def model_ranking(
        self,
        horizon: int = 20,
        n_scenarios: int = 1000,
    ) -> pl.DataFrame:
        """Rank models by their alignment with the ensemble consensus.

        Models that agree with the ensemble median are ranked higher
        (lower model risk).  This is useful for identifying which models
        to investigate or down-weight.

        Args:
            horizon: Forecast horizon.
            n_scenarios: Monte-Carlo paths per model.

        Returns:
            Polars DataFrame with model names, deviation scores, and ranks.
        """
        predictions: dict[str, list[float]] = {}
        for model in self.models:
            try:
                result = model.predict(horizon, n_scenarios)
                predictions[model.config.name] = result.point_forecast
            except RuntimeError:
                continue

        if len(predictions) < 2:
            raise ValueError("Need at least 2 fitted models for ranking")

        names = list(predictions.keys())
        pred_matrix = np.array(
            [predictions[n] for n in names], dtype=np.float64
        )
        ensemble_median = np.median(pred_matrix, axis=0)

        # Compute mean absolute deviation from median
        mad = np.mean(np.abs(pred_matrix - ensemble_median), axis=1)

        rows = [
            {
                "model": name,
                "mad_from_median": float(mad[i]),
                "rank": int(rank),
            }
            for i, (name, rank) in enumerate(
                zip(names, np.argsort(mad) + 1, strict=True)
            )
        ]

        return pl.DataFrame(rows).sort("rank")
