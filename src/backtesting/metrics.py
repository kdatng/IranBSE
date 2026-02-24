"""Evaluation metrics for probabilistic commodity futures forecasts.

Provides a comprehensive suite of scoring rules and calibration diagnostics
for evaluating distributional forecasts: CRPS, log score, interval coverage,
calibration (PIT histograms), and directional accuracy.

Typical usage::

    metrics = EvaluationMetrics()
    crps = metrics.crps(actuals, forecast_samples)
    coverage = metrics.interval_coverage(actuals, lower, upper)
    calibration = metrics.calibration_metrics(actuals, forecast_cdf)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from numpy.typing import NDArray
from scipy import stats


@dataclass(frozen=True)
class ScorecardResult:
    """Complete scorecard for a probabilistic forecast evaluation.

    Attributes:
        crps: Continuous Ranked Probability Score (lower = better).
        log_score: Mean log predictive density (higher = better).
        interval_coverage: Fraction of actuals within the prediction
            interval, keyed by nominal coverage level.
        calibration_pvalue: Kolmogorov-Smirnov test p-value for PIT
            uniformity (higher = better calibrated).
        directional_accuracy: Fraction of correct direction predictions.
        rmse: Root Mean Squared Error of point forecasts.
        mae: Mean Absolute Error of point forecasts.
        mape: Mean Absolute Percentage Error (%).
        bias: Mean signed error (positive = over-prediction).
        sharpness: Average width of prediction intervals.
    """

    crps: float
    log_score: float
    interval_coverage: dict[float, float]
    calibration_pvalue: float
    directional_accuracy: float
    rmse: float
    mae: float
    mape: float
    bias: float
    sharpness: float


class EvaluationMetrics:
    """Evaluation metrics for probabilistic forecasts.

    All methods accept numpy arrays and return scalar or structured
    results.  Designed for use with the walk-forward validator and
    conflict backtester.
    """

    def __init__(self) -> None:
        logger.info("EvaluationMetrics initialised")

    def scorecard(
        self,
        actuals: NDArray[np.float64],
        point_forecasts: NDArray[np.float64],
        forecast_samples: NDArray[np.float64] | None = None,
        lower_bounds: dict[float, NDArray[np.float64]] | None = None,
        upper_bounds: dict[float, NDArray[np.float64]] | None = None,
    ) -> ScorecardResult:
        """Compute a full scorecard of all evaluation metrics.

        Args:
            actuals: Realised values, shape ``(n_steps,)``.
            point_forecasts: Point forecast values, shape ``(n_steps,)``.
            forecast_samples: Monte-Carlo forecast samples, shape
                ``(n_samples, n_steps)``.  Required for CRPS and log score.
            lower_bounds: Prediction interval lower bounds keyed by
                nominal coverage level.
            upper_bounds: Prediction interval upper bounds keyed by
                nominal coverage level.

        Returns:
            ScorecardResult with all metrics.
        """
        n = min(len(actuals), len(point_forecasts))
        a = actuals[:n]
        p = point_forecasts[:n]

        # Point forecast metrics
        errors = a - p
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))
        bias = float(np.mean(errors))

        nonzero = np.abs(a) > 1e-10
        mape = (
            float(np.mean(np.abs(errors[nonzero] / a[nonzero]))) * 100
            if nonzero.any()
            else float("nan")
        )

        dir_acc = self.directional_accuracy(a, p)

        # Probabilistic metrics
        crps_val = float("nan")
        log_score_val = float("nan")
        cal_pvalue = float("nan")

        if forecast_samples is not None and forecast_samples.shape[1] >= n:
            samples = forecast_samples[:, :n]
            crps_val = self.crps(a, samples)
            log_score_val = self.log_score(a, samples)
            cal_pvalue = self.calibration_metrics(a, samples)["ks_pvalue"]

        # Interval coverage
        coverage: dict[float, float] = {}
        sharpness_val = 0.0
        if lower_bounds is not None and upper_bounds is not None:
            for level in sorted(
                set(lower_bounds.keys()) & set(upper_bounds.keys())
            ):
                lower = lower_bounds[level][:n]
                # Find corresponding upper bound level
                # Convention: lower 0.05 pairs with upper 0.95 for 90% coverage
                nominal_coverage = 1 - 2 * min(level, 1 - level)
                upper_level = 1 - level
                if upper_level in upper_bounds:
                    upper = upper_bounds[upper_level][:n]
                    coverage[nominal_coverage] = self.interval_coverage(
                        a, lower, upper
                    )

            # Sharpness: average interval width for the widest interval
            if coverage:
                widest_level = max(coverage.keys())
                lower_key = (1 - widest_level) / 2
                upper_key = 1 - lower_key
                if lower_key in lower_bounds and upper_key in upper_bounds:
                    lower_arr = lower_bounds[lower_key][:n]
                    upper_arr = upper_bounds[upper_key][:n]
                    sharpness_val = float(np.mean(upper_arr - lower_arr))

        return ScorecardResult(
            crps=crps_val,
            log_score=log_score_val,
            interval_coverage=coverage,
            calibration_pvalue=cal_pvalue,
            directional_accuracy=dir_acc,
            rmse=rmse,
            mae=mae,
            mape=mape,
            bias=bias,
            sharpness=sharpness_val,
        )

    def crps(
        self,
        actuals: NDArray[np.float64],
        forecast_samples: NDArray[np.float64],
    ) -> float:
        """Compute the Continuous Ranked Probability Score (CRPS).

        CRPS generalises MAE to probabilistic forecasts.  It measures the
        integrated squared difference between the forecast CDF and the
        step-function CDF of the observation.

        Uses the ensemble representation:
        CRPS = E|X - y| - 0.5 * E|X - X'|

        Args:
            actuals: Realised values, shape ``(n_steps,)``.
            forecast_samples: Samples from the forecast distribution,
                shape ``(n_samples, n_steps)``.

        Returns:
            Mean CRPS across all time steps (lower = better).
        """
        n_samples, n_steps = forecast_samples.shape
        n = min(n_steps, len(actuals))

        crps_values = np.zeros(n, dtype=np.float64)

        for t in range(n):
            samples = forecast_samples[:, t]
            y = actuals[t]

            # E|X - y|
            term1 = float(np.mean(np.abs(samples - y)))

            # E|X - X'| via sorted samples (efficient formula)
            sorted_samples = np.sort(samples)
            n_s = len(sorted_samples)
            # E|X - X'| = (2 / n^2) * sum_{i=1}^{n} (2i - n - 1) * x_(i)
            weights = 2 * np.arange(1, n_s + 1) - n_s - 1
            term2 = float(np.sum(weights * sorted_samples)) / (n_s * n_s)

            crps_values[t] = term1 - term2

        return float(np.mean(crps_values))

    def log_score(
        self,
        actuals: NDArray[np.float64],
        forecast_samples: NDArray[np.float64],
        bandwidth: str = "silverman",
    ) -> float:
        """Compute the log predictive density score.

        Uses kernel density estimation on the forecast samples to evaluate
        the predictive density at the observed value.

        Args:
            actuals: Realised values, shape ``(n_steps,)``.
            forecast_samples: Samples from the forecast distribution,
                shape ``(n_samples, n_steps)``.
            bandwidth: KDE bandwidth selection method.

        Returns:
            Mean log score across all time steps (higher = better).
        """
        n_steps = min(forecast_samples.shape[1], len(actuals))
        log_scores = np.zeros(n_steps, dtype=np.float64)

        for t in range(n_steps):
            samples = forecast_samples[:, t]
            y = actuals[t]

            # Remove NaN samples
            valid_samples = samples[~np.isnan(samples)]
            if len(valid_samples) < 5:
                log_scores[t] = -20.0  # Penalty for insufficient samples
                continue

            # KDE to estimate density
            try:
                kde = stats.gaussian_kde(valid_samples, bw_method=bandwidth)
                density = float(kde(y)[0])
                log_scores[t] = float(np.log(max(density, 1e-20)))
            except (np.linalg.LinAlgError, ValueError):
                log_scores[t] = -20.0

        return float(np.mean(log_scores))

    def interval_coverage(
        self,
        actuals: NDArray[np.float64],
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
    ) -> float:
        """Compute empirical coverage of a prediction interval.

        Args:
            actuals: Realised values.
            lower: Lower bound of the interval.
            upper: Upper bound of the interval.

        Returns:
            Fraction of actuals falling within [lower, upper].
        """
        n = min(len(actuals), len(lower), len(upper))
        within = (actuals[:n] >= lower[:n]) & (actuals[:n] <= upper[:n])
        return float(np.mean(within))

    def calibration_metrics(
        self,
        actuals: NDArray[np.float64],
        forecast_samples: NDArray[np.float64],
    ) -> dict[str, Any]:
        """Assess calibration via Probability Integral Transform (PIT).

        A well-calibrated probabilistic forecast produces PIT values that
        are uniformly distributed on [0, 1].

        Args:
            actuals: Realised values, shape ``(n_steps,)``.
            forecast_samples: Forecast samples, shape ``(n_samples, n_steps)``.

        Returns:
            Dictionary with:
            - ``pit_values``: The PIT values.
            - ``ks_statistic``: KS test statistic vs. Uniform[0,1].
            - ``ks_pvalue``: KS test p-value (higher = better calibrated).
            - ``pit_histogram``: Binned histogram counts for visual
              inspection.
        """
        n_steps = min(forecast_samples.shape[1], len(actuals))
        pit_values = np.zeros(n_steps, dtype=np.float64)

        for t in range(n_steps):
            samples = forecast_samples[:, t]
            valid = samples[~np.isnan(samples)]
            if len(valid) == 0:
                pit_values[t] = 0.5
                continue
            # PIT = F(y) = fraction of samples <= actual
            pit_values[t] = float(np.mean(valid <= actuals[t]))

        # KS test against Uniform[0,1]
        ks_stat, ks_pvalue = stats.kstest(pit_values, "uniform")

        # Histogram for visual calibration check
        n_bins = 10
        hist_counts, _ = np.histogram(pit_values, bins=n_bins, range=(0, 1))

        return {
            "pit_values": pit_values.tolist(),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "pit_histogram": hist_counts.tolist(),
            "n_bins": n_bins,
        }

    def directional_accuracy(
        self,
        actuals: NDArray[np.float64],
        predictions: NDArray[np.float64],
    ) -> float:
        """Compute directional accuracy (hit ratio).

        Measures the fraction of time steps where the predicted direction
        of change matches the actual direction.

        Args:
            actuals: Realised values.
            predictions: Point forecasts.

        Returns:
            Directional accuracy in [0, 1].
        """
        n = min(len(actuals), len(predictions))
        if n < 2:
            return 0.0

        actual_dir = np.diff(actuals[:n]) > 0
        pred_dir = np.diff(predictions[:n]) > 0
        return float(np.mean(actual_dir == pred_dir))

    def winkler_score(
        self,
        actuals: NDArray[np.float64],
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
        alpha: float = 0.10,
    ) -> float:
        """Compute the Winkler interval score.

        Rewards narrow intervals that contain the observation and penalises
        intervals that miss.  Combines sharpness and calibration into a
        single score.

        Args:
            actuals: Realised values.
            lower: Lower prediction interval bounds.
            upper: Upper prediction interval bounds.
            alpha: Nominal miscoverage rate (e.g. 0.10 for 90% interval).

        Returns:
            Mean Winkler score (lower = better).
        """
        n = min(len(actuals), len(lower), len(upper))
        scores = np.zeros(n, dtype=np.float64)

        for i in range(n):
            width = upper[i] - lower[i]
            if actuals[i] < lower[i]:
                scores[i] = width + (2 / alpha) * (lower[i] - actuals[i])
            elif actuals[i] > upper[i]:
                scores[i] = width + (2 / alpha) * (actuals[i] - upper[i])
            else:
                scores[i] = width

        return float(np.mean(scores))

    def brier_score(
        self,
        actual_events: NDArray[np.float64],
        predicted_probabilities: NDArray[np.float64],
    ) -> float:
        """Compute the Brier score for binary event forecasts.

        Relevant for scenario probability assessment (e.g., "probability
        of oil > $150 in 30 days").

        Args:
            actual_events: Binary outcomes (0 or 1).
            predicted_probabilities: Predicted probabilities in [0, 1].

        Returns:
            Brier score (lower = better), range [0, 1].
        """
        n = min(len(actual_events), len(predicted_probabilities))
        return float(
            np.mean((predicted_probabilities[:n] - actual_events[:n]) ** 2)
        )

    def reliability_diagram_data(
        self,
        actual_events: NDArray[np.float64],
        predicted_probabilities: NDArray[np.float64],
        n_bins: int = 10,
    ) -> dict[str, list[float]]:
        """Compute data for a reliability (calibration) diagram.

        Args:
            actual_events: Binary outcomes.
            predicted_probabilities: Predicted probabilities.
            n_bins: Number of probability bins.

        Returns:
            Dictionary with ``bin_centers``, ``observed_frequency``,
            ``predicted_mean``, and ``bin_counts``.
        """
        n = min(len(actual_events), len(predicted_probabilities))
        a = actual_events[:n]
        p = predicted_probabilities[:n]

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers: list[float] = []
        observed_freq: list[float] = []
        predicted_mean: list[float] = []
        bin_counts: list[float] = []

        for i in range(n_bins):
            mask = (p >= bin_edges[i]) & (p < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (p >= bin_edges[i]) & (p <= bin_edges[i + 1])

            count = int(mask.sum())
            bin_counts.append(float(count))
            bin_centers.append(float((bin_edges[i] + bin_edges[i + 1]) / 2))

            if count > 0:
                observed_freq.append(float(np.mean(a[mask])))
                predicted_mean.append(float(np.mean(p[mask])))
            else:
                observed_freq.append(float("nan"))
                predicted_mean.append(float("nan"))

        return {
            "bin_centers": bin_centers,
            "observed_frequency": observed_freq,
            "predicted_mean": predicted_mean,
            "bin_counts": bin_counts,
        }

    def to_dataframe(
        self,
        scorecard: ScorecardResult,
    ) -> pl.DataFrame:
        """Convert a scorecard result to a Polars DataFrame.

        Args:
            scorecard: Scorecard to convert.

        Returns:
            Single-row DataFrame with all scalar metrics.
        """
        row: dict[str, Any] = {
            "crps": scorecard.crps,
            "log_score": scorecard.log_score,
            "calibration_pvalue": scorecard.calibration_pvalue,
            "directional_accuracy": scorecard.directional_accuracy,
            "rmse": scorecard.rmse,
            "mae": scorecard.mae,
            "mape": scorecard.mape,
            "bias": scorecard.bias,
            "sharpness": scorecard.sharpness,
        }
        for level, cov in scorecard.interval_coverage.items():
            row[f"coverage_{int(level*100)}pct"] = cov

        return pl.DataFrame([row])
