"""Walk-forward validation for time-series models.

Implements expanding and rolling window walk-forward cross-validation
tailored for commodity futures forecasting.  Avoids look-ahead bias by
strictly partitioning data into train/test splits that respect temporal
ordering, with optional recalibration at each step.

Typical usage::

    validator = WalkForwardValidator(
        initial_train_size=252,
        test_size=21,
        step_size=21,
    )
    results = validator.run(model, data, horizon=21)
    summary = validator.summary()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
import polars as pl
from loguru import logger
from numpy.typing import NDArray

from src.models.base_model import BaseModel, PredictionResult


class WindowType(str, Enum):
    """Walk-forward window strategy."""

    EXPANDING = "expanding"
    ROLLING = "rolling"


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation.

    Attributes:
        initial_train_size: Minimum number of observations for the first
            training window. 252 trading days provides a full year of data.
        test_size: Number of observations in each test fold. 21 trading
            days (~1 month) is a natural recalibration frequency.
        step_size: Number of observations to advance between folds. Equal
            to ``test_size`` by default for non-overlapping folds.
        window_type: ``EXPANDING`` (growing train window) or ``ROLLING``
            (fixed-size sliding window).
        max_train_size: Maximum training window for rolling mode.
            ``None`` means no limit for expanding mode.
        recalibrate: Whether to refit the model at each fold.
        embargo_size: Number of observations to skip between train and
            test to prevent information leakage from lagged features.
            5 days is conservative for daily data with weekly lags.
    """

    initial_train_size: int = 252
    test_size: int = 21
    step_size: int = 21
    window_type: WindowType = WindowType.EXPANDING
    max_train_size: int | None = None
    recalibrate: bool = True
    embargo_size: int = 5


@dataclass
class FoldResult:
    """Results from a single walk-forward fold.

    Attributes:
        fold_index: Zero-based fold number.
        train_start: Row index of training start.
        train_end: Row index of training end (exclusive).
        test_start: Row index of test start.
        test_end: Row index of test end (exclusive).
        predictions: Model predictions for the test period.
        actuals: Realised values for the test period.
        metrics: Evaluation metrics for this fold.
    """

    fold_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    predictions: list[float]
    actuals: list[float]
    metrics: dict[str, float] = field(default_factory=dict)


class WalkForwardValidator:
    """Walk-forward cross-validation engine.

    Generates temporal train/test splits, fits the model on each training
    window, produces forecasts for the test window, and collects evaluation
    metrics across all folds.

    Args:
        config: Walk-forward configuration parameters.
    """

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()
        self._fold_results: list[FoldResult] = []
        logger.info(
            "WalkForwardValidator initialised: window={}, train={}, test={}, "
            "step={}, recalibrate={}",
            self.config.window_type.value,
            self.config.initial_train_size,
            self.config.test_size,
            self.config.step_size,
            self.config.recalibrate,
        )

    @property
    def fold_results(self) -> list[FoldResult]:
        """Access individual fold results after a run.

        Returns:
            List of :class:`FoldResult` objects.
        """
        return self._fold_results

    def run(
        self,
        model: BaseModel,
        data: pl.DataFrame,
        target_col: str = "target",
        horizon: int | None = None,
        n_scenarios: int = 1000,
        metric_fn: Callable[
            [NDArray[np.float64], NDArray[np.float64]], dict[str, float]
        ]
        | None = None,
    ) -> list[FoldResult]:
        """Execute walk-forward validation.

        Args:
            model: The model to validate.
            data: Full historical DataFrame sorted by date.
            target_col: Column name for the target variable.
            horizon: Forecast horizon (defaults to ``test_size``).
            n_scenarios: Monte-Carlo scenarios per prediction.
            metric_fn: Optional function ``(actuals, predictions) -> metrics``
                for custom evaluation.  If ``None``, uses RMSE and MAE.

        Returns:
            List of FoldResult objects, one per fold.

        Raises:
            ValueError: If data is too short for even one fold.
        """
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        n = data.height
        horizon = horizon or self.config.test_size
        min_required = (
            self.config.initial_train_size
            + self.config.embargo_size
            + self.config.test_size
        )

        if n < min_required:
            raise ValueError(
                f"Data has {n} rows but need at least {min_required} "
                f"for one fold."
            )

        self._fold_results = []
        splits = self._generate_splits(n)

        logger.info(
            "Walk-forward validation: {} folds over {} observations",
            len(splits),
            n,
        )

        for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(
            splits
        ):
            train_data = data.slice(train_start, train_end - train_start)
            test_data = data.slice(test_start, test_end - test_start)
            actuals = test_data[target_col].to_numpy().astype(np.float64)

            # Fit model
            if self.config.recalibrate or fold_idx == 0:
                try:
                    model.fit(train_data)
                except Exception as exc:
                    logger.error(
                        "Fold {}: fit failed: {}", fold_idx, exc
                    )
                    continue

            # Predict
            try:
                pred_horizon = min(horizon, len(actuals))
                result = model.predict(pred_horizon, n_scenarios)
                predictions = result.point_forecast[:len(actuals)]
            except Exception as exc:
                logger.error(
                    "Fold {}: predict failed: {}", fold_idx, exc
                )
                continue

            # Compute metrics
            pred_arr = np.array(predictions, dtype=np.float64)
            actual_arr = actuals[: len(predictions)]

            if metric_fn is not None:
                metrics = metric_fn(actual_arr, pred_arr)
            else:
                metrics = self._default_metrics(actual_arr, pred_arr)

            fold = FoldResult(
                fold_index=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                predictions=predictions,
                actuals=actual_arr.tolist(),
                metrics=metrics,
            )
            self._fold_results.append(fold)

            logger.debug(
                "Fold {}: train=[{}:{}], test=[{}:{}], RMSE={:.4f}",
                fold_idx,
                train_start,
                train_end,
                test_start,
                test_end,
                metrics.get("rmse", float("nan")),
            )

        logger.info(
            "Walk-forward complete: {} successful folds",
            len(self._fold_results),
        )
        return self._fold_results

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics across all folds.

        Returns:
            Dictionary with aggregated metrics (mean, std, min, max)
            across all folds.

        Raises:
            RuntimeError: If no folds have been run.
        """
        if not self._fold_results:
            raise RuntimeError("No fold results available. Run validation first.")

        # Collect all metric names
        all_metric_names: set[str] = set()
        for fold in self._fold_results:
            all_metric_names.update(fold.metrics.keys())

        summary: dict[str, Any] = {
            "n_folds": len(self._fold_results),
            "window_type": self.config.window_type.value,
        }

        for name in sorted(all_metric_names):
            values = [
                f.metrics[name]
                for f in self._fold_results
                if name in f.metrics
            ]
            if values:
                arr = np.array(values, dtype=np.float64)
                summary[f"{name}_mean"] = float(np.mean(arr))
                summary[f"{name}_std"] = float(np.std(arr))
                summary[f"{name}_min"] = float(np.min(arr))
                summary[f"{name}_max"] = float(np.max(arr))

        return summary

    def to_dataframe(self) -> pl.DataFrame:
        """Convert fold results to a Polars DataFrame.

        Returns:
            DataFrame with one row per fold, containing split indices
            and metrics.
        """
        rows: list[dict[str, Any]] = []
        for fold in self._fold_results:
            row: dict[str, Any] = {
                "fold": fold.fold_index,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "n_train": fold.train_end - fold.train_start,
                "n_test": fold.test_end - fold.test_start,
            }
            row.update(fold.metrics)
            rows.append(row)

        return pl.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_splits(
        self,
        n: int,
    ) -> list[tuple[int, int, int, int]]:
        """Generate train/test split indices.

        Args:
            n: Total number of observations.

        Returns:
            List of tuples ``(train_start, train_end, test_start, test_end)``.
        """
        splits: list[tuple[int, int, int, int]] = []
        test_start = self.config.initial_train_size + self.config.embargo_size

        while test_start + self.config.test_size <= n:
            test_end = test_start + self.config.test_size

            if self.config.window_type == WindowType.EXPANDING:
                train_start = 0
            else:
                # Rolling: fixed window size
                train_size = self.config.max_train_size or self.config.initial_train_size
                train_start = max(
                    0, test_start - self.config.embargo_size - train_size
                )

            train_end = test_start - self.config.embargo_size

            if train_end - train_start >= 30:  # Minimum viable training set
                splits.append((train_start, train_end, test_start, test_end))

            test_start += self.config.step_size

        return splits

    @staticmethod
    def _default_metrics(
        actuals: NDArray[np.float64],
        predictions: NDArray[np.float64],
    ) -> dict[str, float]:
        """Compute default evaluation metrics.

        Args:
            actuals: Ground truth values.
            predictions: Model predictions.

        Returns:
            Dictionary with RMSE, MAE, MAPE, and directional accuracy.
        """
        n = min(len(actuals), len(predictions))
        a = actuals[:n]
        p = predictions[:n]
        errors = a - p

        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))

        # MAPE (avoid division by zero)
        nonzero = np.abs(a) > 1e-10
        mape = (
            float(np.mean(np.abs(errors[nonzero] / a[nonzero]))) * 100
            if nonzero.any()
            else float("nan")
        )

        # Directional accuracy
        if n > 1:
            actual_dir = np.diff(a) > 0
            pred_dir = np.diff(p) > 0
            dir_acc = float(np.mean(actual_dir == pred_dir)) * 100
        else:
            dir_acc = float("nan")

        return {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "directional_accuracy": dir_acc,
        }
