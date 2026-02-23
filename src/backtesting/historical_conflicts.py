"""Historical conflict backtesting for the IranBSE model suite.

Tests model performance on documented historical conflict episodes to
validate that the system correctly captures supply-disruption dynamics,
geopolitical risk premia, and cross-asset contagion patterns.  Covers
Gulf War (1990), Iraq War (2003), Libya (2011), Iran sanctions episodes,
and Russia-Ukraine (2022).

Typical usage::

    backtester = ConflictBacktester()
    results = backtester.run_all(model, full_data)
    comparison = backtester.comparison_table()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from numpy.typing import NDArray

from src.models.base_model import BaseModel, PredictionResult


@dataclass(frozen=True)
class ConflictEpisode:
    """Definition of a historical conflict period for backtesting.

    Attributes:
        name: Human-readable conflict label.
        start_date: First day of the conflict / shock period.
        end_date: Last day of the active conflict phase.
        pre_start_date: Start of the pre-conflict calibration window
            (typically 1 year before conflict start).
        peak_oil_move_pct: Observed peak oil price change (%).
        peak_wheat_move_pct: Observed peak wheat price change (%).
        description: Brief narrative of the episode.
        key_features: Which alpha signals were most relevant.
    """

    name: str
    start_date: date
    end_date: date
    pre_start_date: date
    peak_oil_move_pct: float
    peak_wheat_move_pct: float
    description: str
    key_features: list[str] = field(default_factory=list)


# Pre-defined historical conflict episodes
CONFLICT_EPISODES: dict[str, ConflictEpisode] = {
    "gulf_war_1990": ConflictEpisode(
        name="Gulf War (1990-1991)",
        start_date=date(1990, 8, 2),
        end_date=date(1991, 2, 28),
        pre_start_date=date(1989, 8, 1),
        peak_oil_move_pct=95.0,
        peak_wheat_move_pct=15.0,
        description="Iraqi invasion of Kuwait. Oil doubled from $21 to $41/bbl. "
        "OPEC spare capacity cushioned some impact.",
        key_features=["opec_spare_capacity", "contango", "vix"],
    ),
    "iraq_war_2003": ConflictEpisode(
        name="Iraq War (2003)",
        start_date=date(2003, 3, 20),
        end_date=date(2003, 5, 1),
        pre_start_date=date(2002, 3, 1),
        peak_oil_move_pct=35.0,
        peak_wheat_move_pct=8.0,
        description="US-led invasion of Iraq. Oil spiked on supply fears, "
        "then fell as Iraqi fields were not destroyed.",
        key_features=["opec_spare_capacity", "cot_positioning", "dxy"],
    ),
    "libya_2011": ConflictEpisode(
        name="Libyan Civil War (2011)",
        start_date=date(2011, 2, 15),
        end_date=date(2011, 10, 31),
        pre_start_date=date(2010, 2, 1),
        peak_oil_move_pct=25.0,
        peak_wheat_move_pct=20.0,
        description="Arab Spring topples Gaddafi. 1.6 mbd offline. "
        "Wheat impacted by broader MENA instability and food security fears.",
        key_features=[
            "opec_spare_capacity",
            "food_security",
            "dark_fleet",
        ],
    ),
    "iran_sanctions_2012": ConflictEpisode(
        name="Iran Sanctions Tightening (2012)",
        start_date=date(2012, 1, 1),
        end_date=date(2012, 7, 1),
        pre_start_date=date(2011, 1, 1),
        peak_oil_move_pct=15.0,
        peak_wheat_move_pct=5.0,
        description="EU embargo and SWIFT disconnection. Iran exports "
        "fell from ~2.5 to ~1.0 mbd. Dark fleet activity surged.",
        key_features=[
            "iran_exports",
            "dark_fleet",
            "contango",
            "ovx",
        ],
    ),
    "iran_jcpoa_withdrawal_2018": ConflictEpisode(
        name="US JCPOA Withdrawal (2018)",
        start_date=date(2018, 5, 8),
        end_date=date(2018, 11, 5),
        pre_start_date=date(2017, 5, 1),
        peak_oil_move_pct=20.0,
        peak_wheat_move_pct=3.0,
        description="US withdraws from Iran nuclear deal, reimposing sanctions. "
        "Oil rose ~20% on supply reduction fears.",
        key_features=["iran_exports", "opec_spare_capacity", "cot_positioning"],
    ),
    "abqaiq_attack_2019": ConflictEpisode(
        name="Abqaiq-Khurais Attack (2019)",
        start_date=date(2019, 9, 14),
        end_date=date(2019, 10, 15),
        pre_start_date=date(2018, 9, 1),
        peak_oil_move_pct=15.0,
        peak_wheat_move_pct=2.0,
        description="Drone/cruise missile attack on Saudi Aramco facilities. "
        "5.7 mbd disrupted. Oil spiked 15% intraday, recovered within weeks.",
        key_features=["opec_spare_capacity", "ovx", "vol_surface"],
    ),
    "russia_ukraine_2022": ConflictEpisode(
        name="Russia-Ukraine War (2022)",
        start_date=date(2022, 2, 24),
        end_date=date(2022, 7, 31),
        pre_start_date=date(2021, 2, 1),
        peak_oil_move_pct=45.0,
        peak_wheat_move_pct=60.0,
        description="Russian invasion of Ukraine. Brent hit $130, wheat "
        "reached all-time highs. Black Sea exports disrupted.",
        key_features=[
            "black_sea_risk",
            "food_security",
            "contango",
            "vol_surface",
            "dxy",
        ],
    ),
}


@dataclass
class ConflictTestResult:
    """Results from backtesting a model on a conflict episode.

    Attributes:
        episode: The conflict episode tested.
        predictions: Model point forecasts during the conflict period.
        actuals: Realised prices during the conflict period.
        rmse: Root mean squared error.
        mae: Mean absolute error.
        directional_accuracy: Fraction of correct direction calls.
        peak_capture_ratio: How much of the peak move the model captured
            (predicted peak / actual peak).
        timing_error_days: How many days early/late the model predicted
            the peak (negative = early, positive = late).
        scenario_coverage: Fraction of the actual path that fell within
            the model's prediction intervals.
    """

    episode: ConflictEpisode
    predictions: list[float]
    actuals: list[float]
    rmse: float
    mae: float
    directional_accuracy: float
    peak_capture_ratio: float
    timing_error_days: int
    scenario_coverage: float


class ConflictBacktester:
    """Backtests models against historical geopolitical conflict episodes.

    For each episode:
    1. Calibrate the model on pre-conflict data.
    2. Generate forward predictions for the conflict duration.
    3. Compare predictions to realised prices.
    4. Evaluate peak capture, timing, and interval coverage.

    Args:
        episodes: Custom episode library to augment built-in episodes.
            If ``None``, uses the standard conflict library.
        target_col: Column name for the target price series in data.
    """

    def __init__(
        self,
        episodes: dict[str, ConflictEpisode] | None = None,
        target_col: str = "target",
    ) -> None:
        self._episodes: dict[str, ConflictEpisode] = {**CONFLICT_EPISODES}
        if episodes:
            self._episodes.update(episodes)
        self.target_col = target_col
        self._results: dict[str, ConflictTestResult] = {}
        logger.info(
            "ConflictBacktester initialised with {} episodes",
            len(self._episodes),
        )

    @property
    def available_episodes(self) -> list[str]:
        """List all available conflict episode names.

        Returns:
            Sorted list of episode identifiers.
        """
        return sorted(self._episodes.keys())

    def run_single(
        self,
        model: BaseModel,
        data: pl.DataFrame,
        episode_name: str,
        n_scenarios: int = 1000,
    ) -> ConflictTestResult:
        """Backtest a model on a single conflict episode.

        Args:
            model: Model to test.
            data: Full historical DataFrame with a ``date`` column and
                the target column.
            episode_name: Key from the episode library.
            n_scenarios: Monte-Carlo scenarios for prediction intervals.

        Returns:
            ConflictTestResult with evaluation metrics.

        Raises:
            KeyError: If the episode name is not found.
            ValueError: If the data does not cover the episode period.
        """
        if episode_name not in self._episodes:
            raise KeyError(
                f"Episode '{episode_name}' not found. "
                f"Available: {self.available_episodes}"
            )

        episode = self._episodes[episode_name]
        return self._test_episode(model, data, episode, n_scenarios)

    def run_all(
        self,
        model: BaseModel,
        data: pl.DataFrame,
        n_scenarios: int = 1000,
    ) -> dict[str, ConflictTestResult]:
        """Backtest a model on all available conflict episodes.

        Episodes where the data does not cover the required period are
        skipped with a warning.

        Args:
            model: Model to test.
            data: Full historical DataFrame.
            n_scenarios: Monte-Carlo scenarios for prediction intervals.

        Returns:
            Mapping of episode name to ConflictTestResult.
        """
        self._results = {}

        for name, episode in self._episodes.items():
            try:
                result = self._test_episode(model, data, episode, n_scenarios)
                self._results[name] = result
                logger.info(
                    "Episode '{}': RMSE={:.4f}, peak_capture={:.2f}, "
                    "dir_acc={:.1%}",
                    name,
                    result.rmse,
                    result.peak_capture_ratio,
                    result.directional_accuracy,
                )
            except ValueError as exc:
                logger.warning(
                    "Skipping episode '{}': {}", name, exc
                )

        logger.info(
            "Conflict backtesting complete: {}/{} episodes tested",
            len(self._results),
            len(self._episodes),
        )
        return self._results

    def comparison_table(self) -> pl.DataFrame:
        """Generate a comparison table of all tested episodes.

        Returns:
            Polars DataFrame with one row per episode and all evaluation
            metrics.

        Raises:
            RuntimeError: If no episodes have been tested.
        """
        if not self._results:
            raise RuntimeError(
                "No results available. Run backtesting first."
            )

        rows: list[dict[str, Any]] = []
        for name, result in self._results.items():
            rows.append({
                "episode": name,
                "conflict_name": result.episode.name,
                "start_date": result.episode.start_date.isoformat(),
                "end_date": result.episode.end_date.isoformat(),
                "actual_oil_peak_pct": result.episode.peak_oil_move_pct,
                "actual_wheat_peak_pct": result.episode.peak_wheat_move_pct,
                "rmse": result.rmse,
                "mae": result.mae,
                "directional_accuracy": result.directional_accuracy,
                "peak_capture_ratio": result.peak_capture_ratio,
                "timing_error_days": result.timing_error_days,
                "scenario_coverage": result.scenario_coverage,
            })

        return pl.DataFrame(rows).sort("start_date")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _test_episode(
        self,
        model: BaseModel,
        data: pl.DataFrame,
        episode: ConflictEpisode,
        n_scenarios: int,
    ) -> ConflictTestResult:
        """Execute a single episode backtest.

        Args:
            model: Model to test.
            data: Full historical DataFrame.
            episode: Conflict episode definition.
            n_scenarios: Monte-Carlo scenarios.

        Returns:
            ConflictTestResult.

        Raises:
            ValueError: If data does not cover the required period.
        """
        if "date" not in data.columns:
            raise ValueError("Data must contain a 'date' column")

        # Filter data for pre-conflict training period
        train_data = data.filter(
            (pl.col("date") >= pl.lit(episode.pre_start_date))
            & (pl.col("date") < pl.lit(episode.start_date))
        )

        if train_data.height < 30:
            raise ValueError(
                f"Insufficient training data for '{episode.name}': "
                f"{train_data.height} rows (need >= 30)"
            )

        # Filter data for conflict test period
        test_data = data.filter(
            (pl.col("date") >= pl.lit(episode.start_date))
            & (pl.col("date") <= pl.lit(episode.end_date))
        )

        if test_data.height < 5:
            raise ValueError(
                f"Insufficient test data for '{episode.name}': "
                f"{test_data.height} rows"
            )

        # Fit model on pre-conflict data
        model.fit(train_data)

        # Predict for conflict duration
        horizon = test_data.height
        result = model.predict(horizon, n_scenarios)

        actuals = test_data[self.target_col].to_numpy().astype(np.float64)
        predictions = np.array(
            result.point_forecast[:horizon], dtype=np.float64
        )

        # Pad predictions if shorter than actuals
        if len(predictions) < len(actuals):
            last_pred = predictions[-1] if len(predictions) > 0 else 0.0
            predictions = np.concatenate([
                predictions,
                np.full(len(actuals) - len(predictions), last_pred),
            ])

        # Compute metrics
        errors = actuals - predictions
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))

        # Directional accuracy
        if len(actuals) > 1:
            actual_dir = np.diff(actuals) > 0
            pred_dir = np.diff(predictions) > 0
            dir_acc = float(np.mean(actual_dir == pred_dir))
        else:
            dir_acc = 0.0

        # Peak capture
        actual_peak_idx = int(np.argmax(np.abs(actuals - actuals[0])))
        pred_peak_idx = int(np.argmax(np.abs(predictions - predictions[0])))

        actual_peak_move = actuals[actual_peak_idx] - actuals[0]
        pred_peak_move = predictions[pred_peak_idx] - predictions[0]

        peak_capture = (
            pred_peak_move / actual_peak_move
            if abs(actual_peak_move) > 1e-10
            else 0.0
        )

        timing_error = pred_peak_idx - actual_peak_idx

        # Scenario coverage
        coverage = self._compute_coverage(result, actuals)

        return ConflictTestResult(
            episode=episode,
            predictions=predictions.tolist(),
            actuals=actuals.tolist(),
            rmse=rmse,
            mae=mae,
            directional_accuracy=dir_acc,
            peak_capture_ratio=float(peak_capture),
            timing_error_days=int(timing_error),
            scenario_coverage=coverage,
        )

    @staticmethod
    def _compute_coverage(
        result: PredictionResult,
        actuals: NDArray[np.float64],
    ) -> float:
        """Compute what fraction of actuals fall within prediction intervals.

        Uses the 90% prediction interval (5th to 95th percentile).

        Args:
            result: Model prediction result with bounds.
            actuals: Realised values.

        Returns:
            Coverage fraction in [0, 1].
        """
        lower_key = None
        upper_key = None

        for k in result.lower_bounds:
            if abs(k - 0.05) < 0.02:
                lower_key = k
                break
        if lower_key is None and result.lower_bounds:
            lower_key = min(result.lower_bounds.keys())

        for k in result.upper_bounds:
            if abs(k - 0.95) < 0.02:
                upper_key = k
                break
        if upper_key is None and result.upper_bounds:
            upper_key = max(result.upper_bounds.keys())

        if lower_key is None or upper_key is None:
            return 0.0

        lower = np.array(result.lower_bounds[lower_key], dtype=np.float64)
        upper = np.array(result.upper_bounds[upper_key], dtype=np.float64)

        n = min(len(actuals), len(lower), len(upper))
        within = (actuals[:n] >= lower[:n]) & (actuals[:n] <= upper[:n])
        return float(np.mean(within))
