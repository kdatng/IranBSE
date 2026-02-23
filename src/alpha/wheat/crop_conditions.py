"""USDA crop condition ratings and WASDE surprise momentum signals.

Tracks weekly crop condition reports and monthly WASDE (World Agricultural
Supply and Demand Estimates) releases to derive alpha signals for wheat
futures.  Deteriorating crop conditions and negative WASDE surprises are
bullish for wheat, especially when combined with geopolitical supply risks.

Typical usage::

    signal = CropConditionSignal()
    features = signal.compute(crop_data, wasde_data)
    surprise = signal.wasde_surprise_momentum(wasde_data)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from loguru import logger


@dataclass
class CropConditionConfig:
    """Configuration for crop condition signal computation.

    Attributes:
        good_excellent_col: Column for USDA "Good + Excellent" percentage.
        poor_very_poor_col: Column for USDA "Poor + Very Poor" percentage.
        production_estimate_col: Column for WASDE production estimate (mmt).
        prior_estimate_col: Column for prior month WASDE estimate (mmt).
        consensus_col: Column for analyst consensus estimate (mmt).
        lookback: Rolling window for condition momentum. 4 weeks is the
            standard reporting cadence for meaningful week-over-week changes.
        historical_lookback: Longer window for percentile ranking.
            52 weeks provides a full crop year of context.
    """

    good_excellent_col: str = "good_excellent_pct"
    poor_very_poor_col: str = "poor_very_poor_pct"
    production_estimate_col: str = "wasde_production_mmt"
    prior_estimate_col: str = "wasde_prior_mmt"
    consensus_col: str = "consensus_production_mmt"
    lookback: int = 4
    historical_lookback: int = 52


class CropConditionSignal:
    """USDA crop condition and WASDE surprise signals for wheat.

    Computes:
    - **Condition index**: Net crop condition (Good+Excellent minus
      Poor+VeryPoor) as a single sentiment measure.
    - **Condition momentum**: Week-over-week change in condition index
      during the growing season.
    - **Historical percentile**: Current condition ranked against the
      past year's distribution.
    - **WASDE surprise**: Deviation of actual WASDE release from analyst
      consensus, normalised by historical surprise volatility.
    - **WASDE revision momentum**: Cumulative direction of WASDE estimate
      revisions.

    Args:
        config: Configuration parameters.
    """

    def __init__(self, config: CropConditionConfig | None = None) -> None:
        self.config = config or CropConditionConfig()
        logger.info(
            "CropConditionSignal initialised: lookback={}, hist_lookback={}",
            self.config.lookback,
            self.config.historical_lookback,
        )

    def compute(
        self,
        crop_data: pl.DataFrame,
        wasde_data: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Compute all crop condition signals.

        Args:
            crop_data: Weekly DataFrame with USDA crop condition columns.
                Must include a ``date`` column sorted ascending.
            wasde_data: Optional monthly DataFrame with WASDE production
                estimates.

        Returns:
            DataFrame with signal columns:
            ``condition_index``, ``condition_momentum``,
            ``condition_percentile``, and optionally WASDE signals.

        Raises:
            ValueError: If required crop condition columns are missing.
        """
        required = [self.config.good_excellent_col, self.config.poor_very_poor_col]
        self._validate_columns(crop_data, required)

        result = crop_data.clone()

        # --- Net condition index ---
        result = result.with_columns(
            (
                pl.col(self.config.good_excellent_col)
                - pl.col(self.config.poor_very_poor_col)
            ).alias("condition_index")
        )

        # --- Condition momentum (week-over-week change) ---
        result = result.with_columns(
            (pl.col("condition_index") - pl.col("condition_index").shift(1))
            .alias("condition_momentum")
        )

        # --- Rolling momentum (smoothed over lookback weeks) ---
        result = result.with_columns(
            pl.col("condition_momentum")
            .rolling_mean(window_size=self.config.lookback)
            .alias("condition_momentum_smooth")
        )

        # --- Historical percentile ranking ---
        result = self._add_rolling_percentile(
            result, "condition_index", self.config.historical_lookback
        )

        # --- Condition deterioration rate ---
        result = result.with_columns(
            (
                pl.col("condition_index")
                - pl.col("condition_index").shift(self.config.lookback)
            ).alias("condition_change_4w")
        )

        # --- WASDE surprise signals ---
        if wasde_data is not None:
            wasde_signals = self.wasde_surprise_momentum(wasde_data)
            # Only merge if both have date columns
            if "date" in result.columns and "date" in wasde_signals.columns:
                result = result.join(wasde_signals, on="date", how="left")
            else:
                logger.warning(
                    "Cannot merge WASDE signals: 'date' column missing"
                )

        logger.info(
            "Crop condition signals computed: {} rows, "
            "mean condition index={:.1f}",
            result.height,
            float(result["condition_index"].mean()),  # type: ignore[arg-type]
        )
        return result

    def wasde_surprise_momentum(self, wasde_data: pl.DataFrame) -> pl.DataFrame:
        """Compute WASDE release surprise and revision momentum.

        Args:
            wasde_data: Monthly DataFrame with WASDE production estimate,
                prior estimate, and consensus columns.

        Returns:
            DataFrame with columns:
            ``wasde_surprise``, ``wasde_surprise_zscore``,
            ``revision_direction``, ``revision_streak``.
        """
        result = wasde_data.clone()

        # --- Surprise vs consensus ---
        if self.config.consensus_col in wasde_data.columns:
            result = result.with_columns(
                (
                    pl.col(self.config.production_estimate_col)
                    - pl.col(self.config.consensus_col)
                ).alias("wasde_surprise")
            )

            # Z-score of surprise
            result = result.with_columns(
                (
                    (
                        pl.col("wasde_surprise")
                        - pl.col("wasde_surprise").rolling_mean(
                            window_size=self.config.historical_lookback
                        )
                    )
                    / pl.col("wasde_surprise")
                    .rolling_std(window_size=self.config.historical_lookback)
                    .clip(lower_bound=0.001)
                ).alias("wasde_surprise_zscore")
            )

        # --- Revision direction ---
        if self.config.prior_estimate_col in wasde_data.columns:
            result = result.with_columns(
                (
                    pl.col(self.config.production_estimate_col)
                    - pl.col(self.config.prior_estimate_col)
                ).alias("wasde_revision")
            )

            # Revision direction: +1 upward, -1 downward, 0 unchanged
            result = result.with_columns(
                pl.col("wasde_revision")
                .sign()
                .cast(pl.Int8)
                .alias("revision_direction")
            )

            # Revision streak: consecutive same-direction revisions
            directions = result["revision_direction"].to_numpy()
            streaks = self._compute_streak(directions)
            result = result.with_columns(
                pl.Series("revision_streak", streaks)
            )

        return result

    def condition_stress_score(self, crop_data: pl.DataFrame) -> pl.Series:
        """Compute a composite crop stress score.

        Combines condition deterioration rate, absolute level, and
        percentile ranking into a single 0-100 stress score.  Higher
        scores indicate more stressed crops (bullish for wheat).

        Args:
            crop_data: DataFrame with crop condition columns.

        Returns:
            Series named ``crop_stress_score`` in range [0, 100].
        """
        good_exc = crop_data[self.config.good_excellent_col].to_numpy().astype(np.float64)
        poor_vp = crop_data[self.config.poor_very_poor_col].to_numpy().astype(np.float64)

        # Invert good/excellent (lower = more stress)
        ge_stress = 100.0 - good_exc
        # Poor/very poor directly measures stress
        pvp_stress = poor_vp

        # Rate of deterioration (negative momentum = increasing stress)
        net_condition = good_exc - poor_vp
        momentum = np.full_like(net_condition, 0.0)
        momentum[1:] = net_condition[1:] - net_condition[:-1]
        # Negative momentum => positive stress contribution
        momentum_stress = np.clip(-momentum * 5, 0, 100)

        # Composite
        composite = 0.4 * ge_stress + 0.35 * pvp_stress + 0.25 * momentum_stress
        composite = np.clip(composite, 0, 100)

        return pl.Series("crop_stress_score", composite)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_rolling_percentile(
        df: pl.DataFrame,
        column: str,
        window: int,
    ) -> pl.DataFrame:
        """Add a rolling percentile rank for a column.

        Args:
            df: Input DataFrame.
            column: Column to rank.
            window: Window size for percentile computation.

        Returns:
            DataFrame with ``{column}_percentile`` appended.
        """
        values = df[column].to_numpy().astype(np.float64)
        percentiles = np.full_like(values, np.nan)

        for i in range(window, len(values)):
            history = values[i - window : i]
            valid = history[~np.isnan(history)]
            if len(valid) > 0:
                percentiles[i] = float(np.mean(valid <= values[i])) * 100

        return df.with_columns(
            pl.Series(f"{column}_percentile", percentiles)
        )

    @staticmethod
    def _compute_streak(directions: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Count consecutive same-direction values.

        Positive values indicate upward streak length; negative values
        indicate downward streak length.

        Args:
            directions: Array of direction indicators (+1, 0, -1).

        Returns:
            Streak array.
        """
        streaks = np.zeros(len(directions), dtype=np.float64)
        for i in range(1, len(directions)):
            if directions[i] == directions[i - 1] and directions[i] != 0:
                streaks[i] = streaks[i - 1] + directions[i]
            elif directions[i] != 0:
                streaks[i] = float(directions[i])
        return streaks

    @staticmethod
    def _validate_columns(df: pl.DataFrame, required: list[str]) -> None:
        """Validate required columns exist.

        Args:
            df: DataFrame to check.
            required: Required column names.

        Raises:
            ValueError: If columns are missing.
        """
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
