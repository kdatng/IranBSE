"""CFTC Commitments of Traders (CoT) positioning signals.

Tracks managed money net positioning in crude oil and wheat futures from
the weekly CFTC CoT reports.  Extreme positioning (crowding) creates
vulnerability to sharp reversals, especially when a geopolitical catalyst
arrives.  When managed money is max-long oil and a conflict escalates,
the initial price spike can be amplified; when already max-long,
incremental buying power is exhausted and profit-taking risk rises.

Typical usage::

    cot = COTPositioning()
    signals = cot.compute(cot_data)
    crowding = cot.crowding_indicators(cot_data)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from loguru import logger


@dataclass
class COTConfig:
    """Configuration for COT positioning signals.

    Attributes:
        mm_long_col: Column for managed money long contracts.
        mm_short_col: Column for managed money short contracts.
        comm_long_col: Column for commercial (hedger) long contracts.
        comm_short_col: Column for commercial (hedger) short contracts.
        oi_col: Column for open interest (total contracts).
        lookback: Rolling window for percentile ranking. 52 weeks
            provides a full year of CoT reports for context.
        extreme_percentile: Threshold for extreme positioning flag.
            90th percentile captures historically unusual crowding.
        crowding_lookback: Window for crowding acceleration. 13 weeks
            (~1 quarter) captures medium-term positioning shifts.
    """

    mm_long_col: str = "mm_long"
    mm_short_col: str = "mm_short"
    comm_long_col: str = "comm_long"
    comm_short_col: str = "comm_short"
    oi_col: str = "open_interest"
    lookback: int = 52
    extreme_percentile: float = 90.0
    crowding_lookback: int = 13


class COTPositioning:
    """CFTC CoT managed money positioning and crowding signals.

    Computes:
    - **Net positioning**: Managed money long minus short contracts.
    - **Net-to-OI ratio**: Net positioning normalised by open interest
      for cross-commodity comparability.
    - **Historical percentile**: Current net position ranked against
      historical distribution.
    - **Positioning extremes**: Flags when positioning reaches historically
      unusual levels (potential reversal risk).
    - **Crowding velocity**: Rate of change in positioning, detecting
      rapid herding.
    - **Commercial hedger divergence**: When commercials and specs
      disagree, commercials are historically more reliable.

    Args:
        config: Configuration parameters.
    """

    def __init__(self, config: COTConfig | None = None) -> None:
        self.config = config or COTConfig()
        logger.info(
            "COTPositioning initialised: lookback={}, extreme_pct={:.0f}",
            self.config.lookback,
            self.config.extreme_percentile,
        )

    def compute(self, cot_data: pl.DataFrame) -> pl.DataFrame:
        """Compute all COT positioning signals.

        Args:
            cot_data: Weekly DataFrame with managed money and commercial
                positioning columns plus open interest.

        Returns:
            DataFrame with signal columns:
            ``mm_net``, ``mm_net_oi_ratio``, ``mm_net_percentile``,
            ``extreme_long_flag``, ``extreme_short_flag``,
            ``positioning_velocity``, ``comm_mm_divergence``.

        Raises:
            ValueError: If required positioning columns are missing.
        """
        required = [self.config.mm_long_col, self.config.mm_short_col]
        self._validate_columns(cot_data, required)

        result = cot_data.clone()

        # --- Net managed money position ---
        result = result.with_columns(
            (pl.col(self.config.mm_long_col) - pl.col(self.config.mm_short_col))
            .alias("mm_net")
        )

        # --- Net-to-OI ratio ---
        if self.config.oi_col in result.columns:
            result = result.with_columns(
                (
                    pl.col("mm_net")
                    / pl.col(self.config.oi_col).clip(lower_bound=1)
                ).alias("mm_net_oi_ratio")
            )

        # --- Historical percentile ---
        result = self._add_rolling_percentile(
            result, "mm_net", self.config.lookback
        )

        # --- Extreme positioning flags ---
        result = result.with_columns(
            (pl.col("mm_net_percentile") >= self.config.extreme_percentile)
            .cast(pl.Int8)
            .alias("extreme_long_flag"),
            (pl.col("mm_net_percentile") <= (100 - self.config.extreme_percentile))
            .cast(pl.Int8)
            .alias("extreme_short_flag"),
        )

        # --- Positioning velocity ---
        result = result.with_columns(
            (pl.col("mm_net") - pl.col("mm_net").shift(1)).alias("mm_net_change")
        )
        result = result.with_columns(
            pl.col("mm_net_change")
            .rolling_mean(window_size=4)
            .alias("positioning_velocity")
        )

        # --- Positioning acceleration ---
        result = result.with_columns(
            (
                pl.col("positioning_velocity")
                - pl.col("positioning_velocity").shift(1)
            ).alias("positioning_acceleration")
        )

        # --- Commercial hedger divergence ---
        if (
            self.config.comm_long_col in result.columns
            and self.config.comm_short_col in result.columns
        ):
            result = self._add_commercial_divergence(result)

        # --- Z-score of net positioning ---
        result = result.with_columns(
            (
                (
                    pl.col("mm_net")
                    - pl.col("mm_net").rolling_mean(
                        window_size=self.config.lookback
                    )
                )
                / pl.col("mm_net")
                .rolling_std(window_size=self.config.lookback)
                .clip(lower_bound=1.0)
            ).alias("mm_net_zscore")
        )

        logger.info(
            "COT positioning signals computed: {} rows",
            result.height,
        )
        return result

    def crowding_indicators(self, cot_data: pl.DataFrame) -> pl.DataFrame:
        """Compute crowding and herding risk indicators.

        Crowding occurs when managed money positioning is both extreme
        and accelerating.  This creates fragility: any adverse catalyst
        can trigger a stampede of position liquidation.

        Args:
            cot_data: Weekly CoT DataFrame.

        Returns:
            DataFrame with crowding-specific columns:
            ``crowding_score``, ``herding_velocity``,
            ``reversal_risk``, ``crowding_alert``.
        """
        # First compute base signals
        base = self.compute(cot_data)

        # --- Crowding score ---
        # Combines extremity and velocity
        percentile_col = "mm_net_percentile"
        velocity_col = "positioning_velocity"

        if percentile_col not in base.columns or velocity_col not in base.columns:
            logger.warning("Cannot compute crowding: missing base signals")
            return base

        # Normalised distance from 50th percentile (how extreme)
        base = base.with_columns(
            ((pl.col(percentile_col) - 50).abs() / 50).alias("position_extremity")
        )

        # Normalised velocity z-score
        base = base.with_columns(
            (
                (
                    pl.col(velocity_col)
                    - pl.col(velocity_col).rolling_mean(
                        window_size=self.config.crowding_lookback
                    )
                )
                / pl.col(velocity_col)
                .rolling_std(window_size=self.config.crowding_lookback)
                .clip(lower_bound=1.0)
            ).alias("velocity_zscore")
        )

        # Crowding score = extremity * velocity direction alignment
        base = base.with_columns(
            (
                pl.col("position_extremity")
                * pl.col("velocity_zscore").abs()
            ).alias("crowding_score")
        )

        # Herding velocity: rate at which positioning converges to extreme
        base = base.with_columns(
            pl.col("position_extremity")
            .rolling_mean(window_size=4)
            .diff()
            .alias("herding_velocity")
        )

        # Reversal risk: high crowding with decelerating momentum
        base = base.with_columns(
            (
                (pl.col("crowding_score") > 1.0)
                & (pl.col("positioning_acceleration") < 0)
            )
            .cast(pl.Int8)
            .alias("reversal_risk")
        )

        # Crowding alert
        base = base.with_columns(
            (pl.col("crowding_score") > 1.5)
            .cast(pl.Int8)
            .alias("crowding_alert")
        )

        return base

    def spec_commercial_spread(self, cot_data: pl.DataFrame) -> pl.Series:
        """Compute the speculator-commercial positioning spread.

        When speculators and commercials take opposing extreme positions,
        the resulting "tug of war" often resolves in favour of commercial
        hedgers, who have superior information about physical supply/demand.

        Args:
            cot_data: Weekly CoT DataFrame.

        Returns:
            Series named ``spec_comm_spread``.
        """
        mm_net = (
            cot_data[self.config.mm_long_col].to_numpy().astype(np.float64)
            - cot_data[self.config.mm_short_col].to_numpy().astype(np.float64)
        )

        if (
            self.config.comm_long_col in cot_data.columns
            and self.config.comm_short_col in cot_data.columns
        ):
            comm_net = (
                cot_data[self.config.comm_long_col].to_numpy().astype(np.float64)
                - cot_data[self.config.comm_short_col].to_numpy().astype(np.float64)
            )
        else:
            comm_net = np.zeros_like(mm_net)

        spread = mm_net - comm_net
        return pl.Series("spec_comm_spread", spread)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_commercial_divergence(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add commercial-speculator divergence signal.

        Args:
            df: Working DataFrame.

        Returns:
            DataFrame with divergence columns.
        """
        # Commercial net position
        df = df.with_columns(
            (pl.col(self.config.comm_long_col) - pl.col(self.config.comm_short_col))
            .alias("comm_net")
        )

        # Divergence: spec net minus commercial net (normalised)
        df = df.with_columns(
            (
                (pl.col("mm_net") - pl.col("comm_net"))
                / (pl.col("mm_net").abs() + pl.col("comm_net").abs()).clip(lower_bound=1.0)
            ).alias("comm_mm_divergence")
        )

        # Divergence z-score
        df = df.with_columns(
            (
                (
                    pl.col("comm_mm_divergence")
                    - pl.col("comm_mm_divergence").rolling_mean(
                        window_size=self.config.lookback
                    )
                )
                / pl.col("comm_mm_divergence")
                .rolling_std(window_size=self.config.lookback)
                .clip(lower_bound=0.001)
            ).alias("comm_mm_divergence_zscore")
        )

        return df

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
