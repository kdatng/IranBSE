"""Black Sea shipping disruption risk signals for wheat futures.

The Black Sea region (Russia, Ukraine, Romania) accounts for approximately
30% of global wheat exports.  Disruptions at key ports (especially
Novorossiysk) and Russian export policy changes (taxes, quotas, bans)
directly impact global wheat supply and prices.

Typical usage::

    signal = BlackSeaRiskSignal()
    features = signal.compute(shipping_data, policy_data)
    congestion = signal.novorossiysk_congestion(vessel_data)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from loguru import logger


@dataclass
class BlackSeaConfig:
    """Configuration for Black Sea risk signals.

    Attributes:
        vessel_queue_col: Column for vessel queue length at Novorossiysk.
        wait_time_col: Column for average vessel waiting time (days).
        freight_rate_col: Column for Black Sea to Mediterranean freight
            rate (USD/tonne).
        export_tax_col: Column for Russian wheat export tax rate (USD/tonne).
        export_volume_col: Column for weekly/monthly Black Sea wheat exports.
        lookback_short: Short-term window for momentum. 4 weeks reflects
            a typical shipping cycle.
        lookback_long: Long-term window for baseline. 26 weeks (~6 months)
            provides seasonal context.
        congestion_threshold: Queue length (vessels) above which
            congestion is flagged. 50 vessels at Novorossiysk is roughly
            double the normal queue based on historical port data.
    """

    vessel_queue_col: str = "novo_vessel_queue"
    wait_time_col: str = "avg_wait_days"
    freight_rate_col: str = "bs_freight_rate"
    export_tax_col: str = "russia_export_tax"
    export_volume_col: str = "bs_wheat_exports_mt"
    lookback_short: int = 4
    lookback_long: int = 26
    congestion_threshold: int = 50


class BlackSeaRiskSignal:
    """Black Sea shipping disruption and Russian export policy signals.

    Computes:
    - **Novorossiysk congestion index**: Port queue length and wait times
      relative to seasonal norms.
    - **Freight rate premium**: Black Sea freight rates vs. benchmark,
      capturing war risk / insurance premium.
    - **Russian export tax momentum**: Direction and magnitude of export
      tax changes (higher tax = more expensive Russian wheat).
    - **Export flow disruption**: Deviations from seasonal export volume
      patterns.
    - **Composite Black Sea risk score**: Weighted combination of all
      sub-signals.

    Args:
        config: Configuration parameters.
    """

    def __init__(self, config: BlackSeaConfig | None = None) -> None:
        self.config = config or BlackSeaConfig()
        logger.info(
            "BlackSeaRiskSignal initialised: congestion threshold={} vessels",
            self.config.congestion_threshold,
        )

    def compute(
        self,
        shipping_data: pl.DataFrame,
        policy_data: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Compute all Black Sea risk signals.

        Args:
            shipping_data: DataFrame with shipping columns (vessel queue,
                wait times, freight rates, export volumes).
            policy_data: Optional DataFrame with Russian export tax data.

        Returns:
            DataFrame with signal columns:
            ``congestion_index``, ``freight_premium_zscore``,
            ``export_flow_deviation``, ``bs_risk_composite``.

        Raises:
            ValueError: If critical shipping columns are missing.
        """
        result = shipping_data.clone()

        # --- Congestion index ---
        if self.config.vessel_queue_col in result.columns:
            result = self._add_congestion_signals(result)

        # --- Freight rate premium ---
        if self.config.freight_rate_col in result.columns:
            result = self._add_freight_signals(result)

        # --- Export flow deviation ---
        if self.config.export_volume_col in result.columns:
            result = self._add_export_flow_signals(result)

        # --- Export tax signals ---
        if policy_data is not None and self.config.export_tax_col in policy_data.columns:
            tax_signals = self._compute_tax_signals(policy_data)
            if "date" in result.columns and "date" in tax_signals.columns:
                result = result.join(tax_signals, on="date", how="left")

        # --- Composite risk score ---
        result = self._compute_composite(result)

        logger.info(
            "Black Sea risk signals computed: {} rows",
            result.height,
        )
        return result

    def novorossiysk_congestion(
        self,
        vessel_data: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compute detailed Novorossiysk port congestion metrics.

        Args:
            vessel_data: DataFrame with columns ``vessel_queue``,
                ``avg_wait_days``, and ``date``.

        Returns:
            DataFrame with congestion analysis columns:
            ``congestion_severity``, ``congestion_momentum``,
            ``wait_time_zscore``, ``congestion_alert``.
        """
        required = [self.config.vessel_queue_col, self.config.wait_time_col]
        self._validate_columns(vessel_data, required)

        result = vessel_data.clone()

        # Queue severity: normalised 0-1 scale
        queue_vals = result[self.config.vessel_queue_col].to_numpy().astype(np.float64)
        severity = np.clip(queue_vals / self.config.congestion_threshold, 0, 2.0)
        result = result.with_columns(
            pl.Series("congestion_severity", severity)
        )

        # Congestion momentum
        result = result.with_columns(
            (
                pl.col(self.config.vessel_queue_col).rolling_mean(
                    window_size=self.config.lookback_short
                )
                - pl.col(self.config.vessel_queue_col).rolling_mean(
                    window_size=self.config.lookback_long
                )
            ).alias("congestion_momentum")
        )

        # Wait time z-score
        result = result.with_columns(
            (
                (
                    pl.col(self.config.wait_time_col)
                    - pl.col(self.config.wait_time_col).rolling_mean(
                        window_size=self.config.lookback_long
                    )
                )
                / pl.col(self.config.wait_time_col)
                .rolling_std(window_size=self.config.lookback_long)
                .clip(lower_bound=0.01)
            ).alias("wait_time_zscore")
        )

        # Alert flag: severe congestion
        result = result.with_columns(
            (
                (pl.col("congestion_severity") > 1.0)
                | (pl.col("wait_time_zscore") > 2.0)
            )
            .cast(pl.Int8)
            .alias("congestion_alert")
        )

        return result

    def russian_export_tax_impact(
        self,
        policy_data: pl.DataFrame,
        wheat_price_col: str = "wheat_price",
    ) -> pl.DataFrame:
        """Estimate the price impact of Russian export tax changes.

        The Russian floating export tax acts as a floor under global wheat
        prices by raising the effective cost of Russian exports.

        Args:
            policy_data: DataFrame with export tax and wheat price columns.
            wheat_price_col: Column name for the benchmark wheat price.

        Returns:
            DataFrame with ``tax_as_pct_of_price``, ``tax_change``,
            and ``effective_floor_price`` columns.
        """
        self._validate_columns(
            policy_data,
            [self.config.export_tax_col, wheat_price_col],
        )

        result = policy_data.clone()

        # Tax as percentage of wheat price
        result = result.with_columns(
            (
                pl.col(self.config.export_tax_col)
                / pl.col(wheat_price_col).clip(lower_bound=0.01)
                * 100
            ).alias("tax_as_pct_of_price")
        )

        # Tax change (period-over-period)
        result = result.with_columns(
            (pl.col(self.config.export_tax_col) - pl.col(self.config.export_tax_col).shift(1))
            .alias("tax_change")
        )

        # Effective floor: Russian FOB cost + tax
        result = result.with_columns(
            (pl.col(wheat_price_col) + pl.col(self.config.export_tax_col))
            .alias("effective_floor_price")
        )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_congestion_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add congestion-related columns to the working DataFrame.

        Args:
            df: Working DataFrame.

        Returns:
            DataFrame with congestion columns.
        """
        col = self.config.vessel_queue_col

        # Normalised congestion (ratio to threshold)
        df = df.with_columns(
            (pl.col(col) / self.config.congestion_threshold).alias("congestion_index")
        )

        # Congestion z-score
        df = df.with_columns(
            (
                (pl.col(col) - pl.col(col).rolling_mean(window_size=self.config.lookback_long))
                / pl.col(col).rolling_std(window_size=self.config.lookback_long).clip(lower_bound=0.01)
            ).alias("congestion_zscore")
        )

        return df

    def _add_freight_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add freight rate premium signals.

        Args:
            df: Working DataFrame.

        Returns:
            DataFrame with freight columns.
        """
        col = self.config.freight_rate_col

        df = df.with_columns(
            (
                (pl.col(col) - pl.col(col).rolling_mean(window_size=self.config.lookback_long))
                / pl.col(col).rolling_std(window_size=self.config.lookback_long).clip(lower_bound=0.01)
            ).alias("freight_premium_zscore")
        )

        # Freight momentum
        df = df.with_columns(
            (
                pl.col(col).rolling_mean(window_size=self.config.lookback_short)
                - pl.col(col).rolling_mean(window_size=self.config.lookback_long)
            ).alias("freight_momentum")
        )

        return df

    def _add_export_flow_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add export flow deviation signals.

        Args:
            df: Working DataFrame.

        Returns:
            DataFrame with export flow columns.
        """
        col = self.config.export_volume_col

        # Deviation from rolling mean (seasonal proxy)
        df = df.with_columns(
            (
                (pl.col(col) - pl.col(col).rolling_mean(window_size=self.config.lookback_long))
                / pl.col(col).rolling_mean(window_size=self.config.lookback_long).clip(lower_bound=0.01)
            ).alias("export_flow_deviation")
        )

        return df

    def _compute_tax_signals(self, policy_data: pl.DataFrame) -> pl.DataFrame:
        """Compute export tax signals standalone.

        Args:
            policy_data: Policy DataFrame with tax column.

        Returns:
            DataFrame with tax signal columns.
        """
        col = self.config.export_tax_col
        result = policy_data.select(
            [c for c in policy_data.columns if c in ["date", col]]
        )

        result = result.with_columns(
            (pl.col(col) - pl.col(col).shift(1)).alias("export_tax_change"),
            (pl.col(col).pct_change()).alias("export_tax_pct_change"),
        )

        # Tax direction: rising taxes are bullish for wheat
        result = result.with_columns(
            pl.col("export_tax_change").sign().cast(pl.Int8).alias("tax_direction")
        )

        return result

    def _compute_composite(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute a composite Black Sea risk score from available sub-signals.

        The composite is a weighted average of available z-scored signals.
        Missing signals are excluded from the average.

        Args:
            df: DataFrame with individual signal columns.

        Returns:
            DataFrame with ``bs_risk_composite`` column.
        """
        components: dict[str, float] = {
            "congestion_zscore": 0.30,
            "freight_premium_zscore": 0.30,
            "export_flow_deviation": 0.25,
            "export_tax_change": 0.15,
        }

        available = {k: v for k, v in components.items() if k in df.columns}
        if not available:
            df = df.with_columns(pl.lit(np.nan).alias("bs_risk_composite"))
            return df

        # Renormalise weights
        total_weight = sum(available.values())
        normalised = {k: v / total_weight for k, v in available.items()}

        composite_expr = sum(
            pl.col(col) * weight for col, weight in normalised.items()
        )
        df = df.with_columns(composite_expr.alias("bs_risk_composite"))

        return df

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
