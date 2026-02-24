"""Fundamental / supply-demand feature processor.

Computes features related to physical commodity supply-demand dynamics:
inventory surprises, OPEC spare capacity estimates, and crack spreads.
These features capture the fundamental drivers that amplify or dampen
price reactions to geopolitical shocks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from pydantic import Field

from src.data.processors.base_processor import BaseProcessor, ProcessorConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# JUSTIFIED: EIA weekly inventory report consensus expectations are
# typically within +/- 2 million barrels; a "surprise" beyond 1 std
# is market-moving (see Hamilton 2009, Kilian & Murphy 2014).
INVENTORY_SURPRISE_LOOKBACK: int = 52  # JUSTIFIED: 1 year of weekly data
INVENTORY_SURPRISE_THRESHOLD: float = 1.5  # JUSTIFIED: 1.5 sigma = significant surprise

# JUSTIFIED: OPEC effective spare capacity is estimated at ~3-5 mbpd
# (EIA Short-Term Energy Outlook); ratio to global demand (~100 mbpd)
# gives the spare capacity cushion.
GLOBAL_OIL_DEMAND_MBPD: float = 102.0  # JUSTIFIED: IEA 2025 estimate
OPEC_BASELINE_SPARE_MBPD: float = 4.0  # JUSTIFIED: EIA STEO Dec 2025


class FundamentalConfig(ProcessorConfig):
    """Configuration for the fundamental feature processor.

    Attributes:
        inventory_col: Column name for crude oil inventory levels.
        inventory_surprise_lookback: Rolling window for computing inventory
            z-scores (in observation periods, typically weekly).
        crack_spread_products: Mapping of product columns used for crack
            spread computation.
        global_demand_mbpd: Global oil demand baseline for spare capacity
            ratio computation.
        opec_spare_baseline_mbpd: Estimated OPEC spare capacity baseline.
    """

    name: str = "fundamental_features"
    inventory_col: str = Field(default="crude_inventory")
    inventory_surprise_lookback: int = Field(default=INVENTORY_SURPRISE_LOOKBACK, ge=4)
    crack_spread_products: dict[str, str] = Field(
        default_factory=lambda: {
            "gasoline": "gasoline_close",
            "heating_oil": "heating_oil_close",
        }
    )
    global_demand_mbpd: float = Field(default=GLOBAL_OIL_DEMAND_MBPD, gt=0)
    opec_spare_baseline_mbpd: float = Field(default=OPEC_BASELINE_SPARE_MBPD, ge=0)


class FundamentalProcessor(BaseProcessor):
    """Computes supply-demand fundamental features for commodity markets.

    Features produced:
        - Inventory surprise z-scores (bullish/bearish draw vs. build).
        - Inventory momentum (rate of change in inventories).
        - OPEC spare capacity ratio (cushion vs. demand).
        - Spare capacity depletion rate.
        - Crack spread (refinery margin proxy).
        - Crack spread momentum.
        - Days-of-supply estimate.

    Args:
        config: Optional custom configuration.

    Example:
        >>> proc = FundamentalProcessor()
        >>> df = proc.run(data_with_inventories)
        >>> assert "inventory_surprise_zscore" in df.columns
    """

    def __init__(self, config: FundamentalConfig | None = None) -> None:
        super().__init__(config or FundamentalConfig())
        self.fund_config: FundamentalConfig = self.config  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute all fundamental features.

        Args:
            df: Input DataFrame with inventory and price columns.

        Returns:
            DataFrame with fundamental features appended.
        """
        df = self._compute_inventory_features(df)
        df = self._compute_spare_capacity(df)
        df = self._compute_crack_spreads(df)
        df = self._compute_days_of_supply(df)
        return df

    def get_feature_names(self) -> list[str]:
        """Return the list of fundamental feature names.

        Returns:
            Feature column names.
        """
        p = self._prefixed
        return [
            p("inventory_surprise_zscore"),
            p("inventory_change"),
            p("inventory_momentum"),
            p("inventory_vs_seasonal"),
            p("opec_spare_ratio"),
            p("spare_capacity_depletion"),
            p("crack_spread_321"),
            p("crack_spread_momentum"),
            p("days_of_supply"),
        ]

    # ------------------------------------------------------------------
    # Inventory features
    # ------------------------------------------------------------------

    def _compute_inventory_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute inventory-related features.

        Args:
            df: DataFrame potentially containing inventory columns.

        Returns:
            DataFrame with inventory features.
        """
        inv_col = self.fund_config.inventory_col
        lookback = self.fund_config.inventory_surprise_lookback

        zscore_col = self._prefixed("inventory_surprise_zscore")
        change_col = self._prefixed("inventory_change")
        mom_col = self._prefixed("inventory_momentum")
        seasonal_col = self._prefixed("inventory_vs_seasonal")

        if inv_col not in df.columns:
            logger.debug(
                "Inventory column '{col}' not found; adding null features",
                col=inv_col,
            )
            return df.with_columns(
                [
                    pl.lit(None).cast(pl.Float64).alias(zscore_col),
                    pl.lit(None).cast(pl.Float64).alias(change_col),
                    pl.lit(None).cast(pl.Float64).alias(mom_col),
                    pl.lit(None).cast(pl.Float64).alias(seasonal_col),
                ]
            )

        # Week-over-week inventory change.
        df = df.with_columns(
            pl.col(inv_col).diff().alias(change_col)
        )

        # Z-score of inventory change relative to rolling distribution.
        # JUSTIFIED: Z-score normalisation enables comparison across
        # different inventory level regimes.
        df = df.with_columns(
            [
                pl.col(change_col)
                .rolling_mean(window_size=lookback)
                .alias("_inv_change_mean"),
                pl.col(change_col)
                .rolling_std(window_size=lookback)
                .alias("_inv_change_std"),
            ]
        )

        df = df.with_columns(
            self._safe_divide(
                pl.col(change_col) - pl.col("_inv_change_mean"),
                pl.col("_inv_change_std"),
            ).alias(zscore_col)
        )

        # Inventory momentum: 4-week rate of change.
        # JUSTIFIED: 4-week captures monthly draw/build trends that
        # are more signal than noise.
        df = df.with_columns(
            pl.col(inv_col).pct_change(n=4).alias(mom_col)
        )

        # Seasonal deviation: current inventory vs. 52-week rolling mean.
        # JUSTIFIED: Commodities exhibit strong seasonality; deviations
        # from seasonal norms are more informative than absolute levels.
        df = df.with_columns(
            self._safe_divide(
                pl.col(inv_col) - pl.col(inv_col).rolling_mean(window_size=52),
                pl.col(inv_col).rolling_mean(window_size=52),
            ).alias(seasonal_col)
        )

        df = df.drop(["_inv_change_mean", "_inv_change_std"])
        return df

    # ------------------------------------------------------------------
    # OPEC spare capacity
    # ------------------------------------------------------------------

    def _compute_spare_capacity(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute OPEC spare capacity features.

        If an ``opec_spare_capacity`` column exists, uses it directly.
        Otherwise, creates a baseline estimate that can be overridden
        by downstream scenario models.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with spare capacity features.
        """
        ratio_col = self._prefixed("opec_spare_ratio")
        depletion_col = self._prefixed("spare_capacity_depletion")

        spare_col = "opec_spare_capacity"
        demand = self.fund_config.global_demand_mbpd
        baseline_spare = self.fund_config.opec_spare_baseline_mbpd

        if spare_col not in df.columns:
            # Use baseline estimate.
            logger.debug(
                "No opec_spare_capacity column; using baseline {base} mbpd",
                base=baseline_spare,
            )
            df = df.with_columns(
                pl.lit(baseline_spare).alias(spare_col)
            )

        # Spare capacity as fraction of global demand.
        # JUSTIFIED: This ratio is the key metric for supply buffer — values
        # below 2% historically trigger risk premiums (Hamilton 2009).
        df = df.with_columns(
            (pl.col(spare_col) / demand).alias(ratio_col)
        )

        # Depletion rate: how fast spare capacity is being consumed.
        # JUSTIFIED: 4-period lookback for rate of change in spare capacity.
        df = df.with_columns(
            pl.col(spare_col).pct_change(n=4).alias(depletion_col)
        )

        return df

    # ------------------------------------------------------------------
    # Crack spreads
    # ------------------------------------------------------------------

    def _compute_crack_spreads(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute refinery crack spread proxies.

        The 3-2-1 crack spread approximates refinery margins:
            Crack = (2 * Gasoline + 1 * Heating Oil - 3 * Crude) / 3

        If product columns are not available, computes a simplified proxy
        using the crude close price's deviation from trend.

        Args:
            df: DataFrame with commodity price columns.

        Returns:
            DataFrame with crack spread features.
        """
        crack_col = self._prefixed("crack_spread_321")
        crack_mom_col = self._prefixed("crack_spread_momentum")

        products = self.fund_config.crack_spread_products
        gasoline_col = products.get("gasoline", "gasoline_close")
        heating_col = products.get("heating_oil", "heating_oil_close")

        # Try to find a crude close column.
        crude_candidates = [
            c for c in df.columns
            if "crude" in c.lower() and "close" in c.lower()
        ]
        crude_col = crude_candidates[0] if crude_candidates else None

        if (
            gasoline_col in df.columns
            and heating_col in df.columns
            and crude_col is not None
        ):
            # JUSTIFIED: 3-2-1 crack spread is the industry standard refinery
            # margin proxy used by the CME Group and most energy analysts.
            df = df.with_columns(
                (
                    (2 * pl.col(gasoline_col) + pl.col(heating_col) - 3 * pl.col(crude_col))
                    / 3.0
                ).alias(crack_col)
            )
            logger.debug("Computed 3-2-1 crack spread from product prices")
        elif crude_col is not None:
            # Simplified proxy: crude price vs. 60-day trend as margin indicator.
            # JUSTIFIED: Refinery margins mean-revert; deviation from trend
            # captures the directional signal without product price data.
            df = df.with_columns(
                (
                    pl.col(crude_col)
                    - pl.col(crude_col).rolling_mean(window_size=60)
                ).alias(crack_col)
            )
            logger.debug("Computed simplified crack spread proxy (trend deviation)")
        else:
            df = df.with_columns(
                pl.lit(None).cast(pl.Float64).alias(crack_col)
            )

        # Crack spread momentum.
        # JUSTIFIED: 10-day momentum in margins signals demand-side shifts.
        df = df.with_columns(
            pl.col(crack_col).pct_change(n=10).alias(crack_mom_col)
        )

        return df

    # ------------------------------------------------------------------
    # Days of supply
    # ------------------------------------------------------------------

    def _compute_days_of_supply(self, df: pl.DataFrame) -> pl.DataFrame:
        """Estimate days of supply from inventory and implied consumption.

        Days of supply = Inventory / (daily consumption rate).

        Args:
            df: DataFrame with inventory column.

        Returns:
            DataFrame with ``days_of_supply`` column.
        """
        dos_col = self._prefixed("days_of_supply")
        inv_col = self.fund_config.inventory_col
        demand = self.fund_config.global_demand_mbpd

        if inv_col not in df.columns:
            return df.with_columns(
                pl.lit(None).cast(pl.Float64).alias(dos_col)
            )

        # JUSTIFIED: Convert inventory (million barrels) to days at current
        # global consumption rate (mbpd).  This is the standard EIA metric.
        df = df.with_columns(
            (pl.col(inv_col) / demand).alias(dos_col)
        )

        return df
