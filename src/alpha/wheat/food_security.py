"""Food security panic signals from MENA wheat importers.

Monitors distress signals from major wheat-importing nations in the Middle
East and North Africa.  GASC (Egypt's General Authority for Supply
Commodities) tender pricing, import tender frequency, and a food riot
index proxy all capture the panic premium that MENA importers embed in
wheat futures during geopolitical crises.

Typical usage::

    signal = FoodSecuritySignal()
    features = signal.compute(tender_data, social_data)
    riot_index = signal.food_riot_index(social_data)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from loguru import logger


@dataclass
class FoodSecurityConfig:
    """Configuration for food security signal computation.

    Attributes:
        gasc_price_col: Column for GASC tender awarded price (USD/tonne).
        benchmark_price_col: Column for benchmark FOB wheat price (USD/tonne).
        tender_frequency_col: Column for GASC tender frequency (tenders/month).
        tender_volume_col: Column for GASC tender volume (thousand tonnes).
        social_unrest_col: Column for social unrest/protest event count
            in MENA wheat importers.
        food_price_inflation_col: Column for MENA food CPI year-over-year.
        lookback: Rolling window for signal normalisation. 12 months
            provides a full seasonal cycle for GASC tender patterns.
        panic_threshold_zscore: Z-score threshold for flagging panic
            buying. 2.0 standard deviations captures roughly the top
            2.5% of observations.
    """

    gasc_price_col: str = "gasc_tender_price"
    benchmark_price_col: str = "fob_wheat_price"
    tender_frequency_col: str = "gasc_tender_count"
    tender_volume_col: str = "gasc_tender_volume_kt"
    social_unrest_col: str = "mena_unrest_events"
    food_price_inflation_col: str = "mena_food_cpi_yoy"
    lookback: int = 12
    panic_threshold_zscore: float = 2.0


class FoodSecuritySignal:
    """Food security panic signals from MENA wheat importing nations.

    Computes:
    - **GASC premium**: Spread between GASC tender awarded price and
      FOB benchmark, reflecting urgency premium.
    - **Tender urgency index**: Combination of tender frequency and volume
      increases indicating panic stockpiling.
    - **Food riot index**: Proxy for social instability driven by food
      prices in MENA countries (composite of unrest events and food
      inflation).
    - **Import dependency stress**: Cross-country vulnerability assessment
      for nations most exposed to wheat import disruptions.
    - **Panic buying composite**: Unified signal combining all sub-signals
      with crisis-regime detection.

    Args:
        config: Configuration parameters.
    """

    # Key MENA wheat importers and approximate annual import volumes (mmt)
    # for import dependency calculations
    MENA_IMPORTERS: dict[str, float] = {
        "egypt": 12.5,
        "algeria": 7.5,
        "morocco": 5.0,
        "iraq": 4.5,
        "saudi_arabia": 3.5,
        "yemen": 3.5,
        "tunisia": 2.5,
        "libya": 2.0,
        "lebanon": 1.5,
        "jordan": 1.2,
    }

    def __init__(self, config: FoodSecurityConfig | None = None) -> None:
        self.config = config or FoodSecurityConfig()
        logger.info(
            "FoodSecuritySignal initialised: panic threshold={:.1f} sigma",
            self.config.panic_threshold_zscore,
        )

    def compute(
        self,
        tender_data: pl.DataFrame,
        social_data: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Compute all food security signals.

        Args:
            tender_data: DataFrame with GASC tender pricing and volume
                columns.  Must include a ``date`` column.
            social_data: Optional DataFrame with social unrest and food
                inflation indicators.

        Returns:
            DataFrame with signal columns:
            ``gasc_premium``, ``gasc_premium_zscore``,
            ``tender_urgency``, ``panic_composite``.

        Raises:
            ValueError: If critical tender columns are missing.
        """
        result = tender_data.clone()

        # --- GASC premium ---
        if (
            self.config.gasc_price_col in result.columns
            and self.config.benchmark_price_col in result.columns
        ):
            result = self._add_gasc_premium(result)

        # --- Tender urgency ---
        if self.config.tender_frequency_col in result.columns:
            result = self._add_tender_urgency(result)

        # --- Food riot index ---
        if social_data is not None:
            riot_idx = self.food_riot_index(social_data)
            if "date" in result.columns and "date" in riot_idx.columns:
                result = result.join(riot_idx, on="date", how="left")

        # --- Panic composite ---
        result = self._compute_panic_composite(result)

        logger.info(
            "Food security signals computed: {} rows",
            result.height,
        )
        return result

    def food_riot_index(self, social_data: pl.DataFrame) -> pl.DataFrame:
        """Compute a food riot risk index from social unrest and inflation data.

        The index combines protest/unrest event counts with food price
        inflation to proxy the probability of food-related social
        instability in MENA importing nations.

        Args:
            social_data: DataFrame with social unrest event counts and
                food price inflation columns.

        Returns:
            DataFrame with ``food_riot_index``, ``riot_momentum``, and
            ``riot_alert`` columns.
        """
        result = social_data.clone()

        has_unrest = self.config.social_unrest_col in result.columns
        has_inflation = self.config.food_price_inflation_col in result.columns

        if not has_unrest and not has_inflation:
            logger.warning(
                "Neither unrest nor inflation columns found; "
                "returning empty riot index"
            )
            result = result.with_columns(
                pl.lit(np.nan).alias("food_riot_index")
            )
            return result

        components: list[pl.Expr] = []

        if has_unrest:
            # Z-score of unrest events
            unrest_z = (
                (
                    pl.col(self.config.social_unrest_col)
                    - pl.col(self.config.social_unrest_col).rolling_mean(
                        window_size=self.config.lookback
                    )
                )
                / pl.col(self.config.social_unrest_col)
                .rolling_std(window_size=self.config.lookback)
                .clip(lower_bound=0.01)
            )
            result = result.with_columns(unrest_z.alias("unrest_zscore"))
            components.append(pl.col("unrest_zscore") * 0.5)

        if has_inflation:
            # Z-score of food inflation
            infl_z = (
                (
                    pl.col(self.config.food_price_inflation_col)
                    - pl.col(self.config.food_price_inflation_col).rolling_mean(
                        window_size=self.config.lookback
                    )
                )
                / pl.col(self.config.food_price_inflation_col)
                .rolling_std(window_size=self.config.lookback)
                .clip(lower_bound=0.01)
            )
            result = result.with_columns(infl_z.alias("food_inflation_zscore"))
            components.append(pl.col("food_inflation_zscore") * 0.5)

        # Composite riot index
        if len(components) == 1:
            riot_expr = components[0] * 2  # Scale up single component
        else:
            riot_expr = sum(components)

        result = result.with_columns(riot_expr.alias("food_riot_index"))

        # Riot momentum
        result = result.with_columns(
            (
                pl.col("food_riot_index")
                - pl.col("food_riot_index").shift(1)
            ).alias("riot_momentum")
        )

        # Alert flag
        result = result.with_columns(
            (pl.col("food_riot_index") > self.config.panic_threshold_zscore)
            .cast(pl.Int8)
            .alias("riot_alert")
        )

        return result

    def gasc_tender_analysis(self, tender_data: pl.DataFrame) -> pl.DataFrame:
        """Detailed analysis of GASC tender pricing behaviour.

        GASC is the world's largest institutional wheat buyer.  Its tender
        behaviour is a leading indicator of import panic: when GASC
        accelerates purchases or accepts higher premiums, it signals
        supply anxiety across the MENA region.

        Args:
            tender_data: DataFrame with GASC tender columns.

        Returns:
            DataFrame with detailed tender analysis columns.
        """
        result = tender_data.clone()

        if self.config.gasc_price_col not in result.columns:
            logger.warning("GASC price column not found")
            return result

        # Tender-over-tender price change
        result = result.with_columns(
            (pl.col(self.config.gasc_price_col).pct_change() * 100)
            .alias("gasc_price_change_pct")
        )

        # Running max (panic ceiling)
        result = result.with_columns(
            pl.col(self.config.gasc_price_col)
            .rolling_max(window_size=self.config.lookback)
            .alias("gasc_rolling_max")
        )

        # Distance from running max (how close to panic levels)
        result = result.with_columns(
            (
                pl.col(self.config.gasc_price_col)
                / pl.col("gasc_rolling_max").clip(lower_bound=0.01)
            ).alias("gasc_proximity_to_max")
        )

        # Volume acceleration (if available)
        if self.config.tender_volume_col in result.columns:
            result = result.with_columns(
                (
                    pl.col(self.config.tender_volume_col)
                    / pl.col(self.config.tender_volume_col)
                    .rolling_mean(window_size=self.config.lookback)
                    .clip(lower_bound=0.01)
                ).alias("volume_acceleration")
            )

        return result

    def import_dependency_stress(
        self,
        supply_data: pl.DataFrame,
        disrupted_sources: list[str] | None = None,
    ) -> dict[str, float]:
        """Assess import dependency stress for MENA nations.

        Calculates which importers are most vulnerable to disruption of
        specific supply sources (e.g., Russia, Ukraine, France).

        Args:
            supply_data: DataFrame with columns ``country``,
                ``supplier``, ``volume_mt`` representing bilateral
                wheat trade flows.
            disrupted_sources: List of supplier countries assumed to be
                disrupted.  Defaults to ``["russia", "ukraine"]``.

        Returns:
            Dictionary mapping MENA importer to vulnerability score (0-1).
        """
        if disrupted_sources is None:
            disrupted_sources = ["russia", "ukraine"]

        self._validate_columns(supply_data, ["country", "supplier", "volume_mt"])

        vulnerability: dict[str, float] = {}

        for country, total_import in self.MENA_IMPORTERS.items():
            country_data = supply_data.filter(
                pl.col("country").str.to_lowercase() == country
            )
            if country_data.height == 0:
                vulnerability[country] = 0.5  # Unknown, moderate default
                continue

            total_flow = float(country_data["volume_mt"].sum())
            if total_flow < 1e-6:
                vulnerability[country] = 0.5
                continue

            disrupted_flow = float(
                country_data.filter(
                    pl.col("supplier").str.to_lowercase().is_in(disrupted_sources)
                )["volume_mt"].sum()
            )

            vulnerability[country] = disrupted_flow / total_flow

        return vulnerability

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_gasc_premium(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add GASC premium signals.

        Args:
            df: Working DataFrame.

        Returns:
            DataFrame with GASC premium columns.
        """
        # Raw premium
        df = df.with_columns(
            (
                pl.col(self.config.gasc_price_col)
                - pl.col(self.config.benchmark_price_col)
            ).alias("gasc_premium")
        )

        # Premium as percentage of benchmark
        df = df.with_columns(
            (
                pl.col("gasc_premium")
                / pl.col(self.config.benchmark_price_col).clip(lower_bound=0.01)
                * 100
            ).alias("gasc_premium_pct")
        )

        # Z-score of premium
        df = df.with_columns(
            (
                (
                    pl.col("gasc_premium")
                    - pl.col("gasc_premium").rolling_mean(
                        window_size=self.config.lookback
                    )
                )
                / pl.col("gasc_premium")
                .rolling_std(window_size=self.config.lookback)
                .clip(lower_bound=0.01)
            ).alias("gasc_premium_zscore")
        )

        return df

    def _add_tender_urgency(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add tender urgency signals.

        Args:
            df: Working DataFrame.

        Returns:
            DataFrame with tender urgency columns.
        """
        col = self.config.tender_frequency_col

        # Tender frequency z-score
        df = df.with_columns(
            (
                (pl.col(col) - pl.col(col).rolling_mean(window_size=self.config.lookback))
                / pl.col(col).rolling_std(window_size=self.config.lookback).clip(lower_bound=0.01)
            ).alias("tender_frequency_zscore")
        )

        # Combine frequency and volume (if available) into urgency index
        if self.config.tender_volume_col in df.columns:
            vol_col = self.config.tender_volume_col
            df = df.with_columns(
                (
                    (pl.col(col) - pl.col(col).rolling_mean(window_size=self.config.lookback))
                    / pl.col(col).rolling_std(window_size=self.config.lookback).clip(lower_bound=0.01)
                    + (pl.col(vol_col) - pl.col(vol_col).rolling_mean(window_size=self.config.lookback))
                    / pl.col(vol_col).rolling_std(window_size=self.config.lookback).clip(lower_bound=0.01)
                ).alias("tender_urgency")
            )
        else:
            df = df.with_columns(
                pl.col("tender_frequency_zscore").alias("tender_urgency")
            )

        return df

    def _compute_panic_composite(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute panic buying composite score.

        Args:
            df: Working DataFrame with individual signals.

        Returns:
            DataFrame with ``panic_composite`` and ``panic_alert`` columns.
        """
        components: dict[str, float] = {
            "gasc_premium_zscore": 0.35,
            "tender_urgency": 0.25,
            "food_riot_index": 0.25,
            "unrest_zscore": 0.15,
        }

        available = {k: v for k, v in components.items() if k in df.columns}
        if not available:
            df = df.with_columns(
                pl.lit(np.nan).alias("panic_composite"),
                pl.lit(0).cast(pl.Int8).alias("panic_alert"),
            )
            return df

        total_weight = sum(available.values())
        normalised = {k: v / total_weight for k, v in available.items()}

        composite_expr = sum(
            pl.col(col) * weight for col, weight in normalised.items()
        )
        df = df.with_columns(composite_expr.alias("panic_composite"))

        # Panic alert
        df = df.with_columns(
            (pl.col("panic_composite") > self.config.panic_threshold_zscore)
            .cast(pl.Int8)
            .alias("panic_alert")
        )

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
