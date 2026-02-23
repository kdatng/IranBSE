"""OPEC spare capacity and compliance positioning signals.

Estimates OPEC's effective spare production capacity and tracks compliance
deviations from production quotas.  Spare capacity is a critical buffer
against supply disruptions: when spare capacity is thin, any disruption
(e.g., Iran conflict) has outsized price impact.

Typical usage::

    opec = OPECPositioning()
    signals = opec.compute(production_data, quota_data)
    spare = opec.spare_capacity_estimate(production_data)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from loguru import logger


@dataclass
class OPECConfig:
    """Configuration for OPEC positioning signals.

    Attributes:
        production_col: Column for actual OPEC production (mbd).
        quota_col: Column for OPEC quota/target (mbd).
        capacity_col: Column for estimated maximum capacity (mbd).
        saudi_production_col: Column for Saudi Arabia production (key swing
            producer).
        lookback: Rolling window for momentum calculations. 12 months
            captures a full OPEC production cycle.
        compliance_threshold: Deviation (fraction) beyond which
            non-compliance is flagged. 0.03 (3%) is a common materiality
            threshold in OPEC monitoring.
    """

    production_col: str = "opec_production_mbd"
    quota_col: str = "opec_quota_mbd"
    capacity_col: str = "opec_capacity_mbd"
    saudi_production_col: str = "saudi_production_mbd"
    lookback: int = 12
    compliance_threshold: float = 0.03


class OPECPositioning:
    """OPEC spare capacity and compliance deviation signals.

    Computes:
    - **Spare capacity**: Difference between estimated max capacity and
      actual production, normalised by global demand.
    - **Compliance ratio**: Actual production vs. agreed quota.
    - **Compliance deviation momentum**: Rate of change in compliance,
      indicating tightening or loosening discipline.
    - **Saudi swing capacity**: Saudi Arabia's remaining ramp-up room
      (most responsive swing producer).
    - **Effective spare buffer**: Spare capacity adjusted for maintenance,
      political constraints, and Iran's sanctioned capacity.

    Args:
        config: Configuration parameters.
    """

    def __init__(self, config: OPECConfig | None = None) -> None:
        self.config = config or OPECConfig()
        logger.info(
            "OPECPositioning initialised: compliance threshold={:.1%}",
            self.config.compliance_threshold,
        )

    def compute(
        self,
        production_data: pl.DataFrame,
        quota_data: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Compute all OPEC positioning signals.

        Args:
            production_data: Monthly DataFrame with production and capacity
                columns.  Must include a ``date`` column sorted ascending.
            quota_data: Optional DataFrame with quota/target columns.  If
                ``None``, compliance signals are skipped.

        Returns:
            DataFrame with signal columns:
            ``spare_capacity_mbd``, ``spare_capacity_pct``,
            ``compliance_ratio``, ``compliance_momentum``,
            ``spare_capacity_zscore``.

        Raises:
            ValueError: If required production columns are missing.
        """
        required = [self.config.production_col, self.config.capacity_col]
        self._validate_columns(production_data, required)

        result = production_data.clone()

        # --- Spare capacity ---
        result = result.with_columns(
            (pl.col(self.config.capacity_col) - pl.col(self.config.production_col))
            .alias("spare_capacity_mbd")
        )

        result = result.with_columns(
            (
                pl.col("spare_capacity_mbd")
                / pl.col(self.config.capacity_col).clip(lower_bound=0.01)
            ).alias("spare_capacity_pct")
        )

        # --- Spare capacity z-score ---
        result = result.with_columns(
            (
                (
                    pl.col("spare_capacity_mbd")
                    - pl.col("spare_capacity_mbd").rolling_mean(
                        window_size=self.config.lookback
                    )
                )
                / pl.col("spare_capacity_mbd").rolling_std(
                    window_size=self.config.lookback
                )
            ).alias("spare_capacity_zscore")
        )

        # --- Compliance signals ---
        if quota_data is not None and self.config.quota_col in quota_data.columns:
            result = self._add_compliance_signals(result, quota_data)

        # --- Saudi swing capacity ---
        if self.config.saudi_production_col in production_data.columns:
            result = self._add_saudi_signals(result)

        logger.info(
            "OPEC positioning computed: {} rows, mean spare={:.2f} mbd",
            result.height,
            float(result["spare_capacity_mbd"].mean()),  # type: ignore[arg-type]
        )
        return result

    def spare_capacity_estimate(
        self,
        production_data: pl.DataFrame,
        iran_sanctioned_capacity_mbd: float = 1.5,
    ) -> pl.DataFrame:
        """Estimate effective spare capacity with Iran scenario adjustment.

        In a conflict scenario, Iran's production is removed from both
        actual production and available capacity, shrinking the global
        buffer.

        Args:
            production_data: Monthly production/capacity DataFrame.
            iran_sanctioned_capacity_mbd: Iran's estimated capacity at
                risk in a conflict scenario.  1.5 mbd reflects a scenario
                where Iran's ~3.2 mbd capacity is roughly halved by
                sanctions enforcement and infrastructure disruption.

        Returns:
            DataFrame with ``effective_spare_mbd`` and scenario-adjusted
            columns.
        """
        self._validate_columns(
            production_data,
            [self.config.production_col, self.config.capacity_col],
        )

        result = production_data.clone()

        # Baseline spare
        result = result.with_columns(
            (pl.col(self.config.capacity_col) - pl.col(self.config.production_col))
            .alias("baseline_spare_mbd")
        )

        # Iran-disruption adjusted: remove Iran's capacity from both sides
        result = result.with_columns(
            (pl.col("baseline_spare_mbd") - pl.lit(iran_sanctioned_capacity_mbd))
            .clip(lower_bound=0.0)
            .alias("effective_spare_mbd")
        )

        # Cushion ratio: effective spare as fraction of global demand proxy
        result = result.with_columns(
            (
                pl.col("effective_spare_mbd")
                / pl.col(self.config.production_col).clip(lower_bound=0.01)
            ).alias("cushion_ratio")
        )

        # Critical threshold flag: spare < 2 mbd is historically tight
        result = result.with_columns(
            (pl.col("effective_spare_mbd") < 2.0)
            .cast(pl.Int8)
            .alias("spare_critical_flag")
        )

        return result

    def compliance_deviation_momentum(
        self,
        production_data: pl.DataFrame,
        quota_data: pl.DataFrame,
    ) -> pl.Series:
        """Compute the rate of change in OPEC compliance deviation.

        Accelerating non-compliance suggests weakening cartel discipline,
        typically bearish for oil prices.  Tightening compliance is bullish.

        Args:
            production_data: Production DataFrame with date column.
            quota_data: Quota DataFrame with date and quota columns.

        Returns:
            Series named ``compliance_momentum`` representing month-over-month
            change in the compliance ratio.
        """
        self._validate_columns(production_data, [self.config.production_col])
        self._validate_columns(quota_data, [self.config.quota_col])

        prod = production_data[self.config.production_col].to_numpy().astype(np.float64)
        quota = quota_data[self.config.quota_col].to_numpy().astype(np.float64)

        n = min(len(prod), len(quota))
        compliance = prod[:n] / np.where(quota[:n] > 0.01, quota[:n], np.nan)

        # Month-over-month momentum
        momentum = np.full(n, np.nan, dtype=np.float64)
        for i in range(1, n):
            if not (np.isnan(compliance[i]) or np.isnan(compliance[i - 1])):
                momentum[i] = compliance[i] - compliance[i - 1]

        return pl.Series("compliance_momentum", momentum)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_compliance_signals(
        self,
        result: pl.DataFrame,
        quota_data: pl.DataFrame,
    ) -> pl.DataFrame:
        """Add compliance ratio and momentum columns.

        Args:
            result: Working DataFrame.
            quota_data: Quota DataFrame to join.

        Returns:
            DataFrame with compliance columns added.
        """
        # Assume both DataFrames share a date column or are aligned
        quota_col = self.config.quota_col
        n = min(result.height, quota_data.height)

        quota_values = quota_data[quota_col][:n].to_numpy().astype(np.float64)
        prod_values = result[self.config.production_col][:n].to_numpy().astype(np.float64)

        compliance = prod_values / np.where(
            quota_values > 0.01, quota_values, np.nan
        )
        deviation = compliance - 1.0

        # Build padded arrays if lengths differ
        full_compliance = np.full(result.height, np.nan)
        full_deviation = np.full(result.height, np.nan)
        full_compliance[:n] = compliance
        full_deviation[:n] = deviation

        result = result.with_columns(
            pl.Series("compliance_ratio", full_compliance),
            pl.Series("compliance_deviation", full_deviation),
        )

        # Momentum: rolling change in compliance
        result = result.with_columns(
            (
                pl.col("compliance_ratio")
                - pl.col("compliance_ratio").shift(1)
            ).alias("compliance_momentum")
        )

        # Non-compliance flag
        result = result.with_columns(
            (pl.col("compliance_deviation").abs() > self.config.compliance_threshold)
            .cast(pl.Int8)
            .alias("non_compliance_flag")
        )

        return result

    def _add_saudi_signals(self, result: pl.DataFrame) -> pl.DataFrame:
        """Add Saudi swing producer signals.

        Args:
            result: Working DataFrame.

        Returns:
            DataFrame with Saudi-specific columns added.
        """
        # Saudi max capacity estimate (approximately 12.5 mbd nameplate)
        saudi_max_capacity = 12.5

        result = result.with_columns(
            (pl.lit(saudi_max_capacity) - pl.col(self.config.saudi_production_col))
            .alias("saudi_spare_mbd")
        )

        result = result.with_columns(
            (
                pl.col("saudi_spare_mbd")
                / pl.lit(saudi_max_capacity)
            ).alias("saudi_utilisation_room")
        )

        return result

    @staticmethod
    def _validate_columns(df: pl.DataFrame, required: list[str]) -> None:
        """Validate that required columns exist.

        Args:
            df: DataFrame to check.
            required: Required column names.

        Raises:
            ValueError: If columns are missing.
        """
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
