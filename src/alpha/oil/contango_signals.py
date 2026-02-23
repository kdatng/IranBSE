"""Oil futures term structure signals: contango, backwardation, and roll yield.

Extracts alpha from the shape of the crude oil futures curve.  Contango
(upward-sloping curve) signals storage economics and oversupply; backwardation
(downward-sloping) signals spot tightness, often associated with supply
disruption or geopolitical risk.  Roll yield and term structure convexity
provide additional insights into market expectations.

Typical usage::

    signal = ContangoSignal(lookback=60, front_col="CL1", second_col="CL2")
    features = signal.compute(futures_data)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from loguru import logger


@dataclass
class ContangoConfig:
    """Configuration for contango signal computation.

    Attributes:
        front_col: Column name for the front-month futures price.
        second_col: Column name for the second-month futures price.
        third_col: Column name for the third-month futures price (optional,
            for convexity).
        lookback: Rolling window for z-score normalisation. 60 trading days
            (~3 months) captures a full quarterly roll cycle.
        annualisation_factor: Factor to annualise roll yield (252 trading
            days / ~21 days between monthly rolls).
    """

    front_col: str = "CL1"
    second_col: str = "CL2"
    third_col: str | None = "CL3"
    lookback: int = 60
    annualisation_factor: float = 12.0


class ContangoSignal:
    """Term structure alpha signals for crude oil futures.

    Computes:
    - **Spread**: Raw difference between front and second month.
    - **Contango/backwardation flag**: Binary regime indicator.
    - **Roll yield**: Annualised carry from holding the front contract.
    - **Term structure convexity**: Curvature of the 3-contract curve,
      indicating whether the curve bows up (convex) or down (concave).
    - **Z-score normalisation**: Spread relative to recent history for
      cross-regime comparability.

    Args:
        config: Configuration parameters for the signal.
    """

    def __init__(self, config: ContangoConfig | None = None) -> None:
        self.config = config or ContangoConfig()
        logger.info(
            "ContangoSignal initialised: front={}, second={}, lookback={}",
            self.config.front_col,
            self.config.second_col,
            self.config.lookback,
        )

    def compute(self, data: pl.DataFrame) -> pl.DataFrame:
        """Compute all term structure signals.

        Args:
            data: DataFrame with at least front and second month price
                columns.  Must be sorted by date ascending.

        Returns:
            DataFrame augmented with signal columns:
            ``spread``, ``contango_flag``, ``roll_yield``,
            ``spread_zscore``, and optionally ``convexity``.

        Raises:
            ValueError: If required price columns are missing.
        """
        required = [self.config.front_col, self.config.second_col]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        result = data.clone()

        # --- Raw spread ---
        result = result.with_columns(
            (pl.col(self.config.second_col) - pl.col(self.config.front_col))
            .alias("spread")
        )

        # --- Contango flag (1 = contango, 0 = backwardation) ---
        result = result.with_columns(
            (pl.col("spread") > 0).cast(pl.Int8).alias("contango_flag")
        )

        # --- Roll yield (annualised) ---
        result = result.with_columns(
            (
                (pl.col(self.config.front_col) - pl.col(self.config.second_col))
                / pl.col(self.config.second_col)
                * self.config.annualisation_factor
            ).alias("roll_yield")
        )

        # --- Z-score of spread ---
        result = self._add_zscore(result, "spread", self.config.lookback)

        # --- Convexity (butterfly spread) ---
        if (
            self.config.third_col is not None
            and self.config.third_col in data.columns
        ):
            result = result.with_columns(
                (
                    pl.col(self.config.front_col)
                    - 2 * pl.col(self.config.second_col)
                    + pl.col(self.config.third_col)
                ).alias("convexity")
            )
            result = self._add_zscore(result, "convexity", self.config.lookback)

        logger.info(
            "ContangoSignal computed {} rows, contango ratio: {:.1%}",
            result.height,
            float(result["contango_flag"].mean()),  # type: ignore[arg-type]
        )
        return result

    def compute_roll_yield(self, data: pl.DataFrame) -> pl.Series:
        """Compute annualised roll yield as a standalone series.

        Args:
            data: DataFrame with front and second month price columns.

        Returns:
            Polars Series named ``roll_yield``.
        """
        front = data[self.config.front_col].to_numpy().astype(np.float64)
        second = data[self.config.second_col].to_numpy().astype(np.float64)
        roll = (front - second) / np.where(
            np.abs(second) > 1e-8, second, np.nan
        ) * self.config.annualisation_factor
        return pl.Series("roll_yield", roll)

    def compute_contango_duration(self, data: pl.DataFrame) -> pl.Series:
        """Count consecutive days in contango or backwardation.

        Positive values indicate days in contango; negative values indicate
        days in backwardation.

        Args:
            data: DataFrame with front and second month price columns.

        Returns:
            Polars Series named ``contango_duration``.
        """
        front = data[self.config.front_col].to_numpy().astype(np.float64)
        second = data[self.config.second_col].to_numpy().astype(np.float64)
        is_contango = second > front

        duration = np.zeros(len(is_contango), dtype=np.float64)
        count = 0.0
        for i in range(len(is_contango)):
            if is_contango[i]:
                count = max(count, 0) + 1
            else:
                count = min(count, 0) - 1
            duration[i] = count

        return pl.Series("contango_duration", duration)

    def compute_term_structure_slope(self, data: pl.DataFrame) -> pl.Series:
        """Compute the normalised slope of the term structure.

        Slope is expressed as a percentage of the front-month price for
        cross-period comparability.

        Args:
            data: DataFrame with front and second month price columns.

        Returns:
            Polars Series named ``ts_slope``.
        """
        front = data[self.config.front_col].to_numpy().astype(np.float64)
        second = data[self.config.second_col].to_numpy().astype(np.float64)
        slope = (second - front) / np.where(
            np.abs(front) > 1e-8, front, np.nan
        ) * 100.0
        return pl.Series("ts_slope", slope)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_zscore(
        df: pl.DataFrame,
        column: str,
        window: int,
    ) -> pl.DataFrame:
        """Add a rolling z-score column for the given signal.

        Args:
            df: Input DataFrame.
            column: Column to z-score.
            window: Rolling window size.

        Returns:
            DataFrame with ``{column}_zscore`` appended.
        """
        return df.with_columns(
            (
                (pl.col(column) - pl.col(column).rolling_mean(window_size=window))
                / pl.col(column).rolling_std(window_size=window)
            ).alias(f"{column}_zscore")
        )
