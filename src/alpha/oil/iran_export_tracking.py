"""Iran crude oil export monitoring and dark fleet tracking proxy signals.

Under sanctions, Iranian crude exports are partially obscured through
ship-to-ship transfers, AIS transponder manipulation, and use of aging
"dark fleet" tankers.  This module constructs proxy signals for Iranian
export volumes using observable shipping and pricing anomalies.

Typical usage::

    tracker = IranExportTracker()
    signals = tracker.compute(shipping_data, pricing_data)
    dark_fleet = tracker.dark_fleet_signals(ais_data)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from loguru import logger


@dataclass
class IranExportConfig:
    """Configuration for Iran export tracking signals.

    Attributes:
        iran_discount_col: Column for Iran Light/Heavy discount to Brent.
        vlcc_rate_col: Column for VLCC (Very Large Crude Carrier) spot rates.
        persian_gulf_traffic_col: Column for Persian Gulf vessel traffic index.
        lookback_short: Short-term rolling window for momentum signals.
            21 trading days approximates one calendar month.
        lookback_long: Long-term rolling window for baseline normalisation.
            63 trading days approximates one quarter.
        dark_fleet_age_threshold: Vessel age (years) above which a tanker
            is considered potential dark fleet. 15 years aligns with
            typical tanker economic life and observed sanctions-evasion
            fleet profiles.
        ais_gap_hours: AIS signal gap (hours) that flags potential
            transponder manipulation. 48 hours exceeds normal port or
            congestion-related gaps.
    """

    iran_discount_col: str = "iran_discount"
    vlcc_rate_col: str = "vlcc_rate"
    persian_gulf_traffic_col: str = "pg_traffic_index"
    lookback_short: int = 21
    lookback_long: int = 63
    dark_fleet_age_threshold: int = 15
    ais_gap_hours: float = 48.0


class IranExportTracker:
    """Proxy signals for Iranian crude oil export volumes.

    Constructs a multi-factor composite from:

    1. **Iran crude discount**: Discount of Iranian grades to benchmarks
       widens when sanctions enforcement tightens (lower volumes) and
       narrows when evasion increases (higher volumes).
    2. **Dark fleet activity**: Count and movements of suspected dark
       fleet vessels in the Persian Gulf region.
    3. **VLCC rate anomalies**: Unusual tanker rate spikes or dips in the
       Middle East Gulf region correlated with sanctioned-cargo routing.
    4. **AIS gap analysis**: Frequency and duration of AIS transponder
       gaps for vessels transiting Iranian waters.

    Args:
        config: Configuration for signal parameters.
    """

    def __init__(self, config: IranExportConfig | None = None) -> None:
        self.config = config or IranExportConfig()
        logger.info(
            "IranExportTracker initialised: dark fleet age threshold={} yrs, "
            "AIS gap threshold={} hrs",
            self.config.dark_fleet_age_threshold,
            self.config.ais_gap_hours,
        )

    def compute(
        self,
        shipping_data: pl.DataFrame,
        pricing_data: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compute composite Iran export proxy signals.

        Args:
            shipping_data: DataFrame with shipping/traffic columns
                (``pg_traffic_index``, ``vlcc_rate``).
            pricing_data: DataFrame with Iran crude discount column.

        Returns:
            DataFrame with computed signal columns:
            ``discount_zscore``, ``vlcc_anomaly``, ``traffic_momentum``,
            ``export_proxy_composite``.

        Raises:
            ValueError: If required columns are missing.
        """
        # Validate inputs
        self._validate_columns(
            shipping_data,
            [self.config.vlcc_rate_col, self.config.persian_gulf_traffic_col],
        )
        self._validate_columns(pricing_data, [self.config.iran_discount_col])

        # --- Iran discount z-score ---
        discount = pricing_data[self.config.iran_discount_col].to_numpy().astype(np.float64)
        discount_zscore = self._rolling_zscore(
            discount, self.config.lookback_long
        )

        # --- VLCC rate anomaly ---
        vlcc = shipping_data[self.config.vlcc_rate_col].to_numpy().astype(np.float64)
        vlcc_zscore = self._rolling_zscore(vlcc, self.config.lookback_long)
        # Anomaly flag: absolute z-score > 2 standard deviations
        vlcc_anomaly = np.where(np.abs(vlcc_zscore) > 2.0, vlcc_zscore, 0.0)

        # --- Traffic momentum ---
        traffic = shipping_data[self.config.persian_gulf_traffic_col].to_numpy().astype(np.float64)
        traffic_momentum = self._momentum(
            traffic, self.config.lookback_short, self.config.lookback_long
        )

        # --- Composite export proxy ---
        # Higher discount z-score = tighter sanctions = lower exports
        # Higher traffic momentum = more ships = potentially higher exports
        # Anomalous VLCC rates = potential sanction evasion activity
        n = min(len(discount_zscore), len(vlcc_anomaly), len(traffic_momentum))
        composite = (
            -0.4 * discount_zscore[:n]
            + 0.35 * traffic_momentum[:n]
            + 0.25 * vlcc_anomaly[:n]
        )

        result = pl.DataFrame({
            "discount_zscore": discount_zscore[:n],
            "vlcc_anomaly": vlcc_anomaly[:n],
            "traffic_momentum": traffic_momentum[:n],
            "export_proxy_composite": composite,
        })

        logger.info(
            "Iran export proxy computed: {} rows, composite mean={:.3f}, std={:.3f}",
            result.height,
            float(np.nanmean(composite)),
            float(np.nanstd(composite)),
        )
        return result

    def dark_fleet_signals(self, ais_data: pl.DataFrame) -> pl.DataFrame:
        """Compute dark fleet activity indicators from AIS data.

        Args:
            ais_data: DataFrame with columns:
                - ``vessel_age_years``: Age of the vessel in years.
                - ``ais_gap_hours``: Maximum AIS gap duration in hours.
                - ``flag_state``: Flag state of the vessel.
                - ``date``: Observation date.
                - ``vessel_id``: Unique vessel identifier.

        Returns:
            Daily aggregated DataFrame with dark fleet signals:
            ``dark_fleet_count``, ``avg_ais_gap``, ``flag_change_count``,
            ``dark_fleet_intensity``.

        Raises:
            ValueError: If required columns are missing.
        """
        required = ["vessel_age_years", "ais_gap_hours", "date", "vessel_id"]
        self._validate_columns(ais_data, required)

        # Flag vessels as potential dark fleet
        flagged = ais_data.with_columns(
            (
                (pl.col("vessel_age_years") >= self.config.dark_fleet_age_threshold)
                | (pl.col("ais_gap_hours") >= self.config.ais_gap_hours)
            )
            .cast(pl.Int8)
            .alias("is_dark_fleet")
        )

        # Daily aggregation
        daily = flagged.group_by("date").agg(
            pl.col("is_dark_fleet").sum().alias("dark_fleet_count"),
            pl.col("vessel_id").n_unique().alias("total_vessels"),
            pl.col("ais_gap_hours").mean().alias("avg_ais_gap"),
            pl.col("ais_gap_hours").max().alias("max_ais_gap"),
        ).sort("date")

        # Dark fleet intensity: ratio of suspected dark fleet to total vessels
        daily = daily.with_columns(
            (pl.col("dark_fleet_count") / pl.col("total_vessels").clip(lower_bound=1))
            .alias("dark_fleet_intensity")
        )

        logger.info(
            "Dark fleet signals computed: {} days, avg dark fleet count={:.1f}",
            daily.height,
            float(daily["dark_fleet_count"].mean()),  # type: ignore[arg-type]
        )
        return daily

    def sanctioned_flow_estimate(
        self,
        dark_fleet_df: pl.DataFrame,
        avg_cargo_mb: float = 2.0,
    ) -> pl.DataFrame:
        """Estimate daily sanctioned crude flow from dark fleet activity.

        Uses the dark fleet vessel count and average cargo size to
        approximate barrels of sanctioned crude transiting per day.

        Args:
            dark_fleet_df: Output from :meth:`dark_fleet_signals`.
            avg_cargo_mb: Average cargo per dark fleet voyage in million
                barrels.  2.0 mb is a VLCC standard load.

        Returns:
            DataFrame with ``estimated_flow_mbd`` (million barrels/day)
            and ``flow_momentum`` columns.
        """
        result = dark_fleet_df.with_columns(
            (pl.col("dark_fleet_count") * avg_cargo_mb / 30.0)
            .alias("estimated_flow_mbd")
        )

        # Add momentum: short-term vs long-term moving average
        result = result.with_columns(
            (
                pl.col("estimated_flow_mbd").rolling_mean(window_size=self.config.lookback_short)
                - pl.col("estimated_flow_mbd").rolling_mean(window_size=self.config.lookback_long)
            ).alias("flow_momentum")
        )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_zscore(
        values: np.ndarray[Any, np.dtype[np.float64]],
        window: int,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Compute rolling z-score.

        Args:
            values: Input array.
            window: Rolling window size.

        Returns:
            Z-score array (NaN-filled for initial window).
        """
        result = np.full_like(values, np.nan)
        for i in range(window, len(values)):
            segment = values[i - window : i]
            mu = np.nanmean(segment)
            sigma = np.nanstd(segment)
            if sigma > 1e-10:
                result[i] = (values[i] - mu) / sigma
        return result

    @staticmethod
    def _momentum(
        values: np.ndarray[Any, np.dtype[np.float64]],
        short_window: int,
        long_window: int,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Compute momentum as short MA minus long MA, normalised by long MA.

        Args:
            values: Input array.
            short_window: Short-term moving average window.
            long_window: Long-term moving average window.

        Returns:
            Momentum array.
        """
        result = np.full_like(values, np.nan)
        for i in range(long_window, len(values)):
            short_ma = np.nanmean(values[i - short_window : i])
            long_ma = np.nanmean(values[i - long_window : i])
            if abs(long_ma) > 1e-10:
                result[i] = (short_ma - long_ma) / abs(long_ma)
        return result

    @staticmethod
    def _validate_columns(df: pl.DataFrame, required: list[str]) -> None:
        """Check that required columns exist in a DataFrame.

        Args:
            df: DataFrame to check.
            required: List of required column names.

        Raises:
            ValueError: If any required columns are missing.
        """
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
