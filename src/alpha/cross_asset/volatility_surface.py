"""Volatility surface signals: VIX/OVX divergence, skew, and kurtosis.

Extracts alpha from the implied volatility surface across equity (VIX),
oil (OVX), and commodity option markets.  Divergences between asset-class
volatilities, skew steepness, and excess kurtosis provide early warning
of stress regime transitions and tail risk repricing relevant to
geopolitical scenarios.

Typical usage::

    signal = VolatilitySurfaceSignal()
    features = signal.compute(vol_data)
    skew = signal.compute_skew_signals(option_data)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from loguru import logger


@dataclass
class VolSurfaceConfig:
    """Configuration for volatility surface signal computation.

    Attributes:
        vix_col: Column for CBOE VIX index.
        ovx_col: Column for CBOE OVX (oil volatility) index.
        gvz_col: Column for CBOE GVZ (gold volatility) index.
        put_iv_col: Column for OTM put implied volatility (25-delta).
        call_iv_col: Column for OTM call implied volatility (25-delta).
        atm_iv_col: Column for at-the-money implied volatility.
        realised_vol_col: Column for realised (historical) volatility.
        lookback: Rolling window for z-score normalisation. 63 trading
            days (~3 months) captures a full quarterly options cycle.
        skew_lookback: Window for skew momentum. 21 trading days aligns
            with the monthly options expiry cycle.
    """

    vix_col: str = "vix"
    ovx_col: str = "ovx"
    gvz_col: str = "gvz"
    put_iv_col: str = "put_25d_iv"
    call_iv_col: str = "call_25d_iv"
    atm_iv_col: str = "atm_iv"
    realised_vol_col: str = "realised_vol_21d"
    lookback: int = 63
    skew_lookback: int = 21


class VolatilitySurfaceSignal:
    """Volatility surface signals for geopolitical risk detection.

    Computes:
    - **VIX/OVX divergence**: When oil vol decouples from equity vol,
      it signals commodity-specific stress (e.g., supply disruption fears).
    - **OVX/GVZ ratio**: Oil-to-gold vol ratio captures flight-to-safety
      dynamics in geopolitical crises.
    - **Volatility risk premium**: Implied minus realised vol, indicating
      market fear relative to actual movements.
    - **Put skew**: 25-delta put vs. ATM spread, measuring demand for
      downside protection.
    - **Risk reversal**: Put-call IV spread, indicating directional
      sentiment tilt.
    - **Kurtosis proxy**: Butterfly spread (wing vol vs. ATM), measuring
      tail risk pricing.

    Args:
        config: Configuration parameters.
    """

    def __init__(self, config: VolSurfaceConfig | None = None) -> None:
        self.config = config or VolSurfaceConfig()
        logger.info(
            "VolatilitySurfaceSignal initialised: lookback={}, skew_lookback={}",
            self.config.lookback,
            self.config.skew_lookback,
        )

    def compute(self, vol_data: pl.DataFrame) -> pl.DataFrame:
        """Compute all volatility surface signals.

        Args:
            vol_data: DataFrame with volatility index columns (VIX, OVX,
                GVZ) and optionally implied/realised volatility columns.

        Returns:
            DataFrame augmented with signal columns.
        """
        result = vol_data.clone()

        # --- VIX/OVX divergence ---
        if self.config.vix_col in result.columns and self.config.ovx_col in result.columns:
            result = self._add_vix_ovx_divergence(result)

        # --- OVX/GVZ ratio ---
        if self.config.ovx_col in result.columns and self.config.gvz_col in result.columns:
            result = self._add_ovx_gvz_ratio(result)

        # --- Volatility risk premium ---
        if (
            self.config.atm_iv_col in result.columns
            and self.config.realised_vol_col in result.columns
        ):
            result = self._add_vol_risk_premium(result)

        # --- Skew and kurtosis signals ---
        if self.config.put_iv_col in result.columns and self.config.atm_iv_col in result.columns:
            result = self._add_skew_signals(result)

        # --- Risk reversal ---
        if (
            self.config.put_iv_col in result.columns
            and self.config.call_iv_col in result.columns
        ):
            result = self._add_risk_reversal(result)

        # --- Kurtosis / butterfly ---
        if all(
            c in result.columns
            for c in [self.config.put_iv_col, self.config.call_iv_col, self.config.atm_iv_col]
        ):
            result = self._add_butterfly_kurtosis(result)

        logger.info(
            "Volatility surface signals computed: {} rows, {} columns",
            result.height,
            result.width,
        )
        return result

    def compute_skew_signals(self, option_data: pl.DataFrame) -> pl.DataFrame:
        """Compute standalone skew analysis.

        Focuses on the shape of the volatility smile at specific strikes
        to extract directional bias and tail risk pricing.

        Args:
            option_data: DataFrame with put/call IV and ATM IV columns.

        Returns:
            DataFrame with skew-specific signal columns.
        """
        required = [self.config.put_iv_col, self.config.call_iv_col, self.config.atm_iv_col]
        self._validate_columns(option_data, required)

        result = option_data.clone()

        # Put skew (negative = steep downside protection demand)
        result = result.with_columns(
            (pl.col(self.config.put_iv_col) - pl.col(self.config.atm_iv_col))
            .alias("put_skew")
        )

        # Call skew (positive = upside demand / squeeze risk)
        result = result.with_columns(
            (pl.col(self.config.call_iv_col) - pl.col(self.config.atm_iv_col))
            .alias("call_skew")
        )

        # Skew asymmetry: put skew minus call skew
        result = result.with_columns(
            (pl.col("put_skew") - pl.col("call_skew")).alias("skew_asymmetry")
        )

        # Skew momentum
        result = result.with_columns(
            (
                pl.col("put_skew")
                - pl.col("put_skew").shift(self.config.skew_lookback)
            ).alias("skew_momentum")
        )

        # Skew z-score
        result = result.with_columns(
            (
                (
                    pl.col("put_skew")
                    - pl.col("put_skew").rolling_mean(window_size=self.config.lookback)
                )
                / pl.col("put_skew")
                .rolling_std(window_size=self.config.lookback)
                .clip(lower_bound=0.001)
            ).alias("skew_zscore")
        )

        return result

    def vol_regime_detector(self, vol_data: pl.DataFrame) -> pl.DataFrame:
        """Detect volatility regime shifts using multiple indicators.

        Args:
            vol_data: DataFrame with VIX and/or OVX columns.

        Returns:
            DataFrame with ``vol_regime`` (low/normal/elevated/crisis)
            and ``regime_transition_prob`` columns.
        """
        result = vol_data.clone()

        vol_col = (
            self.config.ovx_col
            if self.config.ovx_col in result.columns
            else self.config.vix_col
        )

        if vol_col not in result.columns:
            raise ValueError(f"Need at least '{self.config.vix_col}' or '{self.config.ovx_col}' column")

        vol = result[vol_col].to_numpy().astype(np.float64)

        # Compute rolling percentile for regime classification
        regimes = np.full(len(vol), "normal", dtype=object)
        transition_prob = np.zeros(len(vol), dtype=np.float64)

        for i in range(self.config.lookback, len(vol)):
            window = vol[i - self.config.lookback : i]
            valid = window[~np.isnan(window)]
            if len(valid) < 5:
                continue

            percentile = float(np.mean(valid <= vol[i])) * 100

            if percentile < 20:
                regimes[i] = "low"
            elif percentile < 70:
                regimes[i] = "normal"
            elif percentile < 90:
                regimes[i] = "elevated"
            else:
                regimes[i] = "crisis"

            # Transition probability: how fast are we moving toward crisis?
            if i > 0 and not np.isnan(vol[i - 1]):
                rate_of_change = (vol[i] - vol[i - 1]) / max(vol[i - 1], 0.01)
                transition_prob[i] = float(np.clip(rate_of_change * 5, 0, 1))

        result = result.with_columns(
            pl.Series("vol_regime", regimes.tolist()),
            pl.Series("regime_transition_prob", transition_prob),
        )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_vix_ovx_divergence(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add VIX/OVX divergence signals.

        Args:
            df: Working DataFrame.

        Returns:
            DataFrame with divergence columns.
        """
        # Raw ratio
        df = df.with_columns(
            (pl.col(self.config.ovx_col) / pl.col(self.config.vix_col).clip(lower_bound=0.01))
            .alias("ovx_vix_ratio")
        )

        # Z-score of ratio
        df = df.with_columns(
            (
                (
                    pl.col("ovx_vix_ratio")
                    - pl.col("ovx_vix_ratio").rolling_mean(
                        window_size=self.config.lookback
                    )
                )
                / pl.col("ovx_vix_ratio")
                .rolling_std(window_size=self.config.lookback)
                .clip(lower_bound=0.001)
            ).alias("ovx_vix_divergence_zscore")
        )

        # Divergence momentum
        df = df.with_columns(
            (
                pl.col("ovx_vix_ratio")
                - pl.col("ovx_vix_ratio").shift(self.config.skew_lookback)
            ).alias("ovx_vix_momentum")
        )

        return df

    def _add_ovx_gvz_ratio(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add OVX/GVZ flight-to-safety ratio.

        Args:
            df: Working DataFrame.

        Returns:
            DataFrame with OVX/GVZ ratio columns.
        """
        df = df.with_columns(
            (pl.col(self.config.ovx_col) / pl.col(self.config.gvz_col).clip(lower_bound=0.01))
            .alias("ovx_gvz_ratio")
        )

        df = df.with_columns(
            (
                (
                    pl.col("ovx_gvz_ratio")
                    - pl.col("ovx_gvz_ratio").rolling_mean(
                        window_size=self.config.lookback
                    )
                )
                / pl.col("ovx_gvz_ratio")
                .rolling_std(window_size=self.config.lookback)
                .clip(lower_bound=0.001)
            ).alias("ovx_gvz_zscore")
        )

        return df

    def _add_vol_risk_premium(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add volatility risk premium (implied minus realised).

        Args:
            df: Working DataFrame.

        Returns:
            DataFrame with VRP columns.
        """
        df = df.with_columns(
            (
                pl.col(self.config.atm_iv_col)
                - pl.col(self.config.realised_vol_col)
            ).alias("vol_risk_premium")
        )

        df = df.with_columns(
            (
                (
                    pl.col("vol_risk_premium")
                    - pl.col("vol_risk_premium").rolling_mean(
                        window_size=self.config.lookback
                    )
                )
                / pl.col("vol_risk_premium")
                .rolling_std(window_size=self.config.lookback)
                .clip(lower_bound=0.001)
            ).alias("vrp_zscore")
        )

        return df

    def _add_skew_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add put skew signals.

        Args:
            df: Working DataFrame.

        Returns:
            DataFrame with skew columns.
        """
        df = df.with_columns(
            (pl.col(self.config.put_iv_col) - pl.col(self.config.atm_iv_col))
            .alias("put_skew")
        )

        df = df.with_columns(
            (
                (
                    pl.col("put_skew")
                    - pl.col("put_skew").rolling_mean(
                        window_size=self.config.lookback
                    )
                )
                / pl.col("put_skew")
                .rolling_std(window_size=self.config.lookback)
                .clip(lower_bound=0.001)
            ).alias("put_skew_zscore")
        )

        return df

    def _add_risk_reversal(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add risk reversal (put-call skew spread).

        Args:
            df: Working DataFrame.

        Returns:
            DataFrame with risk reversal columns.
        """
        df = df.with_columns(
            (pl.col(self.config.put_iv_col) - pl.col(self.config.call_iv_col))
            .alias("risk_reversal")
        )

        df = df.with_columns(
            (
                (
                    pl.col("risk_reversal")
                    - pl.col("risk_reversal").rolling_mean(
                        window_size=self.config.lookback
                    )
                )
                / pl.col("risk_reversal")
                .rolling_std(window_size=self.config.lookback)
                .clip(lower_bound=0.001)
            ).alias("risk_reversal_zscore")
        )

        return df

    def _add_butterfly_kurtosis(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add butterfly spread as a kurtosis proxy.

        The butterfly spread measures wing vol relative to ATM vol,
        capturing the market's pricing of tail events.

        Args:
            df: Working DataFrame.

        Returns:
            DataFrame with butterfly/kurtosis columns.
        """
        # Butterfly = 0.5 * (Put_25d + Call_25d) - ATM
        df = df.with_columns(
            (
                0.5
                * (pl.col(self.config.put_iv_col) + pl.col(self.config.call_iv_col))
                - pl.col(self.config.atm_iv_col)
            ).alias("butterfly_spread")
        )

        df = df.with_columns(
            (
                (
                    pl.col("butterfly_spread")
                    - pl.col("butterfly_spread").rolling_mean(
                        window_size=self.config.lookback
                    )
                )
                / pl.col("butterfly_spread")
                .rolling_std(window_size=self.config.lookback)
                .clip(lower_bound=0.001)
            ).alias("kurtosis_zscore")
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
