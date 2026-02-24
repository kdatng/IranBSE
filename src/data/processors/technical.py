"""Technical indicator processor.

Computes standard and advanced technical indicators for commodity price
series: RSI, MACD, Bollinger Bands, ATR, and volume-weighted features.
All computations are vectorised using Polars expressions and NumPy —
no row-level loops.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from pydantic import Field

from src.data.processors.base_processor import BaseProcessor, ProcessorConfig

# ---------------------------------------------------------------------------
# Default parameters — all JUSTIFIED with standard references
# ---------------------------------------------------------------------------
# JUSTIFIED: RSI(14) is the original Wilder (1978) parameterisation and the
# most widely used in both academic and practitioner literature.
DEFAULT_RSI_PERIOD: int = 14

# JUSTIFIED: MACD(12,26,9) is the original Gerald Appel parameterisation.
DEFAULT_MACD_FAST: int = 12
DEFAULT_MACD_SLOW: int = 26
DEFAULT_MACD_SIGNAL: int = 9

# JUSTIFIED: Bollinger(20,2) uses a 20-day SMA with 2-sigma bands — the
# standard John Bollinger recommendation.
DEFAULT_BB_PERIOD: int = 20
DEFAULT_BB_STD: float = 2.0

# JUSTIFIED: ATR(14) matches the Wilder (1978) original parameterisation.
DEFAULT_ATR_PERIOD: int = 14


class TechnicalConfig(ProcessorConfig):
    """Configuration for the technical indicator processor.

    Attributes:
        rsi_period: Lookback period for RSI calculation.
        macd_fast: Fast EMA period for MACD.
        macd_slow: Slow EMA period for MACD.
        macd_signal: Signal line EMA period for MACD.
        bb_period: SMA period for Bollinger Bands.
        bb_std: Standard deviation multiplier for Bollinger Bands.
        atr_period: Lookback period for Average True Range.
        close_col: Name of the close price column to process.
        high_col: Name of the high price column.
        low_col: Name of the low price column.
        volume_col: Name of the volume column.
    """

    name: str = "technical_indicators"
    rsi_period: int = Field(default=DEFAULT_RSI_PERIOD, ge=2)
    macd_fast: int = Field(default=DEFAULT_MACD_FAST, ge=2)
    macd_slow: int = Field(default=DEFAULT_MACD_SLOW, ge=2)
    macd_signal: int = Field(default=DEFAULT_MACD_SIGNAL, ge=2)
    bb_period: int = Field(default=DEFAULT_BB_PERIOD, ge=2)
    bb_std: float = Field(default=DEFAULT_BB_STD, gt=0)
    atr_period: int = Field(default=DEFAULT_ATR_PERIOD, ge=2)
    close_col: str = Field(default="close")
    high_col: str = Field(default="high")
    low_col: str = Field(default="low")
    volume_col: str = Field(default="volume")


class TechnicalProcessor(BaseProcessor):
    """Computes technical analysis indicators for commodity futures.

    All indicators are computed using vectorised Polars/NumPy operations
    for maximum performance on large datasets.

    Indicators produced:
        - RSI (Relative Strength Index)
        - MACD (line, signal, histogram)
        - Bollinger Bands (upper, lower, bandwidth, %B)
        - ATR (Average True Range)
        - VWAP-like features (volume-weighted close, volume ratio)

    Args:
        config: Optional custom configuration.

    Example:
        >>> proc = TechnicalProcessor(TechnicalConfig(close_col="wti_crude_close"))
        >>> result = proc.run(price_df)
        >>> assert "rsi" in result.columns
    """

    def __init__(self, config: TechnicalConfig | None = None) -> None:
        super().__init__(config or TechnicalConfig())
        self.tech_config: TechnicalConfig = self.config  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute all technical indicators and append to the DataFrame.

        Args:
            df: Input DataFrame with OHLCV columns.

        Returns:
            DataFrame with technical indicator columns appended.
        """
        close = self.tech_config.close_col
        high = self.tech_config.high_col
        low = self.tech_config.low_col
        volume = self.tech_config.volume_col

        # Verify at minimum the close column exists.
        if close not in df.columns:
            logger.warning(
                "Close column '{col}' not found; skipping technical indicators",
                col=close,
            )
            return df

        # RSI
        df = self._compute_rsi(df, close)

        # MACD
        df = self._compute_macd(df, close)

        # Bollinger Bands
        df = self._compute_bollinger(df, close)

        # ATR (requires high, low, close)
        if high in df.columns and low in df.columns:
            df = self._compute_atr(df, high, low, close)

        # Volume-weighted features
        if volume in df.columns:
            df = self._compute_volume_features(df, close, volume)

        return df

    def get_feature_names(self) -> list[str]:
        """Return the list of feature names produced.

        Returns:
            List of technical indicator column names.
        """
        p = self._prefixed
        features = [
            p("rsi"),
            p("macd_line"),
            p("macd_signal"),
            p("macd_histogram"),
            p("bb_upper"),
            p("bb_lower"),
            p("bb_mid"),
            p("bb_bandwidth"),
            p("bb_pct_b"),
            p("atr"),
            p("atr_pct"),
            p("vw_close"),
            p("volume_ratio"),
            p("volume_momentum"),
        ]
        return features

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------

    def _compute_rsi(self, df: pl.DataFrame, close_col: str) -> pl.DataFrame:
        """Compute Relative Strength Index (Wilder's smoothing).

        RSI = 100 - 100 / (1 + RS)
        RS = avg_gain / avg_loss over ``rsi_period`` bars.

        Uses exponential moving average for smoothing (Wilder's method),
        not simple moving average.

        Args:
            df: DataFrame with close price column.
            close_col: Name of the close column.

        Returns:
            DataFrame with ``rsi`` column appended.
        """
        period = self.tech_config.rsi_period
        col_name = self._prefixed("rsi")

        # Price changes.
        delta = pl.col(close_col).diff()
        gain = pl.when(delta > 0).then(delta).otherwise(0.0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0.0)

        # Wilder's smoothing uses alpha = 1/period.
        # JUSTIFIED: This matches Wilder (1978) original RSI definition.
        alpha = 1.0 / period

        df = df.with_columns(
            [
                gain.alias("_rsi_gain"),
                loss.alias("_rsi_loss"),
            ]
        )

        df = df.with_columns(
            [
                pl.col("_rsi_gain").ewm_mean(alpha=alpha).alias("_avg_gain"),
                pl.col("_rsi_loss").ewm_mean(alpha=alpha).alias("_avg_loss"),
            ]
        )

        df = df.with_columns(
            (
                100.0
                - 100.0
                / (
                    1.0
                    + self._safe_divide(
                        pl.col("_avg_gain"), pl.col("_avg_loss"), default=1.0
                    )
                )
            ).alias(col_name)
        )

        # Clean up intermediate columns.
        df = df.drop(["_rsi_gain", "_rsi_loss", "_avg_gain", "_avg_loss"])
        return df

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------

    def _compute_macd(self, df: pl.DataFrame, close_col: str) -> pl.DataFrame:
        """Compute MACD line, signal line, and histogram.

        MACD Line = EMA(fast) - EMA(slow)
        Signal = EMA(MACD Line, signal_period)
        Histogram = MACD Line - Signal

        Args:
            df: DataFrame with close price column.
            close_col: Name of the close column.

        Returns:
            DataFrame with MACD columns appended.
        """
        fast = self.tech_config.macd_fast
        slow = self.tech_config.macd_slow
        sig = self.tech_config.macd_signal

        macd_col = self._prefixed("macd_line")
        signal_col = self._prefixed("macd_signal")
        hist_col = self._prefixed("macd_histogram")

        # JUSTIFIED: Standard EMA alpha = 2/(period+1)
        fast_alpha = 2.0 / (fast + 1)
        slow_alpha = 2.0 / (slow + 1)
        sig_alpha = 2.0 / (sig + 1)

        df = df.with_columns(
            [
                pl.col(close_col).ewm_mean(alpha=fast_alpha).alias("_ema_fast"),
                pl.col(close_col).ewm_mean(alpha=slow_alpha).alias("_ema_slow"),
            ]
        )

        df = df.with_columns(
            (pl.col("_ema_fast") - pl.col("_ema_slow")).alias(macd_col)
        )

        df = df.with_columns(
            pl.col(macd_col).ewm_mean(alpha=sig_alpha).alias(signal_col)
        )

        df = df.with_columns(
            (pl.col(macd_col) - pl.col(signal_col)).alias(hist_col)
        )

        df = df.drop(["_ema_fast", "_ema_slow"])
        return df

    # ------------------------------------------------------------------
    # Bollinger Bands
    # ------------------------------------------------------------------

    def _compute_bollinger(self, df: pl.DataFrame, close_col: str) -> pl.DataFrame:
        """Compute Bollinger Bands with bandwidth and %B.

        Upper = SMA + k * std
        Lower = SMA - k * std
        Bandwidth = (Upper - Lower) / SMA
        %B = (Price - Lower) / (Upper - Lower)

        Args:
            df: DataFrame with close price column.
            close_col: Name of the close column.

        Returns:
            DataFrame with Bollinger Band columns appended.
        """
        period = self.tech_config.bb_period
        k = self.tech_config.bb_std

        mid_col = self._prefixed("bb_mid")
        upper_col = self._prefixed("bb_upper")
        lower_col = self._prefixed("bb_lower")
        bw_col = self._prefixed("bb_bandwidth")
        pctb_col = self._prefixed("bb_pct_b")

        df = df.with_columns(
            [
                pl.col(close_col).rolling_mean(window_size=period).alias(mid_col),
                pl.col(close_col).rolling_std(window_size=period).alias("_bb_std"),
            ]
        )

        df = df.with_columns(
            [
                (pl.col(mid_col) + k * pl.col("_bb_std")).alias(upper_col),
                (pl.col(mid_col) - k * pl.col("_bb_std")).alias(lower_col),
            ]
        )

        # Bandwidth: normalised band width.
        df = df.with_columns(
            self._safe_divide(
                pl.col(upper_col) - pl.col(lower_col),
                pl.col(mid_col),
            ).alias(bw_col)
        )

        # %B: position within bands (0 = lower, 1 = upper).
        df = df.with_columns(
            self._safe_divide(
                pl.col(close_col) - pl.col(lower_col),
                pl.col(upper_col) - pl.col(lower_col),
                default=0.5,
            ).alias(pctb_col)
        )

        df = df.drop(["_bb_std"])
        return df

    # ------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------

    def _compute_atr(
        self,
        df: pl.DataFrame,
        high_col: str,
        low_col: str,
        close_col: str,
    ) -> pl.DataFrame:
        """Compute Average True Range.

        True Range = max(H-L, |H-C_prev|, |L-C_prev|)
        ATR = EMA(True Range, period) using Wilder's smoothing.

        Args:
            df: DataFrame with high, low, close columns.
            high_col: High price column.
            low_col: Low price column.
            close_col: Close price column.

        Returns:
            DataFrame with ``atr`` and ``atr_pct`` columns.
        """
        period = self.tech_config.atr_period
        atr_col = self._prefixed("atr")
        atr_pct_col = self._prefixed("atr_pct")

        prev_close = pl.col(close_col).shift(1)

        # True Range components.
        tr1 = pl.col(high_col) - pl.col(low_col)
        tr2 = (pl.col(high_col) - prev_close).abs()
        tr3 = (pl.col(low_col) - prev_close).abs()

        df = df.with_columns(
            pl.max_horizontal(tr1, tr2, tr3).alias("_true_range")
        )

        # Wilder's smoothing: alpha = 1/period.
        alpha = 1.0 / period
        df = df.with_columns(
            pl.col("_true_range").ewm_mean(alpha=alpha).alias(atr_col)
        )

        # ATR as percentage of price — useful for cross-commodity comparison.
        df = df.with_columns(
            self._safe_divide(pl.col(atr_col), pl.col(close_col)).alias(atr_pct_col)
        )

        df = df.drop(["_true_range"])
        return df

    # ------------------------------------------------------------------
    # Volume features
    # ------------------------------------------------------------------

    def _compute_volume_features(
        self,
        df: pl.DataFrame,
        close_col: str,
        volume_col: str,
    ) -> pl.DataFrame:
        """Compute volume-weighted price and volume momentum features.

        Features:
            - ``vw_close``: Volume-weighted close (rolling VWAP approximation).
            - ``volume_ratio``: Current volume / 20-day average volume.
            - ``volume_momentum``: 5-day rate of change in volume.

        Args:
            df: DataFrame with close and volume columns.
            close_col: Close price column.
            volume_col: Volume column.

        Returns:
            DataFrame with volume features appended.
        """
        vw_col = self._prefixed("vw_close")
        ratio_col = self._prefixed("volume_ratio")
        mom_col = self._prefixed("volume_momentum")

        # JUSTIFIED: 20-day rolling for VWAP-like computation matches the
        # standard monthly trading window.
        window = 20

        # Rolling VWAP: sum(price * volume) / sum(volume) over window.
        df = df.with_columns(
            [
                (pl.col(close_col) * pl.col(volume_col)).alias("_pv"),
            ]
        )

        df = df.with_columns(
            [
                pl.col("_pv").rolling_sum(window_size=window).alias("_pv_sum"),
                pl.col(volume_col).rolling_sum(window_size=window).alias("_vol_sum"),
            ]
        )

        df = df.with_columns(
            self._safe_divide(pl.col("_pv_sum"), pl.col("_vol_sum")).alias(vw_col)
        )

        # Volume ratio: today's volume relative to 20-day average.
        df = df.with_columns(
            self._safe_divide(
                pl.col(volume_col),
                pl.col(volume_col).rolling_mean(window_size=window),
                default=1.0,
            ).alias(ratio_col)
        )

        # Volume momentum: 5-day percentage change.
        # JUSTIFIED: 5-day lookback captures within-week volume dynamics.
        df = df.with_columns(
            pl.col(volume_col).pct_change(n=5).alias(mom_col)
        )

        df = df.drop(["_pv", "_pv_sum", "_vol_sum"])
        return df
