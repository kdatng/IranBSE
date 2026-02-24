"""USD dynamics signals: dollar strength, petrodollar flows, and gold/oil ratio.

Commodity prices are typically denominated in USD, creating an inverse
relationship between dollar strength and commodity prices.  During
geopolitical crises involving oil-producing nations, petrodollar recycling
flows shift, and the gold/oil ratio captures relative safe-haven demand
versus supply disruption risk.

Typical usage::

    signal = DollarDynamics()
    features = signal.compute(fx_data, commodity_data)
    petrodollar = signal.petrodollar_flow_proxy(flow_data)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from loguru import logger


@dataclass
class DollarConfig:
    """Configuration for dollar dynamics signals.

    Attributes:
        dxy_col: Column for DXY (Dollar Index).
        eurusd_col: Column for EUR/USD exchange rate.
        gold_col: Column for gold price (USD/oz).
        oil_col: Column for crude oil price (USD/bbl).
        usdcny_col: Column for USD/CNY (China is major commodity importer).
        us10y_col: Column for US 10-year Treasury yield.
        lookback_short: Short-term momentum window. 21 trading days
            (~1 month) for tactical signals.
        lookback_long: Long-term baseline window. 126 trading days
            (~6 months) for structural trends.
    """

    dxy_col: str = "dxy"
    eurusd_col: str = "eurusd"
    gold_col: str = "gold_price"
    oil_col: str = "oil_price"
    usdcny_col: str = "usdcny"
    us10y_col: str = "us_10y_yield"
    lookback_short: int = 21
    lookback_long: int = 126


class DollarDynamics:
    """USD dynamics and petrodollar flow signals.

    Computes:
    - **Dollar strength momentum**: DXY rate of change and z-score, with
      breakout detection.
    - **Dollar-commodity correlation**: Rolling correlation between DXY
      and oil/gold, which breaks down during crisis.
    - **Gold/oil ratio**: Traditional safe-haven vs. growth indicator;
      spikes during geopolitical crises as gold rallies and oil
      faces demand fears.
    - **Petrodollar flow proxy**: Estimated recycling of oil revenues
      back into USD assets (Treasury purchases by oil exporters).
    - **Real effective exchange rate signal**: DXY adjusted for rate
      differentials.

    Args:
        config: Configuration parameters.
    """

    def __init__(self, config: DollarConfig | None = None) -> None:
        self.config = config or DollarConfig()
        logger.info(
            "DollarDynamics initialised: short={}, long={}",
            self.config.lookback_short,
            self.config.lookback_long,
        )

    def compute(
        self,
        fx_data: pl.DataFrame,
        commodity_data: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Compute all dollar dynamics signals.

        Args:
            fx_data: DataFrame with FX columns (DXY, EUR/USD, USD/CNY)
                and optionally Treasury yields.
            commodity_data: Optional DataFrame with gold and oil prices
                for cross-asset signals.

        Returns:
            DataFrame with signal columns:
            ``dxy_momentum``, ``dxy_zscore``, ``gold_oil_ratio``,
            ``dollar_commodity_corr``, ``petrodollar_proxy``.
        """
        result = fx_data.clone()

        # --- Dollar strength signals ---
        if self.config.dxy_col in result.columns:
            result = self._add_dollar_strength(result)

        # --- Cross-asset signals (require commodity data) ---
        if commodity_data is not None:
            result = self._merge_commodity_data(result, commodity_data)

        # Gold/oil ratio
        if self.config.gold_col in result.columns and self.config.oil_col in result.columns:
            result = self._add_gold_oil_ratio(result)

        # Dollar-commodity correlation
        if self.config.dxy_col in result.columns and self.config.oil_col in result.columns:
            result = self._add_dollar_oil_correlation(result)

        # Dollar-gold correlation
        if self.config.dxy_col in result.columns and self.config.gold_col in result.columns:
            result = self._add_dollar_gold_correlation(result)

        # --- Rate differential signal ---
        if self.config.us10y_col in result.columns:
            result = self._add_rate_differential(result)

        logger.info(
            "Dollar dynamics signals computed: {} rows, {} columns",
            result.height,
            result.width,
        )
        return result

    def petrodollar_flow_proxy(
        self,
        flow_data: pl.DataFrame,
        oil_revenue_col: str = "opec_oil_revenue_bn",
        tic_flow_col: str = "tic_opec_treasury_bn",
    ) -> pl.DataFrame:
        """Compute petrodollar recycling flow proxy.

        Petrodollar recycling -- oil exporters reinvesting oil revenues in
        USD-denominated assets -- is a structural dollar support mechanism.
        Disruptions (e.g., sanctions, geopolitical rifts) can weaken the
        dollar independently of traditional macro factors.

        Args:
            flow_data: DataFrame with oil revenue and TIC flow columns.
            oil_revenue_col: Column for estimated OPEC oil revenue.
            tic_flow_col: Column for Treasury International Capital flows
                from OPEC nations.

        Returns:
            DataFrame with ``petrodollar_recycling_ratio``,
            ``recycling_momentum``, and ``recycling_zscore`` columns.
        """
        self._validate_columns(flow_data, [oil_revenue_col, tic_flow_col])

        result = flow_data.clone()

        # Recycling ratio: what fraction of oil revenue goes back into Treasuries
        result = result.with_columns(
            (
                pl.col(tic_flow_col)
                / pl.col(oil_revenue_col).clip(lower_bound=0.01)
            ).alias("petrodollar_recycling_ratio")
        )

        # Momentum
        result = result.with_columns(
            (
                pl.col("petrodollar_recycling_ratio").rolling_mean(
                    window_size=3
                )
                - pl.col("petrodollar_recycling_ratio").rolling_mean(
                    window_size=self.config.lookback_short
                )
            ).alias("recycling_momentum")
        )

        # Z-score
        result = result.with_columns(
            (
                (
                    pl.col("petrodollar_recycling_ratio")
                    - pl.col("petrodollar_recycling_ratio").rolling_mean(
                        window_size=self.config.lookback_short
                    )
                )
                / pl.col("petrodollar_recycling_ratio")
                .rolling_std(window_size=self.config.lookback_short)
                .clip(lower_bound=0.001)
            ).alias("recycling_zscore")
        )

        return result

    def dollar_smile_signal(self, fx_data: pl.DataFrame) -> pl.DataFrame:
        """Compute the dollar smile signal.

        The "dollar smile" theory posits USD strengthens in both risk-off
        (flight to safety) and strong-growth (US exceptionalism) regimes,
        weakening only in benign risk environments.  This signal
        classifies the current regime.

        Args:
            fx_data: DataFrame with DXY and a volatility column.

        Returns:
            DataFrame with ``smile_regime`` and ``smile_signal`` columns.
        """
        if self.config.dxy_col not in fx_data.columns:
            raise ValueError(f"Need '{self.config.dxy_col}' column")

        result = fx_data.clone()
        dxy = result[self.config.dxy_col].to_numpy().astype(np.float64)

        # Compute DXY momentum and volatility
        dxy_returns = np.zeros_like(dxy)
        dxy_returns[1:] = np.diff(dxy) / np.where(dxy[:-1] != 0, dxy[:-1], np.nan)

        smile_regime = np.full(len(dxy), "benign", dtype=object)
        smile_signal = np.zeros(len(dxy), dtype=np.float64)

        window = self.config.lookback_short
        for i in range(window, len(dxy)):
            ret_window = dxy_returns[i - window : i]
            valid = ret_window[~np.isnan(ret_window)]
            if len(valid) < 5:
                continue

            ret = float(np.mean(valid))
            vol = float(np.std(valid))

            # Risk-off: DXY up + high vol
            if ret > 0 and vol > np.percentile(np.abs(dxy_returns[window:i]), 75):
                smile_regime[i] = "risk_off"
                smile_signal[i] = 1.0
            # Growth: DXY up + low vol
            elif ret > 0 and vol <= np.percentile(np.abs(dxy_returns[window:i]), 25):
                smile_regime[i] = "us_growth"
                smile_signal[i] = 0.5
            # Risk-on / benign: DXY down
            elif ret < 0:
                smile_regime[i] = "benign"
                smile_signal[i] = -0.5
            else:
                smile_regime[i] = "neutral"
                smile_signal[i] = 0.0

        result = result.with_columns(
            pl.Series("smile_regime", smile_regime.tolist()),
            pl.Series("smile_signal", smile_signal),
        )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_dollar_strength(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add dollar strength momentum and z-score.

        Args:
            df: Working DataFrame.

        Returns:
            DataFrame with dollar strength columns.
        """
        col = self.config.dxy_col

        # Momentum (short-term MA vs. long-term MA)
        df = df.with_columns(
            (
                pl.col(col).rolling_mean(window_size=self.config.lookback_short)
                - pl.col(col).rolling_mean(window_size=self.config.lookback_long)
            ).alias("dxy_momentum")
        )

        # Z-score
        df = df.with_columns(
            (
                (
                    pl.col(col)
                    - pl.col(col).rolling_mean(
                        window_size=self.config.lookback_long
                    )
                )
                / pl.col(col)
                .rolling_std(window_size=self.config.lookback_long)
                .clip(lower_bound=0.001)
            ).alias("dxy_zscore")
        )

        # Rate of change
        df = df.with_columns(
            (pl.col(col).pct_change(n=self.config.lookback_short) * 100)
            .alias("dxy_roc")
        )

        # Breakout detection: DXY beyond 2-sigma bollinger band
        df = df.with_columns(
            (pl.col("dxy_zscore").abs() > 2.0)
            .cast(pl.Int8)
            .alias("dxy_breakout_flag")
        )

        return df

    def _merge_commodity_data(
        self,
        fx_df: pl.DataFrame,
        commodity_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Merge commodity price columns into the FX DataFrame.

        Args:
            fx_df: FX DataFrame.
            commodity_df: Commodity price DataFrame.

        Returns:
            Merged DataFrame.
        """
        cols_to_add = []
        for col in [self.config.gold_col, self.config.oil_col]:
            if col in commodity_df.columns and col not in fx_df.columns:
                cols_to_add.append(col)

        if not cols_to_add:
            return fx_df

        if "date" in fx_df.columns and "date" in commodity_df.columns:
            select_cols = ["date"] + cols_to_add
            return fx_df.join(
                commodity_df.select(select_cols), on="date", how="left"
            )

        # Fallback: assume aligned by index
        for col in cols_to_add:
            n = min(fx_df.height, commodity_df.height)
            values = np.full(fx_df.height, np.nan, dtype=np.float64)
            values[:n] = commodity_df[col][:n].to_numpy().astype(np.float64)
            fx_df = fx_df.with_columns(pl.Series(col, values))

        return fx_df

    def _add_gold_oil_ratio(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add gold/oil ratio signals.

        The gold/oil ratio is a classic risk barometer.  Historical average
        is roughly 15-20 barrels of oil per ounce of gold.  Ratios above
        25 signal extreme fear / demand destruction; below 12 signals
        extreme oil bullishness.

        Args:
            df: Working DataFrame.

        Returns:
            DataFrame with gold/oil ratio columns.
        """
        df = df.with_columns(
            (
                pl.col(self.config.gold_col)
                / pl.col(self.config.oil_col).clip(lower_bound=0.01)
            ).alias("gold_oil_ratio")
        )

        # Z-score
        df = df.with_columns(
            (
                (
                    pl.col("gold_oil_ratio")
                    - pl.col("gold_oil_ratio").rolling_mean(
                        window_size=self.config.lookback_long
                    )
                )
                / pl.col("gold_oil_ratio")
                .rolling_std(window_size=self.config.lookback_long)
                .clip(lower_bound=0.001)
            ).alias("gold_oil_ratio_zscore")
        )

        # Momentum
        df = df.with_columns(
            (
                pl.col("gold_oil_ratio").rolling_mean(
                    window_size=self.config.lookback_short
                )
                - pl.col("gold_oil_ratio").rolling_mean(
                    window_size=self.config.lookback_long
                )
            ).alias("gold_oil_momentum")
        )

        return df

    def _add_dollar_oil_correlation(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add rolling correlation between DXY and oil.

        Args:
            df: Working DataFrame.

        Returns:
            DataFrame with DXY-oil correlation columns.
        """
        dxy = df[self.config.dxy_col].to_numpy().astype(np.float64)
        oil = df[self.config.oil_col].to_numpy().astype(np.float64)

        corr = self._rolling_correlation(dxy, oil, self.config.lookback_long)

        df = df.with_columns(pl.Series("dxy_oil_corr", corr))

        # Correlation breakdown detection (positive correlation is anomalous)
        df = df.with_columns(
            (pl.col("dxy_oil_corr") > 0)
            .cast(pl.Int8)
            .alias("dxy_oil_corr_breakdown")
        )

        return df

    def _add_dollar_gold_correlation(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add rolling correlation between DXY and gold.

        Args:
            df: Working DataFrame.

        Returns:
            DataFrame with DXY-gold correlation columns.
        """
        dxy = df[self.config.dxy_col].to_numpy().astype(np.float64)
        gold = df[self.config.gold_col].to_numpy().astype(np.float64)

        corr = self._rolling_correlation(dxy, gold, self.config.lookback_long)

        df = df.with_columns(pl.Series("dxy_gold_corr", corr))

        return df

    def _add_rate_differential(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add rate differential signals.

        Args:
            df: Working DataFrame.

        Returns:
            DataFrame with rate differential columns.
        """
        col = self.config.us10y_col

        # Rate momentum
        df = df.with_columns(
            (pl.col(col) - pl.col(col).shift(self.config.lookback_short))
            .alias("rate_momentum")
        )

        # Rate-DXY alignment (both rising = strong dollar environment)
        if self.config.dxy_col in df.columns:
            df = df.with_columns(
                (
                    pl.col("rate_momentum").sign()
                    * pl.col("dxy_momentum").sign()
                ).alias("rate_dxy_alignment")
            )

        return df

    @staticmethod
    def _rolling_correlation(
        x: np.ndarray[Any, np.dtype[np.float64]],
        y: np.ndarray[Any, np.dtype[np.float64]],
        window: int,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Compute rolling Pearson correlation.

        Args:
            x: First time series.
            y: Second time series.
            window: Rolling window size.

        Returns:
            Correlation array (NaN-filled for initial window).
        """
        n = len(x)
        corr = np.full(n, np.nan, dtype=np.float64)

        for i in range(window, n):
            x_w = x[i - window : i]
            y_w = y[i - window : i]
            mask = ~(np.isnan(x_w) | np.isnan(y_w))
            if mask.sum() < 5:
                continue
            x_valid = x_w[mask]
            y_valid = y_w[mask]
            if np.std(x_valid) < 1e-10 or np.std(y_valid) < 1e-10:
                continue
            corr[i] = float(np.corrcoef(x_valid, y_valid)[0, 1])

        return corr

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
