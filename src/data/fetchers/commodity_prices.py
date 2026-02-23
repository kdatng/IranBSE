"""Commodity price fetcher using Yahoo Finance (yfinance).

Fetches OHLCV data for crude oil (WTI CL=F, Brent BZ=F) and wheat (ZW=F)
futures, then computes derived columns: log returns, realised volatility,
and basic spread metrics.  Designed as the primary market data source for
the IranBSE pipeline.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np
import polars as pl
import yfinance as yf
from loguru import logger
from pydantic import Field

from src.data.fetchers.base_fetcher import BaseFetcher, DataFrequency, FetcherConfig

# ---------------------------------------------------------------------------
# Default tickers tracked by this fetcher
# ---------------------------------------------------------------------------
# JUSTIFIED: WTI and Brent are the two global crude benchmarks; ZW is the
# primary CBOT wheat contract.  These three are the core commodities in the
# IranBSE scenario model.
DEFAULT_TICKERS: dict[str, str] = {
    "CL=F": "wti_crude",
    "BZ=F": "brent_crude",
    "ZW=F": "cbot_wheat",
}

# JUSTIFIED: 21 trading days ~ 1 calendar month; standard for realised vol.
REALISED_VOL_WINDOW: int = 21


class CommodityPriceConfig(FetcherConfig):
    """Configuration for the commodity price fetcher.

    Attributes:
        tickers: Mapping of Yahoo Finance ticker symbols to friendly column
            prefixes used throughout the pipeline.
        realised_vol_window: Rolling window (in trading days) for annualised
            realised volatility computation.
        fill_method: Strategy for handling missing prices. ``forward`` uses
            last-observation-carried-forward; ``interpolate`` uses linear
            interpolation.
    """

    name: str = "commodity_prices"
    frequency: DataFrequency = DataFrequency.DAILY
    tickers: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_TICKERS))
    realised_vol_window: int = Field(default=REALISED_VOL_WINDOW, ge=2)
    fill_method: str = Field(default="forward", pattern=r"^(forward|interpolate)$")


class CommodityPriceFetcher(BaseFetcher):
    """Fetches commodity futures OHLCV data and computes derived features.

    Downloads daily price bars from Yahoo Finance for the configured tickers,
    merges them into a single wide DataFrame, and appends:
      - log returns  (``{prefix}_log_return``)
      - realised volatility (``{prefix}_realised_vol``)
      - intraday range (``{prefix}_range``)

    Args:
        config: Optional custom configuration; uses defaults if omitted.

    Example:
        >>> fetcher = CommodityPriceFetcher()
        >>> df = fetcher.fetch(date(2023, 1, 1), date(2023, 12, 31))
        >>> "wti_crude_close" in df.columns
        True
    """

    def __init__(self, config: CommodityPriceConfig | None = None) -> None:
        super().__init__(config or CommodityPriceConfig())
        self.price_config: CommodityPriceConfig = self.config  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fetch(self, start_date: date, end_date: date) -> pl.DataFrame:
        """Fetch OHLCV data for all configured commodity tickers.

        Args:
            start_date: First calendar date (inclusive).
            end_date: Last calendar date (inclusive).

        Returns:
            Wide-format Polars DataFrame with columns:
            ``date``, ``{prefix}_open``, ``{prefix}_high``, ``{prefix}_low``,
            ``{prefix}_close``, ``{prefix}_volume``, ``{prefix}_log_return``,
            ``{prefix}_realised_vol``, ``{prefix}_range``
            for each configured ticker.
        """
        cached = self._read_cache(start_date, end_date)
        if cached is not None:
            return cached

        start_str = start_date.isoformat()
        end_str = end_date.isoformat()

        frames: list[pl.DataFrame] = []

        for ticker, prefix in self.price_config.tickers.items():
            logger.info(
                "Downloading {ticker} ({prefix}) from {start} to {end}",
                ticker=ticker,
                prefix=prefix,
                start=start_str,
                end=end_str,
            )
            raw = self._download_ticker(ticker, start_str, end_str)
            if raw.is_empty():
                logger.warning("No data returned for {ticker}", ticker=ticker)
                continue

            enriched = self._process_single_ticker(raw, prefix)
            frames.append(enriched)

        if not frames:
            logger.error("No data fetched for any ticker")
            return pl.DataFrame({"date": []}).cast({"date": pl.Date})

        df = frames[0]
        for other in frames[1:]:
            df = df.join(other, on="date", how="outer_coalesce")

        df = df.sort("date")
        df = self._handle_missing(df)

        self._write_cache(df, start_date, end_date)
        return df

    def validate(self, df: pl.DataFrame) -> bool:
        """Validate fetched commodity price data.

        Checks:
            1. DataFrame is non-empty.
            2. A ``date`` column exists.
            3. At least one ``*_close`` column is present.
            4. No close column is entirely null.
            5. Close prices are strictly positive where non-null.

        Args:
            df: DataFrame to validate.

        Returns:
            True if all checks pass.
        """
        if df.is_empty():
            logger.warning("Validation failed: empty DataFrame")
            return False

        if "date" not in df.columns:
            logger.warning("Validation failed: missing 'date' column")
            return False

        close_cols = [c for c in df.columns if c.endswith("_close")]
        if not close_cols:
            logger.warning("Validation failed: no close price columns found")
            return False

        for col in close_cols:
            series = df.get_column(col)
            if series.is_null().all():
                logger.warning("Validation failed: {col} is entirely null", col=col)
                return False
            non_null = series.drop_nulls()
            if (non_null <= 0).any():
                logger.warning("Validation failed: {col} has non-positive values", col=col)
                return False

        logger.debug("Validation passed for commodity prices ({n} rows)", n=len(df))
        return True

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata describing the commodity price data source.

        Returns:
            Dictionary with source details, ticker list, and column schema.
        """
        prefixes = list(self.price_config.tickers.values())
        columns: list[str] = ["date"]
        for p in prefixes:
            columns.extend(
                [
                    f"{p}_open",
                    f"{p}_high",
                    f"{p}_low",
                    f"{p}_close",
                    f"{p}_volume",
                    f"{p}_log_return",
                    f"{p}_realised_vol",
                    f"{p}_range",
                ]
            )

        return {
            "source": "yahoo_finance",
            "frequency": self.price_config.frequency.value,
            "description": (
                "Daily OHLCV data for commodity futures with log returns "
                "and realised volatility."
            ),
            "tickers": dict(self.price_config.tickers),
            "columns": columns,
            "realised_vol_window": self.price_config.realised_vol_window,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _download_ticker(ticker: str, start: str, end: str) -> pl.DataFrame:
        """Download OHLCV data for a single ticker via yfinance.

        Args:
            ticker: Yahoo Finance ticker symbol.
            start: ISO start date string.
            end: ISO end date string.

        Returns:
            Polars DataFrame with ``date``, ``open``, ``high``, ``low``,
            ``close``, ``volume`` columns.
        """
        yf_ticker = yf.Ticker(ticker)
        hist = yf_ticker.history(start=start, end=end, auto_adjust=True)

        if hist.empty:
            return pl.DataFrame()

        hist = hist.reset_index()
        hist.columns = [c.lower().replace(" ", "_") for c in hist.columns]

        # yfinance returns a 'date' column as datetime; normalise.
        df = pl.from_pandas(hist)

        # Ensure we have the expected columns; rename 'date' dtype to Date.
        rename_map: dict[str, str] = {}
        if "date" in df.columns:
            pass  # keep as-is
        elif "datetime" in df.columns:
            rename_map["datetime"] = "date"

        if rename_map:
            df = df.rename(rename_map)

        keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
        df = df.select(keep)

        # Cast date column to pl.Date for joins.
        if "date" in df.columns:
            df = df.with_columns(pl.col("date").cast(pl.Date))

        return df

    def _process_single_ticker(self, df: pl.DataFrame, prefix: str) -> pl.DataFrame:
        """Rename columns, compute log returns and realised vol for one ticker.

        Args:
            df: Raw OHLCV DataFrame with generic column names.
            prefix: Friendly name prefix for output columns.

        Returns:
            DataFrame with prefixed columns and derived features.
        """
        window = self.price_config.realised_vol_window

        # Prefix OHLCV columns.
        rename_map = {
            col: f"{prefix}_{col}" for col in df.columns if col != "date"
        }
        df = df.rename(rename_map)

        close_col = f"{prefix}_close"

        # Log returns: ln(P_t / P_{t-1})
        df = df.with_columns(
            pl.col(close_col)
            .log()
            .diff()
            .alias(f"{prefix}_log_return")
        )

        # Realised volatility: annualised std of log returns over rolling window.
        # JUSTIFIED: sqrt(252) annualisation factor for daily data
        # (standard in finance — 252 trading days per year).
        annualisation_factor = math.sqrt(252)  # JUSTIFIED: standard trading-day annualisation
        df = df.with_columns(
            (
                pl.col(f"{prefix}_log_return")
                .rolling_std(window_size=window)
                * annualisation_factor
            ).alias(f"{prefix}_realised_vol")
        )

        # Intraday range: (high - low) / close — normalised measure of daily
        # price dispersion.
        high_col = f"{prefix}_high"
        low_col = f"{prefix}_low"
        if high_col in df.columns and low_col in df.columns:
            df = df.with_columns(
                ((pl.col(high_col) - pl.col(low_col)) / pl.col(close_col)).alias(
                    f"{prefix}_range"
                )
            )

        return df

    def _handle_missing(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fill missing values according to the configured strategy.

        Args:
            df: DataFrame potentially containing null values.

        Returns:
            DataFrame with nulls handled.
        """
        if self.price_config.fill_method == "forward":
            numeric_cols = [
                c for c in df.columns if c != "date" and df[c].dtype.is_numeric()
            ]
            df = df.with_columns([pl.col(c).forward_fill() for c in numeric_cols])
            logger.debug(
                "Applied forward-fill to {n} numeric columns", n=len(numeric_cols)
            )
        elif self.price_config.fill_method == "interpolate":
            numeric_cols = [
                c for c in df.columns if c != "date" and df[c].dtype.is_numeric()
            ]
            df = df.with_columns([pl.col(c).interpolate() for c in numeric_cols])
            logger.debug(
                "Applied linear interpolation to {n} numeric columns",
                n=len(numeric_cols),
            )

        return df

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def fetch_single(
        self, ticker: str, start_date: date, end_date: date
    ) -> pl.DataFrame:
        """Fetch data for a single ticker without merging.

        Args:
            ticker: Yahoo Finance ticker symbol (e.g. ``CL=F``).
            start_date: Inclusive start date.
            end_date: Inclusive end date.

        Returns:
            DataFrame for the single ticker with derived features.
        """
        prefix = self.price_config.tickers.get(ticker, ticker.replace("=", "_"))
        raw = self._download_ticker(ticker, start_date.isoformat(), end_date.isoformat())
        if raw.is_empty():
            return raw
        return self._process_single_ticker(raw, prefix)

    def compute_spread(self, df: pl.DataFrame, col_a: str, col_b: str) -> pl.DataFrame:
        """Compute the price spread between two close-price columns.

        Useful for Brent-WTI spread analysis.

        Args:
            df: DataFrame containing both close columns.
            col_a: First close column name (e.g. ``brent_crude_close``).
            col_b: Second close column name (e.g. ``wti_crude_close``).

        Returns:
            Input DataFrame with an appended ``{col_a}_minus_{col_b}`` column.
        """
        spread_name = f"{col_a}_minus_{col_b}"
        return df.with_columns(
            (pl.col(col_a) - pl.col(col_b)).alias(spread_name)
        )
