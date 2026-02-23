"""Macroeconomic data fetcher using the FRED API (Federal Reserve Economic Data).

Fetches interest rates, the US Dollar Index (DXY), CPI, and inflation
breakeven rates via the ``fredapi`` library.  Handles frequency alignment
across series with different release cadences (daily, monthly, quarterly).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from pydantic import Field, SecretStr

from src.data.fetchers.base_fetcher import BaseFetcher, DataFrequency, FetcherConfig

# ---------------------------------------------------------------------------
# Default FRED series — each mapped to a friendly column name
# ---------------------------------------------------------------------------
# JUSTIFIED: These are the most commonly used macro indicators for commodity
# pricing models (see Hamilton 2009, Kilian 2009).
DEFAULT_FRED_SERIES: dict[str, str] = {
    # Interest rates
    "DFF": "fed_funds_rate",          # Effective Federal Funds Rate (daily)
    "DGS2": "treasury_2y",            # 2-Year Treasury Constant Maturity (daily)
    "DGS10": "treasury_10y",          # 10-Year Treasury Constant Maturity (daily)
    "DGS30": "treasury_30y",          # 30-Year Treasury Constant Maturity (daily)
    # Dollar index
    "DTWEXBGS": "trade_weighted_usd", # Trade-Weighted US Dollar Index (daily)
    # Inflation
    "CPIAUCSL": "cpi_all_urban",      # CPI All Urban Consumers (monthly)
    "T10YIE": "breakeven_10y",        # 10-Year Breakeven Inflation Rate (daily)
    "T5YIE": "breakeven_5y",          # 5-Year Breakeven Inflation Rate (daily)
    # Spreads / risk
    "BAMLH0A0HYM2": "hy_oas",        # ICE BofA US High Yield OAS (daily)
    "TEDRATE": "ted_spread",          # TED Spread (daily, discontinued 2022 — kept for backtest)
}

# Map from FRED series frequency to a canonical label used internally.
SERIES_FREQUENCIES: dict[str, str] = {
    "DFF": "daily",
    "DGS2": "daily",
    "DGS10": "daily",
    "DGS30": "daily",
    "DTWEXBGS": "daily",
    "CPIAUCSL": "monthly",
    "T10YIE": "daily",
    "T5YIE": "daily",
    "BAMLH0A0HYM2": "daily",
    "TEDRATE": "daily",
}


class MacroConfig(FetcherConfig):
    """Configuration for the FRED macro data fetcher.

    Attributes:
        fred_api_key: FRED API key (obtain free at https://fred.stlouisfed.org/docs/api/api_key.html).
        series_map: Mapping of FRED series IDs to friendly column names.
        align_to: Target frequency for output alignment.
        forward_fill_limit: Maximum consecutive forward-fill steps for
            lower-frequency series promoted to daily.
    """

    name: str = "macro_fred"
    frequency: DataFrequency = DataFrequency.DAILY
    fred_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="FRED API key; set via FRED_API_KEY env var or config.",
    )
    series_map: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_FRED_SERIES)
    )
    align_to: str = Field(default="daily", pattern=r"^(daily|weekly|monthly)$")
    forward_fill_limit: int = Field(default=5, ge=1)


class MacroFetcher(BaseFetcher):
    """Fetches macroeconomic indicator series from FRED.

    Retrieves each configured FRED series, converts to Polars, and merges
    into a single daily-frequency DataFrame.  Lower-frequency series
    (monthly, quarterly) are up-sampled via forward-fill.  The output
    includes derived spread columns (e.g. 10Y-2Y term spread).

    Args:
        config: Optional custom configuration.

    Example:
        >>> fetcher = MacroFetcher(MacroConfig(fred_api_key=SecretStr("my_key")))
        >>> df = fetcher.fetch(date(2023, 1, 1), date(2023, 12, 31))
        >>> "treasury_10y" in df.columns
        True
    """

    def __init__(self, config: MacroConfig | None = None) -> None:
        super().__init__(config or MacroConfig())
        self.macro_config: MacroConfig = self.config  # type: ignore[assignment]
        self._fred_client: Any | None = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fetch(self, start_date: date, end_date: date) -> pl.DataFrame:
        """Fetch all configured FRED series and merge into one DataFrame.

        Args:
            start_date: Inclusive start date.
            end_date: Inclusive end date.

        Returns:
            Daily-aligned DataFrame with one column per FRED series plus
            derived spread columns.
        """
        cached = self._read_cache(start_date, end_date)
        if cached is not None:
            return cached

        fred = self._get_fred_client()

        # Build a date spine at daily frequency.
        date_spine = pl.DataFrame(
            {"date": pl.date_range(start_date, end_date, "1d", eager=True)}
        )
        result = date_spine

        for series_id, col_name in self.macro_config.series_map.items():
            series_df = self._fetch_single_series(fred, series_id, col_name, start_date, end_date)
            if series_df is not None:
                result = result.join(series_df, on="date", how="left")
                logger.debug(
                    "Merged {col} ({rows} non-null values)",
                    col=col_name,
                    rows=series_df.get_column(col_name).drop_nulls().len(),
                )
            else:
                # Add null column to preserve schema.
                result = result.with_columns(pl.lit(None).cast(pl.Float64).alias(col_name))

        # Forward-fill lower-frequency series (e.g. monthly CPI) to daily.
        result = self._align_frequencies(result)

        # Derived spreads.
        result = self._compute_derived(result)

        result = result.sort("date")
        self._write_cache(result, start_date, end_date)
        return result

    def validate(self, df: pl.DataFrame) -> bool:
        """Validate macro data quality.

        Args:
            df: DataFrame to validate.

        Returns:
            True if basic schema and value checks pass.
        """
        if df.is_empty():
            logger.warning("Macro validation failed: empty DataFrame")
            return False

        if "date" not in df.columns:
            logger.warning("Macro validation failed: no 'date' column")
            return False

        # At least one indicator column must be mostly non-null.
        indicator_cols = [c for c in df.columns if c != "date"]
        if not indicator_cols:
            logger.warning("Macro validation failed: no indicator columns")
            return False

        any_valid = False
        for col in indicator_cols:
            null_frac = df.get_column(col).null_count() / len(df)
            if null_frac < 0.5:
                any_valid = True
                break

        if not any_valid:
            logger.warning("Macro validation failed: all columns >50%% null")
            return False

        logger.debug("Macro validation passed ({n} rows, {c} cols)", n=len(df), c=len(indicator_cols))
        return True

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata for the FRED macro data source.

        Returns:
            Dictionary with source, series details, and column list.
        """
        return {
            "source": "fred_api",
            "frequency": self.macro_config.frequency.value,
            "description": (
                "Macroeconomic indicators from FRED: rates, dollar index, "
                "CPI, breakevens, and credit spreads."
            ),
            "series": dict(self.macro_config.series_map),
            "columns": ["date"] + list(self.macro_config.series_map.values()) + [
                "term_spread_10y2y",
                "real_rate_10y",
            ],
            "alignment": self.macro_config.align_to,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_fred_client(self) -> Any:
        """Lazily initialise the fredapi client.

        Returns:
            A ``fredapi.Fred`` instance.

        Raises:
            ImportError: If ``fredapi`` is not installed.
        """
        if self._fred_client is not None:
            return self._fred_client

        try:
            from fredapi import Fred
        except ImportError as exc:
            raise ImportError(
                "fredapi is required for MacroFetcher. Install with: pip install fredapi"
            ) from exc

        api_key = self.macro_config.fred_api_key.get_secret_value()
        if not api_key:
            logger.warning(
                "No FRED API key configured — set fred_api_key in MacroConfig "
                "or export FRED_API_KEY env var."
            )

        self._fred_client = Fred(api_key=api_key or None)
        return self._fred_client

    def _fetch_single_series(
        self,
        fred: Any,
        series_id: str,
        col_name: str,
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame | None:
        """Fetch a single FRED series and convert to Polars.

        Args:
            fred: ``fredapi.Fred`` instance.
            series_id: FRED series identifier (e.g. ``DGS10``).
            col_name: Friendly column name for the output.
            start_date: Query start date.
            end_date: Query end date.

        Returns:
            Two-column DataFrame (``date``, ``col_name``) or None on failure.
        """
        try:
            logger.info(
                "Fetching FRED series {sid} ({col})",
                sid=series_id,
                col=col_name,
            )
            pandas_series = fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date,
            )

            if pandas_series is None or pandas_series.empty:
                logger.warning("No data returned for {sid}", sid=series_id)
                return None

            # Convert pandas Series to Polars DataFrame.
            pdf = pandas_series.reset_index()
            pdf.columns = ["date", col_name]
            df = pl.from_pandas(pdf)
            df = df.with_columns(pl.col("date").cast(pl.Date))

            # FRED uses "." for missing in some series; these become NaN.
            if col_name in df.columns:
                df = df.with_columns(
                    pl.when(pl.col(col_name).is_nan())
                    .then(None)
                    .otherwise(pl.col(col_name))
                    .alias(col_name)
                )

            return df

        except Exception as exc:
            logger.error(
                "Failed to fetch FRED series {sid}: {exc}",
                sid=series_id,
                exc=str(exc),
            )
            return None

    def _align_frequencies(self, df: pl.DataFrame) -> pl.DataFrame:
        """Forward-fill lower-frequency columns up to the configured limit.

        Monthly and quarterly series are released with a lag; forward-fill
        carries the last known value into subsequent trading days.

        Args:
            df: Merged DataFrame with mixed-frequency columns.

        Returns:
            DataFrame with all columns aligned to daily frequency.
        """
        numeric_cols = [
            c for c in df.columns
            if c != "date" and df[c].dtype.is_numeric()
        ]

        df = df.with_columns(
            [
                pl.col(c).forward_fill(limit=self.macro_config.forward_fill_limit)
                for c in numeric_cols
            ]
        )

        logger.debug(
            "Aligned {n} columns to daily via forward-fill (limit={lim})",
            n=len(numeric_cols),
            lim=self.macro_config.forward_fill_limit,
        )
        return df

    def _compute_derived(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute derived macro indicators from raw series.

        Currently computes:
            - ``term_spread_10y2y``: 10Y Treasury minus 2Y Treasury yield.
            - ``real_rate_10y``: 10Y Treasury minus 10Y breakeven inflation.

        Args:
            df: DataFrame with raw FRED series columns.

        Returns:
            DataFrame with derived columns appended.
        """
        # Term spread: classic recession predictor / macro regime indicator.
        if "treasury_10y" in df.columns and "treasury_2y" in df.columns:
            df = df.with_columns(
                (pl.col("treasury_10y") - pl.col("treasury_2y")).alias("term_spread_10y2y")
            )
            logger.debug("Computed term_spread_10y2y")

        # Real rate: nominal 10Y minus breakeven inflation.
        if "treasury_10y" in df.columns and "breakeven_10y" in df.columns:
            df = df.with_columns(
                (pl.col("treasury_10y") - pl.col("breakeven_10y")).alias("real_rate_10y")
            )
            logger.debug("Computed real_rate_10y")

        return df
