"""Tests for data fetcher classes.

Tests the BaseFetcher abstract interface via a concrete test implementation,
the CommodityPriceFetcher dispatch logic, data validation routines, and
missing-data handling edge cases.

All external API calls (yfinance, FRED, etc.) are mocked to ensure tests
run offline and deterministically.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from src.data.fetchers.base_fetcher import (
    BaseFetcher,
    DataFrequency,
    FetcherConfig,
    FetchResult,
)


# ---------------------------------------------------------------------------
# Concrete test fetcher
# ---------------------------------------------------------------------------

class MockCommodityFetcher(BaseFetcher):
    """Concrete fetcher for testing that returns predetermined data."""

    def __init__(
        self,
        config: FetcherConfig,
        mock_data: pl.DataFrame | None = None,
        should_fail: bool = False,
        fail_attempts: int = 0,
    ) -> None:
        super().__init__(config)
        self._mock_data = mock_data
        self._should_fail = should_fail
        self._fail_attempts = fail_attempts
        self._attempt_count = 0

    def fetch(self, start_date: date, end_date: date) -> pl.DataFrame:
        """Return mock data or raise an exception.

        Args:
            start_date: Start of date range.
            end_date: End of date range.

        Returns:
            Mock DataFrame.

        Raises:
            RuntimeError: If configured to fail.
        """
        self._attempt_count += 1

        if self._should_fail:
            raise RuntimeError("Simulated fetch failure")

        if self._attempt_count <= self._fail_attempts:
            raise ConnectionError(
                f"Simulated transient error (attempt {self._attempt_count})"
            )

        if self._mock_data is not None:
            return self._mock_data

        # Default: generate synthetic commodity price data.
        n_days = (end_date - start_date).days + 1
        dates = pl.date_range(start_date, end_date, eager=True)
        rng = np.random.default_rng(42)
        prices = 70.0 + np.cumsum(rng.normal(0, 1, n_days))

        return pl.DataFrame({
            "date": dates,
            "close": prices.tolist(),
            "volume": rng.integers(10_000, 100_000, n_days).tolist(),
        })

    def validate(self, df: pl.DataFrame) -> bool:
        """Validate that the DataFrame is non-empty and has required columns.

        Args:
            df: DataFrame to validate.

        Returns:
            True if validation passes.
        """
        if df.height == 0:
            return False
        if "date" not in df.columns:
            return False
        if df.height < 1:
            return False
        return True

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata describing this fetcher.

        Returns:
            Metadata dictionary.
        """
        return {
            "source": "mock_commodity",
            "frequency": self.config.frequency.value,
            "description": "Mock commodity price fetcher for testing",
            "columns": ["date", "close", "volume"],
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fetcher_config(tmp_path: Path) -> FetcherConfig:
    """Create a FetcherConfig with a temporary cache directory."""
    return FetcherConfig(
        name="test_commodity",
        frequency=DataFrequency.DAILY,
        cache_dir=tmp_path / "cache",
        cache_ttl_seconds=3600,
        max_retries=3,
        retry_delay_seconds=0.01,  # Fast retries for testing
        timeout_seconds=5.0,
    )


@pytest.fixture
def sample_price_data() -> pl.DataFrame:
    """Generate a sample commodity price DataFrame."""
    dates = pl.date_range(date(2020, 1, 1), date(2020, 12, 31), eager=True)
    n = len(dates)
    rng = np.random.default_rng(42)
    return pl.DataFrame({
        "date": dates,
        "close": (70.0 + np.cumsum(rng.normal(0, 0.5, n))).tolist(),
        "open": (70.0 + np.cumsum(rng.normal(0, 0.5, n))).tolist(),
        "high": (72.0 + np.cumsum(rng.normal(0, 0.5, n))).tolist(),
        "low": (68.0 + np.cumsum(rng.normal(0, 0.5, n))).tolist(),
        "volume": rng.integers(50_000, 200_000, n).tolist(),
    })


@pytest.fixture
def empty_data() -> pl.DataFrame:
    """Create an empty DataFrame with the expected schema."""
    return pl.DataFrame({
        "date": pl.Series([], dtype=pl.Date),
        "close": pl.Series([], dtype=pl.Float64),
        "volume": pl.Series([], dtype=pl.Int64),
    })


@pytest.fixture
def data_with_missing() -> pl.DataFrame:
    """Create a DataFrame with missing values."""
    dates = pl.date_range(date(2020, 1, 1), date(2020, 3, 31), eager=True)
    n = len(dates)
    rng = np.random.default_rng(42)

    close = rng.normal(70, 2, n).tolist()
    volume = rng.integers(10_000, 100_000, n).tolist()

    # Introduce nulls.
    for i in range(0, n, 7):
        close[i] = None  # type: ignore[call-overload]
    for i in range(0, n, 10):
        volume[i] = None  # type: ignore[call-overload]

    return pl.DataFrame({
        "date": dates,
        "close": close,
        "volume": volume,
    })


# ---------------------------------------------------------------------------
# Tests: CommodityPriceFetcher with mock data
# ---------------------------------------------------------------------------

class TestMockCommodityFetcher:
    """Tests for the mock commodity fetcher (standing in for real fetchers)."""

    def test_fetch_returns_dataframe(
        self, fetcher_config: FetcherConfig
    ) -> None:
        """Fetching a valid date range returns a non-empty DataFrame."""
        fetcher = MockCommodityFetcher(fetcher_config)
        df = fetcher.fetch(date(2020, 1, 1), date(2020, 12, 31))

        assert isinstance(df, pl.DataFrame)
        assert df.height > 0
        assert "date" in df.columns
        assert "close" in df.columns

    def test_fetch_date_range(self, fetcher_config: FetcherConfig) -> None:
        """Fetched data spans the requested date range."""
        start = date(2020, 6, 1)
        end = date(2020, 6, 30)
        fetcher = MockCommodityFetcher(fetcher_config)
        df = fetcher.fetch(start, end)

        dates = df.get_column("date")
        assert dates.min() == start
        assert dates.max() == end

    def test_fetch_with_mock_data(
        self,
        fetcher_config: FetcherConfig,
        sample_price_data: pl.DataFrame,
    ) -> None:
        """Fetcher returns injected mock data when provided."""
        fetcher = MockCommodityFetcher(
            fetcher_config, mock_data=sample_price_data
        )
        df = fetcher.fetch(date(2020, 1, 1), date(2020, 12, 31))

        assert df.height == sample_price_data.height
        assert df.columns == sample_price_data.columns

    def test_fetch_column_types(self, fetcher_config: FetcherConfig) -> None:
        """Verify that columns have expected types."""
        fetcher = MockCommodityFetcher(fetcher_config)
        df = fetcher.fetch(date(2020, 1, 1), date(2020, 3, 31))

        assert df.get_column("date").dtype == pl.Date
        assert df.get_column("close").dtype == pl.Float64
        assert df.get_column("volume").dtype in (pl.Int64, pl.UInt64, pl.Int32)

    def test_metadata(self, fetcher_config: FetcherConfig) -> None:
        """Metadata returns expected keys and values."""
        fetcher = MockCommodityFetcher(fetcher_config)
        meta = fetcher.get_metadata()

        assert "source" in meta
        assert "frequency" in meta
        assert "columns" in meta
        assert meta["source"] == "mock_commodity"
        assert "date" in meta["columns"]


# ---------------------------------------------------------------------------
# Tests: Data validation
# ---------------------------------------------------------------------------

class TestDataValidation:
    """Tests for the validate() method and data quality checks."""

    def test_validate_valid_data(
        self,
        fetcher_config: FetcherConfig,
        sample_price_data: pl.DataFrame,
    ) -> None:
        """Valid data passes validation."""
        fetcher = MockCommodityFetcher(fetcher_config)
        assert fetcher.validate(sample_price_data) is True

    def test_validate_empty_data(
        self,
        fetcher_config: FetcherConfig,
        empty_data: pl.DataFrame,
    ) -> None:
        """Empty DataFrame fails validation."""
        fetcher = MockCommodityFetcher(fetcher_config)
        assert fetcher.validate(empty_data) is False

    def test_validate_missing_date_column(
        self, fetcher_config: FetcherConfig
    ) -> None:
        """DataFrame without a 'date' column fails validation."""
        df = pl.DataFrame({
            "timestamp": [date(2020, 1, 1)],
            "close": [70.0],
        })
        fetcher = MockCommodityFetcher(fetcher_config)
        assert fetcher.validate(df) is False

    def test_validate_data_with_nulls(
        self,
        fetcher_config: FetcherConfig,
        data_with_missing: pl.DataFrame,
    ) -> None:
        """Data with some nulls still passes basic validation."""
        fetcher = MockCommodityFetcher(fetcher_config)
        # Basic validation only checks non-empty + date column.
        assert fetcher.validate(data_with_missing) is True

    def test_null_ratio_detection(
        self, data_with_missing: pl.DataFrame
    ) -> None:
        """Identify columns with excessive null ratios."""
        for col in data_with_missing.columns:
            null_ratio = data_with_missing[col].null_count() / data_with_missing.height
            # Our fixture injects ~14% nulls, which is below the 50% threshold.
            assert null_ratio < 0.5, f"Column {col} has excessive nulls: {null_ratio:.0%}"


# ---------------------------------------------------------------------------
# Tests: Missing data handling
# ---------------------------------------------------------------------------

class TestMissingDataHandling:
    """Tests for edge cases around missing or malformed data."""

    def test_fetch_failure_raises(self, fetcher_config: FetcherConfig) -> None:
        """Fetcher configured to fail raises RuntimeError."""
        fetcher = MockCommodityFetcher(
            fetcher_config, should_fail=True
        )
        with pytest.raises(RuntimeError, match="Simulated fetch failure"):
            fetcher.fetch(date(2020, 1, 1), date(2020, 1, 31))

    def test_retry_logic_succeeds_after_transient_failures(
        self, fetcher_config: FetcherConfig
    ) -> None:
        """Retry logic recovers from transient failures within max_retries."""
        fetcher = MockCommodityFetcher(
            fetcher_config,
            fail_attempts=2,  # Fail twice, succeed on third
        )
        df = fetcher.fetch_with_retry(date(2020, 1, 1), date(2020, 1, 31))
        assert df.height > 0
        assert fetcher._attempt_count == 3  # 2 failures + 1 success

    def test_retry_logic_exhausted(self, fetcher_config: FetcherConfig) -> None:
        """Exhausting all retries raises RuntimeError."""
        fetcher = MockCommodityFetcher(
            fetcher_config, should_fail=True
        )
        with pytest.raises(RuntimeError, match="fetch attempts exhausted"):
            fetcher.fetch_with_retry(date(2020, 1, 1), date(2020, 1, 31))

    def test_fetch_result_metadata(
        self,
        fetcher_config: FetcherConfig,
        sample_price_data: pl.DataFrame,
    ) -> None:
        """FetchResult correctly summarizes a successful fetch."""
        fetcher = MockCommodityFetcher(fetcher_config)
        result = fetcher._build_fetch_result(sample_price_data, elapsed=1.23)

        assert result.success is True
        assert result.row_count == sample_price_data.height
        assert "date" in result.column_names
        assert result.fetch_duration_seconds == 1.23
        assert len(result.source_hash) > 0

    def test_single_row_data(self, fetcher_config: FetcherConfig) -> None:
        """A single-row DataFrame passes validation."""
        df = pl.DataFrame({
            "date": [date(2020, 6, 15)],
            "close": [72.5],
            "volume": [50_000],
        })
        fetcher = MockCommodityFetcher(fetcher_config)
        assert fetcher.validate(df) is True

    def test_all_null_column(self, fetcher_config: FetcherConfig) -> None:
        """DataFrame where an entire column is null."""
        df = pl.DataFrame({
            "date": pl.date_range(date(2020, 1, 1), date(2020, 1, 10), eager=True),
            "close": [None] * 10,
            "volume": list(range(10)),
        })
        fetcher = MockCommodityFetcher(fetcher_config)
        # Still passes basic validation (has date + non-empty).
        assert fetcher.validate(df) is True

        # But quality check should flag it.
        null_ratio = df["close"].null_count() / df.height
        assert null_ratio == 1.0


# ---------------------------------------------------------------------------
# Tests: Caching
# ---------------------------------------------------------------------------

class TestCaching:
    """Tests for the BaseFetcher caching mechanisms."""

    def test_cache_key_deterministic(
        self, fetcher_config: FetcherConfig
    ) -> None:
        """Same inputs produce the same cache key."""
        fetcher = MockCommodityFetcher(fetcher_config)
        key1 = fetcher._cache_key(date(2020, 1, 1), date(2020, 12, 31))
        key2 = fetcher._cache_key(date(2020, 1, 1), date(2020, 12, 31))
        assert key1 == key2

    def test_cache_key_varies_with_dates(
        self, fetcher_config: FetcherConfig
    ) -> None:
        """Different date ranges produce different cache keys."""
        fetcher = MockCommodityFetcher(fetcher_config)
        key1 = fetcher._cache_key(date(2020, 1, 1), date(2020, 6, 30))
        key2 = fetcher._cache_key(date(2020, 7, 1), date(2020, 12, 31))
        assert key1 != key2

    def test_write_and_read_cache(
        self,
        fetcher_config: FetcherConfig,
        sample_price_data: pl.DataFrame,
    ) -> None:
        """Data can be written to cache and read back."""
        fetcher = MockCommodityFetcher(fetcher_config)
        start, end = date(2020, 1, 1), date(2020, 12, 31)

        fetcher._write_cache(sample_price_data, start, end)
        cached = fetcher._read_cache(start, end)

        assert cached is not None
        assert cached.height == sample_price_data.height
        assert cached.columns == sample_price_data.columns

    def test_cache_miss_returns_none(
        self, fetcher_config: FetcherConfig
    ) -> None:
        """Reading from an empty cache returns None."""
        fetcher = MockCommodityFetcher(fetcher_config)
        result = fetcher._read_cache(date(2020, 1, 1), date(2020, 12, 31))
        assert result is None


# ---------------------------------------------------------------------------
# Tests: FetcherConfig validation
# ---------------------------------------------------------------------------

class TestFetcherConfig:
    """Tests for the FetcherConfig pydantic model."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Default config values are applied correctly."""
        cfg = FetcherConfig(name="test", cache_dir=tmp_path / "c")
        assert cfg.frequency == DataFrequency.DAILY
        assert cfg.max_retries == 3
        assert cfg.timeout_seconds == 30.0

    def test_cache_dir_created(self, tmp_path: Path) -> None:
        """Cache directory is created if it does not exist."""
        cache_dir = tmp_path / "new_cache_dir"
        assert not cache_dir.exists()
        cfg = FetcherConfig(name="test", cache_dir=cache_dir)
        assert cfg.cache_dir.exists()

    def test_name_required(self, tmp_path: Path) -> None:
        """Empty name is rejected."""
        with pytest.raises(Exception):
            FetcherConfig(name="", cache_dir=tmp_path)

    def test_parse_date_string(self) -> None:
        """ISO date string is parsed to date object."""
        result = BaseFetcher._parse_date("2020-06-15")
        assert result == date(2020, 6, 15)

    def test_parse_date_object(self) -> None:
        """date object is returned as-is."""
        d = date(2020, 6, 15)
        assert BaseFetcher._parse_date(d) is d

    def test_parse_date_invalid_type(self) -> None:
        """Non-date type raises TypeError."""
        with pytest.raises(TypeError):
            BaseFetcher._parse_date(12345)  # type: ignore[arg-type]
