"""Abstract base class for all data fetchers in the IranBSE pipeline.

Provides a unified interface for fetching, validating, and describing data
from heterogeneous sources (market data, geopolitical indices, macro, etc.).
All fetchers must inherit from BaseFetcher and implement the abstract methods.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from pydantic import BaseModel, Field, field_validator


class DataFrequency(str, Enum):
    """Supported data frequencies for time-series alignment."""

    TICK = "tick"
    MINUTE = "1min"
    HOURLY = "1h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1mo"
    QUARTERLY = "3mo"


class FetcherConfig(BaseModel):
    """Base configuration for all data fetchers.

    Attributes:
        name: Human-readable name for the data source.
        frequency: Expected frequency of the fetched data.
        cache_dir: Directory for caching fetched data locally.
        cache_ttl_seconds: Time-to-live for cached data in seconds.
        max_retries: Maximum number of retry attempts on transient failures.
        retry_delay_seconds: Base delay between retries (exponential backoff applied).
        timeout_seconds: Request timeout for HTTP-based fetchers.
    """

    name: str = Field(..., min_length=1, description="Data source identifier")
    frequency: DataFrequency = DataFrequency.DAILY
    cache_dir: Path = Field(default=Path("data/cache"))
    cache_ttl_seconds: int = Field(default=3600, ge=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1)
    timeout_seconds: float = Field(default=30.0, ge=1.0)

    @field_validator("cache_dir")
    @classmethod
    def _ensure_cache_dir_exists(cls, v: Path) -> Path:
        """Create cache directory if it does not exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    model_config = {"frozen": False}


class FetchResult(BaseModel):
    """Container for fetch operation results with metadata.

    Attributes:
        success: Whether the fetch completed without errors.
        row_count: Number of rows in the fetched DataFrame.
        column_names: List of column names in the fetched DataFrame.
        date_range: Tuple of (earliest, latest) dates in the result.
        fetch_duration_seconds: Wall-clock time for the fetch operation.
        source_hash: SHA-256 hash of the result for change detection.
        warnings: Any non-fatal issues encountered during fetch.
    """

    success: bool
    row_count: int = 0
    column_names: list[str] = Field(default_factory=list)
    date_range: tuple[str, str] | None = None
    fetch_duration_seconds: float = 0.0
    source_hash: str = ""
    warnings: list[str] = Field(default_factory=list)


class BaseFetcher(ABC):
    """Abstract base class for all data fetchers.

    Subclasses must implement ``fetch``, ``validate``, and ``get_metadata``.
    The base class provides caching, retry logic, and standardised logging.

    Args:
        config: Pydantic configuration model for the fetcher.

    Example:
        >>> class MyFetcher(BaseFetcher):
        ...     def fetch(self, start_date, end_date):
        ...         ...
        ...     def validate(self, df):
        ...         return len(df) > 0
        ...     def get_metadata(self):
        ...         return {"source": "my_api"}
    """

    def __init__(self, config: FetcherConfig) -> None:
        self.config = config
        logger.info("Initialised fetcher: {name}", name=config.name)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch(self, start_date: date, end_date: date) -> pl.DataFrame:
        """Fetch raw data for the given date range.

        Args:
            start_date: Inclusive start of the fetch window.
            end_date: Inclusive end of the fetch window.

        Returns:
            A Polars DataFrame with at minimum a ``date`` column and one or
            more value columns.  Column naming conventions are documented in
            each concrete fetcher.

        Raises:
            FetchError: If the data source is unreachable after retries.
            ValidationError: If the returned data fails schema checks.
        """
        ...

    @abstractmethod
    def validate(self, df: pl.DataFrame) -> bool:
        """Run quality checks on a fetched DataFrame.

        Args:
            df: The DataFrame to validate.

        Returns:
            True if the data passes all quality gates, False otherwise.
        """
        ...

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return a metadata dictionary describing the data source.

        Returns:
            Dictionary containing at least ``source``, ``frequency``,
            ``description``, and ``columns`` keys.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def fetch_with_retry(self, start_date: date, end_date: date) -> pl.DataFrame:
        """Fetch data with exponential-backoff retry logic.

        Args:
            start_date: Inclusive start of the fetch window.
            end_date: Inclusive end of the fetch window.

        Returns:
            Validated Polars DataFrame.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.debug(
                    "Fetch attempt {attempt}/{max} for {name} [{start} -> {end}]",
                    attempt=attempt,
                    max=self.config.max_retries,
                    name=self.config.name,
                    start=start_date.isoformat(),
                    end=end_date.isoformat(),
                )
                t0 = time.monotonic()
                df = self.fetch(start_date, end_date)
                elapsed = time.monotonic() - t0

                if not self.validate(df):
                    logger.warning(
                        "Validation failed for {name} on attempt {attempt}",
                        name=self.config.name,
                        attempt=attempt,
                    )
                    continue

                logger.info(
                    "Fetched {rows} rows from {name} in {elapsed:.2f}s",
                    rows=len(df),
                    name=self.config.name,
                    elapsed=elapsed,
                )
                return df

            except Exception as exc:
                last_exc = exc
                delay = self.config.retry_delay_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Fetch error on attempt {attempt} for {name}: {exc}. "
                    "Retrying in {delay:.1f}s",
                    attempt=attempt,
                    name=self.config.name,
                    exc=str(exc),
                    delay=delay,
                )
                time.sleep(delay)

        msg = (
            f"All {self.config.max_retries} fetch attempts exhausted "
            f"for {self.config.name}"
        )
        raise RuntimeError(msg) from last_exc

    def _build_fetch_result(self, df: pl.DataFrame, elapsed: float) -> FetchResult:
        """Build a FetchResult summary from a DataFrame.

        Args:
            df: The fetched DataFrame.
            elapsed: Wall-clock seconds taken by the fetch.

        Returns:
            FetchResult with computed metadata.
        """
        date_range: tuple[str, str] | None = None
        if "date" in df.columns:
            dates = df.get_column("date").cast(pl.Utf8)
            date_range = (dates.min(), dates.max())  # type: ignore[arg-type]

        content_bytes = df.to_pandas().to_csv(index=False).encode("utf-8")
        source_hash = hashlib.sha256(content_bytes).hexdigest()[:16]

        return FetchResult(
            success=True,
            row_count=len(df),
            column_names=df.columns,
            date_range=date_range,
            fetch_duration_seconds=round(elapsed, 4),
            source_hash=source_hash,
        )

    def _cache_key(self, start_date: date, end_date: date) -> Path:
        """Compute a deterministic cache file path.

        Args:
            start_date: Start of the date range.
            end_date: End of the date range.

        Returns:
            Path to the cache parquet file.
        """
        key = f"{self.config.name}_{start_date.isoformat()}_{end_date.isoformat()}"
        hashed = hashlib.sha256(key.encode()).hexdigest()[:12]
        return self.config.cache_dir / f"{self.config.name}_{hashed}.parquet"

    def _read_cache(self, start_date: date, end_date: date) -> pl.DataFrame | None:
        """Attempt to load data from the local cache.

        Args:
            start_date: Start of the date range.
            end_date: End of the date range.

        Returns:
            Cached DataFrame if valid and within TTL, otherwise None.
        """
        cache_path = self._cache_key(start_date, end_date)
        if not cache_path.exists():
            return None

        age = time.time() - cache_path.stat().st_mtime
        if age > self.config.cache_ttl_seconds:
            logger.debug("Cache expired for {name} (age={age:.0f}s)", name=self.config.name, age=age)
            return None

        logger.debug("Cache hit for {name}", name=self.config.name)
        return pl.read_parquet(cache_path)

    def _write_cache(self, df: pl.DataFrame, start_date: date, end_date: date) -> None:
        """Persist a DataFrame to the local cache.

        Args:
            df: DataFrame to cache.
            start_date: Start of the date range.
            end_date: End of the date range.
        """
        cache_path = self._cache_key(start_date, end_date)
        df.write_parquet(cache_path)
        logger.debug("Wrote cache for {name} -> {path}", name=self.config.name, path=str(cache_path))

    @staticmethod
    def _parse_date(d: date | str | datetime) -> date:
        """Normalise various date representations to ``datetime.date``.

        Args:
            d: A date, datetime, or ISO-format string.

        Returns:
            A ``datetime.date`` instance.

        Raises:
            TypeError: If the input type is unsupported.
        """
        if isinstance(d, datetime):
            return d.date()
        if isinstance(d, date):
            return d
        if isinstance(d, str):
            return date.fromisoformat(d)
        raise TypeError(f"Cannot parse date from {type(d).__name__}: {d!r}")
