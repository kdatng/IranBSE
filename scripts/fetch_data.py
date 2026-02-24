#!/usr/bin/env python3
"""Data download script for the IranBSE pipeline.

Downloads all data sources defined in ``config/data_sources.yaml``, applies
validation checks, and persists results to parquet via the DataStore layer.

Features:
    - Progress bars via ``tqdm`` (falls back to plain logging)
    - Exponential-backoff retry logic per source
    - Schema and completeness validation
    - Summary report on completion

Usage::

    # Download everything
    python scripts/fetch_data.py

    # Download specific sources
    python scripts/fetch_data.py --sources commodity_prices macro_indicators

    # Dry run (validate config only)
    python scripts/fetch_data.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_SOURCES_PATH = PROJECT_ROOT / "config" / "data_sources.yaml"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class FetchStatus:
    """Status record for a single data source fetch.

    Attributes:
        source_name: Identifier of the data source.
        provider: Provider backend (e.g. ``"yfinance"``, ``"fred"``).
        status: One of ``"success"``, ``"error"``, ``"skipped"``,
            ``"not_implemented"``.
        row_count: Number of rows fetched (0 if failed).
        elapsed_seconds: Wall-clock duration of the fetch.
        error_message: Human-readable error description if applicable.
        warnings: Non-fatal issues encountered.
    """

    source_name: str
    provider: str
    status: str = "pending"
    row_count: int = 0
    elapsed_seconds: float = 0.0
    error_message: str = ""
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_data_sources_config(path: Path = DEFAULT_DATA_SOURCES_PATH) -> dict[str, Any]:
    """Load and validate the data_sources.yaml configuration.

    Args:
        path: Path to data_sources.yaml.

    Returns:
        Parsed YAML dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config structure is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data sources config not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh) or {}

    if "data_sources" not in config:
        raise ValueError(
            f"Expected top-level 'data_sources' key in {path}"
        )

    sources = config["data_sources"]
    logger.info("Loaded {} data sources from {}", len(sources), path)
    return config


# ---------------------------------------------------------------------------
# Fetcher dispatch
# ---------------------------------------------------------------------------

def _fetch_yfinance(
    source_name: str,
    source_cfg: dict[str, Any],
    output_dir: Path,
    max_retries: int = 3,
) -> FetchStatus:
    """Fetch market data from Yahoo Finance.

    Args:
        source_name: Logical name (e.g. ``"commodity_prices"``).
        source_cfg: Provider-specific config from data_sources.yaml.
        output_dir: Directory to save parquet files.
        max_retries: Maximum retry attempts.

    Returns:
        FetchStatus with outcome details.
    """
    status = FetchStatus(source_name=source_name, provider="yfinance")
    symbols = source_cfg.get("symbols", {})
    start_date = source_cfg.get("start_date", "2000-01-01")

    try:
        import yfinance as yf
        import polars as pl
    except ImportError as exc:
        status.status = "error"
        status.error_message = f"Missing dependency: {exc}"
        return status

    t0 = time.monotonic()
    all_frames: list[pl.DataFrame] = []

    for label, ticker in symbols.items():
        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(
                    "Fetching {} ({}) attempt {}/{}",
                    label, ticker, attempt, max_retries,
                )
                data = yf.download(
                    ticker,
                    start=start_date,
                    auto_adjust=True,
                    progress=False,
                )
                if data.empty:
                    status.warnings.append(f"Empty result for {label} ({ticker})")
                    break

                df = pl.from_pandas(data.reset_index())
                df = df.rename({"Date": "date"})
                # Prefix value columns with the label for disambiguation.
                for col in df.columns:
                    if col != "date":
                        df = df.rename({col: f"{label}_{col.lower()}"})

                all_frames.append(df)
                logger.debug("Fetched {} rows for {}", len(df), label)
                break  # success
            except Exception as exc:
                delay = 2.0 ** (attempt - 1)
                logger.warning(
                    "yfinance error for {} attempt {}: {}. Retrying in {:.1f}s",
                    label, attempt, exc, delay,
                )
                time.sleep(delay)
                if attempt == max_retries:
                    status.warnings.append(
                        f"Failed to fetch {label} after {max_retries} attempts: {exc}"
                    )

    if all_frames:
        try:
            # Join all symbol DataFrames on the date column.
            combined = all_frames[0]
            for frame in all_frames[1:]:
                combined = combined.join(frame, on="date", how="outer_coalesce")

            combined = combined.sort("date")
            output_path = output_dir / f"{source_name}.parquet"
            combined.write_parquet(output_path)
            status.row_count = len(combined)
            status.status = "success"
            logger.info(
                "Saved {} rows for {} -> {}",
                len(combined), source_name, output_path,
            )
        except Exception as exc:
            status.status = "error"
            status.error_message = f"Failed to combine/save: {exc}"
    else:
        status.status = "error"
        status.error_message = "No data frames produced"

    status.elapsed_seconds = round(time.monotonic() - t0, 2)
    return status


def _fetch_fred(
    source_name: str,
    source_cfg: dict[str, Any],
    output_dir: Path,
    max_retries: int = 3,
) -> FetchStatus:
    """Fetch macroeconomic data from FRED.

    Args:
        source_name: Logical name (e.g. ``"macro_indicators"``).
        source_cfg: Provider-specific config including series mappings.
        output_dir: Directory to save parquet files.
        max_retries: Maximum retry attempts.

    Returns:
        FetchStatus with outcome details.
    """
    status = FetchStatus(source_name=source_name, provider="fred")
    series_map = source_cfg.get("series", {})
    start_date = source_cfg.get("start_date", "2000-01-01")

    try:
        import os
        from fredapi import Fred
        import polars as pl
    except ImportError as exc:
        status.status = "error"
        status.error_message = f"Missing dependency: {exc}"
        return status

    api_key = os.environ.get("FRED_API_KEY", "1a72200a7cdeee8aa47553f0ac2a0f29")
    if not api_key:
        status.status = "error"
        status.error_message = (
            "FRED_API_KEY environment variable not set. "
            "Get a key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
        return status

    t0 = time.monotonic()
    fred = Fred(api_key=api_key)
    all_frames: list[pl.DataFrame] = []

    for label, series_id in series_map.items():
        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(
                    "Fetching FRED series {} ({}) attempt {}/{}",
                    label, series_id, attempt, max_retries,
                )
                data = fred.get_series(series_id, observation_start=start_date)
                if data is None or data.empty:
                    status.warnings.append(f"Empty FRED series: {label} ({series_id})")
                    break

                df = pl.DataFrame({
                    "date": data.index.to_list(),
                    label: data.values.tolist(),
                })
                all_frames.append(df)
                logger.debug("Fetched {} rows for FRED {}", len(df), label)
                break
            except Exception as exc:
                delay = 2.0 ** (attempt - 1)
                logger.warning(
                    "FRED error for {} attempt {}: {}. Retrying in {:.1f}s",
                    label, attempt, exc, delay,
                )
                time.sleep(delay)
                if attempt == max_retries:
                    status.warnings.append(
                        f"Failed to fetch FRED {label} after {max_retries} attempts: {exc}"
                    )

    if all_frames:
        try:
            combined = all_frames[0]
            for frame in all_frames[1:]:
                combined = combined.join(frame, on="date", how="outer_coalesce")
            combined = combined.sort("date")

            output_path = output_dir / f"{source_name}.parquet"
            combined.write_parquet(output_path)
            status.row_count = len(combined)
            status.status = "success"
            logger.info(
                "Saved {} rows for {} -> {}",
                len(combined), source_name, output_path,
            )
        except Exception as exc:
            status.status = "error"
            status.error_message = f"Failed to combine/save: {exc}"
    else:
        status.status = "error"
        status.error_message = "No FRED data frames produced"

    status.elapsed_seconds = round(time.monotonic() - t0, 2)
    return status


def _fetch_eia(
    source_name: str,
    source_cfg: dict[str, Any],
    output_dir: Path,
    max_retries: int = 3,
) -> FetchStatus:
    """Fetch petroleum data from the EIA API v2.

    Uses the ``/v2/seriesid/{series_id}`` endpoint which accepts the legacy
    dotted series IDs (e.g. ``PET.WCESTUS1.W``).

    Args:
        source_name: Logical name (e.g. ``"eia_petroleum"``).
        source_cfg: Provider-specific config including series mappings.
        output_dir: Directory to save parquet files.
        max_retries: Maximum retry attempts per series.

    Returns:
        FetchStatus with outcome details.
    """
    import os

    status = FetchStatus(source_name=source_name, provider="eia_api")
    series_map = source_cfg.get("series", {})
    start_date = source_cfg.get("start_date", "2000-01-01")

    api_key = os.environ.get("EIA_API_KEY", "WSRVns7B8xiyTaxp6ynP8pZDs7Z3cLHrxEpajpuc")
    if not api_key:
        status.status = "error"
        status.error_message = "EIA_API_KEY not set."
        return status

    try:
        import requests
        import polars as pl
    except ImportError as exc:
        status.status = "error"
        status.error_message = f"Missing dependency: {exc}"
        return status

    t0 = time.monotonic()
    all_frames: list = []

    for label, series_id in series_map.items():
        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(
                    "Fetching EIA series {} ({}) attempt {}/{}",
                    label, series_id, attempt, max_retries,
                )
                url = f"https://api.eia.gov/v2/seriesid/{series_id}"
                params = {
                    "api_key": api_key,
                    "out": "json",
                    "start": start_date,
                }
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                payload = resp.json()

                # EIA v2 seriesid endpoint returns data under
                # response.data as a list of {period, value} dicts.
                data_list = (
                    payload.get("response", {}).get("data", [])
                )
                if not data_list:
                    status.warnings.append(
                        f"Empty EIA series: {label} ({series_id})"
                    )
                    break

                dates = []
                values = []
                for row in data_list:
                    period = row.get("period")
                    value = row.get("value")
                    if period is None or value is None:
                        continue
                    try:
                        values.append(float(value))
                        dates.append(period)
                    except (ValueError, TypeError):
                        continue

                if not dates:
                    status.warnings.append(
                        f"No parseable rows for EIA {label} ({series_id})"
                    )
                    break

                df = pl.DataFrame({"date": dates, label: values})
                # Period strings may be "YYYY-MM-DD" (weekly) or "YYYY-MM"
                # (monthly).  Normalise to Date.
                df = df.with_columns(
                    pl.col("date").str.to_date(strict=False)
                )
                # Drop any rows where date parsing failed.
                df = df.drop_nulls(subset=["date"])

                all_frames.append(df)
                logger.debug("Fetched {} rows for EIA {}", len(df), label)
                break  # success

            except Exception as exc:
                delay = 2.0 ** (attempt - 1)
                logger.warning(
                    "EIA error for {} attempt {}: {}. Retrying in {:.1f}s",
                    label, attempt, exc, delay,
                )
                time.sleep(delay)
                if attempt == max_retries:
                    status.warnings.append(
                        f"Failed to fetch EIA {label} after {max_retries} attempts: {exc}"
                    )

    if all_frames:
        try:
            combined = all_frames[0]
            for frame in all_frames[1:]:
                combined = combined.join(frame, on="date", how="outer_coalesce")
            combined = combined.sort("date")

            output_path = output_dir / f"{source_name}.parquet"
            combined.write_parquet(output_path)
            status.row_count = len(combined)
            status.status = "success"
            logger.info(
                "Saved {} rows for {} -> {}",
                len(combined), source_name, output_path,
            )
        except Exception as exc:
            status.status = "error"
            status.error_message = f"Failed to combine/save: {exc}"
    else:
        status.status = "error"
        status.error_message = "No EIA data frames produced"

    status.elapsed_seconds = round(time.monotonic() - t0, 2)
    return status


# ---------------------------------------------------------------------------
# LSEG / Refinitiv fetcher
# ---------------------------------------------------------------------------

# Shared session state — opened once, reused across all LSEG sources.
_lseg_session_open: bool = False


def _open_lseg_session(app_key: str, session_type: str) -> None:
    """Open an LSEG Data Library session if not already active.

    Writes ``lseg-data.config.json`` beside this script's project root
    and calls ``ld.open_session()``.

    Args:
        app_key: LSEG / Eikon App Key.
        session_type: ``"desktop"`` (Workspace running) or ``"platform"``.
    """
    global _lseg_session_open
    if _lseg_session_open:
        return

    import json as _json

    import lseg.data as ld

    config_path = PROJECT_ROOT / "lseg-data.config.json"
    if session_type == "platform":
        default_session = "platform.ldp"
    else:
        default_session = "desktop.workspace"

    cfg = {
        "sessions": {
            "default": default_session,
            "desktop": {"workspace": {"app-key": app_key}},
            "platform": {"ldp": {"app-key": app_key}},
        }
    }
    with open(config_path, "w", encoding="utf-8") as fh:
        _json.dump(cfg, fh, indent=2)

    ld.open_session()
    _lseg_session_open = True
    logger.info("LSEG session opened (type={})", session_type)


def _close_lseg_session() -> None:
    """Close the LSEG session if it is open."""
    global _lseg_session_open
    if not _lseg_session_open:
        return
    try:
        import lseg.data as ld

        ld.close_session()
    except Exception:
        pass
    _lseg_session_open = False


def _lseg_resolve_app_key(source_cfg: dict[str, Any]) -> str | None:
    """Resolve the LSEG App Key from env var or config.

    Precedence: ``LSEG_APP_KEY`` env var > ``app_key`` field in
    data_sources.yaml.
    """
    import os

    key = os.environ.get("LSEG_APP_KEY", "") or source_cfg.get("app_key", "")
    return key or None


def _lseg_fetch_timeseries(
    source_cfg: dict[str, Any],
) -> list[tuple[str, Any]]:
    """Fetch daily time-series for a set of RICs.

    Returns:
        List of ``(label, pandas.DataFrame)`` pairs.
    """
    import lseg.data as ld

    rics = source_cfg.get("rics", {})
    fields = source_cfg.get("fields", ["TRDPRC_1"])
    interval = source_cfg.get("interval", "daily")
    start = source_cfg.get("start_date", "2015-01-01")

    results: list[tuple[str, Any]] = []
    for label, ric in rics.items():
        logger.debug("LSEG get_history: {} ({})", label, ric)
        try:
            df = ld.get_history(
                universe=ric,
                fields=fields,
                interval=interval,
                start=start,
            )
            if df is not None and not df.empty:
                results.append((label, df))
                logger.debug("  -> {} rows", len(df))
            else:
                logger.warning("  -> empty result for {}", ric)
        except Exception as exc:
            logger.warning("  -> error for {}: {}", ric, exc)
    return results


def _lseg_fetch_snapshot(
    source_cfg: dict[str, Any],
    ric_key: str = "chain_rics",
) -> list[tuple[str, Any]]:
    """Fetch point-in-time snapshot data (curves, options chains).

    Args:
        source_cfg: Source configuration block.
        ric_key: Key in config holding the RIC mapping (default
            ``"chain_rics"``).

    Returns:
        List of ``(label, pandas.DataFrame)`` pairs.
    """
    import lseg.data as ld

    chain_rics = source_cfg.get(ric_key, {})
    fields = source_cfg.get("fields", [])

    results: list[tuple[str, Any]] = []
    for label, ric in chain_rics.items():
        logger.debug("LSEG get_data: {} ({})", label, ric)
        try:
            # fields is mandatory in lseg-data v2
            df = ld.get_data(universe=ric, fields=fields)
            if df is not None and not df.empty:
                results.append((label, df))
                logger.debug("  -> {} rows", len(df))
            else:
                logger.warning("  -> empty result for {}", ric)
        except Exception as exc:
            logger.warning("  -> error for {}: {}", ric, exc)
    return results


def _lseg_fetch_news(source_cfg: dict[str, Any]) -> list[tuple[str, Any]]:
    """Fetch news headlines for configured queries.

    Returns:
        List of ``(query_label, pandas.DataFrame)`` pairs.
    """
    import lseg.data as ld

    queries = source_cfg.get("queries", [])
    count = source_cfg.get("count_per_query", 100)

    results: list[tuple[str, Any]] = []
    for i, query in enumerate(queries):
        label = f"news_q{i}"
        logger.debug("LSEG get_news_headlines: '{}'", query)
        try:
            df = ld.get_news_headlines(query=query, count=count)
            if df is not None and not df.empty:
                results.append((label, df))
                logger.debug("  -> {} headlines", len(df))
            else:
                logger.warning("  -> no headlines for '{}'", query)
        except Exception as exc:
            logger.warning("  -> error for '{}': {}", query, exc)
    return results


def _fetch_lseg(
    source_name: str,
    source_cfg: dict[str, Any],
    output_dir: Path,
    max_retries: int = 3,
) -> FetchStatus:
    """Fetch data from LSEG via the ``lseg-data`` library.

    Supports multiple dataset types controlled by the ``dataset_type``
    key in the source config:

    - ``timeseries``: Daily OHLCV via ``get_history`` (freight indices, etc.)
    - ``futures_curve``: Term structure snapshots via ``get_data``
    - ``options``: Options chains with IV/Greeks via ``get_data``
    - ``news``: News headlines via ``get_news_headlines``

    Requires either LSEG Workspace running locally (desktop session) or
    platform credentials.  Set ``LSEG_APP_KEY`` env var or ``app_key``
    in data_sources.yaml.

    Args:
        source_name: Logical name from data_sources.yaml.
        source_cfg: Provider-specific config block.
        output_dir: Directory to write parquet output.
        max_retries: Maximum retry attempts.

    Returns:
        FetchStatus with outcome details.
    """
    status = FetchStatus(source_name=source_name, provider="lseg")
    dataset_type = source_cfg.get("dataset_type", "timeseries")

    # --- Resolve App Key ---
    app_key = _lseg_resolve_app_key(source_cfg)
    if not app_key:
        status.status = "error"
        status.error_message = (
            "No LSEG App Key found. Set the LSEG_APP_KEY environment "
            "variable or add app_key to data_sources.yaml. Generate a "
            "key by typing APPKEY in LSEG Workspace."
        )
        return status

    # --- Import check ---
    try:
        import lseg.data  # noqa: F401
        import polars as pl
    except ImportError as exc:
        status.status = "error"
        status.error_message = (
            f"Missing dependency: {exc}. Install with: pip install lseg-data"
        )
        return status

    t0 = time.monotonic()

    # --- Open session (shared across all LSEG sources) ---
    session_type = source_cfg.get("session_type", "desktop")
    for attempt in range(1, max_retries + 1):
        try:
            _open_lseg_session(app_key, session_type)
            break
        except Exception as exc:
            delay = 2.0 ** (attempt - 1)
            logger.warning(
                "LSEG session open attempt {}/{} failed: {}. "
                "Retrying in {:.1f}s",
                attempt, max_retries, exc, delay,
            )
            if attempt == max_retries:
                status.status = "error"
                status.error_message = (
                    f"Failed to open LSEG session after {max_retries} "
                    f"attempts: {exc}"
                )
                status.elapsed_seconds = round(time.monotonic() - t0, 2)
                return status
            time.sleep(delay)

    # --- Fetch based on dataset type ---
    pairs: list[tuple[str, Any]] = []
    try:
        if dataset_type == "timeseries":
            pairs = _lseg_fetch_timeseries(source_cfg)
        elif dataset_type in ("futures_curve", "options"):
            pairs = _lseg_fetch_snapshot(source_cfg)
        elif dataset_type == "news":
            pairs = _lseg_fetch_news(source_cfg)
        else:
            status.status = "error"
            status.error_message = f"Unknown LSEG dataset_type: {dataset_type}"
            status.elapsed_seconds = round(time.monotonic() - t0, 2)
            return status
    except Exception as exc:
        status.status = "error"
        status.error_message = f"LSEG fetch error: {exc}"
        status.elapsed_seconds = round(time.monotonic() - t0, 2)
        return status

    if not pairs:
        status.status = "error"
        status.error_message = "No LSEG data frames produced"
        status.elapsed_seconds = round(time.monotonic() - t0, 2)
        return status

    # --- Convert pandas -> polars, combine, save ---
    try:
        import polars as pl

        polars_frames: list[pl.DataFrame] = []
        for label, pdf in pairs:
            # pandas DataFrames from LSEG have the index as dates.
            pdf_reset = pdf.reset_index()
            plf = pl.from_pandas(pdf_reset)

            # Normalise the date column name.
            date_col = None
            for candidate in ("Date", "date", "Timestamp", "timestamp"):
                if candidate in plf.columns:
                    date_col = candidate
                    break
            if date_col and date_col != "date":
                plf = plf.rename({date_col: "date"})

            # Prefix non-date columns with the label.
            for col in plf.columns:
                if col != "date":
                    plf = plf.rename({col: f"{label}_{col}"})

            polars_frames.append(plf)

        if dataset_type == "news":
            # News frames have different schemas — just concatenate.
            combined = pl.concat(polars_frames, how="diagonal_relaxed")
        else:
            combined = polars_frames[0]
            for frame in polars_frames[1:]:
                combined = combined.join(
                    frame, on="date", how="outer_coalesce",
                )
            combined = combined.sort("date")

        output_path = output_dir / f"{source_name}.parquet"
        combined.write_parquet(output_path)
        status.row_count = len(combined)
        status.status = "success"
        logger.info(
            "Saved {} rows for {} -> {}",
            len(combined), source_name, output_path,
        )
    except Exception as exc:
        status.status = "error"
        status.error_message = f"Failed to combine/save: {exc}"

    status.elapsed_seconds = round(time.monotonic() - t0, 2)
    return status


def _fetch_placeholder(
    source_name: str,
    source_cfg: dict[str, Any],
    output_dir: Path,
) -> FetchStatus:
    """Placeholder fetcher for not-yet-implemented providers.

    Args:
        source_name: Logical name of the data source.
        source_cfg: Provider-specific configuration.
        output_dir: Output directory (unused).

    Returns:
        FetchStatus marked as not implemented.
    """
    provider = source_cfg.get("provider", "unknown")
    logger.warning(
        "Fetcher for provider '{}' not yet implemented (source={})",
        provider, source_name,
    )
    return FetchStatus(
        source_name=source_name,
        provider=provider,
        status="not_implemented",
    )


# Provider dispatch table mapping provider names to fetch functions.
_PROVIDER_DISPATCH: dict[str, Any] = {
    "yfinance": _fetch_yfinance,
    "fred": _fetch_fred,
    "eia_api": _fetch_eia,
    "lseg": _fetch_lseg,
}


def fetch_source(
    source_name: str,
    source_cfg: dict[str, Any],
    output_dir: Path,
) -> FetchStatus:
    """Fetch a single data source by dispatching to the appropriate provider.

    Args:
        source_name: Logical name from data_sources.yaml.
        source_cfg: Provider-specific configuration block.
        output_dir: Directory to write parquet outputs.

    Returns:
        FetchStatus describing the outcome.
    """
    provider = source_cfg.get("provider", "unknown")
    enabled = source_cfg.get("enabled", True)

    if not enabled:
        logger.info("Skipping disabled source: {}", source_name)
        return FetchStatus(
            source_name=source_name,
            provider=provider,
            status="skipped",
        )

    fetch_fn = _PROVIDER_DISPATCH.get(provider, _fetch_placeholder)
    return fetch_fn(source_name, source_cfg, output_dir)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_parquet(path: Path) -> list[str]:
    """Run basic quality checks on a saved parquet file.

    Args:
        path: Path to the parquet file.

    Returns:
        List of warning messages (empty if all checks pass).
    """
    warnings: list[str] = []

    if not path.exists():
        warnings.append(f"File does not exist: {path}")
        return warnings

    try:
        import polars as pl

        df = pl.read_parquet(path)

        if df.height == 0:
            warnings.append(f"Empty DataFrame in {path.name}")
            return warnings

        if "date" not in df.columns:
            warnings.append(f"Missing 'date' column in {path.name}")

        # Check for excessive null ratios.
        for col in df.columns:
            null_ratio = df[col].null_count() / df.height
            if null_ratio > 0.5:  # JUSTIFIED: >50% nulls is a red flag for data quality
                warnings.append(
                    f"Column '{col}' in {path.name} has {null_ratio:.0%} nulls"
                )

        logger.debug(
            "Validated {}: {} rows, {} cols, {} warnings",
            path.name, df.height, df.width, len(warnings),
        )
    except Exception as exc:
        warnings.append(f"Failed to read {path.name}: {exc}")

    return warnings


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(statuses: list[FetchStatus]) -> None:
    """Print a formatted summary table of fetch results.

    Args:
        statuses: List of FetchStatus objects for all sources.
    """
    logger.info("=" * 70)
    logger.info("DATA FETCH SUMMARY")
    logger.info("=" * 70)

    success_count = sum(1 for s in statuses if s.status == "success")
    error_count = sum(1 for s in statuses if s.status == "error")
    skipped_count = sum(1 for s in statuses if s.status in ("skipped", "not_implemented"))
    total_rows = sum(s.row_count for s in statuses)
    total_time = sum(s.elapsed_seconds for s in statuses)

    for s in statuses:
        icon = {
            "success": "[OK]",
            "error": "[FAIL]",
            "skipped": "[SKIP]",
            "not_implemented": "[N/A]",
        }.get(s.status, "[???]")

        logger.info(
            "  {} {:<30} provider={:<12} rows={:<8} time={:.1f}s",
            icon, s.source_name, s.provider, s.row_count, s.elapsed_seconds,
        )
        if s.error_message:
            logger.error("       Error: {}", s.error_message)
        for w in s.warnings:
            logger.warning("       Warning: {}", w)

    logger.info("-" * 70)
    logger.info(
        "  Total: {} success, {} errors, {} skipped | {} rows | {:.1f}s",
        success_count, error_count, skipped_count, total_rows, total_time,
    )
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="fetch_data",
        description="Download data sources for IranBSE pipeline.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        metavar="SOURCE",
        help="Specific source names to fetch (default: all).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_DATA_SOURCES_PATH,
        help="Path to data_sources.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DIR,
        help="Output directory for parquet files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate config and print plan without fetching.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Run the data fetching pipeline.

    Args:
        argv: Optional CLI argument list.

    Returns:
        Exit code (0 on success, 1 on any errors).
    """
    args = parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    try:
        config = load_data_sources_config(args.config)
    except Exception as exc:
        logger.critical("Failed to load config: {}", exc)
        return 1

    sources = config["data_sources"]

    # Filter to requested sources if specified.
    if args.sources:
        unknown = set(args.sources) - set(sources.keys())
        if unknown:
            logger.error("Unknown sources: {}. Available: {}", unknown, list(sources.keys()))
            return 1
        sources = {k: v for k, v in sources.items() if k in args.sources}

    logger.info("Will fetch {} data sources", len(sources))

    if args.dry_run:
        logger.info("DRY RUN -- listing planned fetches:")
        for name, cfg in sources.items():
            provider = cfg.get("provider", "unknown")
            enabled = cfg.get("enabled", True)
            logger.info(
                "  {:<30} provider={:<12} enabled={}",
                name, provider, enabled,
            )
        return 0

    # Ensure output directory exists.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Attempt progress bar import.
    try:
        from tqdm import tqdm
        iterator = tqdm(sources.items(), desc="Fetching", unit="source")
    except ImportError:
        iterator = sources.items()  # type: ignore[assignment]

    statuses: list[FetchStatus] = []
    for source_name, source_cfg in iterator:
        status = fetch_source(source_name, source_cfg, args.output_dir)
        statuses.append(status)

        # Run validation on successfully saved files.
        if status.status == "success":
            parquet_path = args.output_dir / f"{source_name}.parquet"
            validation_warnings = validate_parquet(parquet_path)
            status.warnings.extend(validation_warnings)

    # Clean up LSEG session if it was opened.
    _close_lseg_session()

    print_summary(statuses)

    has_errors = any(s.status == "error" for s in statuses)
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
