#!/usr/bin/env python3
"""Main execution pipeline for IranBSE commodity futures analysis.

Orchestrates the full analysis workflow:
    data fetch -> feature engineering -> model fitting -> scenario generation
    -> risk analysis -> report generation

Supports selective execution via CLI flags (e.g. scenarios-only, backtest-only)
and parallel model fitting via joblib.

Usage::

    # Full pipeline
    python scripts/run_pipeline.py

    # Scenarios only (skip data fetch + model fitting)
    python scripts/run_pipeline.py --scenarios-only

    # Backtest on historical conflicts
    python scripts/run_pipeline.py --backtest-only

    # Subset of models with custom config
    python scripts/run_pipeline.py --models regime_switching extreme_value --config config/model_config.yaml
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.yaml"
DEFAULT_HYPERPARAMS_PATH = PROJECT_ROOT / "config" / "hyperparams.yaml"
DEFAULT_SCENARIOS_PATH = PROJECT_ROOT / "config" / "scenarios.yaml"
DEFAULT_DATA_SOURCES_PATH = PROJECT_ROOT / "config" / "data_sources.yaml"
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure the src package is importable when running as a script.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(level: str = "DEBUG", log_file: str | None = None) -> None:
    """Configure loguru sinks for console and optional file output.

    Args:
        level: Minimum log level for all sinks.
        log_file: Optional path to a rotating log file.  When *None* the
            default from the model config is used.
    """
    logger.remove()  # Remove default stderr handler
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            level=level,
            rotation="10 MB",
            retention="30 days",
            compression="gz",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        )
    logger.info("Logging configured at level={}", level)


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load and parse a YAML configuration file.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If *path* does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    logger.debug("Loaded config from {} ({} top-level keys)", path, len(data))
    return data


def load_all_configs(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and merge all project configuration files.

    Args:
        config_path: Path to the master model_config.yaml.  Sibling config
            files (hyperparams, scenarios, data_sources) are resolved
            relative to the same directory.

    Returns:
        Merged configuration dictionary with top-level keys ``model_config``,
        ``hyperparams``, ``scenarios``, and ``data_sources``.
    """
    config_dir = config_path.parent
    merged: dict[str, Any] = {
        "model_config": load_yaml_config(config_path),
        "hyperparams": load_yaml_config(config_dir / "hyperparams.yaml"),
        "scenarios": load_yaml_config(config_dir / "scenarios.yaml"),
        "data_sources": load_yaml_config(config_dir / "data_sources.yaml"),
    }
    logger.info("All configuration files loaded successfully")
    return merged


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_fetch_data(config: dict[str, Any]) -> dict[str, Any]:
    """Stage 1: Fetch all raw data sources.

    Downloads market data, macro indicators, geopolitical events, and
    alternative data as defined in data_sources.yaml.  Results are persisted
    to parquet via the DataStore abstraction.

    Args:
        config: Merged project configuration.

    Returns:
        Dictionary mapping source names to their fetch status metadata.
    """
    logger.info("=== STAGE 1: Data Fetch ===")
    t0 = time.monotonic()

    data_sources_cfg = config.get("data_sources", {}).get("data_sources", {})
    fetch_results: dict[str, Any] = {}

    for source_name, source_cfg in data_sources_cfg.items():
        provider = source_cfg.get("provider", "unknown")
        enabled = source_cfg.get("enabled", True)

        if not enabled:
            logger.info("Skipping disabled source: {} (provider={})", source_name, provider)
            fetch_results[source_name] = {"status": "skipped", "reason": "disabled"}
            continue

        try:
            logger.info("Fetching {} from provider '{}'", source_name, provider)
            # Import and dispatch to the appropriate fetcher.
            # Each fetcher is expected to be registered in src.data.fetchers.
            fetch_results[source_name] = _dispatch_fetch(source_name, source_cfg)
        except Exception as exc:
            logger.error("Failed to fetch {}: {}", source_name, exc)
            fetch_results[source_name] = {"status": "error", "error": str(exc)}

    elapsed = time.monotonic() - t0
    logger.info("Data fetch completed in {:.1f}s ({} sources)", elapsed, len(fetch_results))
    return fetch_results


def _dispatch_fetch(source_name: str, source_cfg: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a fetch request to the appropriate fetcher implementation.

    Args:
        source_name: Logical name of the data source.
        source_cfg: Provider-specific configuration from data_sources.yaml.

    Returns:
        Status dictionary with row counts and date ranges.
    """
    provider = source_cfg.get("provider", "unknown")
    logger.debug(
        "Dispatching fetch for source={} provider={}", source_name, provider
    )
    # Placeholder: concrete fetcher instantiation goes here once the
    # fetcher modules (commodity_prices, macro, etc.) are implemented.
    logger.warning(
        "Fetcher for provider '{}' not yet implemented -- skipping {}",
        provider,
        source_name,
    )
    return {"status": "not_implemented", "provider": provider}


def stage_feature_engineering(config: dict[str, Any]) -> dict[str, Any]:
    """Stage 2: Engineer features from raw data.

    Computes technical indicators, fundamental signals, cross-asset features,
    and regime-detection features.

    Args:
        config: Merged project configuration.

    Returns:
        Dictionary mapping feature group names to DataFrames (as metadata).
    """
    logger.info("=== STAGE 2: Feature Engineering ===")
    t0 = time.monotonic()

    feature_results: dict[str, Any] = {}

    feature_groups = [
        "technical",
        "fundamental",
        "cross_asset",
        "regime",
        "geopolitical",
    ]

    for group in feature_groups:
        try:
            logger.info("Computing feature group: {}", group)
            # Placeholder for actual processor dispatch.
            feature_results[group] = {"status": "not_implemented"}
        except Exception as exc:
            logger.error("Feature engineering failed for {}: {}", group, exc)
            feature_results[group] = {"status": "error", "error": str(exc)}

    elapsed = time.monotonic() - t0
    logger.info("Feature engineering completed in {:.1f}s", elapsed)
    return feature_results


def stage_model_fitting(
    config: dict[str, Any],
    model_names: list[str] | None = None,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Stage 3: Fit all enabled models (optionally in parallel).

    Uses joblib for parallel model fitting when multiple models are
    requested.  Falls back to sequential fitting if joblib is unavailable.

    Args:
        config: Merged project configuration.
        model_names: Optional explicit list of model names to fit.  When
            *None*, all models listed under ``models.enabled`` in the config
            are fitted.
        n_jobs: Number of parallel jobs for joblib (``-1`` = all CPUs).

    Returns:
        Dictionary mapping model names to fit status / metrics.
    """
    logger.info("=== STAGE 3: Model Fitting ===")
    t0 = time.monotonic()

    model_cfg = config.get("model_config", {})
    enabled_models = _resolve_model_list(model_cfg, model_names)

    if not enabled_models:
        logger.warning("No models to fit -- check config or --models flag")
        return {}

    logger.info("Fitting {} models: {}", len(enabled_models), enabled_models)

    fit_results: dict[str, Any] = {}

    try:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_fit_single_model)(name, config) for name in enabled_models
        )
        for name, result in zip(enabled_models, results):
            fit_results[name] = result
    except ImportError:
        logger.warning("joblib not available -- fitting models sequentially")
        for name in enabled_models:
            fit_results[name] = _fit_single_model(name, config)

    elapsed = time.monotonic() - t0
    logger.info("Model fitting completed in {:.1f}s", elapsed)
    return fit_results


def _resolve_model_list(
    model_cfg: dict[str, Any],
    explicit_names: list[str] | None,
) -> list[str]:
    """Build the flat list of model names to fit.

    Args:
        model_cfg: The ``model_config`` section of the merged config.
        explicit_names: CLI-provided model names (takes precedence).

    Returns:
        De-duplicated list of model names.
    """
    if explicit_names:
        return list(dict.fromkeys(explicit_names))  # preserve order, dedup

    enabled = model_cfg.get("models", {}).get("enabled", {})
    all_names: list[str] = []
    for _category, names in enabled.items():
        if isinstance(names, list):
            all_names.extend(names)
    return list(dict.fromkeys(all_names))


def _fit_single_model(name: str, config: dict[str, Any]) -> dict[str, Any]:
    """Fit a single model by name.

    Args:
        name: Registered model name (e.g. ``"regime_switching"``).
        config: Full merged configuration.

    Returns:
        Dictionary with fit status and timing.
    """
    t0 = time.monotonic()
    try:
        from src.models.registry import ModelRegistry
        from src.models.base_model import ModelConfig

        hyperparams = config.get("hyperparams", {}).get(name, {})
        model_config = ModelConfig(name=name, params=hyperparams)

        registry = ModelRegistry()
        if name in registry.list_models():
            _model = registry.create(name, model_config)
            # Actual fit requires training data -- placeholder for now.
            logger.info("Model '{}' created successfully (fit deferred)", name)
            return {
                "status": "created",
                "elapsed_seconds": round(time.monotonic() - t0, 3),
            }
        else:
            logger.warning("Model '{}' not in registry -- skipping", name)
            return {"status": "not_registered"}
    except Exception as exc:
        logger.error("Error fitting model '{}': {}", name, exc)
        return {
            "status": "error",
            "error": str(exc),
            "elapsed_seconds": round(time.monotonic() - t0, 3),
        }


def stage_scenario_generation(config: dict[str, Any]) -> dict[str, Any]:
    """Stage 4: Generate war-scenario Monte Carlo simulations.

    Produces price path simulations for each escalation level defined in
    scenarios.yaml, using the fitted models as generators.

    Args:
        config: Merged project configuration.

    Returns:
        Scenario results dictionary keyed by escalation level.
    """
    logger.info("=== STAGE 4: Scenario Generation ===")
    t0 = time.monotonic()

    scenarios_cfg = config.get("scenarios", {}).get("scenario", {})
    escalation_levels = scenarios_cfg.get("escalation_levels", [])
    mc_cfg = config.get("hyperparams", {}).get("monte_carlo", {})
    n_simulations = mc_cfg.get("n_simulations", 10_000)

    scenario_results: dict[str, Any] = {}

    for level_cfg in escalation_levels:
        level = level_cfg.get("level", "unknown")
        level_name = level_cfg.get("name", f"Level {level}")
        logger.info(
            "Generating scenarios for Level {} '{}' (n={})",
            level,
            level_name,
            n_simulations,
        )
        try:
            # Placeholder: ScenarioEngine dispatch will go here.
            scenario_results[f"level_{level}"] = {
                "name": level_name,
                "n_simulations": n_simulations,
                "status": "not_implemented",
                "probability": level_cfg.get("probability"),
            }
        except Exception as exc:
            logger.error("Scenario generation failed for level {}: {}", level, exc)
            scenario_results[f"level_{level}"] = {"status": "error", "error": str(exc)}

    elapsed = time.monotonic() - t0
    logger.info("Scenario generation completed in {:.1f}s", elapsed)
    return scenario_results


def stage_risk_analysis(
    config: dict[str, Any],
    scenario_results: dict[str, Any],
) -> dict[str, Any]:
    """Stage 5: Compute risk metrics from scenario outputs.

    Calculates VaR, CVaR, Expected Shortfall, stress-test impacts, and
    model risk bounds for each escalation level.

    Args:
        config: Merged project configuration.
        scenario_results: Output from :func:`stage_scenario_generation`.

    Returns:
        Risk analysis results dictionary.
    """
    logger.info("=== STAGE 5: Risk Analysis ===")
    t0 = time.monotonic()

    confidence_levels = (
        config.get("model_config", {})
        .get("pipeline", {})
        .get("confidence_levels", [0.90, 0.95, 0.99])
    )

    risk_results: dict[str, Any] = {}

    for level_key, level_data in scenario_results.items():
        try:
            logger.info("Computing risk metrics for {}", level_key)
            risk_results[level_key] = {
                "confidence_levels": confidence_levels,
                "status": "not_implemented",
            }
        except Exception as exc:
            logger.error("Risk analysis failed for {}: {}", level_key, exc)
            risk_results[level_key] = {"status": "error", "error": str(exc)}

    elapsed = time.monotonic() - t0
    logger.info("Risk analysis completed in {:.1f}s", elapsed)
    return risk_results


def stage_report(
    config: dict[str, Any],
    fit_results: dict[str, Any],
    scenario_results: dict[str, Any],
    risk_results: dict[str, Any],
) -> Path:
    """Stage 6: Generate the final analysis report.

    Delegates to the report generation script to produce a markdown
    document with embedded charts.

    Args:
        config: Merged project configuration.
        fit_results: Model fitting results.
        scenario_results: Scenario simulation results.
        risk_results: Risk metric results.

    Returns:
        Path to the generated report file.
    """
    logger.info("=== STAGE 6: Report Generation ===")
    t0 = time.monotonic()

    output_dir = PROJECT_ROOT / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "pipeline_report.md"

    try:
        # Placeholder: delegate to generate_report module.
        report_path.write_text(
            "# IranBSE Pipeline Report\n\n"
            f"Models fitted: {len(fit_results)}\n\n"
            f"Scenarios generated: {len(scenario_results)}\n\n"
            f"Risk metrics computed: {len(risk_results)}\n",
            encoding="utf-8",
        )
        logger.info("Report written to {}", report_path)
    except Exception as exc:
        logger.error("Report generation failed: {}", exc)

    elapsed = time.monotonic() - t0
    logger.info("Report generation completed in {:.1f}s", elapsed)
    return report_path


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_backtest(config: dict[str, Any]) -> dict[str, Any]:
    """Execute walk-forward backtests on historical conflict analogs.

    Args:
        config: Merged project configuration.

    Returns:
        Backtest results dictionary.
    """
    logger.info("=== BACKTEST MODE ===")
    t0 = time.monotonic()

    analogs = (
        config.get("scenarios", {})
        .get("scenario", {})
        .get("historical_analogs", [])
    )

    backtest_results: dict[str, Any] = {}

    for analog in analogs:
        event_name = analog.get("event", "Unknown")
        logger.info("Backtesting against analog: '{}'", event_name)
        try:
            backtest_results[event_name] = {
                "oil_peak_pct": analog.get("oil_peak_pct_change"),
                "wheat_peak_pct": analog.get("wheat_peak_pct_change"),
                "duration_days": analog.get("duration_to_peak_days"),
                "status": "not_implemented",
            }
        except Exception as exc:
            logger.error("Backtest failed for '{}': {}", event_name, exc)
            backtest_results[event_name] = {"status": "error", "error": str(exc)}

    elapsed = time.monotonic() - t0
    logger.info(
        "Backtesting completed in {:.1f}s ({} analogs)", elapsed, len(backtest_results)
    )
    return backtest_results


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the pipeline runner.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="IranBSE: Black Swan Event Modeling Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_pipeline.py\n"
            "  python scripts/run_pipeline.py --scenarios-only\n"
            "  python scripts/run_pipeline.py --backtest-only\n"
            "  python scripts/run_pipeline.py --models regime_switching extreme_value\n"
        ),
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--scenarios-only",
        action="store_true",
        default=False,
        help="Run only scenario generation (skip data fetch + model fitting).",
    )
    mode_group.add_argument(
        "--backtest-only",
        action="store_true",
        default=False,
        help="Run only backtesting on historical conflict analogs.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Explicit list of model names to fit (overrides config).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the master model_config.yaml.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for model fitting (-1 = all CPUs).",
    )
    parser.add_argument(
        "--log-level",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Minimum log level.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown_requested: bool = False


def _handle_signal(signum: int, _frame: Any) -> None:
    """Handle SIGINT / SIGTERM for graceful shutdown.

    Args:
        signum: Signal number received.
        _frame: Current stack frame (unused).
    """
    global _shutdown_requested
    _shutdown_requested = True
    sig_name = signal.Signals(signum).name
    logger.warning("Received {} -- requesting graceful shutdown", sig_name)


def _check_shutdown() -> None:
    """Raise ``SystemExit`` if a shutdown signal has been received."""
    if _shutdown_requested:
        logger.info("Graceful shutdown in progress")
        raise SystemExit(130)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Run the IranBSE analysis pipeline.

    Args:
        argv: Optional CLI argument list (for testing).

    Returns:
        Exit code (0 on success, non-zero on failure).
    """
    args = parse_args(argv)

    # Register signal handlers for graceful shutdown.
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Load configuration.
    try:
        config = load_all_configs(args.config)
    except Exception as exc:
        logger.critical("Failed to load configuration: {}", exc)
        return 1

    # Configure logging from config + CLI override.
    log_cfg = config.get("model_config", {}).get("logging", {})
    _configure_logging(
        level=args.log_level or log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file"),
    )

    logger.info(
        "IranBSE Pipeline starting (mode={})",
        "backtest" if args.backtest_only else ("scenarios" if args.scenarios_only else "full"),
    )

    pipeline_t0 = time.monotonic()

    try:
        # ----- Backtest-only mode -----
        if args.backtest_only:
            backtest_results = run_backtest(config)
            logger.info("Backtest results: {}", backtest_results)
            return 0

        # ----- Scenarios-only mode -----
        if args.scenarios_only:
            scenario_results = stage_scenario_generation(config)
            _check_shutdown()
            risk_results = stage_risk_analysis(config, scenario_results)
            _check_shutdown()
            stage_report(config, {}, scenario_results, risk_results)
            return 0

        # ----- Full pipeline -----
        fetch_results = stage_fetch_data(config)
        _check_shutdown()

        feature_results = stage_feature_engineering(config)
        _check_shutdown()

        fit_results = stage_model_fitting(config, args.models, args.n_jobs)
        _check_shutdown()

        scenario_results = stage_scenario_generation(config)
        _check_shutdown()

        risk_results = stage_risk_analysis(config, scenario_results)
        _check_shutdown()

        report_path = stage_report(config, fit_results, scenario_results, risk_results)
        _check_shutdown()

        pipeline_elapsed = time.monotonic() - pipeline_t0
        logger.info(
            "Pipeline completed successfully in {:.1f}s. Report: {}",
            pipeline_elapsed,
            report_path,
        )
        return 0

    except SystemExit as exc:
        logger.info("Pipeline terminated by signal (code={})", exc.code)
        return int(exc.code) if exc.code is not None else 130
    except Exception as exc:
        logger.exception("Pipeline failed with unexpected error: {}", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
