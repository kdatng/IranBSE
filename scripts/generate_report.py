#!/usr/bin/env python3
"""Report generation script for IranBSE scenario analysis.

Loads model outputs and scenario simulation results, computes summary
statistics, generates visualization plots, and produces a markdown report
with embedded charts.

Usage::

    # Generate full report from latest pipeline outputs
    python scripts/generate_report.py

    # Custom input/output directories
    python scripts/generate_report.py --input-dir outputs/run_20260315 --output-dir reports/

    # Skip chart generation (text-only report)
    python scripts/generate_report.py --no-charts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
CONFIG_DIR = PROJECT_ROOT / "config"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scenario_results(input_dir: Path) -> dict[str, Any]:
    """Load scenario simulation results from disk.

    Searches for JSON or parquet files containing scenario outputs.

    Args:
        input_dir: Directory containing pipeline output files.

    Returns:
        Dictionary of scenario results keyed by escalation level.
    """
    results: dict[str, Any] = {}

    # Try JSON first.
    json_path = input_dir / "scenario_results.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as fh:
            results = json.load(fh)
        logger.info("Loaded scenario results from {}", json_path)
        return results

    # Try individual parquet files.
    parquet_files = sorted(input_dir.glob("scenario_level_*.parquet"))
    if parquet_files:
        try:
            import polars as pl
            for pf in parquet_files:
                level_key = pf.stem  # e.g. "scenario_level_1"
                df = pl.read_parquet(pf)
                results[level_key] = {
                    "data": df.to_dict(),
                    "n_simulations": df.height,
                    "columns": df.columns,
                }
            logger.info("Loaded {} scenario parquet files", len(parquet_files))
        except ImportError:
            logger.warning("polars not available -- cannot read parquet files")

    if not results:
        logger.warning("No scenario results found in {}", input_dir)

    return results


def load_model_outputs(input_dir: Path) -> dict[str, Any]:
    """Load fitted model outputs from disk.

    Args:
        input_dir: Directory containing model output files.

    Returns:
        Dictionary of model outputs keyed by model name.
    """
    results: dict[str, Any] = {}

    json_path = input_dir / "model_outputs.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as fh:
            results = json.load(fh)
        logger.info("Loaded model outputs from {}", json_path)
        return results

    # Search for individual model output files.
    for model_file in sorted(input_dir.glob("model_*.json")):
        model_name = model_file.stem.replace("model_", "")
        with open(model_file, "r", encoding="utf-8") as fh:
            results[model_name] = json.load(fh)

    if results:
        logger.info("Loaded {} model output files", len(results))
    else:
        logger.warning("No model outputs found in {}", input_dir)

    return results


def load_config() -> dict[str, Any]:
    """Load project configuration files for report context.

    Returns:
        Merged configuration dictionary.
    """
    config: dict[str, Any] = {}
    for name in ("model_config", "scenarios", "hyperparams"):
        path = CONFIG_DIR / f"{name}.yaml"
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                config[name] = yaml.safe_load(fh) or {}
    return config


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary_statistics(
    scenario_results: dict[str, Any],
) -> dict[str, Any]:
    """Compute aggregate statistics across all scenario levels.

    Args:
        scenario_results: Scenario simulation output.

    Returns:
        Dictionary of summary statistics per level and aggregate.
    """
    summary: dict[str, Any] = {"levels": {}, "aggregate": {}}

    all_oil_peaks: list[float] = []
    all_wheat_peaks: list[float] = []

    for level_key, level_data in scenario_results.items():
        data = level_data.get("data", {})

        # Extract oil and wheat price paths if available.
        oil_peaks = data.get("oil_peak_pct", [])
        wheat_peaks = data.get("wheat_peak_pct", [])

        if oil_peaks:
            oil_arr = np.array(oil_peaks, dtype=np.float64)
            oil_arr = oil_arr[~np.isnan(oil_arr)]
            all_oil_peaks.extend(oil_arr.tolist())
        else:
            oil_arr = np.array([], dtype=np.float64)

        if wheat_peaks:
            wheat_arr = np.array(wheat_peaks, dtype=np.float64)
            wheat_arr = wheat_arr[~np.isnan(wheat_arr)]
            all_wheat_peaks.extend(wheat_arr.tolist())
        else:
            wheat_arr = np.array([], dtype=np.float64)

        level_stats: dict[str, Any] = {
            "n_simulations": level_data.get("n_simulations", 0),
        }

        if len(oil_arr) > 0:
            level_stats["oil"] = {
                "mean": float(np.mean(oil_arr)),
                "median": float(np.median(oil_arr)),
                "std": float(np.std(oil_arr)),
                "p5": float(np.percentile(oil_arr, 5)),
                "p25": float(np.percentile(oil_arr, 25)),
                "p75": float(np.percentile(oil_arr, 75)),
                "p95": float(np.percentile(oil_arr, 95)),
                "p99": float(np.percentile(oil_arr, 99)),
                "max": float(np.max(oil_arr)),
            }

        if len(wheat_arr) > 0:
            level_stats["wheat"] = {
                "mean": float(np.mean(wheat_arr)),
                "median": float(np.median(wheat_arr)),
                "std": float(np.std(wheat_arr)),
                "p5": float(np.percentile(wheat_arr, 5)),
                "p95": float(np.percentile(wheat_arr, 95)),
                "p99": float(np.percentile(wheat_arr, 99)),
                "max": float(np.max(wheat_arr)),
            }

        summary["levels"][level_key] = level_stats

    # Aggregate across all levels (probability-weighted would require config).
    if all_oil_peaks:
        oil_all = np.array(all_oil_peaks)
        summary["aggregate"]["oil_mean"] = float(np.mean(oil_all))
        summary["aggregate"]["oil_p95"] = float(np.percentile(oil_all, 95))
        summary["aggregate"]["oil_p99"] = float(np.percentile(oil_all, 99))

    if all_wheat_peaks:
        wheat_all = np.array(all_wheat_peaks)
        summary["aggregate"]["wheat_mean"] = float(np.mean(wheat_all))
        summary["aggregate"]["wheat_p95"] = float(np.percentile(wheat_all, 95))
        summary["aggregate"]["wheat_p99"] = float(np.percentile(wheat_all, 99))

    return summary


def extract_key_findings(
    summary: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    """Distill the analysis into key findings for the executive summary.

    Args:
        summary: Computed summary statistics.
        config: Project configuration for context.

    Returns:
        List of key finding strings.
    """
    findings: list[str] = []

    scenarios_cfg = config.get("scenarios", {}).get("scenario", {})
    escalation_levels = scenarios_cfg.get("escalation_levels", [])

    # Finding 1: Most probable scenario.
    if escalation_levels:
        most_probable = max(escalation_levels, key=lambda x: x.get("probability", 0))
        findings.append(
            f"Most probable escalation level: **{most_probable.get('name', 'Unknown')}** "
            f"(p={most_probable.get('probability', 0):.0%}) -- "
            f"{most_probable.get('description', '')}"
        )

    # Finding 2: Oil price aggregate.
    agg = summary.get("aggregate", {})
    if "oil_mean" in agg:
        findings.append(
            f"Expected oil price impact (all scenarios): "
            f"mean +{agg['oil_mean']:.1f}%, "
            f"95th percentile +{agg['oil_p95']:.1f}%, "
            f"99th percentile +{agg['oil_p99']:.1f}%"
        )

    # Finding 3: Wheat price aggregate.
    if "wheat_mean" in agg:
        findings.append(
            f"Expected wheat price impact: "
            f"mean +{agg['wheat_mean']:.1f}%, "
            f"95th percentile +{agg['wheat_p95']:.1f}%"
        )

    # Finding 4: Hormuz closure risk.
    hormuz_cfg = scenarios_cfg.get("strait_of_hormuz", {})
    bypass_pct = hormuz_cfg.get("bypass_coverage_pct", 0)
    if bypass_pct:
        findings.append(
            f"Strait of Hormuz bypass capacity covers only "
            f"{bypass_pct:.0%} of daily flow -- "
            f"full closure would remove ~{hormuz_cfg.get('daily_flow_mbpd', 0):.1f} mb/d "
            f"from global markets"
        )

    # Finding 5: Mine clearing timeline.
    mine_duration = hormuz_cfg.get("mine_clearing_duration_weeks", [])
    if mine_duration:
        findings.append(
            f"Mine clearing timeline estimated at {mine_duration[0]}-{mine_duration[1]} weeks, "
            f"implying prolonged supply disruption even after hostilities cease"
        )

    if not findings:
        findings.append(
            "No simulation data available yet. Run the full pipeline to "
            "generate scenario results."
        )

    return findings


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def generate_scenario_fan_chart(
    scenario_results: dict[str, Any],
    output_dir: Path,
) -> Path | None:
    """Generate a fan chart showing price path distributions per scenario.

    Args:
        scenario_results: Scenario simulation output.
        output_dir: Directory to save the chart.

    Returns:
        Path to the generated PNG, or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available -- skipping fan chart")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("IranBSE Scenario Analysis: Price Impact Distribution", fontsize=14)

    level_names = []
    oil_means: list[float] = []
    wheat_means: list[float] = []

    for level_key in sorted(scenario_results.keys()):
        level_data = scenario_results[level_key]
        name = level_data.get("name", level_key)
        level_names.append(name)

        data = level_data.get("data", {})
        oil_vals = data.get("oil_peak_pct", [])
        wheat_vals = data.get("wheat_peak_pct", [])

        oil_means.append(float(np.mean(oil_vals)) if oil_vals else 0.0)
        wheat_means.append(float(np.mean(wheat_vals)) if wheat_vals else 0.0)

    if oil_means:
        axes[0].barh(level_names, oil_means, color="#c0392b", alpha=0.8)
        axes[0].set_xlabel("Mean Oil Price Change (%)")
        axes[0].set_title("Oil (WTI/Brent)")
    else:
        axes[0].text(0.5, 0.5, "No oil data", ha="center", va="center", transform=axes[0].transAxes)

    if wheat_means:
        axes[1].barh(level_names, wheat_means, color="#d4ac0d", alpha=0.8)
        axes[1].set_xlabel("Mean Wheat Price Change (%)")
        axes[1].set_title("Wheat (CBOT)")
    else:
        axes[1].text(0.5, 0.5, "No wheat data", ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    chart_path = output_dir / "scenario_fan_chart.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Fan chart saved to {}", chart_path)
    return chart_path


def generate_risk_heatmap(
    summary: dict[str, Any],
    output_dir: Path,
) -> Path | None:
    """Generate a risk heatmap showing percentile impacts across levels.

    Args:
        summary: Computed summary statistics.
        output_dir: Directory to save the chart.

    Returns:
        Path to the generated PNG, or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available -- skipping risk heatmap")
        return None

    levels = summary.get("levels", {})
    if not levels:
        logger.warning("No level data for heatmap")
        return None

    level_names: list[str] = []
    metrics: list[str] = ["p5", "p25", "median", "p75", "p95", "p99"]
    matrix: list[list[float]] = []

    for level_key in sorted(levels.keys()):
        level_data = levels[level_key]
        oil_stats = level_data.get("oil", {})
        if oil_stats:
            level_names.append(level_key)
            row = [oil_stats.get(m, 0.0) for m in metrics]
            matrix.append(row)

    if not matrix:
        logger.warning("Insufficient data for risk heatmap")
        return None

    fig, ax = plt.subplots(figsize=(10, max(4, len(level_names) * 0.8)))
    data = np.array(matrix)

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(range(len(level_names)))
    ax.set_yticklabels(level_names)
    ax.set_title("Oil Price Impact Percentiles by Scenario Level")

    # Annotate cells.
    for i in range(len(level_names)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{data[i, j]:.1f}%", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, label="Price Change (%)")
    plt.tight_layout()

    chart_path = output_dir / "risk_heatmap.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Risk heatmap saved to {}", chart_path)
    return chart_path


def generate_historical_comparison(
    config: dict[str, Any],
    output_dir: Path,
) -> Path | None:
    """Generate a comparison chart of historical conflict analogs.

    Args:
        config: Project configuration containing historical analogs.
        output_dir: Directory to save the chart.

    Returns:
        Path to the generated PNG, or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available -- skipping comparison chart")
        return None

    analogs = (
        config.get("scenarios", {})
        .get("scenario", {})
        .get("historical_analogs", [])
    )

    if not analogs:
        logger.warning("No historical analogs in config")
        return None

    events = [a.get("event", "?") for a in analogs]
    oil_changes = [a.get("oil_peak_pct_change", 0) for a in analogs]
    wheat_changes = [a.get("wheat_peak_pct_change", 0) for a in analogs]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(events))
    width = 0.35  # JUSTIFIED: standard grouped bar chart width

    ax.bar(x - width / 2, oil_changes, width, label="Oil", color="#c0392b", alpha=0.85)
    ax.bar(x + width / 2, wheat_changes, width, label="Wheat", color="#d4ac0d", alpha=0.85)

    ax.set_ylabel("Peak Price Change (%)")
    ax.set_title("Historical Conflict Analogs: Peak Commodity Impact")
    ax.set_xticks(x)
    ax.set_xticklabels(events, rotation=35, ha="right", fontsize=8)
    ax.legend()
    ax.axhline(y=0, color="grey", linewidth=0.5)

    plt.tight_layout()
    chart_path = output_dir / "historical_comparison.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Historical comparison chart saved to {}", chart_path)
    return chart_path


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_markdown_report(
    summary: dict[str, Any],
    findings: list[str],
    config: dict[str, Any],
    chart_paths: dict[str, Path | None],
    output_path: Path,
) -> Path:
    """Assemble the final markdown report.

    Args:
        summary: Computed summary statistics.
        findings: Key finding strings.
        config: Project configuration.
        chart_paths: Mapping of chart names to file paths.
        output_path: Path to write the markdown file.

    Returns:
        Path to the generated report.
    """
    scenarios_cfg = config.get("scenarios", {}).get("scenario", {})
    model_cfg = config.get("model_config", {})
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []

    # Header.
    lines.append("# IranBSE Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}")
    lines.append(f"**Scenario:** {scenarios_cfg.get('name', 'N/A')}")
    lines.append(
        f"**Period:** {scenarios_cfg.get('start_date', 'N/A')} to "
        f"{scenarios_cfg.get('end_date', 'N/A')}"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Executive summary.
    lines.append("## Executive Summary")
    lines.append("")
    for i, finding in enumerate(findings, 1):
        lines.append(f"{i}. {finding}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Escalation levels table.
    lines.append("## Escalation Levels")
    lines.append("")
    escalation_levels = scenarios_cfg.get("escalation_levels", [])
    if escalation_levels:
        lines.append(
            "| Level | Name | Probability | Oil Disruption | "
            "Hormuz Closure | Wheat Disruption |"
        )
        lines.append("|-------|------|-------------|----------------|----------------|------------------|")
        for lvl in escalation_levels:
            oil_range = lvl.get("oil_supply_disruption_pct", [0, 0])
            wheat_range = lvl.get("wheat_trade_disruption_pct", [0, 0])
            lines.append(
                f"| {lvl.get('level', '?')} "
                f"| {lvl.get('name', '?')} "
                f"| {lvl.get('probability', 0):.0%} "
                f"| {oil_range[0]}-{oil_range[1]}% "
                f"| {lvl.get('hormuz_closure_probability', 0):.0%} "
                f"| {wheat_range[0]}-{wheat_range[1]}% |"
            )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary statistics.
    lines.append("## Summary Statistics")
    lines.append("")
    levels_stats = summary.get("levels", {})
    for level_key in sorted(levels_stats.keys()):
        stats = levels_stats[level_key]
        lines.append(f"### {level_key}")
        lines.append("")
        lines.append(f"- Simulations: {stats.get('n_simulations', 'N/A')}")

        oil = stats.get("oil", {})
        if oil:
            lines.append(f"- **Oil price impact:** mean +{oil.get('mean', 0):.1f}%, "
                          f"median +{oil.get('median', 0):.1f}%, "
                          f"95th pctl +{oil.get('p95', 0):.1f}%")

        wheat = stats.get("wheat", {})
        if wheat:
            lines.append(f"- **Wheat price impact:** mean +{wheat.get('mean', 0):.1f}%, "
                          f"95th pctl +{wheat.get('p95', 0):.1f}%")
        lines.append("")

    # Aggregate.
    agg = summary.get("aggregate", {})
    if agg:
        lines.append("### Aggregate (All Levels)")
        lines.append("")
        if "oil_mean" in agg:
            lines.append(
                f"- Oil: mean +{agg['oil_mean']:.1f}%, "
                f"P95 +{agg['oil_p95']:.1f}%, "
                f"P99 +{agg['oil_p99']:.1f}%"
            )
        if "wheat_mean" in agg:
            lines.append(
                f"- Wheat: mean +{agg['wheat_mean']:.1f}%, "
                f"P95 +{agg['wheat_p95']:.1f}%, "
                f"P99 +{agg['wheat_p99']:.1f}%"
            )
        lines.append("")

    lines.append("---")
    lines.append("")

    # Charts.
    lines.append("## Visualizations")
    lines.append("")
    for chart_name, chart_path in chart_paths.items():
        if chart_path and chart_path.exists():
            # Use relative path from report location.
            try:
                rel = chart_path.relative_to(output_path.parent)
            except ValueError:
                rel = chart_path
            lines.append(f"### {chart_name.replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"![{chart_name}]({rel})")
            lines.append("")

    lines.append("---")
    lines.append("")

    # Model inventory.
    lines.append("## Models Used")
    lines.append("")
    enabled = model_cfg.get("models", {}).get("enabled", {})
    for category, names in enabled.items():
        if isinstance(names, list):
            lines.append(f"- **{category}**: {', '.join(names)}")
    lines.append("")

    # Footer.
    lines.append("---")
    lines.append("")
    lines.append("*This report was auto-generated by the IranBSE pipeline. "
                  "All forecasts are scenario-based projections, not trading recommendations.*")

    report_text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    logger.info("Report written to {} ({} lines)", output_path, len(lines))
    return output_path


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
        prog="generate_report",
        description="Generate IranBSE analysis report with charts.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing pipeline outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated report and charts.",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        default=False,
        help="Skip chart generation (text-only report).",
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
    """Generate the analysis report.

    Args:
        argv: Optional CLI argument list.

    Returns:
        Exit code (0 on success).
    """
    args = parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    t0 = time.monotonic()

    # Load data.
    config = load_config()
    scenario_results = load_scenario_results(args.input_dir)
    _model_outputs = load_model_outputs(args.input_dir)

    # Compute statistics.
    summary = compute_summary_statistics(scenario_results)
    findings = extract_key_findings(summary, config)

    # Generate charts.
    chart_paths: dict[str, Path | None] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_charts:
        chart_paths["scenario_fan_chart"] = generate_scenario_fan_chart(
            scenario_results, args.output_dir
        )
        chart_paths["risk_heatmap"] = generate_risk_heatmap(
            summary, args.output_dir
        )
        chart_paths["historical_comparison"] = generate_historical_comparison(
            config, args.output_dir
        )

    # Generate report.
    report_path = args.output_dir / "analysis_report.md"
    generate_markdown_report(summary, findings, config, chart_paths, report_path)

    elapsed = time.monotonic() - t0
    logger.info("Report generation completed in {:.1f}s", elapsed)
    logger.info("Report: {}", report_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
