"""Fan charts and probability cone visualisations for scenario forecasts.

Provides publication-quality Plotly visualisations of probabilistic
commodity futures forecasts, including fan charts with multiple confidence
bands, escalation scenario comparisons, and probability cone diagrams.

Typical usage::

    fig = plot_scenario_fan_chart(result, title="Brent Crude 60-day Forecast")
    fig = plot_escalation_comparison(results_dict)
    fig = plot_probability_cones(result, horizons=[5, 10, 20, 40, 60])
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from loguru import logger
from numpy.typing import NDArray

from src.models.base_model import PredictionResult


# Colour palette: escalating severity from calm blue to crisis red
SCENARIO_COLOURS: dict[str, str] = {
    "baseline": "#2196F3",
    "limited_strikes": "#FF9800",
    "naval_confrontation": "#F44336",
    "full_escalation": "#9C27B0",
    "hormuz_closure": "#D50000",
}

# Band colours with transparency for fan charts
BAND_COLOURS: list[str] = [
    "rgba(33, 150, 243, 0.10)",   # 90% band (widest)
    "rgba(33, 150, 243, 0.20)",   # 80% band
    "rgba(33, 150, 243, 0.30)",   # 50% band (narrowest)
]


def plot_scenario_fan_chart(
    result: PredictionResult,
    title: str = "Scenario Forecast",
    x_label: str = "Days Forward",
    y_label: str = "Price (USD)",
    historical_prices: list[float] | None = None,
    historical_label: str = "Historical",
    show_scenarios: bool = True,
    max_scenarios_shown: int = 50,
    height: int = 600,
    width: int = 1000,
) -> go.Figure:
    """Create a fan chart with confidence bands and optional scenario paths.

    The fan chart shows the point forecast as a solid line, surrounded by
    progressively wider confidence bands representing different probability
    levels.  Individual Monte Carlo paths can be shown as faint traces.

    Args:
        result: PredictionResult containing point forecast, bounds, and
            optional scenarios.
        title: Chart title.
        x_label: X-axis label.
        y_label: Y-axis label.
        historical_prices: Optional list of recent historical prices to
            prepend for context.
        historical_label: Legend label for historical data.
        show_scenarios: Whether to show individual scenario paths.
        max_scenarios_shown: Cap on number of scenario paths to display
            (for performance).
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    horizon = len(result.point_forecast)
    x_forecast = list(range(horizon))

    # Historical prices (if provided)
    x_offset = 0
    if historical_prices:
        n_hist = len(historical_prices)
        x_hist = list(range(-n_hist, 0))
        fig.add_trace(
            go.Scatter(
                x=x_hist,
                y=historical_prices,
                mode="lines",
                name=historical_label,
                line={"color": "#455A64", "width": 2},
            )
        )

    # Confidence bands (from widest to narrowest)
    sorted_lower = sorted(result.lower_bounds.items(), key=lambda x: x[0])
    sorted_upper = sorted(result.upper_bounds.items(), key=lambda x: x[0], reverse=True)

    band_pairs: list[tuple[float, list[float], float, list[float]]] = []
    for (lk, lv), (uk, uv) in zip(sorted_lower, sorted_upper):
        band_pairs.append((lk, lv, uk, uv))

    for i, (lower_level, lower_vals, upper_level, upper_vals) in enumerate(
        band_pairs
    ):
        colour = BAND_COLOURS[i % len(BAND_COLOURS)]
        coverage = int((upper_level - lower_level) * 100)

        fig.add_trace(
            go.Scatter(
                x=x_forecast + x_forecast[::-1],
                y=upper_vals[:horizon]
                + lower_vals[:horizon][::-1],
                fill="toself",
                fillcolor=colour,
                line={"color": "rgba(0,0,0,0)"},
                name=f"{coverage}% CI",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    # Individual scenario paths
    if show_scenarios and result.scenarios:
        scenario_keys = sorted(result.scenarios.keys())[:max_scenarios_shown]
        for j, key in enumerate(scenario_keys):
            path = result.scenarios[key][:horizon]
            fig.add_trace(
                go.Scatter(
                    x=x_forecast[: len(path)],
                    y=path,
                    mode="lines",
                    line={"color": "rgba(33, 150, 243, 0.05)", "width": 0.5},
                    showlegend=j == 0,
                    name="MC paths" if j == 0 else None,
                    hoverinfo="skip",
                )
            )

    # Point forecast (on top)
    fig.add_trace(
        go.Scatter(
            x=x_forecast,
            y=result.point_forecast,
            mode="lines",
            name="Point Forecast",
            line={"color": "#1565C0", "width": 3},
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=height,
        width=width,
        template="plotly_white",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
        hovermode="x unified",
    )

    # Add vertical line at t=0 if historical data shown
    if historical_prices:
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="grey",
            annotation_text="Forecast Start",
        )

    logger.info("Fan chart created: {} horizon steps", horizon)
    return fig


def plot_escalation_comparison(
    results: dict[str, PredictionResult],
    title: str = "Escalation Scenario Comparison",
    x_label: str = "Days Forward",
    y_label: str = "Price (USD)",
    height: int = 600,
    width: int = 1000,
) -> go.Figure:
    """Compare point forecasts across multiple escalation scenarios.

    Each scenario is plotted as a separate line with its own confidence
    band, using a colour palette that escalates from blue (calm) to red
    (crisis).

    Args:
        results: Mapping of scenario name to PredictionResult.
        title: Chart title.
        x_label: X-axis label.
        y_label: Y-axis label.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    default_colours = list(SCENARIO_COLOURS.values())

    for i, (scenario_name, result) in enumerate(results.items()):
        colour = SCENARIO_COLOURS.get(
            scenario_name, default_colours[i % len(default_colours)]
        )
        horizon = len(result.point_forecast)
        x = list(range(horizon))

        # 90% confidence band for this scenario
        lower_key = min(result.lower_bounds.keys()) if result.lower_bounds else None
        upper_key = max(result.upper_bounds.keys()) if result.upper_bounds else None

        if lower_key is not None and upper_key is not None:
            lower_vals = result.lower_bounds[lower_key][:horizon]
            upper_vals = result.upper_bounds[upper_key][:horizon]

            # Convert colour to rgba for transparency
            r, g, b = _hex_to_rgb(colour)
            band_colour = f"rgba({r}, {g}, {b}, 0.15)"

            fig.add_trace(
                go.Scatter(
                    x=x + x[::-1],
                    y=upper_vals + lower_vals[::-1],
                    fill="toself",
                    fillcolor=band_colour,
                    line={"color": "rgba(0,0,0,0)"},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Point forecast
        fig.add_trace(
            go.Scatter(
                x=x,
                y=result.point_forecast[:horizon],
                mode="lines",
                name=scenario_name.replace("_", " ").title(),
                line={"color": colour, "width": 2.5},
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=height,
        width=width,
        template="plotly_white",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
        hovermode="x unified",
    )

    logger.info(
        "Escalation comparison chart created: {} scenarios",
        len(results),
    )
    return fig


def plot_probability_cones(
    result: PredictionResult,
    horizons: list[int] | None = None,
    title: str = "Probability Cones",
    y_label: str = "Price (USD)",
    height: int = 500,
    width: int = 900,
) -> go.Figure:
    """Plot probability distribution cones at specific horizons.

    Shows the forecast distribution as violin/box plots at selected
    future time points, giving a clear picture of uncertainty growth
    over the forecast horizon.

    Args:
        result: PredictionResult with scenario paths.
        horizons: List of specific horizon days to show cones for.
            Defaults to ``[5, 10, 20, 40, 60]``.
        title: Chart title.
        y_label: Y-axis label.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.
    """
    if horizons is None:
        max_h = len(result.point_forecast)
        horizons = [h for h in [5, 10, 20, 40, 60] if h <= max_h]
        if not horizons:
            horizons = [max_h]

    fig = go.Figure()

    # Extract scenario values at each horizon
    if result.scenarios:
        for h in horizons:
            idx = min(h - 1, len(result.point_forecast) - 1)
            values = []
            for path in result.scenarios.values():
                if len(path) > idx:
                    values.append(path[idx])

            if values:
                fig.add_trace(
                    go.Violin(
                        x=[f"Day {h}"] * len(values),
                        y=values,
                        name=f"Day {h}",
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor="rgba(33, 150, 243, 0.3)",
                        line_color="#1565C0",
                    )
                )

    # Add point forecast markers
    point_vals = []
    point_labels = []
    for h in horizons:
        idx = min(h - 1, len(result.point_forecast) - 1)
        point_vals.append(result.point_forecast[idx])
        point_labels.append(f"Day {h}")

    fig.add_trace(
        go.Scatter(
            x=point_labels,
            y=point_vals,
            mode="markers",
            name="Point Forecast",
            marker={
                "color": "#D50000",
                "size": 12,
                "symbol": "diamond",
            },
        )
    )

    fig.update_layout(
        title=title,
        yaxis_title=y_label,
        height=height,
        width=width,
        template="plotly_white",
        showlegend=True,
        violingap=0.3,
    )

    logger.info(
        "Probability cones created at horizons: {}",
        horizons,
    )
    return fig


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _hex_to_rgb(hex_colour: str) -> tuple[int, int, int]:
    """Convert hex colour string to RGB tuple.

    Args:
        hex_colour: Hex colour string (e.g. ``"#FF0000"``).

    Returns:
        Tuple of (red, green, blue) integers in [0, 255].
    """
    hex_colour = hex_colour.lstrip("#")
    return (
        int(hex_colour[0:2], 16),
        int(hex_colour[2:4], 16),
        int(hex_colour[4:6], 16),
    )
