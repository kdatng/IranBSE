"""Sensitivity analysis visualisations: tornado diagrams, heatmaps, comparisons.

Provides Plotly-based visualisations for parameter sensitivity analysis,
model comparison, and feature importance assessment.

Typical usage::

    fig = plot_sensitivity_tornado(sensitivities, title="Oil Price Sensitivity")
    fig = plot_parameter_heatmap(param_grid, predictions)
    fig = plot_model_comparison(model_results)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots


def plot_sensitivity_tornado(
    sensitivities: dict[str, tuple[float, float]],
    baseline: float = 0.0,
    title: str = "Parameter Sensitivity (Tornado Diagram)",
    x_label: str = "Impact on Forecast",
    height: int = 500,
    width: int = 800,
) -> go.Figure:
    """Create a tornado diagram showing parameter sensitivities.

    Each parameter is shown as a horizontal bar extending from its
    low-scenario impact (left) to its high-scenario impact (right),
    sorted by total range.

    Args:
        sensitivities: Mapping of parameter name to
            ``(low_impact, high_impact)`` tuple.  Low impact is the
            forecast change when the parameter is at its low value;
            high impact when at its high value.
        baseline: Baseline forecast value (centre of the tornado).
        title: Chart title.
        x_label: X-axis label.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.
    """
    # Sort by total range (widest bar on top)
    sorted_params = sorted(
        sensitivities.items(),
        key=lambda x: abs(x[1][1] - x[1][0]),
        reverse=True,
    )

    param_names = [p[0] for p in sorted_params]
    low_vals = [p[1][0] for p in sorted_params]
    high_vals = [p[1][1] for p in sorted_params]

    fig = go.Figure()

    # Low-scenario bars (extending left from baseline)
    fig.add_trace(
        go.Bar(
            y=param_names,
            x=[v - baseline for v in low_vals],
            orientation="h",
            name="Low Scenario",
            marker_color="#2196F3",
            base=baseline,
        )
    )

    # High-scenario bars (extending right from baseline)
    fig.add_trace(
        go.Bar(
            y=param_names,
            x=[v - baseline for v in high_vals],
            orientation="h",
            name="High Scenario",
            marker_color="#F44336",
            base=baseline,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        barmode="overlay",
        height=height,
        width=width,
        template="plotly_white",
        yaxis={"autorange": "reversed"},
    )

    # Add baseline vertical line
    fig.add_vline(
        x=baseline,
        line_dash="dash",
        line_color="black",
        line_width=1,
        annotation_text=f"Baseline: {baseline:.2f}",
    )

    logger.info(
        "Tornado diagram created with {} parameters", len(sensitivities)
    )
    return fig


def plot_parameter_heatmap(
    param1_name: str,
    param1_values: list[float],
    param2_name: str,
    param2_values: list[float],
    predictions: list[list[float]],
    title: str = "Parameter Sensitivity Heatmap",
    colorscale: str = "RdYlBu_r",
    height: int = 500,
    width: int = 600,
) -> go.Figure:
    """Create a heatmap showing forecast sensitivity to two parameters.

    Args:
        param1_name: Name of the first parameter (y-axis).
        param1_values: Values tested for the first parameter.
        param2_name: Name of the second parameter (x-axis).
        param2_values: Values tested for the second parameter.
        predictions: 2D grid of prediction values.
            ``predictions[i][j]`` is the forecast when
            ``param1=param1_values[i]`` and ``param2=param2_values[j]``.
        title: Chart title.
        colorscale: Plotly colour scale name.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.
    """
    z_matrix = np.array(predictions, dtype=np.float64)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=[f"{v:.3g}" for v in param2_values],
            y=[f"{v:.3g}" for v in param1_values],
            colorscale=colorscale,
            colorbar_title="Forecast",
            hovertemplate=(
                f"{param2_name}: %{{x}}<br>"
                f"{param1_name}: %{{y}}<br>"
                "Forecast: %{z:.4f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=param2_name,
        yaxis_title=param1_name,
        height=height,
        width=width,
        template="plotly_white",
    )

    logger.info(
        "Parameter heatmap created: {}x{} grid",
        len(param1_values),
        len(param2_values),
    )
    return fig


def plot_model_comparison(
    model_results: dict[str, dict[str, float]],
    metrics: list[str] | None = None,
    title: str = "Model Performance Comparison",
    height: int = 500,
    width: int = 900,
) -> go.Figure:
    """Create a grouped bar chart comparing models across metrics.

    Args:
        model_results: Mapping of model name to metric dictionary.
            Example::

                {
                    "GARCH": {"rmse": 1.2, "mae": 0.8, "dir_acc": 0.65},
                    "BVAR": {"rmse": 1.1, "mae": 0.9, "dir_acc": 0.70},
                }
        metrics: Subset of metrics to display.  If ``None``, shows all.
        title: Chart title.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.
    """
    if not model_results:
        raise ValueError("model_results must not be empty")

    model_names = list(model_results.keys())

    if metrics is None:
        # Union of all metric keys
        all_metrics: set[str] = set()
        for m_dict in model_results.values():
            all_metrics.update(m_dict.keys())
        metrics = sorted(all_metrics)

    fig = go.Figure()

    colours = [
        "#2196F3",
        "#F44336",
        "#4CAF50",
        "#FF9800",
        "#9C27B0",
        "#00BCD4",
        "#795548",
    ]

    for i, model_name in enumerate(model_names):
        values = [
            model_results[model_name].get(m, 0.0) for m in metrics
        ]
        fig.add_trace(
            go.Bar(
                name=model_name,
                x=metrics,
                y=values,
                marker_color=colours[i % len(colours)],
            )
        )

    fig.update_layout(
        title=title,
        barmode="group",
        height=height,
        width=width,
        template="plotly_white",
        xaxis_title="Metric",
        yaxis_title="Value",
        legend={
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "right",
            "x": 0.99,
        },
    )

    logger.info(
        "Model comparison chart created: {} models, {} metrics",
        len(model_names),
        len(metrics),
    )
    return fig


def plot_feature_importance(
    importances: dict[str, float],
    top_n: int = 20,
    title: str = "Feature Importance",
    height: int = 500,
    width: int = 800,
) -> go.Figure:
    """Create a horizontal bar chart of feature importances.

    Args:
        importances: Mapping of feature name to importance score.
        top_n: Number of top features to show.
        title: Chart title.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.
    """
    sorted_features = sorted(
        importances.items(), key=lambda x: abs(x[1]), reverse=True
    )[:top_n]

    names = [f[0] for f in sorted_features][::-1]
    values = [f[1] for f in sorted_features][::-1]

    colours = [
        "#F44336" if v >= 0 else "#2196F3" for v in values
    ]

    fig = go.Figure(
        go.Bar(
            y=names,
            x=values,
            orientation="h",
            marker_color=colours,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        height=height,
        width=width,
        template="plotly_white",
    )

    logger.info(
        "Feature importance chart created: top {} of {} features",
        top_n,
        len(importances),
    )
    return fig


def plot_residual_diagnostics(
    actuals: list[float],
    predictions: list[float],
    title: str = "Residual Diagnostics",
    height: int = 800,
    width: int = 900,
) -> go.Figure:
    """Create a four-panel residual diagnostic plot.

    Panels:
    1. Residuals over time
    2. Residual histogram
    3. Q-Q plot
    4. Predicted vs actual scatter

    Args:
        actuals: Realised values.
        predictions: Model predictions.
        title: Overall figure title.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure with 4 subplots.
    """
    n = min(len(actuals), len(predictions))
    a = np.array(actuals[:n], dtype=np.float64)
    p = np.array(predictions[:n], dtype=np.float64)
    residuals = a - p

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Residuals Over Time",
            "Residual Distribution",
            "Q-Q Plot",
            "Predicted vs Actual",
        ),
    )

    # Panel 1: Residuals over time
    fig.add_trace(
        go.Scatter(
            x=list(range(n)),
            y=residuals.tolist(),
            mode="markers+lines",
            marker={"color": "#2196F3", "size": 4},
            line={"color": "#2196F3", "width": 0.5},
            name="Residuals",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # Panel 2: Histogram
    fig.add_trace(
        go.Histogram(
            x=residuals.tolist(),
            nbinsx=30,
            marker_color="#4CAF50",
            name="Residuals",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Panel 3: Q-Q plot
    sorted_residuals = np.sort(residuals)
    from scipy import stats as sp_stats

    theoretical = sp_stats.norm.ppf(
        (np.arange(1, n + 1) - 0.5) / n
    )
    fig.add_trace(
        go.Scatter(
            x=theoretical.tolist(),
            y=sorted_residuals.tolist(),
            mode="markers",
            marker={"color": "#FF9800", "size": 4},
            name="Q-Q",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    # 45-degree reference line
    qq_min = float(min(theoretical.min(), sorted_residuals.min()))
    qq_max = float(max(theoretical.max(), sorted_residuals.max()))
    fig.add_trace(
        go.Scatter(
            x=[qq_min, qq_max],
            y=[qq_min, qq_max],
            mode="lines",
            line={"color": "red", "dash": "dash"},
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Panel 4: Predicted vs Actual
    fig.add_trace(
        go.Scatter(
            x=p.tolist(),
            y=a.tolist(),
            mode="markers",
            marker={"color": "#9C27B0", "size": 5},
            name="Pred vs Actual",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    # 45-degree reference line
    pa_min = float(min(p.min(), a.min()))
    pa_max = float(max(p.max(), a.max()))
    fig.add_trace(
        go.Scatter(
            x=[pa_min, pa_max],
            y=[pa_min, pa_max],
            mode="lines",
            line={"color": "red", "dash": "dash"},
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=title,
        height=height,
        width=width,
        template="plotly_white",
    )

    logger.info("Residual diagnostics created: {} observations", n)
    return fig
