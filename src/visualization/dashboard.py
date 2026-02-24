"""Unified interactive dashboard for IranBSE scenario analysis.

Provides a Plotly Dash application with scenario selection, model
comparison, risk metrics, and real-time forecast visualisation.
Designed as the primary interface for analysts to explore conflict
scenario forecasts and their commodity market implications.

Typical usage::

    dashboard = Dashboard(models=model_dict, data=historical_data)
    dashboard.run(port=8050, debug=True)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots

from src.models.base_model import BaseModel, PredictionResult
from src.visualization.scenario_plots import (
    SCENARIO_COLOURS,
    plot_escalation_comparison,
    plot_scenario_fan_chart,
)


# Pre-defined escalation scenarios for the dashboard dropdown
DEFAULT_SCENARIOS: dict[str, dict[str, Any]] = {
    "baseline": {
        "label": "Baseline (No Conflict)",
        "description": "Status quo continuation with current sanctions regime.",
        "probability": 0.60,
    },
    "limited_strikes": {
        "label": "Limited Strikes",
        "description": "Targeted strikes on nuclear facilities; brief disruption.",
        "probability": 0.15,
    },
    "naval_confrontation": {
        "label": "Naval Confrontation",
        "description": "Strait of Hormuz partial blockade; tanker seizures.",
        "probability": 0.10,
    },
    "full_escalation": {
        "label": "Full Escalation",
        "description": "Extended military campaign; major infrastructure damage.",
        "probability": 0.10,
    },
    "hormuz_closure": {
        "label": "Hormuz Closure",
        "description": "Full closure of Strait of Hormuz; 17 mbd disrupted.",
        "probability": 0.05,
    },
}


class Dashboard:
    """Unified dashboard for IranBSE scenario analysis.

    Builds a Plotly Dash application with:
    - Scenario selector dropdown
    - Fan chart with confidence bands
    - Model comparison panel
    - Risk metrics summary
    - Sensitivity analysis controls

    The dashboard can run as a standalone web application or be embedded
    in a Jupyter notebook.

    Args:
        models: Mapping of model name to fitted :class:`BaseModel` instance.
        data: Historical data used for context and backtesting.
        scenarios: Scenario definitions for the dropdown.  If ``None``,
            uses the default escalation scenarios.
        title: Dashboard title.
    """

    def __init__(
        self,
        models: dict[str, BaseModel] | None = None,
        data: Any | None = None,
        scenarios: dict[str, dict[str, Any]] | None = None,
        title: str = "IranBSE Commodity Futures Scenario Dashboard",
    ) -> None:
        self.models = models or {}
        self.data = data
        self.scenarios = scenarios or DEFAULT_SCENARIOS
        self.title = title
        self._app: Any | None = None
        self._cached_results: dict[str, dict[str, PredictionResult]] = {}
        logger.info(
            "Dashboard initialised: {} models, {} scenarios",
            len(self.models),
            len(self.scenarios),
        )

    def build(self) -> Any:
        """Build the Dash application.

        Constructs the layout and registers callbacks.  The Dash
        library is imported lazily to avoid hard dependency for users
        who only need the static plotting functions.

        Returns:
            The Dash application instance.

        Raises:
            ImportError: If ``dash`` is not installed.
        """
        try:
            import dash
            from dash import Input, Output, dcc, html
        except ImportError as exc:
            raise ImportError(
                "The 'dash' package is required for the dashboard. "
                "Install it with: pip install dash"
            ) from exc

        app = dash.Dash(__name__, title=self.title)

        # ----- Layout -----
        app.layout = html.Div(
            [
                html.H1(
                    self.title,
                    style={
                        "textAlign": "center",
                        "fontFamily": "Arial, sans-serif",
                        "color": "#1565C0",
                        "marginBottom": "20px",
                    },
                ),
                # Controls row
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(
                                    "Scenario:",
                                    style={"fontWeight": "bold"},
                                ),
                                dcc.Dropdown(
                                    id="scenario-selector",
                                    options=[
                                        {
                                            "label": s["label"],
                                            "value": k,
                                        }
                                        for k, s in self.scenarios.items()
                                    ],
                                    value="baseline",
                                    clearable=False,
                                ),
                            ],
                            style={
                                "width": "30%",
                                "display": "inline-block",
                                "padding": "10px",
                            },
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Forecast Horizon (days):",
                                    style={"fontWeight": "bold"},
                                ),
                                dcc.Slider(
                                    id="horizon-slider",
                                    min=5,
                                    max=120,
                                    step=5,
                                    value=60,
                                    marks={
                                        h: str(h)
                                        for h in [5, 10, 20, 30, 60, 90, 120]
                                    },
                                ),
                            ],
                            style={
                                "width": "40%",
                                "display": "inline-block",
                                "padding": "10px",
                            },
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Model:",
                                    style={"fontWeight": "bold"},
                                ),
                                dcc.Dropdown(
                                    id="model-selector",
                                    options=[
                                        {"label": name, "value": name}
                                        for name in self.models
                                    ],
                                    value=(
                                        list(self.models.keys())[0]
                                        if self.models
                                        else None
                                    ),
                                    clearable=False,
                                ),
                            ],
                            style={
                                "width": "25%",
                                "display": "inline-block",
                                "padding": "10px",
                            },
                        ),
                    ],
                    style={
                        "backgroundColor": "#F5F5F5",
                        "borderRadius": "8px",
                        "marginBottom": "20px",
                        "padding": "10px",
                    },
                ),
                # Scenario description
                html.Div(
                    id="scenario-description",
                    style={
                        "padding": "10px",
                        "backgroundColor": "#E3F2FD",
                        "borderRadius": "8px",
                        "marginBottom": "20px",
                        "fontFamily": "Arial, sans-serif",
                    },
                ),
                # Main forecast chart
                dcc.Graph(id="forecast-chart"),
                # Model comparison and risk metrics row
                html.Div(
                    [
                        html.Div(
                            [dcc.Graph(id="comparison-chart")],
                            style={
                                "width": "50%",
                                "display": "inline-block",
                            },
                        ),
                        html.Div(
                            [dcc.Graph(id="risk-chart")],
                            style={
                                "width": "50%",
                                "display": "inline-block",
                            },
                        ),
                    ]
                ),
            ],
            style={
                "maxWidth": "1400px",
                "margin": "0 auto",
                "padding": "20px",
                "fontFamily": "Arial, sans-serif",
            },
        )

        # ----- Callbacks -----
        @app.callback(
            Output("scenario-description", "children"),
            Input("scenario-selector", "value"),
        )
        def update_description(scenario: str) -> str:
            s = self.scenarios.get(scenario, {})
            label = s.get("label", scenario)
            desc = s.get("description", "")
            prob = s.get("probability", 0)
            return (
                f"{label}: {desc} "
                f"(Estimated probability: {prob:.0%})"
            )

        @app.callback(
            Output("forecast-chart", "figure"),
            Input("scenario-selector", "value"),
            Input("horizon-slider", "value"),
            Input("model-selector", "value"),
        )
        def update_forecast(
            scenario: str,
            horizon: int,
            model_name: str,
        ) -> go.Figure:
            if not model_name or model_name not in self.models:
                return self._empty_figure("Select a model")

            result = self._get_prediction(model_name, scenario, horizon)
            if result is None:
                return self._empty_figure("Model prediction unavailable")

            return plot_scenario_fan_chart(
                result,
                title=f"{scenario.replace('_', ' ').title()} - {model_name}",
                show_scenarios=True,
                max_scenarios_shown=30,
            )

        @app.callback(
            Output("comparison-chart", "figure"),
            Input("horizon-slider", "value"),
        )
        def update_comparison(horizon: int) -> go.Figure:
            results: dict[str, PredictionResult] = {}
            for scenario_name in self.scenarios:
                for model_name in list(self.models.keys())[:1]:
                    result = self._get_prediction(
                        model_name, scenario_name, horizon
                    )
                    if result is not None:
                        results[scenario_name] = result

            if not results:
                return self._empty_figure("No model results available")

            return plot_escalation_comparison(
                results,
                title="Scenario Comparison",
                height=400,
                width=650,
            )

        @app.callback(
            Output("risk-chart", "figure"),
            Input("scenario-selector", "value"),
            Input("horizon-slider", "value"),
        )
        def update_risk(scenario: str, horizon: int) -> go.Figure:
            return self._build_risk_summary(scenario, horizon)

        self._app = app
        logger.info("Dashboard built successfully")
        return app

    def run(
        self,
        host: str = "127.0.0.1",
        port: int = 8050,
        debug: bool = False,
    ) -> None:
        """Start the dashboard web server.

        Args:
            host: Host address to bind.
            port: Port number.
            debug: Enable Dash debug mode with hot reloading.
        """
        if self._app is None:
            self.build()

        assert self._app is not None
        logger.info("Starting dashboard at http://{}:{}", host, port)
        self._app.run(host=host, port=port, debug=debug)

    def create_static_report(
        self,
        horizon: int = 60,
        n_scenarios: int = 1000,
    ) -> go.Figure:
        """Create a static multi-panel report figure (no Dash required).

        Useful for generating PDF/PNG exports without running the web
        server.

        Args:
            horizon: Forecast horizon.
            n_scenarios: Monte-Carlo scenarios per model.

        Returns:
            Multi-panel Plotly Figure.
        """
        n_scenarios_in = len(self.scenarios)
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Scenario Forecasts",
                "Risk Metrics",
                "Model Weights",
                "Scenario Probabilities",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "pie"}],
            ],
        )

        # Panel 1: Scenario forecasts
        colours = list(SCENARIO_COLOURS.values())
        for i, (scenario_name, scenario_info) in enumerate(
            self.scenarios.items()
        ):
            for model_name in list(self.models.keys())[:1]:
                result = self._get_prediction(
                    model_name, scenario_name, horizon, n_scenarios
                )
                if result is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(horizon)),
                            y=result.point_forecast[:horizon],
                            mode="lines",
                            name=scenario_info.get("label", scenario_name),
                            line={
                                "color": colours[i % len(colours)],
                                "width": 2,
                            },
                        ),
                        row=1,
                        col=1,
                    )

        # Panel 2: Risk metrics placeholder
        risk_metrics = ["VaR 95%", "CVaR 95%", "Max DD", "Tail Index"]
        risk_values = [0.05, 0.08, 0.15, 3.2]  # Placeholder values
        fig.add_trace(
            go.Bar(
                x=risk_metrics,
                y=risk_values,
                marker_color=["#2196F3", "#F44336", "#FF9800", "#4CAF50"],
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Panel 3: Model weights placeholder
        if self.models:
            model_names = list(self.models.keys())
            weights = [1.0 / len(model_names)] * len(model_names)
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=weights,
                    marker_color="#2196F3",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        # Panel 4: Scenario probabilities
        labels = [
            s.get("label", k) for k, s in self.scenarios.items()
        ]
        probs = [s.get("probability", 0) for s in self.scenarios.values()]
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=probs,
                marker_colors=colours[: len(labels)],
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title=self.title,
            height=900,
            width=1200,
            template="plotly_white",
        )

        logger.info("Static report created: {} scenarios", n_scenarios_in)
        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_prediction(
        self,
        model_name: str,
        scenario_name: str,
        horizon: int,
        n_scenarios: int = 500,
    ) -> PredictionResult | None:
        """Get or compute a cached prediction.

        Args:
            model_name: Name of the model.
            scenario_name: Scenario identifier.
            horizon: Forecast horizon.
            n_scenarios: Monte-Carlo scenarios.

        Returns:
            PredictionResult or None if model is not available.
        """
        cache_key = f"{model_name}_{scenario_name}_{horizon}"

        if cache_key in self._cached_results:
            return self._cached_results.get(cache_key, {}).get("result")  # type: ignore[union-attr]

        model = self.models.get(model_name)
        if model is None:
            return None

        try:
            result = model.predict(horizon, n_scenarios)
            self._cached_results[cache_key] = {"result": result}  # type: ignore[assignment]
            return result
        except RuntimeError:
            logger.warning(
                "Model '{}' not fitted; cannot predict", model_name
            )
            return None

    def _build_risk_summary(
        self,
        scenario: str,
        horizon: int,
    ) -> go.Figure:
        """Build a risk metrics summary chart for the selected scenario.

        Args:
            scenario: Active scenario name.
            horizon: Forecast horizon.

        Returns:
            Plotly Figure with risk metric bars.
        """
        fig = go.Figure()

        # Gather predictions across models for uncertainty estimation
        all_predictions: list[list[float]] = []
        for model_name in self.models:
            result = self._get_prediction(model_name, scenario, horizon)
            if result is not None:
                all_predictions.append(result.point_forecast[:horizon])

        if not all_predictions:
            return self._empty_figure("No predictions for risk analysis")

        pred_matrix = np.array(all_predictions, dtype=np.float64)

        # Compute risk metrics from the ensemble
        ensemble_std = np.mean(np.std(pred_matrix, axis=0))
        model_spread = np.mean(
            np.max(pred_matrix, axis=0) - np.min(pred_matrix, axis=0)
        )
        max_drawdown_proxy = float(
            np.max(
                np.max(pred_matrix, axis=1) - np.min(pred_matrix, axis=1)
            )
        )

        metrics = {
            "Ensemble Std": float(ensemble_std),
            "Model Spread": float(model_spread),
            "Max Range": float(max_drawdown_proxy),
        }

        fig.add_trace(
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                marker_color=["#2196F3", "#FF9800", "#F44336"],
            )
        )

        fig.update_layout(
            title=f"Risk Metrics - {scenario.replace('_', ' ').title()}",
            height=400,
            width=650,
            template="plotly_white",
            yaxis_title="Value",
        )

        return fig

    @staticmethod
    def _empty_figure(message: str) -> go.Figure:
        """Create an empty figure with a centered message.

        Args:
            message: Message to display.

        Returns:
            Empty Plotly Figure with annotation.
        """
        fig = go.Figure()
        fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": message,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 16, "color": "grey"},
                    "x": 0.5,
                    "y": 0.5,
                }
            ],
            template="plotly_white",
        )
        return fig
