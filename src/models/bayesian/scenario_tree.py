"""Bayesian scenario tree for escalation-conditional commodity forecasts.

Constructs a discrete scenario tree where each node represents a
geopolitical escalation level (e.g. diplomatic tension, proxy conflict,
direct engagement, full-scale war) and the associated conditional
distribution of commodity price paths.

The tree structure is:

    Root (current state)
      |-- Node 1: De-escalation (prob p_1)
      |     |-- Price distribution F_1(y | de-escalation)
      |-- Node 2: Status quo (prob p_2)
      |     |-- Price distribution F_2(y | status_quo)
      |-- Node 3: Limited conflict (prob p_3)
      |     |-- Price distribution F_3(y | limited_conflict)
      |-- Node 4: Full war (prob p_4)
            |-- Price distribution F_4(y | full_war)

Each price distribution is modelled as a Bayesian regression conditioned
on the escalation level, with posterior predictive draws providing full
distributional forecasts.

Example::

    config = ModelConfig(
        name="scenario_tree",
        params={
            "commodities": ["oil", "wheat"],
            "escalation_levels": [0.0, 0.3, 0.6, 1.0],
            "escalation_priors": [0.3, 0.35, 0.25, 0.1],
        },
    )
    model = ScenarioTreeModel(config)
    model.fit(train_df)
    tree = model.generate_tree(horizon=20)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
import pymc as pm
import arviz as az
from loguru import logger

from src.models.base_model import BaseModel, ModelConfig, ModelState, PredictionResult


# ---------------------------------------------------------------------------
# Tree data structures
# ---------------------------------------------------------------------------


@dataclass
class ScenarioNode:
    """A single node in the scenario tree.

    Attributes:
        name: Human-readable scenario label.
        escalation_level: Numeric escalation intensity in [0, 1].
        probability: Prior or posterior probability of this scenario.
        price_paths: Simulated price trajectories conditioned on this
            scenario, shape ``(n_draws, horizon, n_commodities)``.
        children: Sub-scenarios branching from this node.
    """

    name: str
    escalation_level: float
    probability: float
    price_paths: np.ndarray | None = None
    children: list["ScenarioNode"] = field(default_factory=list)

    def summary_stats(self) -> dict[str, Any]:
        """Compute summary statistics for this node's paths.

        Returns:
            Dictionary with mean, median, quantile forecasts.
        """
        if self.price_paths is None:
            return {"warning": "No paths generated"}
        paths = self.price_paths
        return {
            "name": self.name,
            "escalation_level": self.escalation_level,
            "probability": self.probability,
            "mean": paths.mean(axis=0).tolist(),
            "median": np.median(paths, axis=0).tolist(),
            "q05": np.quantile(paths, 0.05, axis=0).tolist(),
            "q95": np.quantile(paths, 0.95, axis=0).tolist(),
        }


@dataclass
class ScenarioTree:
    """Complete scenario tree rooted at the current state.

    Attributes:
        root: The root node representing the current state.
        horizon: Forecast horizon in time steps.
        n_commodities: Number of modelled commodities.
    """

    root: ScenarioNode
    horizon: int
    n_commodities: int

    def expected_paths(self) -> np.ndarray:
        """Compute probability-weighted expected paths across scenarios.

        Returns:
            Array of shape ``(horizon, n_commodities)``.
        """
        total = np.zeros((self.horizon, self.n_commodities))
        for child in self.root.children:
            if child.price_paths is not None:
                total += child.probability * child.price_paths.mean(axis=0)
        return total

    def to_dict(self) -> dict[str, Any]:
        """Serialise the tree to a nested dictionary."""
        return {
            "horizon": self.horizon,
            "n_commodities": self.n_commodities,
            "scenarios": [child.summary_stats() for child in self.root.children],
            "expected_paths": self.expected_paths().tolist(),
        }


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------


class ScenarioTreeModel(BaseModel):
    """Bayesian scenario tree for escalation-conditional commodity forecasts.

    Builds a discrete tree of geopolitical scenarios and computes
    posterior predictive price distributions conditioned on each
    scenario's escalation level.

    Config params:
        commodities: List of commodity names (default ``["oil", "wheat"]``).
        escalation_col: Column with the escalation index.
        escalation_levels: Discrete escalation levels defining tree nodes
            (default ``[0.0, 0.3, 0.6, 1.0]``).
        escalation_labels: Human-readable labels for each level.
        escalation_priors: Prior probabilities for each level
            (default uniform).
        draws: Posterior draws per scenario (default 1000).
        tune: MCMC tuning steps (default 500).
        chains: Number of chains (default 2).
    """

    _DEFAULT_LEVELS: list[float] = [0.0, 0.3, 0.6, 1.0]
    _DEFAULT_LABELS: list[str] = [
        "de_escalation",
        "status_quo",
        "limited_conflict",
        "full_war",
    ]

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._commodities: list[str] = config.params.get(
            "commodities", ["oil", "wheat"]
        )
        self._escalation_col: str = config.params.get(
            "escalation_col", "escalation_index"
        )
        self._levels: list[float] = config.params.get(
            "escalation_levels", self._DEFAULT_LEVELS
        )
        self._labels: list[str] = config.params.get(
            "escalation_labels", self._DEFAULT_LABELS[: len(self._levels)]
        )
        self._priors: list[float] = config.params.get("escalation_priors", [])
        if not self._priors:
            n = len(self._levels)
            self._priors = [1.0 / n] * n

        self._draws: int = config.params.get("draws", 1000)
        self._tune: int = config.params.get("tune", 500)
        self._chains: int = config.params.get("chains", 2)

        # Fitted artefacts
        self._trace: az.InferenceData | None = None
        self._Y_train: np.ndarray | None = None
        self._esc_train: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def fit(self, data: pl.DataFrame) -> None:
        """Fit the scenario-conditional regression model.

        The model is:
            y_{c,t} = alpha_c + beta_c * esc_t + gamma_c * y_{c,t-1}
                      + delta_c * esc_t * y_{c,t-1} + sigma_c * eps_t

        This captures both the direct effect of escalation and
        its interaction with price momentum.

        Args:
            data: DataFrame with commodity return columns and escalation.
        """
        target_cols = [f"{c}_return" for c in self._commodities]
        required = [self._escalation_col, *target_cols]
        self._validate_data(data, required_columns=required, min_rows=50)

        K = len(self._commodities)
        escalation = data[self._escalation_col].to_numpy().astype(np.float64)
        Y = np.column_stack(
            [data[f"{c}_return"].to_numpy().astype(np.float64) for c in self._commodities]
        )
        Y_lag = np.vstack([Y[:1], Y[:-1]])
        interaction = escalation[:, np.newaxis] * Y_lag

        self._Y_train = Y
        self._esc_train = escalation

        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=0, sigma=1, shape=K)
            beta = pm.Normal("beta", mu=0, sigma=2, shape=K)
            gamma = pm.Normal("gamma", mu=0, sigma=0.5, shape=K)
            delta = pm.Normal("delta", mu=0, sigma=1, shape=K)
            sigma = pm.HalfNormal("sigma", sigma=1, shape=K)

            mu_y = (
                alpha[np.newaxis, :]
                + beta[np.newaxis, :] * escalation[:, np.newaxis]
                + gamma[np.newaxis, :] * Y_lag
                + delta[np.newaxis, :] * interaction
            )
            pm.Normal("y_obs", mu=mu_y, sigma=sigma[np.newaxis, :], observed=Y)

            logger.info("Sampling scenario tree model: {} draws", self._draws)
            self._trace = pm.sample(
                draws=self._draws,
                tune=self._tune,
                chains=self._chains,
                return_inferencedata=True,
                progressbar=True,
            )

        self._mark_fitted(data)

    def predict(
        self,
        horizon: int,
        n_scenarios: int = 1000,
    ) -> PredictionResult:
        """Generate probability-weighted forecasts across all scenarios.

        Args:
            horizon: Forecast horizon.
            n_scenarios: Draws per scenario node.

        Returns:
            PredictionResult with expected (weighted) forecasts.
        """
        self._require_fitted()
        tree = self.generate_tree(horizon=horizon, n_draws=n_scenarios)
        expected = tree.expected_paths()

        # Primary commodity
        point = expected[:, 0].tolist()

        # Aggregate uncertainty from all scenarios
        all_paths: list[np.ndarray] = []
        for child in tree.root.children:
            if child.price_paths is not None:
                all_paths.append(child.price_paths[:, :, 0])
        if all_paths:
            stacked = np.concatenate(all_paths, axis=0)
            lower = np.quantile(stacked, 0.05, axis=0).tolist()
            upper = np.quantile(stacked, 0.95, axis=0).tolist()
        else:
            lower = point
            upper = point

        return PredictionResult(
            point_forecast=point,
            lower_bounds={0.05: lower},
            upper_bounds={0.95: upper},
            scenarios={
                child.name: child.price_paths[:, :, 0].mean(axis=0).tolist()
                for child in tree.root.children
                if child.price_paths is not None
            },
            metadata={
                "model": self.config.name,
                "scenario_probabilities": {
                    child.name: child.probability
                    for child in tree.root.children
                },
            },
        )

    def get_params(self) -> dict[str, Any]:
        """Return posterior summaries for scenario parameters."""
        self._require_fitted()
        assert self._trace is not None

        summary = az.summary(
            self._trace,
            var_names=["alpha", "beta", "gamma", "delta", "sigma"],
            hdi_prob=0.95,
        )
        return {
            "commodities": self._commodities,
            "escalation_levels": self._levels,
            "escalation_labels": self._labels,
            "priors": self._priors,
            "posterior_summary": summary.to_dict(),
        }

    # ------------------------------------------------------------------
    # Tree generation
    # ------------------------------------------------------------------

    def generate_tree(
        self,
        horizon: int = 20,
        n_draws: int = 1000,
    ) -> ScenarioTree:
        """Build the full scenario tree with posterior predictive paths.

        Args:
            horizon: Forecast horizon.
            n_draws: Posterior draws per scenario node.

        Returns:
            :class:`ScenarioTree` with conditional paths at each node.
        """
        self._require_fitted()
        K = len(self._commodities)

        root = ScenarioNode(
            name="current_state",
            escalation_level=float(self._esc_train[-1]) if self._esc_train is not None else 0.0,
            probability=1.0,
        )

        for level, label, prob in zip(
            self._levels, self._labels, self._priors
        ):
            paths = self._simulate_scenario(level, horizon, n_draws)
            node = ScenarioNode(
                name=label,
                escalation_level=level,
                probability=prob,
                price_paths=paths,
            )
            root.children.append(node)

        tree = ScenarioTree(root=root, horizon=horizon, n_commodities=K)
        logger.info(
            "Generated scenario tree: {} scenarios x {} steps",
            len(self._levels),
            horizon,
        )
        return tree

    def scenario_forecast(
        self,
        scenario_label: str,
        horizon: int = 20,
        n_draws: int = 1000,
    ) -> PredictionResult:
        """Forecast conditioned on a specific named scenario.

        Args:
            scenario_label: One of the configured escalation labels.
            horizon: Forecast horizon.
            n_draws: Posterior draws.

        Returns:
            PredictionResult for the specified scenario.
        """
        self._require_fitted()
        if scenario_label not in self._labels:
            raise ValueError(
                f"Unknown scenario '{scenario_label}'. "
                f"Available: {self._labels}"
            )
        idx = self._labels.index(scenario_label)
        level = self._levels[idx]

        paths = self._simulate_scenario(level, horizon, n_draws)
        # Use first commodity as primary
        primary = paths[:, :, 0]

        return PredictionResult(
            point_forecast=primary.mean(axis=0).tolist(),
            lower_bounds={0.05: np.quantile(primary, 0.05, axis=0).tolist()},
            upper_bounds={0.95: np.quantile(primary, 0.95, axis=0).tolist()},
            scenarios={
                self._commodities[i]: paths[:, :, i].mean(axis=0).tolist()
                for i in range(len(self._commodities))
            },
            metadata={
                "scenario": scenario_label,
                "escalation_level": level,
                "n_draws": n_draws,
            },
        )

    def posterior_predictive(
        self,
        escalation_path: list[float],
        n_draws: int = 1000,
    ) -> np.ndarray:
        """Sample posterior predictive paths for an arbitrary escalation path.

        Args:
            escalation_path: Escalation index at each future step.
            n_draws: Number of posterior draws.

        Returns:
            Array of shape ``(n_draws, len(escalation_path), n_commodities)``.
        """
        self._require_fitted()
        return self._simulate_with_path(escalation_path, n_draws)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _simulate_scenario(
        self,
        escalation_level: float,
        horizon: int,
        n_draws: int,
    ) -> np.ndarray:
        """Simulate posterior predictive paths at a fixed escalation level.

        Args:
            escalation_level: Constant escalation value for all steps.
            horizon: Forecast steps.
            n_draws: Number of draws.

        Returns:
            Paths ``(n_draws, horizon, K)``.
        """
        path = [escalation_level] * horizon
        return self._simulate_with_path(path, n_draws)

    def _simulate_with_path(
        self,
        escalation_path: list[float],
        n_draws: int,
    ) -> np.ndarray:
        """Simulate forward using posterior parameter draws.

        Args:
            escalation_path: Escalation at each step.
            n_draws: Number of posterior draws.

        Returns:
            Paths ``(n_draws, len(path), K)``.
        """
        assert self._trace is not None
        assert self._Y_train is not None

        K = len(self._commodities)
        horizon = len(escalation_path)

        posterior = self._trace.posterior
        n_total = posterior.dims["chain"] * posterior.dims["draw"]
        n_use = min(n_draws, n_total)

        alpha = posterior["alpha"].values.reshape(-1, K)[:n_use]
        beta = posterior["beta"].values.reshape(-1, K)[:n_use]
        gamma_p = posterior["gamma"].values.reshape(-1, K)[:n_use]
        delta_p = posterior["delta"].values.reshape(-1, K)[:n_use]
        sigma = posterior["sigma"].values.reshape(-1, K)[:n_use]

        rng = np.random.default_rng(42)
        paths = np.zeros((n_use, horizon, K))
        y_prev = np.tile(self._Y_train[-1], (n_use, 1))

        for t in range(horizon):
            esc = escalation_path[t]
            mu = (
                alpha
                + beta * esc
                + gamma_p * y_prev
                + delta_p * esc * y_prev
            )
            eps = sigma * rng.standard_normal((n_use, K))
            paths[:, t, :] = mu + eps
            y_prev = paths[:, t, :]

        return paths
