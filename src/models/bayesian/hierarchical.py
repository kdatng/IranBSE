"""Hierarchical Bayesian model for cross-commodity shock propagation.

Implements a hierarchical Bayesian structure that shares information
across commodity groups (crude oil, wheat) to estimate the impact of
geopolitical escalation on commodity futures.  The hierarchy enables
partial pooling: commodities within the same group (e.g. energy) share
a common prior on shock sensitivity while retaining commodity-specific
posterior estimates.

Inference is performed via PyMC's NUTS sampler with adaptive step-size
tuning.

Example::

    config = ModelConfig(
        name="hier_bayes",
        params={
            "commodities": ["oil", "wheat"],
            "escalation_col": "escalation_index",
            "draws": 2000,
            "tune": 1000,
            "chains": 4,
        },
    )
    model = HierarchicalBayesModel(config)
    model.fit(train_df)
    result = model.predict(horizon=20)
    model.trace_diagnostics()
"""

from __future__ import annotations

from typing import Any

import arviz as az
import numpy as np
import polars as pl
import pymc as pm
from loguru import logger

from src.models.base_model import BaseModel, ModelConfig, ModelState, PredictionResult


class HierarchicalBayesModel(BaseModel):
    """Hierarchical Bayesian model for multi-commodity geopolitical shocks.

    Implements a two-level hierarchy:

    1. **Group level** -- commodities in the same sector (energy, agriculture)
       share hyper-priors on shock sensitivity and volatility parameters.
    2. **Commodity level** -- each commodity draws its own parameters from
       the group-level distribution, allowing partial pooling.

    The observation model is:

        y_{c,t} = alpha_c + beta_c * escalation_t
                  + gamma_c * y_{c,t-1} + sigma_c * epsilon_t

    with group-level priors:

        alpha_c ~ N(mu_alpha_g, tau_alpha_g)
        beta_c  ~ N(mu_beta_g,  tau_beta_g)

    Config params:
        commodities: List of commodity names (target column suffixes).
        escalation_col: Column with geopolitical escalation index.
        group_map: Dict mapping commodity -> group (e.g. {"oil": "energy"}).
            If not supplied, all commodities share one group.
        draws: Number of posterior draws (default 2000).
        tune: Tuning / burn-in steps (default 1000).
        chains: Number of MCMC chains (default 4).
        target_accept: NUTS target acceptance rate (default 0.9).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._commodities: list[str] = config.params.get(
            "commodities", ["oil", "wheat"]
        )
        self._escalation_col: str = config.params.get(
            "escalation_col", "escalation_index"
        )
        self._group_map: dict[str, str] = config.params.get("group_map", {})
        if not self._group_map:
            self._group_map = {c: "default" for c in self._commodities}

        self._draws: int = config.params.get("draws", 2000)
        self._tune: int = config.params.get("tune", 1000)
        self._chains: int = config.params.get("chains", 4)
        self._target_accept: float = config.params.get("target_accept", 0.9)

        self._trace: az.InferenceData | None = None
        self._model: pm.Model | None = None
        self._groups: list[str] = sorted(set(self._group_map.values()))
        self._commodity_idx: dict[str, int] = {}
        self._group_idx: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def fit(self, data: pl.DataFrame) -> None:
        """Fit the hierarchical model via NUTS sampling.

        Args:
            data: DataFrame with columns for each commodity target
                (``"{commodity}_return"``), the escalation index, and
                lagged features.
        """
        target_cols = [f"{c}_return" for c in self._commodities]
        required = [self._escalation_col, *target_cols]
        self._validate_data(data, required_columns=required, min_rows=50)

        self._commodity_idx = {c: i for i, c in enumerate(self._commodities)}
        self._group_idx = {g: i for i, g in enumerate(self._groups)}
        commodity_group = np.array(
            [self._group_idx[self._group_map[c]] for c in self._commodities]
        )

        n_commodities = len(self._commodities)
        n_groups = len(self._groups)

        # Build data arrays
        escalation = data[self._escalation_col].to_numpy().astype(np.float64)
        Y = np.column_stack(
            [data[f"{c}_return"].to_numpy().astype(np.float64) for c in self._commodities]
        )  # (T, n_commodities)
        Y_lag = np.vstack([Y[:1], Y[:-1]])  # simple lag

        with pm.Model() as model:
            # --- Group-level hyper-priors ---
            mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=1, shape=n_groups)
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.5, shape=n_groups)

            mu_beta = pm.Normal("mu_beta", mu=0, sigma=1, shape=n_groups)
            sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.5, shape=n_groups)

            # --- Commodity-level parameters ---
            alpha = pm.Normal(
                "alpha",
                mu=mu_alpha[commodity_group],
                sigma=sigma_alpha[commodity_group],
                shape=n_commodities,
            )
            beta = pm.Normal(
                "beta",
                mu=mu_beta[commodity_group],
                sigma=sigma_beta[commodity_group],
                shape=n_commodities,
            )
            gamma = pm.Normal("gamma", mu=0.0, sigma=0.5, shape=n_commodities)
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0, shape=n_commodities)

            # --- Observation model ---
            # y_{c,t} = alpha_c + beta_c * esc_t + gamma_c * y_{c,t-1} + eps
            mu_y = (
                alpha[np.newaxis, :]
                + beta[np.newaxis, :] * escalation[:, np.newaxis]
                + gamma[np.newaxis, :] * Y_lag
            )
            pm.Normal(
                "y_obs",
                mu=mu_y,
                sigma=sigma_obs[np.newaxis, :],
                observed=Y,
                shape=Y.shape,
            )

            # --- Sampling ---
            logger.info(
                "Sampling {} draws x {} chains (tune={})",
                self._draws,
                self._chains,
                self._tune,
            )
            self._trace = pm.sample(
                draws=self._draws,
                tune=self._tune,
                chains=self._chains,
                target_accept=self._target_accept,
                return_inferencedata=True,
                progressbar=True,
            )
            self._model = model

        self._mark_fitted(data)

    def predict(
        self,
        horizon: int,
        n_scenarios: int = 1000,
    ) -> PredictionResult:
        """Generate multi-step forecasts via posterior predictive sampling.

        This draws future paths by iteratively sampling from the
        fitted model's posterior predictive distribution.

        Args:
            horizon: Number of forward steps.
            n_scenarios: Number of posterior draws to use.

        Returns:
            PredictionResult with aggregate statistics across commodities.
        """
        self._require_fitted()
        assert self._trace is not None

        posterior = self._trace.posterior
        # Draw parameter samples
        n_total = self._draws * self._chains
        n_use = min(n_scenarios, n_total)

        alpha = posterior["alpha"].values.reshape(-1, len(self._commodities))[:n_use]
        beta = posterior["beta"].values.reshape(-1, len(self._commodities))[:n_use]
        gamma_p = posterior["gamma"].values.reshape(-1, len(self._commodities))[:n_use]
        sigma = posterior["sigma_obs"].values.reshape(-1, len(self._commodities))[:n_use]

        # Simulate forward with escalation=0 (baseline) -- callers should
        # use scenario_predict for conditional forecasts
        rng = np.random.default_rng(42)
        paths = np.zeros((n_use, horizon, len(self._commodities)))
        y_prev = np.zeros((n_use, len(self._commodities)))

        for t in range(horizon):
            esc = 0.0  # baseline scenario
            mu = alpha + beta * esc + gamma_p * y_prev
            paths[:, t, :] = mu + sigma * rng.standard_normal(mu.shape)
            y_prev = paths[:, t, :]

        # Aggregate over commodities (equal-weight average)
        agg = paths.mean(axis=-1)  # (n_use, horizon)
        point = agg.mean(axis=0).tolist()
        lower = np.quantile(agg, 0.05, axis=0).tolist()
        upper = np.quantile(agg, 0.95, axis=0).tolist()

        return PredictionResult(
            point_forecast=point,
            lower_bounds={0.05: lower},
            upper_bounds={0.95: upper},
            scenarios={
                c: paths[:, :, i].mean(axis=0).tolist()
                for i, c in enumerate(self._commodities)
            },
            metadata={"model": self.config.name, "n_draws_used": n_use},
        )

    def get_params(self) -> dict[str, Any]:
        """Return posterior summary statistics."""
        self._require_fitted()
        return self.posterior_summary()

    # ------------------------------------------------------------------
    # Posterior diagnostics
    # ------------------------------------------------------------------

    def posterior_summary(self) -> dict[str, Any]:
        """Compute posterior mean and 95 % HDI for key parameters.

        Returns:
            Dictionary mapping parameter names to summary statistics.
        """
        self._require_fitted()
        assert self._trace is not None

        summary_df = az.summary(
            self._trace,
            var_names=["alpha", "beta", "gamma", "sigma_obs"],
            hdi_prob=0.95,
        )
        result: dict[str, Any] = {
            "commodities": self._commodities,
            "groups": self._groups,
        }
        for param in ["alpha", "beta", "gamma", "sigma_obs"]:
            rows = summary_df.loc[summary_df.index.str.startswith(param)]
            result[param] = {
                "mean": rows["mean"].tolist(),
                "hdi_2.5%": rows["hdi_2.5%"].tolist(),
                "hdi_97.5%": rows["hdi_97.5%"].tolist(),
                "r_hat": rows["r_hat"].tolist(),
            }
        return result

    def trace_diagnostics(self) -> dict[str, Any]:
        """Run convergence diagnostics on the MCMC trace.

        Returns:
            Dictionary with R-hat, ESS, and divergence counts.
        """
        self._require_fitted()
        assert self._trace is not None

        rhat = az.rhat(self._trace)
        ess = az.ess(self._trace)

        # Check for divergences
        sample_stats = self._trace.sample_stats
        n_divergences = int(sample_stats["diverging"].values.sum())

        diag: dict[str, Any] = {
            "n_divergences": n_divergences,
            "max_rhat": {},
            "min_ess": {},
        }
        for var_name in ["alpha", "beta", "gamma", "sigma_obs"]:
            if var_name in rhat:
                diag["max_rhat"][var_name] = float(rhat[var_name].values.max())
            if var_name in ess:
                diag["min_ess"][var_name] = float(ess[var_name].values.min())

        if n_divergences > 0:
            logger.warning(
                "{} divergent transitions detected -- consider increasing "
                "target_accept or reparameterising",
                n_divergences,
            )
        else:
            logger.info("No divergent transitions detected")

        return diag

    def scenario_predict(
        self,
        horizon: int,
        escalation_trajectory: list[float],
        n_scenarios: int = 1000,
    ) -> PredictionResult:
        """Generate forecasts conditional on an escalation trajectory.

        Args:
            horizon: Number of forward steps.
            escalation_trajectory: Escalation index values for each step.
                Must have length >= *horizon*.
            n_scenarios: Number of posterior draws to use.

        Returns:
            PredictionResult conditioned on the given escalation path.
        """
        self._require_fitted()
        assert self._trace is not None
        assert len(escalation_trajectory) >= horizon

        posterior = self._trace.posterior
        n_total = self._draws * self._chains
        n_use = min(n_scenarios, n_total)

        alpha = posterior["alpha"].values.reshape(-1, len(self._commodities))[:n_use]
        beta = posterior["beta"].values.reshape(-1, len(self._commodities))[:n_use]
        gamma_p = posterior["gamma"].values.reshape(-1, len(self._commodities))[:n_use]
        sigma = posterior["sigma_obs"].values.reshape(-1, len(self._commodities))[:n_use]

        rng = np.random.default_rng(42)
        paths = np.zeros((n_use, horizon, len(self._commodities)))
        y_prev = np.zeros((n_use, len(self._commodities)))

        for t in range(horizon):
            esc = escalation_trajectory[t]
            mu = alpha + beta * esc + gamma_p * y_prev
            paths[:, t, :] = mu + sigma * rng.standard_normal(mu.shape)
            y_prev = paths[:, t, :]

        agg = paths.mean(axis=-1)
        point = agg.mean(axis=0).tolist()
        lower = np.quantile(agg, 0.05, axis=0).tolist()
        upper = np.quantile(agg, 0.95, axis=0).tolist()

        return PredictionResult(
            point_forecast=point,
            lower_bounds={0.05: lower},
            upper_bounds={0.95: upper},
            scenarios={
                c: paths[:, :, i].mean(axis=0).tolist()
                for i, c in enumerate(self._commodities)
            },
            metadata={
                "model": self.config.name,
                "escalation_trajectory": escalation_trajectory[:horizon],
            },
        )
