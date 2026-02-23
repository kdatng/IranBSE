"""Bayesian causal impact analysis for commodity futures.

Estimates the causal effect of a geopolitical event (e.g. onset of
US-Iran war) on commodity prices by constructing a Bayesian structural
time series counterfactual: what would prices have been *without* the
intervention?

The approach follows Brodersen et al. (2015, "Inferring causal impact
using Bayesian structural time series models") and uses:

1. **Local linear trend** for capturing the pre-intervention price dynamics.
2. **Regression on covariates** that are unaffected by the intervention
   (e.g. global industrial production, other commodity prices that are
   not directly affected) to improve the counterfactual.
3. **Bayesian inference** via PyMC to quantify posterior uncertainty in
   both the counterfactual and the estimated causal effect.

Example::

    config = ModelConfig(
        name="causal_impact_oil",
        params={
            "target_col": "oil_close",
            "covariate_cols": ["sp500", "dxy_index", "vix"],
            "intervention_date_idx": 500,
        },
    )
    model = CausalImpactModel(config)
    model.fit(full_df)
    impact = model.estimate_impact()
    cum = model.cumulative_impact()
"""

from __future__ import annotations

from typing import Any

import arviz as az
import numpy as np
import polars as pl
import pymc as pm
from loguru import logger

from src.models.base_model import BaseModel, ModelConfig, ModelState, PredictionResult


class CausalImpactModel(BaseModel):
    """Bayesian causal impact estimation for geopolitical interventions.

    Constructs a counterfactual prediction for the post-intervention
    period using a Bayesian structural time series fitted to the
    pre-intervention data, then measures the difference between
    observed and counterfactual as the causal effect.

    Config params:
        target_col: Target price/return column (default ``"target"``).
        covariate_cols: Columns unaffected by the intervention that
            help predict the target (e.g. broad market indices).
        intervention_date_idx: Integer index of the intervention start
            within the DataFrame (required).
        draws: Posterior draws (default 2000).
        tune: Tuning steps (default 1000).
        chains: MCMC chains (default 4).
        target_accept: NUTS acceptance rate (default 0.9).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._target_col: str = config.params.get("target_col", "target")
        self._covariate_cols: list[str] = config.params.get("covariate_cols", [])
        self._intervention_idx: int | None = config.params.get(
            "intervention_date_idx"
        )
        self._draws: int = config.params.get("draws", 2000)
        self._tune: int = config.params.get("tune", 1000)
        self._chains: int = config.params.get("chains", 4)
        self._target_accept: float = config.params.get("target_accept", 0.9)

        # Fitted artefacts
        self._trace: az.InferenceData | None = None
        self._model: pm.Model | None = None
        self._y_pre: np.ndarray | None = None
        self._y_post: np.ndarray | None = None
        self._X_pre: np.ndarray | None = None
        self._X_post: np.ndarray | None = None
        self._counterfactual: np.ndarray | None = None  # (draws, T_post)
        self._y_target_full: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def fit(self, data: pl.DataFrame) -> None:
        """Fit the structural time series on pre-intervention data.

        The model is fit on pre-intervention observations only.  After
        fitting, the posterior predictive is projected forward into the
        post-intervention period to construct the counterfactual.

        Args:
            data: Full DataFrame spanning both pre- and post-intervention.

        Raises:
            ValueError: If intervention_date_idx is not set.
        """
        if self._intervention_idx is None:
            raise ValueError(
                "intervention_date_idx must be set in config params"
            )

        required = [self._target_col, *self._covariate_cols]
        self._validate_data(data, required_columns=required, min_rows=50)

        idx = self._intervention_idx
        y_full = data[self._target_col].to_numpy().astype(np.float64)
        self._y_target_full = y_full

        self._y_pre = y_full[:idx]
        self._y_post = y_full[idx:]
        T_pre = len(self._y_pre)
        T_post = len(self._y_post)

        if self._covariate_cols:
            X_full = (
                data.select(self._covariate_cols)
                .to_numpy()
                .astype(np.float64)
            )
            self._X_pre = X_full[:idx]
            self._X_post = X_full[idx:]
        else:
            self._X_pre = None
            self._X_post = None

        n_covariates = self._X_pre.shape[1] if self._X_pre is not None else 0

        with pm.Model() as model:
            # --- Local linear trend ---
            sigma_level = pm.HalfNormal("sigma_level", sigma=0.5)
            sigma_trend = pm.HalfNormal("sigma_trend", sigma=0.1)

            # Initial states
            level_init = pm.Normal("level_init", mu=self._y_pre[0], sigma=1.0)
            trend_init = pm.Normal("trend_init", mu=0.0, sigma=0.5)

            # State innovations
            level_innov = pm.Normal("level_innov", mu=0, sigma=1, shape=T_pre)
            trend_innov = pm.Normal("trend_innov", mu=0, sigma=1, shape=T_pre)

            # Build states recursively using scan
            def _step(
                l_innov: Any,
                t_innov: Any,
                prev_level: Any,
                prev_trend: Any,
            ) -> tuple[Any, Any]:
                new_trend = prev_trend + sigma_trend * t_innov
                new_level = prev_level + prev_trend + sigma_level * l_innov
                return new_level, new_trend

            (levels, trends), _ = pm.pytensorf.scan(
                fn=_step,
                sequences=[level_innov, trend_innov],
                outputs_info=[level_init, trend_init],
            )

            # --- Regression on covariates ---
            mu = levels
            if n_covariates > 0:
                beta_cov = pm.Normal(
                    "beta_cov", mu=0, sigma=1, shape=n_covariates
                )
                mu = mu + pm.math.dot(self._X_pre, beta_cov)

            # --- Observation noise ---
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)
            pm.Normal("y_obs", mu=mu, sigma=sigma_obs, observed=self._y_pre)

            logger.info(
                "Fitting causal impact model on {} pre-intervention obs "
                "(post-intervention: {})",
                T_pre,
                T_post,
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

        # Construct counterfactual for the post-intervention period
        self._build_counterfactual()
        self._mark_fitted(data)

    def predict(
        self,
        horizon: int,
        n_scenarios: int = 1000,
    ) -> PredictionResult:
        """Return the counterfactual prediction for the post period.

        Args:
            horizon: Clamped to the actual post-intervention length.
            n_scenarios: Number of draws to use.

        Returns:
            PredictionResult with the counterfactual as point forecast.
        """
        self._require_fitted()
        assert self._counterfactual is not None

        h = min(horizon, self._counterfactual.shape[1])
        cf = self._counterfactual[:, :h]

        point = cf.mean(axis=0).tolist()
        lower = np.quantile(cf, 0.05, axis=0).tolist()
        upper = np.quantile(cf, 0.95, axis=0).tolist()

        return PredictionResult(
            point_forecast=point,
            lower_bounds={0.05: lower},
            upper_bounds={0.95: upper},
            metadata={
                "model": self.config.name,
                "type": "counterfactual",
                "post_intervention_length": int(self._counterfactual.shape[1]),
            },
        )

    def get_params(self) -> dict[str, Any]:
        """Return posterior summaries for model parameters."""
        self._require_fitted()
        assert self._trace is not None

        var_names = ["sigma_level", "sigma_trend", "sigma_obs"]
        if self._covariate_cols:
            var_names.append("beta_cov")

        summary = az.summary(self._trace, var_names=var_names, hdi_prob=0.95)
        return {
            "target_col": self._target_col,
            "covariate_cols": self._covariate_cols,
            "intervention_idx": self._intervention_idx,
            "pre_intervention_length": len(self._y_pre) if self._y_pre is not None else 0,
            "post_intervention_length": len(self._y_post) if self._y_post is not None else 0,
            "posterior_summary": summary.to_dict(),
        }

    # ------------------------------------------------------------------
    # Impact estimation
    # ------------------------------------------------------------------

    def estimate_impact(self) -> dict[str, Any]:
        """Estimate the point-wise causal impact.

        The causal impact at each post-intervention time step is:
            impact_t = y_observed_t - y_counterfactual_t

        Returns:
            Dictionary with:
                - ``"observed"``: actual post-intervention values.
                - ``"counterfactual_mean"``: posterior mean counterfactual.
                - ``"counterfactual_lower"``: 5th percentile.
                - ``"counterfactual_upper"``: 95th percentile.
                - ``"impact_mean"``: posterior mean impact.
                - ``"impact_lower"``: 5th percentile of impact.
                - ``"impact_upper"``: 95th percentile of impact.
                - ``"prob_causal_effect"``: posterior probability that the
                  cumulative impact is non-zero (in the observed direction).
        """
        self._require_fitted()
        assert self._y_post is not None
        assert self._counterfactual is not None

        observed = self._y_post
        cf = self._counterfactual  # (draws, T_post)

        impact = observed[np.newaxis, :] - cf  # (draws, T_post)

        # Probability of a positive cumulative effect
        cum_impact = impact.sum(axis=1)
        prob_positive = float((cum_impact > 0).mean())
        prob_negative = float((cum_impact < 0).mean())
        prob_causal = max(prob_positive, prob_negative)

        result = {
            "observed": observed.tolist(),
            "counterfactual_mean": cf.mean(axis=0).tolist(),
            "counterfactual_lower": np.quantile(cf, 0.05, axis=0).tolist(),
            "counterfactual_upper": np.quantile(cf, 0.95, axis=0).tolist(),
            "impact_mean": impact.mean(axis=0).tolist(),
            "impact_lower": np.quantile(impact, 0.05, axis=0).tolist(),
            "impact_upper": np.quantile(impact, 0.95, axis=0).tolist(),
            "prob_causal_effect": prob_causal,
            "cumulative_impact_mean": float(cum_impact.mean()),
            "cumulative_impact_95_hdi": [
                float(np.quantile(cum_impact, 0.025)),
                float(np.quantile(cum_impact, 0.975)),
            ],
        }

        logger.info(
            "Estimated causal impact: cumulative mean={:.4f}, "
            "P(effect)={:.2%}",
            result["cumulative_impact_mean"],
            prob_causal,
        )
        return result

    def cumulative_impact(self) -> dict[str, Any]:
        """Compute the cumulative causal impact over time.

        Returns:
            Dictionary with ``"cumulative_mean"``, ``"cumulative_lower"``,
            ``"cumulative_upper"`` (all lists of cumulated impact).
        """
        self._require_fitted()
        assert self._y_post is not None
        assert self._counterfactual is not None

        observed = self._y_post
        cf = self._counterfactual
        impact = observed[np.newaxis, :] - cf  # (draws, T_post)
        cum = np.cumsum(impact, axis=1)  # (draws, T_post)

        return {
            "cumulative_mean": cum.mean(axis=0).tolist(),
            "cumulative_lower": np.quantile(cum, 0.05, axis=0).tolist(),
            "cumulative_upper": np.quantile(cum, 0.95, axis=0).tolist(),
            "total_mean": float(cum[:, -1].mean()),
            "total_95_hdi": [
                float(np.quantile(cum[:, -1], 0.025)),
                float(np.quantile(cum[:, -1], 0.975)),
            ],
        }

    def plot_impact(self) -> dict[str, Any]:
        """Return structured data for visualising the causal impact.

        This returns all the arrays needed for a plot without importing
        matplotlib, so that the visualisation layer can handle rendering.

        Returns:
            Dictionary with ``"pre_observed"``, ``"post_observed"``,
            ``"counterfactual_*"``, ``"impact_*"``, and
            ``"cumulative_*"`` arrays.
        """
        self._require_fitted()
        assert self._y_pre is not None
        assert self._y_post is not None
        assert self._counterfactual is not None
        assert self._intervention_idx is not None

        impact_data = self.estimate_impact()
        cumulative_data = self.cumulative_impact()

        T_pre = len(self._y_pre)
        T_post = len(self._y_post)

        return {
            "intervention_idx": self._intervention_idx,
            "time_pre": list(range(T_pre)),
            "time_post": list(range(T_pre, T_pre + T_post)),
            "pre_observed": self._y_pre.tolist(),
            "post_observed": self._y_post.tolist(),
            **{k: v for k, v in impact_data.items() if k.startswith("counterfactual")},
            **{k: v for k, v in impact_data.items() if k.startswith("impact")},
            **{k: v for k, v in cumulative_data.items()},
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_counterfactual(self) -> None:
        """Project the fitted structural time series into the post period.

        Uses the posterior samples of the last level and trend to
        recursively simulate forward, adding the regression component
        from post-intervention covariates.
        """
        assert self._trace is not None
        assert self._y_post is not None

        T_post = len(self._y_post)
        posterior = self._trace.posterior
        n_draws = posterior.dims["chain"] * posterior.dims["draw"]

        # Extract last level and trend
        levels = posterior["level_innov"].values  # (chain, draw, T_pre)
        sigma_level = posterior["sigma_level"].values.flatten()
        sigma_trend = posterior["sigma_trend"].values.flatten()
        sigma_obs = posterior["sigma_obs"].values.flatten()

        # Reconstruct final level and trend from the trace
        # We'll approximate by sampling forward from the fitted parameters
        level_init = posterior["level_init"].values.flatten()
        trend_init = posterior["trend_init"].values.flatten()
        level_innov = posterior["level_innov"].values.reshape(n_draws, -1)
        trend_innov = posterior["trend_innov"].values.reshape(n_draws, -1)

        T_pre = level_innov.shape[1]

        # Reconstruct states
        final_levels = np.zeros(n_draws)
        final_trends = np.zeros(n_draws)

        for d in range(n_draws):
            level = level_init[d]
            trend = trend_init[d]
            for t in range(T_pre):
                trend = trend + sigma_trend[d] * trend_innov[d, t]
                level = level + trend + sigma_level[d] * level_innov[d, t]
            final_levels[d] = level
            final_trends[d] = trend

        # Covariate regression coefficients
        has_covariates = self._covariate_cols and self._X_post is not None
        if has_covariates:
            beta = posterior["beta_cov"].values.reshape(n_draws, -1)
        else:
            beta = None

        # Forward simulation
        rng = np.random.default_rng(42)
        counterfactual = np.zeros((n_draws, T_post))

        for d in range(n_draws):
            level = final_levels[d]
            trend = final_trends[d]

            for t in range(T_post):
                trend = trend + sigma_trend[d] * rng.standard_normal()
                level = level + trend + sigma_level[d] * rng.standard_normal()
                mu = level
                if has_covariates and beta is not None and self._X_post is not None:
                    mu += self._X_post[t] @ beta[d]
                counterfactual[d, t] = mu + sigma_obs[d] * rng.standard_normal()

        self._counterfactual = counterfactual
        logger.info(
            "Built counterfactual: {} draws x {} post-intervention steps",
            n_draws,
            T_post,
        )
