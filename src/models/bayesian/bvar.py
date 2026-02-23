"""Bayesian Vector Autoregression with Minnesota prior.

Implements a Bayesian VAR(p) model for jointly forecasting multiple
commodity return series along with geopolitical indicators.  The
Minnesota prior (Litterman, 1986) shrinks coefficients toward a random
walk, providing regularisation that is especially effective in the
high-dimensional, short-sample regime typical of geopolitical event
studies.

Identification follows Sims (1980) with Cholesky-ordered structural
shocks, enabling impulse response analysis of geopolitical shock
propagation through commodity markets.

Example::

    config = ModelConfig(
        name="bvar_commodities",
        params={
            "endogenous_cols": ["oil_return", "wheat_return", "escalation_index"],
            "lags": 4,
            "minnesota_lambda": 0.1,
            "draws": 2000,
            "tune": 1000,
        },
    )
    model = BayesianVARModel(config)
    model.fit(train_df)
    irf = model.impulse_response(shock_var="escalation_index", steps=40)
"""

from __future__ import annotations

from typing import Any

import arviz as az
import numpy as np
import polars as pl
import pymc as pm
from loguru import logger
from scipy import linalg as sla

from src.models.base_model import BaseModel, ModelConfig, ModelState, PredictionResult


class BayesianVARModel(BaseModel):
    """Bayesian VAR(p) model with Minnesota prior and MCMC inference.

    The model estimates:

        Y_t = c + A_1 Y_{t-1} + ... + A_p Y_{t-p} + u_t,  u_t ~ N(0, Sigma)

    with a Minnesota-style prior that shrinks own lags toward 1 (random walk)
    and cross-variable lags toward 0.

    Config params:
        endogenous_cols: Ordered list of endogenous variable columns.
        lags: Number of VAR lags (default 4).
        minnesota_lambda: Overall tightness of the Minnesota prior
            (default 0.1; smaller = tighter).
        minnesota_theta: Cross-variable shrinkage (default 0.5).
        draws: Posterior draws (default 2000).
        tune: Tuning / warm-up (default 1000).
        chains: MCMC chains (default 4).
        target_accept: NUTS target (default 0.9).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._endogenous_cols: list[str] = config.params.get(
            "endogenous_cols", ["oil_return", "wheat_return", "escalation_index"]
        )
        self._lags: int = config.params.get("lags", 4)
        self._lambda: float = config.params.get("minnesota_lambda", 0.1)
        self._theta: float = config.params.get("minnesota_theta", 0.5)
        self._draws: int = config.params.get("draws", 2000)
        self._tune: int = config.params.get("tune", 1000)
        self._chains: int = config.params.get("chains", 4)
        self._target_accept: float = config.params.get("target_accept", 0.9)

        self._trace: az.InferenceData | None = None
        self._model: pm.Model | None = None
        self._Y_train: np.ndarray | None = None  # for IRF initial conditions
        self._var_stds: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Data utilities
    # ------------------------------------------------------------------

    def _build_var_matrices(
        self, data: pl.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct the VAR regression matrices.

        Args:
            data: DataFrame with endogenous columns.

        Returns:
            (Y, X) where Y is ``(T-p, K)`` and X is ``(T-p, K*p+1)``
            (includes intercept).
        """
        Y_full = (
            data.select(self._endogenous_cols).to_numpy().astype(np.float64)
        )
        T, K = Y_full.shape
        p = self._lags

        Y = Y_full[p:]  # (T-p, K)
        X_parts: list[np.ndarray] = [np.ones((T - p, 1))]  # intercept
        for lag in range(1, p + 1):
            X_parts.append(Y_full[p - lag : T - lag])
        X = np.hstack(X_parts)  # (T-p, K*p + 1)
        return Y, X

    def _minnesota_prior_sd(self, K: int) -> np.ndarray:
        """Construct the Minnesota prior standard deviation matrix.

        Args:
            K: Number of endogenous variables.

        Returns:
            Prior SD matrix of shape ``(K*p+1, K)`` for the coefficient
            matrix (intercept row + K*p coefficient rows).
        """
        p = self._lags
        n_coeffs = K * p + 1
        sd = np.ones((n_coeffs, K))

        # Intercept -- diffuse prior
        sd[0, :] = 10.0

        assert self._var_stds is not None
        for lag in range(1, p + 1):
            for i in range(K):  # equation index
                for j in range(K):  # variable index
                    row = 1 + (lag - 1) * K + j
                    if i == j:
                        # Own lag: lambda / lag
                        sd[row, i] = self._lambda / lag
                    else:
                        # Cross lag: lambda * theta * (sigma_i / sigma_j) / lag
                        sd[row, i] = (
                            self._lambda
                            * self._theta
                            * (self._var_stds[i] / (self._var_stds[j] + 1e-8))
                            / lag
                        )
        return sd

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def fit(self, data: pl.DataFrame) -> None:
        """Fit the BVAR via NUTS sampling with Minnesota prior.

        Args:
            data: DataFrame with endogenous columns in chronological order.
        """
        self._validate_data(
            data,
            required_columns=self._endogenous_cols,
            min_rows=self._lags + 30,
        )

        Y, X = self._build_var_matrices(data)
        T_eff, K = Y.shape

        # Estimate variable standard deviations for the Minnesota prior
        self._var_stds = Y.std(axis=0)
        self._Y_train = (
            data.select(self._endogenous_cols).to_numpy().astype(np.float64)
        )

        prior_sd = self._minnesota_prior_sd(K)

        # Minnesota prior centers own lags at 1.0, others at 0.0
        prior_mean = np.zeros_like(prior_sd)
        for lag in range(1, self._lags + 1):
            for j in range(K):
                prior_mean[1 + (lag - 1) * K + j, j] = 1.0 if lag == 1 else 0.0

        with pm.Model() as model:
            # Coefficient matrix: B ~ N(prior_mean, prior_sd)
            B = pm.Normal(
                "B",
                mu=prior_mean,
                sigma=prior_sd,
                shape=(X.shape[1], K),
            )

            # Error covariance (LKJ prior for correlation + half-normal for scale)
            sd_obs = pm.HalfNormal("sd_obs", sigma=self._var_stds * 2, shape=K)
            chol, corr, stds = pm.LKJCholeskyCov(
                "chol_cov",
                n=K,
                eta=2.0,
                sd_dist=pm.HalfNormal.dist(sigma=self._var_stds * 2),
                compute_corr=True,
            )

            # Likelihood
            mu = pm.math.dot(X, B)  # (T_eff, K)
            pm.MvNormal(
                "y_obs",
                mu=mu,
                chol=chol,
                observed=Y,
                shape=(T_eff, K),
            )

            logger.info(
                "Sampling BVAR: {} draws x {} chains (lags={}, K={})",
                self._draws,
                self._chains,
                self._lags,
                K,
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
        """Generate multi-step VAR forecasts via posterior predictive.

        Args:
            horizon: Number of forecast steps.
            n_scenarios: Number of posterior draws to use.

        Returns:
            PredictionResult with per-variable scenario averages.
        """
        self._require_fitted()
        density = self.forecast_density(horizon=horizon, n_scenarios=n_scenarios)
        # density shape: (n_scenarios, horizon, K)
        K = len(self._endogenous_cols)

        # Aggregate: use first variable as primary
        primary = density[:, :, 0]
        point = primary.mean(axis=0).tolist()
        lower = np.quantile(primary, 0.05, axis=0).tolist()
        upper = np.quantile(primary, 0.95, axis=0).tolist()

        scenarios = {}
        for i, col in enumerate(self._endogenous_cols):
            scenarios[col] = density[:, :, i].mean(axis=0).tolist()

        return PredictionResult(
            point_forecast=point,
            lower_bounds={0.05: lower},
            upper_bounds={0.95: upper},
            scenarios=scenarios,
            metadata={
                "model": self.config.name,
                "lags": self._lags,
                "primary_variable": self._endogenous_cols[0],
            },
        )

    def get_params(self) -> dict[str, Any]:
        """Return posterior means of the coefficient matrix."""
        self._require_fitted()
        assert self._trace is not None

        B_post = self._trace.posterior["B"].values
        B_mean = B_post.reshape(-1, *B_post.shape[-2:]).mean(axis=0)

        return {
            "endogenous_cols": self._endogenous_cols,
            "lags": self._lags,
            "B_mean_shape": list(B_mean.shape),
            "minnesota_lambda": self._lambda,
            "minnesota_theta": self._theta,
        }

    # ------------------------------------------------------------------
    # Impulse response functions
    # ------------------------------------------------------------------

    def impulse_response(
        self,
        shock_var: str,
        steps: int = 40,
        shock_size: float = 1.0,
        n_draws: int = 500,
    ) -> dict[str, np.ndarray]:
        """Compute structural impulse response functions.

        Uses Cholesky identification (variable ordering from
        ``endogenous_cols``) to orthogonalise shocks.

        Args:
            shock_var: Name of the variable receiving the shock.
            steps: Number of IRF periods.
            shock_size: Magnitude of the one-standard-deviation shock.
            n_draws: Number of posterior draws for IRF uncertainty bands.

        Returns:
            Dictionary mapping each endogenous variable to an array of
            shape ``(n_draws, steps)`` containing IRF paths.
        """
        self._require_fitted()
        assert self._trace is not None

        K = len(self._endogenous_cols)
        shock_idx = self._endogenous_cols.index(shock_var)
        p = self._lags

        B_post = self._trace.posterior["B"].values.reshape(-1, K * p + 1, K)
        chol_post = self._trace.posterior["chol_cov"].values.reshape(
            -1, K, K
        )

        rng = np.random.default_rng(42)
        draw_idx = rng.choice(B_post.shape[0], size=n_draws, replace=False)

        irfs: dict[str, np.ndarray] = {
            col: np.zeros((n_draws, steps)) for col in self._endogenous_cols
        }

        for d_i, d in enumerate(draw_idx):
            B = B_post[d]  # (Kp+1, K)
            chol = chol_post[d]  # (K, K)

            # Build companion-form A matrices
            A_mats = []
            for lag in range(p):
                A_mats.append(B[1 + lag * K : 1 + (lag + 1) * K, :].T)  # (K, K)

            # Structural shock vector
            shock = np.zeros(K)
            shock[shock_idx] = shock_size

            # Orthogonalised shock
            e0 = chol @ shock

            # Simulate IRF
            Y_irf = np.zeros((steps + p, K))
            Y_irf[p] = e0
            for t in range(p, steps + p):
                for lag, A_l in enumerate(A_mats):
                    if t - lag - 1 >= 0:
                        Y_irf[t] += A_l @ Y_irf[t - lag - 1]

            for v_i, col in enumerate(self._endogenous_cols):
                irfs[col][d_i] = Y_irf[p:, v_i]

        logger.info(
            "Computed IRF for shock to '{}' over {} steps ({} draws)",
            shock_var,
            steps,
            n_draws,
        )
        return irfs

    # ------------------------------------------------------------------
    # Forecast density
    # ------------------------------------------------------------------

    def forecast_density(
        self,
        horizon: int,
        n_scenarios: int = 1000,
    ) -> np.ndarray:
        """Sample from the posterior predictive forecast distribution.

        Args:
            horizon: Number of forward steps.
            n_scenarios: Number of draws.

        Returns:
            Array of shape ``(n_scenarios, horizon, K)``.
        """
        self._require_fitted()
        assert self._trace is not None
        assert self._Y_train is not None

        K = len(self._endogenous_cols)
        p = self._lags

        B_post = self._trace.posterior["B"].values.reshape(-1, K * p + 1, K)
        chol_post = self._trace.posterior["chol_cov"].values.reshape(
            -1, K, K
        )

        rng = np.random.default_rng(42)
        n_total = B_post.shape[0]
        draw_idx = rng.choice(n_total, size=min(n_scenarios, n_total), replace=False)

        Y_last = self._Y_train[-p:]  # (p, K) -- last p observations

        forecasts = np.zeros((len(draw_idx), horizon, K))
        for d_i, d in enumerate(draw_idx):
            B = B_post[d]  # (Kp+1, K)
            chol = chol_post[d]

            # History buffer
            Y_hist = Y_last.copy()  # (p, K)

            for t in range(horizon):
                x = np.concatenate([[1.0], Y_hist[::-1].flatten()])  # (Kp+1,)
                mu = x @ B  # (K,)
                eps = chol @ rng.standard_normal(K)
                y_new = mu + eps
                forecasts[d_i, t] = y_new
                # Roll history
                Y_hist = np.vstack([Y_hist[1:], y_new.reshape(1, -1)])

        return forecasts
