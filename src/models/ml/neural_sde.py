"""Neural Stochastic Differential Equations for continuous-time price dynamics.

Models commodity price evolution as a continuous-time SDE where neural
networks parameterise the drift (mu) and diffusion (sigma) functions,
with an additional jump component for modelling sudden war-shock
discontinuities:

    dX_t = mu(X_t, t; theta) dt + sigma(X_t, t; theta) dW_t + J dN_t

where N_t is a Poisson process capturing geopolitical shocks and J is
a (possibly state-dependent) jump size distribution.

The model is trained by maximising the path log-likelihood using an
Euler-Maruyama discretisation and reparameterised sampling.

Example::

    config = ModelConfig(
        name="nsde_oil",
        params={
            "state_dim": 1,
            "hidden_dim": 64,
            "jump_intensity": 0.05,
            "dt": 1 / 252,
        },
    )
    model = NeuralSDEModel(config)
    model.fit(train_df)
    paths = model.simulate_paths(n_paths=5000, n_steps=60)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from loguru import logger

from src.models.base_model import BaseModel, ModelConfig, PredictionResult


# ---------------------------------------------------------------------------
# Neural network components
# ---------------------------------------------------------------------------


class _DriftNet(nn.Module):
    """Neural network parameterising the drift function mu(x, t)."""

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  # +1 for time
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute drift at state *x* and time *t*.

        Args:
            x: State tensor of shape ``(batch, state_dim)``.
            t: Time tensor of shape ``(batch, 1)`` or scalar.

        Returns:
            Drift values of shape ``(batch, state_dim)``.
        """
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.size(0), 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)


class _DiffusionNet(nn.Module):
    """Neural network parameterising the diffusion function sigma(x, t).

    Output is passed through softplus to ensure positivity.
    """

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute diffusion at state *x* and time *t*.

        Args:
            x: State tensor of shape ``(batch, state_dim)``.
            t: Time tensor of shape ``(batch, 1)`` or scalar.

        Returns:
            Positive diffusion values of shape ``(batch, state_dim)``.
        """
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.size(0), 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)


class _JumpNet(nn.Module):
    """Neural network parameterising the jump size distribution J(x, t).

    Outputs the mean and log-variance of a Gaussian jump size so that
    jump magnitudes can be sampled via the reparameterisation trick.
    """

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, state_dim)
        self.logvar_head = nn.Linear(hidden_dim, state_dim)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (jump_mean, jump_logvar).

        Args:
            x: State tensor ``(batch, state_dim)``.
            t: Time tensor ``(batch, 1)`` or scalar.

        Returns:
            Tuple of (mu_J, logvar_J) each of shape ``(batch, state_dim)``.
        """
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.size(0), 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([x, t], dim=-1)
        h = self.net(inp)
        return self.mu_head(h), self.logvar_head(h)


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------


class NeuralSDEModel(BaseModel):
    """Neural SDE for modelling continuous-time commodity price dynamics.

    Combines learned drift and diffusion with a Poisson jump component
    to capture both smooth mean-reversion and sudden war-related shocks.

    Config params:
        target_col: Price column name (default ``"target"``).
        state_dim: Dimensionality of the state process (default 1).
        hidden_dim: Hidden layer width (default 64).
        jump_intensity: Poisson intensity lambda for the jump process
            (default 0.05, roughly one jump per 20 time steps).
        dt: Time step for Euler-Maruyama discretisation (default 1/252
            for daily data).
        max_epochs: Training epochs (default 200).
        learning_rate: Adam learning rate (default 1e-3).
        batch_size: Training batch size (default 128).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._target_col: str = config.params.get("target_col", "target")
        self._state_dim: int = config.params.get("state_dim", 1)
        self._hidden_dim: int = config.params.get("hidden_dim", 64)
        self._jump_intensity: float = config.params.get("jump_intensity", 0.05)
        self._dt: float = config.params.get("dt", 1.0 / 252.0)
        self._max_epochs: int = config.params.get("max_epochs", 200)
        self._lr: float = config.params.get("learning_rate", 1e-3)
        self._batch_size: int = config.params.get("batch_size", 128)

        self._drift: _DriftNet | None = None
        self._diffusion: _DiffusionNet | None = None
        self._jump: _JumpNet | None = None
        # Learnable log-intensity for the Poisson jump process
        self._log_lambda: nn.Parameter | None = None

        self._x0_mean: float = 0.0
        self._x0_std: float = 1.0
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fit(self, data: pl.DataFrame) -> None:
        """Train the Neural SDE by maximising the Euler-Maruyama log-likelihood.

        The training data is treated as a single observed path.  The model
        learns the drift, diffusion, and jump parameters that maximise the
        likelihood of transitioning between consecutive observations.

        Args:
            data: DataFrame with chronological target column.
        """
        self._validate_data(data, required_columns=[self._target_col], min_rows=50)

        series = data[self._target_col].to_numpy().astype(np.float64)
        # Normalise to log-returns for numerical stability
        log_prices = np.log(np.maximum(series, 1e-8))
        self._x0_mean = float(log_prices[0])
        self._x0_std = float(np.std(log_prices) + 1e-8)
        normalised = (log_prices - self._x0_mean) / self._x0_std

        # Build consecutive transition pairs
        x_t = torch.tensor(
            normalised[:-1].reshape(-1, self._state_dim), dtype=torch.float32
        )
        x_tp1 = torch.tensor(
            normalised[1:].reshape(-1, self._state_dim), dtype=torch.float32
        )
        t_idx = torch.linspace(0, 1, x_t.size(0), dtype=torch.float32)

        # Initialise networks
        self._drift = _DriftNet(self._state_dim, self._hidden_dim).to(self._device)
        self._diffusion = _DiffusionNet(self._state_dim, self._hidden_dim).to(
            self._device
        )
        self._jump = _JumpNet(self._state_dim, self._hidden_dim).to(self._device)
        self._log_lambda = nn.Parameter(
            torch.tensor(np.log(self._jump_intensity), dtype=torch.float32).to(
                self._device
            )
        )

        all_params: list[torch.Tensor] = [
            *self._drift.parameters(),
            *self._diffusion.parameters(),
            *self._jump.parameters(),
            self._log_lambda,
        ]
        optimizer = torch.optim.Adam(all_params, lr=self._lr)

        dataset = torch.utils.data.TensorDataset(x_t, x_tp1, t_idx)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self._batch_size, shuffle=True
        )

        for epoch in range(self._max_epochs):
            total_loss = 0.0
            for xb, xb_next, tb in loader:
                xb = xb.to(self._device)
                xb_next = xb_next.to(self._device)
                tb = tb.to(self._device)

                optimizer.zero_grad()
                loss = -self._transition_log_likelihood(xb, xb_next, tb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 50 == 0:
                logger.info(
                    "NeuralSDE epoch {}/{}: avg NLL={:.6f}",
                    epoch + 1,
                    self._max_epochs,
                    total_loss / len(loader),
                )

        self._mark_fitted(data)

    def predict(
        self,
        horizon: int,
        n_scenarios: int = 1000,
    ) -> PredictionResult:
        """Generate point forecast and prediction intervals via simulation.

        Args:
            horizon: Number of forward steps.
            n_scenarios: Number of Monte-Carlo paths.

        Returns:
            PredictionResult with mean point forecast and quantile bounds.
        """
        self._require_fitted()
        paths = self.simulate_paths(
            n_paths=n_scenarios, n_steps=horizon, return_prices=True
        )
        # paths shape: (n_scenarios, horizon + 1) -- drop initial value
        future = paths[:, 1:]  # (n_scenarios, horizon)

        point = np.mean(future, axis=0).tolist()
        lower = np.quantile(future, 0.05, axis=0).tolist()
        upper = np.quantile(future, 0.95, axis=0).tolist()

        return PredictionResult(
            point_forecast=point,
            lower_bounds={0.05: lower},
            upper_bounds={0.95: upper},
            metadata={
                "model": self.config.name,
                "n_scenarios": n_scenarios,
                "jump_intensity": float(torch.exp(self._log_lambda).item()),  # type: ignore[union-attr]
            },
        )

    def get_params(self) -> dict[str, Any]:
        """Return fitted model parameters and architecture details."""
        self._require_fitted()
        return {
            "state_dim": self._state_dim,
            "hidden_dim": self._hidden_dim,
            "jump_intensity": float(torch.exp(self._log_lambda).item()),  # type: ignore[union-attr]
            "dt": self._dt,
            "x0_mean": self._x0_mean,
            "x0_std": self._x0_std,
            "n_drift_params": sum(p.numel() for p in self._drift.parameters()),  # type: ignore[union-attr]
            "n_diffusion_params": sum(p.numel() for p in self._diffusion.parameters()),  # type: ignore[union-attr]
            "n_jump_params": sum(p.numel() for p in self._jump.parameters()),  # type: ignore[union-attr]
        }

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_paths(
        self,
        n_paths: int = 1000,
        n_steps: int = 60,
        x0: float | None = None,
        return_prices: bool = True,
    ) -> np.ndarray:
        """Simulate price paths via Euler-Maruyama with Poisson jumps.

        Args:
            n_paths: Number of independent paths.
            n_steps: Number of time steps per path.
            x0: Initial price level.  If ``None``, uses the last
                training observation.
            return_prices: If ``True``, return prices; otherwise return
                normalised log-space values.

        Returns:
            Array of shape ``(n_paths, n_steps + 1)`` with simulated paths.
        """
        self._require_fitted()
        assert self._drift is not None
        assert self._diffusion is not None
        assert self._jump is not None
        assert self._log_lambda is not None

        self._drift.eval()
        self._diffusion.eval()
        self._jump.eval()

        # Initial state in normalised log-space
        if x0 is not None:
            z0 = (np.log(max(x0, 1e-8)) - self._x0_mean) / self._x0_std
        else:
            z0 = 0.0  # last training point (approximately)

        dt = self._dt
        sqrt_dt = np.sqrt(dt)
        lam = torch.exp(self._log_lambda).item()

        paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        paths[:, 0] = z0

        with torch.no_grad():
            for step in range(n_steps):
                x_curr = torch.tensor(
                    paths[:, step].reshape(-1, self._state_dim),
                    dtype=torch.float32,
                ).to(self._device)
                t_curr = torch.full(
                    (n_paths, 1), step * dt, dtype=torch.float32
                ).to(self._device)

                mu = self._drift(x_curr, t_curr).cpu().numpy().flatten()
                sigma = self._diffusion(x_curr, t_curr).cpu().numpy().flatten()

                # Brownian increment
                dW = np.random.default_rng().standard_normal(n_paths) * sqrt_dt

                # Poisson jumps
                n_jumps = np.random.default_rng().poisson(lam * dt, size=n_paths)
                jump_mu, jump_logvar = self._jump(x_curr, t_curr)
                j_mu = jump_mu.cpu().numpy().flatten()
                j_std = np.exp(0.5 * jump_logvar.cpu().numpy().flatten())
                jump_sizes = (
                    n_jumps
                    * (j_mu + j_std * np.random.default_rng().standard_normal(n_paths))
                )

                paths[:, step + 1] = (
                    paths[:, step] + mu * dt + sigma * dW + jump_sizes
                )

        if return_prices:
            # Convert back from normalised log-space to prices
            log_prices = paths * self._x0_std + self._x0_mean
            return np.exp(log_prices)

        return paths

    # ------------------------------------------------------------------
    # Log-likelihood
    # ------------------------------------------------------------------

    def log_likelihood(self, data: pl.DataFrame) -> float:
        """Compute the transition log-likelihood of an observed path.

        Args:
            data: DataFrame with the target column.

        Returns:
            Total log-likelihood (scalar).
        """
        self._require_fitted()
        series = data[self._target_col].to_numpy().astype(np.float64)
        log_prices = np.log(np.maximum(series, 1e-8))
        normalised = (log_prices - self._x0_mean) / self._x0_std

        x_t = torch.tensor(
            normalised[:-1].reshape(-1, self._state_dim), dtype=torch.float32
        ).to(self._device)
        x_tp1 = torch.tensor(
            normalised[1:].reshape(-1, self._state_dim), dtype=torch.float32
        ).to(self._device)
        t_idx = torch.linspace(0, 1, x_t.size(0), dtype=torch.float32).to(
            self._device
        )

        with torch.no_grad():
            ll = self._transition_log_likelihood(x_t, x_tp1, t_idx)
        return float(ll.item())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _transition_log_likelihood(
        self,
        x_t: torch.Tensor,
        x_tp1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Euler-Maruyama transition log-likelihood with jump-diffusion.

        Under the Euler-Maruyama scheme with a compound-Poisson jump
        component, the transition density is approximated as a mixture:

            p(x_{t+1} | x_t) ~= (1 - lambda*dt) * N(x_t + mu*dt, sigma^2*dt)
                                + lambda*dt * N(x_t + mu*dt + J_mu, sigma^2*dt + J_var)

        Args:
            x_t: Current states ``(batch, state_dim)``.
            x_tp1: Next states ``(batch, state_dim)``.
            t: Time indices ``(batch,)``.

        Returns:
            Scalar sum of log-likelihoods.
        """
        assert self._drift is not None
        assert self._diffusion is not None
        assert self._jump is not None
        assert self._log_lambda is not None

        dt = self._dt
        mu = self._drift(x_t, t)
        sigma = self._diffusion(x_t, t) + 1e-6  # numerical floor
        j_mu, j_logvar = self._jump(x_t, t)
        j_var = torch.exp(j_logvar)
        lam = torch.exp(self._log_lambda)

        # Diffusion variance
        diff_var = sigma.pow(2) * dt

        # No-jump component
        mean_no_jump = x_t + mu * dt
        log_p_no_jump = (
            -0.5 * ((x_tp1 - mean_no_jump).pow(2) / diff_var + torch.log(diff_var))
            - 0.5 * np.log(2 * np.pi)
        )

        # Jump component
        mean_jump = x_t + mu * dt + j_mu
        var_jump = diff_var + j_var
        log_p_jump = (
            -0.5 * ((x_tp1 - mean_jump).pow(2) / var_jump + torch.log(var_jump))
            - 0.5 * np.log(2 * np.pi)
        )

        # Mixture log-likelihood via log-sum-exp
        log_w_no_jump = torch.log(1 - lam * dt + 1e-8)
        log_w_jump = torch.log(lam * dt + 1e-8)

        log_p = torch.logsumexp(
            torch.stack(
                [log_w_no_jump + log_p_no_jump.sum(-1), log_w_jump + log_p_jump.sum(-1)],
                dim=0,
            ),
            dim=0,
        )
        return log_p.sum()
