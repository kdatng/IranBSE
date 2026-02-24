"""Normalizing flows for full conditional density estimation.

Implements Real NVP (Dinh et al. 2017) and Neural Spline Flows
(Durkan et al. 2019) for learning the complete conditional distribution
of commodity price changes given geopolitical covariates.  Unlike quantile
regression, normalizing flows provide a tractable exact density, enabling
direct computation of probabilities, quantiles, and tail risk measures.

Example::

    config = ModelConfig(
        name="flow_oil",
        params={
            "target_col": "oil_return",
            "flow_type": "neural_spline",
            "n_layers": 8,
            "hidden_dim": 64,
            "n_bins": 16,
        },
    )
    model = NormalizingFlowModel(config)
    model.fit(train_df)
    samples = model.sample(n=5000, context=context_array)
    log_probs = model.log_prob(test_values, context=context_array)
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from loguru import logger

from src.models.base_model import BaseModel, ModelConfig, PredictionResult


# ---------------------------------------------------------------------------
# Coupling layer building blocks
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    """Simple MLP with residual connections for conditioning networks."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_hidden: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Real NVP coupling layer
# ---------------------------------------------------------------------------


class _AffineCouplingLayer(nn.Module):
    """Real NVP affine coupling layer.

    Splits the input into two halves; one is passed through unchanged
    while the other is affine-transformed by parameters that depend on
    the first half plus optional conditioning context.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        hidden_dim: int,
        mask_even: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        # Alternating mask pattern
        mask = torch.zeros(dim)
        if mask_even:
            mask[::2] = 1.0
        else:
            mask[1::2] = 1.0
        self.register_buffer("mask", mask)

        input_dim = dim + context_dim
        self.scale_net = _MLP(input_dim, dim, hidden_dim)
        self.translate_net = _MLP(input_dim, dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (data -> latent).

        Args:
            x: Input tensor ``(batch, dim)``.
            context: Optional conditioning tensor ``(batch, context_dim)``.

        Returns:
            Tuple of (transformed x, log-determinant of Jacobian).
        """
        mask = self.mask  # type: ignore[assignment]
        x_masked = x * mask
        if context is not None:
            inp = torch.cat([x_masked, context], dim=-1)
        else:
            inp = x_masked

        log_scale = self.scale_net(inp) * (1 - mask)
        # Clamp for numerical stability
        log_scale = torch.clamp(log_scale, -5.0, 5.0)
        translate = self.translate_net(inp) * (1 - mask)

        y = x * mask + (1 - mask) * (x * torch.exp(log_scale) + translate)
        log_det = log_scale.sum(dim=-1)
        return y, log_det

    def inverse(
        self, y: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Inverse pass (latent -> data).

        Args:
            y: Latent tensor ``(batch, dim)``.
            context: Optional conditioning tensor.

        Returns:
            Reconstructed data tensor.
        """
        mask = self.mask  # type: ignore[assignment]
        y_masked = y * mask
        if context is not None:
            inp = torch.cat([y_masked, context], dim=-1)
        else:
            inp = y_masked

        log_scale = self.scale_net(inp) * (1 - mask)
        log_scale = torch.clamp(log_scale, -5.0, 5.0)
        translate = self.translate_net(inp) * (1 - mask)

        x = y * mask + (1 - mask) * ((y - translate) * torch.exp(-log_scale))
        return x


# ---------------------------------------------------------------------------
# Neural Spline coupling layer (rational quadratic spline)
# ---------------------------------------------------------------------------


class _NeuralSplineCouplingLayer(nn.Module):
    """Coupling layer using rational-quadratic splines (Durkan et al. 2019).

    Monotone rational-quadratic splines provide more flexible
    transformations than affine coupling while remaining analytically
    invertible.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        hidden_dim: int,
        n_bins: int = 16,
        tail_bound: float = 5.0,
        mask_even: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_bins = n_bins
        self.tail_bound = tail_bound

        mask = torch.zeros(dim)
        if mask_even:
            mask[::2] = 1.0
        else:
            mask[1::2] = 1.0
        self.register_buffer("mask", mask)

        # Each spline needs: n_bins widths + n_bins heights + (n_bins - 1) derivatives
        n_params_per_dim = 3 * n_bins - 1
        n_transform_dims = dim - int(mask.sum().item())
        input_dim = dim + context_dim
        self.param_net = _MLP(input_dim, n_transform_dims * n_params_per_dim, hidden_dim)
        self._n_transform_dims = n_transform_dims
        self._n_params_per_dim = n_params_per_dim

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward transform.

        Args:
            x: Input ``(batch, dim)``.
            context: Optional context ``(batch, context_dim)``.

        Returns:
            (transformed_x, log_det_jacobian).
        """
        mask = self.mask  # type: ignore[assignment]
        x_masked = x * mask
        if context is not None:
            inp = torch.cat([x_masked, context], dim=-1)
        else:
            inp = x_masked

        raw_params = self.param_net(inp)
        raw_params = raw_params.view(-1, self._n_transform_dims, self._n_params_per_dim)

        # Extract components to transform
        x_transform = x[:, mask == 0] if self.dim > 1 else x

        y_transform, log_det = self._spline_forward(x_transform, raw_params)

        y = x.clone()
        if self.dim > 1:
            y[:, mask == 0] = y_transform
        else:
            y = y_transform
        return y, log_det.sum(dim=-1)

    def inverse(
        self, y: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Inverse transform.

        Args:
            y: Latent ``(batch, dim)``.
            context: Optional context.

        Returns:
            Reconstructed data.
        """
        mask = self.mask  # type: ignore[assignment]
        y_masked = y * mask
        if context is not None:
            inp = torch.cat([y_masked, context], dim=-1)
        else:
            inp = y_masked

        raw_params = self.param_net(inp)
        raw_params = raw_params.view(-1, self._n_transform_dims, self._n_params_per_dim)

        y_transform = y[:, mask == 0] if self.dim > 1 else y
        x_transform = self._spline_inverse(y_transform, raw_params)

        x = y.clone()
        if self.dim > 1:
            x[:, mask == 0] = x_transform
        else:
            x = x_transform
        return x

    def _spline_forward(
        self, x: torch.Tensor, raw_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rational-quadratic spline forward transform.

        Uses the parameterisation from Durkan et al. (2019).

        Args:
            x: Values to transform ``(batch, d)``.
            raw_params: Unconstrained spline parameters ``(batch, d, 3K-1)``.

        Returns:
            (transformed values, log derivatives).
        """
        K = self.n_bins
        B = self.tail_bound

        # Split into widths, heights, derivatives
        W = torch.softmax(raw_params[..., :K], dim=-1) * 2 * B
        H = torch.softmax(raw_params[..., K : 2 * K], dim=-1) * 2 * B
        D = torch.softplus(raw_params[..., 2 * K :])

        # Prepend/append boundary derivatives = 1
        ones = torch.ones(*D.shape[:-1], 1, device=D.device)
        D = torch.cat([ones, D, ones], dim=-1)

        # Cumulative sums for bin edges
        cumwidths = torch.cumsum(W, dim=-1) - B
        cumheights = torch.cumsum(H, dim=-1) - B
        cumwidths = torch.cat(
            [torch.full_like(cumwidths[..., :1], -B), cumwidths], dim=-1
        )
        cumheights = torch.cat(
            [torch.full_like(cumheights[..., :1], -B), cumheights], dim=-1
        )

        # Find which bin each x falls in
        # Clamp to valid range; identity outside [-B, B]
        x_clamped = torch.clamp(x, -B + 1e-6, B - 1e-6)
        bin_idx = torch.searchsorted(cumwidths[..., 1:], x_clamped.unsqueeze(-1))
        bin_idx = bin_idx.squeeze(-1).clamp(0, K - 1)

        # Gather bin parameters
        w_k = W.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        h_k = H.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        d_k = D[..., :-1].gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        d_kp1 = D[..., 1:].gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        cw_k = cumwidths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        ch_k = cumheights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)

        # Normalised position within bin
        xi = (x_clamped - cw_k) / w_k
        xi = torch.clamp(xi, 0, 1)

        # Rational quadratic
        numerator = h_k * (d_k * xi.pow(2) + 2 * xi * (1 - xi))
        denominator = d_k + (d_kp1 + d_k - 2) * xi * (1 - xi)
        y = ch_k + numerator / (denominator + 1e-8)

        # Log derivative
        log_deriv = torch.log(
            h_k.pow(2)
            * (d_kp1 * xi.pow(2) + 2 * xi * (1 - xi) + d_k * (1 - xi).pow(2))
            / (denominator.pow(2) * w_k + 1e-8)
            + 1e-8
        )

        return y, log_deriv

    def _spline_inverse(
        self, y: torch.Tensor, raw_params: torch.Tensor
    ) -> torch.Tensor:
        """Inverse rational-quadratic spline (solve numerically).

        For production code a closed-form inverse of the rational-quadratic
        is possible but complex.  Here we use bisection for robustness.

        Args:
            y: Values to invert ``(batch, d)``.
            raw_params: Spline parameters (same as forward).

        Returns:
            Inverted values.
        """
        # Bisection-based inverse
        lo = torch.full_like(y, -self.tail_bound)
        hi = torch.full_like(y, self.tail_bound)

        for _ in range(50):  # bisection iterations
            mid = (lo + hi) / 2
            f_mid, _ = self._spline_forward(mid, raw_params)
            lo = torch.where(f_mid < y, mid, lo)
            hi = torch.where(f_mid >= y, mid, hi)

        return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Flow container
# ---------------------------------------------------------------------------


class _NormalizingFlow(nn.Module):
    """Stack of coupling layers forming a complete normalizing flow."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        hidden_dim: int,
        n_layers: int,
        flow_type: str,
        n_bins: int = 16,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(n_layers):
            mask_even = (i % 2 == 0)
            if flow_type == "real_nvp":
                layers.append(
                    _AffineCouplingLayer(dim, context_dim, hidden_dim, mask_even)
                )
            else:
                layers.append(
                    _NeuralSplineCouplingLayer(
                        dim, context_dim, hidden_dim, n_bins, mask_even=mask_even
                    )
                )
        self.layers = nn.ModuleList(layers)
        self.dim = dim

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward: data -> latent.

        Args:
            x: Data tensor ``(batch, dim)``.
            context: Optional context ``(batch, context_dim)``.

        Returns:
            (latent z, total log-determinant).
        """
        total_log_det = torch.zeros(x.size(0), device=x.device)
        z = x
        for layer in self.layers:
            z, ld = layer(z, context)  # type: ignore[operator]
            total_log_det = total_log_det + ld
        return z, total_log_det

    def inverse(
        self, z: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Inverse: latent -> data.

        Args:
            z: Latent tensor ``(batch, dim)``.
            context: Optional context ``(batch, context_dim)``.

        Returns:
            Reconstructed data.
        """
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x, context)  # type: ignore[operator]
        return x


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------


class NormalizingFlowModel(BaseModel):
    """Normalizing flow for conditional density estimation of price changes.

    Learns the full conditional distribution p(y | x) where *y* is the
    target (e.g. future returns) and *x* are the conditioning covariates
    (geopolitical indicators, lagged prices, etc.).

    Config params:
        target_col: Target column name (default ``"target"``).
        feature_cols: List of feature / context columns (auto-detected
            if ``None``).
        flow_type: ``"real_nvp"`` or ``"neural_spline"`` (default).
        n_layers: Number of coupling layers (default 8).
        hidden_dim: Width of conditioning MLPs (default 64).
        n_bins: Number of spline bins for neural spline flows (default 16).
        max_epochs: Training epochs (default 300).
        learning_rate: Adam LR (default 5e-4).
        batch_size: Training batch size (default 256).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._target_col: str = config.params.get("target_col", "target")
        self._feature_cols: list[str] | None = config.params.get("feature_cols")
        self._flow_type: str = config.params.get("flow_type", "neural_spline")
        self._n_layers: int = config.params.get("n_layers", 8)
        self._hidden_dim: int = config.params.get("hidden_dim", 64)
        self._n_bins: int = config.params.get("n_bins", 16)
        self._max_epochs: int = config.params.get("max_epochs", 300)
        self._lr: float = config.params.get("learning_rate", 5e-4)
        self._batch_size: int = config.params.get("batch_size", 256)

        self._flow: _NormalizingFlow | None = None
        self._fitted_feature_cols: list[str] = []
        self._target_mean: float = 0.0
        self._target_std: float = 1.0
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def fit(self, data: pl.DataFrame) -> None:
        """Train the normalizing flow on historical data.

        The model learns p(y | x) where y = target_col and x = feature_cols.

        Args:
            data: Training DataFrame.
        """
        self._fitted_feature_cols = self._resolve_feature_cols(data)
        self._validate_data(
            data,
            required_columns=[self._target_col, *self._fitted_feature_cols],
            min_rows=100,
        )

        y = data[self._target_col].to_numpy().astype(np.float32)
        self._target_mean = float(y.mean())
        self._target_std = float(y.std() + 1e-8)
        y_norm = (y - self._target_mean) / self._target_std

        X = (
            data.select(self._fitted_feature_cols)
            .to_numpy()
            .astype(np.float32)
        )
        context_dim = X.shape[1]

        self._flow = _NormalizingFlow(
            dim=1,  # univariate target
            context_dim=context_dim,
            hidden_dim=self._hidden_dim,
            n_layers=self._n_layers,
            flow_type=self._flow_type,
            n_bins=self._n_bins,
        ).to(self._device)

        y_t = torch.tensor(y_norm.reshape(-1, 1), dtype=torch.float32)
        x_t = torch.tensor(X, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(y_t, x_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self._batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self._flow.parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._max_epochs
        )

        best_loss = float("inf")
        patience_counter = 0
        patience = 30

        for epoch in range(self._max_epochs):
            self._flow.train()
            epoch_loss = 0.0
            for yb, xb in loader:
                yb = yb.to(self._device)
                xb = xb.to(self._device)

                optimizer.zero_grad()
                z, log_det = self._flow(yb, context=xb)
                # Log-likelihood = log p_z(z) + log |det J|
                log_pz = -0.5 * (z.pow(2) + np.log(2 * np.pi)).sum(dim=-1)
                loss = -(log_pz + log_det).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._flow.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / len(loader)

            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 50 == 0:
                logger.info(
                    "Flow epoch {}/{}: NLL={:.4f}", epoch + 1, self._max_epochs, avg_loss
                )

            if patience_counter >= patience:
                logger.info("Early stopping at epoch {}", epoch + 1)
                break

        self._mark_fitted(data)

    def predict(
        self,
        horizon: int,
        n_scenarios: int = 1000,
    ) -> PredictionResult:
        """Generate scenarios by sampling from the learned density.

        Without conditioning context this draws unconditional samples.
        Prefer :meth:`sample` or :meth:`conditional_density` for
        conditioned predictions.

        Args:
            horizon: Number of forecast steps.
            n_scenarios: Number of samples per step.

        Returns:
            PredictionResult with statistics from sampled distribution.
        """
        self._require_fitted()
        samples = self.sample(n=n_scenarios, context=None)

        point = [float(np.mean(samples))] * horizon
        lower = [float(np.quantile(samples, 0.05))] * horizon
        upper = [float(np.quantile(samples, 0.95))] * horizon

        return PredictionResult(
            point_forecast=point,
            lower_bounds={0.05: lower},
            upper_bounds={0.95: upper},
            metadata={"model": self.config.name, "n_samples": n_scenarios},
        )

    def get_params(self) -> dict[str, Any]:
        """Return model architecture and training info."""
        self._require_fitted()
        return {
            "flow_type": self._flow_type,
            "n_layers": self._n_layers,
            "hidden_dim": self._hidden_dim,
            "n_bins": self._n_bins,
            "feature_cols": self._fitted_feature_cols,
            "target_mean": self._target_mean,
            "target_std": self._target_std,
            "n_parameters": sum(
                p.numel() for p in self._flow.parameters()  # type: ignore[union-attr]
            ),
        }

    # ------------------------------------------------------------------
    # Density estimation interface
    # ------------------------------------------------------------------

    def sample(
        self,
        n: int,
        context: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw samples from the learned density.

        Args:
            n: Number of samples.
            context: Conditioning features ``(n, context_dim)`` or
                ``(1, context_dim)`` to broadcast.  If ``None``, samples
                are unconditional.

        Returns:
            Array of samples in original target space, shape ``(n,)``.
        """
        self._require_fitted()
        assert self._flow is not None
        self._flow.eval()

        z = torch.randn(n, 1, device=self._device)
        ctx: torch.Tensor | None = None
        if context is not None:
            ctx_np = context.astype(np.float32)
            if ctx_np.shape[0] == 1:
                ctx_np = np.repeat(ctx_np, n, axis=0)
            ctx = torch.tensor(ctx_np, dtype=torch.float32).to(self._device)

        with torch.no_grad():
            x = self._flow.inverse(z, context=ctx)
        samples = x.cpu().numpy().flatten()
        return samples * self._target_std + self._target_mean

    def log_prob(
        self,
        y: np.ndarray,
        context: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate the log-density at specified values.

        Args:
            y: Target values ``(n,)`` in original space.
            context: Conditioning features ``(n, context_dim)``.

        Returns:
            Log-probabilities ``(n,)``.
        """
        self._require_fitted()
        assert self._flow is not None
        self._flow.eval()

        y_norm = (y.astype(np.float32) - self._target_mean) / self._target_std
        y_t = torch.tensor(y_norm.reshape(-1, 1), dtype=torch.float32).to(self._device)

        ctx: torch.Tensor | None = None
        if context is not None:
            ctx = torch.tensor(
                context.astype(np.float32), dtype=torch.float32
            ).to(self._device)

        with torch.no_grad():
            z, log_det = self._flow(y_t, context=ctx)
            log_pz = -0.5 * (z.pow(2) + np.log(2 * np.pi)).sum(dim=-1)
            # Correct for standardisation Jacobian
            log_prob_vals = log_pz + log_det - np.log(self._target_std)

        return log_prob_vals.cpu().numpy()

    def conditional_density(
        self,
        context: np.ndarray,
        y_grid: np.ndarray | None = None,
        n_grid: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the full conditional density p(y | x) on a grid.

        Args:
            context: Single context vector ``(1, context_dim)`` or
                ``(context_dim,)``.
            y_grid: Explicit grid of y values.  If ``None``, an
                automatic grid spanning +/- 4 sigma is used.
            n_grid: Number of grid points when auto-generating.

        Returns:
            Tuple of ``(y_values, density_values)``, each ``(n_grid,)``.
        """
        self._require_fitted()

        if y_grid is None:
            y_grid = np.linspace(
                self._target_mean - 4 * self._target_std,
                self._target_mean + 4 * self._target_std,
                n_grid,
            ).astype(np.float32)

        ctx = context.reshape(1, -1) if context.ndim == 1 else context
        ctx_repeated = np.repeat(ctx, len(y_grid), axis=0)

        log_p = self.log_prob(y_grid, context=ctx_repeated)
        density = np.exp(log_p)

        return y_grid, density

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_feature_cols(self, data: pl.DataFrame) -> list[str]:
        """Determine feature columns."""
        if self._feature_cols is not None:
            return self._feature_cols
        numeric_dtypes = {pl.Float32, pl.Float64, pl.Int32, pl.Int64}
        return [
            c
            for c in data.columns
            if c != self._target_col and data[c].dtype in numeric_dtypes
        ]
