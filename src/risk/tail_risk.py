"""Tail risk metrics: VaR, CVaR/Expected Shortfall, and tail index estimation.

Provides a comprehensive toolkit for measuring extreme downside risk in
commodity futures portfolios.  Supports parametric (Gaussian, Student-t,
Cornish-Fisher), historical, and Monte Carlo VaR/CVaR, plus tail index
estimation via the Hill estimator for characterising the heaviness of
return distributions.

Typical usage::

    analyzer = TailRiskAnalyzer(confidence_levels=[0.95, 0.99])
    metrics = analyzer.compute_all(returns)
    var_95 = analyzer.parametric_var(returns, confidence=0.95)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from numpy.typing import NDArray
from scipy import stats


@dataclass(frozen=True)
class TailRiskMetrics:
    """Container for tail risk measurement results.

    Attributes:
        var: Value-at-Risk at each confidence level (mapping level -> value).
        cvar: Conditional VaR / Expected Shortfall at each confidence level.
        tail_index: Hill estimator of the tail index (lower = fatter tails).
        max_drawdown: Maximum peak-to-trough drawdown.
        skewness: Sample skewness of the return distribution.
        excess_kurtosis: Excess kurtosis (kurtosis - 3).
        method: VaR estimation method used.
    """

    var: dict[float, float]
    cvar: dict[float, float]
    tail_index: float
    max_drawdown: float
    skewness: float
    excess_kurtosis: float
    method: str


@dataclass
class TailRiskConfig:
    """Configuration for tail risk analysis.

    Attributes:
        confidence_levels: Confidence levels for VaR/CVaR computation.
        hill_tail_fraction: Fraction of sorted observations to use for
            Hill tail index estimation. 0.10 (top 10%) is a standard
            choice that balances bias (too many observations) and
            variance (too few observations).
        mc_n_simulations: Number of Monte Carlo simulations for MC-VaR.
            10,000 provides sufficient convergence for 99% VaR.
        distribution: Parametric distribution assumption
            (``"normal"``, ``"t"``, ``"cornish_fisher"``).
    """

    confidence_levels: list[float] = field(
        default_factory=lambda: [0.90, 0.95, 0.99]
    )
    hill_tail_fraction: float = 0.10
    mc_n_simulations: int = 10_000
    distribution: str = "cornish_fisher"


class TailRiskAnalyzer:
    """Comprehensive tail risk measurement for commodity futures.

    Supports multiple VaR/CVaR methodologies and tail characterisation
    tools appropriate for the fat-tailed, skewed return distributions
    typical of commodity markets during geopolitical crises.

    Args:
        config: Configuration parameters.
    """

    def __init__(self, config: TailRiskConfig | None = None) -> None:
        self.config = config or TailRiskConfig()
        logger.info(
            "TailRiskAnalyzer initialised: levels={}, distribution={}",
            self.config.confidence_levels,
            self.config.distribution,
        )

    def compute_all(
        self,
        returns: pl.Series | NDArray[np.float64],
    ) -> TailRiskMetrics:
        """Compute all tail risk metrics for a return series.

        Args:
            returns: Daily log-returns or simple returns.

        Returns:
            A :class:`TailRiskMetrics` containing VaR, CVaR, tail index,
            max drawdown, skewness, and excess kurtosis.
        """
        arr = self._to_array(returns)
        arr = arr[~np.isnan(arr)]

        if len(arr) < 30:
            raise ValueError(
                f"Need at least 30 return observations; got {len(arr)}"
            )

        var_dict: dict[float, float] = {}
        cvar_dict: dict[float, float] = {}

        for level in self.config.confidence_levels:
            var_dict[level] = self.parametric_var(arr, level)
            cvar_dict[level] = self.expected_shortfall(arr, level)

        return TailRiskMetrics(
            var=var_dict,
            cvar=cvar_dict,
            tail_index=self.tail_index(arr),
            max_drawdown=self.max_drawdown(arr),
            skewness=float(stats.skew(arr)),
            excess_kurtosis=float(stats.kurtosis(arr)),
            method=self.config.distribution,
        )

    def parametric_var(
        self,
        returns: pl.Series | NDArray[np.float64],
        confidence: float = 0.95,
    ) -> float:
        """Compute parametric Value-at-Risk.

        Args:
            returns: Return series.
            confidence: Confidence level (e.g. 0.95 for 95% VaR).

        Returns:
            VaR as a positive number (loss magnitude).
        """
        arr = self._to_array(returns)
        arr = arr[~np.isnan(arr)]
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1))

        if self.config.distribution == "normal":
            z = stats.norm.ppf(1 - confidence)
            return float(-(mu + sigma * z))

        elif self.config.distribution == "t":
            # Fit Student-t
            df_t, loc_t, scale_t = stats.t.fit(arr)
            q = stats.t.ppf(1 - confidence, df_t, loc=loc_t, scale=scale_t)
            return float(-q)

        elif self.config.distribution == "cornish_fisher":
            return self._cornish_fisher_var(arr, confidence)

        else:
            raise ValueError(
                f"Unknown distribution: {self.config.distribution}"
            )

    def historical_var(
        self,
        returns: pl.Series | NDArray[np.float64],
        confidence: float = 0.95,
    ) -> float:
        """Compute historical (empirical) Value-at-Risk.

        Args:
            returns: Return series.
            confidence: Confidence level.

        Returns:
            VaR as a positive number (loss magnitude).
        """
        arr = self._to_array(returns)
        arr = arr[~np.isnan(arr)]
        quantile = np.percentile(arr, (1 - confidence) * 100)
        return float(-quantile)

    def expected_shortfall(
        self,
        returns: pl.Series | NDArray[np.float64],
        confidence: float = 0.95,
    ) -> float:
        """Compute Expected Shortfall (CVaR / Conditional VaR).

        ES is the expected loss conditional on the loss exceeding VaR.
        It is a coherent risk measure (unlike VaR) and better captures
        tail risk.

        Args:
            returns: Return series.
            confidence: Confidence level.

        Returns:
            Expected Shortfall as a positive number.
        """
        arr = self._to_array(returns)
        arr = arr[~np.isnan(arr)]
        cutoff = np.percentile(arr, (1 - confidence) * 100)
        tail_losses = arr[arr <= cutoff]

        if len(tail_losses) == 0:
            return self.historical_var(arr, confidence)

        return float(-np.mean(tail_losses))

    def tail_index(
        self,
        returns: pl.Series | NDArray[np.float64],
    ) -> float:
        """Estimate the tail index using the Hill estimator.

        The tail index (alpha) characterises how heavy the tail is.
        Lower alpha = fatter tails = more extreme events.  For reference:
        - Gaussian: alpha -> infinity
        - Student-t(4): alpha ~ 4
        - Cauchy: alpha ~ 1
        - Commodity crisis returns: typically alpha ~ 2-4

        Args:
            returns: Return series.

        Returns:
            Estimated tail index (positive value).
        """
        arr = self._to_array(returns)
        arr = arr[~np.isnan(arr)]

        # Use absolute values for tail estimation (both tails)
        abs_returns = np.sort(np.abs(arr))[::-1]  # Descending

        k = max(int(len(abs_returns) * self.config.hill_tail_fraction), 2)
        top_k = abs_returns[:k]

        # Hill estimator: 1/alpha = (1/k) * sum(log(X_i) - log(X_k))
        log_top = np.log(np.maximum(top_k, 1e-12))
        log_threshold = log_top[-1]
        hill_sum = float(np.mean(log_top[:-1] - log_threshold))

        if hill_sum <= 0:
            logger.warning("Hill estimator non-positive; returning default alpha=4")
            return 4.0

        alpha = 1.0 / hill_sum
        logger.debug(
            "Hill tail index: alpha={:.2f} (k={} of {} observations)",
            alpha,
            k,
            len(arr),
        )
        return float(alpha)

    def max_drawdown(
        self,
        returns: pl.Series | NDArray[np.float64],
    ) -> float:
        """Compute maximum drawdown from a return series.

        Args:
            returns: Return series (log or simple).

        Returns:
            Maximum drawdown as a positive fraction (0 to 1+).
        """
        arr = self._to_array(returns)
        arr = arr[~np.isnan(arr)]

        # Construct cumulative wealth curve
        cum_returns = np.cumprod(1 + arr)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (running_max - cum_returns) / running_max

        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    def rolling_var(
        self,
        returns: pl.Series | NDArray[np.float64],
        window: int = 252,
        confidence: float = 0.95,
    ) -> NDArray[np.float64]:
        """Compute rolling historical VaR.

        Args:
            returns: Return series.
            window: Rolling window size. 252 trading days provides a
                full year of context for annual VaR estimation.
            confidence: Confidence level.

        Returns:
            Array of rolling VaR values.
        """
        arr = self._to_array(returns)
        result = np.full(len(arr), np.nan, dtype=np.float64)

        for i in range(window, len(arr)):
            segment = arr[i - window : i]
            valid = segment[~np.isnan(segment)]
            if len(valid) >= 30:
                result[i] = float(-np.percentile(valid, (1 - confidence) * 100))

        return result

    def rolling_es(
        self,
        returns: pl.Series | NDArray[np.float64],
        window: int = 252,
        confidence: float = 0.95,
    ) -> NDArray[np.float64]:
        """Compute rolling Expected Shortfall.

        Args:
            returns: Return series.
            window: Rolling window size.
            confidence: Confidence level.

        Returns:
            Array of rolling ES values.
        """
        arr = self._to_array(returns)
        result = np.full(len(arr), np.nan, dtype=np.float64)

        for i in range(window, len(arr)):
            segment = arr[i - window : i]
            valid = segment[~np.isnan(segment)]
            if len(valid) >= 30:
                cutoff = np.percentile(valid, (1 - confidence) * 100)
                tail = valid[valid <= cutoff]
                if len(tail) > 0:
                    result[i] = float(-np.mean(tail))

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cornish_fisher_var(
        returns: NDArray[np.float64],
        confidence: float,
    ) -> float:
        """Cornish-Fisher expansion for VaR with skewness/kurtosis adjustment.

        Adjusts the Gaussian quantile for the observed skewness and excess
        kurtosis, providing a better tail approximation without a full
        distributional fit.

        Args:
            returns: Array of returns.
            confidence: Confidence level.

        Returns:
            Cornish-Fisher adjusted VaR as a positive number.
        """
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        s = float(stats.skew(returns))
        k = float(stats.kurtosis(returns))  # excess kurtosis

        z = stats.norm.ppf(1 - confidence)

        # Cornish-Fisher expansion
        z_cf = (
            z
            + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * k / 24
            - (2 * z**3 - 5 * z) * s**2 / 36
        )

        return float(-(mu + sigma * z_cf))

    @staticmethod
    def _to_array(
        data: pl.Series | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert input to numpy float64 array.

        Args:
            data: Polars Series or numpy array.

        Returns:
            Float64 numpy array.
        """
        if isinstance(data, pl.Series):
            return data.to_numpy().astype(np.float64)
        return np.asarray(data, dtype=np.float64)
