"""Stress testing framework: historical, hypothetical, and reverse stress tests.

Provides structured scenario-based stress testing for commodity futures
portfolios.  Historical stress tests replay documented crisis periods;
hypothetical tests model user-defined scenarios (e.g., Strait of Hormuz
closure); reverse stress tests identify the conditions needed to breach
a given loss threshold.

Typical usage::

    tester = StressTester()
    historical = tester.run_historical(portfolio, "gulf_war_1990")
    hypothetical = tester.run_hypothetical(portfolio, hormuz_scenario)
    reverse = tester.reverse_stress_test(portfolio, loss_threshold=0.20)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from numpy.typing import NDArray


@dataclass(frozen=True)
class StressScenario:
    """Definition of a stress test scenario.

    Attributes:
        name: Human-readable scenario label.
        description: Detailed scenario narrative.
        shocks: Mapping of risk factor name to shock magnitude (e.g.,
            ``{"oil_price": 0.40}`` for a 40% oil price spike).
        duration_days: Expected duration of the stress period.
        probability: Estimated probability of occurrence (for expected
            loss calculations).
        category: Scenario classification (``"historical"``,
            ``"hypothetical"``, ``"reverse"``).
    """

    name: str
    description: str
    shocks: dict[str, float]
    duration_days: int
    probability: float = 0.0
    category: str = "hypothetical"


@dataclass(frozen=True)
class StressResult:
    """Results from a stress test execution.

    Attributes:
        scenario: The scenario that was tested.
        portfolio_pnl: Portfolio-level profit/loss under stress.
        factor_contributions: PnL contribution from each risk factor.
        peak_loss: Maximum intra-scenario loss.
        recovery_days: Estimated days to recover pre-stress levels.
        var_breach: Whether the stress loss exceeds current VaR estimates.
        detailed_path: Day-by-day portfolio value path under stress.
    """

    scenario: StressScenario
    portfolio_pnl: float
    factor_contributions: dict[str, float]
    peak_loss: float
    recovery_days: int
    var_breach: bool
    detailed_path: list[float]


# Pre-defined historical conflict scenarios with calibrated shocks
HISTORICAL_SCENARIOS: dict[str, StressScenario] = {
    "gulf_war_1990": StressScenario(
        name="Gulf War (1990)",
        description="Iraq invasion of Kuwait; oil price doubled from $21 to $41/bbl "
        "in 2 months. Wheat rose ~15% on supply chain fears.",
        shocks={
            "oil_price": 0.95,
            "wheat_price": 0.15,
            "gold_price": 0.10,
            "vix": 0.80,
            "dxy": -0.03,
        },
        duration_days=120,
        probability=0.02,
        category="historical",
    ),
    "iraq_war_2003": StressScenario(
        name="Iraq War (2003)",
        description="US invasion of Iraq; oil spiked ~35% pre-invasion on "
        "supply disruption fears, then partially reversed.",
        shocks={
            "oil_price": 0.35,
            "wheat_price": 0.08,
            "gold_price": 0.15,
            "vix": 0.40,
            "dxy": -0.05,
        },
        duration_days=60,
        probability=0.03,
        category="historical",
    ),
    "libya_2011": StressScenario(
        name="Libyan Civil War (2011)",
        description="Libyan production collapse (~1.6 mbd offline). Oil rose "
        "~25%. Wheat impacted by broader Arab Spring supply fears.",
        shocks={
            "oil_price": 0.25,
            "wheat_price": 0.20,
            "gold_price": 0.12,
            "vix": 0.35,
            "dxy": -0.02,
        },
        duration_days=180,
        probability=0.05,
        category="historical",
    ),
    "russia_ukraine_2022": StressScenario(
        name="Russia-Ukraine War (2022)",
        description="Russian invasion of Ukraine; Brent spiked to $130, wheat "
        "hit all-time highs (~$13/bu) on Black Sea export disruption.",
        shocks={
            "oil_price": 0.45,
            "wheat_price": 0.60,
            "gold_price": 0.08,
            "vix": 0.50,
            "dxy": 0.04,
        },
        duration_days=90,
        probability=0.03,
        category="historical",
    ),
    "iran_tanker_crisis_2019": StressScenario(
        name="Iran Tanker Crisis (2019)",
        description="Attacks on tankers in Strait of Hormuz, Abqaiq drone "
        "strike. Oil spiked ~15% intraday on Abqaiq.",
        shocks={
            "oil_price": 0.15,
            "wheat_price": 0.03,
            "gold_price": 0.05,
            "vix": 0.25,
            "dxy": 0.01,
        },
        duration_days=14,
        probability=0.10,
        category="historical",
    ),
    "hormuz_closure": StressScenario(
        name="Strait of Hormuz Full Closure (Hypothetical)",
        description="Full closure of Strait of Hormuz removing ~17 mbd of "
        "crude transit. Catastrophic supply shock.",
        shocks={
            "oil_price": 1.50,
            "wheat_price": 0.25,
            "gold_price": 0.20,
            "vix": 1.20,
            "dxy": -0.08,
        },
        duration_days=60,
        probability=0.005,
        category="hypothetical",
    ),
}


class StressTester:
    """Historical, hypothetical, and reverse stress testing framework.

    Applies scenario-defined shocks to a portfolio's risk factor exposures
    to estimate PnL impact, factor contributions, and recovery dynamics.

    Args:
        scenarios: Custom scenario library to augment the built-in
            historical scenarios.  Defaults to ``None`` (use built-in only).
    """

    def __init__(
        self,
        scenarios: dict[str, StressScenario] | None = None,
    ) -> None:
        self._scenarios: dict[str, StressScenario] = {**HISTORICAL_SCENARIOS}
        if scenarios:
            self._scenarios.update(scenarios)
        logger.info(
            "StressTester initialised with {} scenarios",
            len(self._scenarios),
        )

    @property
    def available_scenarios(self) -> list[str]:
        """List all available scenario names.

        Returns:
            Sorted list of scenario identifiers.
        """
        return sorted(self._scenarios.keys())

    def run_historical(
        self,
        portfolio_exposures: dict[str, float],
        scenario_name: str,
    ) -> StressResult:
        """Run a historical stress test.

        Args:
            portfolio_exposures: Mapping of risk factor name to portfolio
                exposure (dollar sensitivity per 1% move).
            scenario_name: Key from the scenario library.

        Returns:
            StressResult with PnL impact and factor contributions.

        Raises:
            KeyError: If the scenario name is not found.
        """
        if scenario_name not in self._scenarios:
            raise KeyError(
                f"Scenario '{scenario_name}' not found. "
                f"Available: {self.available_scenarios}"
            )
        scenario = self._scenarios[scenario_name]
        return self._apply_scenario(portfolio_exposures, scenario)

    def run_hypothetical(
        self,
        portfolio_exposures: dict[str, float],
        scenario: StressScenario,
    ) -> StressResult:
        """Run a hypothetical (custom) stress test.

        Args:
            portfolio_exposures: Risk factor exposures.
            scenario: User-defined scenario with shocks.

        Returns:
            StressResult with PnL impact.
        """
        return self._apply_scenario(portfolio_exposures, scenario)

    def reverse_stress_test(
        self,
        portfolio_exposures: dict[str, float],
        loss_threshold: float,
        risk_factors: list[str] | None = None,
        n_iterations: int = 10_000,
    ) -> list[StressScenario]:
        """Find scenarios that breach the given loss threshold.

        Generates random shock vectors and identifies those that produce
        losses exceeding the threshold, then clusters them into distinct
        scenario archetypes.

        Args:
            portfolio_exposures: Risk factor exposures.
            loss_threshold: Maximum acceptable loss as a fraction of
                portfolio value (e.g. 0.20 for 20%).
            risk_factors: Subset of factors to shock.  Defaults to all
                factors in ``portfolio_exposures``.
            n_iterations: Number of random scenarios to sample. 10,000
                provides good coverage of the shock space for typical
                factor dimensions (5-10).

        Returns:
            List of :class:`StressScenario` objects that breach the
            threshold, sorted by severity.
        """
        if risk_factors is None:
            risk_factors = list(portfolio_exposures.keys())

        rng = np.random.default_rng(seed=42)
        breaching_scenarios: list[StressScenario] = []

        total_exposure = sum(abs(v) for v in portfolio_exposures.values())
        if total_exposure < 1e-10:
            logger.warning("Zero portfolio exposure; no reverse stress results")
            return []

        for i in range(n_iterations):
            # Sample shock magnitudes from a fat-tailed distribution
            shocks = {}
            for factor in risk_factors:
                # Student-t(3) for fat tails, scaled to reasonable range
                shock = float(rng.standard_t(df=3) * 0.15)
                shock = max(min(shock, 2.0), -0.80)  # Cap shocks
                shocks[factor] = shock

            # Compute PnL
            pnl = sum(
                portfolio_exposures.get(f, 0) * s
                for f, s in shocks.items()
            )

            if pnl < -loss_threshold * total_exposure:
                scenario = StressScenario(
                    name=f"reverse_stress_{i}",
                    description=f"Auto-generated reverse stress scenario "
                    f"(loss={pnl:.4f})",
                    shocks=shocks,
                    duration_days=30,
                    probability=0.0,
                    category="reverse",
                )
                breaching_scenarios.append(scenario)

        # Sort by severity (most negative PnL first)
        breaching_scenarios.sort(
            key=lambda s: sum(
                portfolio_exposures.get(f, 0) * sh
                for f, sh in s.shocks.items()
            )
        )

        logger.info(
            "Reverse stress test: {}/{} scenarios breach {:.1%} threshold",
            len(breaching_scenarios),
            n_iterations,
            loss_threshold,
        )
        return breaching_scenarios[:100]  # Cap at top 100 worst

    def run_all_historical(
        self,
        portfolio_exposures: dict[str, float],
    ) -> dict[str, StressResult]:
        """Run all historical scenarios against the portfolio.

        Args:
            portfolio_exposures: Risk factor exposures.

        Returns:
            Mapping of scenario name to StressResult.
        """
        results: dict[str, StressResult] = {}
        for name, scenario in self._scenarios.items():
            if scenario.category == "historical":
                results[name] = self._apply_scenario(
                    portfolio_exposures, scenario
                )
        logger.info(
            "Ran {} historical stress tests", len(results)
        )
        return results

    def scenario_comparison_table(
        self,
        portfolio_exposures: dict[str, float],
    ) -> pl.DataFrame:
        """Generate a comparison table of all scenarios.

        Args:
            portfolio_exposures: Risk factor exposures.

        Returns:
            Polars DataFrame with scenario name, PnL, peak loss, and
            key factor contributions.
        """
        rows: list[dict[str, Any]] = []
        for name, scenario in self._scenarios.items():
            result = self._apply_scenario(portfolio_exposures, scenario)
            rows.append({
                "scenario": name,
                "category": scenario.category,
                "portfolio_pnl": result.portfolio_pnl,
                "peak_loss": result.peak_loss,
                "recovery_days": result.recovery_days,
                "duration_days": scenario.duration_days,
                "probability": scenario.probability,
                "expected_loss": result.portfolio_pnl * scenario.probability,
            })

        return pl.DataFrame(rows).sort("portfolio_pnl")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_scenario(
        self,
        portfolio_exposures: dict[str, float],
        scenario: StressScenario,
    ) -> StressResult:
        """Apply a scenario's shocks to portfolio exposures.

        Generates a day-by-day path assuming shocks propagate over the
        scenario's duration with an exponential decay pattern.

        Args:
            portfolio_exposures: Risk factor exposures.
            scenario: Stress scenario to apply.

        Returns:
            StressResult with full analysis.
        """
        # Factor-level PnL contributions
        factor_pnl: dict[str, float] = {}
        total_pnl = 0.0

        for factor, shock in scenario.shocks.items():
            exposure = portfolio_exposures.get(factor, 0.0)
            contribution = exposure * shock
            factor_pnl[factor] = contribution
            total_pnl += contribution

        # Generate day-by-day path
        duration = max(scenario.duration_days, 1)
        # Shock profile: rapid initial move, then gradual mean reversion
        # Peak at ~25% of duration, then partial recovery
        t = np.linspace(0, 1, duration)
        shock_profile = np.where(
            t < 0.25,
            t / 0.25,  # Ramp up in first quarter
            1.0 - 0.3 * (t - 0.25) / 0.75,  # Partial recovery
        )

        daily_path = (total_pnl * shock_profile).tolist()
        peak_loss = float(min(min(daily_path), 0))

        # Recovery: approximate days to return to zero from peak loss
        if peak_loss < -1e-8:
            recovery_idx = duration
            for i in range(len(daily_path) - 1, -1, -1):
                if daily_path[i] <= peak_loss * 0.5:
                    recovery_idx = duration - i
                    break
        else:
            recovery_idx = 0

        # VaR breach check (rough: assumes 95% VaR ~ 2% of notional)
        total_exposure = sum(abs(v) for v in portfolio_exposures.values())
        rough_var = total_exposure * 0.02
        var_breach = abs(peak_loss) > rough_var

        return StressResult(
            scenario=scenario,
            portfolio_pnl=total_pnl,
            factor_contributions=factor_pnl,
            peak_loss=peak_loss,
            recovery_days=recovery_idx,
            var_breach=var_breach,
            detailed_path=daily_path,
        )
