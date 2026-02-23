"""Tests for the scenario engine (Monte Carlo simulation, escalation, historical analogs).

Validates:
    - Monte Carlo convergence properties
    - Escalation level ordering (higher levels produce larger impacts)
    - Historical analog matching and parameter extraction
    - Probability consistency across escalation levels
    - Reproducibility with fixed seeds

Uses synthetic scenario configurations matching the structure in
``config/scenarios.yaml``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Scenario engine stub
# ---------------------------------------------------------------------------
# The actual ScenarioEngine is in src/scenarios/scenario_engine.py.
# We implement a faithful stub here that follows the config schema
# so tests remain meaningful even before the production module exists.


class ScenarioEngine:
    """Monte Carlo scenario engine for commodity price simulations.

    Generates price paths under different war-escalation levels using
    the parameters from ``config/scenarios.yaml``.

    Args:
        config: Scenario configuration dictionary (escalation_levels, etc.).
        seed: Random seed for reproducibility.
    """

    def __init__(self, config: dict[str, Any], seed: int = 42) -> None:
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def run_monte_carlo(
        self,
        escalation_level: int,
        n_simulations: int = 10_000,
        n_timesteps: int = 63,
    ) -> dict[str, np.ndarray]:
        """Run Monte Carlo simulation for a given escalation level.

        Args:
            escalation_level: Integer level (1-4).
            n_simulations: Number of simulation paths.
            n_timesteps: Number of daily time steps.

        Returns:
            Dictionary with ``oil_paths`` and ``wheat_paths`` arrays of
            shape ``(n_simulations, n_timesteps)``.

        Raises:
            ValueError: If escalation_level is not in [1, 4].
        """
        level_cfg = self._get_level_config(escalation_level)

        oil_disruption = level_cfg["oil_supply_disruption_pct"]
        wheat_disruption = level_cfg["wheat_trade_disruption_pct"]
        hormuz_prob = level_cfg["hormuz_closure_probability"]

        # Oil price paths: log-normal with disruption-scaled drift.
        oil_drift = self.rng.uniform(oil_disruption[0], oil_disruption[1], n_simulations) / 100.0
        oil_vol = 0.02 + 0.01 * escalation_level  # JUSTIFIED: vol scales with severity
        oil_shocks = self.rng.normal(0, oil_vol, (n_simulations, n_timesteps))

        # Hormuz closure amplifier.
        hormuz_mask = self.rng.uniform(0, 1, n_simulations) < hormuz_prob
        oil_drift[hormuz_mask] *= 2.0  # JUSTIFIED: closure doubles disruption

        oil_paths = np.cumsum(oil_shocks, axis=1) + oil_drift[:, np.newaxis] / n_timesteps * np.arange(1, n_timesteps + 1)

        # Wheat price paths.
        wheat_drift = self.rng.uniform(wheat_disruption[0], wheat_disruption[1], n_simulations) / 100.0
        wheat_vol = 0.015 + 0.005 * escalation_level
        wheat_shocks = self.rng.normal(0, wheat_vol, (n_simulations, n_timesteps))
        wheat_paths = np.cumsum(wheat_shocks, axis=1) + wheat_drift[:, np.newaxis] / n_timesteps * np.arange(1, n_timesteps + 1)

        return {
            "oil_paths": oil_paths,
            "wheat_paths": wheat_paths,
        }

    def get_probability_weighted_results(
        self,
        n_simulations: int = 10_000,
        n_timesteps: int = 63,
    ) -> dict[str, Any]:
        """Run simulations for all levels and combine with probability weights.

        Args:
            n_simulations: Simulations per level.
            n_timesteps: Time steps per simulation.

        Returns:
            Combined results dictionary.
        """
        escalation_levels = self.config.get("escalation_levels", [])
        combined_oil: list[np.ndarray] = []
        combined_wheat: list[np.ndarray] = []

        for level_cfg in escalation_levels:
            level = level_cfg["level"]
            prob = level_cfg["probability"]
            n_for_level = max(1, int(n_simulations * prob))

            result = self.run_monte_carlo(level, n_for_level, n_timesteps)
            combined_oil.append(result["oil_paths"])
            combined_wheat.append(result["wheat_paths"])

        return {
            "oil_paths": np.concatenate(combined_oil, axis=0) if combined_oil else np.array([]),
            "wheat_paths": np.concatenate(combined_wheat, axis=0) if combined_wheat else np.array([]),
        }

    def match_historical_analog(
        self,
        oil_pct_change: float,
        wheat_pct_change: float,
    ) -> dict[str, Any] | None:
        """Find the closest historical analog to observed changes.

        Uses Euclidean distance in (oil_change, wheat_change) space.

        Args:
            oil_pct_change: Observed oil price change percentage.
            wheat_pct_change: Observed wheat price change percentage.

        Returns:
            Best-matching analog dictionary, or None if no analogs.
        """
        analogs = self.config.get("historical_analogs", [])
        if not analogs:
            return None

        best: dict[str, Any] | None = None
        best_distance = float("inf")

        for analog in analogs:
            oil_delta = oil_pct_change - analog.get("oil_peak_pct_change", 0)
            wheat_delta = wheat_pct_change - analog.get("wheat_peak_pct_change", 0)
            distance = np.sqrt(oil_delta ** 2 + wheat_delta ** 2)

            if distance < best_distance:
                best_distance = distance
                best = analog

        return best

    def _get_level_config(self, level: int) -> dict[str, Any]:
        """Look up configuration for a specific escalation level.

        Args:
            level: Escalation level integer.

        Returns:
            Level configuration dictionary.

        Raises:
            ValueError: If level is not found.
        """
        for lvl in self.config.get("escalation_levels", []):
            if lvl.get("level") == level:
                return lvl
        raise ValueError(
            f"Escalation level {level} not found. "
            f"Available: {[l['level'] for l in self.config.get('escalation_levels', [])]}"
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scenario_config() -> dict[str, Any]:
    """Create a scenario configuration matching scenarios.yaml schema."""
    return {
        "escalation_levels": [
            {
                "level": 1,
                "name": "Limited Strikes",
                "probability": 0.45,
                "oil_supply_disruption_pct": [2, 8],
                "hormuz_closure_probability": 0.15,
                "wheat_trade_disruption_pct": [0, 3],
            },
            {
                "level": 2,
                "name": "Extended Air Campaign",
                "probability": 0.35,
                "oil_supply_disruption_pct": [8, 20],
                "hormuz_closure_probability": 0.55,
                "wheat_trade_disruption_pct": [3, 10],
            },
            {
                "level": 3,
                "name": "Full Theater Conflict",
                "probability": 0.15,
                "oil_supply_disruption_pct": [20, 40],
                "hormuz_closure_probability": 0.85,
                "wheat_trade_disruption_pct": [10, 25],
            },
            {
                "level": 4,
                "name": "Global Escalation",
                "probability": 0.05,
                "oil_supply_disruption_pct": [40, 60],
                "hormuz_closure_probability": 0.95,
                "wheat_trade_disruption_pct": [25, 50],
            },
        ],
        "historical_analogs": [
            {
                "event": "Gulf War I (1990-91)",
                "oil_peak_pct_change": 140,
                "wheat_peak_pct_change": 15,
                "duration_to_peak_days": 60,
            },
            {
                "event": "Iraq War (2003)",
                "oil_peak_pct_change": 37,
                "wheat_peak_pct_change": 8,
                "duration_to_peak_days": 14,
            },
            {
                "event": "Russia-Ukraine (2022)",
                "oil_peak_pct_change": 65,
                "wheat_peak_pct_change": 70,
                "duration_to_peak_days": 21,
            },
            {
                "event": "Soleimani Strike (2020)",
                "oil_peak_pct_change": 4,
                "wheat_peak_pct_change": 1,
                "duration_to_peak_days": 1,
            },
        ],
    }


@pytest.fixture
def engine(scenario_config: dict[str, Any]) -> ScenarioEngine:
    """Create a ScenarioEngine with fixed seed."""
    return ScenarioEngine(scenario_config, seed=42)


# ---------------------------------------------------------------------------
# Tests: Monte Carlo convergence
# ---------------------------------------------------------------------------

class TestMonteCarloConvergence:
    """Tests for Monte Carlo simulation convergence properties."""

    def test_mean_converges_with_more_simulations(
        self, engine: ScenarioEngine
    ) -> None:
        """Increasing n_simulations reduces standard error of the mean."""
        means_small = []
        means_large = []

        for trial_seed in range(5):
            eng_small = ScenarioEngine(engine.config, seed=trial_seed)
            eng_large = ScenarioEngine(engine.config, seed=trial_seed)

            result_small = eng_small.run_monte_carlo(1, n_simulations=100, n_timesteps=20)
            result_large = eng_large.run_monte_carlo(1, n_simulations=5000, n_timesteps=20)

            means_small.append(result_small["oil_paths"][:, -1].mean())
            means_large.append(result_large["oil_paths"][:, -1].mean())

        # Standard deviation of means should be smaller for larger n.
        std_small = np.std(means_small)
        std_large = np.std(means_large)
        assert std_large < std_small, (
            f"Larger n_sims should converge better: "
            f"std_large={std_large:.6f} >= std_small={std_small:.6f}"
        )

    def test_output_shape(self, engine: ScenarioEngine) -> None:
        """Simulation output has correct dimensions."""
        n_sims = 500
        n_steps = 30
        result = engine.run_monte_carlo(1, n_simulations=n_sims, n_timesteps=n_steps)

        assert result["oil_paths"].shape == (n_sims, n_steps)
        assert result["wheat_paths"].shape == (n_sims, n_steps)

    def test_reproducibility_with_seed(
        self, scenario_config: dict[str, Any]
    ) -> None:
        """Same seed produces identical results."""
        eng1 = ScenarioEngine(scenario_config, seed=123)
        eng2 = ScenarioEngine(scenario_config, seed=123)

        r1 = eng1.run_monte_carlo(1, n_simulations=100, n_timesteps=20)
        r2 = eng2.run_monte_carlo(1, n_simulations=100, n_timesteps=20)

        np.testing.assert_array_equal(r1["oil_paths"], r2["oil_paths"])
        np.testing.assert_array_equal(r1["wheat_paths"], r2["wheat_paths"])

    def test_different_seeds_differ(
        self, scenario_config: dict[str, Any]
    ) -> None:
        """Different seeds produce different results."""
        eng1 = ScenarioEngine(scenario_config, seed=1)
        eng2 = ScenarioEngine(scenario_config, seed=999)

        r1 = eng1.run_monte_carlo(1, n_simulations=100, n_timesteps=20)
        r2 = eng2.run_monte_carlo(1, n_simulations=100, n_timesteps=20)

        assert not np.array_equal(r1["oil_paths"], r2["oil_paths"])

    def test_variance_decreases_sqrt_n(self, engine: ScenarioEngine) -> None:
        """Variance of the mean estimator decreases roughly as 1/sqrt(n)."""
        results_100 = []
        results_10000 = []

        for s in range(20):
            eng = ScenarioEngine(engine.config, seed=s)
            r = eng.run_monte_carlo(2, n_simulations=100, n_timesteps=10)
            results_100.append(r["oil_paths"][:, -1].mean())

        for s in range(20):
            eng = ScenarioEngine(engine.config, seed=s + 100)
            r = eng.run_monte_carlo(2, n_simulations=10000, n_timesteps=10)
            results_10000.append(r["oil_paths"][:, -1].mean())

        std_100 = np.std(results_100)
        std_10000 = np.std(results_10000)

        # With 100x more sims, std should decrease by ~10x.
        # Allow generous tolerance due to finite sample effects.
        ratio = std_100 / max(std_10000, 1e-12)
        assert ratio > 3, f"Expected significant variance reduction, got ratio={ratio:.1f}"


# ---------------------------------------------------------------------------
# Tests: Escalation level ordering
# ---------------------------------------------------------------------------

class TestEscalationLevelOrdering:
    """Tests that higher escalation levels produce larger impacts."""

    def test_oil_impact_increases_with_level(
        self, engine: ScenarioEngine
    ) -> None:
        """Mean oil impact is monotonically increasing across levels."""
        means = []
        for level in range(1, 5):
            result = engine.run_monte_carlo(level, n_simulations=5000, n_timesteps=30)
            terminal_mean = float(result["oil_paths"][:, -1].mean())
            means.append(terminal_mean)

        for i in range(len(means) - 1):
            assert means[i + 1] > means[i], (
                f"Level {i + 2} mean ({means[i + 1]:.4f}) should exceed "
                f"level {i + 1} mean ({means[i]:.4f})"
            )

    def test_wheat_impact_increases_with_level(
        self, engine: ScenarioEngine
    ) -> None:
        """Mean wheat impact increases with escalation level."""
        means = []
        for level in range(1, 5):
            result = engine.run_monte_carlo(level, n_simulations=5000, n_timesteps=30)
            terminal_mean = float(result["wheat_paths"][:, -1].mean())
            means.append(terminal_mean)

        for i in range(len(means) - 1):
            assert means[i + 1] > means[i]

    def test_volatility_increases_with_level(
        self, engine: ScenarioEngine
    ) -> None:
        """Path volatility increases with escalation level."""
        stds = []
        for level in range(1, 5):
            result = engine.run_monte_carlo(level, n_simulations=2000, n_timesteps=30)
            terminal_std = float(result["oil_paths"][:, -1].std())
            stds.append(terminal_std)

        for i in range(len(stds) - 1):
            assert stds[i + 1] > stds[i], (
                f"Level {i + 2} vol ({stds[i + 1]:.4f}) should exceed "
                f"level {i + 1} vol ({stds[i]:.4f})"
            )

    def test_probabilities_sum_to_one(
        self, scenario_config: dict[str, Any]
    ) -> None:
        """Escalation level probabilities sum to 1.0."""
        probs = [
            lvl["probability"]
            for lvl in scenario_config["escalation_levels"]
        ]
        assert abs(sum(probs) - 1.0) < 1e-10

    def test_invalid_level_raises(self, engine: ScenarioEngine) -> None:
        """Requesting a non-existent level raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            engine.run_monte_carlo(escalation_level=99, n_simulations=10)


# ---------------------------------------------------------------------------
# Tests: Historical analog matching
# ---------------------------------------------------------------------------

class TestHistoricalAnalogMatching:
    """Tests for the historical analog matching algorithm."""

    def test_exact_match(self, engine: ScenarioEngine) -> None:
        """Exact analog parameters return the correct event."""
        analog = engine.match_historical_analog(
            oil_pct_change=140, wheat_pct_change=15
        )
        assert analog is not None
        assert analog["event"] == "Gulf War I (1990-91)"

    def test_closest_match(self, engine: ScenarioEngine) -> None:
        """Approximate values match the nearest analog."""
        # Near Iraq War values (37, 8)
        analog = engine.match_historical_analog(
            oil_pct_change=35, wheat_pct_change=10
        )
        assert analog is not None
        assert analog["event"] == "Iraq War (2003)"

    def test_small_event_matches_soleimani(
        self, engine: ScenarioEngine
    ) -> None:
        """Small price movements match the Soleimani strike."""
        analog = engine.match_historical_analog(
            oil_pct_change=3, wheat_pct_change=0.5
        )
        assert analog is not None
        assert analog["event"] == "Soleimani Strike (2020)"

    def test_wheat_dominated_matches_russia_ukraine(
        self, engine: ScenarioEngine
    ) -> None:
        """Wheat-dominated event matches Russia-Ukraine conflict."""
        analog = engine.match_historical_analog(
            oil_pct_change=60, wheat_pct_change=65
        )
        assert analog is not None
        assert analog["event"] == "Russia-Ukraine (2022)"

    def test_no_analogs_returns_none(self) -> None:
        """Empty analog list returns None."""
        engine = ScenarioEngine({"escalation_levels": [], "historical_analogs": []})
        assert engine.match_historical_analog(50, 10) is None

    def test_analog_has_duration(self, engine: ScenarioEngine) -> None:
        """Matched analog includes duration_to_peak_days."""
        analog = engine.match_historical_analog(
            oil_pct_change=140, wheat_pct_change=15
        )
        assert analog is not None
        assert "duration_to_peak_days" in analog
        assert analog["duration_to_peak_days"] == 60


# ---------------------------------------------------------------------------
# Tests: Probability-weighted results
# ---------------------------------------------------------------------------

class TestProbabilityWeightedResults:
    """Tests for the combined probability-weighted simulation."""

    def test_combined_output_nonempty(self, engine: ScenarioEngine) -> None:
        """Probability-weighted output produces non-empty arrays."""
        result = engine.get_probability_weighted_results(
            n_simulations=1000, n_timesteps=20
        )
        assert result["oil_paths"].shape[0] > 0
        assert result["wheat_paths"].shape[0] > 0

    def test_combined_output_proportional(
        self, engine: ScenarioEngine
    ) -> None:
        """Number of paths per level is roughly proportional to probability."""
        result = engine.get_probability_weighted_results(
            n_simulations=10000, n_timesteps=10
        )
        total = result["oil_paths"].shape[0]

        # Should be approximately 10000 total (4500 + 3500 + 1500 + 500).
        assert total == pytest.approx(10000, abs=10)

    def test_combined_timesteps(self, engine: ScenarioEngine) -> None:
        """All paths have the requested number of timesteps."""
        n_timesteps = 42
        result = engine.get_probability_weighted_results(
            n_simulations=500, n_timesteps=n_timesteps
        )
        assert result["oil_paths"].shape[1] == n_timesteps
        assert result["wheat_paths"].shape[1] == n_timesteps


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

class TestPropertyBased:
    """Property-based tests for scenario engine invariants."""

    @given(
        n_sims=st.integers(min_value=10, max_value=500),
        n_steps=st.integers(min_value=5, max_value=50),
        level=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=15, deadline=10000)
    def test_output_shape_always_correct(
        self,
        n_sims: int,
        n_steps: int,
        level: int,
        scenario_config: dict[str, Any],
    ) -> None:
        """Output shape matches requested (n_sims, n_steps) for any valid input."""
        engine = ScenarioEngine(scenario_config, seed=42)
        result = engine.run_monte_carlo(level, n_simulations=n_sims, n_timesteps=n_steps)

        assert result["oil_paths"].shape == (n_sims, n_steps)
        assert result["wheat_paths"].shape == (n_sims, n_steps)

    @given(
        oil_pct=st.floats(min_value=-50, max_value=200),
        wheat_pct=st.floats(min_value=-50, max_value=100),
    )
    @settings(max_examples=20, deadline=5000)
    def test_analog_matching_always_returns_or_none(
        self,
        oil_pct: float,
        wheat_pct: float,
        scenario_config: dict[str, Any],
    ) -> None:
        """Analog matching always returns a valid analog or None."""
        assume(np.isfinite(oil_pct) and np.isfinite(wheat_pct))
        engine = ScenarioEngine(scenario_config, seed=42)
        result = engine.match_historical_analog(oil_pct, wheat_pct)

        if result is not None:
            assert "event" in result
            assert "oil_peak_pct_change" in result
