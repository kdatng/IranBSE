"""Tests for the Strait of Hormuz chokepoint model.

Validates the physical and operational dynamics of Hormuz strait disruption:
    - Mine laying and clearing dynamics
    - Bypass pipeline capacity constraints
    - Insurance/shipping response triggers
    - Supply flow calculations

Parameters are sourced from ``config/scenarios.yaml`` with full
research-backed justifications (IEA, DIA, CENTCOM, etc.).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Hormuz model stub
# ---------------------------------------------------------------------------
# Faithful implementation of the Strait of Hormuz disruption model
# based on config/scenarios.yaml parameters.


class HormuzModel:
    """Model for Strait of Hormuz chokepoint disruption dynamics.

    Simulates mine laying/clearing, bypass pipeline utilization, and
    insurance/shipping market responses during conflict scenarios.

    Args:
        config: Hormuz-specific configuration from scenarios.yaml.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        # Core flow parameters.
        self.daily_flow_mbpd: float = config.get("daily_flow_mbpd", 20.0)
        self.bypass_capacity_mbpd: float = config.get("bypass_pipeline_capacity_mbpd", 4.2)
        self.bypass_coverage_pct: float = config.get("bypass_coverage_pct", 0.21)

        # Mine parameters.
        self.mine_inventory: list[int] = config.get("mine_inventory", [5000, 6000])
        self.mine_laying_rate: int = config.get("mine_laying_rate_per_day", 100)
        self.mine_clearing_weeks: list[int] = config.get("mine_clearing_duration_weeks", [8, 26])

        # Insurance parameters.
        self.war_risk_baseline_pct: float = config.get("war_risk_premium_baseline_pct", 0.07)
        self.war_risk_peak_pct: float = config.get("war_risk_premium_peak_pct", 1.0)
        self.freight_surge_pct: float = config.get("freight_rate_surge_pct", 30)
        self.cape_extra_days: list[int] = config.get("cape_of_good_hope_extra_days", [10, 14])

    def compute_mine_density(
        self,
        days_of_laying: int,
        laying_rate_override: int | None = None,
    ) -> int:
        """Calculate total mines deployed after a given number of days.

        Args:
            days_of_laying: Number of days Iran lays mines.
            laying_rate_override: Override for the daily laying rate.

        Returns:
            Total mines deployed, capped at inventory maximum.
        """
        rate = laying_rate_override or self.mine_laying_rate
        total = rate * days_of_laying
        max_inventory = self.mine_inventory[1]  # Upper bound
        return min(total, max_inventory)

    def compute_clearing_timeline_days(
        self,
        mines_deployed: int,
        mcm_vessels: int = 16,
        mines_per_vessel_per_day: float = 3.0,
    ) -> tuple[int, int]:
        """Estimate mine clearing duration in days.

        Args:
            mines_deployed: Number of mines in the strait.
            mcm_vessels: Available mine countermeasure vessels.
            mines_per_vessel_per_day: Clearance rate per vessel per day.
                JUSTIFIED: Conservative estimate for mixed mine types.

        Returns:
            Tuple of (optimistic_days, pessimistic_days).
        """
        daily_clearance = mcm_vessels * mines_per_vessel_per_day
        if daily_clearance == 0:
            return (365, 365)  # Effectively permanent closure

        base_days = int(np.ceil(mines_deployed / daily_clearance))

        # JUSTIFIED: Pentagon estimates 2x-3x for verification sweeps.
        optimistic = base_days
        pessimistic = base_days * 3  # JUSTIFIED: multiple verification passes

        # Clamp to research-backed range (8-26 weeks).
        min_days = self.mine_clearing_weeks[0] * 7
        max_days = self.mine_clearing_weeks[1] * 7

        return (
            max(optimistic, min_days),
            min(pessimistic, max_days),
        )

    def compute_supply_disruption(
        self,
        closure_fraction: float,
    ) -> dict[str, float]:
        """Calculate oil supply disruption given a strait closure fraction.

        Args:
            closure_fraction: Fraction of Hormuz transit blocked [0, 1].

        Returns:
            Dictionary with disrupted/bypassed/net_lost volumes in mb/d.

        Raises:
            ValueError: If closure_fraction is outside [0, 1].
        """
        if not 0.0 <= closure_fraction <= 1.0:
            raise ValueError(
                f"closure_fraction must be in [0, 1], got {closure_fraction}"
            )

        disrupted_mbpd = self.daily_flow_mbpd * closure_fraction
        bypassed_mbpd = min(self.bypass_capacity_mbpd, disrupted_mbpd)
        net_lost_mbpd = max(0.0, disrupted_mbpd - bypassed_mbpd)

        return {
            "total_flow_mbpd": self.daily_flow_mbpd,
            "disrupted_mbpd": disrupted_mbpd,
            "bypassed_mbpd": bypassed_mbpd,
            "net_lost_mbpd": net_lost_mbpd,
            "closure_fraction": closure_fraction,
            "pct_global_loss": net_lost_mbpd / self.daily_flow_mbpd * 100,
        }

    def compute_insurance_response(
        self,
        mine_density: int,
        active_hostilities: bool = True,
    ) -> dict[str, float]:
        """Calculate war risk insurance premium and shipping response.

        Args:
            mine_density: Number of mines deployed in the strait.
            active_hostilities: Whether active combat is occurring.

        Returns:
            Insurance/shipping response metrics.
        """
        # JUSTIFIED: Insurance market research -- premium scales with threat.
        # Even credible threat alone triggers premium spikes.
        if active_hostilities and mine_density > 0:
            premium_pct = self.war_risk_peak_pct
        elif mine_density > 100:
            # JUSTIFIED: >100 mines = "credible threat" threshold
            premium_pct = self.war_risk_peak_pct * 0.7
        elif mine_density > 0:
            premium_pct = self.war_risk_baseline_pct * 5
        else:
            premium_pct = self.war_risk_baseline_pct

        # Premium per $100M vessel.
        premium_per_vessel = premium_pct / 100 * 100_000_000

        # Freight rate impact.
        if premium_pct > self.war_risk_baseline_pct * 3:
            freight_impact_pct = self.freight_surge_pct
        else:
            freight_impact_pct = self.freight_surge_pct * (
                premium_pct / self.war_risk_peak_pct
            )

        # Rerouting cost.
        extra_days = (
            self.cape_extra_days[1]
            if active_hostilities
            else self.cape_extra_days[0]
        )

        return {
            "war_risk_premium_pct": premium_pct,
            "premium_per_100m_vessel_usd": premium_per_vessel,
            "freight_rate_surge_pct": freight_impact_pct,
            "rerouting_extra_days": extra_days,
            "insurance_withdrawal": premium_pct >= self.war_risk_peak_pct * 0.5,
        }

    def simulate_disruption_timeline(
        self,
        days: int = 31,
        laying_days: int = 3,
        closure_fraction: float = 0.8,
        seed: int = 42,
    ) -> dict[str, np.ndarray]:
        """Simulate daily disruption dynamics over the conflict period.

        Args:
            days: Total simulation days.
            laying_days: Number of days mines are actively laid.
            closure_fraction: Maximum closure fraction reached.
            seed: Random seed.

        Returns:
            Time series of mine counts, flow, and insurance metrics.
        """
        rng = np.random.default_rng(seed)

        mine_count = np.zeros(days)
        daily_flow = np.full(days, self.daily_flow_mbpd)
        insurance_premium = np.zeros(days)

        for day in range(days):
            if day < laying_days:
                mine_count[day] = self.compute_mine_density(day + 1)
                frac = min(closure_fraction * (day + 1) / laying_days, closure_fraction)
            else:
                mine_count[day] = mine_count[min(day - 1, laying_days - 1)]
                frac = closure_fraction

            disruption = self.compute_supply_disruption(frac)
            daily_flow[day] = self.daily_flow_mbpd - disruption["net_lost_mbpd"]

            insurance = self.compute_insurance_response(
                int(mine_count[day]), active_hostilities=(day < laying_days + 7)
            )
            insurance_premium[day] = insurance["war_risk_premium_pct"]

        return {
            "mine_count": mine_count,
            "daily_flow_mbpd": daily_flow,
            "insurance_premium_pct": insurance_premium,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hormuz_config() -> dict[str, Any]:
    """Hormuz model configuration matching scenarios.yaml."""
    return {
        "daily_flow_mbpd": 20.0,
        "pct_global_oil_trade": 0.20,
        "pct_global_lng_trade": 0.22,
        "bypass_pipeline_capacity_mbpd": 4.2,
        "bypass_coverage_pct": 0.21,
        "petroline_design_mbpd": 7.0,
        "petroline_spare_mbpd": [2.8, 4.8],
        "adcop_capacity_mbpd": 1.8,
        "adcop_spare_mbpd": 0.7,
        "mine_inventory": [5000, 6000],
        "mine_laying_rate_per_day": 100,
        "mine_clearing_duration_weeks": [8, 26],
        "mcm_vessels_needed": 16,
        "war_risk_premium_baseline_pct": 0.07,
        "war_risk_premium_peak_pct": 1.0,
        "premium_per_100m_vessel_usd": 1_000_000,
        "freight_rate_surge_pct": 30,
        "cape_of_good_hope_extra_days": [10, 14],
    }


@pytest.fixture
def model(hormuz_config: dict[str, Any]) -> HormuzModel:
    """Create a HormuzModel with default configuration."""
    return HormuzModel(hormuz_config)


# ---------------------------------------------------------------------------
# Tests: Mine laying/clearing dynamics
# ---------------------------------------------------------------------------

class TestMineDynamics:
    """Tests for mine laying and clearing calculations."""

    def test_mine_density_increases_with_days(self, model: HormuzModel) -> None:
        """More laying days produce more mines (up to inventory cap)."""
        density_1 = model.compute_mine_density(1)
        density_3 = model.compute_mine_density(3)
        density_7 = model.compute_mine_density(7)

        assert density_1 < density_3 < density_7

    def test_mine_density_capped_at_inventory(self, model: HormuzModel) -> None:
        """Mine deployment cannot exceed inventory maximum."""
        density = model.compute_mine_density(days_of_laying=100)
        max_inventory = model.mine_inventory[1]
        assert density == max_inventory

    def test_mine_density_day_one(self, model: HormuzModel) -> None:
        """Day 1 mine density equals the daily laying rate."""
        density = model.compute_mine_density(1)
        assert density == model.mine_laying_rate

    def test_mine_density_zero_days(self, model: HormuzModel) -> None:
        """Zero laying days means zero mines."""
        assert model.compute_mine_density(0) == 0

    def test_clearing_timeline_increases_with_mines(
        self, model: HormuzModel
    ) -> None:
        """More mines take longer to clear."""
        opt_100, pess_100 = model.compute_clearing_timeline_days(100)
        opt_1000, pess_1000 = model.compute_clearing_timeline_days(1000)
        opt_5000, pess_5000 = model.compute_clearing_timeline_days(5000)

        assert opt_100 <= opt_1000 <= opt_5000
        assert pess_100 <= pess_1000 <= pess_5000

    def test_clearing_within_research_bounds(self, model: HormuzModel) -> None:
        """Clearing timeline stays within the 8-26 week research range."""
        for mines in [100, 500, 2000, 5000]:
            opt, pess = model.compute_clearing_timeline_days(mines)
            min_days = model.mine_clearing_weeks[0] * 7  # 56 days
            max_days = model.mine_clearing_weeks[1] * 7  # 182 days

            assert opt >= min_days, f"Optimistic {opt} < min {min_days} for {mines} mines"
            assert pess <= max_days, f"Pessimistic {pess} > max {max_days} for {mines} mines"

    def test_clearing_pessimistic_gte_optimistic(
        self, model: HormuzModel
    ) -> None:
        """Pessimistic estimate is always >= optimistic."""
        for mines in [100, 1000, 5000]:
            opt, pess = model.compute_clearing_timeline_days(mines)
            assert pess >= opt

    def test_zero_mcm_vessels_gives_max_duration(
        self, model: HormuzModel
    ) -> None:
        """Zero MCM vessels means effectively permanent closure."""
        opt, pess = model.compute_clearing_timeline_days(1000, mcm_vessels=0)
        assert opt >= 182  # At least max weeks
        assert pess >= 182

    def test_custom_laying_rate(self, model: HormuzModel) -> None:
        """Override laying rate changes mine density."""
        density_default = model.compute_mine_density(5)
        density_fast = model.compute_mine_density(5, laying_rate_override=200)
        assert density_fast > density_default


# ---------------------------------------------------------------------------
# Tests: Bypass pipeline capacity limits
# ---------------------------------------------------------------------------

class TestBypassPipeline:
    """Tests for bypass pipeline capacity constraints."""

    def test_bypass_cannot_exceed_capacity(self, model: HormuzModel) -> None:
        """Bypassed flow never exceeds pipeline capacity."""
        result = model.compute_supply_disruption(closure_fraction=1.0)
        assert result["bypassed_mbpd"] <= model.bypass_capacity_mbpd

    def test_full_closure_net_loss(self, model: HormuzModel) -> None:
        """Full closure net loss is total flow minus bypass capacity."""
        result = model.compute_supply_disruption(closure_fraction=1.0)
        expected_loss = model.daily_flow_mbpd - model.bypass_capacity_mbpd
        assert abs(result["net_lost_mbpd"] - expected_loss) < 0.01

    def test_no_closure_no_disruption(self, model: HormuzModel) -> None:
        """Zero closure means zero disruption."""
        result = model.compute_supply_disruption(closure_fraction=0.0)
        assert result["disrupted_mbpd"] == 0.0
        assert result["bypassed_mbpd"] == 0.0
        assert result["net_lost_mbpd"] == 0.0

    def test_partial_closure_within_bypass(self, model: HormuzModel) -> None:
        """Small closure within bypass capacity means zero net loss."""
        # 10% of 20 mbpd = 2 mbpd, which is within bypass capacity (4.2)
        result = model.compute_supply_disruption(closure_fraction=0.10)
        assert result["net_lost_mbpd"] == 0.0
        assert result["bypassed_mbpd"] == pytest.approx(2.0, abs=0.01)

    def test_bypass_coverage_matches_config(self, model: HormuzModel) -> None:
        """Bypass coverage percentage matches the configured value."""
        coverage = model.bypass_capacity_mbpd / model.daily_flow_mbpd
        assert abs(coverage - model.bypass_coverage_pct) < 0.01

    def test_invalid_closure_fraction_raises(self, model: HormuzModel) -> None:
        """Closure fraction outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError):
            model.compute_supply_disruption(closure_fraction=-0.1)
        with pytest.raises(ValueError):
            model.compute_supply_disruption(closure_fraction=1.5)

    def test_disruption_monotonic(self, model: HormuzModel) -> None:
        """Net loss increases monotonically with closure fraction."""
        losses = []
        for frac in np.linspace(0, 1, 20):
            result = model.compute_supply_disruption(closure_fraction=frac)
            losses.append(result["net_lost_mbpd"])

        for i in range(len(losses) - 1):
            assert losses[i + 1] >= losses[i] - 1e-10


# ---------------------------------------------------------------------------
# Tests: Insurance response triggers
# ---------------------------------------------------------------------------

class TestInsuranceResponse:
    """Tests for insurance premium and shipping market response."""

    def test_baseline_premium_no_mines(self, model: HormuzModel) -> None:
        """No mines and no hostilities returns baseline premium."""
        response = model.compute_insurance_response(
            mine_density=0, active_hostilities=False
        )
        assert response["war_risk_premium_pct"] == model.war_risk_baseline_pct

    def test_active_hostilities_peak_premium(self, model: HormuzModel) -> None:
        """Active hostilities with mines triggers peak premium."""
        response = model.compute_insurance_response(
            mine_density=500, active_hostilities=True
        )
        assert response["war_risk_premium_pct"] == model.war_risk_peak_pct

    def test_credible_threat_premium(self, model: HormuzModel) -> None:
        """Credible threat (>100 mines, no hostilities) triggers elevated premium."""
        response = model.compute_insurance_response(
            mine_density=200, active_hostilities=False
        )
        assert response["war_risk_premium_pct"] > model.war_risk_baseline_pct
        assert response["war_risk_premium_pct"] < model.war_risk_peak_pct

    def test_insurance_withdrawal_threshold(self, model: HormuzModel) -> None:
        """Insurance withdrawal triggers at high premiums."""
        response = model.compute_insurance_response(
            mine_density=500, active_hostilities=True
        )
        assert response["insurance_withdrawal"] is True

    def test_no_withdrawal_at_baseline(self, model: HormuzModel) -> None:
        """No insurance withdrawal at baseline conditions."""
        response = model.compute_insurance_response(
            mine_density=0, active_hostilities=False
        )
        assert response["insurance_withdrawal"] is False

    def test_freight_surge_at_peak(self, model: HormuzModel) -> None:
        """Freight rate surge matches configured value at peak."""
        response = model.compute_insurance_response(
            mine_density=1000, active_hostilities=True
        )
        # At peak premium, freight should be at max surge.
        assert response["freight_rate_surge_pct"] >= model.freight_surge_pct * 0.9

    def test_rerouting_days(self, model: HormuzModel) -> None:
        """Rerouting adds the expected extra transit days."""
        response_active = model.compute_insurance_response(500, active_hostilities=True)
        response_passive = model.compute_insurance_response(200, active_hostilities=False)

        assert response_active["rerouting_extra_days"] == model.cape_extra_days[1]
        assert response_passive["rerouting_extra_days"] == model.cape_extra_days[0]

    def test_premium_per_vessel_calculation(self, model: HormuzModel) -> None:
        """Premium per vessel is correctly derived from percentage."""
        response = model.compute_insurance_response(500, active_hostilities=True)
        expected = model.war_risk_peak_pct / 100 * 100_000_000
        assert response["premium_per_100m_vessel_usd"] == pytest.approx(expected, rel=0.01)


# ---------------------------------------------------------------------------
# Tests: Full disruption timeline simulation
# ---------------------------------------------------------------------------

class TestDisruptionTimeline:
    """Tests for the multi-day disruption simulation."""

    def test_timeline_length(self, model: HormuzModel) -> None:
        """Simulation produces the requested number of days."""
        result = model.simulate_disruption_timeline(days=31)
        assert len(result["mine_count"]) == 31
        assert len(result["daily_flow_mbpd"]) == 31
        assert len(result["insurance_premium_pct"]) == 31

    def test_mine_count_peaks_after_laying(self, model: HormuzModel) -> None:
        """Mine count stabilizes after laying period ends."""
        result = model.simulate_disruption_timeline(days=31, laying_days=5)
        mines = result["mine_count"]

        # After laying period, mines should be constant.
        assert mines[5] == mines[10]
        assert mines[10] == mines[20]

    def test_flow_decreases_during_disruption(self, model: HormuzModel) -> None:
        """Daily oil flow decreases during the disruption period."""
        result = model.simulate_disruption_timeline(
            days=10, laying_days=3, closure_fraction=0.8
        )
        flow = result["daily_flow_mbpd"]

        # Flow on day 0 should be less than initial (partial closure).
        assert flow[3] < model.daily_flow_mbpd

    def test_insurance_spikes_during_hostilities(
        self, model: HormuzModel
    ) -> None:
        """Insurance premium spikes during active hostilities."""
        result = model.simulate_disruption_timeline(
            days=20, laying_days=3, closure_fraction=0.5
        )
        premium = result["insurance_premium_pct"]

        # Premium during hostilities (first ~10 days) should be high.
        assert np.max(premium[:10]) > model.war_risk_baseline_pct * 5

    def test_reproducibility(self, model: HormuzModel) -> None:
        """Same seed produces identical timelines."""
        r1 = model.simulate_disruption_timeline(days=20, seed=42)
        r2 = model.simulate_disruption_timeline(days=20, seed=42)

        np.testing.assert_array_equal(r1["mine_count"], r2["mine_count"])
        np.testing.assert_array_equal(r1["daily_flow_mbpd"], r2["daily_flow_mbpd"])


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

class TestPropertyBased:
    """Property-based tests for Hormuz model invariants."""

    @given(
        closure_fraction=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=30, deadline=5000)
    def test_net_loss_never_exceeds_total_flow(
        self, closure_fraction: float, model: HormuzModel
    ) -> None:
        """Net loss can never exceed the total Hormuz flow."""
        result = model.compute_supply_disruption(closure_fraction)
        assert result["net_lost_mbpd"] <= model.daily_flow_mbpd + 1e-10
        assert result["net_lost_mbpd"] >= -1e-10

    @given(
        closure_fraction=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=30, deadline=5000)
    def test_bypass_never_exceeds_disrupted(
        self, closure_fraction: float, model: HormuzModel
    ) -> None:
        """Bypassed volume cannot exceed disrupted volume."""
        result = model.compute_supply_disruption(closure_fraction)
        assert result["bypassed_mbpd"] <= result["disrupted_mbpd"] + 1e-10

    @given(
        days=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=20, deadline=5000)
    def test_mine_density_never_negative(
        self, days: int, model: HormuzModel
    ) -> None:
        """Mine density is always non-negative."""
        density = model.compute_mine_density(days)
        assert density >= 0

    @given(
        mine_density=st.integers(min_value=0, max_value=6000),
        active=st.booleans(),
    )
    @settings(max_examples=20, deadline=5000)
    def test_insurance_premium_non_negative(
        self, mine_density: int, active: bool, model: HormuzModel
    ) -> None:
        """Insurance premium is always non-negative."""
        response = model.compute_insurance_response(mine_density, active)
        assert response["war_risk_premium_pct"] >= 0
