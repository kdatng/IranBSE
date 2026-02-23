"""Monte Carlo scenario engine for IranBSE commodity-futures modeling.

Orchestrates the three geopolitical sub-models (conflict escalation,
supply disruption, cross-market contagion) into a unified Monte Carlo
simulation framework.  Loads scenario parameters from YAML configuration,
generates N simulation paths per escalation level, and produces aggregate
statistics for downstream risk management and trading-signal generation.

Typical usage::

    engine = ScenarioEngine()
    engine.load_config("config/scenarios.yaml")
    results = engine.generate_scenarios(n_paths=10_000)
    stats = engine.summary_statistics(results)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from loguru import logger

from src.models.base_model import ModelConfig
from src.models.geopolitical.conflict_model import (
    ConflictEscalationModel,
    EscalationLevel,
)
from src.models.geopolitical.contagion import ContagionModel
from src.models.geopolitical.supply_disruption import SupplyDisruptionModel


@dataclass
class ScenarioPath:
    """A single Monte Carlo scenario path.

    Attributes:
        path_id: Unique integer identifier for this path.
        initial_level: Starting escalation level.
        escalation_path: Day-by-day escalation levels.
        oil_impact_pct: Day-by-day oil price impact (%).
        wheat_impact_pct: Day-by-day wheat price impact (%).
        supply_gap_mbpd: Day-by-day supply gap (mb/d).
        peak_escalation: Maximum escalation level reached.
        peak_oil_impact_pct: Maximum oil price impact (%).
        peak_wheat_impact_pct: Maximum wheat price impact (%).
        peak_supply_gap_mbpd: Maximum supply gap (mb/d).
        hormuz_closed: Whether Hormuz was effectively closed.
    """

    path_id: int
    initial_level: int
    escalation_path: list[int]
    oil_impact_pct: list[float]
    wheat_impact_pct: list[float]
    supply_gap_mbpd: list[float]
    peak_escalation: int
    peak_oil_impact_pct: float
    peak_wheat_impact_pct: float
    peak_supply_gap_mbpd: float
    hormuz_closed: bool


@dataclass
class ScenarioResults:
    """Aggregate results from a full Monte Carlo scenario run.

    Attributes:
        paths: List of all simulated paths.
        n_paths: Total number of paths.
        horizon_days: Simulation horizon.
        config_name: Name of the loaded scenario configuration.
        escalation_distribution: Distribution of peak escalation levels.
        metadata: Additional metadata.
    """

    paths: list[ScenarioPath]
    n_paths: int
    horizon_days: int
    config_name: str
    escalation_distribution: dict[int, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ScenarioEngine:
    """Monte Carlo scenario generation engine.

    Combines conflict escalation, supply disruption, and cross-market
    contagion models into a coherent simulation framework.  Configuration
    is loaded from a YAML file matching the project's ``config/scenarios.yaml``
    schema.

    Args:
        config_path: Optional path to scenario YAML; can also be set via
            :meth:`load_config`.
        seed: Global RNG seed for reproducibility.

    Example::

        engine = ScenarioEngine(seed=42)
        engine.load_config("config/scenarios.yaml")
        results = engine.generate_scenarios(n_paths=10_000)
        stats = engine.summary_statistics(results)
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        seed: int = 42,  # JUSTIFIED: model_config.yaml pipeline.seed
    ) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._config: dict[str, Any] = {}
        self._escalation_levels: list[dict[str, Any]] = []
        self._hormuz_params: dict[str, Any] = {}
        self._military_params: dict[str, Any] = {}
        self._proxy_params: dict[str, Any] = {}
        self._insurance_params: dict[str, Any] = {}

        # Sub-models (lazily initialised)
        self._conflict_model: ConflictEscalationModel | None = None
        self._supply_model: SupplyDisruptionModel | None = None
        self._contagion_model: ContagionModel | None = None

        if config_path is not None:
            self.load_config(config_path)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def load_config(self, config_path: str | Path) -> None:
        """Load and validate scenario configuration from a YAML file.

        Args:
            config_path: Path to the scenario YAML file.

        Raises:
            FileNotFoundError: If the config file does not exist.
            KeyError: If required configuration sections are missing.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Scenario config not found: {path}")

        with open(path) as fh:
            raw = yaml.safe_load(fh)

        self._config = raw.get("scenario", raw)
        self._escalation_levels = self._config.get("escalation_levels", [])
        self._hormuz_params = self._config.get("strait_of_hormuz", {})
        self._military_params = self._config.get("iran_military", {})
        self._proxy_params = self._config.get("proxy_capability", {})
        self._insurance_params = self._config.get("insurance_shipping", {})

        if not self._escalation_levels:
            raise KeyError(
                "Scenario config must contain 'escalation_levels' section."
            )

        logger.info(
            "Loaded scenario config '{}' with {} escalation levels",
            self._config.get("name", "unnamed"),
            len(self._escalation_levels),
        )

        # Initialise sub-models
        self._init_sub_models()

    def _init_sub_models(self) -> None:
        """Initialise the three geopolitical sub-models from config."""
        # Conflict model
        self._conflict_model = ConflictEscalationModel(
            ModelConfig(
                name="conflict_escalation",
                params={
                    "initial_level": 1,
                    "hawkes_baseline": 0.1,
                    "hawkes_alpha": 0.5,
                    "hawkes_beta": 1.0,
                },
            )
        )

        # Supply disruption model
        supply_params: dict[str, Any] = {}
        if self._hormuz_params:
            supply_params["daily_flow_mbpd"] = self._hormuz_params.get(
                "daily_flow_mbpd", 20.0
            )
            supply_params["bypass_capacity_mbpd"] = self._hormuz_params.get(
                "bypass_pipeline_capacity_mbpd", 4.2
            )
        if self._military_params:
            supply_params["remaining_missiles"] = self._military_params.get(
                "remaining_missiles", 1500
            )
            supply_params["remaining_launchers"] = self._military_params.get(
                "remaining_launchers", 200
            )
            mine_inv = self._military_params.get("mine_inventory", [5000, 6000])
            if isinstance(mine_inv, list):
                supply_params["mine_inventory"] = (mine_inv[0] + mine_inv[1]) // 2
            else:
                supply_params["mine_inventory"] = mine_inv

        self._supply_model = SupplyDisruptionModel(
            ModelConfig(name="supply_disruption", params=supply_params)
        )

        # Contagion model
        proxy_fronts: list[str] = []
        if self._proxy_params.get("houthi"):
            proxy_fronts.append("red_sea")
        if self._proxy_params.get("hezbollah"):
            proxy_fronts.append("hezbollah")

        self._contagion_model = ContagionModel(
            ModelConfig(
                name="contagion",
                params={
                    "proxy_fronts": proxy_fronts,
                    "enable_proxy_amplification": True,
                },
            )
        )

    # ------------------------------------------------------------------
    # Scenario generation
    # ------------------------------------------------------------------

    def generate_scenarios(
        self,
        n_paths: int = 10_000,
        horizon_days: int | None = None,
    ) -> ScenarioResults:
        """Generate Monte Carlo scenario paths.

        For each path:
            1. Sample an initial escalation level from config probabilities.
            2. Simulate day-by-day escalation using the conflict model.
            3. For each day, compute supply disruption based on level.
            4. Propagate oil shock to wheat via contagion model.

        Args:
            n_paths: Number of Monte Carlo paths to generate.
            horizon_days: Override for simulation horizon; defaults to
                config ``duration_days``.

        Returns:
            ScenarioResults with all paths and aggregate metadata.

        Raises:
            RuntimeError: If sub-models are not fitted.
        """
        if not self._escalation_levels:
            raise RuntimeError(
                "No scenario config loaded. Call load_config() first."
            )

        horizon = horizon_days or self._config.get(
            "duration_days", 31  # JUSTIFIED: scenarios.yaml duration_days default
        )

        # Ensure sub-models are fitted (use synthetic calibration data)
        self._ensure_models_fitted(horizon)

        # Extract level probabilities
        level_probs = np.array(
            [lvl["probability"] for lvl in self._escalation_levels],
            dtype=np.float64,
        )
        level_probs /= level_probs.sum()  # Normalise

        paths: list[ScenarioPath] = []
        rng = np.random.default_rng(self._seed)

        for i in range(n_paths):
            # 1. Sample initial escalation level
            init_idx = rng.choice(len(self._escalation_levels), p=level_probs)
            init_level = self._escalation_levels[init_idx]["level"]
            level_config = self._escalation_levels[init_idx]

            # 2. Simulate escalation path
            assert self._conflict_model is not None
            self._conflict_model._initial_level = EscalationLevel(init_level)
            esc_paths = self._conflict_model.simulate_escalation_path(
                n_paths=1,
                horizon_days=horizon,
                seed=self._seed + i,
            )
            esc_path = esc_paths[0].tolist()

            # 3. Day-by-day supply disruption and contagion
            oil_impacts: list[float] = []
            wheat_impacts: list[float] = []
            supply_gaps: list[float] = []
            hormuz_closed = False

            for day_idx, level_val in enumerate(esc_path):
                level_cfg = self._get_level_config(level_val)

                # Supply gap based on Hormuz closure probability
                closure_prob = level_cfg.get("hormuz_closure_probability", 0.0)
                if rng.random() < closure_prob:
                    hormuz_closed = True
                    # Draw disruption from level range
                    disrupt_range = level_cfg.get(
                        "oil_supply_disruption_pct", [5, 15]
                    )
                    disruption_pct = rng.uniform(
                        disrupt_range[0], disrupt_range[1]
                    ) / 100.0
                    gap = disruption_pct * self._hormuz_params.get(
                        "daily_flow_mbpd", 20.0
                    )
                else:
                    disruption_pct = 0.0
                    gap = 0.0

                supply_gaps.append(gap)

                # Oil price impact from level config
                oil_range = level_cfg.get("oil_price_range_bbl", [70, 80])
                base_price = 70.0  # JUSTIFIED: approximate pre-conflict Brent baseline (Feb 2026)
                oil_price = rng.uniform(oil_range[0], oil_range[1])
                oil_impact = (oil_price - base_price) / base_price * 100.0
                oil_impacts.append(oil_impact)

                # Wheat impact via contagion
                assert self._contagion_model is not None
                wheat_range = level_cfg.get("wheat_trade_disruption_pct", [0, 5])
                wheat_impact = rng.uniform(wheat_range[0], wheat_range[1])
                # Add contagion channel (oil -> wheat)
                if oil_impact > 0:
                    contagion_result = self._contagion_model.cross_market_impact(
                        oil_shock_pct=oil_impact, regime="crisis"
                    )
                    wheat_impact += contagion_result.get("wheat", 0.0)
                wheat_impacts.append(wheat_impact)

            path = ScenarioPath(
                path_id=i,
                initial_level=init_level,
                escalation_path=esc_path,
                oil_impact_pct=oil_impacts,
                wheat_impact_pct=wheat_impacts,
                supply_gap_mbpd=supply_gaps,
                peak_escalation=max(esc_path),
                peak_oil_impact_pct=max(oil_impacts) if oil_impacts else 0.0,
                peak_wheat_impact_pct=max(wheat_impacts) if wheat_impacts else 0.0,
                peak_supply_gap_mbpd=max(supply_gaps) if supply_gaps else 0.0,
                hormuz_closed=hormuz_closed,
            )
            paths.append(path)

        # Escalation distribution
        esc_dist: dict[int, int] = {}
        for p in paths:
            esc_dist[p.peak_escalation] = esc_dist.get(p.peak_escalation, 0) + 1

        results = ScenarioResults(
            paths=paths,
            n_paths=n_paths,
            horizon_days=horizon,
            config_name=self._config.get("name", "unnamed"),
            escalation_distribution=esc_dist,
            metadata={
                "seed": self._seed,
                "escalation_level_probs": level_probs.tolist(),
            },
        )

        logger.info(
            "Generated {} scenario paths over {} days; escalation dist: {}",
            n_paths,
            horizon,
            esc_dist,
        )
        return results

    def aggregate_results(
        self,
        results: ScenarioResults,
    ) -> pl.DataFrame:
        """Aggregate scenario paths into a summary DataFrame.

        Args:
            results: Output of :meth:`generate_scenarios`.

        Returns:
            Polars DataFrame with one row per path and columns for
            key metrics (peak escalation, peak oil/wheat impact, etc.).
        """
        rows: list[dict[str, Any]] = []
        for p in results.paths:
            rows.append(
                {
                    "path_id": p.path_id,
                    "initial_level": p.initial_level,
                    "peak_escalation": p.peak_escalation,
                    "peak_oil_impact_pct": p.peak_oil_impact_pct,
                    "peak_wheat_impact_pct": p.peak_wheat_impact_pct,
                    "peak_supply_gap_mbpd": p.peak_supply_gap_mbpd,
                    "hormuz_closed": p.hormuz_closed,
                    "mean_oil_impact_pct": float(np.mean(p.oil_impact_pct)),
                    "mean_wheat_impact_pct": float(np.mean(p.wheat_impact_pct)),
                }
            )
        return pl.DataFrame(rows)

    def summary_statistics(
        self,
        results: ScenarioResults,
    ) -> dict[str, Any]:
        """Compute summary statistics across all paths.

        Args:
            results: Output of :meth:`generate_scenarios`.

        Returns:
            Dictionary with:
                - ``mean_*``, ``median_*``, ``std_*`` for key metrics
                - ``percentiles`` at 5th, 25th, 50th, 75th, 95th
                - ``hormuz_closure_probability``
                - ``escalation_distribution``
        """
        df = self.aggregate_results(results)

        oil_peaks = df.get_column("peak_oil_impact_pct").to_numpy()
        wheat_peaks = df.get_column("peak_wheat_impact_pct").to_numpy()
        gap_peaks = df.get_column("peak_supply_gap_mbpd").to_numpy()

        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]

        stats: dict[str, Any] = {
            "n_paths": results.n_paths,
            "horizon_days": results.horizon_days,
            "oil_impact_pct": {
                "mean": float(oil_peaks.mean()),
                "median": float(np.median(oil_peaks)),
                "std": float(oil_peaks.std()),
                "percentiles": {
                    f"p{int(q * 100)}": float(np.quantile(oil_peaks, q))
                    for q in quantiles
                },
            },
            "wheat_impact_pct": {
                "mean": float(wheat_peaks.mean()),
                "median": float(np.median(wheat_peaks)),
                "std": float(wheat_peaks.std()),
                "percentiles": {
                    f"p{int(q * 100)}": float(np.quantile(wheat_peaks, q))
                    for q in quantiles
                },
            },
            "supply_gap_mbpd": {
                "mean": float(gap_peaks.mean()),
                "median": float(np.median(gap_peaks)),
                "std": float(gap_peaks.std()),
                "percentiles": {
                    f"p{int(q * 100)}": float(np.quantile(gap_peaks, q))
                    for q in quantiles
                },
            },
            "hormuz_closure_probability": float(
                df.get_column("hormuz_closed").cast(pl.Float64).mean()
            ),
            "escalation_distribution": results.escalation_distribution,
        }

        logger.info(
            "Summary stats: oil mean={:.1f}%, wheat mean={:.1f}%, "
            "Hormuz closure P={:.2f}",
            stats["oil_impact_pct"]["mean"],
            stats["wheat_impact_pct"]["mean"],
            stats["hormuz_closure_probability"],
        )
        return stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_level_config(self, level: int) -> dict[str, Any]:
        """Retrieve config dict for a given escalation level.

        Args:
            level: Escalation level (1-4).

        Returns:
            Configuration dictionary for that level.
        """
        for lvl_cfg in self._escalation_levels:
            if lvl_cfg["level"] == level:
                return lvl_cfg
        # Fallback to highest defined level
        return self._escalation_levels[-1]

    def _ensure_models_fitted(self, horizon: int) -> None:
        """Fit sub-models with synthetic calibration data if not fitted.

        This allows the engine to run even without historical data
        (pure scenario-based simulation).

        Args:
            horizon: Simulation horizon for sizing calibration data.
        """
        n_rows = max(50, horizon * 2)

        # Conflict model
        assert self._conflict_model is not None
        if self._conflict_model.state.value != "fitted":
            rng = np.random.default_rng(self._seed)
            cal_data = pl.DataFrame(
                {
                    "date": pl.date_range(
                        pl.date(2024, 1, 1),
                        pl.date(2024, 1, 1) + pl.duration(days=n_rows - 1),  # type: ignore[operator]
                        eager=True,
                    ).head(n_rows),
                    "escalation_level": rng.choice(
                        [1, 1, 1, 2, 2, 3], size=n_rows
                    ).tolist(),
                }
            )
            self._conflict_model.fit(cal_data)

        # Supply disruption model
        assert self._supply_model is not None
        if self._supply_model.state.value != "fitted":
            cal_supply = pl.DataFrame(
                {
                    "date": pl.date_range(
                        pl.date(2024, 1, 1),
                        pl.date(2024, 1, 1) + pl.duration(days=n_rows - 1),  # type: ignore[operator]
                        eager=True,
                    ).head(n_rows),
                    "supply_change_mbpd": np.random.default_rng(self._seed)
                    .uniform(-5, 0, size=n_rows)
                    .tolist(),
                    "oil_price_change_pct": np.random.default_rng(self._seed + 1)
                    .uniform(0, 50, size=n_rows)
                    .tolist(),
                }
            )
            self._supply_model.fit(cal_supply)

        # Contagion model
        assert self._contagion_model is not None
        if self._contagion_model.state.value != "fitted":
            rng_c = np.random.default_rng(self._seed + 2)
            oil_ret = rng_c.normal(0, 0.02, size=n_rows)
            cal_contagion = pl.DataFrame(
                {
                    "date": pl.date_range(
                        pl.date(2024, 1, 1),
                        pl.date(2024, 1, 1) + pl.duration(days=n_rows - 1),  # type: ignore[operator]
                        eager=True,
                    ).head(n_rows),
                    "oil_return": oil_ret.tolist(),
                    "wheat_return": (oil_ret * 0.2 + rng_c.normal(0, 0.01, size=n_rows)).tolist(),
                    "gold_return": (oil_ret * 0.3 + rng_c.normal(0, 0.008, size=n_rows)).tolist(),
                    "natgas_return": (oil_ret * 0.8 + rng_c.normal(0, 0.03, size=n_rows)).tolist(),
                }
            )
            self._contagion_model.fit(cal_contagion)
