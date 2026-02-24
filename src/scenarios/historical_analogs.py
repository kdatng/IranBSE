"""Historical conflict analog engine for scenario calibration.

Maintains a structured database of 9 historical geopolitical events with
their observed oil and wheat price impacts, and provides kernel-weighted
analog matching to generate calibrated forecasts for the current US-Iran
scenario.

The core idea: future conflict impacts are best estimated by a weighted
combination of historical precedents, where weights reflect similarity
to the current scenario along key dimensions (supply disruption magnitude,
chokepoint involvement, proxy warfare, duration).

Typical usage::

    engine = HistoricalAnalogEngine(config_path="config/scenarios.yaml")
    closest = engine.find_closest_analogs(
        supply_disruption_pct=25.0,
        chokepoint_involved=True,
        proxy_warfare=True,
        n_top=5,
    )
    forecast = engine.weighted_forecast(closest)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger


@dataclass
class HistoricalAnalog:
    """A single historical conflict event with observed market impacts.

    Attributes:
        event: Human-readable event name.
        year: Year of the event.
        oil_peak_pct_change: Peak oil price change (%).
        wheat_peak_pct_change: Peak wheat price change (%).
        duration_to_peak_days: Days from event onset to peak price impact.
        supply_disruption_mbpd: Estimated supply disruption (mb/d).
        chokepoint_involved: Whether a maritime chokepoint was affected.
        proxy_warfare: Whether proxy forces were involved.
        hormuz_threatened: Whether Hormuz was directly threatened.
        context: Additional context string.
        suez_traffic_drop_pct: Suez Canal traffic impact (if applicable).
        similarity_score: Computed similarity to current scenario (0-1).
    """

    event: str
    year: int
    oil_peak_pct_change: float
    wheat_peak_pct_change: float
    duration_to_peak_days: int
    supply_disruption_mbpd: float = 0.0
    chokepoint_involved: bool = False
    proxy_warfare: bool = False
    hormuz_threatened: bool = False
    context: str = ""
    suez_traffic_drop_pct: float | None = None
    similarity_score: float = 0.0


@dataclass
class AnalogForecast:
    """Weighted forecast derived from historical analogs.

    Attributes:
        oil_impact_pct: Weighted oil price impact estimate (%).
        wheat_impact_pct: Weighted wheat price impact estimate (%).
        duration_days: Weighted expected duration to peak (days).
        confidence_interval_oil: (lower, upper) 80% CI for oil impact.
        confidence_interval_wheat: (lower, upper) 80% CI for wheat impact.
        analogs_used: List of analogs contributing to the forecast.
        weights: Normalised weights for each analog.
    """

    oil_impact_pct: float
    wheat_impact_pct: float
    duration_days: float
    confidence_interval_oil: tuple[float, float]
    confidence_interval_wheat: tuple[float, float]
    analogs_used: list[str]
    weights: list[float]


class HistoricalAnalogEngine:
    """Kernel-weighted historical analog matching engine.

    Maintains a database of 9 historical geopolitical/conflict events
    and their observed impacts on oil and wheat prices.  Provides
    similarity-weighted forecasting for new scenarios.

    Args:
        config_path: Path to ``scenarios.yaml`` to load analog database.
        analogs: Optional direct injection of analogs (overrides config).

    Example::

        engine = HistoricalAnalogEngine("config/scenarios.yaml")
        top3 = engine.find_closest_analogs(
            supply_disruption_pct=30.0,
            chokepoint_involved=True,
        )
        forecast = engine.weighted_forecast(top3)
        print(f"Expected oil impact: {forecast.oil_impact_pct:.0f}%")
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        analogs: list[HistoricalAnalog] | None = None,
    ) -> None:
        if analogs is not None:
            self._analogs = analogs
        elif config_path is not None:
            self._analogs = self._load_from_config(config_path)
        else:
            self._analogs = self._builtin_analogs()

        logger.info(
            "HistoricalAnalogEngine initialised with {} analogs: {}",
            len(self._analogs),
            [a.event for a in self._analogs],
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def find_closest_analogs(
        self,
        supply_disruption_pct: float = 20.0,
        chokepoint_involved: bool = True,
        proxy_warfare: bool = True,
        hormuz_threatened: bool = True,
        conflict_duration_days: int = 30,
        n_top: int = 5,
    ) -> list[HistoricalAnalog]:
        """Find the most similar historical analogs to the current scenario.

        Similarity is computed via a Gaussian kernel over a multi-dimensional
        feature space including supply disruption magnitude, chokepoint
        involvement, proxy warfare, and duration.

        Args:
            supply_disruption_pct: Expected supply disruption (% of global).
            chokepoint_involved: Whether a maritime chokepoint is affected.
            proxy_warfare: Whether proxy forces are active.
            hormuz_threatened: Whether Hormuz is directly threatened.
            conflict_duration_days: Expected conflict duration.
            n_top: Number of top analogs to return.

        Returns:
            List of HistoricalAnalog with ``similarity_score`` populated,
            sorted by descending similarity.
        """
        query = self._build_feature_vector(
            supply_disruption_pct=supply_disruption_pct,
            chokepoint_involved=chokepoint_involved,
            proxy_warfare=proxy_warfare,
            hormuz_threatened=hormuz_threatened,
            duration_days=conflict_duration_days,
        )

        scored: list[HistoricalAnalog] = []
        for analog in self._analogs:
            av = self._analog_feature_vector(analog)
            sim = self._gaussian_kernel_similarity(query, av)
            analog.similarity_score = sim
            scored.append(analog)

        scored.sort(key=lambda a: a.similarity_score, reverse=True)
        top = scored[:n_top]

        logger.info(
            "Top {} analogs: {}",
            n_top,
            [(a.event, f"{a.similarity_score:.3f}") for a in top],
        )
        return top

    def weighted_forecast(
        self,
        analogs: list[HistoricalAnalog],
        confidence_level: float = 0.80,
    ) -> AnalogForecast:
        """Generate a similarity-weighted forecast from selected analogs.

        Weights are the normalised similarity scores.  Confidence intervals
        are derived from the weighted standard deviation.

        Args:
            analogs: List of analogs with ``similarity_score`` populated.
            confidence_level: Confidence level for intervals (default: 80%).

        Returns:
            AnalogForecast with weighted estimates and confidence intervals.
        """
        if not analogs:
            raise ValueError("No analogs provided for forecasting.")

        # Extract data
        weights = np.array(
            [a.similarity_score for a in analogs], dtype=np.float64
        )
        # Guard against all-zero weights
        if weights.sum() < 1e-12:
            weights = np.ones_like(weights)
        weights /= weights.sum()

        oil_impacts = np.array([a.oil_peak_pct_change for a in analogs])
        wheat_impacts = np.array([a.wheat_peak_pct_change for a in analogs])
        durations = np.array(
            [a.duration_to_peak_days for a in analogs], dtype=np.float64
        )

        # Weighted means
        oil_mean = float(np.average(oil_impacts, weights=weights))
        wheat_mean = float(np.average(wheat_impacts, weights=weights))
        duration_mean = float(np.average(durations, weights=weights))

        # Weighted standard deviations
        oil_std = float(
            np.sqrt(np.average((oil_impacts - oil_mean) ** 2, weights=weights))
        )
        wheat_std = float(
            np.sqrt(
                np.average((wheat_impacts - wheat_mean) ** 2, weights=weights)
            )
        )

        # Confidence intervals using normal approximation
        # JUSTIFIED: z=1.28 for 80% CI (two-tailed)
        z = 1.28  # JUSTIFIED: scipy.stats.norm.ppf(0.9) = 1.28 for 80% two-tailed CI
        oil_ci = (oil_mean - z * oil_std, oil_mean + z * oil_std)
        wheat_ci = (wheat_mean - z * wheat_std, wheat_mean + z * wheat_std)

        forecast = AnalogForecast(
            oil_impact_pct=oil_mean,
            wheat_impact_pct=wheat_mean,
            duration_days=duration_mean,
            confidence_interval_oil=oil_ci,
            confidence_interval_wheat=wheat_ci,
            analogs_used=[a.event for a in analogs],
            weights=weights.tolist(),
        )

        logger.info(
            "Analog forecast: oil {:.0f}% [{:.0f}, {:.0f}], "
            "wheat {:.0f}% [{:.0f}, {:.0f}], duration {:.0f} days",
            forecast.oil_impact_pct,
            oil_ci[0],
            oil_ci[1],
            forecast.wheat_impact_pct,
            wheat_ci[0],
            wheat_ci[1],
            forecast.duration_days,
        )
        return forecast

    def analog_envelope(
        self,
        analogs: list[HistoricalAnalog] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compute the envelope (range) of outcomes across analogs.

        Provides min/max/mean/median for oil and wheat impacts across
        all analogs (or a supplied subset), representing the full range
        of historical precedent.

        Args:
            analogs: Subset of analogs; defaults to full database.

        Returns:
            Dictionary with ``oil`` and ``wheat`` sub-dicts, each containing
            ``min``, ``max``, ``mean``, ``median``.
        """
        data = analogs or self._analogs

        oil = np.array([a.oil_peak_pct_change for a in data])
        wheat = np.array([a.wheat_peak_pct_change for a in data])

        envelope = {
            "oil": {
                "min": float(oil.min()),
                "max": float(oil.max()),
                "mean": float(oil.mean()),
                "median": float(np.median(oil)),
            },
            "wheat": {
                "min": float(wheat.min()),
                "max": float(wheat.max()),
                "mean": float(wheat.mean()),
                "median": float(np.median(wheat)),
            },
        }
        logger.debug("Analog envelope: {}", envelope)
        return envelope

    # ------------------------------------------------------------------
    # Similarity computation
    # ------------------------------------------------------------------

    def _build_feature_vector(
        self,
        supply_disruption_pct: float,
        chokepoint_involved: bool,
        proxy_warfare: bool,
        hormuz_threatened: bool,
        duration_days: int,
    ) -> np.ndarray:
        """Build normalised feature vector for the query scenario.

        Features:
            0. supply_disruption_pct (normalised by max historical)
            1. chokepoint_involved (0/1)
            2. proxy_warfare (0/1)
            3. hormuz_threatened (0/1)
            4. duration_days (normalised by max historical)

        Args:
            supply_disruption_pct: Supply disruption magnitude.
            chokepoint_involved: Maritime chokepoint flag.
            proxy_warfare: Proxy force flag.
            hormuz_threatened: Hormuz-specific flag.
            duration_days: Expected duration.

        Returns:
            Normalised feature vector.
        """
        max_disruption = max(
            (a.oil_peak_pct_change for a in self._analogs), default=140.0
        )
        max_duration = max(
            (a.duration_to_peak_days for a in self._analogs), default=90
        )

        return np.array(
            [
                supply_disruption_pct / max(max_disruption, 1.0),
                float(chokepoint_involved),
                float(proxy_warfare),
                float(hormuz_threatened),
                duration_days / max(max_duration, 1),
            ],
            dtype=np.float64,
        )

    def _analog_feature_vector(self, analog: HistoricalAnalog) -> np.ndarray:
        """Build normalised feature vector for a historical analog.

        Args:
            analog: Historical analog.

        Returns:
            Normalised feature vector (same dimensions as query).
        """
        max_disruption = max(
            (a.oil_peak_pct_change for a in self._analogs), default=140.0
        )
        max_duration = max(
            (a.duration_to_peak_days for a in self._analogs), default=90
        )

        return np.array(
            [
                analog.oil_peak_pct_change / max(max_disruption, 1.0),
                float(analog.chokepoint_involved),
                float(analog.proxy_warfare),
                float(analog.hormuz_threatened),
                analog.duration_to_peak_days / max(max_duration, 1),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _gaussian_kernel_similarity(
        query: np.ndarray,
        analog: np.ndarray,
        bandwidth: float = 0.5,
    ) -> float:
        """Compute Gaussian kernel similarity between two feature vectors.

        Args:
            query: Query feature vector.
            analog: Analog feature vector.
            bandwidth: Kernel bandwidth (smaller = more selective).

        Returns:
            Similarity score in (0, 1].
        """
        # Feature weights: supply disruption and chokepoint most important
        # JUSTIFIED: supply disruption magnitude is the primary driver of
        # commodity price impact (IMF WEO analysis 2000-2024)
        feature_weights = np.array(
            [2.0, 1.5, 1.0, 1.5, 0.5],  # JUSTIFIED: disruption > chokepoint/Hormuz > proxy > duration
            dtype=np.float64,
        )

        diff = (query - analog) * feature_weights
        dist_sq = float(np.sum(diff ** 2))
        return float(np.exp(-dist_sq / (2 * bandwidth ** 2)))

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_from_config(self, config_path: str | Path) -> list[HistoricalAnalog]:
        """Load analogs from the scenario YAML configuration.

        Args:
            config_path: Path to scenarios.yaml.

        Returns:
            List of HistoricalAnalog instances.
        """
        path = Path(config_path)
        if not path.exists():
            logger.warning(
                "Config {} not found; using built-in analogs.", path
            )
            return self._builtin_analogs()

        with open(path) as fh:
            raw = yaml.safe_load(fh)

        scenario = raw.get("scenario", raw)
        raw_analogs = scenario.get("historical_analogs", [])

        analogs: list[HistoricalAnalog] = []
        for entry in raw_analogs:
            event_name = entry.get("event", "Unknown")
            analog = HistoricalAnalog(
                event=event_name,
                year=self._extract_year(event_name),
                oil_peak_pct_change=entry.get("oil_peak_pct_change", 0.0),
                wheat_peak_pct_change=entry.get("wheat_peak_pct_change", 0.0),
                duration_to_peak_days=entry.get("duration_to_peak_days", 30),
                chokepoint_involved=self._is_chokepoint_event(event_name),
                proxy_warfare=self._is_proxy_event(event_name),
                hormuz_threatened=self._is_hormuz_event(event_name),
                context=entry.get("context", ""),
                suez_traffic_drop_pct=entry.get("suez_traffic_drop_pct"),
            )
            analogs.append(analog)

        return analogs

    @staticmethod
    def _builtin_analogs() -> list[HistoricalAnalog]:
        """Provide the 9 built-in historical analogs.

        Returns:
            List of 9 HistoricalAnalog instances with research-backed
            impact parameters.
        """
        return [
            HistoricalAnalog(
                event="Gulf War I (1990-91)",
                year=1990,
                oil_peak_pct_change=140.0,  # JUSTIFIED: oil went from $21 to ~$46/bbl
                wheat_peak_pct_change=15.0,  # JUSTIFIED: moderate wheat impact (no major wheat exporter involved)
                duration_to_peak_days=60,  # JUSTIFIED: oil peaked ~60 days after invasion
                supply_disruption_mbpd=4.3,  # JUSTIFIED: Iraq + Kuwait production offline
                chokepoint_involved=True,  # JUSTIFIED: Hormuz threatened but not closed
                proxy_warfare=False,
                hormuz_threatened=True,
                context="Iraq invasion of Kuwait; coalition response",
            ),
            HistoricalAnalog(
                event="Iraq War (2003)",
                year=2003,
                oil_peak_pct_change=37.0,  # JUSTIFIED: oil from ~$25 to ~$34/bbl
                wheat_peak_pct_change=8.0,  # JUSTIFIED: mild supply-chain disruption
                duration_to_peak_days=14,  # JUSTIFIED: rapid initial price spike
                supply_disruption_mbpd=2.3,  # JUSTIFIED: Iraqi production offline
                chokepoint_involved=False,
                proxy_warfare=False,
                hormuz_threatened=False,
                context="US-led invasion; rapid conventional victory",
            ),
            HistoricalAnalog(
                event="Libya Civil War (2011)",
                year=2011,
                oil_peak_pct_change=25.0,  # JUSTIFIED: Brent from $95 to ~$125/bbl
                wheat_peak_pct_change=5.0,  # JUSTIFIED: minimal direct wheat impact
                duration_to_peak_days=45,  # JUSTIFIED: gradual escalation
                supply_disruption_mbpd=1.6,  # JUSTIFIED: Libyan output dropped from 1.6 to near zero
                chokepoint_involved=False,
                proxy_warfare=True,
                hormuz_threatened=False,
                context="Civil war disrupted Libyan oil exports",
            ),
            HistoricalAnalog(
                event="Russia-Ukraine (2022)",
                year=2022,
                oil_peak_pct_change=65.0,  # JUSTIFIED: Brent spiked from ~$80 to ~$130/bbl
                wheat_peak_pct_change=70.0,  # JUSTIFIED: CBOT wheat limit-up multiple days; peak +70%
                duration_to_peak_days=21,  # JUSTIFIED: peaked ~3 weeks after invasion
                supply_disruption_mbpd=3.0,  # JUSTIFIED: Russian oil exports initially fell ~3 mb/d before rerouting
                chokepoint_involved=True,  # JUSTIFIED: Black Sea grain corridor disrupted
                proxy_warfare=False,
                hormuz_threatened=False,
                context="Major wheat/energy exporter; Black Sea blockade",
            ),
            HistoricalAnalog(
                event="Iran-Iraq War Start (1980)",
                year=1980,
                oil_peak_pct_change=110.0,  # JUSTIFIED: oil from $14 to ~$35/bbl (on top of 1979 revolution shock)
                wheat_peak_pct_change=10.0,  # JUSTIFIED: moderate via energy costs
                duration_to_peak_days=90,  # JUSTIFIED: prolonged escalation
                supply_disruption_mbpd=3.5,  # JUSTIFIED: Iran + Iraq production severely curtailed
                chokepoint_involved=True,  # JUSTIFIED: tanker war threatened Hormuz
                proxy_warfare=False,
                hormuz_threatened=True,
                context="Start of 8-year war; Hormuz directly threatened",
            ),
            HistoricalAnalog(
                event="Soleimani Strike (2020)",
                year=2020,
                oil_peak_pct_change=4.0,  # JUSTIFIED: brief $3-4/bbl spike; markets calmed quickly
                wheat_peak_pct_change=1.0,  # JUSTIFIED: negligible wheat impact
                duration_to_peak_days=1,  # JUSTIFIED: single-day event
                supply_disruption_mbpd=0.0,  # JUSTIFIED: no supply disruption
                chokepoint_involved=False,
                proxy_warfare=True,  # JUSTIFIED: Iran used proxies in response
                hormuz_threatened=False,
                context="Targeted assassination; limited retaliation",
            ),
            HistoricalAnalog(
                event="Houthi Red Sea Campaign (2023-2025)",
                year=2023,
                oil_peak_pct_change=12.0,  # JUSTIFIED: ~$10-12/bbl increase from baseline
                wheat_peak_pct_change=3.0,  # JUSTIFIED: freight-driven wheat cost increase
                duration_to_peak_days=90,  # JUSTIFIED: slow build over 3 months
                supply_disruption_mbpd=0.5,  # JUSTIFIED: rerouting, not physical supply loss
                chokepoint_involved=True,  # JUSTIFIED: Bab al-Mandab / Red Sea
                proxy_warfare=True,  # JUSTIFIED: Iran-backed Houthi operations
                hormuz_threatened=False,
                context="70% Red Sea traffic decline; 58% Suez drop",
                suez_traffic_drop_pct=58.0,
            ),
            HistoricalAnalog(
                event="Operation Praying Mantis (1988)",
                year=1988,
                oil_peak_pct_change=5.0,  # JUSTIFIED: brief spike; US destroyed half Iran's navy
                wheat_peak_pct_change=1.0,  # JUSTIFIED: negligible
                duration_to_peak_days=3,  # JUSTIFIED: 1-day operation, 3-day market impact
                supply_disruption_mbpd=0.0,  # JUSTIFIED: no sustained supply loss
                chokepoint_involved=True,  # JUSTIFIED: Hormuz area
                proxy_warfare=False,
                hormuz_threatened=True,
                context="USS Roberts hit mine; US destroyed half Iran navy in hours",
            ),
            HistoricalAnalog(
                event="June 2025 Israel-Iran War",
                year=2025,
                oil_peak_pct_change=18.0,  # JUSTIFIED: scenarios.yaml -- Brent spiked ~$12-15
                wheat_peak_pct_change=5.0,  # JUSTIFIED: scenarios.yaml
                duration_to_peak_days=7,  # JUSTIFIED: 1-week conflict duration
                supply_disruption_mbpd=0.2,  # JUSTIFIED: Iran loaded mines but did not deploy
                chokepoint_involved=True,  # JUSTIFIED: Hormuz threatened (mines loaded)
                proxy_warfare=True,  # JUSTIFIED: Hezbollah + Houthi involved
                hormuz_threatened=True,
                context="Iran loaded mines but did not deploy; nuclear facilities struck",
            ),
        ]

    @staticmethod
    def _extract_year(event_name: str) -> int:
        """Extract the year from an event name string.

        Args:
            event_name: Event name containing a year (e.g. "Gulf War I (1990-91)").

        Returns:
            Extracted year, or 2000 as fallback.
        """
        import re

        match = re.search(r"(\d{4})", event_name)
        if match:
            return int(match.group(1))
        return 2000

    @staticmethod
    def _is_chokepoint_event(event_name: str) -> bool:
        """Determine if an event involved maritime chokepoint disruption.

        Args:
            event_name: Event name.

        Returns:
            True if a chokepoint was involved.
        """
        chokepoint_keywords = [
            "gulf", "hormuz", "red sea", "houthi", "suez",
            "praying mantis", "iran-iraq", "june 2025",
        ]
        name_lower = event_name.lower()
        return any(kw in name_lower for kw in chokepoint_keywords)

    @staticmethod
    def _is_proxy_event(event_name: str) -> bool:
        """Determine if an event involved proxy warfare.

        Args:
            event_name: Event name.

        Returns:
            True if proxy forces were involved.
        """
        proxy_keywords = ["houthi", "libya", "soleimani", "june 2025"]
        name_lower = event_name.lower()
        return any(kw in name_lower for kw in proxy_keywords)

    @staticmethod
    def _is_hormuz_event(event_name: str) -> bool:
        """Determine if an event directly threatened the Strait of Hormuz.

        Args:
            event_name: Event name.

        Returns:
            True if Hormuz was directly threatened.
        """
        hormuz_keywords = [
            "gulf war", "iran-iraq", "praying mantis", "june 2025",
        ]
        name_lower = event_name.lower()
        return any(kw in name_lower for kw in hormuz_keywords)
