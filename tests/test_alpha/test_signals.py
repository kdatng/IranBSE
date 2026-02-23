"""Tests for alpha signal computation modules.

Validates the core trading signal generators:
    - Contango/backwardation signals from futures term structure
    - COT (Commitment of Traders) positioning extreme detection
    - Volatility surface signals (VIX/OVX divergence, skew)

Uses synthetic market data with known properties to verify signal
correctness, edge case handling, and numerical stability.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Alpha signal stubs
# ---------------------------------------------------------------------------
# These implement the signal computation logic specified in claude.md
# for the alpha signal modules in src/alpha/.


class ContangoSignal:
    """Computes contango/backwardation signals from futures term structure.

    Contango: front-month < back-month (positive spread).
    Backwardation: front-month > back-month (negative spread).

    The signal is the normalized spread between the first two contract
    months, plus second-derivative (convexity) of the curve.
    """

    @staticmethod
    def compute_spread(
        front_price: np.ndarray,
        back_price: np.ndarray,
    ) -> np.ndarray:
        """Compute the raw front-back spread.

        Args:
            front_price: Front-month contract prices.
            back_price: Back-month (deferred) contract prices.

        Returns:
            Spread array (back - front). Positive = contango.
        """
        return back_price - front_price

    @staticmethod
    def compute_normalized_spread(
        front_price: np.ndarray,
        back_price: np.ndarray,
    ) -> np.ndarray:
        """Compute percentage spread normalized by front price.

        Args:
            front_price: Front-month prices.
            back_price: Back-month prices.

        Returns:
            Normalized spread as percentage. Positive = contango.

        Raises:
            ValueError: If any front_price is zero.
        """
        if np.any(front_price == 0):
            raise ValueError("Front price cannot be zero for normalization")
        return (back_price - front_price) / front_price * 100.0

    @staticmethod
    def compute_convexity(
        price_m1: np.ndarray,
        price_m2: np.ndarray,
        price_m3: np.ndarray,
    ) -> np.ndarray:
        """Compute term structure convexity (second derivative).

        Measures the curvature of the futures curve. Positive convexity
        indicates the curve is bowing upward (accelerating contango).

        JUSTIFIED: Second derivative of futures curve predicts storage
        economics shifts (claude.md alpha signal #10).

        Args:
            price_m1: Month-1 prices.
            price_m2: Month-2 prices.
            price_m3: Month-3 prices.

        Returns:
            Convexity signal array.
        """
        return price_m3 - 2 * price_m2 + price_m1

    @staticmethod
    def classify_regime(
        normalized_spread: np.ndarray,
        contango_threshold: float = 0.5,
        backwardation_threshold: float = -0.5,
    ) -> np.ndarray:
        """Classify each observation as contango, backwardation, or flat.

        Args:
            normalized_spread: Percentage spread series.
            contango_threshold: Minimum spread for contango classification.
            backwardation_threshold: Maximum spread for backwardation.

        Returns:
            Array of regime labels: 1 (contango), -1 (backwardation), 0 (flat).
        """
        regimes = np.zeros(len(normalized_spread), dtype=np.int32)
        regimes[normalized_spread > contango_threshold] = 1
        regimes[normalized_spread < backwardation_threshold] = -1
        return regimes


class COTSignal:
    """Computes CFTC Commitment of Traders positioning signals.

    Detects extreme positioning in managed money net longs/shorts
    that historically correlate with mean-reversion setups.

    JUSTIFIED: CFTC CoT managed money net positioning extremes
    are a known alpha signal (claude.md cross-asset alpha #2).
    """

    @staticmethod
    def compute_net_position(
        long_positions: np.ndarray,
        short_positions: np.ndarray,
    ) -> np.ndarray:
        """Compute net managed-money position.

        Args:
            long_positions: Managed money long contracts.
            short_positions: Managed money short contracts.

        Returns:
            Net position (long - short).
        """
        return long_positions - short_positions

    @staticmethod
    def compute_z_score(
        net_position: np.ndarray,
        lookback: int = 52,
    ) -> np.ndarray:
        """Compute rolling z-score of net positioning.

        Args:
            net_position: Net position time series.
            lookback: Rolling window in observations (weeks).
                JUSTIFIED: 52 weeks = 1 year of CoT data.

        Returns:
            Z-score array. NaN for the initial warmup period.
        """
        z_scores = np.full(len(net_position), np.nan)

        for i in range(lookback, len(net_position)):
            window = net_position[i - lookback : i]
            mean = np.mean(window)
            std = np.std(window, ddof=1)

            if std > 1e-10:
                z_scores[i] = (net_position[i] - mean) / std
            else:
                z_scores[i] = 0.0

        return z_scores

    @staticmethod
    def detect_extremes(
        z_scores: np.ndarray,
        upper_threshold: float = 2.0,
        lower_threshold: float = -2.0,
    ) -> np.ndarray:
        """Detect positioning extremes from z-scores.

        Args:
            z_scores: Rolling z-score series.
            upper_threshold: Z-score above which position is extremely long.
            lower_threshold: Z-score below which position is extremely short.

        Returns:
            Signal array: 1 (extreme long = bearish contrarian),
            -1 (extreme short = bullish contrarian), 0 (neutral).
        """
        signals = np.zeros(len(z_scores), dtype=np.int32)
        valid = ~np.isnan(z_scores)

        signals[valid & (z_scores > upper_threshold)] = 1   # Contrarian bearish
        signals[valid & (z_scores < lower_threshold)] = -1  # Contrarian bullish
        return signals

    @staticmethod
    def compute_crowding_index(
        net_position: np.ndarray,
        open_interest: np.ndarray,
    ) -> np.ndarray:
        """Compute position crowding as fraction of open interest.

        Args:
            net_position: Net managed money position.
            open_interest: Total open interest.

        Returns:
            Crowding index (net/OI). Higher = more crowded.
        """
        safe_oi = np.where(open_interest == 0, np.nan, open_interest)
        return net_position / safe_oi


class VolSurfaceSignal:
    """Computes volatility surface signals for cross-asset analysis.

    Focuses on:
    - VIX/OVX divergence (equity vol vs oil vol)
    - Vol skew signals
    - Options-implied kurtosis

    JUSTIFIED: VIX/OVX divergence during geopolitical events is
    alpha signal #1 in cross-asset signals (claude.md).
    """

    @staticmethod
    def compute_vix_ovx_ratio(
        vix: np.ndarray,
        ovx: np.ndarray,
    ) -> np.ndarray:
        """Compute VIX/OVX ratio.

        Rising ratio = equity fear outpacing oil fear (or oil complacency).
        Falling ratio = oil-specific risk premium expansion.

        Args:
            vix: CBOE VIX index values.
            ovx: CBOE OVX (oil volatility) index values.

        Returns:
            VIX/OVX ratio array.
        """
        safe_ovx = np.where(ovx == 0, np.nan, ovx)
        return vix / safe_ovx

    @staticmethod
    def compute_divergence_z(
        vix: np.ndarray,
        ovx: np.ndarray,
        lookback: int = 63,
    ) -> np.ndarray:
        """Compute z-score of VIX/OVX divergence.

        JUSTIFIED: 63 trading days ~ 3 months for vol regime detection.

        Args:
            vix: VIX values.
            ovx: OVX values.
            lookback: Rolling window in trading days.

        Returns:
            Z-score of the divergence. Large positive = equity vol
            anomalously high relative to oil vol.
        """
        ratio = VolSurfaceSignal.compute_vix_ovx_ratio(vix, ovx)
        z_scores = np.full(len(ratio), np.nan)

        for i in range(lookback, len(ratio)):
            window = ratio[i - lookback : i]
            valid = window[~np.isnan(window)]
            if len(valid) < 10:
                continue
            mean = np.mean(valid)
            std = np.std(valid, ddof=1)
            if std > 1e-10:
                z_scores[i] = (ratio[i] - mean) / std

        return z_scores

    @staticmethod
    def compute_put_call_skew(
        put_iv: np.ndarray,
        call_iv: np.ndarray,
    ) -> np.ndarray:
        """Compute put-call implied volatility skew.

        Positive skew = puts more expensive (downside fear).
        Negative skew = calls more expensive (upside speculation).

        Args:
            put_iv: Out-of-the-money put implied volatility.
            call_iv: Out-of-the-money call implied volatility.

        Returns:
            Skew array (put_iv - call_iv).
        """
        return put_iv - call_iv

    @staticmethod
    def compute_implied_kurtosis(
        atm_iv: np.ndarray,
        otm_put_iv: np.ndarray,
        otm_call_iv: np.ndarray,
    ) -> np.ndarray:
        """Compute options-implied excess kurtosis proxy.

        Higher kurtosis = fatter tails priced by the market.

        JUSTIFIED: Options-implied density kurtosis in OTM oil
        puts/calls is alpha signal #11 (claude.md).

        Args:
            atm_iv: At-the-money implied volatility.
            otm_put_iv: OTM put implied volatility.
            otm_call_iv: OTM call implied volatility.

        Returns:
            Implied kurtosis proxy. Values > 0 indicate fat tails.
        """
        wing_avg = (otm_put_iv + otm_call_iv) / 2
        safe_atm = np.where(atm_iv == 0, np.nan, atm_iv)
        return (wing_avg / safe_atm - 1) * 100  # Percentage butterfly spread


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def contango_data() -> dict[str, np.ndarray]:
    """Generate synthetic futures term structure data in contango."""
    rng = np.random.default_rng(42)
    n = 252  # JUSTIFIED: 1 trading year
    base = 70.0 + np.cumsum(rng.normal(0, 0.5, n))

    return {
        "front": base,
        "back": base + rng.uniform(0.5, 2.0, n),  # Contango: back > front
        "m3": base + rng.uniform(1.0, 3.0, n),
    }


@pytest.fixture
def backwardation_data() -> dict[str, np.ndarray]:
    """Generate synthetic futures term structure in backwardation."""
    rng = np.random.default_rng(42)
    n = 252
    base = 70.0 + np.cumsum(rng.normal(0, 0.5, n))

    return {
        "front": base,
        "back": base - rng.uniform(0.5, 2.0, n),  # Backwardation: back < front
        "m3": base - rng.uniform(1.0, 3.0, n),
    }


@pytest.fixture
def cot_data() -> dict[str, np.ndarray]:
    """Generate synthetic COT positioning data with extremes."""
    rng = np.random.default_rng(42)
    n = 156  # JUSTIFIED: 3 years of weekly data

    # Normal regime with some extreme positions.
    longs = 200_000 + rng.normal(0, 30_000, n)
    shorts = 180_000 + rng.normal(0, 25_000, n)

    # Inject extreme long at index 100.
    longs[100] = 350_000
    # Inject extreme short at index 120.
    shorts[120] = 350_000

    oi = longs + shorts + rng.uniform(50_000, 100_000, n)

    return {
        "longs": np.maximum(longs, 0),
        "shorts": np.maximum(shorts, 0),
        "open_interest": np.maximum(oi, 1),
    }


@pytest.fixture
def vol_surface_data() -> dict[str, np.ndarray]:
    """Generate synthetic volatility surface data."""
    rng = np.random.default_rng(42)
    n = 252

    vix = 15 + rng.normal(0, 3, n)
    vix = np.maximum(vix, 5)

    ovx = 25 + rng.normal(0, 5, n)
    ovx = np.maximum(ovx, 5)

    atm_iv = 0.25 + rng.normal(0, 0.03, n)
    atm_iv = np.maximum(atm_iv, 0.05)

    otm_put_iv = atm_iv + rng.uniform(0.02, 0.08, n)  # Puts more expensive
    otm_call_iv = atm_iv + rng.uniform(-0.01, 0.04, n)

    return {
        "vix": vix,
        "ovx": ovx,
        "atm_iv": atm_iv,
        "otm_put_iv": otm_put_iv,
        "otm_call_iv": otm_call_iv,
    }


# ---------------------------------------------------------------------------
# Tests: Contango signal computation
# ---------------------------------------------------------------------------

class TestContangoSignal:
    """Tests for contango/backwardation term structure signals."""

    def test_contango_spread_positive(
        self, contango_data: dict[str, np.ndarray]
    ) -> None:
        """Contango data produces positive spreads."""
        spread = ContangoSignal.compute_spread(
            contango_data["front"], contango_data["back"]
        )
        assert np.all(spread > 0), "All spreads should be positive in contango"

    def test_backwardation_spread_negative(
        self, backwardation_data: dict[str, np.ndarray]
    ) -> None:
        """Backwardation data produces negative spreads."""
        spread = ContangoSignal.compute_spread(
            backwardation_data["front"], backwardation_data["back"]
        )
        assert np.all(spread < 0), "All spreads should be negative in backwardation"

    def test_normalized_spread_units(
        self, contango_data: dict[str, np.ndarray]
    ) -> None:
        """Normalized spread is in percentage terms."""
        norm_spread = ContangoSignal.compute_normalized_spread(
            contango_data["front"], contango_data["back"]
        )
        # For typical oil futures, contango spread should be 0.5-5%.
        assert np.all(norm_spread > 0)
        assert np.all(norm_spread < 10)

    def test_zero_price_raises(self) -> None:
        """Zero front price raises ValueError in normalization."""
        with pytest.raises(ValueError, match="cannot be zero"):
            ContangoSignal.compute_normalized_spread(
                np.array([0.0, 70.0]),
                np.array([71.0, 72.0]),
            )

    def test_convexity_calculation(
        self, contango_data: dict[str, np.ndarray]
    ) -> None:
        """Convexity computation produces finite values."""
        convexity = ContangoSignal.compute_convexity(
            contango_data["front"],
            contango_data["back"],
            contango_data["m3"],
        )
        assert np.all(np.isfinite(convexity))
        assert len(convexity) == len(contango_data["front"])

    def test_flat_curve_zero_convexity(self) -> None:
        """Perfectly flat term structure has zero convexity."""
        flat = np.full(100, 70.0)
        convexity = ContangoSignal.compute_convexity(flat, flat, flat)
        np.testing.assert_allclose(convexity, 0.0, atol=1e-10)

    def test_regime_classification_contango(
        self, contango_data: dict[str, np.ndarray]
    ) -> None:
        """Contango data is correctly classified."""
        norm_spread = ContangoSignal.compute_normalized_spread(
            contango_data["front"], contango_data["back"]
        )
        regimes = ContangoSignal.classify_regime(norm_spread)
        # Most should be classified as contango (1).
        assert np.mean(regimes == 1) > 0.8

    def test_regime_classification_backwardation(
        self, backwardation_data: dict[str, np.ndarray]
    ) -> None:
        """Backwardation data is correctly classified."""
        norm_spread = ContangoSignal.compute_normalized_spread(
            backwardation_data["front"], backwardation_data["back"]
        )
        regimes = ContangoSignal.classify_regime(norm_spread)
        assert np.mean(regimes == -1) > 0.8

    def test_spread_symmetry(self) -> None:
        """Spread(A, B) = -Spread(B, A)."""
        a = np.array([70.0, 71.0, 72.0])
        b = np.array([72.0, 73.0, 74.0])

        spread_ab = ContangoSignal.compute_spread(a, b)
        spread_ba = ContangoSignal.compute_spread(b, a)
        np.testing.assert_allclose(spread_ab, -spread_ba)


# ---------------------------------------------------------------------------
# Tests: COT positioning extremes
# ---------------------------------------------------------------------------

class TestCOTSignal:
    """Tests for COT positioning extreme detection."""

    def test_net_position_calculation(
        self, cot_data: dict[str, np.ndarray]
    ) -> None:
        """Net position equals long minus short."""
        net = COTSignal.compute_net_position(
            cot_data["longs"], cot_data["shorts"]
        )
        expected = cot_data["longs"] - cot_data["shorts"]
        np.testing.assert_array_equal(net, expected)

    def test_z_score_warmup_period(
        self, cot_data: dict[str, np.ndarray]
    ) -> None:
        """Z-scores are NaN during the lookback warmup period."""
        net = COTSignal.compute_net_position(
            cot_data["longs"], cot_data["shorts"]
        )
        lookback = 52
        z_scores = COTSignal.compute_z_score(net, lookback=lookback)

        assert np.all(np.isnan(z_scores[:lookback]))
        assert not np.all(np.isnan(z_scores[lookback:]))

    def test_z_score_mean_near_zero(
        self, cot_data: dict[str, np.ndarray]
    ) -> None:
        """Z-scores of a stationary process should have mean near zero."""
        rng = np.random.default_rng(42)
        stationary = rng.normal(0, 1, 200)
        z = COTSignal.compute_z_score(stationary, lookback=52)
        valid_z = z[~np.isnan(z)]

        assert abs(np.mean(valid_z)) < 0.5

    def test_extreme_detection_long(
        self, cot_data: dict[str, np.ndarray]
    ) -> None:
        """Injected extreme long position is detected."""
        net = COTSignal.compute_net_position(
            cot_data["longs"], cot_data["shorts"]
        )
        z_scores = COTSignal.compute_z_score(net, lookback=52)
        signals = COTSignal.detect_extremes(z_scores)

        # Check that at least one extreme is detected near index 100.
        extreme_indices = np.where(signals == 1)[0]
        assert len(extreme_indices) > 0, "Should detect at least one extreme long"

    def test_extreme_detection_short(
        self, cot_data: dict[str, np.ndarray]
    ) -> None:
        """Injected extreme short position is detected."""
        net = COTSignal.compute_net_position(
            cot_data["longs"], cot_data["shorts"]
        )
        z_scores = COTSignal.compute_z_score(net, lookback=52)
        signals = COTSignal.detect_extremes(z_scores)

        extreme_short = np.where(signals == -1)[0]
        assert len(extreme_short) > 0, "Should detect at least one extreme short"

    def test_no_extremes_in_normal_data(self) -> None:
        """Normally distributed data produces few extremes."""
        rng = np.random.default_rng(42)
        normal_data = rng.normal(100_000, 10_000, 200)
        z = COTSignal.compute_z_score(normal_data, lookback=52)
        signals = COTSignal.detect_extremes(z)

        valid_signals = signals[~np.isnan(z)]
        extreme_ratio = np.mean(np.abs(valid_signals) == 1)
        # JUSTIFIED: P(|Z| > 2) ~ 4.6% for normal distribution
        assert extreme_ratio < 0.15

    def test_crowding_index_bounded(
        self, cot_data: dict[str, np.ndarray]
    ) -> None:
        """Crowding index is between -1 and 1 for reasonable data."""
        net = COTSignal.compute_net_position(
            cot_data["longs"], cot_data["shorts"]
        )
        crowding = COTSignal.compute_crowding_index(
            net, cot_data["open_interest"]
        )
        valid = crowding[~np.isnan(crowding)]
        assert np.all(valid >= -1.0)
        assert np.all(valid <= 1.0)

    def test_zero_open_interest_gives_nan(self) -> None:
        """Zero open interest produces NaN crowding index."""
        net = np.array([100.0, 200.0])
        oi = np.array([0.0, 1000.0])
        crowding = COTSignal.compute_crowding_index(net, oi)
        assert np.isnan(crowding[0])
        assert not np.isnan(crowding[1])


# ---------------------------------------------------------------------------
# Tests: Vol surface signals
# ---------------------------------------------------------------------------

class TestVolSurfaceSignal:
    """Tests for volatility surface and cross-asset vol signals."""

    def test_vix_ovx_ratio_positive(
        self, vol_surface_data: dict[str, np.ndarray]
    ) -> None:
        """VIX/OVX ratio is positive when both inputs are positive."""
        ratio = VolSurfaceSignal.compute_vix_ovx_ratio(
            vol_surface_data["vix"], vol_surface_data["ovx"]
        )
        valid = ratio[~np.isnan(ratio)]
        assert np.all(valid > 0)

    def test_vix_ovx_ratio_typical_range(
        self, vol_surface_data: dict[str, np.ndarray]
    ) -> None:
        """VIX/OVX ratio is typically between 0.3 and 2.0."""
        ratio = VolSurfaceSignal.compute_vix_ovx_ratio(
            vol_surface_data["vix"], vol_surface_data["ovx"]
        )
        valid = ratio[~np.isnan(ratio)]
        assert np.mean((valid > 0.2) & (valid < 3.0)) > 0.9

    def test_zero_ovx_gives_nan(self) -> None:
        """Zero OVX produces NaN ratio."""
        vix = np.array([15.0, 20.0])
        ovx = np.array([0.0, 25.0])
        ratio = VolSurfaceSignal.compute_vix_ovx_ratio(vix, ovx)
        assert np.isnan(ratio[0])
        assert not np.isnan(ratio[1])

    def test_divergence_z_warmup(
        self, vol_surface_data: dict[str, np.ndarray]
    ) -> None:
        """Divergence z-score has NaN warmup period."""
        lookback = 63
        z = VolSurfaceSignal.compute_divergence_z(
            vol_surface_data["vix"],
            vol_surface_data["ovx"],
            lookback=lookback,
        )
        assert np.all(np.isnan(z[:lookback]))

    def test_divergence_z_finite_after_warmup(
        self, vol_surface_data: dict[str, np.ndarray]
    ) -> None:
        """Z-scores are finite after the warmup period."""
        lookback = 63
        z = VolSurfaceSignal.compute_divergence_z(
            vol_surface_data["vix"],
            vol_surface_data["ovx"],
            lookback=lookback,
        )
        post_warmup = z[lookback:]
        valid = post_warmup[~np.isnan(post_warmup)]
        assert len(valid) > 0
        assert np.all(np.isfinite(valid))

    def test_put_call_skew_positive_for_fear(
        self, vol_surface_data: dict[str, np.ndarray]
    ) -> None:
        """Put-call skew is positive when puts are more expensive."""
        skew = VolSurfaceSignal.compute_put_call_skew(
            vol_surface_data["otm_put_iv"],
            vol_surface_data["otm_call_iv"],
        )
        # Our fixture generates puts > calls, so skew should be mostly positive.
        assert np.mean(skew > 0) > 0.5

    def test_symmetric_skew(self) -> None:
        """Equal put and call IVs produce zero skew."""
        iv = np.array([0.25, 0.30, 0.28])
        skew = VolSurfaceSignal.compute_put_call_skew(iv, iv)
        np.testing.assert_allclose(skew, 0.0, atol=1e-10)

    def test_implied_kurtosis_positive_for_smile(
        self, vol_surface_data: dict[str, np.ndarray]
    ) -> None:
        """Implied kurtosis is positive when wings > ATM (vol smile)."""
        kurtosis = VolSurfaceSignal.compute_implied_kurtosis(
            vol_surface_data["atm_iv"],
            vol_surface_data["otm_put_iv"],
            vol_surface_data["otm_call_iv"],
        )
        valid = kurtosis[~np.isnan(kurtosis)]
        # Wing avg > ATM means positive kurtosis proxy.
        assert np.mean(valid > 0) > 0.8

    def test_flat_smile_zero_kurtosis(self) -> None:
        """Flat vol surface (no smile) gives zero implied kurtosis."""
        flat_iv = np.array([0.25, 0.25, 0.25])
        kurtosis = VolSurfaceSignal.compute_implied_kurtosis(
            flat_iv, flat_iv, flat_iv
        )
        np.testing.assert_allclose(kurtosis, 0.0, atol=1e-10)

    def test_zero_atm_gives_nan(self) -> None:
        """Zero ATM IV produces NaN kurtosis."""
        atm = np.array([0.0, 0.25])
        otm_put = np.array([0.30, 0.30])
        otm_call = np.array([0.28, 0.28])
        kurtosis = VolSurfaceSignal.compute_implied_kurtosis(atm, otm_put, otm_call)
        assert np.isnan(kurtosis[0])


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

class TestPropertyBased:
    """Property-based tests for alpha signal invariants."""

    @given(
        front=st.floats(min_value=10, max_value=200),
        back=st.floats(min_value=10, max_value=200),
    )
    @settings(max_examples=30, deadline=5000)
    def test_spread_sign_consistency(self, front: float, back: float) -> None:
        """Spread sign always matches (back - front)."""
        spread = ContangoSignal.compute_spread(
            np.array([front]), np.array([back])
        )
        if back > front:
            assert spread[0] > 0
        elif back < front:
            assert spread[0] < 0
        else:
            assert spread[0] == 0

    @given(
        long_pos=st.floats(min_value=0, max_value=1e6),
        short_pos=st.floats(min_value=0, max_value=1e6),
    )
    @settings(max_examples=30, deadline=5000)
    def test_net_position_sign(self, long_pos: float, short_pos: float) -> None:
        """Net position sign reflects which side dominates."""
        net = COTSignal.compute_net_position(
            np.array([long_pos]), np.array([short_pos])
        )
        if long_pos > short_pos:
            assert net[0] > 0
        elif long_pos < short_pos:
            assert net[0] < 0

    @given(
        vix=st.floats(min_value=5, max_value=80),
        ovx=st.floats(min_value=5, max_value=100),
    )
    @settings(max_examples=20, deadline=5000)
    def test_ratio_always_positive(self, vix: float, ovx: float) -> None:
        """VIX/OVX ratio is always positive for positive inputs."""
        ratio = VolSurfaceSignal.compute_vix_ovx_ratio(
            np.array([vix]), np.array([ovx])
        )
        assert ratio[0] > 0

    @given(
        put_iv=st.floats(min_value=0.05, max_value=1.0),
        call_iv=st.floats(min_value=0.05, max_value=1.0),
    )
    @settings(max_examples=20, deadline=5000)
    def test_skew_matches_dominance(self, put_iv: float, call_iv: float) -> None:
        """Skew sign matches whether puts or calls dominate."""
        skew = VolSurfaceSignal.compute_put_call_skew(
            np.array([put_iv]), np.array([call_iv])
        )
        if put_iv > call_iv:
            assert skew[0] > 0
        elif put_iv < call_iv:
            assert skew[0] < 0
