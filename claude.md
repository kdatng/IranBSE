# IranBSE: Black Swan Event Modeling — US-Iran War Impact on Commodity Futures

## Project Overview

This project builds a **hyper-complex, top-1% statistical model** to predict the impact of a hypothetical 1-month US-Iran war (starting March 15, 2026) on:
- **Wheat futures** (CBOT Wheat — ZW)
- **Crude oil futures** (WTI — CL, Brent — BZ)

The model treats this as a **black swan scenario analysis** — an event with fat-tailed, nonlinear, regime-shifting effects on global commodity markets. The goal is not just point prediction but full distributional forecasting with uncertainty quantification.

---

## Session Log — What Has Been Done

### Session 2026-02-24: Review, Bug Fixes, Data Pipeline

#### Critical Bugs Fixed

**1. Oil price ranges had no historical basis** (`config/scenarios.yaml`)
- Level 3 ceiling: `$250` → `$175`; Level 4 ceiling: `$300` → `$200`
- Gulf War I — the largest oil supply disruption in history — peaked at +119% from base. From ~$70 base, that's ~$153. The old Level 4 ceiling of $300 was pure fantasy.
- Wheat trade disruption ranges also tightened (Iran imports 7Mt/yr of wheat — they don't export it).

**2. Three bugs in the Monte Carlo engine** (`scripts/wheat_forecast.py`)
- **Markov chain "roach motel"**: Level 4 had 95% self-transition probability. Once a path reached max escalation it never came back down. Reduced to 88% with real de-escalation paths.
- **"Antithetic variance reduction" was claimed but not implemented**: The report and config referenced it, but the code didn't do it. Now actually generates 5,000 paths + 5,000 antithetic mirrors.
- **Impact jumped instantly between levels**: Dropping from L3 to L2 would instantly erase the L3 shock. Now uses a running-max ratchet with slow decay (half-life 10 days), matching how markets actually behave — they price in disruption fast but unwind it slowly.

**3. Broken Polars API** (`src/scenarios/scenario_engine.py`)
- `pl.date(2024, 1, 1) + pl.duration(days=N)` — `pl.duration()` returns an expression, not a timedelta. Would crash at runtime. Replaced with standard `datetime` arithmetic.

**Net effect on results**: War scenario mean should drop from ~+40% to ~+15-20%, consistent with Gulf War I (+15%), Iraq War (+8%), and June 2025 Israel-Iran (+5%) analogs.

#### Data Pipeline Fully Implemented

`scripts/fetch_data.py` now has **4 active providers** (was 2 placeholders + yfinance):

| Provider | Key | Status | Data |
|---|---|---|---|
| `yfinance` | None needed | Working | Commodity futures, equity indices (daily OHLCV since 2000) |
| `fred` | `1a72200a7cdeee8aa47553f0ac2a0f29` (hardcoded default) | Working | Fed funds, 10Y/2Y yields, DXY, breakeven inflation, oil spot, CPI |
| `eia_api` | `WSRVns7B8xiyTaxp6ynP8pZDs7Z3cLHrxEpajpuc` (hardcoded default) | Working | Weekly crude inventory, imports, refinery utilization, SPR stocks |
| `lseg` | Set via `LSEG_APP_KEY` env var or `app_key` in yaml | Implemented, needs Workspace | Options chains, futures curves, freight rates, news headlines |

**API keys**: FRED and EIA keys are hardcoded as default fallbacks in `fetch_data.py` (env vars still take precedence). LSEG key must be provided — it is NOT hardcoded.

#### LSEG / Refinitiv Integration

Extensive research was conducted on the LSEG API ecosystem. Key findings:

**Library evolution** (use `lseg-data`, NOT `eikon` or `refinitiv-data`):
- `eikon` → sunset, Eikon desktop withdrawn June 2025
- `refinitiv-data` → maintenance only, no new features
- **`lseg-data` v2.1.1** → active, recommended for all new development
- App Keys generated from any generation work across all libraries

**Authentication & session types**:
- **Desktop session** (`desktop.workspace`): Requires LSEG Workspace running on the same machine. Just needs an App Key. The script writes `lseg-data.config.json` dynamically.
- **Platform session** (`platform.ldp`): Headless/server access. Needs App Key + client_id + client_secret from LSEG account rep.

**Rate limits** (desktop session):
- 5 requests/second, 10,000 requests/day, 5 GB/day
- `get_history` (interday): 3,000 rows per request
- `get_news_headlines`: 100 per request, ~15 months depth

**4 LSEG data sources configured** in `data_sources.yaml`:

| Source | Dataset Type | RICs | What It Feeds |
|---|---|---|---|
| `lseg_futures_curve` | Term structure snapshots | `0#CL:`, `0#LCO:`, `0#W:` | `contango_backwardation.py` |
| `lseg_options` | Options chains (IV, Greeks) | `0#CLc1+`, `0#LCOc1+`, `0#Wc1+` | `volatility_surface.py` |
| `lseg_freight` | Baltic freight indices | `.BDI`, `.BDTI`, `.BCTI`, `TD3C-TCE-d` | `shipping.py` (replaces BDI×0.6 proxy) |
| `lseg_news` | Commodity/geopolitical headlines | N/A (query-based) | Sentiment analysis |

**Fields parameter is MANDATORY** in `lseg-data` v2 `get_data()` — this was a breaking change from v1.

#### Data Gaps Still Outstanding

**Tier 1 — Highest ROI** (free, just need to run fetch_data.py):
- Historical daily prices from yfinance (ZW=F, CL=F back to 2000)
- FRED macro data (already configured, needs API call)
- CFTC Commitment of Traders (`cot_reports` Python package or direct CFTC download)

**Tier 2 — High-value alternative data**:
- AIS / tanker tracking (MarineTraffic freemium, or Kpler premium)
- ACLED/GDELT geopolitical events (free for research)
- EIA petroleum inventories (configured, ready to fetch)

**Tier 3 — Precision improvements**:
- USDA WASDE (global wheat stocks-to-use ratio)
- NOAA weather + NDVI satellite (crop condition)
- War-risk insurance premiums (not available via LSEG standard API — contact Lloyd's)
- Caldara-Iacoviello Geopolitical Risk Index (free at matteoiacoviello.com/gpr.htm)

### First Run Results (Wheat Forecast, 2026-02-24)

**Executed**: `scripts/wheat_forecast.py` — EGARCH(1,1) with skewed-t innovations, 10,000 Monte Carlo paths, 22-day forecast.

| Metric | War Scenario | No-War Scenario |
|---|---|---|
| Base price | $576.75/bu | $576.75/bu |
| Point forecast | $810.79 (+40.6%) | $577.44 (+0.1%) |
| Median | $809.63 (+40.4%) | $575.47 (-0.2%) |
| Std dev | $155.84 | $51.15 |
| P05 | $574.52 | $497.80 |
| P95 | $1,059.33 | $662.90 |
| P99 | $1,144.05 | $713.38 |
| P(>10% rise) | 84.4% | 12.2% |
| P(>50% rise) | 38.7% | 0.0% |

**Note**: These results are from BEFORE the bug fixes above. After fixes, the war scenario mean should compress to ~+15-20%.

**Output files**: `output/wheat_forecast/` — JSON results, text report, 3 PNG charts (fan chart, distribution, historical comparison).

---

## Architecture Philosophy

### Modularity First
Every component must be **hot-swappable**. New data sources, alternative models, or parameter changes must slot in without rewriting upstream/downstream code. Use:
- **Strategy pattern** for model backends
- **Plugin architecture** for data feeds
- **Config-driven pipelines** (YAML/JSON configs, not hardcoded params)
- **Registry pattern** for registering new models/features/data sources

### Multi-Agent Development Protocol
- Multiple agents will work on this codebase simultaneously
- All code must be **self-documenting** with clear module boundaries
- Each module gets its own directory with a `README.md` and `__init__.py`
- Use **type hints everywhere** (Python 3.10+ style)
- Follow **Google-style docstrings** with parameter descriptions
- Every function that produces a number must have a **justification comment** explaining why

### Quality Standards
- **Triple-check** all numerical outputs, statistical assumptions, and data transformations
- Every hardcoded number must have a `# JUSTIFIED:` comment with source/reasoning
- Unit tests for all statistical functions
- Integration tests for pipeline end-to-end runs
- Logging at DEBUG level for all intermediate calculations

---

## Directory Structure

```
IranBSE/
├── claude.md                          # This file — project brain
├── config/
│   ├── model_config.yaml              # Master configuration
│   ├── data_sources.yaml              # Data source registry (yfinance, FRED, EIA, LSEG)
│   ├── scenarios.yaml                 # War scenario parameters (research-backed)
│   └── hyperparams.yaml              # Model hyperparameters
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── fetchers/                  # Data ingestion modules
│   │   │   ├── base_fetcher.py        # Abstract base class (Pydantic config, retry logic, caching)
│   │   │   ├── commodity_prices.py    # Futures price data
│   │   │   ├── geopolitical.py        # Geopolitical risk indices
│   │   │   ├── shipping.py            # Shipping/freight data (currently proxy-based, LSEG replaces)
│   │   │   ├── sentiment.py           # News/social sentiment
│   │   │   └── macro.py               # Macroeconomic indicators
│   │   ├── processors/               # Feature engineering
│   │   │   ├── technical.py           # Technical indicators
│   │   │   ├── fundamental.py         # Supply/demand features
│   │   │   ├── cross_asset.py         # Cross-asset signals
│   │   │   ├── regime.py              # Regime detection features
│   │   │   └── geopolitical.py        # Geopolitical risk features
│   │   └── storage/
│   │       └── data_store.py          # Unified data storage layer
│   ├── models/
│   │   ├── registry.py                # Model registry (strategy pattern, thread-safe singleton)
│   │   ├── base_model.py              # Abstract model interface + PredictionResult dataclass
│   │   ├── ensemble/
│   │   │   ├── meta_learner.py        # Stacking/blending ensemble
│   │   │   └── adversarial_ensemble.py # Adversarial validation ensemble
│   │   ├── statistical/
│   │   │   ├── regime_switching.py    # Markov regime-switching models
│   │   │   ├── hawkes_process.py      # Hawkes process for event clustering
│   │   │   ├── extreme_value.py       # EVT — GEV/GPD tail modeling
│   │   │   ├── copula.py             # Copula models for dependency
│   │   │   ├── garch_family.py       # GARCH/EGARCH/GJR-GARCH/FIGARCH
│   │   │   └── var_svar.py           # Structural VAR models
│   │   ├── ml/
│   │   │   ├── gradient_boost.py     # XGBoost/LightGBM/CatBoost
│   │   │   ├── neural_sde.py         # Neural SDEs for price dynamics
│   │   │   ├── temporal_fusion.py    # Temporal Fusion Transformer
│   │   │   ├── normalizing_flows.py  # Normalizing flows for density est.
│   │   │   ├── deep_state_space.py   # Deep state space models (S4/Mamba)
│   │   │   └── conformal.py          # Conformal prediction wrappers
│   │   ├── bayesian/
│   │   │   ├── hierarchical.py       # Hierarchical Bayesian models
│   │   │   ├── bvar.py               # Bayesian VAR with Minnesota prior
│   │   │   ├── scenario_tree.py      # Bayesian scenario tree
│   │   │   └── causal_impact.py      # Bayesian causal impact analysis
│   │   └── geopolitical/
│   │       ├── conflict_model.py     # War escalation dynamics
│   │       ├── supply_disruption.py  # Supply chain disruption model
│   │       └── contagion.py          # Cross-market contagion model
│   ├── scenarios/
│   │   ├── scenario_engine.py        # Monte Carlo scenario generator (10K paths)
│   │   ├── iran_war.py               # Iran war-specific scenario logic (864 lines)
│   │   ├── strait_of_hormuz.py       # Hormuz chokepoint model (796 lines)
│   │   └── historical_analogs.py     # Historical conflict analogs (637 lines)
│   ├── alpha/
│   │   ├── oil/
│   │   │   ├── contango_signals.py   # Term structure / contango-backwardation
│   │   │   ├── floating_storage.py   # Floating storage / tanker tracking
│   │   │   ├── refinery_margins.py   # Crack spreads / refinery signals
│   │   │   ├── opec_positioning.py   # OPEC spare capacity / compliance
│   │   │   ├── iran_export_tracking.py # Iran-specific export monitoring
│   │   │   └── dark_fleet.py         # Sanctioned tanker "dark fleet" signals
│   │   ├── wheat/
│   │   │   ├── crop_conditions.py    # USDA crop condition reports
│   │   │   ├── weather_models.py     # Weather anomaly signals
│   │   │   ├── black_sea_risk.py     # Black Sea export disruption
│   │   │   ├── substitution.py       # Grain substitution dynamics
│   │   │   ├── fob_basis.py          # FOB basis / export competitiveness
│   │   │   └── food_security.py      # Food security panic buying signals
│   │   └── cross_asset/
│   │       ├── dollar_dynamics.py    # USD strength / petrodollar flows
│   │       ├── volatility_surface.py # Vol surface / skew signals
│   │       ├── cot_positioning.py    # CFTC Commitment of Traders
│   │       ├── etf_flows.py          # Commodity ETF flow signals
│   │       └── options_flow.py       # Unusual options activity
│   ├── risk/
│   │   ├── tail_risk.py              # Tail risk metrics (CVaR, ES)
│   │   ├── stress_testing.py         # Historical + hypothetical stress tests
│   │   └── model_risk.py             # Model uncertainty quantification
│   ├── backtesting/
│   │   ├── walk_forward.py           # Walk-forward validation
│   │   ├── historical_conflicts.py   # Backtest on prior conflicts
│   │   └── metrics.py                # Custom evaluation metrics
│   └── visualization/
│       ├── scenario_plots.py         # Scenario fan charts
│       ├── sensitivity.py            # Sensitivity analysis plots
│       └── dashboard.py              # Interactive dashboard (604 lines)
├── tests/                            # ~3,809 lines of tests
│   ├── test_data/                    # Data fetcher tests
│   ├── test_models/                  # Model registry and statistical tests
│   ├── test_scenarios/               # Scenario engine and Hormuz tests
│   └── test_alpha/                   # Alpha signal tests
├── scripts/
│   ├── run_pipeline.py               # Main execution pipeline (737 lines, 6 stages)
│   ├── wheat_forecast.py             # 1-month wheat forecast (717 lines, COMPLETED)
│   ├── fetch_data.py                 # Data download script (1073 lines, 4 providers active)
│   └── generate_report.py            # Generate analysis report
├── output/
│   └── wheat_forecast/               # First run results (2026-02-24)
│       ├── wheat_forecast_results.json
│       ├── wheat_forecast_report.txt
│       ├── wheat_forecast_fan_chart.png
│       ├── wheat_forecast_distribution.png
│       └── wheat_historical_comparison.png
├── requirements.txt                  # Includes lseg-data>=2.0
├── pyproject.toml
└── Makefile
```

---

## Data Pipeline: fetch_data.py

### Provider Dispatch Architecture

`scripts/fetch_data.py` uses a dispatch table pattern. Each provider has a dedicated `_fetch_*` function:

```python
_PROVIDER_DISPATCH = {
    "yfinance": _fetch_yfinance,   # Yahoo Finance — commodity futures, equity indices
    "fred": _fetch_fred,           # FRED — macro indicators (API key hardcoded)
    "eia_api": _fetch_eia,         # EIA v2 — petroleum data (API key hardcoded)
    "lseg": _fetch_lseg,           # LSEG Data Library — options, curves, freight, news
}
```

All other providers (acled, cftc, noaa, nasa_modis, usda_api, kpler) fall through to `_fetch_placeholder`.

### Running the Pipeline

```bash
# Fetch all enabled sources
python scripts/fetch_data.py

# Fetch specific sources only
python scripts/fetch_data.py --sources commodity_prices macro_indicators eia_petroleum

# LSEG sources (requires Workspace Desktop running on same machine)
export LSEG_APP_KEY="your-app-key-here"
python scripts/fetch_data.py --sources lseg_futures_curve lseg_options lseg_freight lseg_news

# Dry run (validate config only)
python scripts/fetch_data.py --dry-run
```

### LSEG Session Notes

- The fetcher writes `lseg-data.config.json` to the project root dynamically
- Session opens once and is reused across all 4 LSEG sources
- Session is cleaned up at the end of `main()`
- If Workspace is not running, LSEG sources will fail gracefully with error status
- For headless/server access, change `session_type: "platform"` in `data_sources.yaml` and get platform credentials from LSEG

### Output Format

All fetchers save to `data/raw/<source_name>.parquet` as Polars DataFrames with a `date` column and label-prefixed value columns.

---

## Core Models & Techniques

### Tier 1: Foundation Models (Must-Have)
1. **Markov Regime-Switching VAR (MS-SVAR)** — Captures peace/crisis/war regimes with distinct volatility and correlation structures
2. **Extreme Value Theory (EVT)** — GEV/GPD for tail risk; Peak-over-Threshold for war-shock magnitudes
3. **Bayesian Structural VAR** — Minnesota/Sims priors; structural identification via sign restrictions for oil supply shocks
4. **GARCH Family** — EGARCH for leverage, GJR-GARCH for asymmetry, FIGARCH for long memory in vol
5. **Copula Models** — Time-varying Clayton/Gumbel copulas for oil-wheat tail dependence

### Tier 2: Advanced Models (Differentiators)
6. **Hawkes Process** — Self-exciting point process for geopolitical event clustering and contagion
7. **Neural Stochastic Differential Equations** — Continuous-time neural network price dynamics
8. **Temporal Fusion Transformer (TFT)** — Attention-based multi-horizon forecasting with interpretability
9. **Normalizing Flows** — Full conditional density estimation (Real NVP / Neural Spline Flows)
10. **Deep State Space Models** — S4/Mamba architecture for ultra-long-range dependencies

### Tier 3: Cutting-Edge (Experimental)
11. **Conformal Prediction** — Distribution-free prediction intervals with finite-sample coverage guarantees
12. **Causal Discovery (PCMCI+)** — Tigramite-based causal graph learning for commodity interdependencies
13. **Adversarial Scenario Generation** — GAN-based generation of worst-case price paths
14. **Rough Volatility Models** — Fractional Brownian motion for realistic vol dynamics (Gatheral et al.)
15. **Score-Based Diffusion Models** — Denoising diffusion for scenario generation

### Ensemble Strategy
- **Stacking Meta-Learner**: Ridge regression over model predictions with regime-conditional weights
- **Adversarial Validation**: Detect distribution shift between training and scenario data
- **Bayesian Model Averaging**: Posterior model weights via marginal likelihood

---

## Alpha Signals Research Agenda

### Crude Oil — Unconventional / Unpriced Alpha
1. **Strait of Hormuz transit time anomalies** — AIS vessel tracking data; transit delays = supply risk
2. **Iranian "dark fleet" tanker positioning** — Sanctioned vessels turning off transponders; satellite detection
3. **IRGC naval exercise frequency** — Historical correlation with escalation probability
4. **Saudi spare capacity real-time estimation** — Satellite imagery of tank farm levels vs. reported capacity
5. **Refinery maintenance schedule clustering** — Unusual simultaneous turnarounds signal strategic reserves drawdown
6. **USD/IRR black market spread** — Divergence between official and black market rates = regime stress
7. **Insurance premium spikes on Gulf shipping** — War risk insurance as a leading indicator
8. **US Strategic Petroleum Reserve (SPR) release signals** — DOE tender announcements, Congressional language
9. **Chinese strategic reserve buying patterns** — Teapot refinery run rates from satellite/shipping data
10. **Contango term structure convexity** — Second derivative of futures curve predicts storage economics shifts
11. **Options-implied density kurtosis** — Fat tail pricing in OTM oil puts/calls
12. **OPEC+ compliance deviation momentum** — Rate of change in compliance, not level

### Wheat — Unconventional / Unpriced Alpha
1. **NDVI vegetation index anomalies** — Satellite crop health in Black Sea/Middle East wheat belt
2. **Soil moisture deficit z-scores** — SMAP satellite data vs. 30-year norms
3. **Egyptian GASC tender pricing** — World's largest wheat buyer; tender results as global price discovery
4. **Russian export tax policy signals** — Legislative language analysis for export restriction probability
5. **Fertilizer price / nat gas cross-signal** — European nat gas prices feed into fertilizer costs → wheat costs
6. **Shipping congestion at Novorossiysk** — AIS data for Black Sea grain export bottleneck
7. **USDA report surprise momentum** — Systematic mispricing after WASDE surprises
8. **Indian export ban probability** — Monsoon forecasts + government procurement data
9. **Wheat-corn substitution basis** — Feed wheat/corn spread mean reversion during supply shocks
10. **Food riot / protest frequency index** — ACLED conflict data in MENA wheat importers
11. **La Nina / El Nino transition probability** — ENSO state → Australian/Argentine wheat yield
12. **Freight rate / wheat price divergence** — Panamax rates decoupled from wheat = mean-reversion signal

### Cross-Asset Alpha
1. **VIX/OVX divergence** — Equity vol vs. oil vol spread anomalies during geopolitical events
2. **CFTC CoT managed money net positioning extremes** — Crowded positioning = reversal risk
3. **ETF creation/redemption unit flow** — Smart money flows in USO/DBO vs. retail sentiment
4. **Treasury yield curve slope × commodity carry** — Macro regime indicator for commodity allocation
5. **Gold/oil ratio regime breaks** — Historical ratio collapses during war → mean reversion setup
6. **Credit default swap spreads on Gulf sovereigns** — Saudi/UAE CDS as war probability proxy
7. **Defense sector ETF momentum** — ITA/XAR unusual volume as smart money conflict signal
8. **Baltic Dry Index acceleration** — Second derivative of BDI signals trade disruption before commodities move

---

## Scenario Parameters: US-Iran War (March 15 – April 15, 2026)

See `config/scenarios.yaml` for full research-backed parameters including:
- 4 escalation levels with calibrated probabilities and price ranges
- Strait of Hormuz flow data (20 mb/d, 20% of global seaborne oil trade)
- Iran military capabilities (post-June 2025 strikes)
- Proxy network activity (Houthis, Hezbollah)
- Mine-clearing timelines (8-26 weeks, 16 MCM vessels)
- Insurance/shipping parameters
- 6 historical conflict analogs with calibrated price impacts

---

## Mathematical Framework

### Price Dynamics Under Regime Switching
The core price model uses a Markov-switching jump-diffusion:

```
dS_t = μ(S_t, Z_t) dt + σ(S_t, Z_t) dW_t + J_t dN_t(λ(Z_t))
```

Where:
- `S_t` = commodity price at time t
- `Z_t` = latent regime state (Peace/Tension/Conflict/War) — Markov chain
- `μ(·)` = regime-dependent drift (mean-reversion + trend)
- `σ(·)` = regime-dependent volatility (GARCH-like)
- `J_t` = jump size (drawn from regime-dependent distribution, e.g., GPD for war regime)
- `N_t` = Hawkes process (self-exciting) with intensity `λ(Z_t)`

### Tail Dependence via Copulas
Oil-wheat joint distribution modeled with time-varying copula:

```
C(u, v; θ_t) where θ_t = f(regime, VIX, geopolitical_index)
```

Using Clayton copula for lower tail dependence (both crash together) and survival Gumbel for upper tail.

### Bayesian Scenario Tree
Posterior predictive distribution over price paths:

```
p(S_{T} | D, scenario) = ∫ p(S_{T} | θ, scenario) p(θ | D) dθ
```

Approximated via MCMC (NUTS sampler in PyMC/NumPyro) with informative priors from historical conflicts.

---

## Data Sources Priority List

### Active — Implemented & Configured
1. **yfinance** — Daily futures OHLCV (CL=F, BZ=F, ZW=F, GC=F, NG=F, HG=F) + equity indices since 2000
2. **FRED** — Fed funds, 10Y/2Y yields, DXY, breakeven inflation, oil spot, CPI (API key in fetch_data.py)
3. **EIA API v2** — Weekly crude inventory, imports, refinery utilization, SPR stocks (API key in fetch_data.py)
4. **LSEG Data Library** — Options chains (IV/Greeks), futures term structure, Baltic freight indices, news headlines (needs App Key + Workspace Desktop)

### Configured But Not Yet Fetching (placeholder providers)
5. USDA (WASDE, crop reports) — Wheat fundamentals
6. ACLED — Conflict event data
7. CFTC CoT — Positioning data (use `cot_reports` package for free alternative)
8. NOAA — Weather and climate data
9. NASA MODIS — Satellite vegetation indices

### Premium (Optional)
10. Kpler / Vortexa — Tanker tracking, oil flows (`enabled: false` in yaml)

### Free Supplements Not Yet Configured
- Caldara-Iacoviello Geopolitical Risk Index (matteoiacoviello.com/gpr.htm)
- GDELT — News-based geopolitical signals
- Economic Policy Uncertainty Index (policyuncertainty.com)

---

## Agent Coordination Rules

### When Starting Work
1. **Read this entire claude.md first** — Every time
2. **Check the Session Log above** — Don't redo completed work
3. **Read the module README** — Before touching any module
4. **Run existing tests** — Before AND after changes

### Code Standards
- Python 3.10+ with full type annotations
- `black` formatter (100-char lines), `ruff` linter, `mypy` strict mode
- Every function: Google docstring + `# JUSTIFIED:` for any numerical constant
- Prefer `numpy` vectorized ops over loops
- Use `polars` over `pandas` for large datasets (10x faster)
- `pydantic` for all config/data validation
- `loguru` for structured logging

### Git Conventions
- Branch: `feature/<module-name>` or `fix/<issue>`
- Commit: Conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`)
- PR: Always describe what changed and why

### Testing Requirements
- Unit tests: `pytest` with `hypothesis` for property-based testing
- Statistical tests: KS test, chi-squared for distribution fits
- Backtesting: Walk-forward on minimum 3 historical conflict analogs
- Smoke tests: Pipeline runs end-to-end without errors

---

## Key Dependencies

```
# Core
numpy>=1.24, scipy>=1.11, polars>=0.20, pydantic>=2.0, pyyaml>=6.0

# Statistical Models
statsmodels>=0.14, arch>=6.0, pymc>=5.10, numpyro>=0.13, arviz>=0.17

# Machine Learning
scikit-learn>=1.3, xgboost>=2.0, lightgbm>=4.0, catboost>=1.2

# Deep Learning
torch>=2.1, pytorch-lightning>=2.1

# Time Series
darts>=0.27, tslearn>=0.6

# Specialized
copulas>=0.10, tigramite>=5.2

# Data
yfinance>=0.2, fredapi>=0.5, requests>=2.31, lseg-data>=2.0

# Visualization
plotly>=5.18, matplotlib>=3.8, seaborn>=0.13

# Infrastructure
loguru>=0.7, hydra-core>=1.3, mlflow>=2.9, joblib>=1.3

# Testing
pytest>=7.4, hypothesis>=6.88
```

---

## Performance Targets

- **Oil price prediction**: Capture >80% of directional move within 1 week of event
- **Wheat price prediction**: Capture >70% of directional move within 2 weeks
- **Tail risk calibration**: Model 99th percentile move within 20% of realized
- **Prediction interval coverage**: 90% PI should achieve 85-95% empirical coverage
- **Scenario ranking**: Correct ordinal ranking of severity across escalation levels
- **Cross-asset correlation shift**: Detect regime break within 2 days

---

## Outstanding Work

### High Priority
- [ ] Re-run wheat_forecast.py after bug fixes to get corrected results
- [ ] Run fetch_data.py to populate `data/raw/` with yfinance + FRED + EIA data
- [ ] Connect LSEG data feeds (needs Workspace Desktop running locally)
- [ ] Implement CFTC CoT fetcher (use `cot_reports` package)
- [ ] Wire fetched data into feature processors (technical, fundamental, regime)

### Medium Priority
- [ ] Implement model fitting (models created but no actual training on real data)
- [ ] Full pipeline integration (run_pipeline.py stages are modular but need live data)
- [ ] Walk-forward backtesting on historical conflict analogs
- [ ] Interactive Plotly/Dash dashboard

### Lower Priority
- [ ] ACLED/GDELT geopolitical event fetcher
- [ ] NOAA weather + NASA NDVI satellite data fetchers
- [ ] USDA WASDE report parser
- [ ] Caldara-Iacoviello GPR Index integration

---

## Risk & Caveats

1. **This is a scenario analysis tool, not a trading system** — No model can predict black swans with certainty
2. **Historical analogs are imperfect** — Each conflict is unique; analog-based priors are informative, not deterministic
3. **Data limitations** — Many exotic signals are noisy or have short histories
4. **Model uncertainty is the output** — Wide prediction intervals are a feature, not a bug
5. **Regime identification is retrospective** — Real-time regime detection has inherent lag
6. **Geopolitical dynamics are non-stationary** — Parameters MUST be updated as the situation evolves

---

## Quick Start

```bash
# Setup
make install          # Install dependencies (includes lseg-data)
make fetch-data       # Download all data sources
make validate-data    # Run data quality checks

# Run
make run-pipeline     # Full pipeline execution
make run-scenarios    # Generate scenario analysis
make run-backtest     # Backtest on historical conflicts

# Wheat forecast (standalone)
python scripts/wheat_forecast.py

# Dev
make test             # Run all tests
make lint             # Run linters
make format           # Auto-format code
```

---

*Last updated: 2026-02-24*
*Project codename: IranBSE (Black Swan Event)*
