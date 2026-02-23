# IranBSE: Black Swan Event Modeling — US-Iran War Impact on Commodity Futures

## Project Overview

This project builds a **hyper-complex, top-1% statistical model** to predict the impact of a hypothetical 1-month US-Iran war (starting March 15, 2026) on:
- **Wheat futures** (CBOT Wheat — ZW)
- **Crude oil futures** (WTI — CL, Brent — BZ)

The model treats this as a **black swan scenario analysis** — an event with fat-tailed, nonlinear, regime-shifting effects on global commodity markets. The goal is not just point prediction but full distributional forecasting with uncertainty quantification.

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
│   ├── data_sources.yaml              # Data source registry
│   ├── scenarios.yaml                 # War scenario parameters
│   └── hyperparams.yaml              # Model hyperparameters
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetchers/                  # Data ingestion modules
│   │   │   ├── __init__.py
│   │   │   ├── base_fetcher.py        # Abstract base class
│   │   │   ├── commodity_prices.py    # Futures price data
│   │   │   ├── geopolitical.py        # Geopolitical risk indices
│   │   │   ├── shipping.py            # Shipping/freight data
│   │   │   ├── satellite.py           # Satellite imagery signals
│   │   │   ├── sentiment.py           # News/social sentiment
│   │   │   ├── macro.py               # Macroeconomic indicators
│   │   │   └── alternative.py         # Alternative/exotic data
│   │   ├── processors/               # Feature engineering
│   │   │   ├── __init__.py
│   │   │   ├── base_processor.py
│   │   │   ├── technical.py           # Technical indicators
│   │   │   ├── fundamental.py         # Supply/demand features
│   │   │   ├── cross_asset.py         # Cross-asset signals
│   │   │   ├── regime.py              # Regime detection features
│   │   │   └── geopolitical.py        # Geopolitical risk features
│   │   └── storage/
│   │       ├── __init__.py
│   │       └── data_store.py          # Unified data storage layer
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py                # Model registry (strategy pattern)
│   │   ├── base_model.py              # Abstract model interface
│   │   ├── ensemble/
│   │   │   ├── __init__.py
│   │   │   ├── meta_learner.py        # Stacking/blending ensemble
│   │   │   └── adversarial_ensemble.py # Adversarial validation ensemble
│   │   ├── statistical/
│   │   │   ├── __init__.py
│   │   │   ├── regime_switching.py    # Markov regime-switching models
│   │   │   ├── hawkes_process.py      # Hawkes process for event clustering
│   │   │   ├── extreme_value.py       # EVT — GEV/GPD tail modeling
│   │   │   ├── copula.py             # Copula models for dependency
│   │   │   ├── garch_family.py       # GARCH/EGARCH/GJR-GARCH/FIGARCH
│   │   │   └── var_svar.py           # Structural VAR models
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── gradient_boost.py     # XGBoost/LightGBM/CatBoost
│   │   │   ├── neural_sde.py         # Neural SDEs for price dynamics
│   │   │   ├── temporal_fusion.py    # Temporal Fusion Transformer
│   │   │   ├── normalizing_flows.py  # Normalizing flows for density est.
│   │   │   ├── deep_state_space.py   # Deep state space models (S4/Mamba)
│   │   │   └── conformal.py          # Conformal prediction wrappers
│   │   ├── bayesian/
│   │   │   ├── __init__.py
│   │   │   ├── hierarchical.py       # Hierarchical Bayesian models
│   │   │   ├── bvar.py               # Bayesian VAR with Minnesota prior
│   │   │   ├── scenario_tree.py      # Bayesian scenario tree
│   │   │   └── causal_impact.py      # Bayesian causal impact analysis
│   │   └── geopolitical/
│   │       ├── __init__.py
│   │       ├── conflict_model.py     # War escalation dynamics
│   │       ├── supply_disruption.py  # Supply chain disruption model
│   │       └── contagion.py          # Cross-market contagion model
│   ├── scenarios/
│   │   ├── __init__.py
│   │   ├── scenario_engine.py        # Monte Carlo scenario generator
│   │   ├── iran_war.py               # Iran war-specific scenario logic
│   │   ├── strait_of_hormuz.py       # Hormuz chokepoint model
│   │   └── historical_analogs.py     # Historical conflict analogs
│   ├── alpha/
│   │   ├── __init__.py
│   │   ├── oil/
│   │   │   ├── __init__.py
│   │   │   ├── contango_signals.py   # Term structure / contango-backwardation
│   │   │   ├── floating_storage.py   # Floating storage / tanker tracking
│   │   │   ├── refinery_margins.py   # Crack spreads / refinery signals
│   │   │   ├── opec_positioning.py   # OPEC spare capacity / compliance
│   │   │   ├── iran_export_tracking.py # Iran-specific export monitoring
│   │   │   └── dark_fleet.py         # Sanctioned tanker "dark fleet" signals
│   │   ├── wheat/
│   │   │   ├── __init__.py
│   │   │   ├── crop_conditions.py    # USDA crop condition reports
│   │   │   ├── weather_models.py     # Weather anomaly signals
│   │   │   ├── black_sea_risk.py     # Black Sea export disruption
│   │   │   ├── substitution.py       # Grain substitution dynamics
│   │   │   ├── fob_basis.py          # FOB basis / export competitiveness
│   │   │   └── food_security.py      # Food security panic buying signals
│   │   └── cross_asset/
│   │       ├── __init__.py
│   │       ├── dollar_dynamics.py    # USD strength / petrodollar flows
│   │       ├── volatility_surface.py # Vol surface / skew signals
│   │       ├── cot_positioning.py    # CFTC Commitment of Traders
│   │       ├── etf_flows.py          # Commodity ETF flow signals
│   │       └── options_flow.py       # Unusual options activity
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── tail_risk.py              # Tail risk metrics (CVaR, ES)
│   │   ├── stress_testing.py         # Historical + hypothetical stress tests
│   │   └── model_risk.py             # Model uncertainty quantification
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── walk_forward.py           # Walk-forward validation
│   │   ├── historical_conflicts.py   # Backtest on prior conflicts
│   │   └── metrics.py                # Custom evaluation metrics
│   └── visualization/
│       ├── __init__.py
│       ├── scenario_plots.py         # Scenario fan charts
│       ├── sensitivity.py            # Sensitivity analysis plots
│       └── dashboard.py              # Interactive dashboard
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_scenario_analysis.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_data/
│   ├── test_models/
│   ├── test_scenarios/
│   └── test_alpha/
├── scripts/
│   ├── run_pipeline.py               # Main execution pipeline
│   ├── fetch_data.py                 # Data download script
│   └── generate_report.py            # Generate analysis report
├── requirements.txt
├── pyproject.toml
└── Makefile
```

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

### Key Variables (Config-Driven)
```yaml
scenario:
  name: "US-Iran War March 2026"
  start_date: "2026-03-15"
  end_date: "2026-04-15"
  duration_days: 31

  escalation_levels:
    - level: 1
      name: "Limited Strikes"
      description: "Targeted strikes on nuclear/military facilities"
      probability: 0.45
      oil_supply_disruption_pct: [2, 8]      # % of global supply
      hormuz_closure_probability: 0.15
      wheat_trade_disruption_pct: [0, 3]

    - level: 2
      name: "Extended Air Campaign"
      description: "Sustained air operations, Iranian retaliation"
      probability: 0.35
      oil_supply_disruption_pct: [8, 20]
      hormuz_closure_probability: 0.55
      wheat_trade_disruption_pct: [3, 10]

    - level: 3
      name: "Full Theater Conflict"
      description: "Regional war with proxy involvement"
      probability: 0.15
      oil_supply_disruption_pct: [20, 40]
      hormuz_closure_probability: 0.85
      wheat_trade_disruption_pct: [10, 25]

    - level: 4
      name: "Global Escalation"
      description: "Multi-party conflict, severe trade disruption"
      probability: 0.05
      oil_supply_disruption_pct: [40, 60]
      hormuz_closure_probability: 0.95
      wheat_trade_disruption_pct: [25, 50]

  iran_specifics:
    daily_oil_production_mbpd: 3.2           # JUSTIFIED: EIA 2025 estimate
    hormuz_daily_flow_mbpd: 17.0             # JUSTIFIED: EIA Strait of Hormuz analysis
    hormuz_pct_global_oil_trade: 0.20        # JUSTIFIED: ~20% of global oil trade
    iran_wheat_import_dependency: 0.30       # JUSTIFIED: FAO food balance sheets
    iran_strategic_oil_reserve_days: 15      # JUSTIFIED: estimated from storage capacity

  historical_analogs:
    - event: "Gulf War I (1990-91)"
      oil_peak_pct_change: 140
      wheat_peak_pct_change: 15
      duration_to_peak_days: 60
    - event: "Iraq War (2003)"
      oil_peak_pct_change: 37
      wheat_peak_pct_change: 8
      duration_to_peak_days: 14
    - event: "Libya Civil War (2011)"
      oil_peak_pct_change: 25
      wheat_peak_pct_change: 5
      duration_to_peak_days: 45
    - event: "Russia-Ukraine (2022)"
      oil_peak_pct_change: 65
      wheat_peak_pct_change: 70
      duration_to_peak_days: 21
    - event: "Iran-Iraq War Start (1980)"
      oil_peak_pct_change: 110
      wheat_peak_pct_change: 10
      duration_to_peak_days: 90
    - event: "Soleimani Strike (2020)"
      oil_peak_pct_change: 4
      wheat_peak_pct_change: 1
      duration_to_peak_days: 1
```

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

### Free / Open Source
1. Yahoo Finance / yfinance — Historical futures prices
2. FRED (Federal Reserve) — Macro indicators, rates
3. USDA (WASDE, crop reports) — Wheat fundamentals
4. EIA (Energy Information Admin) — Oil supply/demand/inventories
5. ACLED (Armed Conflict Location) — Conflict event data
6. GDELT (Global Event Database) — News-based geopolitical signals
7. ERA5 / NOAA — Weather and climate data
8. NASA MODIS / Sentinel — Satellite vegetation indices
9. CFTC CoT — Positioning data
10. World Bank / IMF — Macro/trade data

### Premium (If Available)
11. Refinitiv/LSEG — Tick data, fundamentals, estimates
12. Bloomberg — Terminal data
13. Kpler / Vortexa — Tanker tracking, oil flows
14. Planet Labs — High-frequency satellite imagery
15. Predata — Geopolitical risk signals
16. Orbital Insight — Tank farm satellite analytics

---

## Agent Coordination Rules

### When Starting Work
1. **Read this entire claude.md first** — Every time
2. **Check the TODO list** — Don't duplicate work
3. **Read the module README** — Before touching any module
4. **Run existing tests** — Before AND after changes

### Code Standards
- Python 3.10+ with full type annotations
- `black` formatter, `ruff` linter, `mypy` strict mode
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
numpy>=1.24
scipy>=1.11
polars>=0.20
pydantic>=2.0

# Statistical Models
statsmodels>=0.14
arch>=6.0               # GARCH family
pymc>=5.10              # Bayesian inference
numpyro>=0.13           # Fast Bayesian (JAX-based)
arviz>=0.17             # Bayesian diagnostics

# Machine Learning
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
catboost>=1.2

# Deep Learning
torch>=2.1
pytorch-lightning>=2.1

# Time Series
darts>=0.27             # Unified forecasting API
gluonts>=0.14           # Probabilistic forecasting
tslearn>=0.6            # Time series clustering

# Specialized
tick>=0.7               # Hawkes processes
copulas>=0.10           # Copula models
tigramite>=5.2          # Causal discovery (PCMCI+)
POT>=0.9                # Extreme Value Theory

# Data
yfinance>=0.2
fredapi>=0.5
requests>=2.31

# Visualization
plotly>=5.18
matplotlib>=3.8
seaborn>=0.13

# Infrastructure
loguru>=0.7
hydra-core>=1.3         # Config management
mlflow>=2.9             # Experiment tracking
joblib>=1.3             # Parallel execution
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
make install          # Install dependencies
make fetch-data       # Download all data sources
make validate-data    # Run data quality checks

# Run
make run-pipeline     # Full pipeline execution
make run-scenarios    # Generate scenario analysis
make run-backtest     # Backtest on historical conflicts

# Dev
make test             # Run all tests
make lint             # Run linters
make format           # Auto-format code
```

---

*Last updated: 2026-02-22*
*Project codename: IranBSE (Black Swan Event)*
