# Rule-Based Direct Policy Optimisation (RBDPO)
## Implementation Plan — v1.0

*Step-by-step guide for building the production package from scratch*

---

> **MVP Scope Notice.** This plan covers the v1.0 MVP: single-instrument, OHLCV-only strategy discovery. Two planned extensions are scoped but not yet implemented: **(1)** external time-series data as rule inputs (v2.0); and **(2)** a portfolio-level optimisation loop for multi-instrument joint strategy discovery (v3.0). Architectural placeholders for these extensions are noted where relevant, and a dedicated section is included at the end of this document. All extension work is contingent on MVP validation.

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Environment Setup](#2-environment-setup)
3. [Phase 1 — Data Layer](#3-phase-1--data-layer)
4. [Phase 2 — Indicator Precomputation Engine](#4-phase-2--indicator-precomputation-engine)
5. [Phase 3 — Policy Representation](#5-phase-3--policy-representation)
6. [Phase 4 — Rule Evaluation Engine](#6-phase-4--rule-evaluation-engine)
7. [Phase 5 — Stop-Loss and Take-Profit Engine](#7-phase-5--stop-loss-and-take-profit-engine)
8. [Phase 6 — Backtesting Integration (vectorbt)](#8-phase-6--backtesting-integration-vectorbt)
9. [Phase 7 — Objective Function](#9-phase-7--objective-function)
10. [Phase 8 — Search Space Encoding (ConfigSpace)](#10-phase-8--search-space-encoding-configspace)
11. [Phase 9 — Optimiser Integration](#11-phase-9--optimiser-integration)
12. [Phase 10 — Walk-Forward Validation](#12-phase-10--walk-forward-validation)
13. [Phase 11 — Results and Reporting](#13-phase-11--results-and-reporting)
14. [Phase 12 — CLI and Configuration Layer](#14-phase-12--cli-and-configuration-layer)
15. [Phase 13 — Testing Suite](#15-phase-13--testing-suite)
16. [Future Extension Placeholders](#16-future-extension-placeholders)
17. [Dependency Reference](#17-dependency-reference)
18. [Implementation Order Summary](#18-implementation-order-summary)

---

## 1. Repository Structure

Create the following directory and file layout before writing any code. Every module must have a corresponding `__init__.py`.

```
rbdpo/
├── pyproject.toml
├── uv.lock                     # Committed lock file for reproducibility
├── README.md
├── CHANGELOG.md
├── .python-version             # Pins Python version for uv
├── configs/
│   └── default.yaml            # Default run configuration
├── examples/
│   └── quickstart.ipynb        # Getting started notebook
├── rbdpo/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # OHLCV data loading and validation
│   │   └── preprocessor.py     # Cleaning, resampling, train/test split
│   ├── indicators/
│   │   ├── __init__.py
│   │   ├── precompute.py       # Master precomputation orchestrator
│   │   ├── registry.py         # Indicator definitions and param spaces
│   │   └── cache.py            # In-memory indicator store
│   │   # v2.0 PLACEHOLDER: external_series.py — external time-series registry
│   ├── policy/
│   │   ├── __init__.py
│   │   ├── schema.py           # Policy dataclass definitions
│   │   ├── rules.py            # Rule archetype implementations
│   │   ├── sl_tp.py            # Stop-loss / take-profit implementations
│   │   └── decoder.py          # Convert optimiser config → Policy object
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py           # vectorbt backtest runner
│   │   └── metrics.py          # Performance metric calculations
│   ├── objective/
│   │   ├── __init__.py
│   │   └── function.py         # Named objectives + regularisation
│   ├── search/
│   │   ├── __init__.py
│   │   ├── space.py            # ConfigSpace definition
│   │   ├── dehb_runner.py      # DEHB optimiser wrapper
│   │   └── nevergrad_runner.py # Nevergrad optimiser wrapper
│   ├── validation/
│   │   ├── __init__.py
│   │   └── walk_forward.py     # Walk-forward orchestration
│   └── reporting/
│       ├── __init__.py
│       └── report.py           # Results serialisation and human-readable output
│   # v3.0 PLACEHOLDER: portfolio/ — portfolio-level optimisation loop
└── tests/
    ├── conftest.py
    ├── test_indicators.py
    ├── test_rules.py
    ├── test_backtest.py
    ├── test_objective.py
    └── test_search_space.py
```

---

## 2. Environment Setup

### 2.1 Python Version

Use Python 3.11. Pin it for the project:

```bash
echo "3.11" > .python-version
```

### 2.2 Install uv

`uv` is the project's sole environment and dependency manager. It replaces pip, virtualenv, and pip-tools entirely.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:

```bash
uv --version
```

### 2.3 Initialise the Project

```bash
uv init rbdpo
cd rbdpo
```

This creates `pyproject.toml`, `.python-version`, and a basic project scaffold.

### 2.4 Add Dependencies

Add all dependencies with a single command. `uv` resolves, downloads, and locks them:

```bash
uv add vectorbt pandas numpy pandas-ta ConfigSpace DEHB nevergrad joblib pyyaml click
uv add --dev pytest pytest-cov
```

`uv` writes a `uv.lock` file containing the fully resolved, reproducible dependency tree. Commit this file to version control.

**Note: No TA-Lib.** TA-Lib is explicitly excluded. It requires a compiled C library (`libta-lib`) that varies across operating systems, breaks reproducible containerised environments, and is incompatible with `uv`'s fully managed environment model. `pandas-ta` is a pure-Python library that covers all indicators required by RBDPO and installs without any system prerequisites.

### 2.5 pyproject.toml

`uv init` creates a `pyproject.toml`. Extend it with the following sections:

```toml
[project]
name = "rbdpo"
version = "0.1.0"
description = "Rule-Based Direct Policy Optimisation for trading strategy discovery"
requires-python = ">=3.11"
dependencies = [
    "vectorbt",
    "pandas>=2.0",
    "numpy>=1.26",
    "pandas-ta>=0.3.14b",
    "ConfigSpace>=0.7",
    "DEHB>=0.0.6",
    "nevergrad>=1.0",
    "joblib>=1.3",
    "pyyaml>=6.0",
    "click>=8.1",
]

[project.scripts]
rbdpo = "rbdpo.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Install the package in editable mode so imports resolve during development:

```bash
uv pip install -e .
```

### 2.6 Running Commands

All commands are run inside the `uv`-managed environment. Prefix any command with `uv run` to avoid activating the environment manually:

```bash
uv run rbdpo run --data data/SPY.csv --config configs/default.yaml
uv run pytest tests/
uv run python examples/quickstart.py
```

Alternatively, activate the environment once for a session:

```bash
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```

---

## 3. Phase 1 — Data Layer

### 3.1 `rbdpo/data/loader.py`

Implement a `load_ohlcv(source, freq=None)` function.

**Inputs:** A file path (CSV or Parquet) or a pandas DataFrame passed directly. `freq` is an optional target resampling frequency string (e.g. `'1D'`, `'1H'`).

**Outputs:** A validated pandas DataFrame with a `DatetimeIndex` and lowercase columns: `open`, `high`, `low`, `close`, `volume`. The index must be timezone-naive and monotonically increasing.

**Validation checks:**
- All five columns are present (raise `ValueError` if not)
- No NaN in `close`, `high`, `low` (volume NaN filled with 0)
- `high >= low` at every row
- `high >= close` and `low <= close` at every row
- Index is sorted ascending
- Minimum row count check (configurable, default: 500 bars)

### 3.2 `rbdpo/data/preprocessor.py`

Implement `preprocess(data, config)`:

- **Resampling:** If `freq` is specified, resample using OHLCV aggregation: `open=first`, `high=max`, `low=min`, `close=last`, `volume=sum`.
- **Train/test split:** Accept a split date or a float (fraction for in-sample use). Return `(train_df, test_df)`.

Both returned DataFrames share the same column structure and pass validation.

> **v2.0 Note:** This module will be extended to accept and align external time-series data alongside OHLCV data.

---

## 4. Phase 2 — Indicator Precomputation Engine

### 4.1 `rbdpo/indicators/registry.py`

Define every indicator as a Python dataclass:

```python
@dataclass
class IndicatorDef:
    name: str                   # unique identifier, e.g. 'bb', 'rsi'
    param_grid: dict            # {param_name: [list of values to precompute]}
    outputs: list[str]          # named outputs, e.g. ['upper', 'middle', 'lower']
    compute_fn: Callable        # fn(data: pd.DataFrame, **params) -> dict[str, np.ndarray]
    constraints: Callable | None = None  # optional fn to filter invalid param combos
```

Define the following indicators using `pandas-ta` as the compute backend throughout. Do not use TA-Lib.

| Name | Parameters to grid | Output names | pandas-ta call |
|---|---|---|---|
| `sma` | `period: [5,8,10,13,15,20,25,30,40,50,75,100,150,200]` | `['ma']` | `ta.sma(close, length=period)` |
| `ema` | `period: [5,8,10,13,15,20,25,30,40,50,75,100,150,200]` | `['ma']` | `ta.ema(close, length=period)` |
| `wma` | `period: [10,20,50,100,200]` | `['ma']` | `ta.wma(close, length=period)` |
| `rsi` | `period: [7,9,10,12,14,16,21]` | `['rsi']` | `ta.rsi(close, length=period)` |
| `macd` | `fast: [8,10,12,15]`, `slow: [21,24,26,30]`, `signal: [7,9,12]`; constraint: fast < slow | `['macd','signal','hist']` | `ta.macd(close, fast=fast, slow=slow, signal=signal)` |
| `bb` | `period: [10,15,20,25,30,40,50]`, `std: [1.5,1.75,2.0,2.25,2.5,3.0]`, `ma_type: ['sma','ema']` | `['upper','middle','lower']` | `ta.bbands(close, length=period, std=std, mamode=ma_type)` |
| `atr` | `period: [7,10,14,20]` | `['atr']` | `ta.atr(high, low, close, length=period)` |
| `keltner` | `ema_period: [10,15,20,30]`, `atr_period: [10,14,20]`, `mult: [1.5,2.0,2.5]` | `['upper','lower']` | `ta.kc(high, low, close, length=ema_period, atr_length=atr_period, scalar=mult)` |
| `donchian` | `period: [10,20,30,50,100]` | `['upper','lower']` | `ta.donchian(high, low, length=period)` |
| `adx` | `period: [10,14,20]` | `['adx']` | `ta.adx(high, low, close, length=period)` |
| `stoch` | `k: [5,9,14,21]`, `d: [3,5,7]`, `smooth: [3,5]` | `['k','d']` | `ta.stoch(high, low, close, k=k, d=d, smooth_k=smooth)` |
| `psar` | `step: [0.01,0.02,0.05]`, `max_step: [0.1,0.2,0.3]` | `['psar']` | `ta.psar(high, low, af0=step, max_af=max_step)` |
| `cci` | `period: [10,14,20]` | `['cci']` | `ta.cci(high, low, close, length=period)` |
| `roc` | `period: [5,10,14,20]` | `['roc']` | `ta.roc(close, length=period)` |
| `obv` | (none) | `['obv']` | `ta.obv(close, volume)` |
| `vwap` | (none) | `['vwap']` | `ta.vwap(high, low, close, volume)` |

**MACD constraint:** only compute combinations where `fast < slow`. Implement via:

```python
constraints = lambda params: params['fast'] < params['slow']
```

Apply this filter in the precomputation orchestrator when generating the Cartesian product of `param_grid` values.

> **v2.0 Note:** This registry will be extended with an `ExternalSeriesDef` class for macroeconomic and alternative time-series data.

### 4.2 `rbdpo/indicators/precompute.py`

Implement `precompute_all(data, registry, n_jobs=-1) -> IndicatorCache`:

1. Iterate over every indicator in the registry.
2. Generate the full Cartesian product of `param_grid` values; apply `constraints` filter if present.
3. Compute each combination. Use `joblib.Parallel(n_jobs=n_jobs)` to parallelise across combinations within each indicator.
4. Store results in the cache.
5. Log total memory usage after completion.

**Error handling:** If an individual combination fails (e.g. insufficient data for a given period), log a warning and store `None`. The rule evaluation engine must handle `None` gracefully by returning a `False` array.

### 4.3 `rbdpo/indicators/cache.py`

Implement `IndicatorCache`:

```python
class IndicatorCache:
    def store(self, indicator_name: str, params: tuple, output_name: str, array: np.ndarray) -> None:
        """Store a precomputed 1D array indexed by (indicator, params, output)."""

    def get(self, indicator_name: str, params: tuple, output_name: str) -> np.ndarray | None:
        """Retrieve a precomputed array. Returns None if not found."""

    def memory_usage_gb(self) -> float:
        """Total memory occupied by all stored arrays, in GB."""

    def available_params(self, indicator_name: str) -> list[tuple]:
        """All precomputed parameter tuples for a given indicator."""
```

Internal storage: a nested dict `{indicator_name: {(params_tuple, output_name): np.ndarray}}`. Dict lookups are O(1) and negligible versus backtest time. Store arrays as `float32` where precision allows — this halves memory usage with no meaningful impact on indicator accuracy.

---

## 5. Phase 3 — Policy Representation

### 5.1 `rbdpo/policy/schema.py`

Define all policy components as frozen Python dataclasses (hashable for caching).

```python
@dataclass(frozen=True)
class RuleConfig:
    archetype: str          # 'comparison', 'crossover', 'band_test', 'pattern',
                            #  'stat_test', 'derivative'
    indicator: str          # e.g. 'rsi', 'bb', 'sma'
    params: tuple           # e.g. (14,) or (20, 2.0, 'sma')
    output: str             # e.g. 'rsi', 'upper', 'ma'
    operator: str           # '>', '<', 'crosses_above', 'crosses_below'
    comparand: str | float  # 'price', 'indicator2', or numeric threshold
    indicator2: str | None = None
    params2: tuple | None = None
    output2: str | None = None
    lookback: int | None = None
    band_side: str | None = None  # 'upper' or 'lower' for band_test archetype

@dataclass(frozen=True)
class SLConfig:
    sl_type: str   # 'pct', 'atr_mult', 'swing_low', 'bb_lower', 'trailing_atr'
    params: tuple

@dataclass(frozen=True)
class TPConfig:
    tp_type: str   # 'pct', 'atr_mult', 'risk_reward', 'swing_high', 'bb_upper'
    params: tuple

@dataclass(frozen=True)
class Policy:
    entry_rules: tuple[RuleConfig, ...]
    exit_rules: tuple[RuleConfig, ...]
    sl_config: SLConfig
    tp_config: TPConfig
    n_active_entry: int
    n_active_exit: int
```

### 5.2 `rbdpo/policy/decoder.py`

Implement `decode_config(config: dict, cache: IndicatorCache) -> Policy | None`.

Receives a flat dictionary from the optimiser (sampled from ConfigSpace) and converts it to a `Policy` object:

1. Extract `rule_{i}_active` flags. Skip inactive slots.
2. For each active slot, read archetype, indicator, params.
3. Validate that the required indicator/param combination exists in the cache. Return `None` if not.
4. Construct and return the `Policy` dataclass.

This function is called on every optimiser iteration and must contain no loops over data arrays.

---

## 6. Phase 4 — Rule Evaluation Engine

### 6.1 `rbdpo/policy/rules.py`

Implement `evaluate_rule(rule: RuleConfig, cache: IndicatorCache, price: np.ndarray) -> np.ndarray` where the output is a boolean NumPy array of shape `(T,)`.

Implement each archetype:

**Comparison — `indicator > threshold`:**
```python
arr = cache.get(rule.indicator, rule.params, rule.output)
return arr > rule.comparand
```

**Comparison — `indicator1 > indicator2`:**
```python
arr1 = cache.get(rule.indicator, rule.params, rule.output)
arr2 = cache.get(rule.indicator2, rule.params2, rule.output2)
return arr1 > arr2
```

**Comparison — `price > indicator`:**
```python
arr = cache.get(rule.indicator, rule.params, rule.output)
return price > arr
```

**Crossover — `indicator crosses_above threshold`:**
```python
arr = cache.get(rule.indicator, rule.params, rule.output)
prev = np.roll(arr, 1)
signal = (arr > rule.comparand) & (prev <= rule.comparand)
signal[0] = False   # mask the rolled first element
return signal
```

**Crossover — `indicator1 crosses_above indicator2`:**
```python
arr1 = cache.get(rule.indicator, rule.params, rule.output)
arr2 = cache.get(rule.indicator2, rule.params2, rule.output2)
p1, p2 = np.roll(arr1, 1), np.roll(arr2, 1)
signal = (arr1 > arr2) & (p1 <= p2)
signal[0] = False
return signal
```

**Band test — `price above band_upper`:**
```python
upper = cache.get(rule.indicator, rule.params, 'upper')
return price > upper
```

**Pattern — `price at N-bar high`:**
```python
rolling_max = pd.Series(price).rolling(rule.lookback).max().values
return price >= rolling_max
```

**Pattern — `consecutive(condition, count)`:** Apply rolling sum; return `rolling_sum >= count`.

**Derivative — `slope(indicator) > 0`:**
```python
arr = cache.get(rule.indicator, rule.params, rule.output)
return np.diff(arr, prepend=arr[0]) > 0
```

**Statistical — `zscore > threshold`:**
```python
arr = cache.get(rule.indicator, rule.params, rule.output)
s = pd.Series(arr)
zscore = (s - s.rolling(rule.lookback).mean()) / (s.rolling(rule.lookback).std() + 1e-8)
return zscore.values > rule.comparand
```

**Critical:** All archetype functions must treat NaN-leading array positions (from indicator warm-up) as `False`. Apply `np.nan_to_num(arr, nan=0.0)` before boolean operations where needed.

### 6.2 Combining Rules

Implement `combine_rules(rules, cache, price) -> np.ndarray`:

```python
signals = [evaluate_rule(r, cache, price) for r in rules]
return np.logical_and.reduce(signals)
```

---

## 7. Phase 5 — Stop-Loss and Take-Profit Engine

### 7.1 `rbdpo/policy/sl_tp.py`

Implement functions that compute SL and TP price levels as 1D arrays of shape `(T,)`, given the current close price at each bar. These arrays are passed to vectorbt's stop generators.

**`compute_sl(sl_config, cache, price, high, low) -> np.ndarray`:**

| Type | Implementation |
|---|---|
| `pct` | `price * (1 - pct)` |
| `atr_mult` | `price - mult * cache.get('atr', (period,), 'atr')` |
| `swing_low` | `pd.Series(low).rolling(lookback).min().values` |
| `bb_lower` | `cache.get('bb', params, 'lower')` |
| `trailing_atr` | `price - mult * atr` (initial level; vectorbt handles the trailing update) |

**`compute_tp(tp_config, cache, price, high, sl_levels) -> np.ndarray`:**

| Type | Implementation |
|---|---|
| `pct` | `price * (1 + pct)` |
| `atr_mult` | `price + mult * cache.get('atr', (period,), 'atr')` |
| `risk_reward` | `price + rr * (price - sl_levels)` |
| `swing_high` | `pd.Series(high).rolling(lookback).max().values` |
| `bb_upper` | `cache.get('bb', params, 'upper')` |

---

## 8. Phase 6 — Backtesting Integration (vectorbt)

### 8.1 `rbdpo/backtest/engine.py`

Implement `run_backtest(policy, cache, data, config) -> dict`.

Step-by-step:

1. Extract `close`, `high`, `low` as NumPy arrays.
2. Call `combine_rules(policy.entry_rules, cache, close)` → `entry_signals`.
3. If `policy.exit_rules` is non-empty, call `combine_rules(policy.exit_rules, cache, close)` → `exit_signals`. Otherwise `exit_signals = None`.
4. Call `compute_sl(policy.sl_config, ...)` → `sl_levels`.
5. Call `compute_tp(policy.tp_config, ...)` → `tp_levels`.
6. Convert all arrays to `pd.Series` with the DatetimeIndex from `data` (vectorbt requires this).
7. Build SL exits: `vbt.STOPLOSS.run(close, entries=entry_signals, stop=sl_pct, trailing=is_trailing).exits`.
8. Build TP exits: `vbt.TAKEPROFIT.run(close, entries=entry_signals, stop=tp_pct).exits`.
9. Combine: `combined_exits = sl_exits | tp_exits | (exit_signals if exit_signals is not None else False)`.
10. Run portfolio: `vbt.Portfolio.from_signals(close, entries=entry_signals, exits=combined_exits, freq=freq, init_cash=init_cash, size=np.inf, max_size=1, fees=fee_rate, slippage=slippage_rate)`.
11. Return `extract_metrics(portfolio)`.

**Configuration parameters (via `config` dict):**

| Key | Default | Description |
|---|---|---|
| `init_cash` | 10000.0 | Initial capital |
| `fee_rate` | 0.001 | Flat fee per trade (10bps) |
| `slippage` | 0.0005 | Slippage per trade |
| `freq` | `'1D'` | Data frequency string |

### 8.2 `rbdpo/backtest/metrics.py`

Implement `extract_metrics(portfolio) -> dict`:

```python
{
    'total_return':          portfolio.total_return(),
    'sharpe_ratio':          portfolio.sharpe_ratio(),
    'calmar_ratio':          portfolio.calmar_ratio(),
    'sortino_ratio':         portfolio.sortino_ratio(),
    'max_drawdown':          portfolio.max_drawdown(),
    'n_trades':              portfolio.trades.count(),
    'win_rate':              portfolio.trades.win_rate(),
    'avg_return_per_trade':  portfolio.trades.returns.mean(),
}
```

Replace NaN and infinity values with `-999.0` as a sentinel (signals degenerate strategy to the objective function).

---

## 9. Phase 7 — Objective Function

### 9.1 `rbdpo/objective/function.py`

Implement `compute_objective(metrics, policy, config) -> float`.

The function returns a **single float**. The optimiser minimises this value; objectives are negated internally.

**Supported named objectives (selected via `config['objective']`):**

```python
OBJECTIVES = {
    'sharpe':     lambda m: -m['sharpe_ratio'],
    'calmar':     lambda m: -m['calmar_ratio'],
    'sortino':    lambda m: -m['sortino_ratio'],
    'max_return': lambda m: -m['total_return'],
    'composite':  lambda m, w: -(w[0]*m['sharpe_ratio']
                                + w[1]*m['calmar_ratio']
                                + w[2]*m['total_return']
                                - w[3]*abs(m['max_drawdown'])),
}
```

**Full implementation:**

```python
def compute_objective(metrics, policy, config):
    objective_key = config.get('objective', 'sharpe')
    lam = config.get('complexity_penalty', 0.02)
    min_trades = config.get('min_trades', 10)

    # Guard: penalise degenerate strategies
    if metrics['n_trades'] < min_trades:
        return 999.0

    # Compute base objective
    if objective_key == 'composite':
        weights = config.get('composite_weights', [0.5, 0.3, 0.1, 0.1])
        raw = OBJECTIVES['composite'](metrics, weights)
    else:
        raw = OBJECTIVES[objective_key](metrics)

    # Complexity regularisation (applied before negation is already done above)
    n_rules = policy.n_active_entry + policy.n_active_exit
    return raw + lam * n_rules
```

---

## 10. Phase 8 — Search Space Encoding (ConfigSpace)

### 10.1 `rbdpo/search/space.py`

Implement `build_config_space(n_entry_rules, n_exit_rules, registry) -> ConfigurationSpace`.

Use the `ConfigSpace` library throughout. The space is **conditional**: each choice opens up different sub-spaces depending on which indicator is selected.

**For each rule slot `i` in `range(n_entry_rules)` (repeat analogously for exit slots):**

```python
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter,
                        UniformIntegerHyperparameter, UniformFloatHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition

cs = ConfigurationSpace()

# --- Slot activation ---
active = CategoricalHyperparameter(f'entry_{i}_active', [True, False])
cs.add_hyperparameter(active)

# --- Archetype (conditional on active) ---
archetype = CategoricalHyperparameter(f'entry_{i}_archetype',
    ['comparison', 'crossover', 'band_test', 'pattern', 'stat_test', 'derivative'])
cs.add_hyperparameter(archetype)
cs.add_condition(EqualsCondition(archetype, active, True))

# --- Indicator (conditional on active) ---
indicator = CategoricalHyperparameter(f'entry_{i}_indicator',
    ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr', 'keltner', 'donchian',
     'adx', 'stoch', 'psar', 'cci', 'roc'])
cs.add_hyperparameter(indicator)
cs.add_condition(EqualsCondition(indicator, active, True))

# --- Per-indicator parameters (conditional on indicator choice) ---

# RSI
rsi_period = UniformIntegerHyperparameter(f'entry_{i}_rsi_period', lower=7, upper=21)
cs.add_hyperparameter(rsi_period)
cs.add_condition(EqualsCondition(rsi_period, indicator, 'rsi'))

# Bollinger Bands
bb_period = UniformIntegerHyperparameter(f'entry_{i}_bb_period', lower=10, upper=50)
bb_std    = UniformFloatHyperparameter(f'entry_{i}_bb_std', lower=1.5, upper=3.0)
bb_matype = CategoricalHyperparameter(f'entry_{i}_bb_matype', ['sma', 'ema'])
bb_output = CategoricalHyperparameter(f'entry_{i}_bb_output', ['upper', 'middle', 'lower'])
for hp in [bb_period, bb_std, bb_matype, bb_output]:
    cs.add_hyperparameter(hp)
    cs.add_condition(EqualsCondition(hp, indicator, 'bb'))

# MACD
macd_fast   = UniformIntegerHyperparameter(f'entry_{i}_macd_fast', lower=8, upper=15)
macd_slow   = UniformIntegerHyperparameter(f'entry_{i}_macd_slow', lower=21, upper=30)
macd_signal = UniformIntegerHyperparameter(f'entry_{i}_macd_signal', lower=7, upper=12)
macd_output = CategoricalHyperparameter(f'entry_{i}_macd_output', ['macd', 'signal', 'hist'])
for hp in [macd_fast, macd_slow, macd_signal, macd_output]:
    cs.add_hyperparameter(hp)
    cs.add_condition(EqualsCondition(hp, indicator, 'macd'))

# SMA / EMA (same structure, conditioned separately)
for ind_name in ['sma', 'ema', 'wma']:
    ma_period = UniformIntegerHyperparameter(f'entry_{i}_{ind_name}_period', lower=5, upper=200)
    cs.add_hyperparameter(ma_period)
    cs.add_condition(EqualsCondition(ma_period, indicator, ind_name))

# ATR
atr_period = UniformIntegerHyperparameter(f'entry_{i}_atr_period', lower=7, upper=21)
cs.add_hyperparameter(atr_period)
cs.add_condition(EqualsCondition(atr_period, indicator, 'atr'))

# Keltner Channel
kc_ema   = UniformIntegerHyperparameter(f'entry_{i}_kc_ema_period', lower=10, upper=30)
kc_atr   = UniformIntegerHyperparameter(f'entry_{i}_kc_atr_period', lower=10, upper=20)
kc_mult  = UniformFloatHyperparameter(f'entry_{i}_kc_mult', lower=1.5, upper=2.5)
kc_side  = CategoricalHyperparameter(f'entry_{i}_kc_side', ['upper', 'lower'])
for hp in [kc_ema, kc_atr, kc_mult, kc_side]:
    cs.add_hyperparameter(hp)
    cs.add_condition(EqualsCondition(hp, indicator, 'keltner'))

# Stochastic
st_k = UniformIntegerHyperparameter(f'entry_{i}_stoch_k', lower=5, upper=21)
st_d = UniformIntegerHyperparameter(f'entry_{i}_stoch_d', lower=3, upper=7)
st_s = UniformIntegerHyperparameter(f'entry_{i}_stoch_smooth', lower=3, upper=5)
st_out = CategoricalHyperparameter(f'entry_{i}_stoch_output', ['k', 'd'])
for hp in [st_k, st_d, st_s, st_out]:
    cs.add_hyperparameter(hp)
    cs.add_condition(EqualsCondition(hp, indicator, 'stoch'))

# (Add remaining indicators following the same pattern: ADX, Donchian, PSAR, CCI, ROC)

# --- Operator ---
operator = CategoricalHyperparameter(f'entry_{i}_operator',
    ['>', '<', 'crosses_above', 'crosses_below'])
cs.add_hyperparameter(operator)
cs.add_condition(EqualsCondition(operator, active, True))

# --- Threshold (for comparison-to-value archetypes) ---
threshold = UniformFloatHyperparameter(f'entry_{i}_threshold', lower=0.0, upper=100.0)
cs.add_hyperparameter(threshold)
cs.add_condition(InCondition(threshold, archetype, ['comparison', 'stat_test']))
```

**For the SL block:**

```python
sl_type = CategoricalHyperparameter('sl_type', ['pct', 'atr_mult', 'swing_low', 'bb_lower', 'trailing_atr'])
sl_pct   = UniformFloatHyperparameter('sl_pct', lower=0.005, upper=0.10)
sl_atr_mult   = UniformFloatHyperparameter('sl_atr_mult', lower=1.0, upper=4.0)
sl_atr_period = CategoricalHyperparameter('sl_atr_period', [7, 10, 14, 20])
sl_swing_lookback = UniformIntegerHyperparameter('sl_swing_lookback', lower=5, upper=50)
cs.add_hyperparameters([sl_type, sl_pct, sl_atr_mult, sl_atr_period, sl_swing_lookback])
cs.add_condition(EqualsCondition(sl_pct, sl_type, 'pct'))
cs.add_condition(InCondition(sl_atr_mult, sl_type, ['atr_mult', 'trailing_atr']))
cs.add_condition(InCondition(sl_atr_period, sl_type, ['atr_mult', 'trailing_atr']))
cs.add_condition(EqualsCondition(sl_swing_lookback, sl_type, 'swing_low'))
```

Repeat the same pattern for the TP block.

### 10.2 Validation

Implement `sample_and_validate(cs, n_samples=10)` to verify that sampled configurations decode without errors. Run as a smoke test before starting any optimisation run.

---

## 11. Phase 9 — Optimiser Integration

### 11.1 DEHB Runner — `rbdpo/search/dehb_runner.py`

Implement `run_dehb(objective_fn, config_space, budget, min_fidelity, max_fidelity, n_workers, seed) -> OptimisationResult`.

**DEHB fidelity mapping:** Map the fidelity float to a data subset length:

```python
n_bars = int(fidelity * len(full_data))
data_subset = full_data.iloc[-n_bars:]  # use the most recent N bars
```

**DEHB objective wrapper:**

```python
def dehb_objective(config: dict, fidelity: float, **kwargs) -> dict:
    data_subset = full_data.iloc[-int(fidelity * len(full_data)):]
    policy = decode_config(config, cache)
    if policy is None:
        return {'fitness': 999.0, 'cost': fidelity}
    metrics = run_backtest(policy, cache, data_subset, backtest_config)
    score = compute_objective(metrics, policy, obj_config)
    return {'fitness': score, 'cost': fidelity}
```

**DEHB initialisation with seed:**

```python
from dehb import DEHB
optimizer = DEHB(
    f=dehb_objective,
    cs=config_space,
    dimensions=len(config_space.get_hyperparameters()),
    min_fidelity=min_fidelity,   # e.g. 0.1
    max_fidelity=max_fidelity,   # e.g. 1.0
    eta=3,                       # Hyperband halving factor
    n_workers=n_workers,
    seed=seed                    # ← reproducibility / diversity control
)
history = optimizer.run(total_cost=budget)
```

**Seed usage note.** The same `seed` with the same configuration guarantees an identical sequence of evaluations. Running with `seed=42`, `seed=123`, `seed=999` etc. explores different regions of the search space and produces qualitatively different strategies — useful for building a diverse strategy ensemble.

### 11.2 Nevergrad Runner — `rbdpo/search/nevergrad_runner.py`

Implement `run_nevergrad(objective_fn, config_space, budget, seed) -> OptimisationResult`.

Mirror the ConfigSpace definition into `ng.p.Instrumentation` (one `ng.p.Choice` per categorical, one `ng.p.Scalar` per continuous). Use `ng.optimizers.NGOpt` as the default.

**Seed setting:**

```python
import nevergrad as ng
import numpy as np

optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=budget)
optimizer.parametrization.random_state = np.random.RandomState(seed)
```

**Run loop:**

```python
for _ in range(budget):
    x = optimizer.ask()
    loss = objective_fn(x.value)
    optimizer.tell(x, loss)
recommendation = optimizer.provide_recommendation()
```

### 11.3 Shared `OptimisationResult`

```python
@dataclass
class OptimisationResult:
    best_policy: Policy
    best_metrics: dict
    best_score: float
    n_evaluations: int
    optimiser_name: str
    seed: int
    runtime_seconds: float
    full_history: list[dict]   # all evaluated configs and their scores
```

The `seed` field is included explicitly so that every result is traceable to its exact run configuration.

---

## 12. Phase 10 — Walk-Forward Validation

### 12.1 `rbdpo/validation/walk_forward.py`

Implement `walk_forward_validate(data, optimise_fn, config, n_splits, train_pct) -> WalkForwardResult`.

**Protocol:**

1. Divide the full dataset into `n_splits` sequential folds.
2. For each fold:
   - Define `train = data[fold_start : fold_start + train_window]`
   - Define `test  = data[fold_start + train_window : next_fold_start]`
   - Run indicator precomputation on `train` data **only** (no data leakage).
   - Run optimisation on `train` data; obtain `best_policy`.
   - Backtest `best_policy` on `test` data (out-of-sample). **No refitting on test data.**
   - Record out-of-sample metrics.
3. Aggregate across all folds.

**WalkForwardResult:**

```python
@dataclass
class WalkForwardResult:
    fold_results: list[dict]
    oos_sharpe_mean: float
    oos_sharpe_std: float
    oos_calmar_mean: float
    oos_return_mean: float
    oos_max_drawdown_mean: float
    best_policies_per_fold: list[Policy]
```

**Important:** Indicators must be recomputed on each training fold independently. Never share an indicator cache between folds.

---

## 13. Phase 11 — Results and Reporting

### 13.1 `rbdpo/reporting/report.py`

Implement `generate_report(result, wf_result, output_path)`.

Save as both YAML (human-readable) and JSON (machine-readable). The report must include:

**Strategy description.** A plain-English string describing every active rule. Example output:

```
Objective: sharpe  |  Seed: 42

Entry conditions (ALL must be true simultaneously):
  Rule 1: RSI(14) crosses above 30
  Rule 2: Close price is above EMA(50)
  Rule 3: ATR(14) is above its 20-period rolling mean

Exit conditions:
  (none — exits driven by stop-loss and take-profit only)

Stop-loss:  2.0 × ATR(14) below entry price
Take-profit: 3.0 × ATR(14) above entry price
```

Implement `policy_to_human_readable(policy) -> str` to produce this output for any policy.

**Performance summary.** In-sample and out-of-sample metrics side by side.

**Walk-forward summary.** Per-fold OOS results table.

**Run metadata.** Optimiser name, seed, budget, objective key, runtime.

---

## 14. Phase 12 — CLI and Configuration Layer

### 14.1 `configs/default.yaml`

```yaml
data:
  path: null            # Required at runtime
  freq: '1D'
  min_bars: 500
  train_split: 0.7

indicators:
  n_jobs: -1            # Use all CPU cores

policy:
  n_entry_rules: 3
  n_exit_rules: 1

backtest:
  init_cash: 10000.0
  fee_rate: 0.001
  slippage: 0.0005

objective:
  name: 'sharpe'        # 'sharpe', 'calmar', 'sortino', 'max_return', 'composite'
  composite_weights: [0.5, 0.3, 0.1, 0.1]  # only used when name='composite'
  complexity_penalty: 0.02
  min_trades: 10

optimiser:
  name: 'dehb'          # 'dehb', 'nevergrad'
  budget: 500
  seed: 42              # Set for reproducibility; change for diversity
  min_fidelity: 0.1
  max_fidelity: 1.0
  n_workers: 1

validation:
  walk_forward: true
  n_splits: 5
  train_pct: 0.7

output:
  path: './results/'
```

### 14.2 CLI Entry Points — `rbdpo/cli.py`

Using `click`:

```bash
# Run full optimisation pipeline
rbdpo run --data path/to/data.csv --config configs/default.yaml --output results/

# Run with a specific objective and seed (override config file)
rbdpo run --data data.csv --objective calmar --seed 123

# Generate human-readable report from saved result
rbdpo report --result results/result.json

# Validate a saved policy on new out-of-sample data
rbdpo validate --policy results/best_policy.yaml --data path/to/new_data.csv

# Run multiple seeds in sequence to build a diverse strategy set
rbdpo sweep --data data.csv --seeds 42 123 999 777 --objective sharpe
```

---

## 15. Phase 13 — Testing Suite

### 15.1 `tests/conftest.py`

Shared fixtures:
- `sample_ohlcv_data`: A 2,000-bar synthetic OHLCV DataFrame (generate prices via geometric Brownian motion; use `np.random.seed(0)` for fixture determinism).
- `small_cache`: An `IndicatorCache` built from `sample_ohlcv_data` with a reduced indicator set (SMA, RSI, BB only) for fast testing.
- `simple_policy`: A hardcoded `Policy` using SMA comparison entry and ATR stop-loss, suitable as a smoke-test strategy.

### 15.2 Test Modules

**`test_indicators.py`:**
- Verify precomputed SMA(20) matches a manually computed SMA(20) on the same data (within float32 tolerance).
- Verify that cache memory usage is within expected bounds for the reduced indicator set.
- Verify that requesting a non-precomputed parameter returns `None` from the cache.
- Verify that pandas-ta computes the same RSI as the reference formula on known input data.

**`test_rules.py`:**
- Test each rule archetype with known synthetic inputs and verify the boolean output is exactly correct.
- Test that NaN-leading arrays (indicator warm-up) produce `False` at those positions.
- Test crossover detection with a manually constructed series that crosses a threshold at a known bar.

**`test_backtest.py`:**
- Verify that `run_backtest` returns a dict with all expected metric keys.
- Verify that a trivially active strategy produces more trades than a restrictive one.
- Verify that `n_trades == 0` results in the `999.0` sentinel from the objective function.

**`test_objective.py`:**
- Verify each named objective returns lower values (better score) for a better-performing strategy vs a worse one.
- Verify that the complexity penalty correctly reduces the score for strategies with more active rules.
- Verify that `composite` objective with custom weights behaves as expected.

**`test_search_space.py`:**
- Verify that `build_config_space` can be sampled 100 times without error.
- Verify that `decode_config` maps every sampled config to a valid `Policy` or `None`.
- Verify all conditional parameters are respected (e.g. BB params only appear when indicator is `'bb'`).
- Verify that `seed` produces deterministic sampling: two spaces sampled with the same seed return the same sequence.

### 15.3 Running Tests

```bash
uv run pytest tests/ -v --cov=rbdpo --cov-report=term-missing
```

All tests must pass before proceeding to integration testing or live validation.

---

## 16. Future Extension Placeholders

### 16.1 External Time-Series Data (v2.0)

The following modules are reserved for the v2.0 external data extension. Do not implement in v1.0.

- `rbdpo/indicators/external_series.py` — external series registry and alignment utilities
- Additional archetype `EXTERNAL_COMPARISON` in `rbdpo/policy/rules.py`
- Additional ConfigSpace slots in `rbdpo/search/space.py` for external series parameters
- Updated `configs/default.yaml` with an `external_data` section

The v1.0 architecture should be designed so that adding these modules does not require changes to the backtesting engine, optimiser integration, or reporting layer.

### 16.2 Portfolio-Level Optimisation Loop (v3.0)

> **[PLACEHOLDER — To Be Specified]**
>
> The `rbdpo/portfolio/` directory is reserved for the v3.0 portfolio extension. This will be designed and implemented after the MVP has been validated in production.
>
> Expected components:
> - `rbdpo/portfolio/universe.py` — multi-instrument data management
> - `rbdpo/portfolio/portfolio_objective.py` — portfolio-level objective (joint Sharpe, correlation, etc.)
> - `rbdpo/portfolio/outer_loop.py` — second-level optimisation loop selecting and sizing per-instrument strategies
> - `rbdpo/portfolio/risk_constraints.py` — position limits, volatility targets, drawdown budgets
>
> The single-instrument backtest engine (`rbdpo/backtest/`) will be reused without modification as the inner evaluation loop.
>
> Full specification to follow once MVP is validated.

---

## 17. Dependency Reference

| Library | Min Version | Purpose |
|---|---|---|
| Python | 3.11 | Runtime |
| vectorbt | 0.26 | Vectorised backtesting and indicator precomputation |
| pandas | 2.0 | Data manipulation |
| numpy | 1.26 | Numerical arrays |
| pandas-ta | 0.3.14b | Technical indicator computation (pure Python; no C dependencies) |
| ConfigSpace | 0.7 | Conditional hyperparameter space definition |
| DEHB | 0.0.6 | Primary optimiser |
| nevergrad | 1.0 | Secondary / prototyping optimiser |
| joblib | 1.3 | Parallel precomputation |
| pyyaml | 6.0 | Configuration files |
| click | 8.1 | CLI |
| pytest | 7.4 | Test framework |
| pytest-cov | 4.1 | Coverage reporting |
| uv | latest | Environment and dependency management |

**Explicitly excluded:**

| Library | Reason |
|---|---|
| TA-Lib | Requires compiled C system library; incompatible with reproducible uv environments |
| ta (technical analysis) | Superseded by pandas-ta |
| Julia / PyCall | FastBack.jl is insufficiently mature; speed advantage over vectorbt is marginal when indicators are precomputed |

---

## 18. Implementation Order Summary

Build and test each phase before starting the next.

| Phase | Module(s) | Gate Criterion Before Proceeding |
|---|---|---|
| 1 | `data/` | Loader returns validated DataFrame; split produces correct proportions |
| 2 | `indicators/` | All indicators precompute without error; memory usage is logged; pandas-ta output matches expected values |
| 3 | `policy/schema.py`, `decoder.py` | Policy objects are hashable; decoder correctly maps sample configs to Policy or None |
| 4 | `policy/rules.py` | Every archetype produces correct boolean arrays on known synthetic inputs |
| 5 | `policy/sl_tp.py` | SL and TP level arrays are correctly shaped and valued on known inputs |
| 6 | `backtest/engine.py`, `metrics.py` | End-to-end backtest completes and returns all metric keys; `n_trades` is plausible |
| 7 | `objective/function.py` | All named objectives return lower scores for better strategies; complexity penalty applies correctly |
| 8 | `search/space.py` | ConfigSpace samples without error 100 times; all conditions resolve correctly; same seed produces identical samples |
| 9 | `search/dehb_runner.py` | 10-evaluation test run with `seed=42` completes and returns an `OptimisationResult`; rerun with `seed=42` produces identical result |
| 10 | `validation/walk_forward.py` | 3-fold walk-forward on small data completes; OOS metrics are logged per fold |
| 11 | `reporting/report.py` | YAML and JSON report files written correctly; human-readable policy string is accurate |
| 12 | `cli.py`, `configs/` | `uv run rbdpo run` executes end-to-end on sample data without error |
| 13 | `tests/` | All tests pass with ≥ 80% coverage |

---

*End of Implementation Plan — v1.0*
