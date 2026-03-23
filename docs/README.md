# COBRA-py â€” Rule-Based Direct Policy Optimisation

**Automatically discover interpretable trading strategies using black-box optimisation.**

COBRA-py searches a structured space of human-readable technical trading rules and finds the best-performing strategy for a given objective â€” without neural networks, without black-box models, and without any post-hoc explanation needed.

Every strategy produced by COBRA-py is a plain set of indicator conditions you can read, understand, and deploy in a few lines of code.

---

## What It Does

Given historical OHLCV price data and a performance objective (e.g. maximise Sharpe ratio), COBRA-py:

1. **Precomputes** thousands of technical indicator arrays across all parameter combinations
2. **Searches** the space of possible rule combinations using a multi-fidelity black-box optimiser
3. **Evaluates** each candidate strategy via vectorised backtesting
4. **Returns** the best strategy as a human-readable rule set, with a full performance report

Example output:

```
Objective: sharpe  |  Seed: 42

Entry conditions (ALL must be true simultaneously):
  Rule 1: RSI(14) crosses above 30
  Rule 2: Close price is above EMA(50)
  Rule 3: ATR(14) is above its 20-period rolling mean

Stop-loss:  2.0 Ã— ATR(14) below entry price
Take-profit: 3.0 Ã— ATR(14) above entry price

In-sample Sharpe:  1.43
Out-of-sample Sharpe:  1.11
```

---

## MVP Scope

Version 1.0 operates on **single-instrument OHLCV data only**. Two extensions are planned:

- **v2.0:** External time-series data as rule inputs (e.g. `policy_rate > 2%`, `yield_curve < 0`)
- **v3.0:** Portfolio-level optimisation loop for multi-instrument joint strategy discovery

---

## Installation

COBRA-py uses [`uv`](https://github.com/astral-sh/uv) for environment management. All dependencies are pure Python â€” no C library compilation required.

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and install

```bash
git clone https://github.com/your-org/cobra-py.git
cd cobra-py
uv sync          # installs all dependencies from uv.lock
uv pip install -e .
```

### 3. Verify

```bash
uv run cobra-py --help
```

---

## Quick Start

### From the command line

```bash
# Run with default settings (Sharpe objective, seed 42)
uv run cobra-py run --data data/SPY_daily.csv --output results/

# Choose a different objective
uv run cobra-py run --data data/SPY_daily.csv --objective calmar

# Change the seed (produces a different strategy)
uv run cobra-py run --data data/SPY_daily.csv --seed 123

# Run multiple seeds and compare results
uv run cobra-py sweep --data data/SPY_daily.csv --seeds 42 123 999 --objective sharpe
```

### From Python

```python
from cobra_py.data.loader import load_ohlcv
from cobra_py.data.preprocessor import preprocess
from cobra_py.indicators.precompute import precompute_all
from cobra_py.indicators.registry import DEFAULT_REGISTRY
from cobra_py.search.dehb_runner import run_dehb
from cobra_py.search.space import build_config_space
from cobra_py.reporting.report import generate_report

# Load data
data = load_ohlcv('data/SPY_daily.csv')
train, test = preprocess(data, config={'train_split': 0.7})

# Precompute all indicators once
cache = precompute_all(train, DEFAULT_REGISTRY, n_jobs=-1)

# Build search space
cs = build_config_space(n_entry_rules=3, n_exit_rules=1, registry=DEFAULT_REGISTRY)

# Run optimisation (Sharpe objective, reproducible with seed=42)
result = run_dehb(
    cache=cache,
    data=train,
    config_space=cs,
    obj_config={'objective': 'sharpe', 'min_trades': 10},
    backtest_config={'init_cash': 10000, 'fee_rate': 0.001},
    budget=500,
    seed=42,
)

print(result.best_policy)
print(f"In-sample Sharpe: {result.best_metrics['sharpe_ratio']:.2f}")
```

See `examples/quickstart.ipynb` for a fully worked walkthrough.

---

## Configuration

All settings are controlled via a YAML config file. The defaults are in `configs/default.yaml`. Key options:

```yaml
objective:
  name: 'sharpe'          # Options: sharpe, calmar, sortino, max_return, composite
  complexity_penalty: 0.02

optimiser:
  name: 'dehb'            # Options: dehb, nevergrad
  budget: 500             # Number of strategy evaluations
  seed: 42                # Seed for reproducibility (change for diversity)

policy:
  n_entry_rules: 3        # Maximum number of simultaneous entry conditions
  n_exit_rules: 1         # Maximum number of simultaneous exit conditions

validation:
  walk_forward: true
  n_splits: 5
```

Override any setting at the CLI:

```bash
uv run cobra-py run --data data.csv --objective sortino --seed 999 --budget 1000
```

---

## Reproducibility and Diversity

COBRA-py is designed to be both **reproducible** and **diverse**:

- **Reproducibility:** The same seed + same config always produces the same strategy. The seed is recorded in every output file.
- **Diversity:** Running different seeds with the same config explores different regions of the search space, producing qualitatively different strategies. This is useful for building a portfolio of uncorrelated rules.

```bash
# Build a diverse strategy set
uv run cobra-py sweep --data data.csv --seeds 42 123 456 789 999 --objective sharpe --output results/ensemble/
```

---

## Supported Objectives

| Key | Description |
|---|---|
| `sharpe` | Maximise Sharpe ratio |
| `calmar` | Maximise Calmar ratio (return / max drawdown) |
| `sortino` | Maximise Sortino ratio (penalises downside vol only) |
| `max_return` | Maximise total return |
| `composite` | Weighted blend (weights configurable in YAML) |

---

## Indicators Available

All indicators are computed using `pandas-ta` (pure Python, no system dependencies):

- **Moving averages:** SMA, EMA, WMA
- **Oscillators:** RSI, MACD, Stochastic, ROC, CCI
- **Volatility / Bands:** Bollinger Bands, ATR, Keltner Channel, Donchian Channel
- **Trend strength:** ADX, Parabolic SAR
- **Volume:** OBV, VWAP

---

## Stop-Loss and Take-Profit Types

The optimiser selects both the *type* and *parameters* of the risk management block:

**Stop-loss:** Fixed %, ATR multiple, Swing low, Bollinger lower band, Trailing ATR

**Take-profit:** Fixed %, ATR multiple, Risk-reward ratio, Swing high, Bollinger upper band

---

## Rule Archetypes

Rules are parameterised templates, not hardcoded instances. The optimiser selects the archetype and its parameters:

- **Comparison:** `indicator > threshold`, `indicator1 > indicator2`, `price > indicator`
- **Crossover:** `indicator crosses_above threshold`, `indicator1 crosses_above indicator2`
- **Band test:** `price above band_upper`, `price below band_lower`
- **Pattern:** `price at N-bar high`, `consecutive(condition, N)`
- **Derivative:** `slope(indicator) > 0`, `change(indicator, lag) > threshold`
- **Statistical:** `zscore(indicator, lookback) > threshold`

---

## Project Structure

```
cobra_py/
â”œâ”€â”€ configs/default.yaml       # Default configuration
â”œâ”€â”€ examples/quickstart.ipynb  # Worked example notebook
â”œâ”€â”€ cobra_py/
â”‚   â”œâ”€â”€ data/                  # Data loading and preprocessing
â”‚   â”œâ”€â”€ indicators/            # Indicator precomputation and cache
â”‚   â”œâ”€â”€ policy/                # Policy schema, rules, SL/TP, decoder
â”‚   â”œâ”€â”€ backtest/              # vectorbt integration and metrics
â”‚   â”œâ”€â”€ objective/             # Named objectives and regularisation
â”‚   â”œâ”€â”€ search/                # ConfigSpace + DEHB / Nevergrad runners
â”‚   â”œâ”€â”€ validation/            # Walk-forward validation
â”‚   â””â”€â”€ reporting/             # Results serialisation and human-readable output
â””â”€â”€ tests/                     # Full test suite
```

---

## Running Tests

```bash
uv run pytest tests/ -v --cov=cobra_py --cov-report=term-missing
```

---

## Design Philosophy

COBRA-py is built around three principles:

1. **Interpretability by construction.** The policy grammar only produces strategies a practitioner can read and reason about. Interpretability is not a post-processing step.

2. **Objective-driven search.** The user specifies what they are optimising for. The system finds the best strategy for *that* objective, not for a surrogate reward.

3. **Reproducibility first.** Every run records its seed. Results are always traceable. Diversity is achieved deliberately by varying the seed, not accidentally.

---

## Academic Context

COBRA-py draws on several bodies of literature:

- **Genetic programming for trading** (Allen & Karjalainen 1999; Neely et al. 1997) â€” shares the rule-space philosophy but replaces evolutionary operators with modern black-box optimisers
- **Hyperparameter optimisation** (DEHB â€” Awad et al. 2021; ConfigSpace â€” Lindauer et al. 2019) â€” the core algorithmic machinery is borrowed from the AutoML literature
- **Direct policy search** (Salimans et al. 2017; Wang et al. 2020) â€” offline black-box search over a bounded policy class
- **Backtest bias and overfitting** (Bailey et al. 2017; Sullivan et al. 1999) â€” walk-forward validation and complexity regularisation address documented pitfalls

See the [Summary & Design Document](COBRA-py_Summary_Design.md) for a detailed literature comparison and full reference list.

---

## Licence

MIT

---

## Contributing

Contributions welcome. Please open an issue before starting work on a new feature. All PRs must pass the test suite (`uv run pytest tests/`) and include tests for any new functionality.



