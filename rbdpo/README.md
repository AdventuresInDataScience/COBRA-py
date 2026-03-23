# COBRA-py MVP

COBRA-py Stage 1 (MVP) implementation for interpretable rule-based trading strategy discovery.

## First Run (3 Commands)

From the project root:

```bash
uv venv .venv
uv pip install -e .
uv pip install yfinance ipython jupyter matplotlib
```

Then run the showcase:

```bash
uv run python examples/spy_showcase.py
```

Expected outputs are written under [examples/showcase_results](examples/showcase_results).

## Examples

Interactive, block-runnable demo scripts live in [examples/README.md](examples/README.md).

- [examples/spy_demo.py](examples/spy_demo.py): quick run + config override demo.
- [examples/spy_showcase.py](examples/spy_showcase.py): scenario comparison, fee modeling, strategy vs buy-and-hold, plots, and mini walk-forward OOS benchmarking.

## Quick Start (CLI)

```bash
uv pip install -e .
cobra-py run --data /path/to/data.csv
```

## Repository Structure

Core project layout and purpose:

- [pyproject.toml](pyproject.toml): package metadata, dependencies, CLI entry points, and build configuration.
- [configs/default.yaml](configs/default.yaml): default runtime settings for data, policy, objective, optimiser, backtest, and output.
- [cobra_py/cli.py](cobra_py/cli.py): CLI commands (`run`, `report`, `validate`, `sweep`).
- [cobra_py/data](cobra_py/data): OHLCV loading, validation, and train/test preprocessing.
- [cobra_py/indicators](cobra_py/indicators): indicator registry, precomputation orchestration, and cache.
- [cobra_py/policy](cobra_py/policy): rule schema, rule evaluation, stop-loss/take-profit logic, and config decoder.
- [cobra_py/backtest](cobra_py/backtest): backtest engine and metric extraction.
- [cobra_py/objective](cobra_py/objective): objective scoring and complexity regularization.
- [cobra_py/search](cobra_py/search): search space and optimisation runners.
- [cobra_py/validation](cobra_py/validation): walk-forward validation flow.
- [cobra_py/reporting](cobra_py/reporting): JSON/YAML reports and human-readable strategy rendering.
- [examples](examples): block-runnable demos and showcase scripts.
- [tests](tests): pytest suite for indicators, rules, backtest, objective, and search-space behavior.

## What The Main Example Files Do

- [examples/spy_demo.py](examples/spy_demo.py):
	- loads SPY OHLCV from yfinance
	- loads base config from file
	- runs baseline and override experiments
	- writes result reports and effective configs

- [examples/spy_showcase.py](examples/spy_showcase.py):
	- runs multiple scenarios (objective, seed, fees, slippage)
	- compares strategy vs buy-and-hold
	- plots equity/return/sharpe comparisons
	- runs mini walk-forward OOS comparison
	- writes CSV summaries and PNG charts

