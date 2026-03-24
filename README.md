# COBRA-py MVP

COBRA-py Stage 1 (MVP) implementation for interpretable rule-based trading strategy discovery.

## First Run (Single Root)

From the `COBRA-py` root folder:

```bash
uv sync
uv run pytest tests -q
uv run cobra-py run --data smoke_data.csv --config smoke_config.yaml
```

Then run the showcase:

```bash
uv run python examples/spy_showcase.py
```

Expected outputs are written under [smoke_results](smoke_results) and [examples/showcase_results](examples/showcase_results).

## Examples

Interactive, block-runnable demo scripts live in [examples/README.md](examples/README.md).

- [examples/spy_demo.py](examples/spy_demo.py): quick run + config override demo.
- [examples/spy_showcase.py](examples/spy_showcase.py): diverse optimiser/objective/cost scenario comparison using the helper API.

## Quick Start (CLI)

```bash
uv run cobra-py run --data /path/to/data.csv
```

List available indicators and parameter ranges (after applying config include/exclude/overrides):

```bash
uv run cobra-py indicators --config configs/default.yaml
```

## Quick Start (Python Script Helpers)

For script-first workflows, use the helper API in [cobra_py/helpers.py](cobra_py/helpers.py):

```python
from cobra_py import fetch_yfinance_ohlcv, load_config, plot_equity_curves, run_optimiser, summarise_reports

cfg = load_config("configs/default.yaml")
spy = fetch_yfinance_ohlcv("SPY", start="2018-01-01", interval="1d")

run_a = run_optimiser(spy, config=cfg, overrides={"optimiser": {"name": "dehb", "budget": 10000, "seed": 42}})
run_b = run_optimiser(spy, config=cfg, overrides={"optimiser": {"name": "tpe", "budget": 10000, "seed": 42}})

summary = summarise_reports({
	"dehb": run_a["report"],
	"tpe": run_b["report"],
})
print(summary[["run", "objective", "best_metric_name", "best_metric_value", "best_score"]])

plot_equity_curves({"dehb": run_a["report"], "tpe": run_b["report"]}, save_path="equity_curves.png")
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
	- loads SPY OHLCV from yfinance through helper API
	- runs multiple optimiser/objective combinations
	- uses 10,000-budget search baselines
	- writes per-run reports, summary CSV, and equity chart

- [examples/spy_showcase.py](examples/spy_showcase.py):
	- runs broader scenario matrix across optimiser, objective, cost regime, and indicator subset
	- uses helper API for pipeline execution and report aggregation
	- writes scenario and helper summary CSVs
	- writes equity and metric comparison charts

## Configuration Guide

Main config file: [configs/default.yaml](configs/default.yaml).

- `data.freq`: resample frequency (`1D`, `4H`, etc.).
- `data.min_bars`: minimum rows after load/resample.
- `data.train_split`: in-sample fraction used for optimisation.

- `indicators.include`: optional whitelist of indicator names.
- `indicators.exclude`: optional blacklist applied after include.
- `indicators.param_ranges`: optional parameter overrides per indicator.
- `indicators.n_jobs`: parallel workers for indicator precompute (`-1` means all cores).

- `policy.n_entry_rules`: number of entry rule slots sampled.
- `policy.n_exit_rules`: number of exit rule slots sampled.

- `backtest.init_cash`: starting portfolio value.
- `backtest.fee_rate`: decimal transaction fee per trade notional (`0.001` = 0.1%).
- `backtest.slippage`: decimal execution slippage (`0.0005` = 0.05%).

- `objective.name`: one of `sharpe`, `calmar`, `sortino`, `ulcer`, `max_return`, `composite`.
- `objective.composite_weights`: used only for `composite`, with score:

$$
	ext{score} = -\left(w_0\cdot\text{sharpe} + w_1\cdot\text{calmar} + w_2\cdot\text{return} - w_3\cdot|\text{max\_drawdown}|\right) + \lambda\cdot n_{rules}
$$

- `objective.complexity_penalty` (`\lambda`): penalty per active rule.
- `objective.min_trades`: if trades are below this, score becomes `999.0`.

- `optimiser.name`: one of `dehb`, `nevergrad`, `tpe`.
- `optimiser.budget`: number of sampled configurations.
- `optimiser.seed`: random seed for reproducibility.

## Optimiser Options

- `dehb`: default optimiser mode.
- `nevergrad`: alternate optimiser entrypoint.
- `tpe`: TPE entrypoint.

In this MVP, these options currently share the same lightweight search backend and differ mainly by configuration intent and run labeling in reports.

## Interpreting Scores

- Internal optimisation minimizes `best_score`.
- For non-composite objectives, this is mostly negative of your chosen metric plus complexity penalty.
- Reports also provide human-readable fields: `best_metric_name` and `best_metric_value`.

## Why Demo Runs Are Fast

Very fast runs in `examples/spy_demo.py` are expected for this MVP because:

- Search loop is currently a lightweight random-search style loop.
- Data window in demos is modest.
- Core indicator/backtest operations are vectorized and cached.
- The objective evaluation is simple and low-overhead.

Larger datasets, more granular bars, stricter walk-forward settings, and bigger budgets will increase runtime substantially.

