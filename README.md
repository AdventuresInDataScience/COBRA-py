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

Note: default optimiser budget in [configs/default.yaml](configs/default.yaml) is now `100` for faster iteration. Increase this when you want deeper search.

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

# Interactive Plotly chart (legend toggle, zoom, range slider)
fig = plot_equity_curves(
	{"dehb": run_a["report"], "tpe": run_b["report"]},
	backend="plotly",
	save_path="equity_curves.html",
)
fig.show()
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
- `indicators.param_ranges` supports either explicit lists or range shorthand:

```yaml
indicators:
	param_ranges:
		bb:
			period: {start: 10, stop: 200, step: 5}
			std: {range: [1.5, 3.0, 0.5]}
			ma_type: ["sma", "ema"]
```

- `indicators.n_jobs`: parallel workers for indicator precompute (`-1` means all cores).

- `policy.n_entry_rules`: number of entry rule slots sampled.
- `policy.n_exit_rules`: number of exit rule slots sampled.

- `backtest.init_cash`: starting portfolio value.
- `backtest.fee_rate`: decimal transaction fee per trade notional (`0.001` = 0.1%).
- `backtest.slippage`: decimal execution slippage (`0.0005` = 0.05%).
- `backtest.leverage`: gross leverage multiplier (`1.0` = unlevered).
- `backtest.borrow_cost_rate`: annualized borrow rate applied to leveraged notional above equity (`0.06` = 6%/year).
- `backtest.leverage_range`: optimiser search range for learned leverage values.
- `backtest.borrow_cost_rate_range`: optimiser search range for learned financing costs.

- `objective.name`: one of `sharpe`, `calmar`, `car_mdd`, `cagr`, `sortino`, `ulcer`, `max_return`, `max_return_dd_cap`, `composite`.
- `calmar` uses trailing 3-year CAGR and trailing 3-year max drawdown (36-month convention).
- `car_mdd` uses full-period CAGR and full-period max drawdown (backward-looking full-sample ratio).
- `objective.max_drawdown_cap`: used by `max_return_dd_cap` as a hard max drawdown limit.
- `objective.composite_weights`: used only for `composite`, with score:

$$
	ext{score} = -\left(w_0\cdot\text{sharpe} + w_1\cdot\text{calmar} + w_2\cdot\text{return} - w_3\cdot|\text{max\_drawdown}|\right) + \lambda\cdot n_{rules}
$$

- `objective.complexity_penalty` (`\lambda`): penalty per active rule.
- `objective.min_trades`: if trades are below this, score becomes `999.0`.

- `optimiser.name`: one of `dehb`, `nevergrad`, `tpe`.
- `optimiser.budget`: number of sampled configurations (default `100` for quick first runs).
- `optimiser.seed`: random seed for reproducibility.
- `optimiser.dehb_backend`: DEHB backend selector (`auto`, `native`, `seed_de`).
- `optimiser.min_fidelity`: lower fidelity used by native DEHB (fraction of train bars).
- `optimiser.max_fidelity`: upper fidelity used by native DEHB.
- `optimiser.n_workers`: DEHB parallel workers (native backend only).
- `optimiser.nevergrad_algorithm`: algorithm class used by Nevergrad (`NGOpt`, `OnePlusOne`, `CMA`, ...).

## Optimiser Options

- `dehb`:
	- `dehb_backend: auto`: use true DEHB package when available, otherwise fallback to internal seed-DE surrogate.
	- `dehb_backend: native`: force true DEHB package path and fail if package is missing.
	- `dehb_backend: seed_de`: force internal surrogate backend.
	- `dehb_mutation_factor`, `dehb_crossover_rate`, `dehb_population_size`: evolution controls for `seed_de` backend.
- `nevergrad`: Nevergrad runner with configurable algorithm class from `optimiser.nevergrad_algorithm`.
	- `nevergrad_num_workers`: parallel workers where supported by the selected Nevergrad optimizer.
- `tpe`: Optuna TPE runner.
	- `tpe_multivariate`, `tpe_group`, `tpe_n_startup_trials`, `tpe_constant_liar`: TPE efficiency controls for conditional spaces.

### Parameter Type Handling

The search space is strongly typed and conditional:

- Categorical: connector logic (`and`/`or`/`dnf`), operators (`>`, `<`, `==`, `crosses_*`, pattern op), archetype, indicator names, output selectors.
- Integer: lookbacks and discrete indicator lengths/periods.
- Float / quantized float: risk settings such as SL/TP percentages, ATR multipliers, borrow costs, leverage ranges.

This is important for optimiser efficiency on mixed conditional spaces, especially for TPE and Nevergrad.

### Tuning Presets

Use these as practical defaults by budget size; they apply across all optimisers.

`small_budget` (fast iteration, <= 500 evals):

```yaml
optimiser:
	name: tpe
	budget: 300
	seed: 42
	tpe_multivariate: true
	tpe_group: true
	tpe_n_startup_trials: 40
	tpe_constant_liar: false
```

`medium_budget` (balanced, 1k-5k evals):

```yaml
optimiser:
	name: dehb
	dehb_backend: auto
	budget: 2500
	seed: 42
	min_fidelity: 0.2
	max_fidelity: 1.0
	n_workers: 1
	dehb_mutation_factor: 0.8
	dehb_crossover_rate: 0.7
	dehb_population_size: 24
```

`large_budget` (deep search, >= 10k evals):

```yaml
optimiser:
	name: nevergrad
	nevergrad_algorithm: NGOpt
	budget: 10000
	seed: 42
	nevergrad_num_workers: 4
```

For heavily conditional spaces, start with `tpe` and `dehb(native/auto)`; use Nevergrad for broad global exploration and diversity across seeds.

Example config snippets:

```yaml
optimiser:
	name: dehb
	dehb_backend: auto
	budget: 10000
	min_fidelity: 0.2
	max_fidelity: 1.0
	n_workers: 1
```

```yaml
optimiser:
	name: dehb
	dehb_backend: native
	budget: 10000
	min_fidelity: 0.2
	max_fidelity: 1.0
	n_workers: 4
```

For `nevergrad` and `tpe`, install optional optimizer dependencies:

```bash
uv pip install -e .[optim]
```

For interactive Plotly charting in helper scripts:

```bash
uv pip install plotly
```

## Interpreting Scores

- Internal optimisation minimizes `best_score`.
- For non-composite objectives, this is mostly negative of your chosen metric plus complexity penalty.
- Reports also provide human-readable fields: `best_metric_name` and `best_metric_value`.
- Example scripts now print the final interpreted strategy text and include explicit out-of-sample metric output.

## Why Demo Runs Are Fast

Very fast runs in `examples/spy_demo.py` are expected for this MVP because:

- Search loop is currently a lightweight random-search style loop.
- Data window in demos is modest.
- Core indicator/backtest operations are vectorized and cached.
- The objective evaluation is simple and low-overhead.

Larger datasets, more granular bars, stricter walk-forward settings, and bigger budgets will increase runtime substantially.

