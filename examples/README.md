# Examples (Python "Vignettes")

In Python projects, the common convention is to use an `examples/` folder (and sometimes `tutorials/`) rather than the R term "vignettes".

This folder contains block-runnable `#%%` scripts that work like notebooks in VS Code.

## Files

- `quick_api_run.py`: shortest end-to-end helper API example for script users.
- `spy_demo.py`: Minimal helper-API demo with diverse optimiser/objective runs (`dehb`, `nevergrad`, `tpe`) at high-search budgets.
- `spy_showcase.py`: Rich scenario showcase with:
  - multiple optimisers/objectives/seeds
  - full-history SPY data from 1996
  - explicit `max_return` objective run
  - `max_return_dd_cap` (return under drawdown cap) objective run
  - fee/slippage sensitivity
  - leverage and borrow-cost sensitivity run
  - indicator subset/range filtering example
  - strategy printout text for interpretability
  - explicit out-of-sample metric columns per scenario
  - buy-and-hold baseline in comparison charts
  - plots and CSV exports
  - helper-API-first workflow patterns

## Prerequisites

From project root:

```bash
uv pip install -e .[optim]
uv pip install yfinance ipython jupyter matplotlib plotly
```

## How To Run

### Option 1: Run block-by-block in VS Code

Open either script and run each `#%%` block interactively.

### Option 2: Run as full scripts

```bash
uv run python examples/spy_demo.py
uv run python examples/spy_showcase.py
```

## What Gets Saved

`spy_demo.py` writes to `examples/results_api_demo/` including per-run reports, effective configs, `summary.csv`, and `equity_curves.png`.

`spy_demo.py` also prints:

- the exact strategy logic found by each run
- out-of-sample metrics for that strategy on the held-out split

`spy_showcase.py` writes to `examples/showcase_results/` including:

- `scenario_summary.csv`
- `helper_summary.csv`
- `equity_curves.png`
- `metric_score_comparison.png`
- per-run `strategy.txt` files
- per-scenario report files and effective config snapshots

When a run uses native DEHB, checkpoint/history artifacts are written under that run's output directory (for example, `examples/results_api_demo/<run>/checkpoints/dehb/`).

## Interpreting The Showcase

- Use `scenario_summary.csv` for an at-a-glance comparison of optimiser/objective combinations.
- Use `helper_summary.csv` for broader metric columns across runs.
- Compare low/high fee scenarios to understand strategy robustness to trading costs.
- Compare `best_metric_value` and `best_score` together: `best_score` is minimized, while `best_metric_value` is the human-readable target metric.

## Optimiser Tuning Notes

The examples are intentionally high-budget. For faster local iterations:

- Lower `optimiser.budget` first.
- For `tpe`, prefer `tpe_multivariate: true` and `tpe_group: true` on conditional spaces.
- For `dehb` seed-DE backend, tune `dehb_population_size`, `dehb_mutation_factor`, and `dehb_crossover_rate`.
- For `nevergrad`, set `nevergrad_num_workers` > 1 when parallel resources are available.
