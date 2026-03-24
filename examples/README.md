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
  - buy-and-hold baseline in comparison charts
  - plots and CSV exports
  - helper-API-first workflow patterns

## Prerequisites

From project root:

```bash
uv pip install -e .
uv pip install yfinance ipython jupyter matplotlib
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

`spy_showcase.py` writes to `examples/showcase_results/` including:

- `scenario_summary.csv`
- `helper_summary.csv`
- `equity_curves.png`
- `metric_score_comparison.png`
- per-run `strategy.txt` files
- per-scenario report files and effective config snapshots

## Interpreting The Showcase

- Use `scenario_summary.csv` for an at-a-glance comparison of optimiser/objective combinations.
- Use `helper_summary.csv` for broader metric columns across runs.
- Compare low/high fee scenarios to understand strategy robustness to trading costs.
- Compare `best_metric_value` and `best_score` together: `best_score` is minimized, while `best_metric_value` is the human-readable target metric.
