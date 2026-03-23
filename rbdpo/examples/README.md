# Examples (Python "Vignettes")

In Python projects, the common convention is to use an `examples/` folder (and sometimes `tutorials/`) rather than the R term "vignettes".

This folder contains block-runnable `#%%` scripts that work like notebooks in VS Code.

## Files

- `spy_demo.py`: Minimal end-to-end SPY run with config loading and override demo.
- `spy_showcase.py`: Rich scenario showcase with:
  - multiple objectives/seeds
  - fee/slippage sensitivity
  - strategy vs buy-and-hold comparisons
  - plots and CSV exports
  - mini walk-forward out-of-sample (OOS) benchmark vs buy-and-hold

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

`spy_demo.py` writes to `examples/results_file_config/` and `examples/results_override_config/`.

`spy_showcase.py` writes to `examples/showcase_results/` including:

- `scenario_summary.csv`
- `scenario_summary_vs_buyhold.csv`
- `equity_curves_vs_buyhold.png`
- `return_sharpe_comparison.png`
- `walk_forward_oos_summary.csv`
- `walk_forward_oos_folds.csv`
- per-scenario report files and effective config snapshots

## Interpreting The Showcase

- Use `scenario_summary_vs_buyhold.csv` for in-sample excess return/sharpe vs buy-and-hold.
- Use `walk_forward_oos_summary.csv` for mean OOS comparison across folds.
- Compare low/high fee scenarios to understand strategy robustness to trading costs.
