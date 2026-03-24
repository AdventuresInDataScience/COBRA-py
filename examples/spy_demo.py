#%%
from __future__ import annotations

import importlib.metadata as im
from pathlib import Path

import pandas as pd
import yaml

from cobra_py import (
    fetch_yfinance_ohlcv,
    list_available_objectives,
    list_available_optimisers,
    load_config,
    plot_equity_curves,
    run_optimiser,
    summarise_reports,
)


# %% 1. Project paths
def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() and (p / "configs" / "default.yaml").exists():
            return p
    raise RuntimeError("Could not find project root containing pyproject.toml and configs/default.yaml")


try:
    project_root = find_project_root(Path(__file__).resolve())
except NameError:  # pragma: no cover - interactive fallback
    project_root = find_project_root(Path.cwd())

examples_dir = project_root / "examples"
results_dir = examples_dir / "results_api_demo"
results_dir.mkdir(parents=True, exist_ok=True)

print("Project root:", project_root)
print("Results dir:", results_dir)


# %% 2. Environment and API availability
print("cobra-py version:", im.version("cobra-py"))
print("Available optimisers:", list_available_optimisers())
print("Available objectives:", list_available_objectives())


# %% 3. Base config and data
base_cfg = load_config(project_root / "configs" / "default.yaml")
base_cfg["optimiser"]["budget"] = max(10000, int(base_cfg["optimiser"].get("budget", 10000)))
base_cfg["indicators"]["n_jobs"] = 1

spy_ohlcv = fetch_yfinance_ohlcv("SPY", start="2018-01-01", interval="1d")
print("SPY rows:", len(spy_ohlcv))
print("Date range:", spy_ohlcv.index.min().date(), "to", spy_ohlcv.index.max().date())


# %% 4. Diverse runs (optimiser + objective combinations)
runs = {
    "dehb_sharpe_10k": {
        "optimiser": {"name": "dehb", "budget": 10000, "seed": 42},
        "objective": {"name": "sharpe", "min_trades": 1},
    },
    "nevergrad_sortino_10k": {
        "optimiser": {"name": "nevergrad", "budget": 10000, "seed": 42},
        "objective": {"name": "sortino", "min_trades": 1},
    },
    "tpe_ulcer_10k": {
        "optimiser": {"name": "tpe", "budget": 10000, "seed": 42},
        "objective": {"name": "ulcer", "min_trades": 1},
    },
    "dehb_composite_10k": {
        "optimiser": {"name": "dehb", "budget": 10000, "seed": 123},
        "objective": {
            "name": "composite",
            "composite_weights": [0.45, 0.30, 0.15, 0.10],
            "min_trades": 1,
        },
        "policy": {"n_entry_rules": 2, "n_exit_rules": 1},
        "indicators": {
            "param_ranges": {
                "bb": {
                    "period": {"start": 10, "stop": 200, "step": 5},
                    "std": {"range": [1.5, 3.0, 0.5]},
                    "ma_type": ["sma", "ema"],
                }
            }
        },
    },
}

named_reports: dict[str, dict] = {}
for run_name, overrides in runs.items():
    print("Running:", run_name)
    out_dir = results_dir / run_name
    result = run_optimiser(
        source=spy_ohlcv,
        config=base_cfg,
        overrides=overrides,
        output_path=out_dir,
        run_walk_forward=False,
        evaluate_oos=True,
    )
    named_reports[run_name] = result["report"]

    with (out_dir / "effective_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(result["config"], f, sort_keys=False)

    print(f"\nStrategy for {run_name}:\n{result['report'].get('policy_human_readable', '')}\n")
    if result.get("oos_metrics"):
        oos = result["oos_metrics"]
        print(
            "OOS metrics:",
            {
                "total_return": oos.get("total_return"),
                "sharpe_ratio": oos.get("sharpe_ratio"),
                "max_drawdown": oos.get("max_drawdown"),
            },
        )


# %% 5. Summary table
comparison = summarise_reports(named_reports).sort_values("run").reset_index(drop=True)
print(comparison[["run", "optimiser", "objective", "best_metric_name", "best_metric_value", "best_score", "evals"]])

summary_csv = results_dir / "summary.csv"
comparison.to_csv(summary_csv, index=False)
print("Saved:", summary_csv)


# %% 6. Quick chart from helper API
eq_png = results_dir / "equity_curves.png"
plot_equity_curves(named_reports, normalize=True, title="SPY helper API demo (10k budgets)", save_path=eq_png)
print("Saved:", eq_png)


