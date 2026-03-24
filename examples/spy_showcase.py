# %% 1. Setup: imports and paths
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from cobra_py import fetch_yfinance_ohlcv, load_config, plot_equity_curves, run_optimiser, summarise_reports


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
showcase_dir = examples_dir / "showcase_results"
showcase_dir.mkdir(parents=True, exist_ok=True)

print("Project root:", project_root)
print("Showcase dir:", showcase_dir)


# %% 2. Load config and data
base_cfg = load_config(project_root / "configs" / "default.yaml")
base_cfg["optimiser"]["budget"] = max(10000, int(base_cfg["optimiser"].get("budget", 10000)))
base_cfg["indicators"]["n_jobs"] = 1

spy = fetch_yfinance_ohlcv("SPY", start="2018-01-01", interval="1d")
print("SPY rows:", len(spy), "date range:", spy.index.min().date(), "to", spy.index.max().date())


# %% 3. Scenario matrix: diverse optimisers, objectives, and cost assumptions
scenarios: list[dict] = [
    {
        "name": "A_dehb_sharpe_10k",
        "overrides": {
            "optimiser": {"name": "dehb", "budget": 10000, "seed": 42},
            "objective": {"name": "sharpe", "min_trades": 1},
        },
    },
    {
        "name": "B_nevergrad_sortino_10k",
        "overrides": {
            "optimiser": {"name": "nevergrad", "budget": 10000, "seed": 42},
            "objective": {"name": "sortino", "min_trades": 1},
        },
    },
    {
        "name": "C_tpe_ulcer_10k",
        "overrides": {
            "optimiser": {"name": "tpe", "budget": 10000, "seed": 42},
            "objective": {"name": "ulcer", "min_trades": 1},
        },
    },
    {
        "name": "D_dehb_composite_10k",
        "overrides": {
            "optimiser": {"name": "dehb", "budget": 10000, "seed": 123},
            "objective": {
                "name": "composite",
                "composite_weights": [0.40, 0.30, 0.20, 0.10],
                "min_trades": 1,
            },
            "policy": {"n_entry_rules": 2, "n_exit_rules": 1},
        },
    },
    {
        "name": "E_low_cost_regime",
        "overrides": {
            "optimiser": {"name": "tpe", "budget": 10000, "seed": 7},
            "objective": {"name": "calmar", "min_trades": 1},
            "backtest": {"fee_rate": 0.0002, "slippage": 0.0001},
        },
    },
    {
        "name": "F_high_cost_regime",
        "overrides": {
            "optimiser": {"name": "nevergrad", "budget": 10000, "seed": 7},
            "objective": {"name": "calmar", "min_trades": 1},
            "backtest": {"fee_rate": 0.0020, "slippage": 0.0010},
        },
    },
    {
        "name": "G_indicator_subset",
        "overrides": {
            "optimiser": {"name": "dehb", "budget": 10000, "seed": 99},
            "objective": {"name": "sharpe", "min_trades": 1},
            "indicators": {
                "include": ["sma", "ema", "rsi", "macd", "bb", "atr"],
                "param_ranges": {"rsi": {"period": [7, 14, 21]}, "bb": {"period": [20, 30], "std": [2.0, 2.5], "ma_type": ["sma", "ema"]}},
            },
        },
    },
]

print("Scenarios:")
for s in scenarios:
    print(" -", s["name"])


# %% 4. Execute scenarios
run_records: list[dict] = []
named_reports: dict[str, dict] = {}

for scenario in scenarios:
    print("Running:", scenario["name"])
    out_dir = showcase_dir / scenario["name"]
    out = run_optimiser(
        source=spy,
        config=base_cfg,
        overrides=scenario["overrides"],
        output_path=out_dir,
        run_walk_forward=False,
    )
    named_reports[scenario["name"]] = out["report"]

    with (out_dir / "effective_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(out["config"], f, sort_keys=False)

    summary = out["report"]["summary"]
    run_records.append(
        {
            "scenario": scenario["name"],
            "optimiser": summary.get("optimiser_name"),
            "objective": summary.get("objective"),
            "best_metric_name": summary.get("best_metric_name"),
            "best_metric_value": summary.get("best_metric_value"),
            "best_score": summary.get("best_score"),
            "evals": summary.get("n_evaluations"),
        }
    )


# %% 5. Summaries
helper_summary = summarise_reports(named_reports).sort_values("run").reset_index(drop=True)
scenario_summary = pd.DataFrame(run_records).sort_values("scenario").reset_index(drop=True)

print("Scenario summary:")
print(scenario_summary)

summary_csv = showcase_dir / "scenario_summary.csv"
helper_csv = showcase_dir / "helper_summary.csv"
scenario_summary.to_csv(summary_csv, index=False)
helper_summary.to_csv(helper_csv, index=False)
print("Saved:", summary_csv)
print("Saved:", helper_csv)


# %% 6. Equity curves via helper API
curve_png = showcase_dir / "equity_curves.png"
plot_equity_curves(named_reports, normalize=True, title="SPY showcase: optimisers/objectives (budget=10k)", save_path=curve_png)
print("Saved:", curve_png)


# %% 7. Metric comparison chart
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].bar(scenario_summary["scenario"], scenario_summary["best_metric_value"])
axes[0].set_title("Best metric value by scenario")
axes[0].set_ylabel("Best metric value")
axes[0].tick_params(axis="x", labelrotation=35)
axes[0].grid(axis="y", alpha=0.25)

axes[1].bar(scenario_summary["scenario"], scenario_summary["best_score"])
axes[1].set_title("Best score (minimised) by scenario")
axes[1].set_ylabel("Best score")
axes[1].tick_params(axis="x", labelrotation=35)
axes[1].grid(axis="y", alpha=0.25)

plt.tight_layout()
bars_png = showcase_dir / "metric_score_comparison.png"
fig.savefig(bars_png, dpi=140)
print("Saved:", bars_png)


# %% 8. CLI equivalents
print("\nCLI equivalents from project root:")
print("uv run cobra-py run --data <your_csv> --config configs/default.yaml --budget 10000")
print("uv run cobra-py run --data <your_csv> --objective sortino --budget 10000 --seed 42")
print("uv run cobra-py run --data <your_csv> --objective ulcer --budget 10000 --seed 42")

