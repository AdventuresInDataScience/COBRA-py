# %% 1. Setup: imports and paths
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from cobra_py import fetch_yfinance_ohlcv, load_config, run_optimiser, summarise_reports


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

spy = fetch_yfinance_ohlcv("SPY", start="1996-01-01", interval="1d")
print("SPY rows:", len(spy), "date range:", spy.index.min().date(), "to", spy.index.max().date())


def compute_perf(equity_curve: np.ndarray) -> dict[str, float]:
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) < 2:
        return {"total_return": -999.0, "sharpe": -999.0, "max_drawdown": -999.0}

    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    total_return = float(eq[-1] / max(eq[0], 1e-12) - 1.0)
    sharpe = float((rets.mean() / (rets.std(ddof=0) + 1e-12)) * np.sqrt(252.0))
    peak = np.maximum.accumulate(eq)
    dd = eq / np.maximum(peak, 1e-12) - 1.0
    max_dd = float(dd.min())
    return {"total_return": total_return, "sharpe": sharpe, "max_drawdown": max_dd}


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
        "name": "C_dehb_max_return_10k",
        "overrides": {
            "optimiser": {"name": "dehb", "budget": 10000, "seed": 42},
            "objective": {"name": "max_return", "min_trades": 1},
        },
    },
    {
        "name": "D_dehb_max_return_dd20_10k",
        "overrides": {
            "optimiser": {"name": "dehb", "budget": 10000, "seed": 42},
            "objective": {"name": "max_return_dd_cap", "max_drawdown_cap": 0.20, "min_trades": 1},
        },
    },
    {
        "name": "E_tpe_ulcer_10k",
        "overrides": {
            "optimiser": {"name": "tpe", "budget": 10000, "seed": 42},
            "objective": {"name": "ulcer", "min_trades": 1},
        },
    },
    {
        "name": "F_dehb_composite_10k",
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
        "name": "G_low_cost_regime",
        "overrides": {
            "optimiser": {"name": "tpe", "budget": 10000, "seed": 7},
            "objective": {"name": "calmar", "min_trades": 1},
            "backtest": {"fee_rate": 0.0002, "slippage": 0.0001},
        },
    },
    {
        "name": "H_high_cost_regime",
        "overrides": {
            "optimiser": {"name": "nevergrad", "budget": 10000, "seed": 7},
            "objective": {"name": "calmar", "min_trades": 1},
            "backtest": {"fee_rate": 0.0020, "slippage": 0.0010},
        },
    },
    {
        "name": "I_indicator_subset",
        "overrides": {
            "optimiser": {"name": "dehb", "budget": 10000, "seed": 99},
            "objective": {"name": "sharpe", "min_trades": 1},
            "indicators": {
                "include": ["sma", "ema", "rsi", "macd", "bb", "atr"],
                "param_ranges": {"rsi": {"period": [7, 14, 21]}, "bb": {"period": [20, 30], "std": [2.0, 2.5], "ma_type": ["sma", "ema"]}},
            },
        },
    },
    {
        "name": "J_leverage2_dd20_with_borrow",
        "overrides": {
            "optimiser": {"name": "dehb", "budget": 10000, "seed": 21},
            "objective": {"name": "max_return_dd_cap", "max_drawdown_cap": 0.20, "min_trades": 1},
            "backtest": {"leverage": 2.0, "borrow_cost_rate": 0.06},
        },
    },
]

print("Scenarios:")
for s in scenarios:
    print(" -", s["name"])


# %% 4. Execute scenarios
run_records: list[dict] = []
named_reports: dict[str, dict] = {}
run_curves: dict[str, dict] = {}

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

    strategy_text = str(out["report"].get("policy_human_readable", ""))
    print(f"\nStrategy for {scenario['name']}:\n{strategy_text}\n")

    strategy_txt = out_dir / "strategy.txt"
    strategy_txt.write_text(strategy_text + "\n", encoding="utf-8")

    with (out_dir / "effective_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(out["config"], f, sort_keys=False)

    summary = out["report"]["summary"]
    strategy_eq = np.asarray(out["report"].get("best_metrics", {}).get("equity_curve", []), dtype=float)
    close_train = out["train"]["close"].to_numpy(dtype=float)
    init_cash = float(out["config"]["backtest"].get("init_cash", 10000.0))
    buyhold_eq = init_cash * (close_train / max(close_train[0], 1e-12))

    strategy_perf = compute_perf(strategy_eq)
    bh_perf = compute_perf(buyhold_eq)

    run_curves[scenario["name"]] = {
        "index": out["train"].index,
        "strategy_eq": strategy_eq,
        "buyhold_eq": buyhold_eq,
    }

    run_records.append(
        {
            "scenario": scenario["name"],
            "optimiser": summary.get("optimiser_name"),
            "objective": summary.get("objective"),
            "best_metric_name": summary.get("best_metric_name"),
            "best_metric_value": summary.get("best_metric_value"),
            "best_score": summary.get("best_score"),
            "evals": summary.get("n_evaluations"),
            "strategy_total_return": strategy_perf["total_return"],
            "strategy_sharpe": strategy_perf["sharpe"],
            "buyhold_total_return": bh_perf["total_return"],
            "buyhold_sharpe": bh_perf["sharpe"],
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
fig_eq, ax_eq = plt.subplots(figsize=(12, 6))

for name, curves in run_curves.items():
    eq = curves["strategy_eq"]
    if len(eq) < 2:
        continue
    norm = eq / max(eq[0], 1e-12)
    idx = curves["index"][: len(norm)]
    ax_eq.plot(idx, norm, linewidth=1.6, label=f"{name} strategy")

first_key = next(iter(run_curves))
bh = run_curves[first_key]["buyhold_eq"]
bh_norm = bh / max(bh[0], 1e-12)
bh_idx = run_curves[first_key]["index"][: len(bh_norm)]
ax_eq.plot(bh_idx, bh_norm, color="black", linestyle="--", linewidth=2.2, label="Buy & Hold baseline")

ax_eq.set_title("SPY showcase: date-based normalized equity (1996+)")
ax_eq.set_xlabel("Date")
ax_eq.set_ylabel("Normalized equity")
ax_eq.grid(alpha=0.25)
ax_eq.legend(loc="best", fontsize=8)
fig_eq.tight_layout()
fig_eq.savefig(curve_png, dpi=140)
print("Saved:", curve_png)


# %% 7. Metric comparison chart (with buy-and-hold baseline)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

x = np.arange(len(scenario_summary))
width = 0.38

axes[0].bar(x - width / 2, scenario_summary["strategy_total_return"], width, label="Strategy")
axes[0].bar(x + width / 2, scenario_summary["buyhold_total_return"], width, label="Buy & Hold")
axes[0].set_xticks(x)
axes[0].set_xticklabels(scenario_summary["scenario"], rotation=35, ha="right")
axes[0].set_title("Total return: strategy vs buy-and-hold")
axes[0].set_ylabel("Total return")
axes[0].grid(axis="y", alpha=0.25)
axes[0].legend()

axes[1].bar(x - width / 2, scenario_summary["strategy_sharpe"], width, label="Strategy")
axes[1].bar(x + width / 2, scenario_summary["buyhold_sharpe"], width, label="Buy & Hold")
axes[1].set_xticks(x)
axes[1].set_xticklabels(scenario_summary["scenario"], rotation=35, ha="right")
axes[1].set_title("Sharpe: strategy vs buy-and-hold")
axes[1].set_ylabel("Sharpe")
axes[1].grid(axis="y", alpha=0.25)
axes[1].legend()

plt.tight_layout()
bars_png = showcase_dir / "metric_score_comparison.png"
fig.savefig(bars_png, dpi=140)
print("Saved:", bars_png)


# %% 8. CLI equivalents
print("\nCLI equivalents from project root:")
print("uv run cobra-py run --data <your_csv> --config configs/default.yaml --budget 10000")
print("uv run cobra-py run --data <your_csv> --objective sortino --budget 10000 --seed 42")
print("uv run cobra-py run --data <your_csv> --objective ulcer --budget 10000 --seed 42")
print("uv run cobra-py run --data <your_csv> --objective max_return --budget 10000 --seed 42")

