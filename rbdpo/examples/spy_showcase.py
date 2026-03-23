from __future__ import annotations

# %% 1. Setup: imports and paths
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import yaml

from rbdpo.backtest.engine import run_backtest
from rbdpo.data.loader import load_ohlcv
from rbdpo.data.preprocessor import preprocess
from rbdpo.indicators.precompute import precompute_all
from rbdpo.indicators.registry import DEFAULT_REGISTRY
from rbdpo.reporting.report import generate_report
from rbdpo.search.dehb_runner import run_dehb
from rbdpo.search.space import build_config_space

try:
    project_root = Path(__file__).resolve().parents[1]
except NameError:  # pragma: no cover - interactive fallback
    cwd = Path.cwd()
    project_root = cwd if cwd.name == "rbdpo" else cwd / "rbdpo"

examples_dir = project_root / "examples"
showcase_dir = examples_dir / "showcase_results"
showcase_dir.mkdir(parents=True, exist_ok=True)

print("Project root:", project_root)
print("Showcase dir:", showcase_dir)


# %% 2. Helpers: config merge, metrics, and SPY loader
def deep_update(base: dict, updates: dict) -> dict:
    out = deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


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


def fetch_spy_ohlcv(start: str = "2018-01-01", end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    raw = yf.download("SPY", start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError("No SPY data returned. Check internet and ticker.")
    raw = raw.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    raw.index.name = "datetime"
    return raw[["open", "high", "low", "close", "volume"]]


def make_obj_cfg(cfg: dict) -> dict:
    return {
        "objective": cfg["objective"].get("name", "sharpe"),
        "composite_weights": cfg["objective"].get("composite_weights", [0.5, 0.3, 0.1, 0.1]),
        "complexity_penalty": cfg["objective"].get("complexity_penalty", 0.02),
        "min_trades": cfg["objective"].get("min_trades", 10),
        "n_entry_rules": int(cfg["policy"].get("n_entry_rules", 3)),
        "n_exit_rules": int(cfg["policy"].get("n_exit_rules", 1)),
    }


# %% 3. Load base config and fetch data once
with (project_root / "configs" / "default.yaml").open("r", encoding="utf-8") as f:
    base_cfg = yaml.safe_load(f)

spy = fetch_spy_ohlcv(start="2018-01-01", interval="1d")
print("SPY rows:", len(spy), "date range:", spy.index.min().date(), "to", spy.index.max().date())


# %% 4. Define scenarios (ways to run + what to compare)
scenarios: list[dict] = [
    {
        "name": "A_file_config_baseline",
        "overrides": {
            "objective": {"name": "sharpe", "min_trades": 1},
            "optimiser": {"budget": 25, "seed": 42},
            "indicators": {"n_jobs": 1},
        },
    },
    {
        "name": "B_low_fees",
        "overrides": {
            "objective": {"name": "sharpe", "min_trades": 1},
            "optimiser": {"budget": 25, "seed": 42},
            "backtest": {"fee_rate": 0.0002, "slippage": 0.0001},
            "indicators": {"n_jobs": 1},
        },
    },
    {
        "name": "C_high_fees",
        "overrides": {
            "objective": {"name": "sharpe", "min_trades": 1},
            "optimiser": {"budget": 25, "seed": 42},
            "backtest": {"fee_rate": 0.0020, "slippage": 0.0010},
            "indicators": {"n_jobs": 1},
        },
    },
    {
        "name": "D_calmar_objective",
        "overrides": {
            "objective": {"name": "calmar", "min_trades": 1},
            "optimiser": {"budget": 25, "seed": 123},
            "policy": {"n_entry_rules": 2, "n_exit_rules": 1},
            "indicators": {"n_jobs": 1},
        },
    },
]

print("Scenarios:")
for s in scenarios:
    print(" -", s["name"])


# %% 5. Core scenario run helper

def run_scenario(base_config: dict, scenario: dict, ohlcv: pd.DataFrame) -> dict:
    cfg = deep_update(base_config, scenario["overrides"])

    data = load_ohlcv(ohlcv, freq=cfg["data"].get("freq"), min_bars=int(cfg["data"].get("min_bars", 500)))
    train, _test = preprocess(data, cfg["data"])

    cache = precompute_all(train, DEFAULT_REGISTRY, n_jobs=int(cfg["indicators"].get("n_jobs", 1)))
    cs = build_config_space(
        n_entry_rules=int(cfg["policy"].get("n_entry_rules", 3)),
        n_exit_rules=int(cfg["policy"].get("n_exit_rules", 1)),
        registry=DEFAULT_REGISTRY,
        seed=int(cfg["optimiser"].get("seed", 42)),
    )

    result = run_dehb(
        cache=cache,
        data=train,
        config_space=cs,
        obj_config=make_obj_cfg(cfg),
        backtest_config=cfg["backtest"],
        budget=int(cfg["optimiser"].get("budget", 25)),
        seed=int(cfg["optimiser"].get("seed", 42)),
    )

    run_dir = showcase_dir / scenario["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    generate_report(result=result, wf_result=None, output_path=run_dir)

    strategy_eq = np.asarray(result.best_metrics.get("equity_curve", []), dtype=float)
    init_cash = float(cfg["backtest"].get("init_cash", 10000.0))
    close_train = train["close"].to_numpy(dtype=float)
    bh_eq = init_cash * (close_train / max(close_train[0], 1e-12))

    strategy_perf = compute_perf(strategy_eq)
    bh_perf = compute_perf(bh_eq)

    row = {
        "scenario": scenario["name"],
        "objective": cfg["objective"].get("name"),
        "seed": cfg["optimiser"].get("seed"),
        "budget": cfg["optimiser"].get("budget"),
        "fee_rate": cfg["backtest"].get("fee_rate"),
        "slippage": cfg["backtest"].get("slippage"),
        "strategy_total_return": strategy_perf["total_return"],
        "strategy_sharpe": strategy_perf["sharpe"],
        "strategy_max_dd": strategy_perf["max_drawdown"],
        "buyhold_total_return": bh_perf["total_return"],
        "buyhold_sharpe": bh_perf["sharpe"],
        "buyhold_max_dd": bh_perf["max_drawdown"],
        "best_score": result.best_score,
        "n_eval": result.n_evaluations,
    }

    with (run_dir / "effective_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return {
        "row": row,
        "index": train.index,
        "strategy_eq": strategy_eq,
        "buyhold_eq": bh_eq,
        "cfg": cfg,
        "dir": run_dir,
    }


# %% 6. Execute all scenarios (in-sample)
runs: list[dict] = []
for scenario in scenarios:
    print("Running:", scenario["name"])
    runs.append(run_scenario(base_cfg, scenario, spy))

summary = pd.DataFrame([r["row"] for r in runs]).sort_values("scenario").reset_index(drop=True)
summary


# %% 7. Compare to baseline: strategy vs buy-and-hold table
summary_vs_bh = summary.copy()
summary_vs_bh["excess_return_vs_bh"] = summary_vs_bh["strategy_total_return"] - summary_vs_bh["buyhold_total_return"]
summary_vs_bh["excess_sharpe_vs_bh"] = summary_vs_bh["strategy_sharpe"] - summary_vs_bh["buyhold_sharpe"]
summary_vs_bh[[
    "scenario",
    "fee_rate",
    "objective",
    "strategy_total_return",
    "buyhold_total_return",
    "excess_return_vs_bh",
    "strategy_sharpe",
    "buyhold_sharpe",
    "excess_sharpe_vs_bh",
]]


# %% 8. Results graph: equity curves vs buy-and-hold
fig, ax = plt.subplots(figsize=(12, 6))
for r in runs:
    eq = r["strategy_eq"]
    if len(eq) > 1:
        norm = eq / max(eq[0], 1e-12)
        ax.plot(r["index"][: len(norm)], norm, label=f"{r['row']['scenario']} strategy", linewidth=1.8)

bh = runs[0]["buyhold_eq"]
bh_norm = bh / max(bh[0], 1e-12)
ax.plot(runs[0]["index"][: len(bh_norm)], bh_norm, label="Buy & Hold", color="black", linestyle="--", linewidth=2.2)

ax.set_title("Normalized Equity Curves: COBRA-py Scenarios vs Buy & Hold")
ax.set_ylabel("Normalized Equity")
ax.set_xlabel("Date")
ax.grid(alpha=0.25)
ax.legend(loc="best")
plt.tight_layout()

curve_png = showcase_dir / "equity_curves_vs_buyhold.png"
fig.savefig(curve_png, dpi=140)
print("Saved:", curve_png)


# %% 9. Results graph: return and Sharpe bars
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(summary))
width = 0.38

axes[0].bar(x - width / 2, summary["strategy_total_return"], width, label="Strategy")
axes[0].bar(x + width / 2, summary["buyhold_total_return"], width, label="Buy & Hold")
axes[0].set_xticks(x)
axes[0].set_xticklabels(summary["scenario"], rotation=25, ha="right")
axes[0].set_title("Total Return Comparison")
axes[0].grid(axis="y", alpha=0.25)
axes[0].legend()

axes[1].bar(x - width / 2, summary["strategy_sharpe"], width, label="Strategy")
axes[1].bar(x + width / 2, summary["buyhold_sharpe"], width, label="Buy & Hold")
axes[1].set_xticks(x)
axes[1].set_xticklabels(summary["scenario"], rotation=25, ha="right")
axes[1].set_title("Sharpe Comparison")
axes[1].grid(axis="y", alpha=0.25)
axes[1].legend()

plt.tight_layout()
bars_png = showcase_dir / "return_sharpe_comparison.png"
fig2.savefig(bars_png, dpi=140)
print("Saved:", bars_png)


# %% 10. Mini walk-forward OOS comparison vs buy-and-hold

def walk_forward_oos(base_config: dict, scenario: dict, ohlcv: pd.DataFrame, n_splits: int = 3, train_pct: float = 0.7) -> dict:
    cfg = deep_update(base_config, scenario["overrides"])
    data = load_ohlcv(ohlcv, freq=cfg["data"].get("freq"), min_bars=int(cfg["data"].get("min_bars", 500)))

    fold_size = len(data) // n_splits
    rows: list[dict] = []

    for i in range(n_splits):
        start = i * fold_size
        end = len(data) if i == n_splits - 1 else (i + 1) * fold_size
        fold = data.iloc[start:end]
        if len(fold) < 80:
            continue

        split_idx = int(len(fold) * train_pct)
        train = fold.iloc[:split_idx]
        test = fold.iloc[split_idx:]
        if train.empty or test.empty:
            continue

        cache_train = precompute_all(train, DEFAULT_REGISTRY, n_jobs=int(cfg["indicators"].get("n_jobs", 1)))
        cs = build_config_space(
            n_entry_rules=int(cfg["policy"].get("n_entry_rules", 3)),
            n_exit_rules=int(cfg["policy"].get("n_exit_rules", 1)),
            registry=DEFAULT_REGISTRY,
            seed=int(cfg["optimiser"].get("seed", 42)) + i,
        )

        result = run_dehb(
            cache=cache_train,
            data=train,
            config_space=cs,
            obj_config=make_obj_cfg(cfg),
            backtest_config=cfg["backtest"],
            budget=max(10, int(cfg["optimiser"].get("budget", 25)) // 2),
            seed=int(cfg["optimiser"].get("seed", 42)) + i,
        )

        cache_test = precompute_all(test, DEFAULT_REGISTRY, n_jobs=int(cfg["indicators"].get("n_jobs", 1)))
        oos_metrics = run_backtest(result.best_policy, cache_test, test, cfg["backtest"])

        init_cash = float(cfg["backtest"].get("init_cash", 10000.0))
        bh_eq = init_cash * (test["close"].to_numpy(dtype=float) / max(float(test["close"].iloc[0]), 1e-12))
        bh_perf = compute_perf(bh_eq)

        rows.append(
            {
                "scenario": scenario["name"],
                "fold": i + 1,
                "oos_strategy_return": float(oos_metrics.get("total_return", -999.0)),
                "oos_strategy_sharpe": float(oos_metrics.get("sharpe_ratio", -999.0)),
                "oos_strategy_max_dd": float(oos_metrics.get("max_drawdown", -999.0)),
                "oos_bh_return": bh_perf["total_return"],
                "oos_bh_sharpe": bh_perf["sharpe"],
                "oos_bh_max_dd": bh_perf["max_drawdown"],
            }
        )

    fold_df = pd.DataFrame(rows)
    if fold_df.empty:
        return {"folds": fold_df, "mean": {"scenario": scenario["name"]}}

    mean_row = {
        "scenario": scenario["name"],
        "oos_strategy_return_mean": float(fold_df["oos_strategy_return"].mean()),
        "oos_bh_return_mean": float(fold_df["oos_bh_return"].mean()),
        "oos_strategy_sharpe_mean": float(fold_df["oos_strategy_sharpe"].mean()),
        "oos_bh_sharpe_mean": float(fold_df["oos_bh_sharpe"].mean()),
        "oos_excess_return_mean": float((fold_df["oos_strategy_return"] - fold_df["oos_bh_return"]).mean()),
    }
    return {"folds": fold_df, "mean": mean_row}


wf_runs = [walk_forward_oos(base_cfg, s, spy, n_splits=3, train_pct=0.7) for s in scenarios]
wf_mean = pd.DataFrame([x["mean"] for x in wf_runs]).sort_values("scenario").reset_index(drop=True)
wf_folds = pd.concat([x["folds"] for x in wf_runs if not x["folds"].empty], ignore_index=True)

print("OOS mean summary:")
print(wf_mean)


# %% 11. Persist summaries and show CLI equivalents
summary_csv = showcase_dir / "scenario_summary.csv"
summary_vs_bh_csv = showcase_dir / "scenario_summary_vs_buyhold.csv"
wf_mean_csv = showcase_dir / "walk_forward_oos_summary.csv"
wf_folds_csv = showcase_dir / "walk_forward_oos_folds.csv"

summary.to_csv(summary_csv, index=False)
summary_vs_bh.to_csv(summary_vs_bh_csv, index=False)
wf_mean.to_csv(wf_mean_csv, index=False)
wf_folds.to_csv(wf_folds_csv, index=False)

print("Saved:")
for p in [summary_csv, summary_vs_bh_csv, curve_png, bars_png, wf_mean_csv, wf_folds_csv]:
    print(" -", p)

print("\nCLI equivalents from project root:")
print("uv run cobra-py run --data <your_csv> --config configs/default.yaml")
print("uv run cobra-py run --data <your_csv> --objective calmar --seed 123 --budget 25")
print("uv run cobra-py sweep --data <your_csv> --seeds 42 123 999 --objective sharpe")
