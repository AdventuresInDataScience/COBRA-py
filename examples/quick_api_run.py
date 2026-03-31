from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from cobra_py.backtest.metrics import extract_metrics
from cobra_py import fetch_yfinance_ohlcv, find_strategy, load_config


OBJECTIVE_TO_METRIC_KEY = {
    "sharpe": "sharpe_ratio",
    "calmar": "calmar_ratio",
    "car_mdd": "car_mdd_ratio",
    "cagr": "cagr",
    "sortino": "sortino_ratio",
    "ulcer": "ulcer_index",
    "max_return": "total_return",
    "max_return_dd_cap": "total_return",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _fmt_period(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "(no data)"
    return f"{df.index.min().date()} -> {df.index.max().date()} ({len(df)} bars)"


def _equity_series(values: object, index: pd.Index) -> pd.Series:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return pd.Series(dtype=float, index=index[:0], name="equity")
    if len(arr) == len(index):
        eq_index = index
    elif len(arr) < len(index):
        eq_index = index[-len(arr) :]
    else:
        eq_index = pd.RangeIndex(len(arr))
    return pd.Series(arr, index=eq_index, name="equity")


def _save_is_oos_chart(result, output_dir: Path) -> Path | None:
    if result.train_data is None or result.train_data.empty:
        return None
    if result.test_data is None or result.test_data.empty:
        return None
    if not result.oos_metrics:
        return None

    train_eq = _equity_series(result.metrics.get("equity_curve", []), result.train_data.index)
    test_eq = _equity_series(result.oos_metrics.get("equity_curve", []), result.test_data.index)
    if train_eq.empty or test_eq.empty:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    # Rescale OOS equity so the stitched curve is visually continuous at the split.
    scale = float(train_eq.iloc[-1]) / max(float(test_eq.iloc[0]), 1e-12)
    test_scaled = test_eq * scale
    combined = pd.concat([train_eq, test_scaled])
    split_time = test_scaled.index[0]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(combined.index, combined.values, lw=1.8, color="#1565c0", label="strategy equity")
    ax.axvline(split_time, color="#d32f2f", lw=1.5, ls="--", label="IS/OOS split")
    ax.set_title("Best Strategy Equity (In-sample + Out-of-sample)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    chart_path = output_dir / "equity_is_oos_split.png"
    fig.savefig(chart_path, dpi=140)
    plt.close(fig)
    return chart_path


def _buy_hold_metric(df: pd.DataFrame | None, metric_key: str, freq: str, rf_annual: float) -> float | None:
    if df is None or df.empty:
        return None
    metrics = extract_metrics(
        {
            "equity_curve": df["close"].to_numpy(dtype=float),
            "trade_returns": [],
            "n_trades": 0,
        },
        freq=freq,
        risk_free_rate_annual=rf_annual,
    )
    value = metrics.get(metric_key)
    return float(value) if value is not None else None

cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
spy = fetch_yfinance_ohlcv("SPY", start="2018-01-01", interval="1d")

quick_budget = int(os.getenv("COBRA_EXAMPLE_BUDGET", str(cfg["optimiser"].get("budget", 300))))
quick_seed = int(os.getenv("COBRA_EXAMPLE_SEED", str(cfg["optimiser"].get("seed", 42))))
quick_objective = str(os.getenv("COBRA_OBJECTIVE", "cagr")).strip().lower()
quick_min_trades = int(os.getenv("COBRA_MIN_TRADES", str(cfg["objective"].get("min_trades", 10))))
complexity_penalty = float(
    os.getenv("COBRA_COMPLEXITY_PENALTY", str(cfg["objective"].get("complexity_penalty", 0.02)))
)
run_walk_forward = _env_bool("COBRA_RUN_WALK_FORWARD", bool(cfg.get("validation", {}).get("walk_forward", True)))

result = find_strategy(
    source=spy,
    config=cfg,
    overrides={
        "optimiser": {"name": "tpe", "budget": max(100, quick_budget), "seed": quick_seed},
        "objective": {
            "name": quick_objective,
            "min_trades": quick_min_trades,
            "complexity_penalty": complexity_penalty,
        },
    },
    output_path=Path(__file__).resolve().parents[0] / "quick_api_results",
    run_walk_forward=run_walk_forward,
    evaluate_oos=True,
)

summary = result.report["summary"]
metric_key = OBJECTIVE_TO_METRIC_KEY.get(str(summary.get("objective", "")).lower())
metric_name = str(summary.get("best_metric_name", metric_key or "metric"))

is_metric = result.metrics.get(metric_key) if metric_key else summary.get("best_metric_value")
oos_metric = result.oos_metrics.get(metric_key) if result.oos_metrics and metric_key else None
bh_is_metric = None
bh_oos_metric = None
if metric_key:
    bt_cfg = result.config.get("backtest", {}) if isinstance(result.config, dict) else {}
    freq = str(bt_cfg.get("freq", "1D"))
    rf_annual = float(bt_cfg.get("risk_free_rate_annual", 0.0))
    bh_is_metric = _buy_hold_metric(result.train_data, metric_key, freq, rf_annual)
    bh_oos_metric = _buy_hold_metric(result.test_data, metric_key, freq, rf_annual)

print("optimiser:", result.optimiser)
print("objective:", result.objective)
print("complexity_penalty:", complexity_penalty)
print("min_trades:", quick_min_trades)
print("walk_forward_enabled:", run_walk_forward)
print("best metric:", summary["best_metric_name"], summary["best_metric_value"])
print("best score:", summary["best_score"])
print("equity points:", len(result.equity_curve))
print("in-sample period:", _fmt_period(result.train_data))
print("out-of-sample period:", _fmt_period(result.test_data))
print(f"in-sample {metric_name}:", is_metric)
print(f"out-of-sample {metric_name}:", oos_metric)
if metric_key:
    print(f"buy-and-hold in-sample {metric_name}:", bh_is_metric)
    print(f"buy-and-hold out-of-sample {metric_name}:", bh_oos_metric)
print("\nBest rules:\n")
print(result.rules)

wf = result.walk_forward
if wf is not None:
    print("\nWalk-forward summary:")
    print("  folds completed:", f"{wf.n_completed_folds}/{wf.n_requested_folds}")
    print("  folds skipped:", wf.n_skipped_folds)
    print("  oos_sharpe_mean:", wf.oos_sharpe_mean)
    print("  oos_sharpe_std:", wf.oos_sharpe_std)
    print("  oos_calmar_mean:", wf.oos_calmar_mean)
    print("  oos_return_mean:", wf.oos_return_mean)
    print("  oos_max_drawdown_mean:", wf.oos_max_drawdown_mean)

chart_file = _save_is_oos_chart(result, Path(__file__).resolve().parents[0] / "quick_api_results")
if chart_file is not None:
    print("saved split chart:", chart_file)
else:
    print("split chart not generated (missing OOS data or matplotlib)")
