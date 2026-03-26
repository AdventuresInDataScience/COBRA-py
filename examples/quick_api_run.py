from __future__ import annotations

from pathlib import Path

from cobra_py import fetch_yfinance_ohlcv, find_strategy, load_config

cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
spy = fetch_yfinance_ohlcv("SPY", start="2018-01-01", interval="1d")

result = find_strategy(
    source=spy,
    config=cfg,
    overrides={
        "optimiser": {"name": "dehb", "budget": 10000, "seed": 42},
        "objective": {"name": "sortino", "min_trades": 1},
    },
    output_path=Path(__file__).resolve().parents[0] / "quick_api_results",
)

summary = result.report["summary"]
print("optimiser:", result.optimiser)
print("objective:", result.objective)
print("best metric:", summary["best_metric_name"], summary["best_metric_value"])
print("best score:", summary["best_score"])
print("equity points:", len(result.equity_curve))
