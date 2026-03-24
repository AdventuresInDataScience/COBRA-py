from __future__ import annotations

from pathlib import Path

from cobra_py import fetch_yfinance_ohlcv, load_config, run_optimiser

cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
spy = fetch_yfinance_ohlcv("SPY", start="2018-01-01", interval="1d")

out = run_optimiser(
    source=spy,
    config=cfg,
    overrides={
        "optimiser": {"name": "dehb", "budget": 10000, "seed": 42},
        "objective": {"name": "sortino", "min_trades": 1},
    },
    output_path=Path(__file__).resolve().parents[0] / "quick_api_results",
)

summary = out["report"]["summary"]
print("optimiser:", summary["optimiser_name"])
print("objective:", summary["objective"])
print("best metric:", summary["best_metric_name"], summary["best_metric_value"])
print("best score:", summary["best_score"])
