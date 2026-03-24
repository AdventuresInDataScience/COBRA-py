from __future__ import annotations

import numpy as np


SENTINEL_BAD = -999.0


def _safe(x: float) -> float:
    if not np.isfinite(x):
        return SENTINEL_BAD
    return float(x)


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min())


def _ulcer_index(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / np.maximum(peak, 1e-12) - 1.0
    drawdown_mag = np.minimum(dd, 0.0)
    return float(np.sqrt(np.mean(np.square(drawdown_mag))))


def _annualisation_factor(freq: str) -> float:
    if freq.endswith("H"):
        return 24.0 * 252.0
    if freq.endswith("T") or freq.endswith("min"):
        return 390.0 * 252.0
    return 252.0


def extract_metrics(results: dict, freq: str = "1D") -> dict:
    equity = np.asarray(results.get("equity_curve", []), dtype=float)
    trade_returns = np.asarray(results.get("trade_returns", []), dtype=float)
    n_trades = int(results.get("n_trades", 0))

    if len(equity) < 2:
        return {
            "total_return": SENTINEL_BAD,
            "sharpe_ratio": SENTINEL_BAD,
            "calmar_ratio": SENTINEL_BAD,
            "sortino_ratio": SENTINEL_BAD,
            "ulcer_index": SENTINEL_BAD,
            "max_drawdown": SENTINEL_BAD,
            "n_trades": 0,
            "win_rate": SENTINEL_BAD,
            "avg_return_per_trade": SENTINEL_BAD,
        }

    rets = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    ann = _annualisation_factor(freq)
    mean = rets.mean()
    std = rets.std(ddof=0)
    downside = rets[rets < 0]
    down_std = downside.std(ddof=0) if len(downside) else 0.0

    total_return = equity[-1] / max(equity[0], 1e-12) - 1.0
    max_dd = _max_drawdown(equity)
    ulcer = _ulcer_index(equity)

    sharpe = (mean / (std + 1e-12)) * np.sqrt(ann)
    sortino = (mean / (down_std + 1e-12)) * np.sqrt(ann)
    calmar = (mean * ann) / (abs(max_dd) + 1e-12)

    win_rate = float((trade_returns > 0).mean()) if len(trade_returns) else np.nan
    avg_trade = float(trade_returns.mean()) if len(trade_returns) else np.nan

    return {
        "total_return": _safe(total_return),
        "sharpe_ratio": _safe(sharpe),
        "calmar_ratio": _safe(calmar),
        "sortino_ratio": _safe(sortino),
        "ulcer_index": _safe(ulcer),
        "max_drawdown": _safe(max_dd),
        "n_trades": n_trades,
        "win_rate": _safe(win_rate),
        "avg_return_per_trade": _safe(avg_trade),
    }
