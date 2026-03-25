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


def _safe_ratio(numerator: float, denominator: float, eps: float = 1e-12) -> float:
    if abs(denominator) <= eps:
        return np.nan
    return float(numerator / denominator)


def _cagr(equity: np.ndarray, ann_factor: float) -> float:
    years = max((len(equity) - 1) / max(ann_factor, 1.0), 1e-12)
    if equity[-1] <= 0.0 or equity[0] <= 0.0:
        return -1.0
    return float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0)


def extract_metrics(results: dict, freq: str = "1D", risk_free_rate_annual: float = 0.0) -> dict:
    equity = np.asarray(results.get("equity_curve", []), dtype=float)
    trade_returns = np.asarray(results.get("trade_returns", []), dtype=float)
    n_trades = int(results.get("n_trades", 0))

    if len(equity) < 2:
        return {
            "total_return": SENTINEL_BAD,
            "cagr": SENTINEL_BAD,
            "sharpe_ratio": SENTINEL_BAD,
            "calmar_ratio": SENTINEL_BAD,
            "car_mdd_ratio": SENTINEL_BAD,
            "sortino_ratio": SENTINEL_BAD,
            "ulcer_index": SENTINEL_BAD,
            "max_drawdown": SENTINEL_BAD,
            "n_trades": 0,
            "win_rate": SENTINEL_BAD,
            "avg_return_per_trade": SENTINEL_BAD,
        }

    prev_equity = equity[:-1]
    rets = np.divide(
        np.diff(equity),
        prev_equity,
        out=np.zeros_like(prev_equity, dtype=float),
        where=prev_equity > 1e-12,
    )
    ann = _annualisation_factor(freq)
    rf_annual = float(risk_free_rate_annual)
    rf_per_bar = (1.0 + rf_annual) ** (1.0 / max(ann, 1.0)) - 1.0
    excess_rets = rets - rf_per_bar
    mean_excess = float(excess_rets.mean())
    std_excess = float(excess_rets.std(ddof=0))
    downside = np.minimum(excess_rets, 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside))))

    total_return = equity[-1] / max(equity[0], 1e-12) - 1.0
    max_dd = _max_drawdown(equity)
    ulcer = _ulcer_index(equity)
    cagr = _cagr(equity, ann)

    # Calmar ratio conventionally uses a trailing 36-month horizon.
    trailing_bars = max(int(round(3.0 * ann)), 2)
    if len(equity) > trailing_bars:
        trailing_equity = equity[-(trailing_bars + 1) :]
    else:
        trailing_equity = equity

    trailing_cagr = _cagr(trailing_equity, ann)
    trailing_max_dd = _max_drawdown(trailing_equity)

    sharpe = _safe_ratio(mean_excess, std_excess) * np.sqrt(ann)
    sortino = _safe_ratio(mean_excess, downside_dev) * np.sqrt(ann)
    calmar = _safe_ratio(trailing_cagr, abs(trailing_max_dd))
    car_mdd = _safe_ratio(cagr, abs(max_dd))

    win_rate = float((trade_returns > 0).mean()) if len(trade_returns) else np.nan
    avg_trade = float(trade_returns.mean()) if len(trade_returns) else np.nan

    return {
        "total_return": _safe(total_return),
        "cagr": _safe(cagr),
        "sharpe_ratio": _safe(sharpe),
        "calmar_ratio": _safe(calmar),
        "car_mdd_ratio": _safe(car_mdd),
        "sortino_ratio": _safe(sortino),
        "ulcer_index": _safe(ulcer),
        "max_drawdown": _safe(max_dd),
        "n_trades": n_trades,
        "win_rate": _safe(win_rate),
        "avg_return_per_trade": _safe(avg_trade),
    }
