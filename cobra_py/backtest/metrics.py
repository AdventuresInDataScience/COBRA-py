"""Performance metric extraction from vectorbt portfolio objects.

Implements spec Phase 6 section 8.2: extract_metrics(portfolio) -> dict.
Extends with ulcer index, CAGR, and borrow-cost adjustments for leverage.
"""
from __future__ import annotations

import numpy as np


SENTINEL_BAD = -999.0


def _safe(x: float) -> float:
    """Replace non-finite values with sentinel."""
    if not np.isfinite(x):
        return SENTINEL_BAD
    return float(x)


def _ulcer_index(equity: np.ndarray) -> float:
    """Root-mean-square of drawdown percentages (Ulcer Index)."""
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


def _max_drawdown_from_equity(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min())


def extract_metrics(
    portfolio,
    freq: str = "1D",
    risk_free_rate_annual: float = 0.0,
    leverage: float = 1.0,
    borrow_cost_rate: float = 0.0,
) -> dict:
    """Extract performance metrics from a vectorbt Portfolio object.

    Uses vectorbt's native methods where available, with manual computation
    for metrics VBT doesn't provide (ulcer index, CAGR, CAR/MDD, borrow costs).
    """
    # -- Core metrics from vectorbt --
    try:
        total_return = float(portfolio.total_return())
    except Exception:
        total_return = SENTINEL_BAD

    try:
        sharpe = float(portfolio.sharpe_ratio())
    except Exception:
        sharpe = SENTINEL_BAD

    try:
        sortino = float(portfolio.sortino_ratio())
    except Exception:
        sortino = SENTINEL_BAD

    try:
        max_dd = float(portfolio.max_drawdown())
    except Exception:
        max_dd = SENTINEL_BAD

    try:
        n_trades = int(portfolio.trades.count())
    except Exception:
        n_trades = 0

    try:
        win_rate = float(portfolio.trades.win_rate())
    except Exception:
        win_rate = np.nan

    try:
        trade_returns = portfolio.trades.returns.values
        avg_return_per_trade = float(trade_returns.mean()) if len(trade_returns) > 0 else np.nan
    except Exception:
        trade_returns = np.array([], dtype=float)
        avg_return_per_trade = np.nan

    # -- Equity curve and derived metrics --
    try:
        equity = portfolio.value().to_numpy(dtype=float)
    except Exception:
        equity = np.array([10000.0], dtype=float)

    ann = _annualisation_factor(freq)

    # Borrow cost adjustment for leveraged positions
    if leverage > 1.0 and borrow_cost_rate > 0.0 and len(equity) > 1:
        # Apply per-bar borrow cost on the borrowed portion of each position
        borrow_per_bar = borrow_cost_rate / max(ann, 1.0)
        borrowed_fraction = (leverage - 1.0) / leverage
        # Approximate: reduce equity by borrow cost on borrowed notional each bar
        for i in range(1, len(equity)):
            cost = equity[i] * borrowed_fraction * borrow_per_bar
            equity[i] = max(equity[i] - cost, 0.0)

    # Recompute return-based metrics on (possibly borrow-adjusted) equity
    if len(equity) >= 2:
        cagr_val = _cagr(equity, ann)
        max_dd_adj = _max_drawdown_from_equity(equity)
        ulcer = _ulcer_index(equity)

        # Calmar: trailing 3-year window
        trailing_bars = max(int(round(3.0 * ann)), 2)
        if len(equity) > trailing_bars:
            trailing_equity = equity[-(trailing_bars + 1):]
        else:
            trailing_equity = equity
        trailing_cagr = _cagr(trailing_equity, ann)
        trailing_max_dd = _max_drawdown_from_equity(trailing_equity)
        calmar = _safe_ratio(trailing_cagr, abs(trailing_max_dd))
        car_mdd = _safe_ratio(cagr_val, abs(max_dd_adj))

        # Recompute Sharpe/Sortino on adjusted equity if borrow costs applied
        if leverage > 1.0 and borrow_cost_rate > 0.0:
            prev_eq = equity[:-1]
            rets = np.divide(
                np.diff(equity), prev_eq,
                out=np.zeros_like(prev_eq, dtype=float),
                where=prev_eq > 1e-12,
            )
            rf_per_bar = (1.0 + risk_free_rate_annual) ** (1.0 / max(ann, 1.0)) - 1.0
            excess = rets - rf_per_bar
            mean_excess = float(excess.mean())
            std_excess = float(excess.std(ddof=0))
            downside = np.minimum(excess, 0.0)
            downside_dev = float(np.sqrt(np.mean(np.square(downside))))
            sharpe = _safe_ratio(mean_excess, std_excess) * np.sqrt(ann)
            sortino = _safe_ratio(mean_excess, downside_dev) * np.sqrt(ann)
            total_return = equity[-1] / max(equity[0], 1e-12) - 1.0
            max_dd = max_dd_adj
    else:
        cagr_val = SENTINEL_BAD
        calmar = SENTINEL_BAD
        car_mdd = SENTINEL_BAD
        ulcer = SENTINEL_BAD

    return {
        "total_return": _safe(total_return),
        "cagr": _safe(cagr_val),
        "sharpe_ratio": _safe(sharpe),
        "calmar_ratio": _safe(calmar),
        "car_mdd_ratio": _safe(car_mdd),
        "sortino_ratio": _safe(sortino),
        "ulcer_index": _safe(ulcer),
        "max_drawdown": _safe(max_dd),
        "n_trades": n_trades,
        "win_rate": _safe(win_rate),
        "avg_return_per_trade": _safe(avg_return_per_trade),
        "equity_curve": equity,
        "trade_returns": trade_returns,
    }
