from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt

from cobra_py.indicators.cache import IndicatorCache
from cobra_py.policy.rules import combine_rules_with_logic
from cobra_py.policy.schema import Policy
from cobra_py.policy.sl_tp import compute_sl, compute_tp

from .metrics import extract_metrics


def _validate_risk_levels(levels: np.ndarray, expected_len: int, label: str, level_type: str) -> None:
    if len(levels) != expected_len:
        raise ValueError(f"{label} returned length {len(levels)}, expected {expected_len}")
    if np.all(~np.isfinite(levels)):
        raise ValueError(
            f"{label} produced all-NaN levels. Ensure required indicators are precomputed for {level_type}."
        )


def _as_series(values: np.ndarray, index: pd.Index, name: str) -> pd.Series:
    return pd.Series(np.asarray(values), index=index, name=name)


def _stop_pct(close: np.ndarray, levels: np.ndarray, is_take_profit: bool) -> np.ndarray:
    close_safe = np.maximum(np.asarray(close, dtype=float), 1e-12)
    lvl = np.asarray(levels, dtype=float)
    if is_take_profit:
        pct = (lvl - close_safe) / close_safe
    else:
        pct = (close_safe - lvl) / close_safe
    pct = np.where(np.isfinite(pct) & (pct > 0.0), pct, np.nan)
    return pct


def _maybe_call(x: Any) -> Any:
    return x() if callable(x) else x


def run_backtest(policy: Policy, cache: IndicatorCache, data: pd.DataFrame, config: dict[str, Any] | None = None) -> dict:
    cfg = config or {}
    init_cash = float(cfg.get("init_cash", 10000.0))
    fee_rate = float(cfg.get("fee_rate", 0.001))
    slippage = float(cfg.get("slippage", 0.0005))
    risk_free_rate_annual = float(cfg.get("risk_free_rate_annual", 0.0))
    freq = str(cfg.get("freq", "1D"))

    close = data["close"].to_numpy(dtype=float)
    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)

    entry_signals = combine_rules_with_logic(policy.entry_rules, cache, close, logic=policy.entry_logic)
    exit_signals = (
        combine_rules_with_logic(policy.exit_rules, cache, close, logic=policy.exit_logic)
        if policy.exit_rules
        else np.zeros(len(close), dtype=bool)
    )

    sl_levels = compute_sl(policy.sl_config, cache, close, high, low)
    tp_levels = compute_tp(policy.tp_config, cache, close, high, sl_levels)
    _validate_risk_levels(sl_levels, len(close), "compute_sl", policy.sl_config.sl_type)
    _validate_risk_levels(tp_levels, len(close), "compute_tp", policy.tp_config.tp_type)

    index = data.index
    close_s = _as_series(close, index, "close")
    entries_s = _as_series(entry_signals.astype(bool), index, "entries")
    exits_s = _as_series(exit_signals.astype(bool), index, "exits")
    sl_stop_s = _as_series(_stop_pct(close, sl_levels, is_take_profit=False), index, "sl_stop")
    tp_stop_s = _as_series(_stop_pct(close, tp_levels, is_take_profit=True), index, "tp_stop")

    portfolio = vbt.Portfolio.from_signals(
        close=close_s,
        entries=entries_s,
        exits=exits_s,
        sl_stop=sl_stop_s,
        sl_trail=(policy.sl_config.sl_type == "trailing_atr"),
        tp_stop=tp_stop_s,
        init_cash=init_cash,
        fees=fee_rate,
        slippage=slippage,
        freq=freq,
        size=np.inf,
    )

    metrics = extract_metrics(portfolio, freq=freq, risk_free_rate_annual=risk_free_rate_annual)
    equity_obj = _maybe_call(portfolio.value)
    equity_curve = np.asarray(equity_obj.to_numpy(), dtype=float) if hasattr(equity_obj, "to_numpy") else np.asarray(equity_obj, dtype=float)
    records = portfolio.trades.records_readable
    if "Return" in records:
        trade_returns = np.asarray(records["Return"].to_numpy(), dtype=float)
    else:
        trade_returns_obj = _maybe_call(portfolio.trades.returns)
        if hasattr(trade_returns_obj, "to_pd"):
            trade_returns = np.asarray(trade_returns_obj.to_pd().dropna().to_numpy(), dtype=float)
        elif hasattr(trade_returns_obj, "to_numpy"):
            trade_returns = np.asarray(trade_returns_obj.to_numpy(), dtype=float)
        else:
            trade_returns = np.asarray(trade_returns_obj, dtype=float)
    metrics["equity_curve"] = equity_curve
    metrics["trade_returns"] = trade_returns
    return metrics

