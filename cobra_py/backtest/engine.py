"""Backtesting engine using vectorbt as per spec Phase 6 (section 8.1).

Uses vbt.Portfolio.from_signals() for portfolio simulation and metric extraction.
Extends the spec with leverage and borrow-cost support added by the user.
"""
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


def _levels_to_pct_stop(levels: np.ndarray, close: np.ndarray, direction: str) -> np.ndarray:
    """Convert absolute price levels to fractional stop distances from close.

    For SL (direction='below'): pct = (close - sl_level) / close  (positive fraction below close)
    For TP (direction='above'): pct = (tp_level - close) / close  (positive fraction above close)

    vectorbt uses these as percentage offsets from entry price.
    NaN levels are preserved as NaN (no stop at that bar).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        if direction == "below":
            pct = (close - levels) / close
        else:
            pct = (levels - close) / close
    # Clamp negative percentages to a tiny positive value (invalid stop direction)
    pct = np.where(np.isfinite(pct) & (pct > 0), pct, np.nan)
    return pct


def _bars_per_year(freq: str) -> float:
    if freq.endswith("H"):
        return 24.0 * 252.0
    if freq.endswith("T") or freq.endswith("min"):
        return 390.0 * 252.0
    return 252.0


def run_backtest(
    policy: Policy,
    cache: IndicatorCache,
    data: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict:
    """Run a vectorbt-based backtest for the given policy on OHLCV data.

    Parameters
    ----------
    policy : Policy
        Decoded strategy policy with entry/exit rules and SL/TP config.
    cache : IndicatorCache
        Precomputed indicator values.
    data : pd.DataFrame
        OHLCV data with DatetimeIndex.
    config : dict, optional
        Backtest configuration (init_cash, fee_rate, slippage, freq, leverage, etc.)

    Returns
    -------
    dict
        Metrics dictionary including equity_curve and trade_returns.
    """
    cfg = config or {}
    init_cash = float(cfg.get("init_cash", 10000.0))
    fee_rate = float(cfg.get("fee_rate", 0.001))
    slippage = float(cfg.get("slippage", 0.0005))
    leverage = float(cfg.get("leverage", 1.0))
    borrow_cost_rate = float(cfg.get("borrow_cost_rate", 0.0))
    risk_free_rate_annual = float(cfg.get("risk_free_rate_annual", 0.0))
    freq = str(cfg.get("freq", "1D"))

    close = data["close"].to_numpy(dtype=float)
    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)

    # ---- 1. Generate entry/exit signals from rules ----
    entry_signals = combine_rules_with_logic(
        policy.entry_rules, cache, close, logic=policy.entry_logic
    )
    exit_signals = (
        combine_rules_with_logic(
            policy.exit_rules, cache, close, logic=policy.exit_logic
        )
        if policy.exit_rules
        else np.zeros(len(close), dtype=bool)
    )

    # ---- 2. Compute SL/TP price levels ----
    sl_levels = compute_sl(policy.sl_config, cache, close, high, low)
    tp_levels = compute_tp(policy.tp_config, cache, close, high, sl_levels)
    _validate_risk_levels(sl_levels, len(close), "compute_sl", policy.sl_config.sl_type)
    _validate_risk_levels(tp_levels, len(close), "compute_tp", policy.tp_config.tp_type)

    # ---- 3. Convert to pd.Series with DatetimeIndex (vectorbt requires this) ----
    idx = data.index
    close_s = pd.Series(close, index=idx, name="close")
    entry_s = pd.Series(entry_signals, index=idx, name="entries")
    exit_s = pd.Series(exit_signals, index=idx, name="exits")

    # ---- 4. Convert SL/TP levels to percentage stops for vectorbt ----
    sl_pct = _levels_to_pct_stop(sl_levels, close, direction="below")
    tp_pct = _levels_to_pct_stop(tp_levels, close, direction="above")

    sl_pct_s = pd.Series(sl_pct, index=idx)
    tp_pct_s = pd.Series(tp_pct, index=idx)

    is_trailing = policy.sl_config.sl_type == "trailing_atr"

    # ---- 5. Build portfolio via vectorbt ----
    # Determine position sizing for leverage
    leverage = max(leverage, 1.0)
    size = leverage  # fraction of portfolio value to invest
    size_type = "percent"

    try:
        portfolio = vbt.Portfolio.from_signals(
            close_s,
            entries=entry_s,
            exits=exit_s,
            freq=freq,
            init_cash=init_cash,
            size=size,
            size_type=size_type,
            fees=fee_rate,
            slippage=slippage,
            sl_stop=sl_pct_s,
            sl_trail=is_trailing,
            tp_stop=tp_pct_s,
            accumulate=False,
        )
    except TypeError:
        # Fallback for older vectorbt versions that don't support sl_stop/tp_stop
        # in from_signals — pre-compute exit signals manually
        combined_exits = _precompute_sl_tp_exits(
            close, high, low, entry_signals, exit_signals, sl_levels, tp_levels
        )
        combined_exit_s = pd.Series(combined_exits, index=idx, name="exits")
        portfolio = vbt.Portfolio.from_signals(
            close_s,
            entries=entry_s,
            exits=combined_exit_s,
            freq=freq,
            init_cash=init_cash,
            size=size,
            size_type=size_type,
            fees=fee_rate,
            slippage=slippage,
            accumulate=False,
        )

    # ---- 6. Extract metrics from vectorbt portfolio ----
    metrics = extract_metrics(
        portfolio,
        freq=freq,
        risk_free_rate_annual=risk_free_rate_annual,
        leverage=leverage,
        borrow_cost_rate=borrow_cost_rate,
    )
    return metrics


def _precompute_sl_tp_exits(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    entries: np.ndarray,
    exits: np.ndarray,
    sl_levels: np.ndarray,
    tp_levels: np.ndarray,
) -> np.ndarray:
    """Pre-compute combined exit signals including SL/TP triggers.

    Used as a fallback when vectorbt version does not support sl_stop/tp_stop
    in from_signals(). Tracks position state to know when to check SL/TP.
    """
    n = len(close)
    combined = np.copy(exits)
    in_pos = False
    sl = np.nan
    tp = np.nan

    for i in range(n):
        if not in_pos and entries[i]:
            in_pos = True
            sl = sl_levels[i]
            tp = tp_levels[i]
        elif in_pos:
            hit_sl = np.isfinite(sl) and low[i] <= sl
            hit_tp = np.isfinite(tp) and high[i] >= tp
            if hit_sl or hit_tp or exits[i]:
                combined[i] = True
                in_pos = False
        if combined[i] and in_pos:
            in_pos = False

    return combined
