from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cobra_py.indicators.cache import IndicatorCache
from cobra_py.policy.rules import combine_rules
from cobra_py.policy.schema import Policy
from cobra_py.policy.sl_tp import compute_sl, compute_tp

from .metrics import extract_metrics


def _bars_per_year(freq: str) -> float:
    if freq.endswith("H"):
        return 24.0 * 252.0
    if freq.endswith("T") or freq.endswith("min"):
        return 390.0 * 252.0
    return 252.0


def _simulate_single_position(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    entries: np.ndarray,
    exits: np.ndarray,
    sl_levels: np.ndarray,
    tp_levels: np.ndarray,
    init_cash: float,
    fee_rate: float,
    slippage: float,
    leverage: float,
    borrow_cost_rate: float,
    freq: str,
) -> dict:
    cash = float(init_cash)
    pos_qty = 0.0
    in_pos = False
    entry_price = 0.0
    borrowed_principal = 0.0
    accrued_borrow = 0.0
    stop = np.nan
    take = np.nan
    bars_per_year = _bars_per_year(freq)
    borrow_per_bar = max(float(borrow_cost_rate), 0.0) / max(bars_per_year, 1.0)
    leverage = max(float(leverage), 1.0)

    equity_curve = []
    trade_returns = []

    for i in range(len(close)):
        px = float(close[i])

        if not in_pos and bool(entries[i]):
            fill = px * (1.0 + slippage)
            gross_notional = cash * leverage
            fee = gross_notional * fee_rate
            equity_after_fee = cash - fee
            if equity_after_fee <= 0:
                equity_curve.append(cash)
                continue
            pos_qty = gross_notional / max(fill, 1e-12)
            borrowed_principal = max(gross_notional - equity_after_fee, 0.0)
            accrued_borrow = 0.0
            cash = 0.0
            in_pos = True
            entry_price = fill
            stop = sl_levels[i]
            take = tp_levels[i]

        elif in_pos:
            accrued_borrow += borrowed_principal * borrow_per_bar
            hit_stop = np.isfinite(stop) and low[i] <= stop
            hit_take = np.isfinite(take) and high[i] >= take
            explicit_exit = bool(exits[i])
            if hit_stop or hit_take or explicit_exit:
                if hit_stop:
                    exit_px = float(stop)
                elif hit_take:
                    exit_px = float(take)
                else:
                    exit_px = px
                exit_px *= (1.0 - slippage)
                gross = pos_qty * exit_px
                fee = gross * fee_rate
                cash = gross - fee - borrowed_principal - accrued_borrow
                trade_returns.append(cash / max(init_cash, 1e-12) - 1.0)
                pos_qty = 0.0
                in_pos = False
                entry_price = 0.0
                borrowed_principal = 0.0
                accrued_borrow = 0.0
                stop = np.nan
                take = np.nan

        equity = cash if not in_pos else pos_qty * px
        equity_curve.append(equity)

    if in_pos:
        accrued_borrow += borrowed_principal * borrow_per_bar
        final_px = close[-1] * (1.0 - slippage)
        gross = pos_qty * final_px
        fee = gross * fee_rate
        cash = gross - fee - borrowed_principal - accrued_borrow
        trade_returns.append(cash / max(init_cash, 1e-12) - 1.0)
        equity_curve[-1] = cash

    return {
        "equity_curve": np.asarray(equity_curve, dtype=float),
        "trade_returns": np.asarray(trade_returns, dtype=float),
        "n_trades": int(len(trade_returns)),
    }


def run_backtest(policy: Policy, cache: IndicatorCache, data: pd.DataFrame, config: dict[str, Any] | None = None) -> dict:
    cfg = config or {}
    init_cash = float(cfg.get("init_cash", 10000.0))
    fee_rate = float(cfg.get("fee_rate", 0.001))
    slippage = float(cfg.get("slippage", 0.0005))
    leverage = float(cfg.get("leverage", 1.0))
    borrow_cost_rate = float(cfg.get("borrow_cost_rate", 0.0))
    freq = str(cfg.get("freq", "1D"))

    close = data["close"].to_numpy(dtype=float)
    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)

    entry_signals = combine_rules(policy.entry_rules, cache, close)
    exit_signals = combine_rules(policy.exit_rules, cache, close) if policy.exit_rules else np.zeros(len(close), dtype=bool)

    sl_levels = compute_sl(policy.sl_config, cache, close, high, low)
    tp_levels = compute_tp(policy.tp_config, cache, close, high, sl_levels)

    raw = _simulate_single_position(
        close=close,
        high=high,
        low=low,
        entries=entry_signals,
        exits=exit_signals,
        sl_levels=sl_levels,
        tp_levels=tp_levels,
        init_cash=init_cash,
        fee_rate=fee_rate,
        slippage=slippage,
        leverage=leverage,
        borrow_cost_rate=borrow_cost_rate,
        freq=freq,
    )
    metrics = extract_metrics(raw, freq=freq)
    metrics["equity_curve"] = raw["equity_curve"]
    return metrics

