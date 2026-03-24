from __future__ import annotations

import numpy as np
import pandas as pd

from cobra_py.indicators.cache import IndicatorCache

from .schema import SLConfig, TPConfig


def _align_len(arr: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if len(x) == target_len:
        return x
    if len(x) > target_len:
        return x[-target_len:]
    out = np.full(target_len, np.nan, dtype=float)
    out[-len(x):] = x
    return out


def compute_sl(sl_config: SLConfig, cache: IndicatorCache, price: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
    t = sl_config.sl_type
    p = sl_config.params

    if t == "pct":
        pct = float(p[0])
        return price * (1.0 - pct)
    if t == "atr_mult":
        mult, period = float(p[0]), int(p[1])
        atr = cache.get("atr", (period,), "atr")
        if atr is None:
            return np.full_like(price, np.nan)
        atr = _align_len(atr, len(price))
        return price - mult * atr
    if t == "swing_low":
        lookback = int(p[0])
        return pd.Series(low).rolling(lookback).min().to_numpy()
    if t == "bb_lower":
        period, std, ma_type = int(p[0]), float(p[1]), str(p[2])
        lower = cache.get("bb", (period, std, ma_type), "lower")
        if lower is None:
            return np.full_like(price, np.nan)
        lower = _align_len(lower, len(price))
        return lower
    if t == "trailing_atr":
        mult, period = float(p[0]), int(p[1])
        atr = cache.get("atr", (period,), "atr")
        if atr is None:
            return np.full_like(price, np.nan)
        atr = _align_len(atr, len(price))
        return price - mult * atr
    return np.full_like(price, np.nan)


def compute_tp(tp_config: TPConfig, cache: IndicatorCache, price: np.ndarray, high: np.ndarray, sl_levels: np.ndarray) -> np.ndarray:
    t = tp_config.tp_type
    p = tp_config.params

    if t == "pct":
        pct = float(p[0])
        return price * (1.0 + pct)
    if t == "atr_mult":
        mult, period = float(p[0]), int(p[1])
        atr = cache.get("atr", (period,), "atr")
        if atr is None:
            return np.full_like(price, np.nan)
        atr = _align_len(atr, len(price))
        return price + mult * atr
    if t == "risk_reward":
        rr = float(p[0])
        return price + rr * (price - sl_levels)
    if t == "swing_high":
        lookback = int(p[0])
        return pd.Series(high).rolling(lookback).max().to_numpy()
    if t == "bb_upper":
        period, std, ma_type = int(p[0]), float(p[1]), str(p[2])
        upper = cache.get("bb", (period, std, ma_type), "upper")
        if upper is None:
            return np.full_like(price, np.nan)
        upper = _align_len(upper, len(price))
        return upper
    return np.full_like(price, np.nan)

