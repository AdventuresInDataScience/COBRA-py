from __future__ import annotations

import numpy as np
import pandas as pd

from rbdpo.indicators.cache import IndicatorCache

from .schema import RuleConfig


def _false_like(price: np.ndarray) -> np.ndarray:
    return np.zeros(len(price), dtype=bool)


def _safe(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
        return None
    x = np.asarray(arr, dtype=float)
    return np.nan_to_num(x, nan=np.nan)


def _align_len(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) == target_len:
        return arr
    if len(arr) > target_len:
        return arr[-target_len:]
    out = np.full(target_len, np.nan, dtype=float)
    out[-len(arr):] = arr
    return out


def _cmp(lhs: np.ndarray, rhs: np.ndarray | float, op: str) -> np.ndarray:
    if op == ">":
        out = lhs > rhs
    elif op == "<":
        out = lhs < rhs
    else:
        out = np.zeros_like(lhs, dtype=bool)
    out &= np.isfinite(lhs)
    if isinstance(rhs, np.ndarray):
        out &= np.isfinite(rhs)
    return out


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def evaluate_rule(rule: RuleConfig, cache: IndicatorCache, price: np.ndarray) -> np.ndarray:
    arr = _safe(cache.get(rule.indicator, rule.params, rule.output))
    if arr is None:
        return _false_like(price)
    arr = _align_len(arr, len(price))

    if rule.archetype == "comparison":
        if isinstance(rule.comparand, (int, float)):
            return _cmp(arr, float(rule.comparand), rule.operator)
        if rule.comparand == "price":
            return _cmp(price, arr, rule.operator)
        if rule.comparand == "indicator2" and rule.indicator2 and rule.params2 and rule.output2:
            arr2 = _safe(cache.get(rule.indicator2, rule.params2, rule.output2))
            if arr2 is None:
                return _false_like(price)
            arr2 = _align_len(arr2, len(price))
            return _cmp(arr, arr2, rule.operator)

    if rule.archetype == "crossover":
        if isinstance(rule.comparand, (int, float)):
            prev = np.roll(arr, 1)
            if rule.operator == "crosses_above":
                signal = (arr > float(rule.comparand)) & (prev <= float(rule.comparand))
            else:
                signal = (arr < float(rule.comparand)) & (prev >= float(rule.comparand))
            signal[0] = False
            signal &= np.isfinite(arr) & np.isfinite(prev)
            return signal
        if rule.comparand == "indicator2" and rule.indicator2 and rule.params2 and rule.output2:
            arr2 = _safe(cache.get(rule.indicator2, rule.params2, rule.output2))
            if arr2 is None:
                return _false_like(price)
            arr2 = _align_len(arr2, len(price))
            p1, p2 = np.roll(arr, 1), np.roll(arr2, 1)
            if rule.operator == "crosses_above":
                signal = (arr > arr2) & (p1 <= p2)
            else:
                signal = (arr < arr2) & (p1 >= p2)
            signal[0] = False
            signal &= np.isfinite(arr) & np.isfinite(arr2) & np.isfinite(p1) & np.isfinite(p2)
            return signal

    if rule.archetype == "band_test":
        side = rule.band_side or "upper"
        band = _safe(cache.get(rule.indicator, rule.params, side))
        if band is None:
            return _false_like(price)
        band = _align_len(band, len(price))
        if rule.operator in {">", "crosses_above"}:
            out = price > band
        else:
            out = price < band
        out &= np.isfinite(band)
        return out

    if rule.archetype == "pattern":
        lookback = int(rule.lookback or 20)
        s = pd.Series(price)
        if rule.operator == "nbar_high":
            rm = s.rolling(lookback).max().to_numpy()
            out = price >= rm
        elif rule.operator == "nbar_low":
            rl = s.rolling(lookback).min().to_numpy()
            out = price <= rl
        else:
            # consecutive positive bars as a default pattern primitive
            cond = (s.diff().fillna(0.0) > 0).astype(int)
            out = cond.rolling(lookback).sum().to_numpy() >= lookback
        return np.nan_to_num(out.astype(bool), nan=False)

    if rule.archetype == "derivative":
        diff = np.diff(arr, prepend=arr[0])
        threshold = _as_float(rule.comparand, 0.0)
        if rule.operator == ">":
            out = diff > threshold
        else:
            out = diff < threshold
        out &= np.isfinite(diff)
        return out

    if rule.archetype == "stat_test":
        lookback = int(rule.lookback or 20)
        s = pd.Series(arr)
        z = (s - s.rolling(lookback).mean()) / (s.rolling(lookback).std(ddof=0) + 1e-8)
        zarr = z.to_numpy()
        threshold = _as_float(rule.comparand, 0.0)
        if rule.operator == ">":
            out = zarr > threshold
        else:
            out = zarr < threshold
        out &= np.isfinite(zarr)
        return out

    return _false_like(price)


def combine_rules(rules: tuple[RuleConfig, ...], cache: IndicatorCache, price: np.ndarray) -> np.ndarray:
    if not rules:
        return np.zeros(len(price), dtype=bool)
    signals = [evaluate_rule(r, cache, price) for r in rules]
    return np.logical_and.reduce(signals)
