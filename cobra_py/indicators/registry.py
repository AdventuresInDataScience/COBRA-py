from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class IndicatorDef:
    name: str
    param_grid: dict[str, list]
    outputs: list[str]
    compute_fn: Callable[..., dict[str, np.ndarray]]
    constraints: Callable[[dict], bool] | None = None


def _sma(close: pd.Series, period: int) -> dict[str, np.ndarray]:
    return {"ma": close.rolling(period).mean().to_numpy()}


def _ema(close: pd.Series, period: int) -> dict[str, np.ndarray]:
    return {"ma": close.ewm(span=period, adjust=False).mean().to_numpy()}


def _wma(close: pd.Series, period: int) -> dict[str, np.ndarray]:
    weights = np.arange(1, period + 1, dtype=float)
    out = close.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    return {"ma": out.to_numpy()}


def _rsi(close: pd.Series, period: int) -> dict[str, np.ndarray]:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_up / (avg_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return {"rsi": rsi.to_numpy()}


def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> dict[str, np.ndarray]:
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    macd = fast_ema - slow_ema
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return {"macd": macd.to_numpy(), "signal": sig.to_numpy(), "hist": hist.to_numpy()}


def _bb(close: pd.Series, period: int, std: float, ma_type: str) -> dict[str, np.ndarray]:
    if ma_type == "ema":
        mid = close.ewm(span=period, adjust=False).mean()
    else:
        mid = close.rolling(period).mean()
    sigma = close.rolling(period).std(ddof=0)
    up = mid + std * sigma
    low = mid - std * sigma
    return {"upper": up.to_numpy(), "middle": mid.to_numpy(), "lower": low.to_numpy()}


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> dict[str, np.ndarray]:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return {"atr": atr.to_numpy()}


def _keltner(high: pd.Series, low: pd.Series, close: pd.Series, ema_period: int, atr_period: int, mult: float) -> dict[str, np.ndarray]:
    center = close.ewm(span=ema_period, adjust=False).mean()
    atr = _atr(high, low, close, atr_period)["atr"]
    up = center.to_numpy() + mult * atr
    lowb = center.to_numpy() - mult * atr
    return {"upper": up, "lower": lowb}


def _donchian(high: pd.Series, low: pd.Series, period: int) -> dict[str, np.ndarray]:
    up = high.rolling(period).max()
    lowb = low.rolling(period).min()
    return {"upper": up.to_numpy(), "lower": lowb.to_numpy()}


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> dict[str, np.ndarray]:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = _atr(high, low, close, period)["atr"]
    plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(alpha=1 / period, adjust=False).mean().to_numpy() / (atr + 1e-12)
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(alpha=1 / period, adjust=False).mean().to_numpy() / (atr + 1e-12)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)
    adx = pd.Series(dx, index=close.index).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    return {"adx": adx}


def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int, d: int, smooth: int) -> dict[str, np.ndarray]:
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    raw_k = 100 * (close - ll) / (hh - ll + 1e-12)
    k_line = raw_k.rolling(smooth).mean()
    d_line = k_line.rolling(d).mean()
    return {"k": k_line.to_numpy(), "d": d_line.to_numpy()}


def _psar(high: pd.Series, low: pd.Series, step: float, max_step: float) -> dict[str, np.ndarray]:
    # Lightweight approximation suitable for MVP smoke usage.
    mid = (high + low) / 2.0
    sar = mid.ewm(alpha=min(max(step, 0.001), max_step), adjust=False).mean()
    return {"psar": sar.to_numpy()}


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> dict[str, np.ndarray]:
    tp = (high + low + close) / 3.0
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    cci = (tp - ma) / (0.015 * (md + 1e-12))
    return {"cci": cci.to_numpy()}


def _roc(close: pd.Series, period: int) -> dict[str, np.ndarray]:
    roc = 100 * (close / close.shift(period) - 1.0)
    return {"roc": roc.to_numpy()}


def _obv(close: pd.Series, volume: pd.Series) -> dict[str, np.ndarray]:
    direction = np.sign(close.diff().fillna(0.0))
    obv = (direction * volume).cumsum()
    return {"obv": obv.to_numpy()}


def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> dict[str, np.ndarray]:
    tp = (high + low + close) / 3.0
    cumulative = (tp * volume).cumsum()
    cum_vol = volume.cumsum() + 1e-12
    return {"vwap": (cumulative / cum_vol).to_numpy()}


def _compute(ind: str, data: pd.DataFrame, params: dict) -> dict[str, np.ndarray]:
    c = data["close"]
    h = data["high"]
    l = data["low"]
    v = data["volume"]
    if ind == "sma":
        return _sma(c, params["period"])
    if ind == "ema":
        return _ema(c, params["period"])
    if ind == "wma":
        return _wma(c, params["period"])
    if ind == "rsi":
        return _rsi(c, params["period"])
    if ind == "macd":
        return _macd(c, params["fast"], params["slow"], params["signal"])
    if ind == "bb":
        return _bb(c, params["period"], params["std"], params["ma_type"])
    if ind == "atr":
        return _atr(h, l, c, params["period"])
    if ind == "keltner":
        return _keltner(h, l, c, params["ema_period"], params["atr_period"], params["mult"])
    if ind == "donchian":
        return _donchian(h, l, params["period"])
    if ind == "adx":
        return _adx(h, l, c, params["period"])
    if ind == "stoch":
        return _stoch(h, l, c, params["k"], params["d"], params["smooth"])
    if ind == "psar":
        return _psar(h, l, params["step"], params["max_step"])
    if ind == "cci":
        return _cci(h, l, c, params["period"])
    if ind == "roc":
        return _roc(c, params["period"])
    if ind == "obv":
        return _obv(c, v)
    if ind == "vwap":
        return _vwap(h, l, c, v)
    raise KeyError(ind)


def make_default_registry() -> list[IndicatorDef]:
    defs = [
        IndicatorDef("sma", {"period": [5, 8, 10, 13, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]}, ["ma"], lambda d, **p: _compute("sma", d, p)),
        IndicatorDef("ema", {"period": [5, 8, 10, 13, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]}, ["ma"], lambda d, **p: _compute("ema", d, p)),
        IndicatorDef("wma", {"period": [10, 20, 50, 100, 200]}, ["ma"], lambda d, **p: _compute("wma", d, p)),
        IndicatorDef("rsi", {"period": [7, 9, 10, 12, 14, 16, 21]}, ["rsi"], lambda d, **p: _compute("rsi", d, p)),
        IndicatorDef("macd", {"fast": [8, 10, 12, 15], "slow": [21, 24, 26, 30], "signal": [7, 9, 12]}, ["macd", "signal", "hist"], lambda d, **p: _compute("macd", d, p), constraints=lambda params: params["fast"] < params["slow"]),
        IndicatorDef("bb", {"period": [10, 15, 20, 25, 30, 40, 50], "std": [1.5, 1.75, 2.0, 2.25, 2.5, 3.0], "ma_type": ["sma", "ema"]}, ["upper", "middle", "lower"], lambda d, **p: _compute("bb", d, p)),
        IndicatorDef("atr", {"period": [7, 10, 14, 20]}, ["atr"], lambda d, **p: _compute("atr", d, p)),
        IndicatorDef("keltner", {"ema_period": [10, 15, 20, 30], "atr_period": [10, 14, 20], "mult": [1.5, 2.0, 2.5]}, ["upper", "lower"], lambda d, **p: _compute("keltner", d, p)),
        IndicatorDef("donchian", {"period": [10, 20, 30, 50, 100]}, ["upper", "lower"], lambda d, **p: _compute("donchian", d, p)),
        IndicatorDef("adx", {"period": [10, 14, 20]}, ["adx"], lambda d, **p: _compute("adx", d, p)),
        IndicatorDef("stoch", {"k": [5, 9, 14, 21], "d": [3, 5, 7], "smooth": [3, 5]}, ["k", "d"], lambda data, **p: _compute("stoch", data, p)),
        IndicatorDef("psar", {"step": [0.01, 0.02, 0.05], "max_step": [0.1, 0.2, 0.3]}, ["psar"], lambda d, **p: _compute("psar", d, p)),
        IndicatorDef("cci", {"period": [10, 14, 20]}, ["cci"], lambda d, **p: _compute("cci", d, p)),
        IndicatorDef("roc", {"period": [5, 10, 14, 20]}, ["roc"], lambda d, **p: _compute("roc", d, p)),
        IndicatorDef("obv", {}, ["obv"], lambda d, **p: _compute("obv", d, p)),
        IndicatorDef("vwap", {}, ["vwap"], lambda d, **p: _compute("vwap", d, p)),
    ]
    return defs


DEFAULT_REGISTRY = make_default_registry()


def build_registry_from_config(
    base_registry: list[IndicatorDef],
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    param_ranges: dict[str, dict[str, list[Any]]] | None = None,
) -> list[IndicatorDef]:
    include_set = {str(x).strip() for x in (include or []) if str(x).strip()}
    exclude_set = {str(x).strip() for x in (exclude or []) if str(x).strip()}
    overrides = param_ranges or {}

    out: list[IndicatorDef] = []
    for ind in base_registry:
        if include_set and ind.name not in include_set:
            continue
        if ind.name in exclude_set:
            continue

        if ind.name not in overrides:
            out.append(ind)
            continue

        override_grid = overrides[ind.name] or {}
        new_grid: dict[str, list[Any]] = {}
        for key, values in ind.param_grid.items():
            if key not in override_grid:
                new_grid[key] = values
                continue
            custom_values = list(override_grid[key])
            if not custom_values:
                raise ValueError(f"param_ranges for '{ind.name}.{key}' cannot be empty")
            new_grid[key] = custom_values

        unknown_keys = sorted(set(override_grid.keys()) - set(ind.param_grid.keys()))
        if unknown_keys:
            raise ValueError(f"Unknown parameter override(s) for '{ind.name}': {unknown_keys}")

        out.append(
            IndicatorDef(
                name=ind.name,
                param_grid=new_grid,
                outputs=ind.outputs,
                compute_fn=ind.compute_fn,
                constraints=ind.constraints,
            )
        )

    if not out:
        raise ValueError("Indicator selection is empty after applying include/exclude filters")

    return out


def list_indicator_specs(registry: list[IndicatorDef]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for ind in registry:
        specs.append(
            {
                "name": ind.name,
                "outputs": list(ind.outputs),
                "param_grid": {k: list(v) for k, v in ind.param_grid.items()},
            }
        )
    return specs


def param_product(param_grid: dict[str, list]) -> list[dict]:
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]
