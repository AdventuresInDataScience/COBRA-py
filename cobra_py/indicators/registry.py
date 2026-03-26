from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable

import numpy as np
import pandas as pd
import pandas_ta_classic as ta


@dataclass
class IndicatorDef:
    name: str
    param_grid: dict[str, list]
    outputs: list[str]
    compute_fn: Callable[..., dict[str, np.ndarray]]
    constraints: Callable[[dict], bool] | None = None


def _sma(close: pd.Series, period: int) -> dict[str, np.ndarray]:
    out = ta.sma(close=close, length=period)
    return {"ma": np.asarray(out, dtype=float)}


def _ema(close: pd.Series, period: int) -> dict[str, np.ndarray]:
    out = ta.ema(close=close, length=period)
    return {"ma": np.asarray(out, dtype=float)}


def _wma(close: pd.Series, period: int) -> dict[str, np.ndarray]:
    out = ta.wma(close=close, length=period)
    return {"ma": np.asarray(out, dtype=float)}


def _rsi(close: pd.Series, period: int) -> dict[str, np.ndarray]:
    out = ta.rsi(close=close, length=period)
    return {"rsi": np.asarray(out, dtype=float)}


def _col(df: pd.DataFrame, prefix: str) -> pd.Series:
    for c in df.columns:
        if str(c).startswith(prefix):
            return df[c]
    raise KeyError(f"Missing expected pandas-ta column prefix '{prefix}' in {list(df.columns)}")


def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> dict[str, np.ndarray]:
    out = ta.macd(close=close, fast=fast, slow=slow, signal=signal)
    if out is None:
        nan = np.full(len(close), np.nan, dtype=float)
        return {"macd": nan, "signal": nan, "hist": nan}
    return {
        "macd": np.asarray(_col(out, "MACD_"), dtype=float),
        "signal": np.asarray(_col(out, "MACDs_"), dtype=float),
        "hist": np.asarray(_col(out, "MACDh_"), dtype=float),
    }


def _bb(close: pd.Series, period: int, std: float, ma_type: str) -> dict[str, np.ndarray]:
    mamode = "ema" if str(ma_type).lower() == "ema" else "sma"
    out = ta.bbands(close=close, length=period, std=std, mamode=mamode)
    if out is None:
        nan = np.full(len(close), np.nan, dtype=float)
        return {"upper": nan, "middle": nan, "lower": nan}
    return {
        "upper": np.asarray(_col(out, "BBU_"), dtype=float),
        "middle": np.asarray(_col(out, "BBM_"), dtype=float),
        "lower": np.asarray(_col(out, "BBL_"), dtype=float),
    }


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> dict[str, np.ndarray]:
    out = ta.atr(high=high, low=low, close=close, length=period)
    return {"atr": np.asarray(out, dtype=float)}


def _keltner(high: pd.Series, low: pd.Series, close: pd.Series, ema_period: int, atr_period: int, mult: float) -> dict[str, np.ndarray]:
    center = ta.ema(close=close, length=ema_period)
    atr = ta.atr(high=high, low=low, close=close, length=atr_period)
    up = np.asarray(center, dtype=float) + float(mult) * np.asarray(atr, dtype=float)
    lowb = np.asarray(center, dtype=float) - float(mult) * np.asarray(atr, dtype=float)
    return {"upper": up, "lower": lowb}


def _donchian(high: pd.Series, low: pd.Series, period: int) -> dict[str, np.ndarray]:
    out = ta.donchian(high=high, low=low, lower_length=period, upper_length=period)
    if out is None:
        nan = np.full(len(high), np.nan, dtype=float)
        return {"upper": nan, "lower": nan}
    return {
        "upper": np.asarray(_col(out, "DCU_"), dtype=float),
        "lower": np.asarray(_col(out, "DCL_"), dtype=float),
    }


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> dict[str, np.ndarray]:
    out = ta.adx(high=high, low=low, close=close, length=period)
    if out is None:
        return {"adx": np.full(len(close), np.nan, dtype=float)}
    return {"adx": np.asarray(_col(out, "ADX_"), dtype=float)}


def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int, d: int, smooth: int) -> dict[str, np.ndarray]:
    out = ta.stoch(high=high, low=low, close=close, k=k, d=d, smooth_k=smooth)
    if out is None:
        nan = np.full(len(close), np.nan, dtype=float)
        return {"k": nan, "d": nan}
    return {
        "k": np.asarray(_col(out, "STOCHk_"), dtype=float),
        "d": np.asarray(_col(out, "STOCHd_"), dtype=float),
    }


def _psar(high: pd.Series, low: pd.Series, step: float, max_step: float) -> dict[str, np.ndarray]:
    out = ta.psar(high=high, low=low, af=step, max_af=max_step)
    if out is None:
        return {"psar": np.full(len(high), np.nan, dtype=float)}
    long_sar = _col(out, "PSARl_")
    short_sar = _col(out, "PSARs_")
    sar = long_sar.fillna(short_sar)
    return {"psar": np.asarray(sar, dtype=float)}


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> dict[str, np.ndarray]:
    out = ta.cci(high=high, low=low, close=close, length=period)
    return {"cci": np.asarray(out, dtype=float)}


def _roc(close: pd.Series, period: int) -> dict[str, np.ndarray]:
    out = ta.roc(close=close, length=period)
    return {"roc": np.asarray(out, dtype=float)}


def _obv(close: pd.Series, volume: pd.Series) -> dict[str, np.ndarray]:
    out = ta.obv(close=close, volume=volume)
    return {"obv": np.asarray(out, dtype=float)}


def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> dict[str, np.ndarray]:
    out = ta.vwap(high=high, low=low, close=close, volume=volume)
    return {"vwap": np.asarray(out, dtype=float)}


def _ichimoku(high: pd.Series, low: pd.Series, close: pd.Series, tenkan: int, kijun: int, senkou: int) -> dict[str, np.ndarray]:
    out = ta.ichimoku(high=high, low=low, close=close, tenkan=tenkan, kijun=kijun, senkou=senkou)
    if out is None:
        nan = np.full(len(close), np.nan, dtype=float)
        return {
            "tenkan": nan,
            "kijun": nan,
            "senkou_a": nan,
            "senkou_b": nan,
            "chikou": nan,
        }
    lead_df = out[0] if isinstance(out, tuple) else out
    return {
        "tenkan": np.asarray(_col(lead_df, "ITS_"), dtype=float),
        "kijun": np.asarray(_col(lead_df, "IKS_"), dtype=float),
        "senkou_a": np.asarray(_col(lead_df, "ISA_"), dtype=float),
        "senkou_b": np.asarray(_col(lead_df, "ISB_"), dtype=float),
        "chikou": np.asarray(_col(lead_df, "ICS_"), dtype=float),
    }


def _aroon(high: pd.Series, low: pd.Series, period: int) -> dict[str, np.ndarray]:
    out = ta.aroon(high=high, low=low, length=period)
    if out is None:
        nan = np.full(len(high), np.nan, dtype=float)
        return {"up": nan, "down": nan, "osc": nan}
    return {
        "up": np.asarray(_col(out, "AROONU_"), dtype=float),
        "down": np.asarray(_col(out, "AROOND_"), dtype=float),
        "osc": np.asarray(_col(out, "AROONOSC_"), dtype=float),
    }


def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int, mult: float) -> dict[str, np.ndarray]:
    out = ta.supertrend(high=high, low=low, close=close, length=period, multiplier=mult)
    if out is None:
        nan = np.full(len(close), np.nan, dtype=float)
        return {"supertrend": nan, "direction": nan}
    return {
        "supertrend": np.asarray(_col(out, "SUPERT_"), dtype=float),
        "direction": np.asarray(_col(out, "SUPERTd_"), dtype=float),
    }


def _willr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> dict[str, np.ndarray]:
    out = ta.willr(high=high, low=low, close=close, length=period)
    return {"willr": np.asarray(out, dtype=float)}


def _ppo(close: pd.Series, fast: int, slow: int, signal: int) -> dict[str, np.ndarray]:
    out = ta.ppo(close=close, fast=fast, slow=slow, signal=signal)
    if out is None:
        nan = np.full(len(close), np.nan, dtype=float)
        return {"ppo": nan, "signal": nan, "hist": nan}
    return {
        "ppo": np.asarray(_col(out, "PPO_"), dtype=float),
        "signal": np.asarray(_col(out, "PPOs_"), dtype=float),
        "hist": np.asarray(_col(out, "PPOh_"), dtype=float),
    }


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int) -> dict[str, np.ndarray]:
    out = ta.mfi(high=high, low=low, close=close, volume=volume, length=period)
    return {"mfi": np.asarray(out, dtype=float)}


def _ao(high: pd.Series, low: pd.Series, fast: int, slow: int) -> dict[str, np.ndarray]:
    out = ta.ao(high=high, low=low, fast=fast, slow=slow)
    return {"ao": np.asarray(out, dtype=float)}


def _tsi(close: pd.Series, fast: int, slow: int, signal: int) -> dict[str, np.ndarray]:
    out = ta.tsi(close=close, fast=fast, slow=slow, signal=signal)
    if out is None:
        nan = np.full(len(close), np.nan, dtype=float)
        return {"tsi": nan, "signal": nan}
    return {
        "tsi": np.asarray(_col(out, "TSI_"), dtype=float),
        "signal": np.asarray(_col(out, "TSIs_"), dtype=float),
    }


def _uo(high: pd.Series, low: pd.Series, close: pd.Series, fast: int, medium: int, slow: int) -> dict[str, np.ndarray]:
    out = ta.uo(high=high, low=low, close=close, fast=fast, medium=medium, slow=slow)
    return {"uo": np.asarray(out, dtype=float)}


def _stdev(close: pd.Series, period: int) -> dict[str, np.ndarray]:
    out = ta.stdev(close=close, length=period)
    return {"stdev": np.asarray(out, dtype=float)}


def _ad(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> dict[str, np.ndarray]:
    out = ta.ad(high=high, low=low, close=close, volume=volume)
    return {"ad": np.asarray(out, dtype=float)}


def _cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int) -> dict[str, np.ndarray]:
    out = ta.cmf(high=high, low=low, close=close, volume=volume, length=period)
    return {"cmf": np.asarray(out, dtype=float)}


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
    if ind == "ichimoku":
        return _ichimoku(h, l, c, params["tenkan"], params["kijun"], params["senkou"])
    if ind == "aroon":
        return _aroon(h, l, params["period"])
    if ind == "supertrend":
        return _supertrend(h, l, c, params["period"], params["mult"])
    if ind == "willr":
        return _willr(h, l, c, params["period"])
    if ind == "ppo":
        return _ppo(c, params["fast"], params["slow"], params["signal"])
    if ind == "mfi":
        return _mfi(h, l, c, v, params["period"])
    if ind == "ao":
        return _ao(h, l, params["fast"], params["slow"])
    if ind == "tsi":
        return _tsi(c, params["fast"], params["slow"], params["signal"])
    if ind == "uo":
        return _uo(h, l, c, params["fast"], params["medium"], params["slow"])
    if ind == "stdev":
        return _stdev(c, params["period"])
    if ind == "ad":
        return _ad(h, l, c, v)
    if ind == "cmf":
        return _cmf(h, l, c, v, params["period"])
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
        IndicatorDef("ichimoku", {"tenkan": [9], "kijun": [26], "senkou": [52]}, ["tenkan", "kijun", "senkou_a", "senkou_b", "chikou"], lambda d, **p: _compute("ichimoku", d, p)),
        IndicatorDef("aroon", {"period": [14, 25]}, ["up", "down", "osc"], lambda d, **p: _compute("aroon", d, p)),
        IndicatorDef("supertrend", {"period": [7, 10, 14], "mult": [2.0, 3.0, 4.0]}, ["supertrend", "direction"], lambda d, **p: _compute("supertrend", d, p)),
        IndicatorDef("willr", {"period": [10, 14, 20]}, ["willr"], lambda d, **p: _compute("willr", d, p)),
        IndicatorDef("ppo", {"fast": [8, 10, 12], "slow": [21, 26, 30], "signal": [7, 9, 12]}, ["ppo", "signal", "hist"], lambda d, **p: _compute("ppo", d, p), constraints=lambda params: params["fast"] < params["slow"]),
        IndicatorDef("mfi", {"period": [10, 14, 20]}, ["mfi"], lambda d, **p: _compute("mfi", d, p)),
        IndicatorDef("ao", {"fast": [5], "slow": [34]}, ["ao"], lambda d, **p: _compute("ao", d, p)),
        IndicatorDef("tsi", {"fast": [13], "slow": [25], "signal": [13]}, ["tsi", "signal"], lambda d, **p: _compute("tsi", d, p)),
        IndicatorDef("uo", {"fast": [7], "medium": [14], "slow": [28]}, ["uo"], lambda d, **p: _compute("uo", d, p)),
        IndicatorDef("stdev", {"period": [10, 20, 30]}, ["stdev"], lambda d, **p: _compute("stdev", d, p)),
        IndicatorDef("ad", {}, ["ad"], lambda d, **p: _compute("ad", d, p)),
        IndicatorDef("cmf", {"period": [10, 20]}, ["cmf"], lambda d, **p: _compute("cmf", d, p)),
    ]
    return defs


DEFAULT_REGISTRY = make_default_registry()


def _expand_range_spec(value: Any) -> list[Any]:
    if isinstance(value, dict):
        if "values" in value:
            return list(value["values"])
        if "range" in value:
            seq = list(value["range"])
            if len(seq) != 3:
                raise ValueError("Range spec must be [start, stop, step]")
            start, stop, step = seq
        elif {"start", "stop", "step"}.issubset(value.keys()):
            start = value["start"]
            stop = value["stop"]
            step = value["step"]
        else:
            raise ValueError("Range dict must contain either 'values', 'range', or start/stop/step")

        start_f = float(start)
        stop_f = float(stop)
        step_f = float(step)
        if step_f <= 0:
            raise ValueError("Range step must be > 0")
        out: list[Any] = []
        cur = start_f
        while cur <= stop_f + 1e-12:
            if float(start).is_integer() and float(stop).is_integer() and float(step).is_integer():
                out.append(int(round(cur)))
            else:
                out.append(round(cur, 10))
            cur += step_f
        return out

    if isinstance(value, list):
        return list(value)

    raise ValueError("Parameter range override must be a list or range dict")


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
            custom_values = _expand_range_spec(override_grid[key])
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
