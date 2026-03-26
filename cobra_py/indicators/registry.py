"""Indicator registry using pandas-ta as the compute backend.

Covers 30 indicators across four categories:
  - Trend (10): SMA, EMA, WMA, MACD, PSAR, ADX, Ichimoku, Aroon, SuperTrend, Alligator
  - Momentum (10): RSI, Stoch, CCI, Williams %R, PPO, MFI, ROC, Awesome Oscillator, TSI, Ultimate Oscillator
  - Volatility (6): Bollinger Bands, ATR, StdDev, Keltner, Donchian, Chaikin Volatility
  - Volume (4): OBV, A/D Line, CMF, VWAP

Note: Zig Zag is excluded — it requires future data to confirm pivots,
making it unsuitable for forward-looking signal generation.

Supports both pandas-ta-classic (preferred, actively maintained) and the
original pandas-ta as a fallback. Both expose an identical function API.
Install one of:
    pip install pandas-ta-classic   # recommended
    pip install pandas-ta           # original (less actively maintained)
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable

import numpy as np
import pandas as pd

# Try pandas-ta-classic first (actively maintained community fork),
# then fall back to original pandas-ta. Both expose the same API.
try:
    import pandas_ta_classic as ta
except ImportError:
    try:
        import pandas_ta as ta
    except ImportError as exc:
        raise ImportError(
            "No pandas-ta backend found. Install one of:\n"
            "  pip install pandas-ta-classic   (recommended)\n"
            "  pip install pandas-ta"
        ) from exc


@dataclass
class IndicatorDef:
    name: str
    param_grid: dict[str, list]
    outputs: list[str]
    compute_fn: Callable[..., dict[str, np.ndarray]]
    constraints: Callable[[dict], bool] | None = None


# ==========================================================================
# TREND INDICATORS (10)
# ==========================================================================

def _sma(data: pd.DataFrame, period: int) -> dict[str, np.ndarray]:
    result = ta.sma(data["close"], length=period)
    return {"ma": result.to_numpy(dtype=np.float64)}


def _ema(data: pd.DataFrame, period: int) -> dict[str, np.ndarray]:
    result = ta.ema(data["close"], length=period)
    return {"ma": result.to_numpy(dtype=np.float64)}


def _wma(data: pd.DataFrame, period: int) -> dict[str, np.ndarray]:
    result = ta.wma(data["close"], length=period)
    return {"ma": result.to_numpy(dtype=np.float64)}


def _macd(data: pd.DataFrame, fast: int, slow: int, signal: int) -> dict[str, np.ndarray]:
    result = ta.macd(data["close"], fast=fast, slow=slow, signal=signal)
    cols = list(result.columns)
    macd_col = [c for c in cols if c.startswith("MACD_")][0]
    hist_col = [c for c in cols if c.startswith("MACDh_")][0]
    signal_col = [c for c in cols if c.startswith("MACDs_")][0]
    return {
        "macd": result[macd_col].to_numpy(dtype=np.float64),
        "signal": result[signal_col].to_numpy(dtype=np.float64),
        "hist": result[hist_col].to_numpy(dtype=np.float64),
    }


def _psar(data: pd.DataFrame, step: float, max_step: float) -> dict[str, np.ndarray]:
    result = ta.psar(data["high"], data["low"], af0=step, af=step, max_af=max_step)
    cols = list(result.columns)
    long_col = [c for c in cols if c.startswith("PSARl_")]
    short_col = [c for c in cols if c.startswith("PSARs_")]
    if long_col and short_col:
        long_vals = result[long_col[0]].to_numpy(dtype=np.float64)
        short_vals = result[short_col[0]].to_numpy(dtype=np.float64)
        psar = np.where(np.isfinite(long_vals), long_vals, short_vals)
    elif long_col:
        psar = result[long_col[0]].to_numpy(dtype=np.float64)
    elif short_col:
        psar = result[short_col[0]].to_numpy(dtype=np.float64)
    else:
        psar = np.full(len(data), np.nan, dtype=np.float64)
    return {"psar": psar}


def _adx(data: pd.DataFrame, period: int) -> dict[str, np.ndarray]:
    result = ta.adx(data["high"], data["low"], data["close"], length=period)
    cols = list(result.columns)
    adx_col = [c for c in cols if c.startswith("ADX_")][0]
    return {"adx": result[adx_col].to_numpy(dtype=np.float64)}


def _ichimoku(data: pd.DataFrame, tenkan: int, kijun: int, senkou: int) -> dict[str, np.ndarray]:
    result_tuple = ta.ichimoku(data["high"], data["low"], data["close"],
                               tenkan=tenkan, kijun=kijun, senkou=senkou)
    # ta.ichimoku returns (ichimoku_df, span_df); we use the first
    ich = result_tuple[0] if isinstance(result_tuple, tuple) else result_tuple
    cols = list(ich.columns)
    # Columns: ISA_t (Senkou A), ISB_k (Senkou B), ITS_t (Tenkan), IKS_k (Kijun), ICS_k (Chikou)
    tenkan_col = [c for c in cols if c.startswith("ITS_")][0]
    kijun_col = [c for c in cols if c.startswith("IKS_")][0]
    senkou_a_col = [c for c in cols if c.startswith("ISA_")][0]
    senkou_b_col = [c for c in cols if c.startswith("ISB_")][0]
    out = {
        "tenkan": ich[tenkan_col].to_numpy(dtype=np.float64),
        "kijun": ich[kijun_col].to_numpy(dtype=np.float64),
        "senkou_a": ich[senkou_a_col].to_numpy(dtype=np.float64),
        "senkou_b": ich[senkou_b_col].to_numpy(dtype=np.float64),
    }
    chikou_cols = [c for c in cols if c.startswith("ICS_")]
    if chikou_cols:
        out["chikou"] = ich[chikou_cols[0]].to_numpy(dtype=np.float64)
    return out


def _aroon(data: pd.DataFrame, period: int) -> dict[str, np.ndarray]:
    result = ta.aroon(data["high"], data["low"], length=period)
    cols = list(result.columns)
    up_col = [c for c in cols if "AROONU" in c][0]
    down_col = [c for c in cols if "AROOND" in c][0]
    osc_col = [c for c in cols if "AROONOSC" in c][0]
    return {
        "aroon_up": result[up_col].to_numpy(dtype=np.float64),
        "aroon_down": result[down_col].to_numpy(dtype=np.float64),
        "aroon_osc": result[osc_col].to_numpy(dtype=np.float64),
    }


def _supertrend(data: pd.DataFrame, period: int, mult: float) -> dict[str, np.ndarray]:
    result = ta.supertrend(data["high"], data["low"], data["close"],
                           length=period, multiplier=mult)
    cols = list(result.columns)
    st_col = [c for c in cols if c.startswith("SUPERT_") and "d" not in c.split("_")[-1]][0]
    dir_col = [c for c in cols if c.startswith("SUPERTd_")][0]
    return {
        "supertrend": result[st_col].to_numpy(dtype=np.float64),
        "direction": result[dir_col].to_numpy(dtype=np.float64),
    }


def _alligator(data: pd.DataFrame, jaw_period: int, teeth_period: int, lips_period: int) -> dict[str, np.ndarray]:
    """Bill Williams Alligator: three smoothed moving averages (SMMA).

    Jaw  = SMMA(median_price, jaw_period)   shifted 8 bars forward
    Teeth = SMMA(median_price, teeth_period) shifted 5 bars forward
    Lips  = SMMA(median_price, lips_period)  shifted 3 bars forward

    Uses pandas-ta SMA as an approximation of SMMA for the smoothed component,
    then shifts forward. The shift means the last N values are NaN.
    """
    median_price = (data["high"] + data["low"]) / 2.0
    jaw = ta.sma(median_price, length=jaw_period).shift(8)
    teeth = ta.sma(median_price, length=teeth_period).shift(5)
    lips = ta.sma(median_price, length=lips_period).shift(3)
    return {
        "jaw": jaw.to_numpy(dtype=np.float64),
        "teeth": teeth.to_numpy(dtype=np.float64),
        "lips": lips.to_numpy(dtype=np.float64),
    }


# ==========================================================================
# MOMENTUM INDICATORS / OSCILLATORS (10)
# ==========================================================================

def _rsi(data: pd.DataFrame, period: int) -> dict[str, np.ndarray]:
    result = ta.rsi(data["close"], length=period)
    return {"rsi": result.to_numpy(dtype=np.float64)}


def _stoch(data: pd.DataFrame, k: int, d: int, smooth: int) -> dict[str, np.ndarray]:
    result = ta.stoch(data["high"], data["low"], data["close"],
                      k=k, d=d, smooth_k=smooth)
    cols = list(result.columns)
    k_col = [c for c in cols if c.startswith("STOCHk_")][0]
    d_col = [c for c in cols if c.startswith("STOCHd_")][0]
    return {
        "k": result[k_col].to_numpy(dtype=np.float64),
        "d": result[d_col].to_numpy(dtype=np.float64),
    }


def _cci(data: pd.DataFrame, period: int) -> dict[str, np.ndarray]:
    result = ta.cci(data["high"], data["low"], data["close"], length=period)
    return {"cci": result.to_numpy(dtype=np.float64)}


def _willr(data: pd.DataFrame, period: int) -> dict[str, np.ndarray]:
    result = ta.willr(data["high"], data["low"], data["close"], length=period)
    return {"willr": result.to_numpy(dtype=np.float64)}


def _ppo(data: pd.DataFrame, fast: int, slow: int, signal: int) -> dict[str, np.ndarray]:
    result = ta.ppo(data["close"], fast=fast, slow=slow, signal=signal)
    cols = list(result.columns)
    ppo_col = [c for c in cols if c.startswith("PPO_")][0]
    hist_col = [c for c in cols if c.startswith("PPOh_")][0]
    sig_col = [c for c in cols if c.startswith("PPOs_")][0]
    return {
        "ppo": result[ppo_col].to_numpy(dtype=np.float64),
        "ppo_signal": result[sig_col].to_numpy(dtype=np.float64),
        "ppo_hist": result[hist_col].to_numpy(dtype=np.float64),
    }


def _mfi(data: pd.DataFrame, period: int) -> dict[str, np.ndarray]:
    result = ta.mfi(data["high"], data["low"], data["close"], data["volume"], length=period)
    return {"mfi": result.to_numpy(dtype=np.float64)}


def _roc(data: pd.DataFrame, period: int) -> dict[str, np.ndarray]:
    result = ta.roc(data["close"], length=period)
    return {"roc": result.to_numpy(dtype=np.float64)}


def _ao(data: pd.DataFrame, fast: int, slow: int) -> dict[str, np.ndarray]:
    result = ta.ao(data["high"], data["low"], fast=fast, slow=slow)
    return {"ao": result.to_numpy(dtype=np.float64)}


def _tsi(data: pd.DataFrame, fast: int, slow: int) -> dict[str, np.ndarray]:
    result = ta.tsi(data["close"], fast=fast, slow=slow)
    cols = list(result.columns)
    tsi_col = [c for c in cols if c.startswith("TSI_")][0]
    sig_col = [c for c in cols if c.startswith("TSIs_")]
    out = {"tsi": result[tsi_col].to_numpy(dtype=np.float64)}
    if sig_col:
        out["tsi_signal"] = result[sig_col[0]].to_numpy(dtype=np.float64)
    return out


def _uo(data: pd.DataFrame, fast: int, medium: int, slow: int) -> dict[str, np.ndarray]:
    result = ta.uo(data["high"], data["low"], data["close"],
                   fast=fast, medium=medium, slow=slow)
    return {"uo": result.to_numpy(dtype=np.float64)}


# ==========================================================================
# VOLATILITY INDICATORS (6)
# ==========================================================================

def _bb(data: pd.DataFrame, period: int, std: float, ma_type: str) -> dict[str, np.ndarray]:
    result = ta.bbands(data["close"], length=period, std=std, mamode=ma_type)
    cols = list(result.columns)
    lower_col = [c for c in cols if c.startswith("BBL_")][0]
    mid_col = [c for c in cols if c.startswith("BBM_")][0]
    upper_col = [c for c in cols if c.startswith("BBU_")][0]
    return {
        "upper": result[upper_col].to_numpy(dtype=np.float64),
        "middle": result[mid_col].to_numpy(dtype=np.float64),
        "lower": result[lower_col].to_numpy(dtype=np.float64),
    }


def _atr(data: pd.DataFrame, period: int) -> dict[str, np.ndarray]:
    result = ta.atr(data["high"], data["low"], data["close"], length=period)
    return {"atr": result.to_numpy(dtype=np.float64)}


def _stdev(data: pd.DataFrame, period: int) -> dict[str, np.ndarray]:
    result = ta.stdev(data["close"], length=period)
    return {"stdev": result.to_numpy(dtype=np.float64)}


def _keltner(data: pd.DataFrame, ema_period: int, atr_period: int, mult: float) -> dict[str, np.ndarray]:
    result = ta.kc(data["high"], data["low"], data["close"],
                   length=ema_period, scalar=mult)
    cols = list(result.columns)
    lower_col = [c for c in cols if c.startswith("KCL")][0]
    upper_col = [c for c in cols if c.startswith("KCU")][0]
    return {
        "upper": result[upper_col].to_numpy(dtype=np.float64),
        "lower": result[lower_col].to_numpy(dtype=np.float64),
    }


def _donchian(data: pd.DataFrame, period: int) -> dict[str, np.ndarray]:
    result = ta.donchian(data["high"], data["low"], lower_length=period, upper_length=period)
    cols = list(result.columns)
    lower_col = [c for c in cols if c.startswith("DCL_")][0]
    upper_col = [c for c in cols if c.startswith("DCU_")][0]
    return {
        "upper": result[upper_col].to_numpy(dtype=np.float64),
        "lower": result[lower_col].to_numpy(dtype=np.float64),
    }


def _chaikin_vol(data: pd.DataFrame, period: int) -> dict[str, np.ndarray]:
    """Chaikin Volatility: rate of change of the EMA of the High-Low range.

    CV = 100 * (EMA(H-L, period) - EMA(H-L, period).shift(period)) / EMA(H-L, period).shift(period)
    """
    hl_range = data["high"] - data["low"]
    ema_range = ta.ema(hl_range, length=period)
    shifted = ema_range.shift(period)
    cv = 100.0 * (ema_range - shifted) / (shifted.abs() + 1e-12)
    return {"chaikin_vol": cv.to_numpy(dtype=np.float64)}


# ==========================================================================
# VOLUME & BREADTH INDICATORS (4)
# ==========================================================================

def _obv(data: pd.DataFrame) -> dict[str, np.ndarray]:
    result = ta.obv(data["close"], data["volume"])
    return {"obv": result.to_numpy(dtype=np.float64)}


def _ad(data: pd.DataFrame) -> dict[str, np.ndarray]:
    result = ta.ad(data["high"], data["low"], data["close"], data["volume"])
    return {"ad": result.to_numpy(dtype=np.float64)}


def _cmf(data: pd.DataFrame, period: int) -> dict[str, np.ndarray]:
    result = ta.cmf(data["high"], data["low"], data["close"], data["volume"], length=period)
    return {"cmf": result.to_numpy(dtype=np.float64)}


def _vwap(data: pd.DataFrame) -> dict[str, np.ndarray]:
    result = ta.vwap(data["high"], data["low"], data["close"], data["volume"])
    return {"vwap": result.to_numpy(dtype=np.float64)}


# ==========================================================================
# Registry builder
# ==========================================================================

def make_default_registry() -> list[IndicatorDef]:
    """Build the full default indicator registry (29 indicators)."""
    defs = [
        # ---- TREND (10) ----
        IndicatorDef("sma",
                     {"period": [5, 8, 10, 13, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]},
                     ["ma"],
                     lambda d, **p: _sma(d, p["period"])),
        IndicatorDef("ema",
                     {"period": [5, 8, 10, 13, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]},
                     ["ma"],
                     lambda d, **p: _ema(d, p["period"])),
        IndicatorDef("wma",
                     {"period": [10, 20, 50, 100, 200]},
                     ["ma"],
                     lambda d, **p: _wma(d, p["period"])),
        IndicatorDef("macd",
                     {"fast": [8, 10, 12, 15], "slow": [21, 24, 26, 30], "signal": [7, 9, 12]},
                     ["macd", "signal", "hist"],
                     lambda d, **p: _macd(d, p["fast"], p["slow"], p["signal"]),
                     constraints=lambda params: params["fast"] < params["slow"]),
        IndicatorDef("psar",
                     {"step": [0.01, 0.02, 0.05], "max_step": [0.1, 0.2, 0.3]},
                     ["psar"],
                     lambda d, **p: _psar(d, p["step"], p["max_step"])),
        IndicatorDef("adx",
                     {"period": [10, 14, 20]},
                     ["adx"],
                     lambda d, **p: _adx(d, p["period"])),
        IndicatorDef("ichimoku",
                     {"tenkan": [7, 9, 12], "kijun": [22, 26, 30], "senkou": [44, 52, 60]},
                     ["tenkan", "kijun", "senkou_a", "senkou_b"],
                     lambda d, **p: _ichimoku(d, p["tenkan"], p["kijun"], p["senkou"])),
        IndicatorDef("aroon",
                     {"period": [10, 14, 20, 25]},
                     ["aroon_up", "aroon_down", "aroon_osc"],
                     lambda d, **p: _aroon(d, p["period"])),
        IndicatorDef("supertrend",
                     {"period": [7, 10, 14, 20], "mult": [1.5, 2.0, 2.5, 3.0]},
                     ["supertrend", "direction"],
                     lambda d, **p: _supertrend(d, p["period"], p["mult"])),
        IndicatorDef("alligator",
                     {"jaw_period": [13, 21], "teeth_period": [8, 13], "lips_period": [5, 8]},
                     ["jaw", "teeth", "lips"],
                     lambda d, **p: _alligator(d, p["jaw_period"], p["teeth_period"], p["lips_period"]),
                     constraints=lambda p: p["jaw_period"] > p["teeth_period"] > p["lips_period"]),

        # ---- MOMENTUM / OSCILLATORS (10) ----
        IndicatorDef("rsi",
                     {"period": [7, 9, 10, 12, 14, 16, 21]},
                     ["rsi"],
                     lambda d, **p: _rsi(d, p["period"])),
        IndicatorDef("stoch",
                     {"k": [5, 9, 14, 21], "d": [3, 5, 7], "smooth": [3, 5]},
                     ["k", "d"],
                     lambda d, **p: _stoch(d, p["k"], p["d"], p["smooth"])),
        IndicatorDef("cci",
                     {"period": [10, 14, 20]},
                     ["cci"],
                     lambda d, **p: _cci(d, p["period"])),
        IndicatorDef("willr",
                     {"period": [7, 10, 14, 21]},
                     ["willr"],
                     lambda d, **p: _willr(d, p["period"])),
        IndicatorDef("ppo",
                     {"fast": [8, 10, 12], "slow": [21, 26, 30], "signal": [7, 9, 12]},
                     ["ppo", "ppo_signal", "ppo_hist"],
                     lambda d, **p: _ppo(d, p["fast"], p["slow"], p["signal"]),
                     constraints=lambda params: params["fast"] < params["slow"]),
        IndicatorDef("mfi",
                     {"period": [7, 10, 14, 21]},
                     ["mfi"],
                     lambda d, **p: _mfi(d, p["period"])),
        IndicatorDef("roc",
                     {"period": [5, 10, 14, 20]},
                     ["roc"],
                     lambda d, **p: _roc(d, p["period"])),
        IndicatorDef("ao",
                     {"fast": [5, 7], "slow": [21, 34]},
                     ["ao"],
                     lambda d, **p: _ao(d, p["fast"], p["slow"])),
        IndicatorDef("tsi",
                     {"fast": [10, 13], "slow": [21, 25]},
                     ["tsi", "tsi_signal"],
                     lambda d, **p: _tsi(d, p["fast"], p["slow"])),
        IndicatorDef("uo",
                     {"fast": [5, 7], "medium": [10, 14], "slow": [21, 28]},
                     ["uo"],
                     lambda d, **p: _uo(d, p["fast"], p["medium"], p["slow"])),

        # ---- VOLATILITY (6) ----
        IndicatorDef("bb",
                     {"period": [10, 15, 20, 25, 30, 40, 50],
                      "std": [1.5, 1.75, 2.0, 2.25, 2.5, 3.0],
                      "ma_type": ["sma", "ema"]},
                     ["upper", "middle", "lower"],
                     lambda d, **p: _bb(d, p["period"], p["std"], p["ma_type"])),
        IndicatorDef("atr",
                     {"period": [7, 10, 14, 20]},
                     ["atr"],
                     lambda d, **p: _atr(d, p["period"])),
        IndicatorDef("stdev",
                     {"period": [10, 14, 20, 30]},
                     ["stdev"],
                     lambda d, **p: _stdev(d, p["period"])),
        IndicatorDef("keltner",
                     {"ema_period": [10, 15, 20, 30], "atr_period": [10, 14, 20], "mult": [1.5, 2.0, 2.5]},
                     ["upper", "lower"],
                     lambda d, **p: _keltner(d, p["ema_period"], p["atr_period"], p["mult"])),
        IndicatorDef("donchian",
                     {"period": [10, 20, 30, 50, 100]},
                     ["upper", "lower"],
                     lambda d, **p: _donchian(d, p["period"])),
        IndicatorDef("chaikin_vol",
                     {"period": [10, 14, 20]},
                     ["chaikin_vol"],
                     lambda d, **p: _chaikin_vol(d, p["period"])),

        # ---- VOLUME & BREADTH (4) ----
        IndicatorDef("obv", {}, ["obv"],
                     lambda d, **p: _obv(d)),
        IndicatorDef("ad", {}, ["ad"],
                     lambda d, **p: _ad(d)),
        IndicatorDef("cmf",
                     {"period": [10, 14, 20, 30]},
                     ["cmf"],
                     lambda d, **p: _cmf(d, p["period"])),
        IndicatorDef("vwap", {}, ["vwap"],
                     lambda d, **p: _vwap(d)),
    ]
    return defs


DEFAULT_REGISTRY = make_default_registry()


# ==========================================================================
# Config overrides and filtering utilities
# ==========================================================================

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
