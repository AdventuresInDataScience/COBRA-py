from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def _normalize_column_key(column: Any) -> str:
    if isinstance(column, tuple):
        parts = [str(p).strip().lower() for p in column if str(p).strip()]
        raw = "_".join(parts)
    else:
        raw = str(column).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", raw).strip("_")


def _canonicalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [_normalize_column_key(c) for c in data.columns]

    alias_map = {
        "open": {"open", "o"},
        "high": {"high", "h"},
        "low": {"low", "l"},
        "close": {"close", "c", "adj_close", "adjusted_close", "close_adj"},
        "volume": {"volume", "vol", "v"},
    }

    # Prefer exact alias matches, then prefixed forms like open_spy from yfinance MultiIndex columns.
    normalized = [(idx, col, _normalize_column_key(col)) for idx, col in enumerate(data.columns)]
    rename_map: dict[Any, str] = {}
    used_targets: set[str] = set()

    for idx, col, key in normalized:
        best_target = None
        best_score = 99
        for target, aliases in alias_map.items():
            if target in used_targets:
                continue
            if key in aliases:
                score = 0
            elif any(key.startswith(f"{alias}_") for alias in aliases):
                score = 1
            else:
                continue
            if score < best_score:
                best_score = score
                best_target = target
        if best_target is not None:
            rename_map[col] = best_target
            used_targets.add(best_target)

    if rename_map:
        return data.rename(columns=rename_map)
    return data


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        lower_map = {str(c).strip().lower(): c for c in df.columns}
        if "datetime" in lower_map:
            df = df.set_index(lower_map["datetime"])
        elif "date" in lower_map:
            df = df.set_index(lower_map["date"])
        else:
            for candidate in df.columns:
                name = str(candidate).strip().lower()
                if is_numeric_dtype(df[candidate]) and not ("date" in name or "time" in name or name.startswith("unnamed")):
                    continue
                parsed = pd.to_datetime(df[candidate], errors="coerce", utc=False)
                if parsed.notna().mean() >= 0.95:
                    df = df.set_index(candidate)
                    break
            else:
                raise ValueError("Data must have a DatetimeIndex or a 'datetime'/'date' column")
    df.index = pd.to_datetime(df.index, utc=False)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


def validate_ohlcv(df: pd.DataFrame, min_bars: int = 500) -> pd.DataFrame:
    data = _canonicalize_ohlcv_columns(df.copy())
    data.columns = [str(c).strip().lower() for c in data.columns]
    data = _ensure_datetime_index(data)

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    data = data[REQUIRED_COLUMNS]
    data = data.sort_index()

    if not data.index.is_monotonic_increasing:
        raise ValueError("Index must be monotonically increasing")

    if data[["close", "high", "low"]].isna().any().any():
        raise ValueError("Columns close/high/low cannot contain NaN")

    data["volume"] = data["volume"].fillna(0.0)

    if (data["high"] < data["low"]).any():
        raise ValueError("Invalid bars: high < low")
    if (data["high"] < data["close"]).any():
        raise ValueError("Invalid bars: high < close")
    if (data["low"] > data["close"]).any():
        raise ValueError("Invalid bars: low > close")

    if len(data) < min_bars:
        raise ValueError(f"Insufficient rows: got {len(data)}, require at least {min_bars}")

    return data


def load_ohlcv(source: str | Path | pd.DataFrame, freq: str | None = None, min_bars: int = 500) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        data = source.copy()
    else:
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            data = pd.read_csv(path)
        elif suffix in {".parquet", ".pq"}:
            data = pd.read_parquet(path)
        else:
            raise ValueError("source must be a DataFrame or a .csv/.parquet file")

    data = validate_ohlcv(data, min_bars=min_bars)

    if freq:
        data = (
            data.resample(freq)
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna(subset=["open", "high", "low", "close"])
        )
        data = validate_ohlcv(data, min_bars=min_bars)

    return data
