from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


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
    data = df.copy()
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
