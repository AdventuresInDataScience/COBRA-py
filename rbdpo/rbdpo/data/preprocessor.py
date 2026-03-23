from __future__ import annotations

from typing import Any

import pandas as pd

from .loader import validate_ohlcv


def preprocess(data: pd.DataFrame, config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or {}
    freq = cfg.get("freq")
    split = cfg.get("train_split", 0.7)
    min_bars = int(cfg.get("min_bars", 500))

    frame = data.copy()
    if freq:
        frame = (
            frame.resample(freq)
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna(subset=["open", "high", "low", "close"])
        )

    frame = validate_ohlcv(frame, min_bars=min_bars)

    if isinstance(split, str):
        split_ts = pd.Timestamp(split)
        train = frame.loc[frame.index < split_ts].copy()
        test = frame.loc[frame.index >= split_ts].copy()
    else:
        split = float(split)
        if not 0.1 <= split <= 0.95:
            raise ValueError("train_split must be in [0.1, 0.95] when provided as float")
        idx = int(len(frame) * split)
        train = frame.iloc[:idx].copy()
        test = frame.iloc[idx:].copy()

    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty set")

    train = validate_ohlcv(train, min_bars=min(10, len(train)))
    test = validate_ohlcv(test, min_bars=min(10, len(test)))
    return train, test
