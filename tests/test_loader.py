from __future__ import annotations

import pandas as pd
import pytest

from cobra_py.data.loader import load_ohlcv


def test_load_ohlcv_accepts_capitalized_and_adj_close_columns() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "Open": [1, 2, 3, 4, 5],
            "High": [2, 3, 4, 5, 6],
            "Low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "Adj Close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "Volume": [100, 110, 120, 130, 140],
        }
    )

    out = load_ohlcv(df, min_bars=5)

    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert len(out) == 5


def test_load_ohlcv_accepts_yfinance_style_multiindex_columns() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    cols = pd.MultiIndex.from_tuples(
        [
            ("Open", "SPY"),
            ("High", "SPY"),
            ("Low", "SPY"),
            ("Close", "SPY"),
            ("Volume", "SPY"),
        ]
    )
    df = pd.DataFrame(
        [
            [1, 2, 0.5, 1.5, 100],
            [2, 3, 1.5, 2.5, 110],
            [3, 4, 2.5, 3.5, 120],
            [4, 5, 3.5, 4.5, 130],
            [5, 6, 4.5, 5.5, 140],
        ],
        index=idx,
        columns=cols,
    )

    out = load_ohlcv(df, min_bars=5)

    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert len(out) == 5


def test_load_ohlcv_still_fails_on_missing_ohlcv_columns() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=5, freq="D"),
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
        }
    )

    with pytest.raises(ValueError, match="Missing required OHLCV columns"):
        load_ohlcv(df, min_bars=5)
