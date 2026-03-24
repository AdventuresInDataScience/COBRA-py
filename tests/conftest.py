from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cobra_py.indicators.precompute import precompute_all
from cobra_py.indicators.registry import DEFAULT_REGISTRY
from cobra_py.policy.schema import Policy, RuleConfig, SLConfig, TPConfig


@pytest.fixture()
def sample_ohlcv_data() -> pd.DataFrame:
    np.random.seed(0)
    n = 2000
    dt = 1 / 252
    mu = 0.08
    sigma = 0.2
    shocks = np.random.normal((mu - 0.5 * sigma ** 2) * dt, sigma * np.sqrt(dt), size=n)
    close = 100 * np.exp(np.cumsum(shocks))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.maximum(0.001, np.random.uniform(0.001, 0.02, size=n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = np.random.randint(1000, 10000, size=n).astype(float)
    idx = pd.date_range("2010-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


@pytest.fixture()
def small_cache(sample_ohlcv_data):
    reduced = [r for r in DEFAULT_REGISTRY if r.name in {"sma", "rsi", "bb", "atr"}]
    return precompute_all(sample_ohlcv_data, reduced, n_jobs=1)


@pytest.fixture()
def simple_policy() -> Policy:
    rule = RuleConfig(
        archetype="comparison",
        indicator="sma",
        params=(20,),
        output="ma",
        operator=">",
        comparand="price",
    )
    return Policy(
        entry_rules=(rule,),
        exit_rules=(),
        sl_config=SLConfig(sl_type="atr_mult", params=(2.0, 14)),
        tp_config=TPConfig(tp_type="pct", params=(0.03,)),
        n_active_entry=1,
        n_active_exit=0,
    )

