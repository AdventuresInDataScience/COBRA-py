import numpy as np


def test_sma_matches_manual(sample_ohlcv_data, small_cache):
    close = sample_ohlcv_data["close"]
    manual = close.rolling(20).mean().to_numpy()
    cached = small_cache.get("sma", (20,), "ma")
    assert cached is not None
    mask = np.isfinite(manual) & np.isfinite(cached)
    assert np.allclose(cached[mask], manual[mask], atol=1e-5)


def test_cache_memory_usage_nonzero(small_cache):
    assert small_cache.memory_usage_gb() > 0


def test_missing_param_returns_none(small_cache):
    assert small_cache.get("sma", (999,), "ma") is None
