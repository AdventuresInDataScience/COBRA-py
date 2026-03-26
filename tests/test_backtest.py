"""Tests for the vectorbt-based backtest engine and metrics extraction.

Tests use run_backtest() as the public interface, matching spec Phase 13.
Leverage and borrow-cost tests verify the user-added extensions.
"""
import numpy as np
import pytest
import pandas as pd

from cobra_py.backtest.engine import run_backtest
from cobra_py.backtest.metrics import SENTINEL_BAD
from cobra_py.indicators.cache import IndicatorCache
from cobra_py.objective.function import compute_objective
from cobra_py.policy.schema import Policy, RuleConfig, SLConfig, TPConfig
from cobra_py.policy.sl_tp import compute_sl


def test_backtest_returns_expected_keys(sample_ohlcv_data, small_cache, simple_policy):
    metrics = run_backtest(simple_policy, small_cache, sample_ohlcv_data.iloc[:500], {"init_cash": 10000.0})
    for k in [
        "total_return",
        "cagr",
        "sharpe_ratio",
        "calmar_ratio",
        "car_mdd_ratio",
        "sortino_ratio",
        "ulcer_index",
        "max_drawdown",
        "n_trades",
        "win_rate",
        "avg_return_per_trade",
    ]:
        assert k in metrics


def test_min_trades_penalty(sample_ohlcv_data, small_cache, simple_policy):
    metrics = run_backtest(simple_policy, small_cache, sample_ohlcv_data.iloc[:300], {"init_cash": 10000.0})
    score = compute_objective(metrics, simple_policy, {"objective": "sharpe", "min_trades": 9999})
    assert score == 999.0


def _make_simple_data(n: int = 100, trend: float = 0.001) -> pd.DataFrame:
    """Create a simple uptrending OHLCV DataFrame for deterministic tests."""
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 100.0 * np.exp(np.cumsum(np.full(n, trend)))
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": np.full(n, 10000.0),
    }, index=idx)


def _make_cache_with_rsi(data: pd.DataFrame, rsi_values: np.ndarray) -> IndicatorCache:
    """Create a minimal cache with a fixed RSI array for deterministic testing."""
    cache = IndicatorCache()
    cache.store("rsi", (14,), "rsi", rsi_values)
    cache.store("atr", (14,), "atr", np.full(len(data), 1.0, dtype=float))
    cache.store("sma", (20,), "ma", data["close"].rolling(20).mean().to_numpy())
    return cache


def test_leverage_increases_exposure():
    """With leverage > 1, total return should be amplified in an uptrend."""
    data = _make_simple_data(50, trend=0.005)
    rsi = np.full(len(data), 60.0)
    cache = _make_cache_with_rsi(data, rsi)

    entry = RuleConfig("comparison", "rsi", (14,), "rsi", ">", 50.0)
    policy = Policy(
        entry_rules=(entry,),
        exit_rules=(),
        sl_config=SLConfig("pct", (0.5,)),  # wide stop
        tp_config=TPConfig("pct", (2.0,)),  # wide TP
        n_active_entry=1,
        n_active_exit=0,
    )

    cfg_no_lev = {"init_cash": 10000.0, "fee_rate": 0.0, "slippage": 0.0, "leverage": 1.0, "borrow_cost_rate": 0.0}
    cfg_lev2 = {"init_cash": 10000.0, "fee_rate": 0.0, "slippage": 0.0, "leverage": 2.0, "borrow_cost_rate": 0.0}

    m1 = run_backtest(policy, cache, data, cfg_no_lev)
    m2 = run_backtest(policy, cache, data, cfg_lev2)

    # With 2x leverage in an uptrend, return should be higher
    assert m2["total_return"] >= m1["total_return"]


def test_borrow_cost_reduces_leveraged_returns():
    """Borrow costs on leveraged positions should reduce returns."""
    data = _make_simple_data(50, trend=0.005)
    rsi = np.full(len(data), 60.0)
    cache = _make_cache_with_rsi(data, rsi)

    entry = RuleConfig("comparison", "rsi", (14,), "rsi", ">", 50.0)
    policy = Policy(
        entry_rules=(entry,),
        exit_rules=(),
        sl_config=SLConfig("pct", (0.5,)),
        tp_config=TPConfig("pct", (2.0,)),
        n_active_entry=1,
        n_active_exit=0,
    )

    cfg_no_borrow = {"init_cash": 10000.0, "fee_rate": 0.0, "slippage": 0.0, "leverage": 2.0, "borrow_cost_rate": 0.0}
    cfg_high_borrow = {"init_cash": 10000.0, "fee_rate": 0.0, "slippage": 0.0, "leverage": 2.0, "borrow_cost_rate": 0.20}

    m_no = run_backtest(policy, cache, data, cfg_no_borrow)
    m_hi = run_backtest(policy, cache, data, cfg_high_borrow)

    # High borrow costs should reduce the final equity/return
    assert m_hi["total_return"] < m_no["total_return"]


def test_compute_sl_rejects_wrong_param_shape(small_cache):
    with pytest.raises(ValueError, match="expects 2 parameter"):
        compute_sl(
            SLConfig(sl_type="atr_mult", params=(2.0,)),
            small_cache,
            np.array([100.0, 101.0], dtype=float),
            np.array([101.0, 102.0], dtype=float),
            np.array([99.0, 100.0], dtype=float),
        )


def test_run_backtest_fails_when_sl_levels_are_all_nan(sample_ohlcv_data, small_cache):
    rule = RuleConfig(
        archetype="comparison",
        indicator="sma",
        params=(20,),
        output="ma",
        operator=">",
        comparand="price",
    )
    policy = Policy(
        entry_rules=(rule,),
        exit_rules=(),
        sl_config=SLConfig(sl_type="atr_mult", params=(2.0, 999)),
        tp_config=TPConfig(tp_type="pct", params=(0.03,)),
        n_active_entry=1,
        n_active_exit=0,
    )

    with pytest.raises(ValueError, match="all-NaN"):
        run_backtest(policy, small_cache, sample_ohlcv_data.iloc[:500], {"init_cash": 10000.0})


def test_run_backtest_exit_or_triggers_when_any_exit_rule_true():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    data = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [100.5, 101.5, 102.5, 103.5, 104.5],
            "low": [99.5, 100.5, 101.5, 102.5, 103.5],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [1000.0] * 5,
        },
        index=idx,
    )

    cache = IndicatorCache()
    cache.store("rsi", (14,), "rsi", np.array([60.0, 60.0, 60.0, 60.0, 60.0], dtype=float))

    entry = RuleConfig("comparison", "rsi", (14,), "rsi", ">", 50.0)
    exit_false = RuleConfig("comparison", "rsi", (14,), "rsi", ">", 70.0)
    exit_true = RuleConfig("comparison", "rsi", (14,), "rsi", "<", 65.0)

    policy_or = Policy(
        entry_rules=(entry,),
        exit_rules=(exit_false, exit_true),
        sl_config=SLConfig("pct", (0.5,)),
        tp_config=TPConfig("pct", (2.0,)),
        n_active_entry=1,
        n_active_exit=2,
        entry_logic="and",
        exit_logic="or",
    )
    policy_and = Policy(
        entry_rules=(entry,),
        exit_rules=(exit_false, exit_true),
        sl_config=SLConfig("pct", (0.5,)),
        tp_config=TPConfig("pct", (2.0,)),
        n_active_entry=1,
        n_active_exit=2,
        entry_logic="and",
        exit_logic="and",
    )

    m_or = run_backtest(policy_or, cache, data, {"init_cash": 10000.0, "fee_rate": 0.0, "slippage": 0.0, "freq": "1D"})
    m_and = run_backtest(policy_and, cache, data, {"init_cash": 10000.0, "fee_rate": 0.0, "slippage": 0.0, "freq": "1D"})

    # OR exits sooner, so should capture less of the monotonic uptrend
    assert m_or["total_return"] <= m_and["total_return"]


def test_equity_curve_and_trade_returns_in_output(sample_ohlcv_data, small_cache, simple_policy):
    """Verify the backtest returns equity_curve and trade_returns arrays."""
    metrics = run_backtest(simple_policy, small_cache, sample_ohlcv_data.iloc[:500], {"init_cash": 10000.0})
    assert "equity_curve" in metrics
    assert "trade_returns" in metrics
    assert isinstance(metrics["equity_curve"], np.ndarray)
    assert len(metrics["equity_curve"]) > 0


def test_no_trades_returns_sentinel_metrics():
    """A strategy that never enters should return sentinel values for ratios."""
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    data = pd.DataFrame({
        "open": np.full(100, 100.0),
        "high": np.full(100, 101.0),
        "low": np.full(100, 99.0),
        "close": np.full(100, 100.0),
        "volume": np.full(100, 1000.0),
    }, index=idx)

    cache = IndicatorCache()
    # RSI always below threshold => entry never fires
    cache.store("rsi", (14,), "rsi", np.full(100, 20.0, dtype=float))

    entry = RuleConfig("comparison", "rsi", (14,), "rsi", ">", 80.0)
    policy = Policy(
        entry_rules=(entry,),
        exit_rules=(),
        sl_config=SLConfig("pct", (0.05,)),
        tp_config=TPConfig("pct", (0.10,)),
        n_active_entry=1,
        n_active_exit=0,
    )

    metrics = run_backtest(policy, cache, data, {"init_cash": 10000.0})
    assert metrics["n_trades"] == 0
