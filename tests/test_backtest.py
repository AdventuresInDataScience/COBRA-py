import numpy as np
import pytest
import pandas as pd

from cobra_py.backtest.engine import run_backtest
from cobra_py.backtest.engine import _simulate_single_position
from cobra_py.backtest.metrics import SENTINEL_BAD, extract_metrics
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


def test_leverage_increases_upside_without_borrow_cost():
    close = np.array([100.0, 105.0, 110.0])
    high = close.copy()
    low = close.copy()
    entries = np.array([True, False, False])
    exits = np.array([False, False, False])
    sl = np.array([np.nan, np.nan, np.nan])
    tp = np.array([np.nan, np.nan, np.nan])

    no_lev = _simulate_single_position(
        close=close,
        high=high,
        low=low,
        entries=entries,
        exits=exits,
        sl_levels=sl,
        tp_levels=tp,
        init_cash=10000.0,
        fee_rate=0.0,
        slippage=0.0,
        leverage=1.0,
        borrow_cost_rate=0.0,
        freq="1D",
    )
    lev2 = _simulate_single_position(
        close=close,
        high=high,
        low=low,
        entries=entries,
        exits=exits,
        sl_levels=sl,
        tp_levels=tp,
        init_cash=10000.0,
        fee_rate=0.0,
        slippage=0.0,
        leverage=2.0,
        borrow_cost_rate=0.0,
        freq="1D",
    )

    assert lev2["equity_curve"][-1] > no_lev["equity_curve"][-1]


def test_borrow_cost_reduces_leveraged_equity():
    close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    high = close.copy()
    low = close.copy()
    entries = np.array([True, False, False, False, False])
    exits = np.array([False, False, False, False, False])
    sl = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    tp = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

    low_cost = _simulate_single_position(
        close=close,
        high=high,
        low=low,
        entries=entries,
        exits=exits,
        sl_levels=sl,
        tp_levels=tp,
        init_cash=10000.0,
        fee_rate=0.0,
        slippage=0.0,
        leverage=2.0,
        borrow_cost_rate=0.0,
        freq="1D",
    )
    high_cost = _simulate_single_position(
        close=close,
        high=high,
        low=low,
        entries=entries,
        exits=exits,
        sl_levels=sl,
        tp_levels=tp,
        init_cash=10000.0,
        fee_rate=0.0,
        slippage=0.0,
        leverage=2.0,
        borrow_cost_rate=0.20,
        freq="1D",
    )

    assert high_cost["equity_curve"][-1] < low_cost["equity_curve"][-1]


def test_in_position_equity_is_net_of_borrowed_principal():
    close = np.array([100.0, 100.0, 100.0])
    high = close.copy()
    low = close.copy()
    entries = np.array([True, False, False])
    exits = np.array([False, False, False])
    sl = np.array([np.nan, np.nan, np.nan])
    tp = np.array([np.nan, np.nan, np.nan])

    out = _simulate_single_position(
        close=close,
        high=high,
        low=low,
        entries=entries,
        exits=exits,
        sl_levels=sl,
        tp_levels=tp,
        init_cash=10000.0,
        fee_rate=0.0,
        slippage=0.0,
        leverage=2.0,
        borrow_cost_rate=0.0,
        freq="1D",
    )

    # Net liquidation value should remain at starting equity for a flat market.
    assert out["equity_curve"][0] == 10000.0


def test_sortino_uses_stable_guard_when_no_downside_deviation():
    results = {
        "equity_curve": np.array([100.0, 101.0, 102.0, 103.0], dtype=float),
        "trade_returns": np.array([], dtype=float),
        "n_trades": 0,
    }

    metrics = extract_metrics(results, freq="1D", risk_free_rate_annual=0.0)
    assert metrics["sortino_ratio"] == SENTINEL_BAD


def test_calmar_uses_trailing_3y_while_car_mdd_uses_full_history():
    # Construct a long history with an early deep drawdown and a smoother recent 3-year period.
    early_crash = np.linspace(100.0, 50.0, 80)
    early_recovery = np.linspace(50.0, 100.0, 80)
    base = np.linspace(100.0, 220.0, 756)
    wiggle = 3.0 * np.sin(np.linspace(0.0, 24.0 * np.pi, 756))
    recent_bull = base + wiggle
    equity = np.concatenate([early_crash, early_recovery, recent_bull]).astype(float)

    results = {
        "equity_curve": equity,
        "trade_returns": np.array([], dtype=float),
        "n_trades": 5,
    }

    metrics = extract_metrics(results, freq="1D", risk_free_rate_annual=0.0)

    # Full-history CAR/MDD still reflects the early 50% drawdown,
    # while trailing-3y Calmar should be materially higher.
    assert metrics["calmar_ratio"] > metrics["car_mdd_ratio"]


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
    # Constant oscillator values create deterministic entry/exit signals.
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

    # OR exits sooner in this setup, so it should capture less of the monotonic uptrend.
    assert m_or["total_return"] < m_and["total_return"]

