import numpy as np

from cobra_py.backtest.engine import run_backtest
from cobra_py.backtest.engine import _simulate_single_position
from cobra_py.backtest.metrics import SENTINEL_BAD, extract_metrics
from cobra_py.objective.function import compute_objective


def test_backtest_returns_expected_keys(sample_ohlcv_data, small_cache, simple_policy):
    metrics = run_backtest(simple_policy, small_cache, sample_ohlcv_data.iloc[:500], {"init_cash": 10000.0})
    for k in ["total_return", "sharpe_ratio", "calmar_ratio", "sortino_ratio", "ulcer_index", "max_drawdown", "n_trades", "win_rate", "avg_return_per_trade"]:
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

