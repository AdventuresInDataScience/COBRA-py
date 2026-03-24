from cobra_py.backtest.engine import run_backtest
from cobra_py.objective.function import compute_objective


def test_backtest_returns_expected_keys(sample_ohlcv_data, small_cache, simple_policy):
    metrics = run_backtest(simple_policy, small_cache, sample_ohlcv_data.iloc[:500], {"init_cash": 10000.0})
    for k in ["total_return", "sharpe_ratio", "calmar_ratio", "sortino_ratio", "ulcer_index", "max_drawdown", "n_trades", "win_rate", "avg_return_per_trade"]:
        assert k in metrics


def test_min_trades_penalty(sample_ohlcv_data, small_cache, simple_policy):
    metrics = run_backtest(simple_policy, small_cache, sample_ohlcv_data.iloc[:300], {"init_cash": 10000.0})
    score = compute_objective(metrics, simple_policy, {"objective": "sharpe", "min_trades": 9999})
    assert score == 999.0

