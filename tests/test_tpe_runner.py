from cobra_py.indicators.registry import DEFAULT_REGISTRY, build_registry_from_config
from cobra_py.search.space import build_config_space
from cobra_py.search.tpe_runner import run_tpe


def test_tpe_runs_on_mixed_indicator_space(sample_ohlcv_data, small_cache):
    reg = build_registry_from_config(DEFAULT_REGISTRY, include=["rsi", "bb", "macd"])
    cs = build_config_space(2, 1, reg, seed=42)

    result = run_tpe(
        cache=small_cache,
        data=sample_ohlcv_data.iloc[:300],
        config_space=cs,
        obj_config={"objective": "sharpe", "n_entry_rules": 2, "n_exit_rules": 1, "min_trades": 0},
        backtest_config={"init_cash": 10000.0, "freq": "1D"},
        budget=10,
        seed=42,
    )

    assert result.n_evaluations == 10
    assert result.full_history