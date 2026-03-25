import pytest

from cobra_py.indicators.registry import DEFAULT_REGISTRY, build_registry_from_config
from cobra_py.search.dehb_runner import run_dehb
from cobra_py.search.space import build_config_space


def test_dehb_backend_validation_raises():
    with pytest.raises(ValueError, match="dehb_backend"):
        run_dehb(
            cache=None,
            data=None,
            config_space=None,
            obj_config={},
            backtest_config={},
            budget=1,
            seed=1,
            dehb_backend="invalid_backend",
        )


def test_dehb_auto_backend_returns_result(sample_ohlcv_data, small_cache):
    reg = build_registry_from_config(DEFAULT_REGISTRY, include=["rsi"])
    cs = build_config_space(1, 1, reg, seed=1)

    result = run_dehb(
        cache=small_cache,
        data=sample_ohlcv_data.iloc[:300],
        config_space=cs,
        obj_config={"objective": "sharpe", "n_entry_rules": 1, "n_exit_rules": 1, "min_trades": 0},
        backtest_config={"init_cash": 10000.0, "freq": "1D"},
        budget=8,
        seed=1,
        dehb_backend="auto",
    )

    assert result.n_evaluations >= 1
    assert result.optimiser_name in {"dehb(native)", "dehb(seed-de-mvp)"}
    assert result.full_history
    assert "entry_logic" in result.full_history[0]["config"]
