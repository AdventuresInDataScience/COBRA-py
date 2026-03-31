from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from cobra_py.backtest.engine import run_backtest
from cobra_py.helpers import find_strategy
from cobra_py.helpers import _run_with_optimiser
from cobra_py.helpers import _extract_ohlcv_from_yfinance
from cobra_py.helpers import list_available_objectives, list_available_optimisers, summarise_reports
from cobra_py.helpers import plot_equity_curves


def test_run_with_optimiser_forwards_tpe_options(monkeypatch):
    captured = {}

    def _fake_run_tpe(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr("cobra_py.helpers.run_tpe", _fake_run_tpe)

    out = _run_with_optimiser(
        "tpe",
        cache=None,
        data=None,
        config_space=None,
        obj_config={},
        backtest_config={},
        budget=10,
        seed=1,
        tpe_multivariate=False,
        tpe_group=False,
        tpe_n_startup_trials=7,
        tpe_constant_liar=True,
    )

    assert out == "ok"
    assert captured["multivariate"] is False
    assert captured["group"] is False
    assert captured["n_startup_trials"] == 7
    assert captured["constant_liar"] is True


def test_run_with_optimiser_forwards_dehb_options(monkeypatch):
    captured = {}

    def _fake_run_dehb(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr("cobra_py.helpers.run_dehb", _fake_run_dehb)

    out = _run_with_optimiser(
        "dehb",
        cache=None,
        data=None,
        config_space=None,
        obj_config={},
        backtest_config={},
        budget=10,
        seed=1,
        dehb_mutation_factor=0.55,
        dehb_crossover_rate=0.65,
        dehb_population_size=17,
    )

    assert out == "ok"
    assert captured["mutation_factor"] == 0.55
    assert captured["crossover_rate"] == 0.65
    assert captured["population_size"] == 17


def test_run_with_optimiser_does_not_leak_foreign_kwargs_to_dehb(monkeypatch):
    captured = {}

    def _fake_run_dehb(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr("cobra_py.helpers.run_dehb", _fake_run_dehb)

    out = _run_with_optimiser(
        "dehb",
        cache=None,
        data=None,
        config_space=None,
        obj_config={},
        backtest_config={},
        budget=10,
        seed=1,
        nevergrad_algorithm="NGOpt",
        nevergrad_num_workers=4,
        tpe_multivariate=False,
    )

    assert out == "ok"
    assert "nevergrad_algorithm" not in captured
    assert "nevergrad_num_workers" not in captured
    assert "tpe_multivariate" not in captured


def test_run_with_optimiser_forwards_nevergrad_workers(monkeypatch):
    captured = {}

    def _fake_run_nevergrad(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr("cobra_py.helpers.run_nevergrad", _fake_run_nevergrad)

    out = _run_with_optimiser(
        "nevergrad",
        cache=None,
        data=None,
        config_space=None,
        obj_config={},
        backtest_config={},
        budget=10,
        seed=1,
        nevergrad_algorithm="NGOpt",
        nevergrad_num_workers=3,
    )

    assert out == "ok"
    assert captured["optimiser_name"] == "NGOpt"
    assert captured["num_workers"] == 3


def test_list_available_optimisers_contains_expected_options():
    opts = list_available_optimisers()
    assert "dehb" in opts
    assert "nevergrad" in opts
    assert "tpe" in opts


def test_list_available_objectives_contains_ulcer_and_sortino():
    objs = list_available_objectives()
    assert "sortino" in objs
    assert "ulcer" in objs
    assert "max_return_dd_cap" in objs
    assert "cagr" in objs
    assert "car_mdd" in objs


def test_summarise_reports_includes_human_and_ulcer_fields():
    report = {
        "summary": {
            "optimiser_name": "dehb(random-search-mvp)",
            "objective": "ulcer",
            "best_metric_name": "Ulcer index",
            "best_metric_value": 0.05,
            "best_score": 0.07,
            "n_evaluations": 20,
        },
        "best_metrics": {
            "total_return": 0.2,
            "cagr": 0.12,
            "sharpe_ratio": 1.2,
            "sortino_ratio": 1.4,
            "calmar_ratio": 0.8,
            "car_mdd_ratio": 0.8,
            "ulcer_index": 0.05,
            "max_drawdown": -0.1,
        },
    }

    df = summarise_reports({"run_a": report})
    row = df.iloc[0]

    assert row["run"] == "run_a"
    assert row["best_metric_name"] == "Ulcer index"
    assert row["ulcer_index"] == 0.05
    assert row["cagr"] == 0.12
    assert row["car_mdd_ratio"] == 0.8


def test_extract_ohlcv_from_yfinance_prefers_close_over_adj_close() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    raw = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0],
            "High": [12.0, 13.0, 14.0],
            "Low": [9.0, 10.0, 11.0],
            "Adj Close": [8.0, 8.5, 9.0],
            "Close": [11.0, 12.0, 13.0],
            "Volume": [1000, 1100, 1200],
        },
        index=idx,
    )

    out = _extract_ohlcv_from_yfinance(raw)

    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out["close"].tolist() == [11.0, 12.0, 13.0]


def test_extract_ohlcv_from_yfinance_drops_nan_core_ohlc_rows() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    raw = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0],
            "High": [12.0, float("nan"), 14.0],
            "Low": [9.0, 10.0, 11.0],
            "Close": [11.0, float("nan"), 13.0],
            "Volume": [1000, 1100, 1200],
        },
        index=idx,
    )

    out = _extract_ohlcv_from_yfinance(raw)

    assert len(out) == 2
    assert out.index.min() == idx[0]
    assert out.index.max() == idx[2]


def test_plot_equity_curves_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        plot_equity_curves({}, backend="unknown")


def test_run_backtest_exposes_trade_returns(sample_ohlcv_data, small_cache, simple_policy):
    metrics = run_backtest(simple_policy, small_cache, sample_ohlcv_data.iloc[:300], {"init_cash": 10000.0})
    assert "trade_returns" in metrics


def test_find_strategy_returns_structured_object(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=3, freq="D")

    @dataclass
    class _StubOptimiserResult:
        best_policy: object

    def _fake_run_optimiser(**kwargs):
        return {
            "config": {"objective": {"name": "sharpe"}},
            "train": pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=idx),
            "test": pd.DataFrame(),
            "result": _StubOptimiserResult(best_policy={"ok": True}),
            "report": {
                "summary": {"optimiser_name": "dehb", "objective": "sharpe"},
                "best_metrics": {
                    "equity_curve": [10000.0, 10100.0, 10200.0],
                    "trade_returns": [0.01, -0.005],
                    "sharpe_ratio": 1.1,
                },
                "policy_human_readable": "Entry conditions (...)",
            },
            "oos_metrics": None,
            "walk_forward": None,
        }

    monkeypatch.setattr("cobra_py.helpers.run_optimiser", _fake_run_optimiser)
    monkeypatch.setattr("cobra_py.helpers.plot_equity_curves", lambda *args, **kwargs: "chart")

    result = find_strategy(pd.DataFrame({"close": [1.0, 2.0]}))

    assert list(result.equity_curve.index) == list(idx)
    assert result.metrics["sharpe_ratio"] == 1.1
    assert len(result.trade_history) == 2
    assert result.trade_history["trade_id"].tolist() == [1, 2]
    assert result.rules.startswith("Entry conditions")
    assert result.equity_chart == "chart"
    assert result.backend == "cobra_py_native"
    assert result.raw_backend_object is None
    assert result.train_data is not None


def test_run_optimiser_walk_forward_uses_full_data(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    full = pd.DataFrame(
        {
            "open": [float(100 + i) for i in range(10)],
            "high": [float(101 + i) for i in range(10)],
            "low": [float(99 + i) for i in range(10)],
            "close": [float(100 + i) for i in range(10)],
            "volume": [1000.0 for _ in range(10)],
        },
        index=idx,
    )
    train = full.iloc[:7].copy()
    test = full.iloc[7:].copy()

    class _StubOptimiserResult:
        best_policy = {"ok": True}
        best_score = 0.0

    captured: dict[str, object] = {}

    monkeypatch.setattr("cobra_py.helpers.load_ohlcv", lambda *args, **kwargs: full)
    monkeypatch.setattr("cobra_py.helpers.preprocess", lambda *args, **kwargs: (train, test))
    monkeypatch.setattr("cobra_py.helpers.get_active_registry", lambda *args, **kwargs: [])
    monkeypatch.setattr("cobra_py.helpers.precompute_all", lambda *args, **kwargs: object())
    monkeypatch.setattr("cobra_py.helpers.build_config_space", lambda *args, **kwargs: object())
    monkeypatch.setattr("cobra_py.helpers._run_with_optimiser", lambda *args, **kwargs: _StubOptimiserResult())
    monkeypatch.setattr("cobra_py.helpers.generate_report", lambda *args, **kwargs: {"summary": {}, "best_metrics": {}})

    def _fake_walk_forward_validate(data, *args, **kwargs):
        captured["rows"] = len(data)
        captured["start"] = data.index.min()
        captured["end"] = data.index.max()
        return {"wf": True}

    monkeypatch.setattr("cobra_py.helpers.walk_forward_validate", _fake_walk_forward_validate)
    monkeypatch.setattr("cobra_py.helpers.plot_equity_curves", lambda *args, **kwargs: "chart")

    config = {
        "data": {"min_bars": 1, "train_split": 0.7},
        "policy": {"n_entry_rules": 1, "n_exit_rules": 0},
        "validation": {"walk_forward": True, "n_splits": 3, "train_pct": 0.7},
    }
    out = find_strategy(source=full, config=config, run_walk_forward=True, evaluate_oos=False)

    assert captured["rows"] == len(full)
    assert captured["rows"] != len(test)
    assert captured["start"] == full.index.min()
    assert captured["end"] == full.index.max()
    assert out.walk_forward == {"wf": True}
