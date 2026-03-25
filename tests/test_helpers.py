from __future__ import annotations

import pandas as pd
import pytest

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
