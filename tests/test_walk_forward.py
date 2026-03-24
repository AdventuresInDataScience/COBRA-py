from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from cobra_py.policy.schema import Policy, SLConfig, TPConfig
from cobra_py.validation.walk_forward import walk_forward_validate


@dataclass
class _DummyOptimiseResult:
    best_policy: Policy
    best_score: float


def _make_price_data(n: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.Series(range(100, 100 + n), index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0,
        }
    )


def _optimise_fn(_train: pd.DataFrame, _cfg: dict):
    policy = Policy(
        entry_rules=(),
        exit_rules=(),
        sl_config=SLConfig(sl_type="pct", params=(0.02,)),
        tp_config=TPConfig(tp_type="pct", params=(0.03,)),
        n_active_entry=0,
        n_active_exit=0,
    )
    return _DummyOptimiseResult(best_policy=policy, best_score=0.0)


def test_walk_forward_rejects_invalid_n_splits():
    data = _make_price_data(100)
    with pytest.raises(ValueError, match="n_splits"):
        walk_forward_validate(data, _optimise_fn, {"backtest": {}}, n_splits=0, train_pct=0.7)


def test_walk_forward_rejects_invalid_train_pct():
    data = _make_price_data(100)
    with pytest.raises(ValueError, match="train_pct"):
        walk_forward_validate(data, _optimise_fn, {"backtest": {}}, n_splits=3, train_pct=1.0)


def test_walk_forward_raises_if_all_folds_skipped():
    data = _make_price_data(30)
    with pytest.raises(ValueError, match="zero valid folds"):
        walk_forward_validate(data, _optimise_fn, {"backtest": {}}, n_splits=3, train_pct=0.7)


def test_walk_forward_reports_skipped_fold_metadata():
    data = _make_price_data(59)
    out = walk_forward_validate(data, _optimise_fn, {"backtest": {}}, n_splits=3, train_pct=0.7)

    assert out.n_requested_folds == 3
    assert out.n_completed_folds == 1
    assert out.n_skipped_folds == 2
    assert all(s["reason"] in {"empty_train_or_test", "fold_too_small"} for s in out.skipped_folds)
