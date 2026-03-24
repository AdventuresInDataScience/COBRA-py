import pytest

from cobra_py.objective.function import compute_objective
from cobra_py.policy.schema import Policy, RuleConfig, SLConfig, TPConfig


RULE = RuleConfig("comparison", "rsi", (14,), "rsi", ">", 50.0)
POLICY_SMALL = Policy((RULE,), (), SLConfig("pct", (0.02,)), TPConfig("pct", (0.03,)), 1, 0)
POLICY_LARGE = Policy((RULE, RULE, RULE), (RULE,), SLConfig("pct", (0.02,)), TPConfig("pct", (0.03,)), 3, 1)


def test_objective_prefers_better_metrics():
    good = {"sharpe_ratio": 2.0, "calmar_ratio": 1.5, "sortino_ratio": 3.0, "total_return": 0.4, "max_drawdown": -0.1, "n_trades": 20}
    bad = {"sharpe_ratio": 0.5, "calmar_ratio": 0.2, "sortino_ratio": 0.7, "total_return": 0.1, "max_drawdown": -0.3, "n_trades": 20}
    assert compute_objective(good, POLICY_SMALL, {"objective": "sharpe"}) < compute_objective(bad, POLICY_SMALL, {"objective": "sharpe"})


def test_complexity_penalty():
    m = {"sharpe_ratio": 1.0, "calmar_ratio": 1.0, "sortino_ratio": 1.0, "total_return": 0.2, "max_drawdown": -0.1, "n_trades": 20}
    small = compute_objective(m, POLICY_SMALL, {"objective": "sharpe", "complexity_penalty": 0.1})
    large = compute_objective(m, POLICY_LARGE, {"objective": "sharpe", "complexity_penalty": 0.1})
    assert large > small


def test_ulcer_objective_prefers_lower_ulcer_index():
    low_risk = {"ulcer_index": 0.03, "n_trades": 20}
    high_risk = {"ulcer_index": 0.15, "n_trades": 20}
    assert compute_objective(low_risk, POLICY_SMALL, {"objective": "ulcer"}) < compute_objective(high_risk, POLICY_SMALL, {"objective": "ulcer"})


def test_max_return_dd_cap_prefers_feasible_high_return():
    feasible_hi = {"total_return": 0.30, "max_drawdown": -0.15, "n_trades": 20}
    feasible_lo = {"total_return": 0.10, "max_drawdown": -0.10, "n_trades": 20}
    cfg = {"objective": "max_return_dd_cap", "max_drawdown_cap": 0.20}
    assert compute_objective(feasible_hi, POLICY_SMALL, cfg) < compute_objective(feasible_lo, POLICY_SMALL, cfg)


def test_max_return_dd_cap_penalises_drawdown_breach():
    feasible = {"total_return": 0.10, "max_drawdown": -0.15, "n_trades": 20}
    violating = {"total_return": 0.60, "max_drawdown": -0.35, "n_trades": 20}
    cfg = {"objective": "max_return_dd_cap", "max_drawdown_cap": 0.20}
    assert compute_objective(violating, POLICY_SMALL, cfg) > compute_objective(feasible, POLICY_SMALL, cfg)


def test_composite_weights_must_have_exactly_four_values():
    metrics = {"sharpe_ratio": 1.0, "calmar_ratio": 1.0, "total_return": 0.1, "max_drawdown": -0.1, "n_trades": 20}
    with pytest.raises(ValueError, match="exactly 4"):
        compute_objective(metrics, POLICY_SMALL, {"objective": "composite", "composite_weights": [0.6, 0.4, 0.2]})


def test_composite_weights_must_be_finite():
    metrics = {"sharpe_ratio": 1.0, "calmar_ratio": 1.0, "total_return": 0.1, "max_drawdown": -0.1, "n_trades": 20}
    with pytest.raises(ValueError, match="finite"):
        compute_objective(metrics, POLICY_SMALL, {"objective": "composite", "composite_weights": [0.6, 0.4, float("nan"), 0.1]})

