from rbdpo.objective.function import compute_objective
from rbdpo.policy.schema import Policy, RuleConfig, SLConfig, TPConfig


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
