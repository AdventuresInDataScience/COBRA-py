from __future__ import annotations

from typing import Any

from cobra_py.policy.schema import Policy


def compute_objective(metrics: dict[str, float], policy: Policy, config: dict[str, Any] | None = None) -> float:
    cfg = config or {}
    objective_key = cfg.get("objective", "sharpe")
    lam = float(cfg.get("complexity_penalty", 0.02))
    min_trades = int(cfg.get("min_trades", 10))

    if int(metrics.get("n_trades", 0)) < min_trades:
        return 999.0

    if objective_key == "sharpe":
        raw = -float(metrics.get("sharpe_ratio", -999.0))
    elif objective_key == "calmar":
        raw = -float(metrics.get("calmar_ratio", -999.0))
    elif objective_key == "sortino":
        raw = -float(metrics.get("sortino_ratio", -999.0))
    elif objective_key == "ulcer":
        # Ulcer index is a downside-risk metric where lower values are better.
        raw = float(metrics.get("ulcer_index", 999.0))
    elif objective_key == "max_return":
        raw = -float(metrics.get("total_return", -999.0))
    elif objective_key == "composite":
        w = cfg.get("composite_weights", [0.5, 0.3, 0.1, 0.1])
        raw = -(
            float(w[0]) * float(metrics.get("sharpe_ratio", -999.0))
            + float(w[1]) * float(metrics.get("calmar_ratio", -999.0))
            + float(w[2]) * float(metrics.get("total_return", -999.0))
            - float(w[3]) * abs(float(metrics.get("max_drawdown", 0.0)))
        )
    else:
        raise ValueError(f"Unknown objective: {objective_key}")

    n_rules = policy.n_active_entry + policy.n_active_exit
    return float(raw + lam * n_rules)

