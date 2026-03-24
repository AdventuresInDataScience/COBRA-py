from __future__ import annotations

import math
from typing import Any

from cobra_py.policy.schema import Policy


def _validate_composite_weights(raw_weights: Any) -> tuple[float, float, float, float]:
    if not isinstance(raw_weights, (list, tuple)) or len(raw_weights) != 4:
        raise ValueError("objective.composite_weights must be a list/tuple of exactly 4 numeric values")
    weights = tuple(float(w) for w in raw_weights)
    if not all(math.isfinite(w) for w in weights):
        raise ValueError("objective.composite_weights must contain finite numeric values")
    return weights  # type: ignore[return-value]


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
    elif objective_key == "max_return_dd_cap":
        dd_cap = abs(float(cfg.get("max_drawdown_cap", 0.20)))
        max_dd = abs(float(metrics.get("max_drawdown", -999.0)))
        if max_dd > dd_cap:
            # Hard feasibility penalty: maximize return only among strategies that satisfy drawdown cap.
            return 999.0 + (max_dd - dd_cap)
        raw = -float(metrics.get("total_return", -999.0))
    elif objective_key == "composite":
        w0, w1, w2, w3 = _validate_composite_weights(cfg.get("composite_weights", [0.5, 0.3, 0.1, 0.1]))
        raw = -(
            w0 * float(metrics.get("sharpe_ratio", -999.0))
            + w1 * float(metrics.get("calmar_ratio", -999.0))
            + w2 * float(metrics.get("total_return", -999.0))
            - w3 * abs(float(metrics.get("max_drawdown", 0.0)))
        )
    else:
        raise ValueError(f"Unknown objective: {objective_key}")

    n_rules = policy.n_active_entry + policy.n_active_exit
    return float(raw + lam * n_rules)

