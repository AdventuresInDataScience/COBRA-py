from __future__ import annotations

from dataclasses import dataclass

from cobra_py.policy.schema import Policy


@dataclass
class OptimisationResult:
    best_policy: Policy
    best_metrics: dict
    best_score: float
    n_evaluations: int
    optimiser_name: str
    seed: int
    runtime_seconds: float
    full_history: list[dict]

