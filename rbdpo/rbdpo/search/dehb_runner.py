from __future__ import annotations

import time
from typing import Any

import pandas as pd

from rbdpo.backtest.engine import run_backtest
from rbdpo.indicators.cache import IndicatorCache
from rbdpo.objective.function import compute_objective
from rbdpo.policy.decoder import decode_config
from rbdpo.search.types import OptimisationResult


def run_dehb(
    cache: IndicatorCache,
    data: pd.DataFrame,
    config_space,
    obj_config: dict[str, Any],
    backtest_config: dict[str, Any],
    budget: int,
    seed: int = 42,
) -> OptimisationResult:
    start = time.time()
    history: list[dict] = []

    best_score = float("inf")
    best_policy = None
    best_metrics = None

    n_eval = int(max(1, budget))
    for _ in range(n_eval):
        cfg = config_space.sample_configuration()
        cfg["n_entry_rules"] = int(obj_config.get("n_entry_rules", 3))
        cfg["n_exit_rules"] = int(obj_config.get("n_exit_rules", 1))

        policy = decode_config(cfg, cache)
        if policy is None:
            score = 999.0
            metrics = {"n_trades": 0}
        else:
            metrics = run_backtest(policy, cache, data, backtest_config)
            score = compute_objective(metrics, policy, obj_config)

        history.append({"config": cfg, "score": float(score), "metrics": metrics})
        if policy is not None and score < best_score:
            best_score = float(score)
            best_policy = policy
            best_metrics = metrics

    if best_policy is None or best_metrics is None:
        # decode the first config to satisfy type contract if all invalid
        fallback = decode_config(config_space.sample_configuration(), cache)
        if fallback is None:
            raise RuntimeError("No valid policy sampled. Increase budget or adjust search space.")
        best_policy = fallback
        best_metrics = {"n_trades": 0}
        best_score = 999.0

    return OptimisationResult(
        best_policy=best_policy,
        best_metrics=best_metrics,
        best_score=best_score,
        n_evaluations=n_eval,
        optimiser_name="dehb(random-search-mvp)",
        seed=seed,
        runtime_seconds=time.time() - start,
        full_history=history,
    )
