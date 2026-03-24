from __future__ import annotations

import time
from typing import Any

import pandas as pd

from cobra_py.backtest.engine import run_backtest
from cobra_py.indicators.cache import IndicatorCache
from cobra_py.objective.function import compute_objective
from cobra_py.policy.decoder import decode_config
from cobra_py.search.types import OptimisationResult


def run_nevergrad(
    cache: IndicatorCache,
    data: pd.DataFrame,
    config_space,
    obj_config: dict[str, Any],
    backtest_config: dict[str, Any],
    budget: int,
    seed: int = 42,
    optimiser_name: str = "NGOpt",
) -> OptimisationResult:
    try:
        import nevergrad as ng
    except ImportError as exc:
        raise ImportError("nevergrad optimiser selected but package is not installed. Install with `pip install nevergrad`.") from exc

    start = time.time()
    history: list[dict] = []
    best_score = float("inf")
    best_policy = None
    best_metrics = None

    n_eval = int(max(1, budget))
    parametrization = ng.p.Scalar(lower=0, upper=2_147_483_647).set_integer_casting()
    optimiser_key = str(optimiser_name).strip()
    optimiser_cls = getattr(ng.optimizers, optimiser_key, None)
    if optimiser_cls is None:
        raise ValueError(f"Unknown nevergrad optimizer '{optimiser_key}'.")
    try:
        optimizer = optimiser_cls(parametrization=parametrization, budget=n_eval, num_workers=1)
    except TypeError:
        optimizer = optimiser_cls(parametrization=parametrization, budget=n_eval)

    def sample_cfg(sample_seed: int) -> dict[str, Any]:
        if hasattr(config_space, "sample_with_seed"):
            return config_space.sample_with_seed(int(sample_seed))
        return config_space.sample_configuration()

    for _ in range(n_eval):
        candidate = optimizer.ask()
        sample_seed = int(candidate.value)
        cfg = sample_cfg(sample_seed)
        cfg["n_entry_rules"] = int(obj_config.get("n_entry_rules", 3))
        cfg["n_exit_rules"] = int(obj_config.get("n_exit_rules", 1))

        policy = decode_config(cfg, cache)
        if policy is None:
            score = 999.0
            metrics = {"n_trades": 0}
        else:
            eval_bt_cfg = dict(backtest_config)
            if "leverage" in cfg:
                eval_bt_cfg["leverage"] = float(cfg["leverage"])
            if "borrow_cost_rate" in cfg:
                eval_bt_cfg["borrow_cost_rate"] = float(cfg["borrow_cost_rate"])

            metrics = run_backtest(policy, cache, data, eval_bt_cfg)
            score = compute_objective(metrics, policy, obj_config)

        optimizer.tell(candidate, float(score))
        history.append({"config": cfg, "score": float(score), "metrics": metrics, "sample_seed": sample_seed})

        if policy is not None and score < best_score:
            best_score = float(score)
            best_policy = policy
            best_metrics = metrics

    if best_policy is None or best_metrics is None:
        fallback_cfg = sample_cfg(seed)
        fallback_cfg["n_entry_rules"] = int(obj_config.get("n_entry_rules", 3))
        fallback_cfg["n_exit_rules"] = int(obj_config.get("n_exit_rules", 1))
        fallback = decode_config(fallback_cfg, cache)
        if fallback is None:
            raise RuntimeError("No valid policy sampled by Nevergrad. Increase budget or adjust search space.")
        best_policy = fallback
        best_metrics = {"n_trades": 0}
        best_score = 999.0

    return OptimisationResult(
        best_policy=best_policy,
        best_metrics=best_metrics,
        best_score=best_score,
        objective_name=str(obj_config.get("objective", "sharpe")),
        n_evaluations=n_eval,
        optimiser_name=f"nevergrad({optimiser_key})",
        seed=seed,
        runtime_seconds=time.time() - start,
        full_history=history,
    )

