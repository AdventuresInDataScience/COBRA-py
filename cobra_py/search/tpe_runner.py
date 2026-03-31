from __future__ import annotations

import time
from typing import Any

import pandas as pd

from cobra_py.backtest.engine import run_backtest
from cobra_py.indicators.cache import IndicatorCache
from cobra_py.objective.function import compute_objective
from cobra_py.policy.decoder import decode_config
from cobra_py.search.types import OptimisationResult


def run_tpe(
    cache: IndicatorCache,
    data: pd.DataFrame,
    config_space,
    obj_config: dict[str, Any],
    backtest_config: dict[str, Any],
    budget: int,
    seed: int = 42,
    multivariate: bool = True,
    group: bool = True,
    n_startup_trials: int = 10,
    constant_liar: bool = False,
) -> OptimisationResult:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("tpe optimiser selected but optuna is not installed. Install with `pip install optuna`.") from exc

    start = time.time()
    history: list[dict] = []
    best_score = float("inf")
    best_policy = None
    best_metrics = None

    n_eval = int(max(1, budget))
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        multivariate=bool(multivariate),
        group=bool(group),
        n_startup_trials=int(max(1, n_startup_trials)),
        constant_liar=bool(constant_liar),
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=int(max(1, n_startup_trials)), n_warmup_steps=0)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    def sample_cfg(sample_seed: int) -> dict[str, Any]:
        if hasattr(config_space, "sample_with_seed"):
            return config_space.sample_with_seed(int(sample_seed))
        return config_space.sample_configuration()

    def objective(trial) -> float:
        nonlocal best_score, best_policy, best_metrics
        if hasattr(config_space, "suggest_with_optuna"):
            cfg = config_space.suggest_with_optuna(trial)
        else:
            sample_seed = int(trial.suggest_int("sample_seed", 0, 2_147_483_647))
            cfg = sample_cfg(sample_seed)
        cfg["n_entry_rules"] = int(obj_config.get("n_entry_rules", 3))
        cfg["n_exit_rules"] = int(obj_config.get("n_exit_rules", 1))

        policy = decode_config(cfg, cache)
        if policy is None:
            score = 999.0
            metrics = {"n_trades": 0}
            history.append(
                {
                    "config": cfg,
                    "score": float(score),
                    "metrics": metrics,
                    "trial_number": int(trial.number),
                    "pruned": False,
                }
            )
            return float(score)
        else:
            eval_bt_cfg = dict(backtest_config)
            if "leverage" in cfg:
                eval_bt_cfg["leverage"] = float(cfg["leverage"])
            if "borrow_cost_rate" in cfg:
                eval_bt_cfg["borrow_cost_rate"] = float(cfg["borrow_cost_rate"])

            subset_n = max(50, int(len(data) * 0.3))
            subset_data = data.iloc[:subset_n] if subset_n < len(data) else data
            intermediate_metrics = run_backtest(policy, cache, subset_data, eval_bt_cfg)
            intermediate_score = compute_objective(intermediate_metrics, policy, obj_config)
            trial.report(float(intermediate_score), step=0)
            if trial.should_prune():
                history.append(
                    {
                        "config": cfg,
                        "score": float(intermediate_score),
                        "metrics": intermediate_metrics,
                        "trial_number": int(trial.number),
                        "pruned": True,
                        "stage": "subset_30pct",
                    }
                )
                raise optuna.TrialPruned()

            metrics = run_backtest(policy, cache, data, eval_bt_cfg)
            score = compute_objective(metrics, policy, obj_config)

        history.append(
            {
                "config": cfg,
                "score": float(score),
                "metrics": metrics,
                "trial_number": int(trial.number),
                "pruned": False,
            }
        )

        if policy is not None and score < best_score:
            best_score = float(score)
            best_policy = policy
            best_metrics = metrics

        return float(score)

    study.optimize(objective, n_trials=n_eval, show_progress_bar=False)

    if best_policy is None or best_metrics is None:
        fallback_cfg = sample_cfg(seed)
        fallback_cfg["n_entry_rules"] = int(obj_config.get("n_entry_rules", 3))
        fallback_cfg["n_exit_rules"] = int(obj_config.get("n_exit_rules", 1))
        fallback = decode_config(fallback_cfg, cache)
        if fallback is None:
            raise RuntimeError("No valid policy sampled by TPE. Increase budget or adjust search space.")
        best_policy = fallback
        best_metrics = {"n_trades": 0}
        best_score = 999.0

    return OptimisationResult(
        best_policy=best_policy,
        best_metrics=best_metrics,
        best_score=best_score,
        objective_name=str(obj_config.get("objective", "sharpe")),
        n_evaluations=n_eval,
        optimiser_name="tpe(optuna)",
        seed=seed,
        runtime_seconds=time.time() - start,
        full_history=history,
    )
