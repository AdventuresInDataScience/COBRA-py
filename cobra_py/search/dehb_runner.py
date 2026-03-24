from __future__ import annotations

import random
import time
from typing import Any

import numpy as np
import pandas as pd

from cobra_py.backtest.engine import run_backtest
from cobra_py.indicators.cache import IndicatorCache
from cobra_py.objective.function import compute_objective
from cobra_py.policy.decoder import decode_config
from cobra_py.search.types import OptimisationResult


_MAX_SEED = 2_147_483_647


def _clamp_seed(value: int) -> int:
    return int(max(0, min(_MAX_SEED, int(value))))


def _seed_from_native_config(config: Any) -> int:
    if isinstance(config, dict):
        for key in ("sample_seed", "seed", "x", "param"):
            if key in config:
                return _clamp_seed(int(round(float(config[key]))))

    if isinstance(config, np.ndarray):
        if config.size == 0:
            return 0
        value = float(config.ravel()[0])
    elif isinstance(config, (list, tuple)):
        if len(config) == 0:
            return 0
        value = float(config[0])
    else:
        value = float(config)

    # Some DEHB versions emit normalized vectors in [0, 1].
    if 0.0 <= value <= 1.0:
        value = value * _MAX_SEED
    return _clamp_seed(int(round(value)))


def _subset_by_fidelity(data: pd.DataFrame, fidelity: float | None) -> pd.DataFrame:
    if fidelity is None:
        return data
    frac = float(max(0.05, min(1.0, fidelity)))
    n = max(50, int(len(data) * frac))
    if n >= len(data):
        return data
    return data.iloc[:n]


def _build_eval_fn(cache: IndicatorCache, data: pd.DataFrame, config_space, obj_config: dict[str, Any], backtest_config: dict[str, Any]):
    def sample_cfg(sample_seed: int) -> dict[str, Any]:
        if hasattr(config_space, "sample_with_seed"):
            return config_space.sample_with_seed(int(sample_seed))
        return config_space.sample_configuration()

    def evaluate_seed(sample_seed: int, fidelity: float | None = None) -> tuple[float, Any, dict[str, Any], dict[str, Any]]:
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

            eval_data = _subset_by_fidelity(data, fidelity)
            metrics = run_backtest(policy, cache, eval_data, eval_bt_cfg)
            score = compute_objective(metrics, policy, obj_config)

        return float(score), policy, metrics, cfg

    return evaluate_seed, sample_cfg


def _run_seed_de_backend(
    cache: IndicatorCache,
    data: pd.DataFrame,
    config_space,
    obj_config: dict[str, Any],
    backtest_config: dict[str, Any],
    budget: int,
    seed: int,
    mutation_factor: float,
    crossover_rate: float,
    population_size: int,
) -> OptimisationResult:
    start = time.time()
    history: list[dict] = []
    rng = random.Random(seed)
    evaluate_seed, sample_cfg = _build_eval_fn(cache, data, config_space, obj_config, backtest_config)

    best_score = float("inf")
    best_policy = None
    best_metrics = None

    n_eval = int(max(1, budget))

    # Differential-evolution style search over deterministic config seeds.
    pop_size = int(max(4, min(population_size, n_eval)))
    population = [rng.randint(0, _MAX_SEED) for _ in range(pop_size)]
    population_scores = [999.0 for _ in range(pop_size)]

    evaluations = 0
    for i in range(pop_size):
        score, policy, metrics, cfg = evaluate_seed(population[i])
        population_scores[i] = score
        history.append(
            {
                "config": cfg,
                "score": score,
                "metrics": metrics,
                "sample_seed": int(population[i]),
                "stage": "init",
            }
        )
        evaluations += 1
        if policy is not None and score < best_score:
            best_score = score
            best_policy = policy
            best_metrics = metrics

    while evaluations < n_eval:
        for i in range(pop_size):
            if evaluations >= n_eval:
                break

            pool = [idx for idx in range(pop_size) if idx != i]
            a, b, c = rng.sample(pool, 3) if len(pool) >= 3 else (i, i, i)
            mutant = int(round(population[a] + float(mutation_factor) * (population[b] - population[c])))
            mutant = _clamp_seed(mutant)
            trial_seed = mutant if rng.random() < float(crossover_rate) else int(population[i])

            score, policy, metrics, cfg = evaluate_seed(trial_seed)
            accepted = score <= population_scores[i]
            if accepted:
                population[i] = trial_seed
                population_scores[i] = score

            history.append(
                {
                    "config": cfg,
                    "score": score,
                    "metrics": metrics,
                    "sample_seed": int(trial_seed),
                    "stage": "evolve",
                    "accepted": bool(accepted),
                }
            )
            evaluations += 1

            if policy is not None and score < best_score:
                best_score = float(score)
                best_policy = policy
                best_metrics = metrics

    if n_eval == 1 and best_policy is None:
        score, policy, metrics, _ = evaluate_seed(seed)
        if policy is not None:
            best_score = float(score)
            best_policy = policy
            best_metrics = metrics

    if best_policy is None or best_metrics is None:
        fallback = decode_config(sample_cfg(seed), cache)
        if fallback is None:
            raise RuntimeError("No valid policy sampled. Increase budget or adjust search space.")
        best_policy = fallback
        best_metrics = {"n_trades": 0}
        best_score = 999.0

    return OptimisationResult(
        best_policy=best_policy,
        best_metrics=best_metrics,
        best_score=best_score,
        objective_name=str(obj_config.get("objective", "sharpe")),
        n_evaluations=n_eval,
        optimiser_name="dehb(seed-de-mvp)",
        seed=seed,
        runtime_seconds=time.time() - start,
        full_history=history,
    )


def _run_native_dehb_backend(
    cache: IndicatorCache,
    data: pd.DataFrame,
    config_space,
    obj_config: dict[str, Any],
    backtest_config: dict[str, Any],
    budget: int,
    seed: int,
    min_fidelity: float,
    max_fidelity: float,
    n_workers: int,
) -> OptimisationResult:
    try:
        from dehb import DEHB
    except ImportError as exc:
        raise ImportError("Native DEHB backend requested but package is not installed. Install with `pip install DEHB`.") from exc

    start = time.time()
    history: list[dict] = []
    evaluate_seed, sample_cfg = _build_eval_fn(cache, data, config_space, obj_config, backtest_config)

    best_score = float("inf")
    best_policy = None
    best_metrics = None

    def objective(config, fidelity=None, **kwargs):
        nonlocal best_score, best_policy, best_metrics
        sample_seed = _seed_from_native_config(config)
        score, policy, metrics, cfg = evaluate_seed(sample_seed, fidelity=float(fidelity) if fidelity is not None else None)

        history.append(
            {
                "config": cfg,
                "score": score,
                "metrics": metrics,
                "sample_seed": int(sample_seed),
                "stage": "native_dehb",
                "fidelity": float(fidelity) if fidelity is not None else None,
            }
        )

        if policy is not None and score < best_score:
            best_score = float(score)
            best_policy = policy
            best_metrics = metrics

        return {"fitness": float(score), "cost": float(fidelity) if fidelity is not None else 1.0, "info": {"sample_seed": int(sample_seed)}}

    n_eval = int(max(1, budget))
    dehb_kwargs = {
        "f": objective,
        "dimensions": 1,
        "min_fidelity": float(min_fidelity),
        "max_fidelity": float(max(max_fidelity, min_fidelity)),
        "n_workers": int(max(1, n_workers)),
        "seed": int(seed),
    }

    try:
        optimiser = DEHB(**dehb_kwargs)
    except TypeError:
        dehb_kwargs.pop("seed", None)
        optimiser = DEHB(**dehb_kwargs)

    ran = False
    for kwargs in (
        {"fevals": n_eval, "verbose": False},
        {"fevals": n_eval},
        {"total_cost": n_eval, "verbose": False},
        {"total_cost": n_eval},
    ):
        try:
            optimiser.run(**kwargs)
            ran = True
            break
        except TypeError:
            continue

    if not ran:
        raise RuntimeError("Unable to run native DEHB due to unexpected API signature.")

    if best_policy is None or best_metrics is None:
        fallback = decode_config(sample_cfg(seed), cache)
        if fallback is None:
            raise RuntimeError("No valid policy sampled by native DEHB. Increase budget or adjust search space.")
        best_policy = fallback
        best_metrics = {"n_trades": 0}
        best_score = 999.0

    return OptimisationResult(
        best_policy=best_policy,
        best_metrics=best_metrics,
        best_score=best_score,
        objective_name=str(obj_config.get("objective", "sharpe")),
        n_evaluations=int(max(1, min(n_eval, len(history) if history else n_eval))),
        optimiser_name="dehb(native)",
        seed=seed,
        runtime_seconds=time.time() - start,
        full_history=history,
    )


def run_dehb(
    cache: IndicatorCache,
    data: pd.DataFrame,
    config_space,
    obj_config: dict[str, Any],
    backtest_config: dict[str, Any],
    budget: int,
    seed: int = 42,
    mutation_factor: float = 0.8,
    crossover_rate: float = 0.7,
    population_size: int = 24,
    dehb_backend: str = "auto",
    min_fidelity: float = 0.2,
    max_fidelity: float = 1.0,
    n_workers: int = 1,
) -> OptimisationResult:
    backend = str(dehb_backend).strip().lower()
    if backend not in {"auto", "native", "seed_de"}:
        raise ValueError("dehb_backend must be one of: auto, native, seed_de")

    if backend in {"auto", "native"}:
        try:
            return _run_native_dehb_backend(
                cache=cache,
                data=data,
                config_space=config_space,
                obj_config=obj_config,
                backtest_config=backtest_config,
                budget=budget,
                seed=seed,
                min_fidelity=min_fidelity,
                max_fidelity=max_fidelity,
                n_workers=n_workers,
            )
        except ImportError:
            if backend == "native":
                raise
        except Exception:
            if backend == "native":
                raise

    return _run_seed_de_backend(
        cache=cache,
        data=data,
        config_space=config_space,
        obj_config=obj_config,
        backtest_config=backtest_config,
        budget=budget,
        seed=seed,
        mutation_factor=mutation_factor,
        crossover_rate=crossover_rate,
        population_size=population_size,
    )

