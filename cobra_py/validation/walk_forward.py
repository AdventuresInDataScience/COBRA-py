from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from cobra_py.backtest.engine import run_backtest
from cobra_py.indicators.precompute import precompute_all
from cobra_py.indicators.registry import DEFAULT_REGISTRY


@dataclass
class WalkForwardResult:
    fold_results: list[dict]
    oos_sharpe_mean: float
    oos_sharpe_std: float
    oos_calmar_mean: float
    oos_return_mean: float
    oos_max_drawdown_mean: float
    best_policies_per_fold: list
    n_requested_folds: int
    n_completed_folds: int
    n_skipped_folds: int
    skipped_folds: list[dict]


def walk_forward_validate(
    data: pd.DataFrame,
    optimise_fn: Callable[[pd.DataFrame, Any], Any],
    config: dict[str, Any],
    n_splits: int,
    train_pct: float,
    registry=None,
) -> WalkForwardResult:
    if int(n_splits) < 1:
        raise ValueError("n_splits must be >= 1")
    train_pct_f = float(train_pct)
    if not 0.05 <= train_pct_f <= 0.95:
        raise ValueError("train_pct must be in [0.05, 0.95]")

    n = len(data)
    fold_size = n // n_splits
    folds = []
    policies = []
    skipped: list[dict] = []

    for i in range(n_splits):
        start = i * fold_size
        end = n if i == n_splits - 1 else (i + 1) * fold_size
        fold = data.iloc[start:end]
        if len(fold) < 20:
            skipped.append({"fold": i, "reason": "fold_too_small", "rows": int(len(fold))})
            continue

        split_idx = int(len(fold) * train_pct_f)
        train = fold.iloc[:split_idx]
        test = fold.iloc[split_idx:]
        if train.empty or test.empty:
            skipped.append({"fold": i, "reason": "empty_train_or_test", "rows": int(len(fold)), "split_idx": int(split_idx)})
            continue

        result = optimise_fn(train, config)
        cache_registry = registry if registry is not None else DEFAULT_REGISTRY
        cache_test = precompute_all(test, cache_registry, n_jobs=1)
        test_metrics = run_backtest(result.best_policy, cache_test, test, config.get("backtest", {}))

        folds.append({"fold": i, "metrics": test_metrics, "best_score": result.best_score})
        policies.append(result.best_policy)

    if not folds:
        raise ValueError(
            f"Walk-forward produced zero valid folds (requested={n_splits}, skipped={len(skipped)}). "
            "Adjust n_splits/train_pct or provide more data."
        )

    sharpe = [f["metrics"].get("sharpe_ratio", -999.0) for f in folds]
    calmar = [f["metrics"].get("calmar_ratio", -999.0) for f in folds]
    ret = [f["metrics"].get("total_return", -999.0) for f in folds]
    dd = [f["metrics"].get("max_drawdown", -999.0) for f in folds]

    return WalkForwardResult(
        fold_results=folds,
        oos_sharpe_mean=float(np.mean(sharpe)) if sharpe else -999.0,
        oos_sharpe_std=float(np.std(sharpe)) if sharpe else -999.0,
        oos_calmar_mean=float(np.mean(calmar)) if calmar else -999.0,
        oos_return_mean=float(np.mean(ret)) if ret else -999.0,
        oos_max_drawdown_mean=float(np.mean(dd)) if dd else -999.0,
        best_policies_per_fold=policies,
        n_requested_folds=int(n_splits),
        n_completed_folds=int(len(folds)),
        n_skipped_folds=int(len(skipped)),
        skipped_folds=skipped,
    )

