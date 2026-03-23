from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from rbdpo.backtest.engine import run_backtest
from rbdpo.indicators.precompute import precompute_all
from rbdpo.indicators.registry import DEFAULT_REGISTRY


@dataclass
class WalkForwardResult:
    fold_results: list[dict]
    oos_sharpe_mean: float
    oos_sharpe_std: float
    oos_calmar_mean: float
    oos_return_mean: float
    oos_max_drawdown_mean: float
    best_policies_per_fold: list


def walk_forward_validate(
    data: pd.DataFrame,
    optimise_fn: Callable[[pd.DataFrame, Any], Any],
    config: dict[str, Any],
    n_splits: int,
    train_pct: float,
) -> WalkForwardResult:
    n = len(data)
    fold_size = n // n_splits
    folds = []
    policies = []

    for i in range(n_splits):
        start = i * fold_size
        end = n if i == n_splits - 1 else (i + 1) * fold_size
        fold = data.iloc[start:end]
        if len(fold) < 20:
            continue

        split_idx = int(len(fold) * train_pct)
        train = fold.iloc[:split_idx]
        test = fold.iloc[split_idx:]
        if train.empty or test.empty:
            continue

        result = optimise_fn(train, config)
        cache_test = precompute_all(test, DEFAULT_REGISTRY, n_jobs=1)
        test_metrics = run_backtest(result.best_policy, cache_test, test, config.get("backtest", {}))

        folds.append({"fold": i, "metrics": test_metrics, "best_score": result.best_score})
        policies.append(result.best_policy)

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
    )
