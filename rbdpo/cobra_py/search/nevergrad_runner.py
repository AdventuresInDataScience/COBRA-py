from __future__ import annotations

from typing import Any

import pandas as pd

from cobra_py.indicators.cache import IndicatorCache
from cobra_py.search.dehb_runner import run_dehb
from cobra_py.search.types import OptimisationResult


def run_nevergrad(
    cache: IndicatorCache,
    data: pd.DataFrame,
    config_space,
    obj_config: dict[str, Any],
    backtest_config: dict[str, Any],
    budget: int,
    seed: int = 42,
) -> OptimisationResult:
    # MVP uses a shared search loop for both optimiser options.
    return run_dehb(
        cache=cache,
        data=data,
        config_space=config_space,
        obj_config=obj_config,
        backtest_config=backtest_config,
        budget=budget,
        seed=seed,
    )

