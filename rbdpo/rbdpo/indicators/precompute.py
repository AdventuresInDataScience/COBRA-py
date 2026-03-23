from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .cache import IndicatorCache
from .registry import IndicatorDef, param_product

logger = logging.getLogger(__name__)


def _compute_one(indicator: IndicatorDef, data: pd.DataFrame, params: dict) -> tuple[tuple, dict[str, np.ndarray] | None, str | None]:
    try:
        output = indicator.compute_fn(data, **params)
        return tuple(params.get(k) for k in indicator.param_grid.keys()), output, None
    except Exception as exc:  # pragma: no cover - defensive branch
        return tuple(params.get(k) for k in indicator.param_grid.keys()), None, str(exc)


def precompute_all(data: pd.DataFrame, registry: Iterable[IndicatorDef], n_jobs: int = -1) -> IndicatorCache:
    cache = IndicatorCache()

    for indicator in registry:
        combos = param_product(indicator.param_grid)
        if indicator.constraints is not None:
            combos = [c for c in combos if indicator.constraints(c)]

        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_one)(indicator, data, combo) for combo in combos
        )

        for params_tuple, output, err in results:
            if err is not None or output is None:
                logger.warning("Indicator combo failed: %s %s (%s)", indicator.name, params_tuple, err)
                for out_name in indicator.outputs:
                    cache.store(indicator.name, params_tuple, out_name, np.full(len(data), np.nan, dtype=np.float32))
                continue
            for out_name, arr in output.items():
                cache.store(indicator.name, params_tuple, out_name, np.asarray(arr))

    logger.info("Indicator cache memory usage: %.3f GB", cache.memory_usage_gb())
    return cache
