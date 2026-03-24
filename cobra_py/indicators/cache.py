from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict

import numpy as np


class IndicatorCache:
    def __init__(self) -> None:
        self._store: DefaultDict[str, dict[tuple[tuple, str], np.ndarray]] = defaultdict(dict)

    def store(self, indicator_name: str, params: tuple, output_name: str, array: np.ndarray) -> None:
        arr = np.asarray(array)
        if arr.dtype.kind == "f":
            arr = arr.astype(np.float32, copy=False)
        self._store[indicator_name][(tuple(params), output_name)] = arr

    def get(self, indicator_name: str, params: tuple, output_name: str) -> np.ndarray | None:
        return self._store.get(indicator_name, {}).get((tuple(params), output_name))

    def memory_usage_gb(self) -> float:
        total_bytes = 0
        for outputs in self._store.values():
            for arr in outputs.values():
                total_bytes += int(arr.nbytes)
        return total_bytes / (1024 ** 3)

    def available_params(self, indicator_name: str) -> list[tuple]:
        outputs = self._store.get(indicator_name, {})
        params = sorted({k[0] for k in outputs})
        return list(params)
