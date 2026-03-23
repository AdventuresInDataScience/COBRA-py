from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from rbdpo.indicators.registry import IndicatorDef


@dataclass
class SimpleConfigSpace:
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def sample_configuration(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "n_entry_rules": 3,
            "n_exit_rules": 1,
            "sl_type": self._rng.choice(["pct", "atr_mult", "swing_low", "bb_lower", "trailing_atr"]),
            "tp_type": self._rng.choice(["pct", "atr_mult", "risk_reward", "swing_high", "bb_upper"]),
            "sl_pct": self._rng.uniform(0.005, 0.1),
            "sl_atr_mult": self._rng.uniform(1.0, 4.0),
            "sl_atr_period": self._rng.choice([7, 10, 14, 20]),
            "sl_swing_lookback": self._rng.randint(5, 50),
            "tp_pct": self._rng.uniform(0.005, 0.2),
            "tp_atr_mult": self._rng.uniform(1.0, 6.0),
            "tp_atr_period": self._rng.choice([7, 10, 14, 20]),
            "tp_rr": self._rng.uniform(1.0, 5.0),
            "tp_swing_lookback": self._rng.randint(5, 100),
        }

        for i in range(3):
            active = self._rng.choice([True, False])
            indicator = self._rng.choice(["sma", "ema", "rsi", "macd", "bb", "atr"])
            cfg[f"entry_{i}_active"] = active
            cfg[f"entry_{i}_archetype"] = self._rng.choice(["comparison", "crossover", "band_test", "pattern", "stat_test", "derivative"])
            cfg[f"entry_{i}_indicator"] = indicator
            cfg[f"entry_{i}_operator"] = self._rng.choice([">", "<", "crosses_above", "crosses_below"])
            cfg[f"entry_{i}_threshold"] = self._rng.uniform(0.0, 100.0)
            cfg[f"entry_{i}_comparand"] = self._rng.choice(["price", cfg[f"entry_{i}_threshold"]])
            cfg[f"entry_{i}_lookback"] = self._rng.randint(5, 50)
            cfg[f"entry_{i}_rsi_period"] = self._rng.choice([7, 9, 10, 12, 14, 16, 21])
            cfg[f"entry_{i}_sma_period"] = self._rng.choice([5, 10, 20, 50, 100, 200])
            cfg[f"entry_{i}_ema_period"] = self._rng.choice([5, 10, 20, 50, 100, 200])
            cfg[f"entry_{i}_atr_period"] = self._rng.choice([7, 10, 14, 20])
            cfg[f"entry_{i}_bb_period"] = self._rng.choice([10, 20, 30, 50])
            cfg[f"entry_{i}_bb_std"] = self._rng.choice([1.5, 2.0, 2.5, 3.0])
            cfg[f"entry_{i}_bb_matype"] = self._rng.choice(["sma", "ema"])
            cfg[f"entry_{i}_bb_output"] = self._rng.choice(["upper", "middle", "lower"])
            cfg[f"entry_{i}_macd_fast"] = self._rng.choice([8, 10, 12, 15])
            cfg[f"entry_{i}_macd_slow"] = self._rng.choice([21, 24, 26, 30])
            cfg[f"entry_{i}_macd_signal"] = self._rng.choice([7, 9, 12])
            cfg[f"entry_{i}_macd_output"] = self._rng.choice(["macd", "signal", "hist"])

        for i in range(1):
            active = self._rng.choice([True, False])
            indicator = self._rng.choice(["sma", "ema", "rsi", "macd", "bb", "atr"])
            cfg[f"exit_{i}_active"] = active
            cfg[f"exit_{i}_archetype"] = self._rng.choice(["comparison", "crossover", "band_test", "pattern", "stat_test", "derivative"])
            cfg[f"exit_{i}_indicator"] = indicator
            cfg[f"exit_{i}_operator"] = self._rng.choice([">", "<", "crosses_above", "crosses_below"])
            cfg[f"exit_{i}_threshold"] = self._rng.uniform(0.0, 100.0)
            cfg[f"exit_{i}_comparand"] = self._rng.choice(["price", cfg[f"exit_{i}_threshold"]])
            cfg[f"exit_{i}_lookback"] = self._rng.randint(5, 50)
            cfg[f"exit_{i}_rsi_period"] = self._rng.choice([7, 9, 10, 12, 14, 16, 21])
            cfg[f"exit_{i}_sma_period"] = self._rng.choice([5, 10, 20, 50, 100, 200])
            cfg[f"exit_{i}_ema_period"] = self._rng.choice([5, 10, 20, 50, 100, 200])
            cfg[f"exit_{i}_atr_period"] = self._rng.choice([7, 10, 14, 20])
            cfg[f"exit_{i}_bb_period"] = self._rng.choice([10, 20, 30, 50])
            cfg[f"exit_{i}_bb_std"] = self._rng.choice([1.5, 2.0, 2.5, 3.0])
            cfg[f"exit_{i}_bb_matype"] = self._rng.choice(["sma", "ema"])
            cfg[f"exit_{i}_bb_output"] = self._rng.choice(["upper", "middle", "lower"])
            cfg[f"exit_{i}_macd_fast"] = self._rng.choice([8, 10, 12, 15])
            cfg[f"exit_{i}_macd_slow"] = self._rng.choice([21, 24, 26, 30])
            cfg[f"exit_{i}_macd_signal"] = self._rng.choice([7, 9, 12])
            cfg[f"exit_{i}_macd_output"] = self._rng.choice(["macd", "signal", "hist"])

        return cfg


def build_config_space(n_entry_rules: int, n_exit_rules: int, registry: list[IndicatorDef], seed: int | None = None):
    # If ConfigSpace is present, users can replace this with a strict conditional space.
    return SimpleConfigSpace(seed=seed)


def sample_and_validate(cs: SimpleConfigSpace, n_samples: int = 10) -> list[dict[str, Any]]:
    return [cs.sample_configuration() for _ in range(n_samples)]
