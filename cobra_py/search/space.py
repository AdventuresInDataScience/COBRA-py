from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from cobra_py.indicators.registry import IndicatorDef

SUPPORTED_RULE_INDICATORS = {"sma", "ema", "wma", "rsi", "macd", "bb", "atr"}


def _valid_rule_indicators(registry: list[IndicatorDef]) -> list[IndicatorDef]:
    selected = [ind for ind in registry if ind.name in SUPPORTED_RULE_INDICATORS]
    if not selected:
        raise ValueError(
            "No rule-compatible indicators selected. Include at least one of: "
            f"{sorted(SUPPORTED_RULE_INDICATORS)}"
        )
    return selected


@dataclass
class SimpleConfigSpace:
    n_entry_rules: int
    n_exit_rules: int
    registry: list[IndicatorDef]
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._rule_indicators = _valid_rule_indicators(self.registry)

    def _pick(self, values: list[Any], fallback: list[Any]) -> Any:
        candidates = values if values else fallback
        return self._rng.choice(candidates)

    def _indicator(self, name: str) -> IndicatorDef | None:
        for ind in self._rule_indicators:
            if ind.name == name:
                return ind
        return None

    def _sample_rule_params(self, prefix: str, indicator: str, cfg: dict[str, Any]) -> None:
        ind = self._indicator(indicator)
        if ind is None:
            return

        grid = ind.param_grid
        if indicator in {"sma", "ema", "wma"}:
            cfg[f"{prefix}_{indicator}_period"] = int(self._pick(list(grid.get("period", [])), [20]))
        elif indicator == "rsi":
            cfg[f"{prefix}_rsi_period"] = int(self._pick(list(grid.get("period", [])), [14]))
        elif indicator == "atr":
            cfg[f"{prefix}_atr_period"] = int(self._pick(list(grid.get("period", [])), [14]))
        elif indicator == "bb":
            cfg[f"{prefix}_bb_period"] = int(self._pick(list(grid.get("period", [])), [20]))
            cfg[f"{prefix}_bb_std"] = float(self._pick(list(grid.get("std", [])), [2.0]))
            cfg[f"{prefix}_bb_matype"] = str(self._pick(list(grid.get("ma_type", [])), ["sma"]))
            cfg[f"{prefix}_bb_output"] = self._pick(["upper", "middle", "lower"], ["middle"])
        elif indicator == "macd":
            fast_values = [int(v) for v in list(grid.get("fast", []))] or [12]
            slow_values = [int(v) for v in list(grid.get("slow", []))] or [26]
            signal_values = [int(v) for v in list(grid.get("signal", []))] or [9]
            valid_pairs = [(f, s) for f in fast_values for s in slow_values if f < s]
            if not valid_pairs:
                valid_pairs = [(12, 26)]
            fast, slow = self._rng.choice(valid_pairs)
            cfg[f"{prefix}_macd_fast"] = fast
            cfg[f"{prefix}_macd_slow"] = slow
            cfg[f"{prefix}_macd_signal"] = self._rng.choice(signal_values)
            cfg[f"{prefix}_macd_output"] = self._pick(["macd", "signal", "hist"], ["macd"])

    def _sample_rule_slot(self, prefix: str, cfg: dict[str, Any]) -> None:
        active = self._rng.choice([True, False])
        indicator = self._rng.choice([ind.name for ind in self._rule_indicators])
        cfg[f"{prefix}_active"] = active
        cfg[f"{prefix}_archetype"] = self._rng.choice(["comparison", "crossover", "band_test", "pattern", "stat_test", "derivative"])
        cfg[f"{prefix}_indicator"] = indicator
        cfg[f"{prefix}_operator"] = self._rng.choice([">", "<", "crosses_above", "crosses_below"])
        cfg[f"{prefix}_threshold"] = self._rng.uniform(0.0, 100.0)
        cfg[f"{prefix}_comparand"] = self._rng.choice(["price", cfg[f"{prefix}_threshold"]])
        cfg[f"{prefix}_lookback"] = self._rng.randint(5, 50)
        self._sample_rule_params(prefix, indicator, cfg)

    def sample_configuration(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "n_entry_rules": self.n_entry_rules,
            "n_exit_rules": self.n_exit_rules,
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

        for i in range(self.n_entry_rules):
            self._sample_rule_slot(f"entry_{i}", cfg)

        for i in range(self.n_exit_rules):
            self._sample_rule_slot(f"exit_{i}", cfg)

        return cfg


def build_config_space(n_entry_rules: int, n_exit_rules: int, registry: list[IndicatorDef], seed: int | None = None):
    # If ConfigSpace is present, users can replace this with a strict conditional space.
    return SimpleConfigSpace(n_entry_rules=n_entry_rules, n_exit_rules=n_exit_rules, registry=registry, seed=seed)


def sample_and_validate(cs: SimpleConfigSpace, n_samples: int = 10) -> list[dict[str, Any]]:
    return [cs.sample_configuration() for _ in range(n_samples)]

