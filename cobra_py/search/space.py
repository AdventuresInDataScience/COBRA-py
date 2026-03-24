from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from cobra_py.indicators.registry import IndicatorDef

SUPPORTED_RULE_INDICATORS = {"sma", "ema", "wma", "rsi", "macd", "bb", "atr"}

_RATIO_STEP_VALUES = [round(x, 4) for x in [i / 1000.0 for i in range(5, 201, 5)]]
_RSI_THRESHOLDS = [float(x) for x in range(10, 91, 5)]
_SL_ATR_MULT_VALUES = [round(x, 2) for x in [i / 4.0 for i in range(4, 17)]]
_TP_ATR_MULT_VALUES = [round(x, 2) for x in [i / 4.0 for i in range(4, 25)]]
_TP_RR_VALUES = [round(x, 2) for x in [i / 4.0 for i in range(4, 21)]]


def _expand_numeric_range_spec(value: Any, fallback: list[float]) -> list[float]:
    if value is None:
        return fallback
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, dict):
        if "values" in value:
            return [float(x) for x in list(value["values"])]
        if "range" in value:
            seq = list(value["range"])
            if len(seq) != 3:
                raise ValueError("Range spec must be [start, stop, step]")
            start, stop, step = seq
        elif {"start", "stop", "step"}.issubset(value.keys()):
            start = value["start"]
            stop = value["stop"]
            step = value["step"]
        else:
            raise ValueError("Range dict must contain 'values', 'range', or start/stop/step")
        start_f = float(start)
        stop_f = float(stop)
        step_f = float(step)
        if step_f <= 0:
            raise ValueError("Range step must be > 0")
        out: list[float] = []
        cur = start_f
        while cur <= stop_f + 1e-12:
            out.append(round(cur, 10))
            cur += step_f
        return out
    return fallback


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
    backtest_config: dict[str, Any] | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._rule_indicators = _valid_rule_indicators(self.registry)
        bt = self.backtest_config or {}
        self._leverage_range = _expand_numeric_range_spec(bt.get("leverage_range"), [float(bt.get("leverage", 1.0))])
        self._borrow_cost_rate_range = _expand_numeric_range_spec(bt.get("borrow_cost_rate_range"), [float(bt.get("borrow_cost_rate", 0.0))])

    def _pick(self, values: list[Any], fallback: list[Any]) -> Any:
        candidates = values if values else fallback
        return self._rng.choice(candidates)

    def _pick_rng(self, rng: random.Random, values: list[Any], fallback: list[Any]) -> Any:
        candidates = values if values else fallback
        return rng.choice(candidates)

    def _indicator(self, name: str) -> IndicatorDef | None:
        for ind in self._rule_indicators:
            if ind.name == name:
                return ind
        return None

    def _sample_rule_params(self, rng: random.Random, prefix: str, indicator: str, cfg: dict[str, Any]) -> None:
        ind = self._indicator(indicator)
        if ind is None:
            return

        grid = ind.param_grid
        if indicator in {"sma", "ema", "wma"}:
            cfg[f"{prefix}_{indicator}_period"] = int(self._pick_rng(rng, list(grid.get("period", [])), [20]))
        elif indicator == "rsi":
            cfg[f"{prefix}_rsi_period"] = int(self._pick_rng(rng, list(grid.get("period", [])), [14]))
        elif indicator == "atr":
            cfg[f"{prefix}_atr_period"] = int(self._pick_rng(rng, list(grid.get("period", [])), [14]))
        elif indicator == "bb":
            cfg[f"{prefix}_bb_period"] = int(self._pick_rng(rng, list(grid.get("period", [])), [20]))
            cfg[f"{prefix}_bb_std"] = float(self._pick_rng(rng, list(grid.get("std", [])), [2.0]))
            cfg[f"{prefix}_bb_matype"] = str(self._pick_rng(rng, list(grid.get("ma_type", [])), ["sma"]))
            cfg[f"{prefix}_bb_output"] = self._pick_rng(rng, ["upper", "middle", "lower"], ["middle"])
        elif indicator == "macd":
            fast_values = [int(v) for v in list(grid.get("fast", []))] or [12]
            slow_values = [int(v) for v in list(grid.get("slow", []))] or [26]
            signal_values = [int(v) for v in list(grid.get("signal", []))] or [9]
            valid_pairs = [(f, s) for f in fast_values for s in slow_values if f < s]
            if not valid_pairs:
                valid_pairs = [(12, 26)]
            fast, slow = rng.choice(valid_pairs)
            cfg[f"{prefix}_macd_fast"] = fast
            cfg[f"{prefix}_macd_slow"] = slow
            cfg[f"{prefix}_macd_signal"] = rng.choice(signal_values)
            cfg[f"{prefix}_macd_output"] = self._pick_rng(rng, ["macd", "signal", "hist"], ["macd"])

    def _sample_rule_slot(self, rng: random.Random, prefix: str, cfg: dict[str, Any]) -> None:
        active = rng.choice([True, False])
        indicator = rng.choice([ind.name for ind in self._rule_indicators])
        cfg[f"{prefix}_active"] = active

        if indicator == "bb":
            archetype = rng.choice(["comparison", "crossover", "band_test"])
        else:
            archetype = rng.choice(["comparison", "crossover"])

        if archetype == "band_test":
            operator = rng.choice([">", "<"])
            comparand: Any = "price"
            cfg[f"{prefix}_band_side"] = rng.choice(["upper", "middle", "lower"])
        elif indicator == "rsi":
            operator = rng.choice([">", "<", "crosses_above", "crosses_below"])
            comparand = float(self._pick_rng(rng, _RSI_THRESHOLDS, [50.0]))
        else:
            operator = rng.choice([">", "<", "crosses_above", "crosses_below"])
            comparand = "price"

        cfg[f"{prefix}_archetype"] = archetype
        cfg[f"{prefix}_indicator"] = indicator
        cfg[f"{prefix}_operator"] = operator
        cfg[f"{prefix}_threshold"] = float(comparand) if isinstance(comparand, (int, float)) else 50.0
        cfg[f"{prefix}_comparand"] = comparand
        cfg[f"{prefix}_lookback"] = rng.randint(5, 50)
        self._sample_rule_params(rng, prefix, indicator, cfg)

    def _sample_configuration_with_rng(self, rng: random.Random) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "n_entry_rules": self.n_entry_rules,
            "n_exit_rules": self.n_exit_rules,
            "sl_type": rng.choice(["pct", "atr_mult", "swing_low", "bb_lower", "trailing_atr"]),
            "tp_type": rng.choice(["pct", "atr_mult", "risk_reward", "swing_high", "bb_upper"]),
            "sl_pct": float(self._pick_rng(rng, [x for x in _RATIO_STEP_VALUES if x <= 0.1], [0.02])),
            "sl_atr_mult": float(self._pick_rng(rng, _SL_ATR_MULT_VALUES, [2.0])),
            "sl_atr_period": rng.choice([7, 10, 14, 20]),
            "sl_swing_lookback": rng.randint(5, 50),
            "tp_pct": float(self._pick_rng(rng, _RATIO_STEP_VALUES, [0.03])),
            "tp_atr_mult": float(self._pick_rng(rng, _TP_ATR_MULT_VALUES, [3.0])),
            "tp_atr_period": rng.choice([7, 10, 14, 20]),
            "tp_rr": float(self._pick_rng(rng, _TP_RR_VALUES, [2.0])),
            "tp_swing_lookback": rng.randint(5, 100),
            "leverage": float(self._pick_rng(rng, [float(x) for x in self._leverage_range], [1.0])),
            "borrow_cost_rate": float(self._pick_rng(rng, [float(x) for x in self._borrow_cost_rate_range], [0.0])),
        }

        for i in range(self.n_entry_rules):
            self._sample_rule_slot(rng, f"entry_{i}", cfg)

        for i in range(self.n_exit_rules):
            self._sample_rule_slot(rng, f"exit_{i}", cfg)

        return cfg

    def sample_configuration(self) -> dict[str, Any]:
        return self._sample_configuration_with_rng(self._rng)

    def sample_with_seed(self, sample_seed: int) -> dict[str, Any]:
        rng = random.Random(int(sample_seed))
        return self._sample_configuration_with_rng(rng)


def build_config_space(
    n_entry_rules: int,
    n_exit_rules: int,
    registry: list[IndicatorDef],
    seed: int | None = None,
    backtest_config: dict[str, Any] | None = None,
):
    # If ConfigSpace is present, users can replace this with a strict conditional space.
    return SimpleConfigSpace(
        n_entry_rules=n_entry_rules,
        n_exit_rules=n_exit_rules,
        registry=registry,
        backtest_config=backtest_config,
        seed=seed,
    )


def sample_and_validate(cs: SimpleConfigSpace, n_samples: int = 10) -> list[dict[str, Any]]:
    return [cs.sample_configuration() for _ in range(n_samples)]

