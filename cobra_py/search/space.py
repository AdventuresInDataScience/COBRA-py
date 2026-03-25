from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from cobra_py.indicators.registry import IndicatorDef

_RATIO_STEP_VALUES = [round(x, 4) for x in [i / 1000.0 for i in range(5, 201, 5)]]
_RSI_THRESHOLDS = [float(x) for x in range(10, 91, 5)]
_SL_ATR_MULT_VALUES = [round(x, 2) for x in [i / 4.0 for i in range(4, 17)]]
_TP_ATR_MULT_VALUES = [round(x, 2) for x in [i / 4.0 for i in range(4, 25)]]
_TP_RR_VALUES = [round(x, 2) for x in [i / 4.0 for i in range(4, 21)]]
_RULE_ARCHETYPES = ["comparison", "crossover", "band_test", "pattern", "stat_test", "derivative"]
_RULE_OPERATORS = [">", "<", "==", "crosses_above", "crosses_below", "nbar_high", "nbar_low", "consecutive"]
_LOGIC_OPTIONS = ["and", "or", "dnf"]


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
    if not registry:
        raise ValueError("Indicator registry is empty")
    return list(registry)


def _indicator_output_choices(indicator: IndicatorDef) -> list[str]:
    return list(indicator.outputs) if indicator.outputs else ["ma"]


def _encode_cat_index(choices: list[Any], value: Any) -> float:
    try:
        idx = choices.index(value)
    except ValueError:
        idx = 0
    return float(idx)


def _decode_cat_index(choices: list[Any], raw: float) -> Any:
    if not choices:
        return None
    idx = int(round(float(raw)))
    idx = max(0, min(len(choices) - 1, idx))
    return choices[idx]


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
        self._indicator_names = [ind.name for ind in self._rule_indicators]
        bt = self.backtest_config or {}
        self._leverage_range = _expand_numeric_range_spec(bt.get("leverage_range"), [float(bt.get("leverage", 1.0))])
        self._borrow_cost_rate_range = _expand_numeric_range_spec(bt.get("borrow_cost_rate_range"), [float(bt.get("borrow_cost_rate", 0.0))])
        self._dims = self._build_dimensions()

    def _build_dimensions(self) -> list[dict[str, Any]]:
        dims: list[dict[str, Any]] = []

        def add_cat(name: str, choices: list[Any]) -> None:
            dims.append({"name": name, "kind": "cat", "choices": list(choices)})

        def add_int(name: str, low: int, high: int) -> None:
            dims.append({"name": name, "kind": "int", "low": int(low), "high": int(high)})

        def add_float(name: str, low: float, high: float) -> None:
            dims.append({"name": name, "kind": "float", "low": float(low), "high": float(high)})

        add_cat("entry_logic", _LOGIC_OPTIONS)
        add_cat("exit_logic", _LOGIC_OPTIONS)

        add_cat("sl_type", ["pct", "atr_mult", "swing_low", "bb_lower", "trailing_atr"])
        add_cat("tp_type", ["pct", "atr_mult", "risk_reward", "swing_high", "bb_upper"])
        add_cat("sl_pct", [x for x in _RATIO_STEP_VALUES if x <= 0.1])
        add_cat("sl_atr_mult", _SL_ATR_MULT_VALUES)
        add_cat("sl_atr_period", [7, 10, 14, 20])
        add_int("sl_swing_lookback", 5, 50)
        add_cat("tp_pct", _RATIO_STEP_VALUES)
        add_cat("tp_atr_mult", _TP_ATR_MULT_VALUES)
        add_cat("tp_atr_period", [7, 10, 14, 20])
        add_cat("tp_rr", _TP_RR_VALUES)
        add_int("tp_swing_lookback", 5, 100)
        add_cat("leverage", [float(x) for x in self._leverage_range])
        add_cat("borrow_cost_rate", [float(x) for x in self._borrow_cost_rate_range])

        for prefix in [f"entry_{i}" for i in range(self.n_entry_rules)] + [f"exit_{i}" for i in range(self.n_exit_rules)]:
            add_cat(f"{prefix}_active", [False, True])
            add_cat(f"{prefix}_archetype", _RULE_ARCHETYPES)
            add_cat(f"{prefix}_indicator", self._indicator_names)
            add_cat(f"{prefix}_operator", _RULE_OPERATORS)
            add_cat(f"{prefix}_comparand_mode", ["price", "threshold"])
            add_cat(f"{prefix}_threshold", _RSI_THRESHOLDS)
            add_int(f"{prefix}_lookback", 5, 100)
            add_int(f"{prefix}_group_id", 0, 2)
            add_cat(f"{prefix}_band_side", ["upper", "middle", "lower"])

            for ind in self._rule_indicators:
                for p_name, p_values in ind.param_grid.items():
                    add_cat(f"{prefix}_{ind.name}_{p_name}", list(p_values))
                add_cat(f"{prefix}_{ind.name}_output", _indicator_output_choices(ind))

        return dims

    def _rule_prefixes(self) -> list[str]:
        return [f"entry_{i}" for i in range(self.n_entry_rules)] + [f"exit_{i}" for i in range(self.n_exit_rules)]

    def _supports_band(self, indicator_name: str) -> bool:
        ind = self._indicator(indicator_name)
        if ind is None:
            return False
        outs = set(ind.outputs)
        return "upper" in outs and "lower" in outs

    def _archetypes_for_indicator(self, indicator_name: str) -> list[str]:
        if self._supports_band(indicator_name):
            return list(_RULE_ARCHETYPES)
        return [x for x in _RULE_ARCHETYPES if x != "band_test"]

    def _operators_for_archetype(self, archetype: str) -> list[str]:
        key = str(archetype).strip().lower()
        if key == "band_test":
            return [">", "<"]
        if key == "crossover":
            return ["crosses_above", "crosses_below"]
        if key == "pattern":
            return ["nbar_high", "nbar_low", "consecutive"]
        if key in {"comparison", "stat_test", "derivative"}:
            return [">", "<", "=="]
        return [">", "<", "=="]

    def _threshold_capable(self, indicator_name: str, archetype: str) -> bool:
        if str(archetype) in {"band_test", "pattern"}:
            return False
        return str(indicator_name) in {"rsi", "stoch", "cci", "roc", "adx"}

    def _set_rule_indicator_params_from_rng(self, rng: random.Random, prefix: str, indicator_name: str, cfg: dict[str, Any]) -> None:
        ind = self._indicator(indicator_name)
        if ind is None:
            return
        for p_name, p_values in ind.param_grid.items():
            cfg[f"{prefix}_{indicator_name}_{p_name}"] = self._pick_rng(rng, list(p_values), list(p_values))
        cfg[f"{prefix}_{indicator_name}_output"] = self._pick_rng(rng, _indicator_output_choices(ind), _indicator_output_choices(ind))

    def _set_rule_indicator_params_from_optuna(self, trial, prefix: str, indicator_name: str, cfg: dict[str, Any]) -> None:
        ind = self._indicator(indicator_name)
        if ind is None:
            return
        for p_name, p_values in ind.param_grid.items():
            cfg[f"{prefix}_{indicator_name}_{p_name}"] = trial.suggest_categorical(
                f"{prefix}_{indicator_name}_{p_name}",
                list(p_values),
            )
        cfg[f"{prefix}_{indicator_name}_output"] = trial.suggest_categorical(
            f"{prefix}_{indicator_name}_output",
            _indicator_output_choices(ind),
        )

    def _sample_rule_slot_rng(self, rng: random.Random, prefix: str, cfg: dict[str, Any]) -> None:
        indicator_name = self._pick_rng(rng, self._indicator_names, self._indicator_names)
        archetype = self._pick_rng(rng, self._archetypes_for_indicator(indicator_name), self._archetypes_for_indicator(indicator_name))
        operator = self._pick_rng(rng, self._operators_for_archetype(archetype), self._operators_for_archetype(archetype))

        cfg[f"{prefix}_active"] = self._pick_rng(rng, [False, True], [False, True])
        cfg[f"{prefix}_indicator"] = indicator_name
        cfg[f"{prefix}_archetype"] = archetype
        cfg[f"{prefix}_operator"] = operator
        cfg[f"{prefix}_lookback"] = int(rng.randint(5, 100))
        cfg[f"{prefix}_group_id"] = int(rng.randint(0, 2))
        cfg[f"{prefix}_band_side"] = self._pick_rng(rng, ["upper", "middle", "lower"], ["upper", "middle", "lower"])

        if self._threshold_capable(indicator_name, archetype):
            cfg[f"{prefix}_comparand_mode"] = self._pick_rng(rng, ["price", "threshold"], ["price", "threshold"])
        else:
            cfg[f"{prefix}_comparand_mode"] = "price"
        cfg[f"{prefix}_threshold"] = float(self._pick_rng(rng, _RSI_THRESHOLDS, _RSI_THRESHOLDS))

        self._set_rule_indicator_params_from_rng(rng, prefix, indicator_name, cfg)

    def _suggest_rule_slot_optuna(self, trial, prefix: str, cfg: dict[str, Any]) -> None:
        indicator_name = trial.suggest_categorical(f"{prefix}_indicator", self._indicator_names)
        # Optuna requires a fixed categorical domain per parameter name across all trials.
        archetype = trial.suggest_categorical(f"{prefix}_archetype", _RULE_ARCHETYPES)
        operator = trial.suggest_categorical(f"{prefix}_operator", _RULE_OPERATORS)

        cfg[f"{prefix}_active"] = trial.suggest_categorical(f"{prefix}_active", [False, True])
        cfg[f"{prefix}_indicator"] = indicator_name
        cfg[f"{prefix}_archetype"] = archetype
        cfg[f"{prefix}_operator"] = operator
        cfg[f"{prefix}_lookback"] = trial.suggest_int(f"{prefix}_lookback", 5, 100)
        cfg[f"{prefix}_group_id"] = trial.suggest_int(f"{prefix}_group_id", 0, 2)
        cfg[f"{prefix}_band_side"] = trial.suggest_categorical(f"{prefix}_band_side", ["upper", "middle", "lower"])

        cfg[f"{prefix}_comparand_mode"] = trial.suggest_categorical(f"{prefix}_comparand_mode", ["price", "threshold"])
        cfg[f"{prefix}_threshold"] = float(trial.suggest_categorical(f"{prefix}_threshold", _RSI_THRESHOLDS))

        self._set_rule_indicator_params_from_optuna(trial, prefix, indicator_name, cfg)

    @property
    def dimension_count(self) -> int:
        return len(self._dims)

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

    def _sample_configuration_with_rng(self, rng: random.Random) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "n_entry_rules": self.n_entry_rules,
            "n_exit_rules": self.n_exit_rules,
            "entry_logic": self._pick_rng(rng, _LOGIC_OPTIONS, _LOGIC_OPTIONS),
            "exit_logic": self._pick_rng(rng, _LOGIC_OPTIONS, _LOGIC_OPTIONS),
            "sl_type": self._pick_rng(rng, ["pct", "atr_mult", "swing_low", "bb_lower", "trailing_atr"], ["pct"]),
            "tp_type": self._pick_rng(rng, ["pct", "atr_mult", "risk_reward", "swing_high", "bb_upper"], ["pct"]),
            "sl_pct": self._pick_rng(rng, [x for x in _RATIO_STEP_VALUES if x <= 0.1], [0.02]),
            "sl_atr_mult": self._pick_rng(rng, _SL_ATR_MULT_VALUES, [2.0]),
            "sl_atr_period": self._pick_rng(rng, [7, 10, 14, 20], [14]),
            "sl_swing_lookback": int(rng.randint(5, 50)),
            "tp_pct": self._pick_rng(rng, _RATIO_STEP_VALUES, [0.03]),
            "tp_atr_mult": self._pick_rng(rng, _TP_ATR_MULT_VALUES, [3.0]),
            "tp_atr_period": self._pick_rng(rng, [7, 10, 14, 20], [14]),
            "tp_rr": self._pick_rng(rng, _TP_RR_VALUES, [2.0]),
            "tp_swing_lookback": int(rng.randint(5, 100)),
            "leverage": self._pick_rng(rng, [float(x) for x in self._leverage_range], [1.0]),
            "borrow_cost_rate": self._pick_rng(rng, [float(x) for x in self._borrow_cost_rate_range], [0.0]),
        }

        for prefix in self._rule_prefixes():
            self._sample_rule_slot_rng(rng, prefix, cfg)
        return self._finalize_config(cfg)

    def _finalize_config(self, cfg: dict[str, Any]) -> dict[str, Any]:
        for prefix in self._rule_prefixes():
            indicator = str(cfg.get(f"{prefix}_indicator", "rsi"))
            valid_arch = self._archetypes_for_indicator(indicator)
            archetype = str(cfg.get(f"{prefix}_archetype", valid_arch[0]))
            if archetype not in valid_arch:
                archetype = valid_arch[0]
            cfg[f"{prefix}_archetype"] = archetype

            valid_ops = self._operators_for_archetype(archetype)
            op = str(cfg.get(f"{prefix}_operator", valid_ops[0]))
            if op not in valid_ops:
                op = valid_ops[0]
            cfg[f"{prefix}_operator"] = op

            if archetype == "band_test":
                cfg[f"{prefix}_comparand_mode"] = "price"
            elif not self._threshold_capable(indicator, archetype):
                cfg[f"{prefix}_comparand_mode"] = "price"

            mode = str(cfg.get(f"{prefix}_comparand_mode", "price"))
            if mode == "threshold":
                cfg[f"{prefix}_comparand"] = float(cfg.get(f"{prefix}_threshold", 50.0))
            else:
                cfg[f"{prefix}_comparand"] = "price"

            # Enforce indicator-specific structural constraints.
            if indicator == "macd":
                fast = int(cfg.get(f"{prefix}_macd_fast", 12))
                slow = int(cfg.get(f"{prefix}_macd_slow", 26))
                if fast >= slow:
                    slow = fast + 1
                cfg[f"{prefix}_macd_fast"] = fast
                cfg[f"{prefix}_macd_slow"] = slow
        return cfg

    def sample_configuration(self) -> dict[str, Any]:
        return self._sample_configuration_with_rng(self._rng)

    def sample_with_seed(self, sample_seed: int) -> dict[str, Any]:
        rng = random.Random(int(sample_seed))
        return self._sample_configuration_with_rng(rng)

    def suggest_with_optuna(self, trial) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "n_entry_rules": self.n_entry_rules,
            "n_exit_rules": self.n_exit_rules,
            "entry_logic": trial.suggest_categorical("entry_logic", _LOGIC_OPTIONS),
            "exit_logic": trial.suggest_categorical("exit_logic", _LOGIC_OPTIONS),
            "sl_type": trial.suggest_categorical("sl_type", ["pct", "atr_mult", "swing_low", "bb_lower", "trailing_atr"]),
            "tp_type": trial.suggest_categorical("tp_type", ["pct", "atr_mult", "risk_reward", "swing_high", "bb_upper"]),
            "sl_pct": trial.suggest_categorical("sl_pct", [x for x in _RATIO_STEP_VALUES if x <= 0.1]),
            "sl_atr_mult": trial.suggest_categorical("sl_atr_mult", _SL_ATR_MULT_VALUES),
            "sl_atr_period": trial.suggest_categorical("sl_atr_period", [7, 10, 14, 20]),
            "sl_swing_lookback": trial.suggest_int("sl_swing_lookback", 5, 50),
            "tp_pct": trial.suggest_categorical("tp_pct", _RATIO_STEP_VALUES),
            "tp_atr_mult": trial.suggest_categorical("tp_atr_mult", _TP_ATR_MULT_VALUES),
            "tp_atr_period": trial.suggest_categorical("tp_atr_period", [7, 10, 14, 20]),
            "tp_rr": trial.suggest_categorical("tp_rr", _TP_RR_VALUES),
            "tp_swing_lookback": trial.suggest_int("tp_swing_lookback", 5, 100),
            "leverage": trial.suggest_categorical("leverage", [float(x) for x in self._leverage_range]),
            "borrow_cost_rate": trial.suggest_categorical("borrow_cost_rate", [float(x) for x in self._borrow_cost_rate_range]),
        }
        for prefix in self._rule_prefixes():
            self._suggest_rule_slot_optuna(trial, prefix, cfg)
        return self._finalize_config(cfg)

    def get_nevergrad_parametrization(self):
        import nevergrad as ng

        kwargs: dict[str, Any] = {}
        for dim in self._dims:
            name = dim["name"]
            if dim["kind"] == "cat":
                kwargs[name] = ng.p.Choice(dim["choices"])
            elif dim["kind"] == "int":
                kwargs[name] = ng.p.Scalar(lower=int(dim["low"]), upper=int(dim["high"])).set_integer_casting()
            else:
                kwargs[name] = ng.p.Scalar(lower=float(dim["low"]), upper=float(dim["high"]))
        return ng.p.Instrumentation(**kwargs)

    def decode_nevergrad_candidate(self, candidate) -> dict[str, Any]:
        cfg = {"n_entry_rules": self.n_entry_rules, "n_exit_rules": self.n_exit_rules}
        cfg.update(candidate.kwargs)
        return self._finalize_config(cfg)

    def sample_vector(self, rng: random.Random) -> list[float]:
        vec: list[float] = []
        for dim in self._dims:
            if dim["kind"] == "cat":
                vec.append(float(rng.randint(0, len(dim["choices"]) - 1)))
            elif dim["kind"] == "int":
                vec.append(float(rng.randint(int(dim["low"]), int(dim["high"]))))
            else:
                low = float(dim["low"])
                high = float(dim["high"])
                vec.append(float(low + (high - low) * rng.random()))
        return vec

    def vector_to_config(self, vector: Any) -> dict[str, Any]:
        if isinstance(vector, dict):
            for key in ("x", "param", "params", "config"):
                if key in vector:
                    vector = vector[key]
                    break
        raw = list(vector)
        cfg: dict[str, Any] = {"n_entry_rules": self.n_entry_rules, "n_exit_rules": self.n_exit_rules}
        for i, dim in enumerate(self._dims):
            name = dim["name"]
            value = float(raw[i]) if i < len(raw) else 0.0
            if dim["kind"] == "cat":
                cfg[name] = _decode_cat_index(dim["choices"], value)
            elif dim["kind"] == "int":
                iv = int(round(value))
                cfg[name] = max(int(dim["low"]), min(int(dim["high"]), iv))
            else:
                low = float(dim["low"])
                high = float(dim["high"])
                cfg[name] = max(low, min(high, float(value)))
        return self._finalize_config(cfg)

    def config_to_vector(self, config: dict[str, Any]) -> list[float]:
        vec: list[float] = []
        for dim in self._dims:
            name = dim["name"]
            val = config.get(name)
            if dim["kind"] == "cat":
                vec.append(_encode_cat_index(dim["choices"], val))
            elif dim["kind"] == "int":
                vec.append(float(int(val if val is not None else dim["low"])))
            else:
                vec.append(float(val if val is not None else dim["low"]))
        return vec


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

