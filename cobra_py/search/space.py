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


def _rule_slot_index(prefix: str) -> int | None:
    try:
        return int(str(prefix).split("_", 2)[1])
    except (IndexError, ValueError):
        return None


def _active_choices_for_prefix(prefix: str) -> list[bool]:
    slot_idx = _rule_slot_index(prefix)
    if str(prefix).startswith("entry_") and slot_idx is not None:
        if slot_idx == 0:
            return [True]
        if slot_idx in {1, 2}:
            return [True, True, True, False]
    return [False, True]


def _active_choices_and_weights_for_configspace(prefix: str) -> tuple[list[bool], list[float]]:
    slot_idx = _rule_slot_index(prefix)
    if str(prefix).startswith("entry_") and slot_idx is not None:
        if slot_idx == 0:
            return [True], [1.0]
        if slot_idx in {1, 2}:
            return [True, False], [0.75, 0.25]
    return [False, True], [0.5, 0.5]


def _supports_band_from_outputs(outputs: list[str]) -> bool:
    outs = set(outputs)
    return "upper" in outs and "lower" in outs


def _archetypes_for_indicator_name(indicator_name: str, registry_by_name: dict[str, IndicatorDef]) -> list[str]:
    ind = registry_by_name.get(str(indicator_name))
    if ind is None:
        return [x for x in _RULE_ARCHETYPES if x != "band_test"]
    if _supports_band_from_outputs(list(ind.outputs)):
        return list(_RULE_ARCHETYPES)
    return [x for x in _RULE_ARCHETYPES if x != "band_test"]


def _operators_for_archetype_name(archetype: str) -> list[str]:
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


def _threshold_capable_for_indicator(indicator_name: str, archetype: str) -> bool:
    if str(archetype) in {"band_test", "pattern"}:
        return False
    return str(indicator_name) in {"rsi", "stoch", "cci", "roc", "adx"}


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
        self._conditional_cs_cache = None
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
            add_cat(f"{prefix}_active", _active_choices_for_prefix(prefix))
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
        by_name = {ind.name: ind for ind in self._rule_indicators}
        return _archetypes_for_indicator_name(indicator_name, by_name)

    def _operators_for_archetype(self, archetype: str) -> list[str]:
        return _operators_for_archetype_name(archetype)

    def _threshold_capable(self, indicator_name: str, archetype: str) -> bool:
        return _threshold_capable_for_indicator(indicator_name, archetype)

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

        active_choices = _active_choices_for_prefix(prefix)
        cfg[f"{prefix}_active"] = self._pick_rng(rng, active_choices, active_choices)
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
        archetypes = self._archetypes_for_indicator(indicator_name)
        archetype = trial.suggest_categorical(f"{prefix}_archetype|{indicator_name}", archetypes)
        operators = self._operators_for_archetype(archetype)
        operator = trial.suggest_categorical(f"{prefix}_operator|{indicator_name}|{archetype}", operators)

        cfg[f"{prefix}_active"] = trial.suggest_categorical(f"{prefix}_active", _active_choices_for_prefix(prefix))
        cfg[f"{prefix}_indicator"] = indicator_name
        cfg[f"{prefix}_archetype"] = archetype
        cfg[f"{prefix}_operator"] = operator
        cfg[f"{prefix}_lookback"] = trial.suggest_int(f"{prefix}_lookback", 5, 100)
        cfg[f"{prefix}_group_id"] = trial.suggest_int(f"{prefix}_group_id", 0, 2)
        cfg[f"{prefix}_band_side"] = trial.suggest_categorical(f"{prefix}_band_side", ["upper", "middle", "lower"])

        if self._threshold_capable(indicator_name, archetype):
            cfg[f"{prefix}_comparand_mode"] = trial.suggest_categorical(
                f"{prefix}_comparand_mode|{indicator_name}|{archetype}",
                ["price", "threshold"],
            )
        else:
            cfg[f"{prefix}_comparand_mode"] = "price"
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
            if str(prefix).startswith("entry_") and _rule_slot_index(prefix) == 0:
                cfg[f"{prefix}_active"] = True

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

        kwargs: dict[str, Any] = {
            "entry_logic": ng.p.Choice(_LOGIC_OPTIONS),
            "exit_logic": ng.p.Choice(_LOGIC_OPTIONS),
            "sl_type": ng.p.Choice(["pct", "atr_mult", "swing_low", "bb_lower", "trailing_atr"]),
            "tp_type": ng.p.Choice(["pct", "atr_mult", "risk_reward", "swing_high", "bb_upper"]),
            "sl_pct": ng.p.Choice([x for x in _RATIO_STEP_VALUES if x <= 0.1]),
            "sl_atr_mult": ng.p.Choice(_SL_ATR_MULT_VALUES),
            "sl_atr_period": ng.p.Choice([7, 10, 14, 20]),
            "sl_swing_lookback": ng.p.Scalar(lower=5, upper=50).set_integer_casting(),
            "tp_pct": ng.p.Choice(_RATIO_STEP_VALUES),
            "tp_atr_mult": ng.p.Choice(_TP_ATR_MULT_VALUES),
            "tp_atr_period": ng.p.Choice([7, 10, 14, 20]),
            "tp_rr": ng.p.Choice(_TP_RR_VALUES),
            "tp_swing_lookback": ng.p.Scalar(lower=5, upper=100).set_integer_casting(),
            "leverage": ng.p.Choice([float(x) for x in self._leverage_range]),
            "borrow_cost_rate": ng.p.Choice([float(x) for x in self._borrow_cost_rate_range]),
        }

        for prefix in self._rule_prefixes():
            kwargs[f"{prefix}_active"] = ng.p.Choice(_active_choices_for_prefix(prefix))

            indicator_branches = []
            for ind in self._rule_indicators:
                branch_kwargs: dict[str, Any] = {
                    "indicator": ind.name,
                    "archetype": ng.p.Choice(self._archetypes_for_indicator(ind.name)),
                    "operator": ng.p.Choice(_RULE_OPERATORS),
                    "comparand_mode": ng.p.Choice(["price", "threshold"]),
                    "threshold": ng.p.Choice(_RSI_THRESHOLDS),
                    "lookback": ng.p.Scalar(lower=5, upper=100).set_integer_casting(),
                    "group_id": ng.p.Scalar(lower=0, upper=2).set_integer_casting(),
                    "band_side": ng.p.Choice(["upper", "middle", "lower"]),
                }

                for p_name, p_values in ind.param_grid.items():
                    branch_kwargs[f"{ind.name}_{p_name}"] = ng.p.Choice(list(p_values))
                branch_kwargs[f"{ind.name}_output"] = ng.p.Choice(_indicator_output_choices(ind))

                indicator_branches.append(ng.p.Instrumentation(**branch_kwargs))

            kwargs[f"{prefix}_rule"] = ng.p.Choice(indicator_branches)

        return ng.p.Instrumentation(**kwargs)

    def decode_nevergrad_candidate(self, candidate) -> dict[str, Any]:
        raw = dict(candidate.kwargs)
        cfg: dict[str, Any] = {"n_entry_rules": self.n_entry_rules, "n_exit_rules": self.n_exit_rules}

        scalar_keys = [
            "entry_logic",
            "exit_logic",
            "sl_type",
            "tp_type",
            "sl_pct",
            "sl_atr_mult",
            "sl_atr_period",
            "sl_swing_lookback",
            "tp_pct",
            "tp_atr_mult",
            "tp_atr_period",
            "tp_rr",
            "tp_swing_lookback",
            "leverage",
            "borrow_cost_rate",
        ]
        for key in scalar_keys:
            if key in raw:
                cfg[key] = raw[key]

        for prefix in self._rule_prefixes():
            cfg[f"{prefix}_active"] = bool(raw.get(f"{prefix}_active", _active_choices_for_prefix(prefix)[0]))
            payload = raw.get(f"{prefix}_rule")
            if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[1], dict):
                branch = dict(payload[1])
            elif isinstance(payload, dict):
                branch = dict(payload)
            elif hasattr(payload, "kwargs"):
                branch = dict(getattr(payload, "kwargs"))
            else:
                branch = {}

            indicator_name = str(branch.get("indicator", self._indicator_names[0]))
            cfg[f"{prefix}_indicator"] = indicator_name
            cfg[f"{prefix}_archetype"] = str(branch.get("archetype", self._archetypes_for_indicator(indicator_name)[0]))
            cfg[f"{prefix}_operator"] = str(branch.get("operator", self._operators_for_archetype(cfg[f"{prefix}_archetype"])[0]))
            cfg[f"{prefix}_comparand_mode"] = str(branch.get("comparand_mode", "price"))
            cfg[f"{prefix}_threshold"] = float(branch.get("threshold", 50.0))
            cfg[f"{prefix}_lookback"] = int(branch.get("lookback", 20))
            cfg[f"{prefix}_group_id"] = int(branch.get("group_id", 0))
            cfg[f"{prefix}_band_side"] = str(branch.get("band_side", "middle"))

            ind = self._indicator(indicator_name)
            if ind is None:
                continue
            for p_name, p_values in ind.param_grid.items():
                cfg[f"{prefix}_{indicator_name}_{p_name}"] = branch.get(f"{indicator_name}_{p_name}", list(p_values)[0])
            cfg[f"{prefix}_{indicator_name}_output"] = branch.get(
                f"{indicator_name}_output",
                _indicator_output_choices(ind)[0],
            )

        return self._finalize_config(cfg)

    def get_dehb_configspace(self):
        if self._conditional_cs_cache is not None:
            return self._conditional_cs_cache
        try:
            self._conditional_cs_cache = build_configspace_conditional(
                n_entry_rules=self.n_entry_rules,
                n_exit_rules=self.n_exit_rules,
                registry=self._rule_indicators,
                seed=self.seed,
                backtest_config=self.backtest_config,
            )
        except ImportError:
            self._conditional_cs_cache = None
        return self._conditional_cs_cache

    def conditional_config_to_config(self, raw_config: Any) -> dict[str, Any]:
        if hasattr(raw_config, "get_dictionary"):
            raw = dict(raw_config.get_dictionary())
        elif isinstance(raw_config, dict):
            raw = dict(raw_config)
        else:
            try:
                raw = dict(raw_config)
            except Exception:
                raw = {}

        cfg: dict[str, Any] = {
            "n_entry_rules": self.n_entry_rules,
            "n_exit_rules": self.n_exit_rules,
            "entry_logic": raw.get("entry_logic", _LOGIC_OPTIONS[0]),
            "exit_logic": raw.get("exit_logic", _LOGIC_OPTIONS[0]),
            "sl_type": raw.get("sl_type", "pct"),
            "tp_type": raw.get("tp_type", "pct"),
            "sl_pct": float(raw.get("sl_pct", 0.02)),
            "sl_atr_mult": float(raw.get("sl_atr_mult", 2.0)),
            "sl_atr_period": int(raw.get("sl_atr_period", 14)),
            "sl_swing_lookback": int(raw.get("sl_swing_lookback", 20)),
            "tp_pct": float(raw.get("tp_pct", 0.03)),
            "tp_atr_mult": float(raw.get("tp_atr_mult", 3.0)),
            "tp_atr_period": int(raw.get("tp_atr_period", 14)),
            "tp_rr": float(raw.get("tp_rr", 2.0)),
            "tp_swing_lookback": int(raw.get("tp_swing_lookback", 30)),
            "leverage": float(raw.get("leverage", 1.0)),
            "borrow_cost_rate": float(raw.get("borrow_cost_rate", 0.0)),
        }

        for prefix in self._rule_prefixes():
            active_choices = _active_choices_for_prefix(prefix)
            cfg[f"{prefix}_active"] = bool(raw.get(f"{prefix}_active", active_choices[0]))

            indicator_name = str(raw.get(f"{prefix}_indicator", self._indicator_names[0]))
            cfg[f"{prefix}_indicator"] = indicator_name

            archetype_key = f"{prefix}_archetype|{indicator_name}"
            archetype = str(raw.get(archetype_key, self._archetypes_for_indicator(indicator_name)[0]))
            cfg[f"{prefix}_archetype"] = archetype

            operator_key = f"{prefix}_operator|{indicator_name}|{archetype}"
            cfg[f"{prefix}_operator"] = str(raw.get(operator_key, self._operators_for_archetype(archetype)[0]))

            cfg[f"{prefix}_lookback"] = int(raw.get(f"{prefix}_lookback", 20))
            cfg[f"{prefix}_group_id"] = int(raw.get(f"{prefix}_group_id", 0))
            cfg[f"{prefix}_band_side"] = str(raw.get(f"{prefix}_band_side", "middle"))

            comparand_mode_key = f"{prefix}_comparand_mode|{indicator_name}|{archetype}"
            if self._threshold_capable(indicator_name, archetype):
                cfg[f"{prefix}_comparand_mode"] = str(raw.get(comparand_mode_key, "price"))
                threshold_key = f"{prefix}_threshold|{indicator_name}|{archetype}"
                cfg[f"{prefix}_threshold"] = float(raw.get(threshold_key, 50.0))
            else:
                cfg[f"{prefix}_comparand_mode"] = "price"
                cfg[f"{prefix}_threshold"] = float(raw.get(f"{prefix}_threshold", 50.0))

            ind = self._indicator(indicator_name)
            if ind is None:
                continue
            for p_name, p_values in ind.param_grid.items():
                cfg[f"{prefix}_{indicator_name}_{p_name}"] = raw.get(
                    f"{prefix}_{indicator_name}_{p_name}",
                    list(p_values)[0],
                )
            cfg[f"{prefix}_{indicator_name}_output"] = raw.get(
                f"{prefix}_{indicator_name}_output",
                _indicator_output_choices(ind)[0],
            )

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


def build_configspace_conditional(
    n_entry_rules: int,
    n_exit_rules: int,
    registry: list[IndicatorDef],
    seed: int | None = None,
    backtest_config: dict[str, Any] | None = None,
):
    try:
        from ConfigSpace import (
            CategoricalHyperparameter,
            ConfigurationSpace,
            Constant,
            UniformIntegerHyperparameter,
        )
        from ConfigSpace.conditions import EqualsCondition
    except ImportError as exc:
        raise ImportError("Conditional ConfigSpace requires ConfigSpace package. Install with `pip install ConfigSpace`.") from exc

    bt = backtest_config or {}
    leverage_range = _expand_numeric_range_spec(bt.get("leverage_range"), [float(bt.get("leverage", 1.0))])
    borrow_cost_rate_range = _expand_numeric_range_spec(bt.get("borrow_cost_rate_range"), [float(bt.get("borrow_cost_rate", 0.0))])

    rule_indicators = _valid_rule_indicators(registry)
    indicator_names = [ind.name for ind in rule_indicators]
    registry_by_name = {ind.name: ind for ind in rule_indicators}

    cs = ConfigurationSpace(seed=seed)
    cs.add_hyperparameter(CategoricalHyperparameter("entry_logic", _LOGIC_OPTIONS))
    cs.add_hyperparameter(CategoricalHyperparameter("exit_logic", _LOGIC_OPTIONS))

    cs.add_hyperparameter(CategoricalHyperparameter("sl_type", ["pct", "atr_mult", "swing_low", "bb_lower", "trailing_atr"]))
    cs.add_hyperparameter(CategoricalHyperparameter("tp_type", ["pct", "atr_mult", "risk_reward", "swing_high", "bb_upper"]))
    cs.add_hyperparameter(CategoricalHyperparameter("sl_pct", [x for x in _RATIO_STEP_VALUES if x <= 0.1]))
    cs.add_hyperparameter(CategoricalHyperparameter("sl_atr_mult", _SL_ATR_MULT_VALUES))
    cs.add_hyperparameter(CategoricalHyperparameter("sl_atr_period", [7, 10, 14, 20]))
    cs.add_hyperparameter(UniformIntegerHyperparameter("sl_swing_lookback", lower=5, upper=50))
    cs.add_hyperparameter(CategoricalHyperparameter("tp_pct", _RATIO_STEP_VALUES))
    cs.add_hyperparameter(CategoricalHyperparameter("tp_atr_mult", _TP_ATR_MULT_VALUES))
    cs.add_hyperparameter(CategoricalHyperparameter("tp_atr_period", [7, 10, 14, 20]))
    cs.add_hyperparameter(CategoricalHyperparameter("tp_rr", _TP_RR_VALUES))
    cs.add_hyperparameter(UniformIntegerHyperparameter("tp_swing_lookback", lower=5, upper=100))
    cs.add_hyperparameter(CategoricalHyperparameter("leverage", [float(x) for x in leverage_range]))
    cs.add_hyperparameter(CategoricalHyperparameter("borrow_cost_rate", [float(x) for x in borrow_cost_rate_range]))

    prefixes = [f"entry_{i}" for i in range(int(n_entry_rules))] + [f"exit_{i}" for i in range(int(n_exit_rules))]
    for prefix in prefixes:
        active_choices, active_weights = _active_choices_and_weights_for_configspace(prefix)
        if len(active_choices) == 1:
            active_hp = Constant(f"{prefix}_active", active_choices[0])
        else:
            active_hp = CategoricalHyperparameter(f"{prefix}_active", active_choices, weights=active_weights)
        cs.add_hyperparameter(active_hp)

        indicator_hp = CategoricalHyperparameter(f"{prefix}_indicator", indicator_names)
        cs.add_hyperparameter(indicator_hp)
        cs.add_condition(EqualsCondition(indicator_hp, active_hp, True))

        lookback_hp = UniformIntegerHyperparameter(f"{prefix}_lookback", lower=5, upper=100)
        group_hp = UniformIntegerHyperparameter(f"{prefix}_group_id", lower=0, upper=2)
        band_side_hp = CategoricalHyperparameter(f"{prefix}_band_side", ["upper", "middle", "lower"])
        cs.add_hyperparameter(lookback_hp)
        cs.add_hyperparameter(group_hp)
        cs.add_hyperparameter(band_side_hp)
        cs.add_condition(EqualsCondition(lookback_hp, active_hp, True))
        cs.add_condition(EqualsCondition(group_hp, active_hp, True))
        cs.add_condition(EqualsCondition(band_side_hp, active_hp, True))

        for ind in rule_indicators:
            archetypes = _archetypes_for_indicator_name(ind.name, registry_by_name)
            archetype_hp = CategoricalHyperparameter(f"{prefix}_archetype|{ind.name}", archetypes)
            cs.add_hyperparameter(archetype_hp)
            cs.add_condition(EqualsCondition(archetype_hp, indicator_hp, ind.name))

            for archetype in archetypes:
                operators = _operators_for_archetype_name(archetype)
                operator_hp = CategoricalHyperparameter(f"{prefix}_operator|{ind.name}|{archetype}", operators)
                cs.add_hyperparameter(operator_hp)
                cs.add_condition(EqualsCondition(operator_hp, archetype_hp, archetype))

                if _threshold_capable_for_indicator(ind.name, archetype):
                    comparand_mode_hp = CategoricalHyperparameter(
                        f"{prefix}_comparand_mode|{ind.name}|{archetype}",
                        ["price", "threshold"],
                    )
                    threshold_hp = CategoricalHyperparameter(
                        f"{prefix}_threshold|{ind.name}|{archetype}",
                        _RSI_THRESHOLDS,
                    )
                    cs.add_hyperparameter(comparand_mode_hp)
                    cs.add_hyperparameter(threshold_hp)
                    cs.add_condition(EqualsCondition(comparand_mode_hp, archetype_hp, archetype))
                    cs.add_condition(EqualsCondition(threshold_hp, comparand_mode_hp, "threshold"))

            for p_name, p_values in ind.param_grid.items():
                hp = CategoricalHyperparameter(f"{prefix}_{ind.name}_{p_name}", list(p_values))
                cs.add_hyperparameter(hp)
                cs.add_condition(EqualsCondition(hp, indicator_hp, ind.name))
            out_hp = CategoricalHyperparameter(f"{prefix}_{ind.name}_output", _indicator_output_choices(ind))
            cs.add_hyperparameter(out_hp)
            cs.add_condition(EqualsCondition(out_hp, indicator_hp, ind.name))

    return cs

