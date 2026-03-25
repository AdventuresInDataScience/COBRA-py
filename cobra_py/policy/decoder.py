from __future__ import annotations

from typing import Any

from cobra_py.indicators.cache import IndicatorCache

from .schema import Policy, RuleConfig, SLConfig, TPConfig


def _indicator_params_output(prefix: str, indicator: str, config: dict[str, Any]) -> tuple[tuple, str]:
    if indicator == "rsi":
        return (int(config.get(f"{prefix}_rsi_period", 14)),), "rsi"
    if indicator in {"sma", "ema", "wma"}:
        return (int(config.get(f"{prefix}_{indicator}_period", 20)),), "ma"
    if indicator == "atr":
        return (int(config.get(f"{prefix}_atr_period", 14)),), "atr"
    if indicator == "bb":
        params = (
            int(config.get(f"{prefix}_bb_period", 20)),
            float(config.get(f"{prefix}_bb_std", 2.0)),
            str(config.get(f"{prefix}_bb_matype", "sma")),
        )
        return params, str(config.get(f"{prefix}_bb_output", "middle"))
    if indicator == "macd":
        params = (
            int(config.get(f"{prefix}_macd_fast", 12)),
            int(config.get(f"{prefix}_macd_slow", 26)),
            int(config.get(f"{prefix}_macd_signal", 9)),
        )
        return params, str(config.get(f"{prefix}_macd_output", "macd"))
    if indicator == "keltner":
        params = (
            int(config.get(f"{prefix}_keltner_ema_period", 20)),
            int(config.get(f"{prefix}_keltner_atr_period", 14)),
            float(config.get(f"{prefix}_keltner_mult", 2.0)),
        )
        return params, str(config.get(f"{prefix}_keltner_output", "upper"))
    if indicator == "donchian":
        params = (int(config.get(f"{prefix}_donchian_period", 20)),)
        return params, str(config.get(f"{prefix}_donchian_output", "upper"))
    if indicator == "adx":
        return (int(config.get(f"{prefix}_adx_period", 14)),), "adx"
    if indicator == "stoch":
        params = (
            int(config.get(f"{prefix}_stoch_k", 14)),
            int(config.get(f"{prefix}_stoch_d", 3)),
            int(config.get(f"{prefix}_stoch_smooth", 3)),
        )
        return params, str(config.get(f"{prefix}_stoch_output", "k"))
    if indicator == "psar":
        params = (
            float(config.get(f"{prefix}_psar_step", 0.02)),
            float(config.get(f"{prefix}_psar_max_step", 0.2)),
        )
        return params, "psar"
    if indicator == "cci":
        return (int(config.get(f"{prefix}_cci_period", 14)),), "cci"
    if indicator == "roc":
        return (int(config.get(f"{prefix}_roc_period", 10)),), "roc"
    if indicator == "obv":
        return (), "obv"
    if indicator == "vwap":
        return (), "vwap"

    params = (int(config.get(f"{prefix}_{indicator}_period", 14)),)
    return params, str(config.get(f"{prefix}_output", "ma"))


def _decode_rule(prefix: str, config: dict[str, Any], cache: IndicatorCache) -> RuleConfig | None:
    indicator = config.get(f"{prefix}_indicator", "rsi")
    archetype = config.get(f"{prefix}_archetype", "comparison")
    operator = config.get(f"{prefix}_operator", ">")

    params, output = _indicator_params_output(prefix, indicator, config)
    band_side = str(config.get(f"{prefix}_band_side", output))

    required_output = band_side if archetype == "band_test" else output
    if cache.get(indicator, params, required_output) is None:
        return None

    threshold = float(config.get(f"{prefix}_threshold", 50.0))
    comparand_mode = str(config.get(f"{prefix}_comparand_mode", "price")).lower()
    if comparand_mode == "threshold":
        comparand: str | float = threshold
    elif comparand_mode == "indicator2":
        comparand = "indicator2"
    else:
        comparand = "price"

    lookback = int(config.get(f"{prefix}_lookback", 20))
    group_id = int(config.get(f"{prefix}_group_id", 0))

    return RuleConfig(
        archetype=archetype,
        indicator=indicator,
        params=params,
        output=output,
        operator=operator,
        comparand=comparand,
        indicator2=config.get(f"{prefix}_indicator2"),
        params2=config.get(f"{prefix}_params2"),
        output2=config.get(f"{prefix}_output2"),
        lookback=lookback,
        band_side=band_side,
        group_id=group_id,
    )


def decode_config(config: dict[str, Any], cache: IndicatorCache) -> Policy | None:
    n_entry_slots = int(config.get("n_entry_rules", 3))
    n_exit_slots = int(config.get("n_exit_rules", 1))

    entry_rules = []
    for i in range(n_entry_slots):
        if not bool(config.get(f"entry_{i}_active", False)):
            continue
        rule = _decode_rule(f"entry_{i}", config, cache)
        if rule is None:
            return None
        entry_rules.append(rule)

    exit_rules = []
    for i in range(n_exit_slots):
        if not bool(config.get(f"exit_{i}_active", False)):
            continue
        rule = _decode_rule(f"exit_{i}", config, cache)
        if rule is None:
            return None
        exit_rules.append(rule)

    sl_type = str(config.get("sl_type", "pct"))
    if sl_type == "pct":
        sl_params = (float(config.get("sl_pct", 0.02)),)
    elif sl_type in {"atr_mult", "trailing_atr"}:
        sl_params = (float(config.get("sl_atr_mult", 2.0)), int(config.get("sl_atr_period", 14)))
    elif sl_type == "swing_low":
        sl_params = (int(config.get("sl_swing_lookback", 20)),)
    else:
        sl_params = (
            int(config.get("sl_bb_period", 20)),
            float(config.get("sl_bb_std", 2.0)),
            str(config.get("sl_bb_matype", "sma")),
        )

    tp_type = str(config.get("tp_type", "pct"))
    if tp_type == "pct":
        tp_params = (float(config.get("tp_pct", 0.03)),)
    elif tp_type == "atr_mult":
        tp_params = (float(config.get("tp_atr_mult", 3.0)), int(config.get("tp_atr_period", 14)))
    elif tp_type == "risk_reward":
        tp_params = (float(config.get("tp_rr", 2.0)),)
    elif tp_type == "swing_high":
        tp_params = (int(config.get("tp_swing_lookback", 30)),)
    else:
        tp_params = (
            int(config.get("tp_bb_period", 20)),
            float(config.get("tp_bb_std", 2.0)),
            str(config.get("tp_bb_matype", "sma")),
        )

    return Policy(
        entry_rules=tuple(entry_rules),
        exit_rules=tuple(exit_rules),
        sl_config=SLConfig(sl_type=sl_type, params=sl_params),
        tp_config=TPConfig(tp_type=tp_type, params=tp_params),
        n_active_entry=len(entry_rules),
        n_active_exit=len(exit_rules),
        entry_logic=str(config.get("entry_logic", "and")).lower(),
        exit_logic=str(config.get("exit_logic", "or")).lower(),
    )

