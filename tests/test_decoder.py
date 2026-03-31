from cobra_py.indicators.precompute import precompute_all
from cobra_py.indicators.registry import DEFAULT_REGISTRY, build_registry_from_config
from cobra_py.policy.decoder import decode_config


def _base_rule_config(indicator_name: str) -> dict:
    return {
        "n_entry_rules": 1,
        "n_exit_rules": 0,
        "entry_logic": "and",
        "exit_logic": "or",
        "entry_0_active": True,
        "entry_0_indicator": indicator_name,
        "entry_0_archetype": "comparison",
        "entry_0_operator": ">",
        "entry_0_comparand_mode": "price",
        "entry_0_threshold": 50.0,
        "entry_0_lookback": 10,
        "entry_0_group_id": 0,
        "entry_0_band_side": "middle",
        "sl_type": "pct",
        "sl_pct": 0.02,
        "tp_type": "pct",
        "tp_pct": 0.03,
    }


def test_decode_config_supports_registry_defined_indicator_params_and_outputs(sample_ohlcv_data):
    include = ["ichimoku", "supertrend", "ad", "cmf", "tsi"]
    reg = build_registry_from_config(DEFAULT_REGISTRY, include=include)
    cache = precompute_all(sample_ohlcv_data.iloc[:600], reg, n_jobs=1)
    by_name = {ind.name: ind for ind in reg}

    for indicator_name in include:
        ind = by_name[indicator_name]
        cfg = _base_rule_config(indicator_name)

        # Use last output to ensure decoder reads indicator-specific output keys.
        if ind.outputs:
            cfg[f"entry_0_{indicator_name}_output"] = ind.outputs[-1]
            cfg["entry_0_band_side"] = ind.outputs[-1]

        for p_name, p_values in ind.param_grid.items():
            cfg[f"entry_0_{indicator_name}_{p_name}"] = p_values[-1]

        policy = decode_config(cfg, cache)

        assert policy is not None, f"Expected valid policy for indicator={indicator_name}"
        assert len(policy.entry_rules) == 1
        assert policy.entry_rules[0].indicator == indicator_name
        if ind.outputs:
            assert policy.entry_rules[0].output == ind.outputs[-1]
