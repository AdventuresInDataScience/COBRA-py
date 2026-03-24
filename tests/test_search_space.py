from cobra_py.indicators.registry import DEFAULT_REGISTRY, build_registry_from_config
from cobra_py.search.space import build_config_space, sample_and_validate


def test_space_samples_without_error():
    cs = build_config_space(3, 1, DEFAULT_REGISTRY, seed=42)
    samples = sample_and_validate(cs, n_samples=100)
    assert len(samples) == 100
    assert all(isinstance(s, dict) for s in samples)


def test_seed_determinism():
    cs1 = build_config_space(3, 1, DEFAULT_REGISTRY, seed=123)
    cs2 = build_config_space(3, 1, DEFAULT_REGISTRY, seed=123)
    seq1 = [cs1.sample_configuration() for _ in range(5)]
    seq2 = [cs2.sample_configuration() for _ in range(5)]
    assert seq1 == seq2


def test_registry_filter_and_param_override():
    reg = build_registry_from_config(
        DEFAULT_REGISTRY,
        include=["rsi", "bb"],
        param_ranges={"rsi": {"period": [14, 21]}},
    )
    names = [x.name for x in reg]
    assert names == ["rsi", "bb"]
    rsi = next(x for x in reg if x.name == "rsi")
    assert rsi.param_grid["period"] == [14, 21]


def test_registry_param_range_shorthand():
    reg = build_registry_from_config(
        DEFAULT_REGISTRY,
        include=["bb"],
        param_ranges={
            "bb": {
                "period": {"start": 10, "stop": 20, "step": 5},
                "std": {"range": [1.5, 2.5, 0.5]},
                "ma_type": ["sma", "ema"],
            }
        },
    )
    bb = reg[0]
    assert bb.param_grid["period"] == [10, 15, 20]
    assert bb.param_grid["std"] == [1.5, 2.0, 2.5]


def test_space_uses_selected_indicators_and_rule_counts():
    reg = build_registry_from_config(DEFAULT_REGISTRY, include=["rsi"])
    cs = build_config_space(2, 2, reg, seed=42)
    sample = cs.sample_configuration()

    assert sample["n_entry_rules"] == 2
    assert sample["n_exit_rules"] == 2
    assert sample["entry_0_indicator"] == "rsi"
    assert sample["entry_1_indicator"] == "rsi"
    assert sample["exit_0_indicator"] == "rsi"
    assert sample["exit_1_indicator"] == "rsi"


def test_space_samples_leverage_from_configured_range():
    reg = build_registry_from_config(DEFAULT_REGISTRY, include=["rsi"])
    cs = build_config_space(
        1,
        1,
        reg,
        seed=42,
        backtest_config={"leverage_range": {"start": 0.5, "stop": 1.5, "step": 0.5}},
    )
    values = {cs.sample_configuration()["leverage"] for _ in range(20)}
    assert values.issubset({0.5, 1.0, 1.5})


def test_space_samples_discrete_stop_and_target_percentages():
    reg = build_registry_from_config(DEFAULT_REGISTRY, include=["rsi"])
    cs = build_config_space(1, 1, reg, seed=7)
    sample = cs.sample_configuration()

    assert abs(sample["sl_pct"] * 1000 - round(sample["sl_pct"] * 1000)) < 1e-9
    assert abs(sample["tp_pct"] * 1000 - round(sample["tp_pct"] * 1000)) < 1e-9


def test_space_uses_indicator_aware_comparand_logic():
    reg = build_registry_from_config(DEFAULT_REGISTRY, include=["sma"])
    cs = build_config_space(1, 1, reg, seed=42)
    sample = cs.sample_configuration()

    assert sample["entry_0_indicator"] == "sma"
    assert sample["entry_0_comparand"] == "price"

