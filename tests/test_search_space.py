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

