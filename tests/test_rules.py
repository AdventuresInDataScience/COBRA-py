import numpy as np

from cobra_py.policy.rules import combine_rules_with_logic, evaluate_rule
from cobra_py.policy.schema import RuleConfig


def test_comparison_rule_output_shape(sample_ohlcv_data, small_cache):
    price = sample_ohlcv_data["close"].to_numpy()
    rule = RuleConfig(
        archetype="comparison",
        indicator="rsi",
        params=(14,),
        output="rsi",
        operator=">",
        comparand=50.0,
    )
    out = evaluate_rule(rule, small_cache, price)
    assert out.dtype == bool
    assert out.shape == price.shape


def test_crossover_triggers_known_cross(sample_ohlcv_data, small_cache):
    price = sample_ohlcv_data["close"].to_numpy()
    rule = RuleConfig(
        archetype="crossover",
        indicator="rsi",
        params=(14,),
        output="rsi",
        operator="crosses_above",
        comparand=50.0,
    )
    out = evaluate_rule(rule, small_cache, price)
    assert not bool(out[0])
    assert np.any(out)


def test_rule_combination_or_is_less_restrictive(sample_ohlcv_data, small_cache):
    price = sample_ohlcv_data["close"].to_numpy()
    r1 = RuleConfig("comparison", "rsi", (14,), "rsi", ">", 70.0)
    r2 = RuleConfig("comparison", "rsi", (14,), "rsi", "<", 30.0)
    rules = (r1, r2)

    sig_and = combine_rules_with_logic(rules, small_cache, price, logic="and")
    sig_or = combine_rules_with_logic(rules, small_cache, price, logic="or")

    assert sig_or.sum() >= sig_and.sum()


def test_rule_combination_dnf_uses_and_blocks_or_connected(sample_ohlcv_data, small_cache):
    price = sample_ohlcv_data["close"].to_numpy()
    # Group 0: loose pair; Group 1: loose pair. DNF should trigger if either group holds.
    rules = (
        RuleConfig("comparison", "rsi", (14,), "rsi", ">", 45.0, group_id=0),
        RuleConfig("comparison", "rsi", (14,), "rsi", "<", 80.0, group_id=0),
        RuleConfig("comparison", "rsi", (14,), "rsi", ">", 20.0, group_id=1),
        RuleConfig("comparison", "rsi", (14,), "rsi", "<", 55.0, group_id=1),
    )

    sig_dnf = combine_rules_with_logic(rules, small_cache, price, logic="dnf")
    sig_and = combine_rules_with_logic(rules, small_cache, price, logic="and")

    assert sig_dnf.sum() >= sig_and.sum()


def test_comparison_rule_supports_equality_operator(sample_ohlcv_data, small_cache):
    price = sample_ohlcv_data["close"].to_numpy()
    rsi = small_cache.get("rsi", (14,), "rsi")
    assert rsi is not None

    finite_idx = int(np.where(np.isfinite(rsi))[0][0])
    eq_value = float(rsi[finite_idx])

    rule = RuleConfig(
        archetype="comparison",
        indicator="rsi",
        params=(14,),
        output="rsi",
        operator="==",
        comparand=eq_value,
    )
    out = evaluate_rule(rule, small_cache, price)

    assert bool(out[finite_idx])

