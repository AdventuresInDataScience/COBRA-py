import numpy as np

from cobra_py.policy.rules import evaluate_rule
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

