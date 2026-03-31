from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from cobra_py.policy.schema import Policy, RuleConfig, SLConfig, TPConfig
from cobra_py.reporting.report import generate_report
from cobra_py.reporting.report import policy_to_human_readable
from cobra_py.search.types import OptimisationResult


def test_generate_report_uses_json_safe_metric_fallback(tmp_path):
    policy = Policy(
        entry_rules=(),
        exit_rules=(),
        sl_config=SLConfig(sl_type="pct", params=(0.02,)),
        tp_config=TPConfig(tp_type="pct", params=(0.03,)),
        n_active_entry=0,
        n_active_exit=0,
    )
    result = OptimisationResult(
        best_policy=policy,
        best_metrics={"total_return": 0.1},
        best_score=1.0,
        objective_name="custom_objective",
        n_evaluations=1,
        optimiser_name="test",
        seed=1,
        runtime_seconds=0.01,
        full_history=[],
    )

    payload = generate_report(result, wf_result=None, output_path=tmp_path)

    assert payload["summary"]["best_metric_value"] is None

    raw = (tmp_path / "result.json").read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert parsed["summary"]["best_metric_value"] is None


def test_policy_to_human_readable_shows_logic_and_output_series_names():
    entry_rule = RuleConfig(
        archetype="band_test",
        indicator="keltner",
        params=(20, 14, 2.0),
        output="upper",
        operator="<",
        comparand="price",
        band_side="upper",
        group_id=1,
    )
    exit_rule = RuleConfig(
        archetype="comparison",
        indicator="macd",
        params=(12, 26, 9),
        output="signal",
        operator=">",
        comparand=0.0,
    )
    policy = Policy(
        entry_rules=(entry_rule,),
        exit_rules=(exit_rule,),
        sl_config=SLConfig(sl_type="pct", params=(0.02,)),
        tp_config=TPConfig(tp_type="pct", params=(0.03,)),
        n_active_entry=1,
        n_active_exit=1,
        entry_logic="or",
        exit_logic="and",
    )

    text = policy_to_human_readable(policy)

    assert "logic: OR" in text
    assert "logic: AND" in text
    assert "keltner(20, 14, 2.0).upper" in text
    assert "macd(12, 26, 9).signal" in text


def test_policy_to_human_readable_reflects_comparison_vs_price_engine_semantics():
    entry_rule = RuleConfig(
        archetype="comparison",
        indicator="sma",
        params=(20,),
        output="ma",
        operator=">",
        comparand="price",
    )
    policy = Policy(
        entry_rules=(entry_rule,),
        exit_rules=(),
        sl_config=SLConfig(sl_type="pct", params=(0.02,)),
        tp_config=TPConfig(tp_type="pct", params=(0.03,)),
        n_active_entry=1,
        n_active_exit=0,
        entry_logic="and",
        exit_logic="or",
    )

    text = policy_to_human_readable(policy)
    assert "[comparison]" in text
    assert "price > sma(20,).ma" in text


def test_policy_to_human_readable_formats_pattern_without_indicator_noise():
    entry_rule = RuleConfig(
        archetype="pattern",
        indicator="rsi",
        params=(14,),
        output="rsi",
        operator="nbar_low",
        comparand="price",
        lookback=17,
    )
    policy = Policy(
        entry_rules=(entry_rule,),
        exit_rules=(),
        sl_config=SLConfig(sl_type="pct", params=(0.02,)),
        tp_config=TPConfig(tp_type="pct", params=(0.03,)),
        n_active_entry=1,
        n_active_exit=0,
        entry_logic="and",
        exit_logic="or",
    )

    text = policy_to_human_readable(policy)
    assert "[pattern]" in text
    assert "price makes 17-bar low" in text


def test_generate_report_serializes_dataclass_with_nested_numpy(tmp_path):
    @dataclass
    class _DummyWf:
        fold_results: list[dict]

    policy = Policy(
        entry_rules=(),
        exit_rules=(),
        sl_config=SLConfig(sl_type="pct", params=(0.02,)),
        tp_config=TPConfig(tp_type="pct", params=(0.03,)),
        n_active_entry=0,
        n_active_exit=0,
    )
    result = OptimisationResult(
        best_policy=policy,
        best_metrics={"total_return": 0.1},
        best_score=1.0,
        objective_name="custom_objective",
        n_evaluations=1,
        optimiser_name="test",
        seed=1,
        runtime_seconds=0.01,
        full_history=[],
    )
    wf = _DummyWf(fold_results=[{"metrics": {"equity_curve": np.array([1.0, 2.0])}}])

    payload = generate_report(result, wf_result=wf, output_path=tmp_path)

    assert payload["walk_forward"]["fold_results"][0]["metrics"]["equity_curve"] == [1.0, 2.0]
    raw = (tmp_path / "result.json").read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert parsed["walk_forward"]["fold_results"][0]["metrics"]["equity_curve"] == [1.0, 2.0]
