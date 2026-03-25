from __future__ import annotations

import json

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
