from __future__ import annotations

import json

from cobra_py.policy.schema import Policy, SLConfig, TPConfig
from cobra_py.reporting.report import generate_report
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
