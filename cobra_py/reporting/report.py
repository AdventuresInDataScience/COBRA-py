from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cobra_py.search.types import OptimisationResult
from cobra_py.validation.walk_forward import WalkForwardResult


OBJECTIVE_TO_METRIC = {
    "sharpe": ("sharpe_ratio", "Sharpe ratio"),
    "calmar": ("calmar_ratio", "Calmar ratio"),
    "sortino": ("sortino_ratio", "Sortino ratio"),
    "ulcer": ("ulcer_index", "Ulcer index"),
    "max_return": ("total_return", "Total return"),
}


def _as_serialisable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _as_serialisable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_serialisable(v) for v in value]
    return value


def policy_to_human_readable(policy) -> str:
    lines = []
    lines.append("Entry conditions (ALL must be true simultaneously):")
    if policy.entry_rules:
        for i, r in enumerate(policy.entry_rules, start=1):
            lines.append(f"  Rule {i}: {r.indicator}{r.params} {r.operator} {r.comparand}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("Exit conditions:")
    if policy.exit_rules:
        for i, r in enumerate(policy.exit_rules, start=1):
            lines.append(f"  Rule {i}: {r.indicator}{r.params} {r.operator} {r.comparand}")
    else:
        lines.append("  (none - exits driven by stop-loss and take-profit only)")

    lines.append("")
    lines.append(f"Stop-loss: {policy.sl_config.sl_type} {policy.sl_config.params}")
    lines.append(f"Take-profit: {policy.tp_config.tp_type} {policy.tp_config.params}")
    return "\n".join(lines)


def generate_report(result: OptimisationResult, wf_result: WalkForwardResult | None, output_path: str | Path) -> dict[str, Any]:
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    objective_name = str(result.objective_name)
    metric_key, metric_name = OBJECTIVE_TO_METRIC.get(objective_name, (None, objective_name))
    if metric_key and metric_key in result.best_metrics:
        best_metric_value = float(result.best_metrics[metric_key])
    else:
        # Fallback: keep a useful human-readable value even for unsupported custom objectives.
        best_metric_value = float("nan")

    payload = {
        "summary": {
            "optimiser_name": result.optimiser_name,
            "objective": objective_name,
            "seed": result.seed,
            "best_score": result.best_score,
            "best_metric_name": metric_name,
            "best_metric_value": best_metric_value,
            "n_evaluations": result.n_evaluations,
            "runtime_seconds": result.runtime_seconds,
        },
        "best_metrics": _as_serialisable(result.best_metrics),
        "policy_human_readable": policy_to_human_readable(result.best_policy),
        "best_policy": _as_serialisable(result.best_policy),
        "walk_forward": _as_serialisable(wf_result) if wf_result else None,
    }

    json_path = out_dir / "result.json"
    yaml_path = out_dir / "result.yaml"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload

