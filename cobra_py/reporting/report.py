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
    "car_mdd": ("car_mdd_ratio", "CAR/MDD ratio"),
    "cagr": ("cagr", "CAGR"),
    "sortino": ("sortino_ratio", "Sortino ratio"),
    "ulcer": ("ulcer_index", "Ulcer index"),
    "max_return": ("total_return", "Total return"),
    "max_return_dd_cap": ("total_return", "Total return (DD constrained)"),
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


def _logic_caption(logic: str, section: str) -> str:
    key = str(logic).strip().lower()
    if key == "or":
        return f"{section} conditions (logic: OR, any rule can trigger):"
    if key == "dnf":
        return f"{section} conditions (logic: DNF, OR across group-wise AND clauses):"
    return f"{section} conditions (logic: AND, all rules must be true):"


def _rule_series_name(rule) -> str:
    # Band tests reference a selected band side; other archetypes use the configured output line.
    output = (rule.band_side if rule.archetype == "band_test" else rule.output) or "value"
    return f"{rule.indicator}{rule.params}.{output}"


def _format_comparand(rule) -> str:
    if isinstance(rule.comparand, (int, float)):
        return str(float(rule.comparand))
    if rule.comparand == "price":
        return "price"
    if rule.comparand == "indicator2" and rule.indicator2:
        rhs_output = rule.output2 or "value"
        rhs_params = rule.params2 if rule.params2 is not None else ()
        return f"{rule.indicator2}{rhs_params}.{rhs_output}"
    return str(rule.comparand)


def _format_rule(rule, idx: int, logic: str) -> str:
    lhs = _rule_series_name(rule)
    rhs = _format_comparand(rule)
    group_suffix = ""
    if str(logic).strip().lower() == "dnf":
        group_suffix = f" [group {int(rule.group_id or 0)}]"
    return f"  Rule {idx}:{group_suffix} {lhs} {rule.operator} {rhs}"


def policy_to_human_readable(policy) -> str:
    lines = []
    lines.append(_logic_caption(getattr(policy, "entry_logic", "and"), "Entry"))
    if policy.entry_rules:
        for i, r in enumerate(policy.entry_rules, start=1):
            lines.append(_format_rule(r, i, getattr(policy, "entry_logic", "and")))
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append(_logic_caption(getattr(policy, "exit_logic", "or"), "Exit"))
    if policy.exit_rules:
        for i, r in enumerate(policy.exit_rules, start=1):
            lines.append(_format_rule(r, i, getattr(policy, "exit_logic", "or")))
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
        # Keep JSON standards-compliant for unknown/custom objectives.
        best_metric_value = None

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

