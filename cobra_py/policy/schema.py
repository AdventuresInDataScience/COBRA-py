from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleConfig:
    archetype: str
    indicator: str
    params: tuple
    output: str
    operator: str
    comparand: str | float
    indicator2: str | None = None
    params2: tuple | None = None
    output2: str | None = None
    lookback: int | None = None
    band_side: str | None = None
    group_id: int | None = None


@dataclass(frozen=True)
class SLConfig:
    sl_type: str
    params: tuple


@dataclass(frozen=True)
class TPConfig:
    tp_type: str
    params: tuple


@dataclass(frozen=True)
class Policy:
    entry_rules: tuple[RuleConfig, ...]
    exit_rules: tuple[RuleConfig, ...]
    sl_config: SLConfig
    tp_config: TPConfig
    n_active_entry: int
    n_active_exit: int
    entry_logic: str = "and"
    exit_logic: str = "or"
