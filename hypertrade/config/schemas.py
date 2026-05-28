from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json

CANDIDATE_POLICY_MODES = {
    "utopia_distance",
    "max_total_pnl",
    "max_profit_factor",
    "min_max_drawdown",
    "min_avg_hold_minutes",
}


@dataclass
class ObjectiveMetric:
    metric: str
    direction: str

    def normalized_direction(self) -> str:
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError(f"Unsupported objective direction: {self.direction}")
        return self.direction


@dataclass
class ConstraintRule:
    metric: str
    min_value: float | None = None
    max_value: float | None = None


@dataclass
class CandidateSelectionPolicy:
    mode: str = "utopia_distance"

    def normalized_mode(self) -> str:
        if self.mode not in CANDIDATE_POLICY_MODES:
            raise ValueError(f"Unsupported candidate policy mode: {self.mode}")
        return self.mode


@dataclass
class ObjectiveProfile:
    name: str
    objectives: list[ObjectiveMetric]
    constraints: list[ConstraintRule] = field(default_factory=list)
    candidate_policy: CandidateSelectionPolicy = field(default_factory=CandidateSelectionPolicy)

    def directions(self) -> list[str]:
        return [objective.normalized_direction() for objective in self.objectives]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ObjectiveProfile":
        objectives = [ObjectiveMetric(**row) for row in payload.get("objectives", [])]
        if not objectives:
            raise ValueError("Objective profile must define at least one objective")
        constraints = [ConstraintRule(**row) for row in payload.get("constraints", [])]
        candidate_policy = CandidateSelectionPolicy(**payload.get("candidate_policy", {}))
        candidate_policy.normalized_mode()
        return cls(
            name=payload["name"],
            objectives=objectives,
            constraints=constraints,
            candidate_policy=candidate_policy,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ObjectiveProfile":
        raw = json.loads(Path(path).read_text())
        return cls.from_dict(raw)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


@dataclass
class FilterSearchSpace:
    sl_min: float = 2.0
    sl_max: float = 7.0
    sl_step: float = 0.5
    tp_min: float = 3.0
    tp_max: float = 15.0
    tp_step: float = 0.5

    atr_use: bool = False
    atr_period_min: int = 10
    atr_period_max: int = 28
    atr_period_step: int = 2
    atr_sl_min: float = 0.75
    atr_sl_max: float = 3.0
    atr_sl_step: float = 0.25
    atr_tp_min: float = 0.5
    atr_tp_max: float = 3.0
    atr_tp_step: float = 0.25

    donchian_use: bool = False
    psar_use: bool = False
    ts_use: bool = False
    ema_use: bool = False
    rsi_use: bool = False
    adx_use: bool = False
    macd_use: bool = False
    bb_use: bool = False
    vwap_use: bool = False
    volume_use: bool = False

    delay_open_min: int = 0
    delay_open_max: int = 600
    delay_open_step: int = 30
    holding_minutes_min: int = 60 * 24
    holding_minutes_max: int = 60 * 24 * 4
    holding_minutes_step: int = 60 * 3
    confirm_bars_min: int = 1
    confirm_bars_max: int = 3
    confirm_bars_step: int = 1
    time_gate_use: bool = False
    accept_start_min: int = 0
    accept_start_max: int = 240
    accept_start_step: int = 30
    accept_end_min: int = 120
    accept_end_max: int = 390
    accept_end_step: int = 30
    cooldown_use: bool = False
    cooldown_min: int = 0
    cooldown_max: int = 240
    cooldown_step: int = 30
    max_signals_per_side_min: int = 1
    max_signals_per_side_max: int = 10
    ranking_mode_choices: list[str] = field(default_factory=lambda: ["none", "strength"])

    commission: float = 0.02
    slippage: float = 0.0004
    bar_minutes: int = 15

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FilterSearchSpace":
        known = {field_name: payload[field_name] for field_name in cls.__dataclass_fields__ if field_name in payload}
        return cls(**known)

    @classmethod
    def load(cls, path: str | Path) -> "FilterSearchSpace":
        raw = json.loads(Path(path).read_text())
        return cls.from_dict(raw)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


@dataclass
class OptimizationRunConfig:
    signals_path: str
    n_trials: int = 100
    benchmark_name: str | None = None
    run_label: str | None = None
    objective_profile_path: str | None = None
    study_name: str | None = None
    artifacts_root: str = "artifacts/experiments"
    seed: int = 42
