from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import math
import re

CANDIDATE_POLICY_MODES = {
    "utopia_distance",
    "max_sum_trade_returns_pct",
    "max_profit_factor",
    "min_proxy_drawdown_usd",
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

    def validate(self) -> None:
        from hypertrade.config.defaults import OBJECTIVE_CATALOG
        if not isinstance(self.name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', self.name):
            raise ValueError('Profile name must be 1-64 letters, digits, underscores or hyphens, starting with a letter/digit')
        if not self.objectives:
            raise ValueError('Objective profile must define at least one objective')
        if len({o.metric for o in self.objectives}) != len(self.objectives):
            raise ValueError('Duplicate objective metric')
        for row in [*self.objectives, *self.constraints]:
            if row.metric not in OBJECTIVE_CATALOG:
                raise ValueError(f'Unknown objective/constraint metric: {row.metric}')
        for objective in self.objectives:
            objective.normalized_direction()
        for rule in self.constraints:
            if rule.min_value is None and rule.max_value is None:
                raise ValueError(f'Constraint {rule.metric} needs a minimum or maximum')
            for value in (rule.min_value, rule.max_value):
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                    raise ValueError(f'Constraint {rule.metric} bounds must be finite numbers')
            if rule.min_value is not None and rule.max_value is not None and rule.min_value > rule.max_value:
                raise ValueError(f'Constraint {rule.metric} minimum exceeds maximum')
        self.candidate_policy.normalized_mode()

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
        profile = cls(
            name=payload["name"],
            objectives=objectives,
            constraints=constraints,
            candidate_policy=candidate_policy,
        )
        profile.validate()
        return profile

    @classmethod
    def load(cls, path: str | Path) -> "ObjectiveProfile":
        raw = json.loads(Path(path).read_text())
        return cls.from_dict(raw)

    def save(self, path: str | Path) -> None:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_text(json.dumps(self.to_dict(), indent=2))
        temporary.replace(target)


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

    def __post_init__(self):
        import math
        for key, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f'Non-finite search parameter: {key}')
            if key.endswith('_min'):
                upper = getattr(self, key[:-4]+'_max', value)
                if value < 0 or value > upper:
                    raise ValueError(f'Invalid search range: {key}')
            if key.endswith('_step') and value <= 0:
                raise ValueError(f'Step must be positive: {key}')
        if self.sl_min <= 0 or self.tp_min <= 0 or self.max_signals_per_side_min < 1:
            raise ValueError('SL, TP and side cap must be positive')
        if self.commission < 0 or not 0 <= self.slippage < 1 or self.bar_minutes != 15:
            raise ValueError('Require nonnegative USD/share fee, fractional slippage and 15-minute bars')
        if self.time_gate_use and self.accept_start_max >= self.accept_end_max:
            raise ValueError('Time gate needs an end after every possible start')
        if not self.ranking_mode_choices or set(self.ranking_mode_choices)-{'none','strength'}:
            raise ValueError('Unsupported ranking modes')

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FilterSearchSpace":
        unknown = set(payload)-set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'Unknown search parameters: {sorted(unknown)}')
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
