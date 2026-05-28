from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

IndicatorValue = int | float | bool | str | None
IndicatorSettings = dict[str, IndicatorValue]
IndicatorConfig = dict[str, IndicatorSettings]


DEFAULT_INDICATOR_CONFIG: IndicatorConfig = {
    "ema": {"enabled": False, "sign": None, "fast": None, "slow": None},
    "rsi": {"enabled": False, "sign": None, "level": None, "period": None},
    "volume": {"enabled": False, "sign": None, "ma_period": None, "threshold": None},
    "adx": {"enabled": False, "mode": None, "minimum": None, "period": None},
    "atr": {"enabled": False, "period": None},
    "macd": {"enabled": False, "sign": None, "fast": None, "slow": None, "signal": None},
    "bb": {"enabled": False, "sign": None, "period": None, "std": None},
    "donchian": {"enabled": False, "period": None},
    "vwap": {"enabled": False, "sign": None, "threshold": None},
}


def default_indicator_config() -> IndicatorConfig:
    return {name: dict(settings) for name, settings in DEFAULT_INDICATOR_CONFIG.items()}


def indicator_settings(indicator_config: IndicatorConfig | None, name: str) -> IndicatorSettings:
    base = dict(DEFAULT_INDICATOR_CONFIG.get(name, {}))
    if not isinstance(indicator_config, dict):
        return base
    current = indicator_config.get(name, {})
    if not isinstance(current, dict):
        raise ValueError(f"Indicator config for '{name}' must be a dict, got {type(current).__name__}")
    base.update(current)
    return base


def indicator_enabled(indicator_config: IndicatorConfig | None, name: str) -> bool:
    return bool(indicator_settings(indicator_config, name).get("enabled", False))


def validate_indicator_config(indicator_config: IndicatorConfig) -> None:
    if not isinstance(indicator_config, dict):
        raise ValueError("indicator_config must be a dict")
    for name, template in DEFAULT_INDICATOR_CONFIG.items():
        if name not in indicator_config:
            raise ValueError(f"indicator_config missing '{name}'")
        settings = indicator_config[name]
        if not isinstance(settings, dict):
            raise ValueError(f"indicator_config['{name}'] must be a dict")
        if "enabled" not in settings:
            raise ValueError(f"indicator_config['{name}'] missing 'enabled'")
        unknown = set(settings) - set(template)
        if unknown:
            raise ValueError(f"indicator_config['{name}'] has unknown keys: {sorted(unknown)}")


@dataclass
class StrategyParams:
    sl: float | None = None
    tp: float | None = None
    delay_open: int = 0
    holding_minutes: int = 600
    atr_use: bool = False
    atr_period: int | None = 14
    atr_sl: float | None = 1.0
    atr_tp: float | None = 1.5
    psar_enabled: bool = False
    psar_max: float | None = None
    psar_step: float | None = None
    ts_enabled: bool = False
    ts_dist: float = 2.0
    ts_step: float = 0.5
    confirm_bars: int = 1
    accept_start_minute: int | None = None
    accept_end_minute: int | None = None
    cooldown_minutes: int = 0
    max_signals_per_side: int | None = None
    ranking_mode: str = "none"
    bar_minutes: int = 15
    commission: float = 0.02
    slippage: float = 0.0004
    indicator_config: IndicatorConfig = field(default_factory=default_indicator_config)

    def __post_init__(self) -> None:
        validate_indicator_config(self.indicator_config)


def clone_indicator_config(indicator_config: IndicatorConfig | None) -> IndicatorConfig:
    config = default_indicator_config()
    if not isinstance(indicator_config, dict):
        return config
    for name, settings in indicator_config.items():
        if name not in config or not isinstance(settings, dict):
            continue
        config[name].update(settings)
    validate_indicator_config(config)
    return config
