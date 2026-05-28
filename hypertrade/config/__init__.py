"""Configuration models, defaults, and UI catalog for HyperTrade."""

from hypertrade.config.schemas import (
    CandidateSelectionPolicy,
    ConstraintRule,
    FilterSearchSpace,
    ObjectiveMetric,
    ObjectiveProfile,
    OptimizationRunConfig,
)

OBJECTIVE_CATALOG = {
    "total_pnl": {
        "label": "Total Return %",
        "direction": "maximize",
        "unit": "%",
        "description": "Sum of trade returns in percent.",
    },
    "max_drawdown": {
        "label": "Max Drawdown %",
        "direction": "minimize",
        "unit": "%",
        "description": "Max drawdown on cumulative trade-return curve.",
    },
    "profit_factor": {
        "label": "Profit Factor",
        "direction": "maximize",
        "unit": "ratio",
        "description": "Gross profit divided by gross loss.",
    },
    "avg_hold_minutes": {
        "label": "Average Hold Minutes",
        "direction": "minimize",
        "unit": "minutes",
        "description": "Average trade hold time measured in minutes.",
    },
    "trades": {
        "label": "Trade Count",
        "direction": "maximize",
        "unit": "count",
        "description": "Number of accepted trades.",
    },
    "calmar": {
        "label": "Calmar",
        "direction": "maximize",
        "unit": "ratio",
        "description": "Total return divided by max drawdown.",
    },
    "cvar_5": {
        "label": "CVaR 5%",
        "direction": "minimize",
        "unit": "ratio",
        "description": "Average loss in the 5% left tail of trade returns.",
    },
}

DEFAULT_OBJECTIVE_PROFILE = ObjectiveProfile(
    name="alpha_pareto_default",
    objectives=[
        ObjectiveMetric(metric="total_pnl", direction="maximize"),
        ObjectiveMetric(metric="max_drawdown", direction="minimize"),
        ObjectiveMetric(metric="profit_factor", direction="maximize"),
        ObjectiveMetric(metric="avg_hold_minutes", direction="minimize"),
    ],
    constraints=[
        ConstraintRule(metric="trades", min_value=20.0),
        ConstraintRule(metric="profit_factor", min_value=1.05),
    ],
    candidate_policy=CandidateSelectionPolicy(mode="utopia_distance"),
)

DEFAULT_FILTER_SEARCH_SPACE = FilterSearchSpace()

__all__ = [
    "CandidateSelectionPolicy",
    "ConstraintRule",
    "DEFAULT_FILTER_SEARCH_SPACE",
    "DEFAULT_OBJECTIVE_PROFILE",
    "FilterSearchSpace",
    "OBJECTIVE_CATALOG",
    "ObjectiveMetric",
    "ObjectiveProfile",
    "OptimizationRunConfig",
]
