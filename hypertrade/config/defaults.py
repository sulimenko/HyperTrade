"""Explicit metric units and default Pareto objectives."""
from hypertrade.config.schemas import CandidateSelectionPolicy, ConstraintRule, FilterSearchSpace, ObjectiveMetric, ObjectiveProfile, OptimizationRunConfig

OBJECTIVE_CATALOG = {
    'sum_trade_returns_pct': dict(label='Sum of trade returns (research)', direction='maximize', unit='percentage points'),
    'proxy_drawdown_usd': dict(label='Sequential proxy drawdown (research)', direction='minimize', unit='normalized USD'),
    'profit_factor': dict(label='Profit factor (research)', direction='maximize', unit='ratio'),
    'avg_hold_minutes': dict(label='Average hold', direction='minimize', unit='minutes'),
    'trades': dict(label='Research trade count', direction='maximize', unit='count'),
    'cvar_loss_5_pct': dict(label='Left-tail loss CVaR 5%', direction='minimize', unit='percent'),
    'portfolio_return_pct': dict(label='Portfolio return', direction='maximize', unit='percent of initial equity'),
    'portfolio_max_drawdown_pct': dict(label='Portfolio drawdown', direction='minimize', unit='percent of peak equity'),
}
DEFAULT_OBJECTIVE_PROFILE = ObjectiveProfile(
    name='research_pareto',
    objectives=[ObjectiveMetric('sum_trade_returns_pct', 'maximize'), ObjectiveMetric('proxy_drawdown_usd', 'minimize'),
                ObjectiveMetric('profit_factor', 'maximize'), ObjectiveMetric('avg_hold_minutes', 'minimize')],
    constraints=[ConstraintRule('trades', min_value=20), ConstraintRule('profit_factor', min_value=1.05)],
)
DEFAULT_FILTER_SEARCH_SPACE = FilterSearchSpace()
