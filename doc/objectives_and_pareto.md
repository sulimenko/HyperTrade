# Objective profiles and Pareto research

`ObjectiveProfile` defines metric directions, optional min/max constraints and a preferred-candidate policy. NSGA-II is multi-objective; no weighted objective score is implemented. Defaults maximize `sum_trade_returns_pct` and `profit_factor`, minimize `proxy_drawdown_usd` and `avg_hold_minutes`, and require 20 trades / PF ≥1.05.

The sum is research percentage points across fixed-nominal trades. Proxy drawdown is normalized dollars on an entry-ordered sequential curve, anchored at zero PnL. Portfolio objectives explicitly use `portfolio_return_pct` and `portfolio_max_drawdown_pct` on shared chronological equity. Both labels and metric keys preserve this distinction.

Other catalog metrics include trade count and nonnegative `cvar_loss_5_pct` (minimize). Undefined PF with no losses is null; candidates requiring it as objective/constraint are pruned with a reason. No-trade and violated constraints are normal research outcomes. Unexpected exceptions become FAIL, with full diagnostics and a failed run status.

Every nondominated valid trial is retained. Preferred candidates are a subset selected by normalized Euclidean distance to the ideal point, or by a configured primary metric with that distance as tie-breaker. This is shortlist presentation, not a scalar optimization objective.

Conditional sampling omits disabled indicators, unused confirmation count, ranking when no ranked indicator is active, unused trailing step, and percentage SL/TP under ATR exits. Bollinger uses actual %B rules:

| Mode | Long | Short |
| --- | --- | --- |
| `avoid_extremes` | lower ≤ %B ≤ upper | lower ≤ %B ≤ upper |
| `mean_reversion` | %B ≤ lower | %B ≥ upper |
| `trend_continuation` | %B ≥ upper | %B ≤ lower |

Both thresholds are sampled and change eligibility. There is no `bb_sign`. Existing RTH UTC-day VWAP and intentional entry OHLC/same-bar SL/TP assumptions are retained.
