# Objective Profiles and Pareto Workflow

## Objective Model

HyperTrade now uses true multi-objective optimization.

Typical objectives:

- maximize `total_pnl`
- minimize `max_drawdown`
- maximize `profit_factor`
- minimize `avg_hold_minutes`

Constraints are configured separately, for example:

- minimum trade count
- minimum profit factor

## Objective Profiles

Objective profiles are stored under:

```text
artifacts/objective_profiles/
```

Each profile defines:

- objective list
- direction for each objective
- constraint rules
- candidate selection policy

The default profile is `alpha_pareto_default`.

## UI Workflow

Use the `Objectives` page to:

- load the current run profile or a saved profile;
- select objective metrics;
- choose maximize/minimize direction per metric;
- edit constraints directly as rows of `metric / min_value / max_value`;
- choose the preferred candidate policy;
- save reusable profiles.

Use the `Launcher` page to:

- choose a benchmark or custom signal file;
- choose a saved objective profile;
- edit or load a search space;
- start a new optimization run.

## Pareto Workflow

After a run:

1. open `Pareto` to inspect the frontier;
2. use 2D/3D plots to understand trade-offs;
3. use the scatter matrix and correlation heatmap to inspect metric/parameter interactions;
4. inspect `Closest to Utopia` for the currently preferred candidates;
5. go to `Trades`, `Filters`, and `Trials` to validate how and why a candidate behaves.

Important:

- the preferred candidate is selected from the Pareto frontier;
- supported candidate policies currently include `utopia_distance`, `max_total_pnl`, `max_profit_factor`, `min_max_drawdown`, and `min_avg_hold_minutes`;
- that selection does not restore the old weighted-score optimizer.
