# HyperTrade Codex Task Pack

Use [blueprint_for_codex.md](/Users/alexey/site/HyperTrade/blueprint_for_codex.md) as the authoritative architecture document.

## Implementation Rules

- Phase 1 optimizes filters over existing external signals only.
- Do not preserve the old scalar weighted score.
- Do not preserve backward compatibility if the new structure is cleaner.
- Do not add a portfolio layer.
- Do not implement walk-forward in this phase.
- Use Streamlit + Plotly for the analytics UI.
- Treat `data/signals/PF20250597.csv` and `data/signals/signals.csv` as required benchmarks.

## Workstream 1. Restructure the Codebase

Goal:

- replace the script-first layout with explicit modules for data, features, signals, simulation, optimization, experiments, reporting, and UI.

Tasks:

1. Introduce the new package structure.
2. Move domain logic out of ad-hoc scripts into modules.
3. Keep thin CLI entrypoints only.
4. Remove or quarantine inactive legacy modules from the active execution path.

Acceptance criteria:

- no core business logic lives in `run_*.py` scripts;
- the package layout clearly separates concerns;
- the active runtime no longer depends on the old directory layout even if some legacy files remain for reference.

## Workstream 2. Define Strong Config Contracts

Goal:

- replace fragile positional indicator configs and mismatched contracts with explicit schemas or typed models.

Tasks:

1. Create typed configs for:
   - market settings
   - execution settings
   - indicator/filter settings
   - optimization objective profile
   - benchmark profile
2. Normalize indicator configuration definitions across all modules.
3. Fix Bollinger and ADX contract mismatches.
4. Remove magic list-index semantics from the active path.

Acceptance criteria:

- no active module relies on ambiguous list ordering for indicator meaning;
- Bollinger and ADX paths are internally consistent;
- config validation fails early and with clear errors.

## Workstream 3. Rebuild the Objective System

Goal:

- replace the old weighted score with true multi-objective optimization.

Tasks:

1. Build an objective catalog with metadata for each metric.
2. Implement Optuna multi-objective studies.
3. Add constraint support for hard gates.
4. Add Pareto extraction utilities.
5. Add candidate-picker logic for dashboard display only.

Acceptance criteria:

- the optimizer no longer exposes the old weighted-score path;
- studies run with multiple directions;
- Pareto candidates are persisted and visible in artifacts;
- the UI can select a preferred candidate from the Pareto set without changing the underlying study objective.

## Workstream 4. Build the Filter Engine for Existing Signals

Goal:

- optimize only filter logic over incoming client signals.

Tasks:

1. Keep the external signal loader as a first-class input.
2. Implement or clean up filter families:
   - EMA
   - RSI
   - MACD
   - ADX
   - ATR gate
   - Donchian gate
   - VWAP
   - volume ratio
   - confirm-bars
   - delay-open
   - time-based acceptance gates
3. Ensure filters act as:
   - accept/reject
   - delay
   - optional ranking inputs
4. Keep long/short support.

Acceptance criteria:

- the system can optimize filters on top of externally supplied signals;
- it does not depend on native signal generation;
- long and short paths both work, even if short remains secondary in research focus.

## Workstream 5. Rebuild Result Persistence

Goal:

- replace fragmented CSV saving with a proper experiment registry.

Tasks:

1. Create a run registry with a stable run id.
2. Save one artifact bundle per run.
3. Store study data in a queryable format.
4. Save objective profile, benchmark profile, environment info, and runtime metadata.
5. Support optional exports for CSV, not CSV as the primary artifact format.

Recommended artifact set:

- `manifest.yaml`
- `objective_profile.yaml`
- `study.sqlite3`
- `trials.parquet`
- `pareto_trials.parquet`
- `candidate_shortlist.parquet`
- `trades.parquet`
- `signal_stats.parquet`
- `metrics_summary.json`
- `environment.json`

Acceptance criteria:

- no run spreads outputs across multiple timestamped folders;
- one run id maps to one artifact directory;
- benchmark runs are reproducible and easy to compare.

## Workstream 6. Build the Streamlit + Plotly Dashboard

Goal:

- replace script-based plots with a real research UI.

Tasks:

1. Build a multipage Streamlit app.
2. Implement pages:
   - Overview
   - Experiments
   - Objectives
   - Pareto
   - Trials
   - Trades
   - Filters
   - Benchmarks
3. Use Plotly for interactive visualizations.
4. Add filters, hover details, and run comparison controls.

Acceptance criteria:

- the dashboard opens and navigates between pages;
- the objective editor is usable from the UI;
- benchmark results for `PF20250597.csv` and `signals.csv` are explorable in the UI.

## Workstream 7. Visual Analytics

Goal:

- implement the required analytics views for research and decision support.

Tasks:

1. Pareto scatter views in 2D and 3D.
2. Equity and drawdown views.
3. Trial diagnostics and parameter plots.
4. Trade return and hold-time distributions.
5. Exit reason views.
6. Signal acceptance funnel and rejection analysis.
7. Long vs short split views.
8. Time-of-day and weekday behavior.
9. Symbol contribution views.
10. Benchmark comparison views.

Acceptance criteria:

- all listed views are present in the dashboard;
- each view works on stored artifacts rather than only live in-memory objects;
- charts are interactive and usable for research.

## Workstream 8. Benchmarks and Smoke Validation

Goal:

- require working end-to-end runs on the selected datasets.

Tasks:

1. Add benchmark definitions for:
   - `data/signals/PF20250597.csv`
   - `data/signals/signals.csv`
2. Create small-trial smoke runs for both.
3. Ensure each run creates full artifacts.
4. Ensure the dashboard can load both.

Acceptance criteria:

- both benchmark inputs complete at least one smoke optimization run;
- both produce complete artifact bundles;
- both are visible in the UI and compare cleanly.

## Workstream 9. Test Coverage

Goal:

- add tests for the parts most likely to break or lie.

Required tests:

1. indicator config schema and validation tests;
2. Bollinger and ADX regression tests;
3. filter engine unit tests;
4. metrics contract tests;
5. result registry tests;
6. Pareto persistence tests;
7. benchmark smoke tests;
8. dashboard data-loading tests.

Acceptance criteria:

- the test suite covers the known defect classes from the current repository;
- failures are precise enough to block silent methodological regressions.

## Workstream 10. Documentation

Goal:

- align repository docs with the real system.

Tasks:

1. Rewrite README to describe the actual architecture.
2. Document benchmark usage.
3. Document artifact layout.
4. Document the objective editor and Pareto workflow.
5. Document what is deliberately deferred to phase 2.

Acceptance criteria:

- README no longer claims unsupported RL/TensorTrade capabilities;
- users can run the new optimizer and dashboard from docs alone.

## Explicit Non-Goals

Do not spend phase 1 time on:

- walk-forward implementation;
- native signal generation;
- capital allocation;
- execution integration;
- portfolio analytics;
- corporate-action modeling;
- survivorship-bias correction work beyond what is already implicitly supported by the current data source.

## Suggested Delivery Order

1. package restructure;
2. config contracts;
3. objective system;
4. filter engine cleanup;
5. result registry;
6. dashboard and visualizations;
7. benchmarks;
8. tests and documentation.

## Minimum Validation Before Completion

At minimum, run and verify:

1. static import / compile checks;
2. smoke optimization on `PF20250597.csv`;
3. smoke optimization on `signals.csv`;
4. artifact generation integrity;
5. dashboard launch and load of benchmark runs.

## Final Delivery Checklist

- old weighted score removed;
- multi-objective optimization active;
- objective editor implemented in UI;
- artifact registry implemented;
- Plotly dashboard implemented;
- `PF20250597.csv` and `signals.csv` benchmarked;
- known BB/ADX/config/result-layout defects fixed;
- walk-forward clearly deferred;
- README updated.
