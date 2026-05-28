# HyperTrade Blueprint for Codex

## 1. Purpose

Re-architect HyperTrade from a small script-based backtesting utility into a cleaner research-lab platform focused on one core job:

- optimize filters over already existing external trading signals;
- maximize alpha;
- support intraday and overnight workflows first;
- keep long and short support, with short treated as secondary in product priority.

This blueprint is the source of truth for phase 1 implementation.

## 2. Product Positioning

Current and near-term positioning:

- primary mode: research-lab;
- future path: semi-auto decision support, then optionally fully automated execution;
- v1 signal workflow: external signals already exist and are loaded into the system;
- v1 goal: find the best filter, acceptance policy, delay policy, and confirmation stack for those signals.

Not in scope for phase 1:

- native signal generation as the primary production path;
- portfolio allocation layer;
- capital sizing logic;
- walk-forward implementation;
- backward compatibility with current result layout or old CLI shape.

## 3. Core Decisions

These are non-negotiable phase 1 decisions.

1. Replace the old scalar weighted score.
   The system must move to true multi-objective optimization with Pareto output. Do not preserve the legacy weighted score mode.

2. Keep signal generation out of phase 1.
   Signals come from the client robot or working strategy. Phase 1 only optimizes filters on top of them.

3. Do not preserve the old file/result structure.
   If a cleaner architecture requires new paths, schemas, and artifacts, use them.

4. Walk-forward is phase 2.
   The existing broken walk-forward code should be quarantined, removed from the active architecture, or clearly marked as deferred. Do not spend phase 1 scope implementing the full replacement.

5. Build a user-facing optimization goal editor in UI.
   The old hard-coded score formula is removed. The user must be able to configure objectives, constraints, and the displayed "preferred candidate" rule in the dashboard.

6. Use Streamlit + Plotly for the new analytics UI.
   Reference: [Streamlit navigation docs](https://docs.streamlit.io/1.50.0/develop/api-reference/navigation/st.navigation), [Plotly Python docs](https://plotly.com/python/), [Dash/Graph docs](https://dash.plotly.com/dash-core-components/graph).

## 4. Why Multi-Objective Replaces Weighted Score

The current weighted score is not equivalent to multi-objective optimization.

Weighted score problems:

- it hard-codes one risk preference into the optimizer;
- it hides the Pareto frontier and therefore hides viable alpha/risk alternatives;
- it makes research brittle because every change in business preference requires redesigning the score formula;
- it can over-reward one metric while masking instability in another.

True multi-objective optimization is better here because:

- it exposes trade-offs explicitly;
- it is easier to explain to a user in a research UI;
- it supports later regime-specific candidate selection without changing the study definition;
- it matches the user's stated goal that "all metrics matter".

Important nuance:

- optimization itself must be multi-objective;
- the UI may still let the user choose a "preferred candidate" from the Pareto set;
- that selection rule is not the optimization objective and must not restore the old weighted-score engine as a hidden legacy path.

## 5. Phase 1 Scope

### 5.1 Must Build

- clean package layout with explicit module boundaries;
- typed configuration objects or schemas;
- external signal ingestion;
- feature and indicator pipeline;
- filter engine for existing signals;
- trade simulation;
- metrics engine;
- Optuna multi-objective study runner;
- experiment registry and reproducible artifacts;
- Streamlit + Plotly dashboard;
- optimization goal editor in UI;
- rich result visualization;
- benchmark support for:
  - `data/signals/PF20250597.csv`
  - `data/signals/signals.csv`

### 5.2 Must Repair

- indicator configuration contract mismatches;
- broken Bollinger path;
- broken ADX path;
- unstable single-run result persistence design;
- unclear metrics/objective contract;
- README mismatch with actual system behavior;
- broken legacy walk-forward module, at least by isolation from the active system.

### 5.3 Must Not Build Yet

- production order execution;
- portfolio construction layer;
- risk budgeting across simultaneous positions;
- walk-forward engine;
- full regime auto-adaptation;
- foundation-model trading signals.

## 6. Target Architecture

Recommended package layout:

```text
hypertrade/
  app/
    cli/
      run_backtest.py
      run_optimize.py
      run_dashboard.py
  config/
    schemas.py
    defaults.py
    objective_catalog.py
  data/
    market_store.py
    signal_store.py
    artifact_store.py
    indicator_store.py
  features/
    indicator_engine.py
    feature_views.py
    feature_registry.py
  signals/
    loaders.py
    filters.py
    filter_space.py
    acceptance.py
  simulation/
    market_time.py
    trade_simulator.py
    backtester.py
    metrics.py
  optimization/
    study_runner.py
    objective_spec.py
    pareto.py
    candidate_picker.py
  experiments/
    registry.py
    manifests.py
    benchmark.py
  reporting/
    exports.py
    analytics.py
  ui/
    streamlit_app.py
    pages/
      01_overview.py
      02_experiments.py
      03_objectives.py
      04_pareto.py
      05_trials.py
      06_trades.py
      07_filters.py
      08_benchmarks.py
tests/
```

Design rules:

- no business logic in CLI entrypoints;
- no implicit config packing via loosely ordered lists;
- no hidden coupling between optimizer and metric names;
- no writing result files from multiple helper calls that each create a new run directory;
- no active dependency on the legacy module layout once migration is complete.

## 7. Functional Architecture

### 7.1 Data Flow

1. Load external signals.
2. Load or hydrate market data.
3. Load or compute indicators and derived features.
4. Apply candidate filters to each incoming signal.
5. Simulate trades for accepted candidates.
6. Compute per-trade and per-run metrics.
7. Run multi-objective optimization over filter/search parameters.
8. Persist the full study and derived analytics.
9. Explore results in UI.

### 7.2 Signal Optimization v1

Phase 1 signal work is filter optimization only.

Suggested filter families:

- trend confirmation:
  - EMA alignment
  - MACD confirmation
  - ADX trend gate
- momentum confirmation:
  - RSI range / threshold behavior
  - breakout confirmation
- volatility and market quality:
  - ATR range gate
  - Donchian proximity filter
  - volume ratio filter
- execution timing:
  - delay-open optimization
  - time-of-day acceptance windows
  - cooldown after prior signal on same symbol
- quality controls:
  - reject near-extreme stretch states
  - reject low-information flat states
  - confirm with multiple bars

Output modes for v1:

- accept or reject signal;
- optionally rank accepted signal candidates;
- optionally apply delayed entry rules;
- preserve long/short support.

Do not add native signal generation to the mainline architecture in phase 1.

## 8. Objective System

### 8.1 Optimization Model

The optimizer must use real multi-objective Optuna studies.

Expected objective examples:

- maximize total return;
- minimize max drawdown;
- maximize profit factor;
- minimize average hold time;
- maximize trade count subject to quality gates;
- later, optionally maximize stability metrics.

Use constraints separately from objectives.

Examples of hard gates:

- minimum trades;
- minimum total return;
- maximum drawdown;
- minimum profit factor;
- maximum average hold time.

### 8.2 UI Goal Editor

Build an "Optimization Goal Editor" page in Streamlit where the user can:

- choose active objectives from a metric catalog;
- choose direction per metric;
- define hard constraints;
- save/load named objective profiles;
- choose a "preferred candidate" policy for viewing Pareto results.

Allowed preferred candidate policies:

- lexicographic order;
- utopia-point distance;
- knee-point selection;
- filtered shortlist view.

Not allowed:

- restoring the old legacy weighted-score engine as a hidden compatibility path.

## 9. Metrics

Metrics must be organized into a catalog with:

- id;
- label;
- description;
- unit;
- direction;
- category.

Minimum metric categories:

- return
- risk
- trade quality
- timing
- filter behavior
- study diagnostics

Recommended phase 1 metric set:

- total return percent;
- median return percent;
- profit factor;
- max drawdown;
- CVaR / left tail loss;
- win rate;
- expectancy;
- average hold minutes;
- trade count;
- exit-reason distribution;
- acceptance ratio vs rejected ratio.

## 10. Experiment Registry and Artifact Schema

Move away from ad-hoc CSV-only result folders.

Recommended artifact structure:

```text
artifacts/
  experiments/
    {run_id}/
      manifest.yaml
      objective_profile.yaml
      study.sqlite3
      trials.parquet
      pareto_trials.parquet
      candidate_shortlist.parquet
      trades.parquet
      signal_stats.parquet
      metrics_summary.json
      filters.json
      environment.json
      logs.txt
```

Requirements:

- every run has one stable run id;
- one run id maps to one directory only;
- all outputs for one run live together;
- save config hash, data source, benchmark name, git commit if available, timestamps, and runtime environment;
- use Parquet for analytic tables and JSON/YAML for metadata;
- CSV export may exist as an optional reporting feature, not as the primary storage format.

## 11. Visualization and UI

Use Streamlit multipage navigation and Plotly interactive charts.

Required pages:

1. Overview
   - recent runs
   - benchmark summaries
   - top Pareto candidates

2. Experiments
   - run table
   - filters by dataset, objective profile, date, study size

3. Objectives
   - goal editor
   - objective catalog
   - constraint editor
   - candidate picking rules

4. Pareto
   - 2D and 3D Pareto exploration
   - hover details
   - shortlist extraction

5. Trials
   - objective values by trial
   - parameter importance
   - interaction plots
   - acceptance / rejection funnel

6. Trades
   - equity curve
   - drawdown curve
   - exit reason mix
   - hold time distribution
   - symbol-level contribution

7. Filters
   - acceptance heatmaps
   - rejection reason analysis
   - long vs short behavior
   - time-of-day and day-of-week views

8. Benchmarks
   - compare `PF20250597.csv` and `signals.csv`
   - compare objective profiles
   - compare selected Pareto candidates

Required visualizations:

- Pareto scatter views;
- equity and drawdown;
- rolling metrics;
- trade return histogram;
- hold time histogram;
- exit reason stacked bars;
- parameter heatmaps;
- filter acceptance funnel;
- signal-to-trade conversion chart;
- symbol contribution table;
- long vs short split;
- time-of-day behavior;
- benchmark comparison table and charts.

## 12. Benchmarks

Phase 1 benchmarks:

- `data/signals/PF20250597.csv`
- `data/signals/signals.csv`

Benchmark requirements:

- every major optimizer and UI change must be smoke-tested on both;
- each benchmark run must produce a complete artifact bundle;
- benchmark outputs must be visible in the dashboard.

## 13. Technical Debt to Eliminate

Known issues from the current repository:

- walk-forward module is broken and not connected to the real architecture;
- Bollinger config ordering is inconsistent across modules;
- ADX config ordering is inconsistent across modules;
- result directory creation is fragmented across save calls;
- metrics returned by the engine do not fully match what optimization code expects;
- README overstates RL/TensorTrade capabilities that are not present;
- current structure mixes orchestration and domain logic.

These should be fixed as part of the refactor, not carried forward.

## 14. Phase 2 and Future Directions

Phase 2:

- proper walk-forward validation;
- purge-gap aware split logic;
- train/test degradation reporting;
- parameter drift analysis.

Future R&D:

- regime classifier for automatic strategy-profile switching;
- meta-labeling and ranking overlays;
- foundation time-series models as regime prior, uncertainty signal, or ranker only.

Reference directions:

- [Optuna multi-objective tutorial](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html)
- [Optuna Dashboard docs](https://optuna-dashboard.readthedocs.io/en/latest/)
- [Chronos repository](https://github.com/amazon-science/chronos-forecasting)
- [Moirai-MoE overview](https://www.salesforce.com/blog/time-series-morai-moe/)
- [MoiraiAgent overview](https://www.salesforce.com/blog/moiraiagent/)

## 15. Definition of Done for Phase 1

Phase 1 is done only when all of the following are true:

- the active codebase follows the new module boundaries;
- the legacy weighted score is gone;
- optimization is multi-objective and produces Pareto outputs;
- the UI exposes objective and constraint editing;
- benchmark runs for `PF20250597.csv` and `signals.csv` work end-to-end;
- artifacts are reproducible and saved under one run directory per study;
- Plotly-based visual analytics cover the required views;
- walk-forward is explicitly deferred, not half-implemented;
- the system remains focused on filtering existing signals rather than expanding scope into portfolio execution.
