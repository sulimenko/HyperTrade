# HyperTrade

HyperTrade is a research-lab for optimizing filters over existing external trading signals.

Phase 1 is intentionally narrow:

- input is an existing signal CSV from a client robot or working strategy;
- the system optimizes filters, acceptance policy, delay, and confirmation logic on top of those signals;
- optimization is true multi-objective Optuna with Pareto output;
- results are stored as reproducible artifact bundles;
- analysis happens in a Streamlit + Plotly dashboard.

What this repository is not doing in phase 1:

- native signal generation as the main workflow;
- portfolio allocation or capital sizing;
- live execution;
- walk-forward validation;
- weighted-score optimization.

## Architecture

The active implementation lives in [hypertrade](/Users/alexey/site/HyperTrade/hypertrade).

High-level layout:

- `hypertrade/data`: signal and market-data loading
- `hypertrade/features`: indicator hydration
- `hypertrade/signals`: filter space and signal acceptance policy
- `hypertrade/simulation`: trade simulation and metrics
- `hypertrade/optimization`: Optuna multi-objective study runner and Pareto handling
- `hypertrade/experiments`: artifact registry and benchmark harness
- `hypertrade/ui`: Streamlit dashboard

The supported runtime surface is the root entrypoints plus `hypertrade/`.

## Requirements

- Python environment with dependencies from [requirements.txt](/Users/alexey/site/HyperTrade/requirements.txt)
- benchmark inputs:
  - `benchmarks/fixtures/PF20250597.csv`
  - `benchmarks/fixtures/signals.csv`

If you use the local conda environment from this workspace, examples below can be run with `./.conda/bin/python`.

## CLI

Run one optimization:

```bash
./.conda/bin/python run_optimize.py \
  --signals benchmarks/fixtures/PF20250597.csv \
  --n_trials 25 \
  --benchmark_name PF20250597 \
  --run_label PF20250597_manual
```

`benchmark_name` identifies the dataset family for comparison. `run_label` names the specific run directory and study.

Run the required benchmark suite:

```bash
./.conda/bin/python run_benchmarks.py --n_trials 1
```

Launch the dashboard:

```bash
./.conda/bin/python run_dashboard.py
```

## Dashboard

The dashboard currently includes:

- `Overview`: recent runs, run activity, top Pareto candidates, benchmark suite status
- `Experiments`: run table with benchmark/date/profile/study-size filters
- `Launcher`: benchmark selection, objective-profile selection, search-space editor, run trigger
- `Objectives`: objective/constraint editor and saved profile management
- `Pareto`: 2D/3D frontier views, interaction matrix, correlation heatmap
- `Trials`: trial metric trends and parameter interaction plots
- `Trades`: equity, drawdown, rolling win rate, long/short split, weekday/hour analysis
- `Filters`: acceptance funnel, cumulative signal PnL, side/time acceptance views
- `Benchmarks`: benchmark suite inspection and run-to-run comparison

## Artifact Model

Each optimization run writes one stable bundle under `artifacts/experiments/<run_id>/`.

Core files:

- `manifest.json` and `manifest.yaml`
- `objective_profile.json` and `objective_profile.yaml`
- `search_space.json` and `search_space.yaml`
- `benchmark_profile.json`
- `metrics_summary.json`
- `environment.json`
- `artifact_index.json`
- `logs.txt`
- `study.sqlite3`
- `trials.parquet`
- `pareto_trials.parquet`
- `candidate_shortlist.parquet`
- `trades.parquet`
- `signal_stats.parquet`

Benchmark suite runs write a summary bundle under `artifacts/benchmarks/<suite_id>/`.

Detailed docs:

- [Architecture and Scope](/Users/alexey/site/HyperTrade/docs/architecture.md)
- [Artifacts and Metadata](/Users/alexey/site/HyperTrade/docs/artifacts.md)
- [Benchmarks](/Users/alexey/site/HyperTrade/docs/benchmarks.md)
- [Objective Profiles and Pareto Workflow](/Users/alexey/site/HyperTrade/docs/objectives_and_pareto.md)
- [UI Guide](/Users/alexey/site/HyperTrade/docs/ui_guide.md)
- [Phase 2 Deferred Work](/Users/alexey/site/HyperTrade/docs/phase2_deferred.md)

## Current Status

Phase 1 implementation now includes:

- multi-objective optimization in place of the old weighted score;
- objective profile editing in UI;
- modular dashboard pages;
- benchmark suite runner for `PF20250597` and `signals`;
- benchmark suite status distinguishes valid frontier runs from `no_valid_candidates`;
- strengthened artifact schema and metadata contract;
- regression coverage for BB/ADX contracts, acceptance policy, registry, benchmark flow, optimizer end-to-end, and UI data loading.
