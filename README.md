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

The active implementation lives in `hypertrade/`.

High-level layout:

- `hypertrade/data`: signal and market-data loading
- `hypertrade/features`: indicator hydration
- `hypertrade/signals`: filter space and signal acceptance policy
- `hypertrade/simulation`: trade simulation and metrics
- `hypertrade/optimization`: Optuna multi-objective study runner and Pareto handling
- `hypertrade/artifacts.py`: artifact registry and benchmark harness
- `hypertrade/ui`: Streamlit dashboard

The supported runtime surface is the root entrypoints plus `hypertrade/`.

Module docs:

- [Data](doc/modules/data.md)
- [Features](doc/modules/features.md)
- [Signals](doc/modules/signals.md)
- [Simulation](doc/modules/simulation.md)
- [Optimization](doc/modules/optimization.md)
- [Experiments](doc/modules/experiments.md)
- [Reporting](doc/modules/reporting.md)
- [UI](doc/modules/ui.md)

## Requirements

- Python 3 with a project virtual environment
- dependencies installed from [requirements.txt]
- benchmark inputs:
  - `benchmarks/fixtures/PF20250597.csv`
  - `benchmarks/fixtures/signals.csv`

Set up a clean local environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Conda can still be used as a personal local environment, but the official project commands use the active standard Python environment.

## CLI

Run one optimization:

```bash
python run_optimize.py \
  --signals benchmarks/fixtures/PF20250597.csv \
  --n_trials 25 \
  --benchmark_name PF20250597 \
  --run_label PF20250597_manual
```

`benchmark_name` identifies the dataset family for comparison. `run_label` names the specific run directory and study.

Run the required benchmark suite:

```bash
python run_benchmarks.py --n_trials 1
```

Launch the dashboard:

```bash
python run_dashboard.py
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

- [Architecture and Scope](doc/architecture.md)
- [Artifacts and Metadata](doc/artifacts.md)
- [Benchmarks](doc/benchmarks.md)
- [Objective Profiles and Pareto Workflow](doc/objectives_and_pareto.md)
- [UI Guide](doc/ui_guide.md)
- [Phase 2 Deferred Work](doc/phase2_deferred.md)

## Current Status

Phase 1 implementation now includes:

- multi-objective optimization in place of the old weighted score;
- objective profile editing in UI;
- modular dashboard pages;
- benchmark suite runner for `PF20250597` and `signals`;
- benchmark suite status distinguishes valid frontier runs from `no_valid_candidates`;
- strengthened artifact schema and metadata contract;
- regression coverage for BB/ADX contracts, acceptance policy, registry, benchmark flow, optimizer end-to-end, and UI data loading.
