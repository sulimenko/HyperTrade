# UI Guide

## Important Distinction

These commands do different things:

```bash
python run_dashboard.py
python run_benchmarks.py --n_trials 1
```

- `run_dashboard` starts the Streamlit web UI and keeps running
- `run_benchmarks` runs benchmark studies in the terminal and exits

If your browser was connected to an older Streamlit session and that process is gone, the browser will show a `Connection error`. That does not mean the benchmark run failed. It only means the UI process is not running.

To recover, run:

```bash
python run_dashboard.py
```

## Start the Dashboard

```bash
python run_dashboard.py
```

Open:

```text
http://localhost:8501
```

## Normal User Workflow

1. Open `Home` for the quick instructions.
2. Open `Objectives` and create or load an objective profile.
3. Open `Launcher`.
4. Choose a benchmark or provide a custom `Signals path`.
5. Set `Run label` and `Trials`.
6. Adjust the search space if needed.
7. Click `Run optimization`.
8. Inspect results in `Pareto`, `Trials`, `Trades`, and `Filters`.
9. Use `Benchmarks` for suite-level comparison.

## Launcher Fields

- `Benchmark`
  - dataset family used for grouping and comparison
- `Signals path`
  - CSV file to optimize against
- `Run label`
  - specific name for this run directory and study
- `Trials`
  - Optuna trial count
- `Objective profile`
  - multi-objective definition and constraints
- `Search space`
  - parameter ranges for optimization

## Page Reference

- `Home`: startup help and workflow
- `Overview`: recent runs and benchmark suite status
- `Experiments`: filterable list of experiment runs
- `Launcher`: new optimization run
- `Objectives`: objective profile editor
- `Pareto`: frontier analysis and preferred candidates
- `Trials`: trial progression and parameter interactions
- `Trades`: trade-level analytics
- `Filters`: signal acceptance and conversion analytics
- `Benchmarks`: benchmark suite and benchmark-run comparison
