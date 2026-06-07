# Optimization Module

Code location: `hypertrade/optimization`.

The optimization module runs true multi-objective Optuna studies. It loads external signals, samples strategy parameters from the filter search space, runs backtests, applies objective constraints, and writes Pareto outputs.

`run_optimization` is the main programmatic entrypoint. The root `run_optimize.py` script exposes it for CLI use.

Important outputs:

- `trials.parquet`
- `pareto_trials.parquet`
- `candidate_shortlist.parquet`
- `metrics_summary.json`
- study metadata and config snapshots

Important boundaries:

- Candidate preference is selected from the Pareto frontier.
- The removed weighted-score optimizer is not restored.
- Walk-forward validation is deferred.
