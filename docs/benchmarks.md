# Benchmarks

## Required Benchmark Inputs

Phase 1 benchmark coverage is based on:

- `benchmarks/fixtures/PF20250597.csv`
- `benchmarks/fixtures/signals.csv`

These are the required benchmark datasets for smoke validation and dashboard comparison.

## Run the Benchmark Suite

```bash
./.conda/bin/python run_benchmarks.py --n_trials 1
```

This command:

- runs the selected benchmark inputs sequentially;
- creates full optimization artifact bundles under `artifacts/experiments/`;
- creates one suite summary bundle under `artifacts/benchmarks/`.

## What to Check After a Run

- both benchmark rows exist in `benchmark_runs.parquet`;
- `status` is `success` for each required dataset;
- if a dataset technically runs but produces no valid Pareto candidates, status must be `no_valid_candidates`, not `success`;
- `artifact_complete` is `true`;
- benchmark runs appear in the dashboard under `Benchmarks`;
- suite summary appears in the dashboard under `Benchmark Suites`.

## Current Caveat

`signals.csv` may finish as `no_valid_candidates` if no trades survive constraints. That is a research outcome, not a benchmark-runner failure.
