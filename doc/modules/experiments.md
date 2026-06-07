# Experiments Module

Code location: `hypertrade/artifacts.py` and root benchmark entrypoints.

Experiment responsibilities are implemented through the artifact registry and benchmark-suite helpers. The registry creates run directories, writes JSON/YAML/parquet outputs, records environment metadata, builds artifact inventories, and validates required files.

Benchmark orchestration uses the configured benchmark definitions and calls optimization for each required dataset. The root `run_benchmarks.py` script is the supported CLI entrypoint.

Important boundaries:

- Experiment outputs live under `artifacts/experiments/` and `artifacts/benchmarks/`.
- Runtime artifacts are generated data and should not be committed.
- Benchmark status distinguishes successful runs from `no_valid_candidates`.
