# Artifacts and Metadata

## Experiment Run Bundle

Each optimization run writes exactly one bundle under:

```text
artifacts/experiments/<run_id>/
```

Important files:

- `manifest.json` and `manifest.yaml`
  - run id
  - timestamps
  - duration
  - benchmark name
  - signal path
  - study name
  - trial count
  - seed
  - objective profile name/path
  - config hashes
  - signal-file hash
  - runtime metadata
- `objective_profile.json` and `objective_profile.yaml`
- `search_space.json` and `search_space.yaml`
- `benchmark_profile.json`
- `metrics_summary.json`
- `environment.json`
- `artifact_index.json`
  - file inventory with file sizes and sha256 values
- `logs.txt`
- `study.sqlite3`
- `trials.parquet`
- `pareto_trials.parquet`
- `candidate_shortlist.parquet`
- `trades.parquet`
- `signal_stats.parquet`

## Benchmark Suite Bundle

Each benchmark suite run writes one bundle under:

```text
artifacts/benchmarks/<suite_id>/
```

Files:

- `manifest.json` and `manifest.yaml`
- `summary.json`
- `objective_profile.json` and `objective_profile.yaml`
- `search_space.json` and `search_space.yaml`
- `benchmark_runs.parquet`
- `artifact_index.json`

## Contract Rules

- one run id maps to one directory only;
- one directory contains the complete artifact set for that run;
- Parquet is the primary storage format for analytic tables;
- JSON/YAML is used for metadata;
- CSV export, if added later, is optional and secondary.

## Why This Contract Exists

The old project layout could spread outputs across multiple timestamped directories and made runs hard to reproduce or compare. The current registry is designed to make every study inspectable, hashable, and benchmarkable as one unit.
