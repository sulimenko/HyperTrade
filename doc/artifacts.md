# Artifact schema 3

One study ID identifies one directory under `artifacts/experiments`. No historical reader is maintained. Artifact sets are written for `success`, legitimate `no_valid_candidates`, and `failed` outcomes, including failures before the first trial.

| Artifact | Content |
| --- | --- |
| `manifest.json`, `.yaml` | status, EXPERIMENTAL flag, timestamps, configuration hashes, costs/model constants, baseline settings, assumptions, interpreter and Git context |
| `objective_profile.json/.yaml`, `search_space.json/.yaml` | exact objective/constraint and sampling settings |
| `benchmark_profile.json`, `data_snapshot.json` | CSV identity/SHA256; used source parquet sizes/mtimes (not full content hashes) |
| `environment.json`, `logs.txt`, `errors.json` | installed dependency versions, runtime, summary log and full unexpected-error stack traces |
| `study.sqlite3`, `trials.parquet` | all trial states, parameters, metric attrs, errors/constraint reasons |
| `pareto_trials.parquet`, `candidate_shortlist.parquet` | valid nondominated candidates and deterministic preferred subset |
| `candidate_parameters.json` | full resolved settings for every Pareto candidate |
| `trades.parquet`, `portfolio_trades.parquet` | every Pareto candidate's admitted trades, costs and position fields; `candidate_trial` links tables |
| `signal_stats.parquet`, `portfolio_signal_stats.parquet` | per-model counts and JSON per-symbol rejection details |
| `equity.parquet`, `sequential.parquet` | time-based capital ledger versus sequential normalized PnL/proxy drawdown |
| `baseline_metrics.json`, `baseline_*.parquet` | same models' fixed unfiltered reference outcomes, statistics and curves |
| `artifact_index.json` | file sizes/SHA256; its own entry deliberately has no self-hash |

`REQUIRED_RUN_ARTIFACTS` in `hypertrade/experiments/artifacts.py` is the executable completeness list. Empty results are valid Parquet files, not missing files. Every Pareto candidate has analytics; other trials retain summary metrics/status in `trials.parquet` and the study. Unexpected simulator errors produce Optuna `FAIL` and diagnostics; normal failed constraints/no trades produce `PRUNED`. Any unexpected trial error marks the whole run `failed` even if other candidates succeed.

Benchmark suites have their own directory, manifest, exact profiles, `benchmark_runs.parquet`, summary and inventory, linking full study directories. A valid no-candidate study is not a benchmark failure. Missing artifact bundles or failed studies make the benchmark CLI exit nonzero.

Source CSV/parquet/secret files are never modified. Additional indicator caches live separately in `data/cache`. Metadata records source size/mtime without promising a full cache rebuild or immutable market-data snapshot.
