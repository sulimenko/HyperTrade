# HyperTrade Research

EXPERIMENTAL external CSV signal research on CPython 3.14. Compare an unfiltered robot baseline with filtered candidates using multi-objective Optuna / NSGA-II and a Pareto frontier. Results separate fixed-nominal research quality from capacity-limited portfolio equity.

## Install

Use a CPython 3.14 interpreter. On macOS with Homebrew installed:

```sh
/opt/homebrew/opt/python@3.14/bin/python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
```

On other platforms, substitute `python3.14 -m venv .venv` and the platform's virtual-environment Python path. Runtime and transitive dependencies are pinned in `requirements.txt`; `requirements-dev.txt` adds pytest. Installation requires access to PyPI. No system package upgrade or removal is performed by the project.

```sh
.venv/bin/python run_optimize.py --signals benchmarks/fixtures/PF20250597.csv --n_trials 20
.venv/bin/python run_benchmarks.py --n_trials 1
.venv/bin/python run_dashboard.py
bash doc/ai/project-checks.sh
CHECK_MODE=benchmark bash doc/ai/project-checks.sh
```

`PYTHON_BIN` can override the interpreter in checks; it must be Python 3.14. Both CLIs print the artifact directory; failed runs/suites exit nonzero. `no_valid_candidates` is a valid research outcome and exits zero.

## Data and models

CSV is semicolon-separated, with `datetime`, `long_symbols` (or `symbols`) and optional `short_symbols`. The historical fixture spelling `shotr_symbols` is also read. ISO and `dd.mm.yyyy` dates are accepted. Naive dates explicitly mean UTC; timezone offsets preserve their instant. Equal timestamps are grouped, symbols deduplicated per side, and signals sorted. Inputs are never rewritten.

Research runs read local `data/ohlc/*.parquet` and `data/indicators/*.parquet`. They never download or overwrite source data. Missing market files are explicit eligibility rejections. Additional indicator columns go into `data/cache/indicators`; uploaded CSVs go into `data/cache/uploads`. The existing RTH-only source uses UTC-day VWAP unchanged. Full indicator-cache invalidation is deferred.

- **Research / Normalized:** unlimited budget, identical $10,000 nominal for baseline and candidates, fractional analytical exposure. `sum_trade_returns_pct` is a sum of percentage points, **not** a return on capital. Net PnL, mean/median trade return, win rate, profit factor and a deterministic sequential proxy drawdown are available.
- **Capital / Portfolio:** $100,000 initial equity, maximum $10,000 gross notional per position, affordable integer shares, entry/exit commission in USD/share/side and separate directional fractional slippage. Long notional and 100% short collateral are reserved; short proceeds remain locked. Time equity includes realized and unrealized PnL; drawdown is a percentage of peak equity.

Every study saves a fixed baseline without indicator filters, time gate, ranking, cap or cooldown. Its SL/TP/delay/holding settings use the configured search minima and are persisted. Candidate improvements can include both filters and exit tuning. Both models use the same costs and nominal settings. See [simulation semantics](doc/simulation.md).

## Dashboard

Streamlit + Plotly use a dark navy institutional research theme throughout five views:

1. Research Home: experiment explorer, latest run and baseline status.
2. New Experiment: CSV source/upload, objective profile, full search configuration and start.
3. Optimizer Explorer: 3D/2D Pareto, selectable axes/color, trial statuses, table/selector fallback for 3D, session shortlist.
4. Candidate Analysis: linked Pareto candidate, baseline comparison, normalized/portfolio curves, outcomes, rejections and settings.
5. Compare & Validation: benchmark evidence, run comparison and objective profiles.

No demo results are generated. Walk-forward/OOS validation is not implemented. See [UI guide](doc/ui_guide.md).

## Development and delivery

Responsibilities and artifact schema: [architecture](doc/architecture.md), [artifacts](doc/artifacts.md), [Pareto objectives](doc/objectives_and_pareto.md). There is one public API in `hypertrade/api.py` and no alternate historical results reader.

T-194 is one AI Pipeline v8.5 task. Use the installed `~/.ai-pipeline-v85/bin/ai-pipeline`; local legacy task queues/scripts were removed. Independent checks, review of the exact Pipeline-owned commit, and Draft PR are required before delivery. Executor never commits, pushes, merges or deploys. Old queue tasks T-003/T-004 are superseded and must not run. See [pipeline](doc/ai/pipeline.md).
