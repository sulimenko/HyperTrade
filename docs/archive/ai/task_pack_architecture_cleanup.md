# Task Pack: Architecture Cleanup and Legacy Removal

## Goal

Bring the repository to one clear runtime architecture with:

- one source of truth for execution;
- fewer files and fewer package layers;
- no active legacy code paths;
- clean root-level entrypoints;
- a sane `.gitignore`;
- reproducible tracked fixtures policy for benchmarks/tests.

This task pack is for cleanup and consolidation, not feature expansion.

## Current State Summary

The repository currently has two architectures at once:

- legacy runtime in `core/`, `loader/`, `config/`, `utils/`, root scripts, and old visualizers;
- active phase-1 runtime in `hypertrade/`.

Main problems:

- root-level scripts still exist with legacy behavior and legacy imports;
- active package is over-fragmented into many tiny files;
- `hypertrade/app/cli/` duplicates root entrypoint responsibility;
- `hypertrade/ui/pages/` splits the UI into many small files with weak cohesion;
- root contains AI/planning artifacts that are not product code;
- `artifacts/` is present in the repository workspace and is not ignored;
- `.gitignore` ignores `data/signals/*`, while the product depends on benchmark signal files from that location;
- the repository mixes runtime code, generated outputs, private datasets, benchmark fixtures, and historical scripts.

## Source Of Truth

After cleanup, the only active runtime must be the new architecture.

Allowed active product code:

- `hypertrade/`
- root entrypoints
- `tests/`
- `docs/`

Everything else must be either:

- deleted;
- moved under `docs/archive/` if needed for historical reference;
- ignored by git if generated or local-only.

## Target Repository Shape

Target structure:

```text
/
  README.md
  requirements.txt
  .gitignore
  run_optimize.py
  run_benchmarks.py
  run_dashboard.py
  streamlit_app.py
  hypertrade/
    __init__.py
    api.py
    config.py
    data.py
    signals.py
    simulation.py
    optimization.py
    artifacts.py
    ui/
      app.py
      launcher.py
      objectives.py
      analysis.py
  tests/
  docs/
```

Notes:

- root entrypoints become thin wrappers only;
- `hypertrade/api.py` becomes the public package surface;
- internal code is consolidated by responsibility, not split by micro-file;
- UI is grouped into a few cohesive modules instead of many page files.

## Main Product Functions To Expose At The Root

These are the real top-level functions of the project and should be obvious from the repository root:

1. `optimize existing signals`
   - run multi-objective optimization over external signal CSVs
2. `run benchmark suite`
   - execute standard benchmark datasets and validate artifact completeness
3. `launch dashboard`
   - open the research UI for analysis, launching new studies, and inspecting outputs
4. `inspect artifacts`
   - load, compare, and summarize experiment runs and benchmark suites

Recommended public API in `hypertrade/api.py`:

- `optimize_run(...)`
- `run_benchmark_suite(...)`
- `launch_dashboard(...)`
- `load_run(...)`
- `load_benchmark_suite(...)`

Recommended root wrappers:

- `run_optimize.py`
- `run_benchmarks.py`
- `run_dashboard.py`
- `streamlit_app.py`

## Workstream 1: Freeze One Runtime

### Objective

Remove ambiguity about which code is active.

### Tasks

1. Replace root-level legacy scripts with thin wrappers to the new runtime:
   - `run_optuna.py` -> delete or replace with `run_optimize.py`
   - `run_single.py` -> delete unless there is a confirmed modern use case
   - `visual_optuna.py` -> delete
   - `visual_single.py` -> delete
2. Add explicit root wrappers:
   - `run_optimize.py`
   - `run_benchmarks.py`
   - `run_dashboard.py`
3. Remove `hypertrade/app/cli/` after moving its logic to root wrappers or `hypertrade/api.py`.
4. Update README and docs so every example points to the same entrypoints.

### Acceptance Criteria

- there is exactly one supported CLI path for optimization;
- there is exactly one supported CLI path for benchmark suites;
- there is exactly one supported CLI path for the dashboard;
- no user-facing docs reference deleted legacy scripts.

## Workstream 2: Delete Legacy Runtime

### Objective

Delete the old architecture once the new root entrypoints are wired.

### Delete Entirely

- `core/`
- `loader/`
- `config/`
- `utils/`

### Preconditions

- grep confirms no imports remain from these directories in active code or tests;
- root wrappers point only to `hypertrade` runtime;
- tests pass using only the new runtime.

### Acceptance Criteria

- repository contains no active imports from legacy packages;
- all old runtime paths are gone from code, docs, and tests.

## Workstream 3: Reduce File Count Inside `hypertrade/`

### Objective

Reduce fragmentation while keeping coherent module boundaries.

### Consolidation Map

1. Merge `hypertrade/config/defaults.py`, `hypertrade/config/objective_catalog.py`, and the public config surface of `hypertrade/config/schemas.py`
   - target: `hypertrade/config.py`
2. Merge `hypertrade/data/api_client.py`, `hypertrade/data/market_store.py`, `hypertrade/data/indicator_store.py`, `hypertrade/data/signal_store.py`, `hypertrade/signals/loaders.py`
   - target: `hypertrade/data.py`
3. Merge `hypertrade/signals/filter_space.py`, `hypertrade/signals/filters.py`, `hypertrade/signals/acceptance.py`
   - target: `hypertrade/signals.py`
4. Merge `hypertrade/simulation/backtester.py`, `hypertrade/simulation/metrics.py`, `hypertrade/simulation/trade_simulator.py`, `hypertrade/simulation/market_time.py`
   - target: `hypertrade/simulation.py`
5. Merge `hypertrade/optimization/study_runner.py` and `hypertrade/optimization/pareto.py`
   - target: `hypertrade/optimization.py`
6. Merge `hypertrade/experiments/registry.py`, `hypertrade/experiments/benchmark.py`, and `hypertrade/reporting/analytics.py`
   - target: `hypertrade/artifacts.py`
7. Remove empty `__init__.py` files where package nesting is no longer needed.

### Rules

- do not merge into huge 1000+ line “god files”;
- aim for 8-12 meaningful Python modules total inside `hypertrade/`;
- prefer merging thin adapter files first.

### Acceptance Criteria

- file count inside `hypertrade/` is materially reduced;
- no tiny one-function wrapper modules remain without strong justification;
- import graph is flatter and easier to understand.

## Workstream 4: Simplify UI Structure

### Objective

Reduce page-file sprawl and make the dashboard easier to maintain.

### Tasks

1. Replace `hypertrade/ui/pages/` micro-pages with grouped UI modules:
   - `hypertrade/ui/app.py`
   - `hypertrade/ui/launcher.py`
   - `hypertrade/ui/objectives.py`
   - `hypertrade/ui/analysis.py`
2. Merge `hypertrade/ui/state.py` into either `ui/app.py` or a single `ui/helpers.py`.
3. Keep navigation labels in one place.
4. Keep `streamlit_app.py` as the only Streamlit entrypoint.

### Suggested Grouping

- `launcher.py`
  - benchmark selection
  - search space editing
  - run execution
- `objectives.py`
  - objective profile editing
- `analysis.py`
  - overview
  - experiments
  - pareto
  - trials
  - trades
  - filters
  - benchmarks

### Acceptance Criteria

- no `hypertrade/ui/pages/` folder;
- UI is still modular, but not split into 9-10 small files;
- Streamlit does not auto-detect accidental multipage scripts.

## Workstream 5: Fix Data And Fixture Policy

### Objective

Separate tracked benchmark fixtures from local/private/generated market data.

### Problem

Current repo behavior is inconsistent:

- benchmark runner depends on files in `data/signals/`;
- `.gitignore` ignores `data/signals/*`;
- local market/indicator caches are mixed into `data/`.

### Tasks

1. Create tracked benchmark fixture location:
   - `benchmarks/fixtures/`
2. Move or copy required benchmark signal CSVs there:
   - `PF20250597.csv`
   - `signals.csv`
3. Update benchmark definitions to use tracked fixture paths.
4. Keep large/local/generated data under ignored directories only:
   - `data/ohlc/`
   - `data/indicators/`
   - `data/raw/`
   - optional private `data/signals/private/`
5. Document what data is:
   - tracked fixture
   - generated cache
   - private local input

### Acceptance Criteria

- benchmark suite works on tracked fixtures;
- git state does not depend on untracked local benchmark CSVs;
- generated caches are clearly non-source artifacts.

## Workstream 6: Clean Root Directory

### Objective

Make the root understandable in one screen.

### Tasks

1. Remove AI coordination documents from the root:
   - `blueprint_for_codex.md`
   - `codex_task_pack.md`
   - `codex_reviewer_master_prompt.md`
2. Either:
   - move them to `docs/archive/ai/`, or
   - delete them if they are no longer needed
3. Keep only product-facing files at root:
   - README
   - requirements
   - gitignore
   - root entrypoints
   - `hypertrade/`
   - `tests/`
   - `docs/`

### Acceptance Criteria

- repository root is product-facing, not process-facing;
- no abandoned one-off scripts remain at root.

## Workstream 7: `.gitignore` Policy

### Objective

Stop tracking or showing generated local files.

### Additions Required

```gitignore
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# Local envs and editor noise
.conda/
.venv/
.DS_Store
.env

# Streamlit and notebooks
.streamlit/
.ipynb_checkpoints/

# Generated outputs
artifacts/
results/
*.sqlite3

# Local data caches
data/ohlc/
data/indicators/
data/raw/

# Optional private user signals
data/signals/private/
```

### Important Rule

Do not blanket-ignore tracked benchmark fixtures. If benchmark fixtures remain under `data/signals/`, use explicit allow-list exceptions or move them out of that directory.

### Acceptance Criteria

- `git status` is not polluted by generated runs or caches;
- tracked fixtures remain available;
- local machine noise is ignored by default.

## Workstream 8: Public API Cleanup

### Objective

Make the main project behavior obvious to both Python users and CLI users.

### Tasks

1. Add `hypertrade/api.py`.
2. Re-export stable root functions in `hypertrade/__init__.py`.
3. Ensure internal modules do not import through accidental UI layers.
4. Keep package usage simple:

```python
from hypertrade import optimize_run, run_benchmark_suite
```

### Acceptance Criteria

- one public API surface exists;
- CLI and UI both use the same underlying functions.

## Workstream 9: Docs Consolidation

### Objective

Reduce docs sprawl the same way we reduce code sprawl.

### Tasks

1. Keep:
   - `README.md`
   - `docs/architecture.md`
   - `docs/user_guide.md`
2. Merge the rest into these or archive them.
3. Remove docs that only exist to coordinate AI work unless they are intentionally archived.

### Acceptance Criteria

- docs are short, current, and tied to the actual code structure.

## Workstream 10: Verification

### Must Pass

1. import smoke:
   - `python -c "import hypertrade"`
2. optimize smoke:
   - one trial on `PF20250597`
3. benchmark smoke:
   - `python -m run_benchmarks --n_trials 1` or equivalent root entrypoint
4. dashboard startup:
   - `python -m run_dashboard` or equivalent root entrypoint
5. tests:
   - all unit and integration tests green
6. grep checks:
   - no imports from `core`, `loader`, `config`, `utils`
   - no docs references to deleted scripts

## Non-Goals

- do not add walk-forward in this cleanup pass;
- do not add portfolio layer;
- do not add native signal generation;
- do not redesign metrics logic unless required by cleanup;
- do not expand scope beyond architecture cleanup.

## Priority Order

1. freeze root entrypoints
2. delete legacy runtime
3. consolidate `hypertrade/`
4. simplify UI structure
5. fix fixture/data policy
6. clean root/docs
7. fix `.gitignore`
8. run verification

## Definition Of Done

The project is done when:

- a new engineer can understand the repository from the root in under 5 minutes;
- there is one runtime architecture, not two;
- the root contains only the main entrypoints and product docs;
- `git status` stays clean during normal local runs;
- benchmark fixtures are reproducible;
- the file count is reduced without creating new ambiguity.
