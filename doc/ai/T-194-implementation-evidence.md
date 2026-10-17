# T-194 implementation evidence · 2026-10-17

Implementation handoff only. Independent Pipeline CHECK, exact-SHA ChatGPT Review and Draft PR remain pending. No commit, push, PR, merge or deployment was performed by Executor. Git HEAD remains `c5710961f868afaad5e1ca6117504526766e0651`; implementation is in the working tree.

## Deterministic checks

- Homebrew CPython 3.14.8, repository `.venv`; pinned runtime/dev dependencies installed. `pip check` reports no broken requirements.
- `bash doc/ai/project-checks.sh`: syntax compilation, API/dashboard imports and **24 passed** pytest cases.
- Existing tests/fixtures were reused. Exactly three new focused cases cover execution prices/entry exits/costs, portfolio admission/equity, and unexpected failed-trial diagnostic bundles. Obsolete BB sign assertions were replaced with behavior-changing mode/threshold assertions; metric/import expectations follow the new contract.
- A local wheel built successfully using `pip wheel --no-deps --no-build-isolation .`; temporary build outputs were removed.
- Real all-indicator calculation paths and ISO/dd.mm.yyyy/explicit-offset date equivalence were exercised without modifying source data. Required EMA/MACD periods use the actual requested configuration, removing the missing-middle-period reconstruction bug.

## Real benchmark smoke

`CHECK_MODE=benchmark bash doc/ai/project-checks.sh` passed. Suite: `artifacts/benchmarks/benchmark_suite_20261017T115226Z`.

| Fixture | Status | Complete bundle |
| --- | --- | --- |
| PF20250597 | success | yes |
| signals | no_valid_candidates | yes |

The PF20250597 candidate has 353 normalized research trades and 214 portfolio trades; the different admission counts are intentional. Research sum is 359.327614 percentage points; portfolio return is 15.955017%, with 11.443953% peak-equity drawdown. These are different quantities and are labelled separately. This is a smoke result, not OOS validation or an investment performance claim.

All required files, Parquet readability and inventory SHA256 values were verified for both final study bundles. The PF portfolio cash/security/collateral/liability identity held across 6,685 event/mark rows, with nonnegative buying power. Both bundles have empty unexpected-error lists. Early incomplete smoke bundles were removed from approved generated directories.

## Dashboard execution checks

Streamlit server started successfully on loopback and was stopped after checks. AppTest exercised all five views, 3D and 2D Explorer, normalized/portfolio Candidate Analysis, session candidate/shortlist persistence across navigation, and a real New Experiment → stored run → Candidate Analysis flow. No framework exceptions occurred. All displayed data came from actual local runs.

The in-app Browser backend was unavailable in this session, so a screenshot/visual browser inspection could not be completed. Framework execution checks do not replace that visual inspection; independent review should verify layout/contrast and pointer interactions in a browser.

## Preservation and boundaries

A before/after file inventory matched for **2,625 protected files**, using relative path, byte size and nanosecond mtime. It includes 1,813 OHLC parquet, 797 indicator parquet, signals/raw files, fixture CSVs and `.env`. Protected data was never written. New indicator columns only use `data/cache`.

The intentional entry OHLC/ranking and same-bar SL/TP assumptions remain. Full cache invalidation, walk-forward/OOS, borrow/maintenance-margin/corporate-action modeling remain outside delivery; details are in `doc/simulation.md` and `doc/phase2_deferred.md`. Behavior remains EXPERIMENTAL pending a separately approved hardening task.
