# T-194 implementation evidence · 2026-10-17

REPAIR implementation handoff only. Independent Pipeline CHECK, exact-SHA ChatGPT Review and Draft PR remain pending. No commit, push, PR, merge or deployment was performed by Executor. The repair is in the working tree on `ai/T-194-v85`, with unchanged C0 HEAD `bb1e8ac64037eafc1ad1edd0634555580e4879fa`.

## C0 review repairs R-1 through R-4

- R-1: settle executed intrabar exits at actual fill prices before close marks; retain genuine preceding closes before open/gap/time exits. Other active positions mark normally. Exclude the exit candle's subsequent Close from each position's marks. Portfolio objectives consume this corrected equity. Entry/settlement/collateral conventions are unchanged.
- R-2: validate ATR protective/target direction and positive finite prices. Ignore each wrong-side Donchian bound independently, retaining its valid ATR level; signed distances replace `abs()`. Invalid ATR levels have an explicit technical rejection. Existing same-candle precedence remains.
- R-3: explicit `python -m hypertrade.data.market --signals <CSV>` preparation uses process-environment `DATA_BASE_URL` with the previous US symbol-search/chart API contract. It writes only `data/cache/market`; prepared-history indicators are calculated without stale source columns. Missing/empty instruments and insufficient early history fail before optimization trials with `market_data_preparation` diagnostics. Existing histories ending early retain actionable per-signal no-candles details (QMMM in the historical signals fixture); explicit preparation checks both ends of the CSV span and fails if the API cannot supply them. Complete prepared research is offline. Live external acquisition was not invoked; the transport contract and staged parquet were verified with a response stub.
- R-4: New Experiment saves/loads named reusable profiles outside archived runs, restores saved directions, validates name/catalog/directions/constraints/policy, and still launches unsaved edits. Reload explicitly restores saved values.

`CHECK_MODE=benchmark bash doc/ai/project-checks.sh` passed on the final source: Python 3.14, pip dependency check, compilation, API/dashboard imports, **27 tests passed**. Exactly three new regression cases cover executed-exit equity, ATR/Donchian levels, and explicit preparation/missing-data failure; the existing profile roundtrip case now also covers changed directions, policy and invalid configuration.

The long entry-100 / OHLC=(100,120,94,115) / SL=5% / no-cost / initial-1000 / cap-100 reproduction finishes at **995 USD**, with **0.5% maximum drawdown** and no 1015 peak. The symmetric short, targets, time/gap exits, genuine prior-close peaks, a stop while another position stays active, nonnegative buying power and accounting identity pass. The documented wrong-side Donchian target now exits at ATR TP **102**, with **+2 per share**, rather than synthetic TP105.19; short mirror, valid bounds and invalid ATR levels pass.

Final suite: `artifacts/benchmarks/benchmark_suite_20261017T121203Z`: PF20250597 **success**, signals **no_valid_candidates**, two complete bundles, no failures. Artifact inventory hashes and Parquet readability were checked. PF portfolio accounting identity and nonnegative buying power hold across **6,684** curve rows.

Streamlit AppTest created a named two-objective profile with reversed directions, a trade-count constraint and max-PF policy, saved it, loaded it in a fresh session and launched a real cached-data run. Manifest/profile match the saved configuration, status is success and the bundle is complete. Invalid names/metrics display errors and do not replace the saved profile. Temporary UI runs/profiles stayed outside project result directories. No native browser screenshot was produced in this repair.

`git diff --check` passes. Protected-path Git diff is empty; source parquet counts remain **1,813 OHLC / 797 indicators**. No source CSV, fixture, `.env` or protected source cache was written. No configured external service was contacted. Behavior remains EXPERIMENTAL. Independent exact-SHA approval is still required after Pipeline creates its repair commit.

The following records describe the original implementation before C0 review; its numerical results are historical. That initial working tree was based on `c5710961f868afaad5e1ca6117504526766e0651` and was later committed by Pipeline as C0.

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
