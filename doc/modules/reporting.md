# Reporting Module

Code location: `hypertrade/artifacts.py` and `hypertrade/ui/state.py`.

Reporting responsibilities load experiment and benchmark artifacts for analytics surfaces. This includes listing runs, reading manifests, loading parquet tables, exposing benchmark suite summaries, and providing safe fallbacks when optional files are missing.

The current implementation keeps reporting helpers close to the artifact registry and UI state layer rather than in a separate `hypertrade/reporting` package.

Important boundaries:

- Reporting reads persisted artifacts; it does not mutate study results.
- Dashboard pages should go through shared loading helpers where possible.
- Missing or empty artifact tables should become empty dataframes or user-visible empty states.
