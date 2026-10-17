# Changes

## T-194 · 2026-10-17 · implementation, review pending

- CPython 3.14 virtual environment, pinned runtime/dev dependencies and portable project checks.
- Explicit domain modules replace implementation-heavy package initializers, duplicate loading paths and the UI analysis monolith.
- PSAR fills at SAR with directional gap handling; SL/TP and known entry trailing level apply once on the entry bar. Per-share fees/slippage are explicit.
- Final admission separates eligibility, non-mutating cooldown, ranking, side caps and capital-aware fallback. Three behavior-changing Bollinger modes replace the inert sign.
- Research normalized metrics/baseline and chronological integer-share portfolio accounting are separate. Short proceeds/collateral are separated and locked.
- Schema-3 complete bundles persist status, baseline, Pareto analytics, rejection details and unexpected-failure tracebacks.
- Five dark Streamlit/Plotly views provide 3D/2D Pareto and linked candidate/baseline analysis.
- Approved historical generated results and obsolete local AI queue instructions removed; source inputs retained.

Independent Pipeline checks, exact-SHA review and Draft PR are still required. Intentional lookahead/intrabar assumptions and deferred model limitations are documented; no merge/deploy is authorized.
