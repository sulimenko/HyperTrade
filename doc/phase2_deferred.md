# Explicit limitations

T-194 behavior is EXPERIMENTAL until a later user-approved Weekly Hardening task. Focused financial regressions and existing project tests are evidence, not full stabilization.

- Walk-forward, OOS/CPCV/purge-gap validation are deferred.
- Full indicator-cache invalidation/rebuild is deferred. Source OHLC/indicator parquet is read-only; required missing columns are computed into the derived cache. Source size/mtime provenance is recorded, without a complete content-addressed data snapshot.
- Corporate actions, survivorship corrections, borrow availability/fees, liquidity/partial fills and maintenance-margin liquidation are outside this model. Short collateral is fixed at 100% entry notional; catastrophic gaps can exceed collateral. Negative cash/equity losses are recorded, with zero remaining buying power.
- Missing candles use the first available entry and last known marks; exhausted history closes at its last candle with `data_truncated=true`. There are no synthetic overnight prices.
- Research curves ignore overlaps/capital constraints. They are proxies and never actual capital returns. Baseline is a fixed unfiltered reference; candidates may tune exits as well as filters.
- Entry candle OHLC/ranking at open and the same-bar SL/TP heuristic remain intentionally enabled. Later trailing stops retain their existing OHLC update heuristic. Intrabar timestamps do not reconstruct a tick path.
- 3D selection uses a linked selector/table; session shortlist changes are not durable edits to the stored study. The complete Pareto frontier has linked trade/equity artifacts; other trials expose summaries and diagnostics.

The product scope excludes live execution, native signal generation, ML ranking, an allocation optimizer and the old weighted objective optimizer. Homebrew upgrades and OS/package removals belong to the operator, outside this repository and Executor.
