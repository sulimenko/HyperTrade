# Architecture — T-194

One Python pipeline accepts external CSV signals, evaluates a fixed baseline, samples candidate settings with NSGA-II, and persists a schema-3 research bundle. Behavior is EXPERIMENTAL pending a separately approved Weekly Hardening task.

| Zone | Owner |
| --- | --- |
| Typed configuration, named indicator settings | `hypertrade/config/{schemas,strategy,defaults}.py` |
| External CSV dates/grouping, read-only OHLC | `hypertrade/data/{signals,market}.py` |
| Feature calculation, additional cached columns | `hypertrade/features/{indicators,indicator_cache}.py` |
| Indicator rules, read-only time eligibility, ranking | `hypertrade/filters/{rules,acceptance,ranking}.py` |
| Trading calendar, execution prices | `hypertrade/simulation/{market_time,trade_engine}.py` |
| Research metrics, capital ledger/time equity, orchestration | `hypertrade/simulation/{metrics,portfolio,backtest}.py` |
| Conditional sampling, Pareto selection, studies | `hypertrade/optimization/{search_space,pareto,runner}.py` |
| Registry, benchmark orchestration, artifact readers | `hypertrade/experiments/{registry,benchmarks,artifacts}.py` |
| Five views and shared chart/candidate components | `hypertrade/ui/{app,state,views,components}` |
| Supported public calls | `hypertrade/api.py` |

Package initializers contain no implementation. There is one CSV loader, one simulator and one results schema. Domain zones are modules rather than empty class hierarchies.

`evaluate()` prepares technical candidates without reserving capital. Two independent admissions then process the same candidates: Research with no money limit, and Portfolio with fixed position sizing and available capital. Each model updates cooldown only after definitive acceptance. Research and portfolio counts can differ, including later cooldown effects.

All timestamps are explicit UTC instants. Actual entry uses the first available candle open at or after the delayed market entry time. Calendar arithmetic counts market minutes. Events use candle-open time for entries/time exits/gap fills, candle-close time for other exits. Trading-day gaps have no synthetic interpolated prices.

The optimizer's objectives use research or explicitly named portfolio fields. No allocation sizes are optimized. The dashboard reads persisted Pareto candidate trades/equity and metadata rather than running a second analysis engine. Every Pareto candidate has linked artifacts; non-Pareto trials retain metrics/parameters/statuses in the trial table and SQLite study.

No live execution, signal generation, weighted objective optimizer, ML ranking, or portfolio allocation optimizer is included. [Limitations](phase2_deferred.md) and [execution assumptions](simulation.md) are explicit.
