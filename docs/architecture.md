# Architecture and Scope

## Purpose

HyperTrade phase 1 is a research-lab for improving existing trading signals by optimizing filters and acceptance rules.

The primary workflow is:

1. load external signal CSVs;
2. hydrate market data and indicators;
3. apply filter and acceptance policies;
4. simulate trades;
5. compute metrics;
6. run a multi-objective Optuna study;
7. persist one artifact bundle per run;
8. analyze results in the dashboard.

## Active Package Layout

- `hypertrade/data`
  - external signal loading
  - market-data access
  - indicator store access
- `hypertrade/features`
  - indicator hydration and feature preparation
- `hypertrade/signals`
  - filter parameter search space
  - indicator filter definitions
  - signal acceptance policy
- `hypertrade/simulation`
  - market-time handling
  - trade simulation
  - run metrics
- `hypertrade/optimization`
  - Optuna study runner
  - Pareto candidate extraction and ranking
- `hypertrade/experiments`
  - experiment registry
  - benchmark harness
- `hypertrade/reporting`
  - run and suite loading for analytics
- `hypertrade/ui`
  - Streamlit app and pages

Supported root entrypoints:

- `run_optimize.py`
- `run_benchmarks.py`
- `run_dashboard.py`
- `streamlit_app.py`

## Core Product Decisions

- weighted-score optimization is removed;
- optimization is true multi-objective with Pareto output;
- v1 works on existing external signals only;
- backward compatibility with legacy result layout is not required;
- walk-forward is explicitly deferred to phase 2;
- no portfolio layer in phase 1.

## Signal Optimization Scope

Phase 1 optimizes:

- indicator-based filters;
- time-gated acceptance windows;
- cooldown policy;
- confirmation bars;
- delay-open policy;
- ranking and per-side signal caps.

It does not yet implement native signal generation as the main product path.
