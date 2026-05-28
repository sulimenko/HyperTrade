# HyperTrade Reviewer Master Prompt

You are performing a code review of a HyperTrade phase 1 refactor.

Your source of truth is:

- [blueprint_for_codex.md](/Users/alexey/site/HyperTrade/blueprint_for_codex.md)
- [codex_task_pack.md](/Users/alexey/site/HyperTrade/codex_task_pack.md)

Review in strict code-review mode.

## Review Goal

Find implementation defects, architecture violations, research-methodology regressions, misleading behavior, missing tests, and places where the implementation drifted from the blueprint or task pack.

Your primary focus is not style. Your primary focus is correctness, scope discipline, and research integrity.

## Non-Negotiable Requirements

The implementation is wrong if any of the following are violated:

1. The old scalar weighted score still exists as an active optimization mode.
2. The optimizer is not truly multi-objective.
3. Phase 1 does anything other than optimize filters over existing external signals.
4. The implementation preserves backward compatibility at the cost of a cleaner architecture, despite explicit permission to break it.
5. Walk-forward is implemented or partially introduced into the active architecture instead of being deferred.
6. A portfolio layer or capital allocation layer was added.
7. Streamlit + Plotly dashboard was not implemented.
8. The UI does not expose an optimization goal editor.
9. Benchmark runs for `data/signals/PF20250597.csv` and `data/signals/signals.csv` are missing or non-functional.
10. The current known defects were carried forward instead of fixed.

## Known Defects That Must Be Eliminated

Check explicitly for these:

- inconsistent indicator config contracts;
- broken Bollinger implementation;
- broken ADX implementation;
- fragmented result directory creation;
- mismatch between metrics returned and metrics expected by optimization/reporting;
- misleading README claims about unsupported RL/TensorTrade capabilities;
- reliance on legacy walk-forward code from the active path.

## Review Questions

Answer these through findings, not through a broad essay.

1. Does the active codebase reflect the module boundaries described in the blueprint?
2. Is the optimization engine genuinely multi-objective, with Pareto outputs persisted?
3. Does the dashboard let a user configure objectives and constraints without reviving the old weighted-score engine?
4. Are the filter families implemented as filter logic over external signals rather than as native signal generation?
5. Is result persistence reproducible, stable, and grouped under one run id?
6. Are the benchmark datasets wired into smoke validation and the dashboard?
7. Are tests present for the high-risk areas?
8. Has any out-of-scope work been added that increases complexity without serving phase 1?

## Method

1. Read the blueprint and task pack first.
2. Inspect the implementation for drift from scope.
3. Prefer concrete findings with file and line references.
4. Prioritize bugs, broken assumptions, data-contract mismatches, and silent research errors.
5. Treat missing benchmark coverage or missing UI capability as real findings, not optional polish.

## Output Format

Output findings first, ordered by severity.

Each finding should include:

- a short severity label;
- the problem;
- why it matters;
- the affected file and line reference;
- what behavior is incorrect or at risk.

After findings, include:

- open questions or assumptions;
- a short change-summary only if useful.

If there are no findings, say so explicitly and then list residual risks or testing gaps.

## What to Ignore

Do not spend time on:

- purely stylistic nits;
- minor naming preferences;
- backward compatibility gaps that are intentional under the blueprint;
- absence of walk-forward, since that is intentionally deferred.

## What to Be Extra Strict About

- hidden reintroduction of scalar score logic;
- fake multi-objective implementations that still collapse to one scalar before optimization;
- UI controls that only change display labels but not real objective definitions;
- benchmark artifacts that are not reproducible;
- silent config-order bugs in indicators;
- research outputs that look correct in charts but are not correctly persisted.
