# AI Pipeline v8.5

T-194 is a single parent task for the approved HyperTrade reorganization, trading/money semantics and five-view research dashboard. Use the installed entrypoint `~/.ai-pipeline-v85/bin/ai-pipeline`. The package itself is outside this checkout's implementation scope.

The lifecycle is inspect → implement/refactor/math/UI/docs → focused checks → independent mandatory CHECK → exact-SHA independent ChatGPT Review → Draft PR. Pipeline owns the commit and delivery. Executor does not run Git history mutations, push, create a PR, merge or deploy. An implementation handoff is not final reviewed delivery.

The former local queue and v8 task scripts/instructions are removed. T-003/T-004 in the separate historical queue branch are superseded by this task and must not execute. This checkout does not reset, normalize or mutate that branch.

New/materially changed behavior stays EXPERIMENTAL. Existing genuine regression assertions remain; three new focused cases cover execution money invariants, admission/portfolio equity, and failed-trial diagnostics. Broad regression hardening belongs to a separately approved Weekly Hardening task.

Independent deterministic checks run `bash doc/ai/project-checks.sh`; use `CHECK_MODE=benchmark` to include real fixtures. Preserve `data/ohlc`, `data/indicators`, `data/signals`, `data/raw`, benchmark fixtures, `.env` and secrets. No external OS changes are part of implementation.
