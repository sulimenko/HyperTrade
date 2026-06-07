# AI project settings

## AI Pipeline v8

This repository uses AI Pipeline v8 with the shared local worker:

```bash
$HOME/.codex/ai-pipeline/bin/watch-and-run-tasks.sh
```

Only this path is supported. Do not document or use legacy worker paths, aliases, or wrapper variables.

## Branches

- Default/base branch: `develop`.
- Queue branch: `ai-task-queue`.
- Work branches: `ai/T-XXX-short-title`.
- Branch title must be short: task number plus 1-2 words.

Examples:

```text
ai/T-001-docs
ai/T-002-checks
ai/T-003-ui
```

## Task IDs

Task IDs use the `T-XXX` format in every repository.

Before creating a new task, inspect existing task files in:

```text
doc/tasks/ready
doc/tasks/in-progress
doc/tasks/review
doc/tasks/done
doc/tasks/failed
```

Create the next incremental task number. If follow-up suffixes are already used, continue with the next suffix, for example `T-041A`, `T-041B`.

## Task contract block

Every task must contain exactly one machine-readable contract block named:

```text
ai-task-contract
```

The old `pbull-task-contract` name is forbidden and must not be used for new tasks. Compatibility aliases are not allowed.

## Queue rules

Tasks are created only in `ai-task-queue` under:

```text
doc/tasks/ready/*.md
```

The worker may move task files between:

```text
doc/tasks/ready
doc/tasks/in-progress
doc/tasks/review
doc/tasks/done
doc/tasks/failed
```

Production work branches must not contain task queue files or AI workflow artifacts.

Forbidden in work branches:

```text
doc/tasks/**
doc/ai/runs/**
doc/ai/context/**
doc/ai/review/**
diff.patch
diff-stat.txt
CHATGPT_REVIEW_REQUEST.md
commits.txt
status.txt
```

The directories below are local-only and must not contain tracked files, including `.gitkeep`:

```text
doc/ai/runs
doc/ai/context
doc/ai/review
```

## GitHub connector rules

Use GitHub connector actions only for coarse-grained operations:

- fetch PR metadata;
- fetch PR patch or per-file patch;
- create/update a complete UTF-8 markdown task file;
- update PR body;
- create PR.

Avoid many small range reads. If a PR was closed after force-push/recreated branch and cannot be reopened, create a new PR from the current branch to `develop`.

Task markdown must be plain UTF-8 markdown. Do not use binary, zip, gzip, base64, or encoded payloads for tasks.

## Review policy

ChatGPT Architect must not submit GitHub `APPROVE` on its own PR. Use a written architectural review conclusion instead:

```text
Approved with notes
```

or:

```text
Blocked
```

Only real production/test/documentation gaps should block review. Outdated PR body is a note unless it can mislead deployment or review.

## PR body requirements

Every PR should include:

- Summary;
- Validation commands and results;
- Manual checks, if performed;
- Known limitations;
- Review notes, if relevant.

## Safety constraints

Do not change without explicit approval:

- `.env`;
- secrets, credentials, tokens;
- production config;
- dependencies, including `requirements.txt`;
- benchmark fixtures under `benchmarks/fixtures/**`;
- generated runtime artifacts under `artifacts/**`;
- artifact schema or metadata contract;
- unrelated code;
- unrelated refactoring.

Do not inspect or modify:

- `vendor`;
- `node_modules`;
- `.git`;
- build/cache/storage artifacts.

## HyperTrade product constraints

HyperTrade phase 1 is a Python research lab for optimizing filters over existing external trading signals.

Do not implement without explicit approval:

- weighted-score optimizer restoration;
- native signal generation as the main workflow;
- walk-forward validation;
- portfolio allocation or capital sizing;
- live execution;
- changes that break the one-run-one-bundle artifact contract;
- changes that make `no_valid_candidates` a benchmark-runner failure for datasets where this is a valid research outcome.

## Validation policy

Default validation:

```bash
PYTHON_BIN=python CHECK_MODE=default bash doc/ai/project-checks.sh
```

Benchmark validation when required:

```bash
PYTHON_BIN=python CHECK_MODE=benchmark bash doc/ai/project-checks.sh
```

`CHECK_MODE=benchmark` additionally runs:

```bash
python run_benchmarks.py --n_trials 1
```

## Repository profile

Repository: `HyperTrade`
Project type: Python research lab / Optuna multi-objective optimization / Streamlit dashboard
