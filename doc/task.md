# AI task workflow

This repository uses AI Pipeline v8.

## Worker command

Run one task:

```bash
RUN_ONCE=1 bash $HOME/.codex/ai-pipeline/bin/watch-and-run-tasks.sh
```

Do not use legacy worker paths, aliases, or wrapper variables.

## Queue branch

All task markdown files are created only in:

```text
ai-task-queue:doc/tasks/ready/*.md
```

## Work branches

Worker branches use:

```text
ai/T-XXX-short
```

Branch title must be short: task number plus 1-2 words.

## Task contract

Only this block name is valid:

```text
ai-task-contract
```

`pbull-task-contract` is forbidden.

## Queue policy

Active task files live only in the standard queue directories:

```text
doc/tasks/ready
doc/tasks/in-progress
doc/tasks/review
doc/tasks/done
doc/tasks/failed
```

Completed task history in `doc/tasks/done` must be preserved unless explicitly requested otherwise.

New tasks should inspect `doc/tasks/done` and active task directories to choose the next task number.
