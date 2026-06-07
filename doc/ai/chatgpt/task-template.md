# Task T-XXX: short title

```ai-task-contract
version: 1
task_id: T-XXX
type: primary
human_summary: "Короткое описание задачи"
execution_mode: codex-simple

git:
  base_branch: develop
  queue_branch: ai-task-queue
  parent_branch: none
  work_branch: ai/T-XXX-short
  work_branch_policy: create_task_branch
  allow_new_branch: true
  allow_codex_git: false

scope:
  allowed_files:
    - path/to/file.ext
  forbidden_files:
    - doc/tasks/**
    - doc/ai/**
    - .env
    - requirements.txt
    - benchmarks/fixtures/**
    - artifacts/**

validation:
  commands:
    - PYTHON_BIN=python CHECK_MODE=default bash doc/ai/project-checks.sh

benchmark_validation:
  commands:
    - PYTHON_BIN=python CHECK_MODE=benchmark bash doc/ai/project-checks.sh

diff_budget:
  max_files_changed: 5
  max_added_lines: 300
  max_deleted_lines: 120

commit:
  message: "feat(ai): implement T-XXX short"
```

## Task routing

Task type: primary
Base branch: develop
Parent task: none
Parent branch: none
Work branch policy: create-task-branch
Queue branch: ai-task-queue

## Execution mode

codex-simple

## Цель

Опиши цель.

## Контекст

Опиши контекст.

## Affected files

- `path/to/file.ext`

## Suspected cause

Для bugfix-задач опиши предполагаемую причину. Для feature/refactor задач укажи `n/a`.

## Exact manual reproduction

Для bugfix-задач опиши точное ручное воспроизведение. Для feature/refactor задач укажи `n/a`.

## Expected fix areas

- Опиши ожидаемые области изменений.

## Constraints

- Не менять `.env`, secrets, credentials, tokens.
- Не добавлять и не менять dependencies, включая `requirements.txt`, без отдельного approval.
- Не менять `benchmarks/fixtures/**` без отдельного approval.
- Не коммитить generated runtime artifacts under `artifacts/**` без отдельного approval.
- Не менять artifact schema или metadata contract без отдельного approval.
- Не восстанавливать weighted-score optimizer.
- Не добавлять live execution, portfolio allocation, capital sizing, walk-forward validation или native signal generation как основной workflow без отдельного approval.
- Не делать unrelated refactoring.
- Не коммитить `doc/tasks/**` и `doc/ai/**` в рабочую ветку.

## Validation commands

```bash
PYTHON_BIN=python CHECK_MODE=default bash doc/ai/project-checks.sh
```

Для задач, которые затрагивают optimizer, benchmark flow, artifact registry, dashboard benchmark loading или signal simulation, дополнительно выполнить:

```bash
PYTHON_BIN=python CHECK_MODE=benchmark bash doc/ai/project-checks.sh
```

## Acceptance criteria

- Критерий 1.
- Критерий 2.
