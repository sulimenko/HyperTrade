# Task T-002: doc modules

```ai-task-contract
version: 1
task_id: T-002
type: primary
human_summary: "Перенести старую документацию из docs в doc, добавить описание модулей проекта и зафиксировать команды запуска без conda"
execution_mode: codex-simple

git:
  base_branch: develop
  queue_branch: ai-task-queue
  parent_branch: none
  work_branch: ai/T-002-doc-modules
  work_branch_policy: create_task_branch
  allow_new_branch: true
  allow_codex_git: false

scope:
  allowed_files:
    - README.md
    - .gitignore
    - doc/*.md
    - doc/modules/*.md
    - docs/*.md
  forbidden_files:
    - doc/tasks/**
    - doc/ai/**
    - .env
    - requirements.txt
    - benchmarks/fixtures/**
    - artifacts/**
    - results/**
    - data/**
    - .conda/**
    - .venv/**

validation:
  commands:
    - PYTHON_BIN=python CHECK_MODE=default bash doc/ai/project-checks.sh

benchmark_validation:
  commands:
    - PYTHON_BIN=python CHECK_MODE=benchmark bash doc/ai/project-checks.sh

diff_budget:
  max_files_changed: 16
  max_added_lines: 500
  max_deleted_lines: 500

commit:
  message: "docs(ai): move docs and document modules"
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

Привести документацию HyperTrade к единому расположению в `doc/`, перенести прежние материалы из `docs/`, добавить описание основных модулей проекта в `doc/modules/` и зафиксировать команды запуска проекта без conda.

## Контекст

В проекте сейчас есть старая документация в папке `docs/`, а AI Pipeline v8 и пользовательские инструкции используют `doc/`. Нужно убрать раздвоение `docs`/`doc`, чтобы вся проектная документация находилась в одном месте.

Также после отказа от conda официальный путь запуска должен быть стандартным Python окружением:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_optimize.py --signals benchmarks/fixtures/PF20250597.csv --n_trials 25 --benchmark_name PF20250597 --run_label PF20250597_manual
python run_benchmarks.py --n_trials 1
python run_dashboard.py
```

Локальная папка `.conda/` уже указана в `.gitignore`, поэтому она не должна коммититься. Удаление `.conda/` является локальной cleanup-операцией на машине разработчика, а не tracked изменением репозитория.

## Affected files

- `README.md`
- `doc/*.md`
- `doc/modules/*.md`
- `docs/*.md`
- `.gitignore`, только если нужно уточнить игнорирование локальных окружений

## Suspected cause

n/a

## Exact manual reproduction

n/a

## Expected fix areas

- Перенести содержимое старой папки `docs/` в `doc/`:
  - `docs/architecture.md` -> `doc/architecture.md`
  - `docs/artifacts.md` -> `doc/artifacts.md`
  - `docs/benchmarks.md` -> `doc/benchmarks.md`
  - `docs/objectives_and_pareto.md` -> `doc/objectives_and_pareto.md`
  - `docs/ui_guide.md` -> `doc/ui_guide.md`
  - `docs/phase2_deferred.md` -> `doc/phase2_deferred.md`
- Обновить ссылки в `README.md` с `docs/...` на `doc/...`.
- После переноса удалить старые markdown-файлы из `docs/`, чтобы не осталось двух источников истины.
- Создать папку `doc/modules/` и добавить описание основных модулей HyperTrade. Минимальный набор:
  - `doc/modules/README.md` с навигацией;
  - `doc/modules/data.md` для `hypertrade/data`;
  - `doc/modules/features.md` для `hypertrade/features`;
  - `doc/modules/signals.md` для `hypertrade/signals`;
  - `doc/modules/simulation.md` для `hypertrade/simulation`;
  - `doc/modules/optimization.md` для `hypertrade/optimization`;
  - `doc/modules/experiments.md` для `hypertrade/experiments`;
  - `doc/modules/reporting.md` для `hypertrade/reporting`;
  - `doc/modules/ui.md` для `hypertrade/ui`.
- В `README.md` добавить или уточнить список команд запуска без conda:
  - setup `.venv`;
  - установка зависимостей;
  - запуск optimization;
  - запуск benchmark suite;
  - запуск dashboard.
- Не добавлять обязательную поддержку conda, Poetry, Pipenv, uv, Docker или lock-файлы.
- Не пытаться удалить `.conda/` через git: это локальная ignored директория. В документации можно дать ручную команду для пользователя.

## Constraints

- Не менять `.env`, secrets, credentials, tokens.
- Не менять `requirements.txt` без отдельного approval.
- Не менять `benchmarks/fixtures/**` без отдельного approval.
- Не коммитить generated runtime artifacts under `artifacts/**`, `results/**` или `data/**`.
- Не коммитить `.conda/**` или `.venv/**`.
- Не менять artifact schema или metadata contract.
- Не восстанавливать weighted-score optimizer.
- Не добавлять live execution, portfolio allocation, capital sizing, walk-forward validation или native signal generation как основной workflow.
- Не делать unrelated refactoring.
- Не коммитить `doc/tasks/**` и `doc/ai/**` в рабочую ветку.

## Validation commands

```bash
PYTHON_BIN=python CHECK_MODE=default bash doc/ai/project-checks.sh
```

Если изменения затронут benchmark documentation только текстово, benchmark validation не обязательна. Если будет изменён benchmark runner или runtime code, дополнительно выполнить:

```bash
PYTHON_BIN=python CHECK_MODE=benchmark bash doc/ai/project-checks.sh
```

## Acceptance criteria

- Вся старая проектная документация из `docs/*.md` перенесена в `doc/*.md`.
- В `README.md` ссылки ведут на `doc/...`, а не на `docs/...`.
- Старые markdown-файлы в `docs/` удалены или папка `docs/` больше не используется как источник документации.
- Создана папка `doc/modules/` с описанием основных модулей HyperTrade.
- README содержит полный список команд запуска без conda через стандартный Python, `.venv` и `pip install -r requirements.txt`.
- `.conda/` не попадает в git и остаётся локальной ignored директорией.
- `requirements.txt` не изменён.
- Default validation проходит или в PR явно указана причина, если локальное окружение не содержит установленных зависимостей.
