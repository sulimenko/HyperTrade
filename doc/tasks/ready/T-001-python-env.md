# Task T-001: python env

```ai-task-contract
version: 1
task_id: T-001
type: primary
human_summary: "Отказаться от conda-специфичных команд в пользу запуска через стандартный Python, venv и pip install -r requirements.txt"
execution_mode: codex-simple

git:
  base_branch: develop
  queue_branch: ai-task-queue
  parent_branch: none
  work_branch: ai/T-001-python-env
  work_branch_policy: create_task_branch
  allow_new_branch: true
  allow_codex_git: false

scope:
  allowed_files:
    - README.md
    - docs/*.md
    - doc/ai/project-checks.sh
  forbidden_files:
    - doc/tasks/**
    - doc/ai/runs/**
    - doc/ai/context/**
    - doc/ai/review/**
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
  max_files_changed: 4
  max_added_lines: 180
  max_deleted_lines: 80

commit:
  message: "docs(ai): document standard python environment"
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

Убрать conda как официальный путь запуска HyperTrade и заменить документацию на переносимый сценарий: стандартный Python, виртуальное окружение через `venv`, установка зависимостей из `requirements.txt`, запуск CLI и dashboard через `python`.

## Контекст

README уже говорит, что проекту нужен Python environment с зависимостями из `requirements.txt`, но примеры команд используют локальный путь `./.conda/bin/python`. Это привязывает документацию к конкретному workspace и мешает переносимому запуску AI worker, CI или другому разработчику.

Архитектурное решение: conda не запрещается как личная локальная альтернатива разработчика, но официальная документация проекта должна быть conda-neutral. Официальный путь:

1. создать окружение через `python -m venv .venv`;
2. активировать `.venv`;
3. установить зависимости через `python -m pip install -r requirements.txt`;
4. запускать проектные команды через `python`, без `./.conda/bin/python`.

## Affected files

- `README.md`
- `docs/*.md`, только если там есть conda-specific команды или противоречащие инструкции
- `doc/ai/project-checks.sh`, только если нужна минимальная правка для conda-neutral validation

## Suspected cause

n/a

## Exact manual reproduction

n/a

## Expected fix areas

- Обновить раздел Requirements/CLI в `README.md`:
  - добавить setup для чистого Python окружения;
  - заменить примеры `./.conda/bin/python ...` на `python ...`;
  - явно указать, что зависимости устанавливаются из `requirements.txt`.
- Проверить `docs/*.md` на conda-specific команды и заменить их на `python ...`, если такие команды есть.
- Не менять `requirements.txt`: задача про использование существующего списка зависимостей, а не про изменение dependency set.
- Сохранить AI validation conda-neutral: `PYTHON_BIN=python CHECK_MODE=default bash doc/ai/project-checks.sh`.

## Constraints

- Не менять `.env`, secrets, credentials, tokens.
- Не менять `requirements.txt` без отдельного approval.
- Не добавлять новые dependency managers, lock-файлы или packaging migration без отдельного approval.
- Не добавлять Poetry, Pipenv, uv, Conda env-файлы или Docker как обязательный путь запуска.
- Не менять `benchmarks/fixtures/**` без отдельного approval.
- Не коммитить generated runtime artifacts under `artifacts/**` без отдельного approval.
- Не менять artifact schema или metadata contract.
- Не восстанавливать weighted-score optimizer.
- Не добавлять live execution, portfolio allocation, capital sizing, walk-forward validation или native signal generation как основной workflow.
- Не делать unrelated refactoring.
- Не коммитить `doc/tasks/**` и `doc/ai/**` в рабочую ветку.

## Validation commands

```bash
PYTHON_BIN=python CHECK_MODE=default bash doc/ai/project-checks.sh
```

Если документация или validation затронут benchmark workflow, дополнительно выполнить:

```bash
PYTHON_BIN=python CHECK_MODE=benchmark bash doc/ai/project-checks.sh
```

## Acceptance criteria

- В README больше нет официальных команд с `./.conda/bin/python`.
- README содержит переносимый сценарий установки: `python -m venv .venv` и `python -m pip install -r requirements.txt`.
- Основные команды запуска optimization, benchmarks и dashboard используют `python`.
- `requirements.txt` не изменён.
- `doc/ai/project-checks.sh` остаётся conda-neutral и использует `PYTHON_BIN=python` по умолчанию.
- Default validation проходит или в PR явно указана причина, если локальное окружение не содержит установленных зависимостей.
