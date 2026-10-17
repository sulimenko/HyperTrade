# Task T-003: psar exit

```ai-task-contract
version: 1
task_id: T-003
type: primary
human_summary: "Исправить цену исполнения PSAR exit: использовать уровень SAR вместо extreme point"
execution_mode: codex-simple

git:
  base_branch: develop
  queue_branch: ai-task-queue
  parent_branch: none
  work_branch: ai/T-003-psar-exit
  work_branch_policy: create_task_branch
  allow_new_branch: true
  allow_codex_git: false

scope:
  allowed_files:
    - hypertrade/simulation/trade_simulator.py
  forbidden_files:
    - doc/tasks/**
    - doc/ai/**
    - .env
    - requirements.txt
    - benchmarks/fixtures/**
    - artifacts/**
    - data/**
    - results/**

validation:
  commands:
    - PYTHON_BIN=python CHECK_MODE=default bash doc/ai/project-checks.sh

benchmark_validation:
  commands:
    - PYTHON_BIN=python CHECK_MODE=benchmark bash doc/ai/project-checks.sh

diff_budget:
  max_files_changed: 1
  max_added_lines: 70
  max_deleted_lines: 35

commit:
  message: "fix(simulation): execute PSAR exit at SAR level"
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

Исправить ошибку расчёта цены выхода по Parabolic SAR для long/short. При пробое PSAR использовать текущий рассчитанный уровень `psar` с существующей моделью учёта gap-open, а не extreme point (`ep`). Не менять стратегию расчёта индикаторов или логику приоритета внутри свечи.

## Контекст

В `hypertrade/simulation/trade_simulator.py`, в `_psar_update_and_check`, при срабатывании long сейчас возвращается `min(ep, o)`, а при short — `max(ep, o)`. `ep` отражает экстремум тренда и не является стоп-уровнем PSAR. Должен использоваться рассчитанный на этом баре `psar` (после ограничения предыдущими high/low).

## Affected files

- `hypertrade/simulation/trade_simulator.py`

## Suspected cause

Перепутаны переменные `ep` и `psar` при формировании цены исполнения стопа после пересечения SAR.

## Exact manual reproduction

1. В Numba-совместимом `_psar_update_and_check` смоделировать long с актуальным `psar = 98`, `ep = 110`, `open = 100`, `low = 97` и без срабатывания clamp, меняющего SAR. Существующая формула `min(ep, open)` даёт исполнение около 100 вместо SAR около 98 (при разрыве вниз — по более неблагоприятной цене открытия).
2. Для short проверить аналогичный случай `psar = 102`, `ep = 90`, `open = 100`, `high = 103`: ожидается выход около 102, не около 100.
3. Проверить обычное срабатывание, gap-through-SAR и отсутствие срабатывания по обеим сторонам. Проверку выполнять в активном стандартном Python `.venv`, не в Conda.

## Expected fix areas

- Использовать `min(psar, o)` для long exit и `max(psar, o)` для short exit, если текущие правила заполнения стопа и clamp это подтверждают.
- Сохранить существующий порядок проверок `time_exit -> psar -> trailing -> sl/tp`.
- Сохранить семантику `EXIT_REASON[3] = "psar"`, формат возвращаемой сделки, slippage и commission.
- Не менять допущения о доступности OHLC свечи входа и порядок SL/TP внутри свечи: они намеренно приняты пользователем.
- Провести 2-4 ручные детерминированные проверки через Python/Numba по описанным сценариям; новый полноценный тестовый suite не создавать.

## Constraints

- Никаких изменений `.env`, секретов, зависимостей, `requirements.txt`, benchmark fixtures, runtime data и artifact schema.
- Не вводить новую модель intrabar path, walk-forward, live execution, weighted-score или portfolio sizing.
- Не делать unrelated refactoring.
- Работать только в ветке `ai/T-003-psar-exit`; файлы очереди задач не включать в PR.

## Validation commands

```bash
PYTHON_BIN=python CHECK_MODE=default bash doc/ai/project-checks.sh
PYTHON_BIN=python CHECK_MODE=benchmark bash doc/ai/project-checks.sh
```

В PR записать результат проверок обычного выхода и gap-through-SAR для long/short. Если окружение не подготовлено, честно сообщить о блокере и не заявлять об успешном benchmark.

## Acceptance criteria

- Выход long/short по PSAR использует действительный уровень SAR с учётом текущей gap-модели, а не EP.
- При неизменных данных и параметрах результат воспроизводим.
- Иные режимы SL/TP, trailing, time exit и намеренные допущения не изменены.
- Изменён только разрешённый production-файл; ручные проверки и известные ограничения описаны в PR.
