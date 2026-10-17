# Task T-004: entry-bar exits

```ai-task-contract
version: 1
task_id: T-004
type: primary
human_summary: "Проверять SL/TP и активные stop-механизмы на свече входа, сохраняя сознательные допущения intrabar"
execution_mode: codex-plan

git:
  base_branch: develop
  queue_branch: ai-task-queue
  parent_branch: none
  work_branch: ai/T-004-entry-bar
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
  max_added_lines: 140
  max_deleted_lines: 70

commit:
  message: "fix(simulation): evaluate exits on entry candle"
```

## Task routing

Task type: primary
Base branch: develop
Parent task: none
Parent branch: none
Work branch policy: create-task-branch
Queue branch: ai-task-queue

## Execution mode

codex-plan

## Цель

Сделать возможным закрытие сделки на той же 15-минутной свече, на Open которой произошёл вход. Сейчас симуляция начинает цикл с `entry_idx + 1`, поэтому SL/TP, достигнутые на первой свече, не проверяются и результаты завышаются или искажаются.

## Контекст

В `hypertrade/simulation/trade_simulator.py` вход производится по `ohlc[entry_idx,0]`, но `simulate_trade_core` обходит свечи с `range(entry_idx + 1, ...)`. Пользователь сознательно допускает использование OHLC свечи входа для проверки индикаторов и текущую эвристику определения приоритета SL/TP по цвету бара. Эти два правила не менять.

## Affected files

- `hypertrade/simulation/trade_simulator.py`

## Suspected cause

Инициализация цикла с последующего индекса без отдельной оценки текущей свечи.

## Exact manual reproduction

1. Long: entry Open=100, High=101, Low=94, SL=5%, TP=10% -> выход по SL на entry candle, `exit_idx = entry_idx`, `hold_bars=0`.
2. Short: entry Open=100, High=106, Low=99, SL=5%, TP=10% -> выход по SL на entry candle.
3. Long: entry Open=100, High=111, Low=99, SL=5%, TP=10% -> выход по TP на entry candle.
4. Long: entry Open=100, High=111, Low=94 при одновременно достигнутых SL/TP -> сохранить текущую эвристику `bullish` и existing priority, не заменять на worst-case.
5. Единственная доступная свеча с `entry_idx == last`: если достигается SL/TP, не возвращать автоматически выход по close.
6. Проверить отсутствие двойной обработки свечи и отсутствие использования более ранних свечей для стопа.

## Expected fix areas

- Провести проверку активных на момент входа SL/TP уже на `entry_idx`. Для PSAR/trailing на entry candle явно выбрать и документировать семантику инициализации (не строить несуществующий предыдущий SAR и не использовать максимум бара как цену исполнения до входа); сохранить существующую модель приоритета между механизмами, насколько применима.
- После entry-bar exit не выполнять обработку следующих свечей.
- Сохранить расчёт slippage/commission, exit reason, формат сделки и расчёт времени удержания (`hold_bars=0` разрешён).
- Исправление из T-003: при PSAR exit используется SAR, не EP. Не откатывать его при реализации этой задачи; ветка T-004 создаётся от обновлённого develop после интеграции T-003.
- Не менять намеренно допущения о доступности OHLC свечи входа или текущую эвристику при одновременном SL/TP.

## Constraints

- Не изменять зависимости, данные и артефакты.
- Не переписывать всю симуляцию, кроме минимально необходимого для нового условия.
- Не добавлять полноценный suite; допускаются несколько детерминированных ручных проверок, особенно entry bar и одно-свечная история.
- Создать ветку после интеграции T-003 в develop, чтобы избежать конфликтов в одном модуле. Не запускать одновременно с T-003.

## Validation commands

```bash
PYTHON_BIN=python CHECK_MODE=default bash doc/ai/project-checks.sh
PYTHON_BIN=python CHECK_MODE=benchmark bash doc/ai/project-checks.sh
```

## Acceptance criteria

- SL/TP исполняются на свече входа при достижении уровня.
- Сделка может иметь нулевую длительность в барах и валидную причину выхода.
- Не происходит двойной обработки начальной свечи.
- Сохраняются намеренные допущения о look-ahead и порядке конфликтующих SL/TP.
- Сохранена корректная реализация PSAR из T-003 при интеграции.
- В PR представлены ручные результаты длинного, короткого и одно-свечного сценария и влияние на benchmark, без требования добавлять новый suite.
