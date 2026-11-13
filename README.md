# HyperTrade Research

EXPERIMENTAL external CSV signal research on CPython 3.14. Compare an unfiltered robot baseline with filtered candidates using multi-objective Optuna / NSGA-II and a Pareto frontier. Results separate fixed-nominal research quality from capacity-limited portfolio equity.

## Install

Use a CPython 3.14 interpreter. On macOS with Homebrew installed:

```sh
/opt/homebrew/opt/python@3.14/bin/python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
```

On other platforms, substitute `python3.14 -m venv .venv` and the platform's virtual-environment Python path. Runtime and transitive dependencies are pinned in `requirements.txt`; `requirements-dev.txt` adds pytest. Installation requires access to PyPI. No system package upgrade or removal is performed by the project.

```sh
.venv/bin/python run_optimize.py --signals benchmarks/fixtures/PF20250597.csv --n_trials 20
.venv/bin/python run_benchmarks.py --n_trials 1
.venv/bin/python run_dashboard.py
bash doc/ai/project-checks.sh
CHECK_MODE=benchmark bash doc/ai/project-checks.sh
```

`PYTHON_BIN` can override the interpreter in checks; it must be Python 3.14. Both CLIs print the artifact directory; failed runs/suites exit nonzero. `no_valid_candidates` is a valid research outcome and exits zero.

## Data and models

CSV is semicolon-separated, with `datetime`, `long_symbols` (or `symbols`) and optional `short_symbols`. The historical fixture spelling `shotr_symbols` is also read. ISO and `dd.mm.yyyy` dates are accepted. Naive dates explicitly mean UTC; timezone offsets preserve their instant. Equal timestamps are grouped, symbols deduplicated per side, and signals sorted. Inputs are never rewritten.

Research runs read local `data/ohlc/*.parquet` and `data/indicators/*.parquet`, preferring explicitly prepared histories in `data/cache/market`. They never download or overwrite source data. Missing/empty instruments or history starting after the signal fail the run before trials, with actionable `market_data_preparation` diagnostics rather than a misleading filter outcome. Prepare new or incomplete instruments explicitly:

```sh
# Configure DATA_BASE_URL in the process environment for the existing market API.
.venv/bin/python -m hypertrade.data.market --signals path/to/external.csv
```

Preparation uses the existing US `symbolSearch` / `addChartSymbols` API contract, validates coverage and stages parquet only in `data/cache/market`. Complete local histories need no API or network. API failures/unrecognized symbols fail preparation visibly; no new credentials are required. Existing histories ending before a later signal retain explicit `rejected_no_candles_after_entry` details (including historical delisted instruments in fixtures). The preparation command checks the full CSV signal span; it cannot invent missing history. End-of-history liquidation remains marked `data_truncated` if a trade's requested holding period extends beyond available bars.

Additional indicator columns go into `data/cache/indicators`; indicators for separately prepared histories are computed from those histories without reusing stale source columns. Uploaded CSVs go into `data/cache/uploads`. The existing RTH-only source uses UTC-day VWAP unchanged. Full indicator-cache invalidation is deferred.

- **Research / Normalized:** unlimited budget, identical $10,000 nominal for baseline and candidates, fractional analytical exposure. `sum_trade_returns_pct` is a sum of percentage points, **not** a return on capital. Net PnL, mean/median trade return, win rate, profit factor and a deterministic sequential proxy drawdown are available.
- **Capital / Portfolio:** $100,000 initial equity, maximum $10,000 gross notional per position, affordable integer shares, entry/exit commission in USD/share/side and separate directional fractional slippage. Long notional and 100% short collateral are reserved; short proceeds remain locked. Time equity includes realized and unrealized PnL; drawdown is a percentage of peak equity.

Every study saves a fixed baseline without indicator filters, time gate, ranking, cap or cooldown. Its SL/TP/delay/holding settings use the configured search minima and are persisted. Candidate improvements can include both filters and exit tuning. Both models use the same costs and nominal settings. See [simulation semantics](doc/simulation.md).

## Исследовательский интерфейс

Запуск: `.venv/bin/python run_dashboard.py`. Streamlit и Plotly сохраняют тёмную тему; пять разделов по умолчанию на русском:

1. **Обзор исследований** (Research Home): сохранённые исследования и статусы исходной стратегии.
2. **Новое исследование** (New Experiment): CSV/загрузка, основные настройки по умолчанию, 20/100/200 или 1…10 000 испытаний, три профиля целей, SL/TP и EMA/RSI/ATR. Расширенная группа содержит цели/направления, ограничения, seed, издержки, ранжирование и JSON. Переключение режима сохраняет их общий профиль; окончательное резюме показывает действующие ограничения и seed.
3. **Обзор оптимизации** (Optimizer Explorer): 3D/2D Парето, оси/цвет, статусы испытаний, выбор точкой/таблицей/списком и диагностика.
4. **Анализ кандидата** (Candidate Analysis): исходная стратегия и до трёх кандидатов, исследовательские и портфельные кривые, сделки и причины отказов.
5. **Сравнение и проверка** (Compare & Validation): сохранённые результаты, сопоставимость профилей и короткая проверка по одному испытанию на контрольный CSV.

Чтобы начать, выберите CSV и профиль, задайте диапазоны, прочитайте резюме и нажмите «Запустить исследование». Подсказки у полей объясняют единицы и ограничения. Полные профили сохраняются в `data/cache/profiles/`; старые профили целей читаются без миграции исходного файла. Визуальный редактор и полный JSON выбираются явно, проверяются перед запуском и не перетирают друг друга.

Краткий глоссарий: **исходная стратегия** — Baseline без фильтров, с минимальными процентными SL/TP; **испытание** — Trial Optuna; **допустимые результаты** — COMPLETE; **отклонено по ограничениям** — PRUNED; **ошибка расчёта** — FAIL. Сумма доходностей сделок измеряется в процентных пунктах, не является доходностью капитала. PF — отношение положительных доходностей к модулю отрицательных; CVaR 5% — средний убыток худших 5% сделок. Доходность портфеля — % от начального капитала, просадка — % от достигнутого пика. Комиссия — USD/акция/сторона; проскальзывание — доля цены на сторону; ATR SL/TP — множители ATR, не проценты.

Изменения T-256 **ЭКСПЕРИМЕНТАЛЬНЫ** до отдельно одобренного Weekly Hardening. Это перевод отображения: исходные метрики, состояния, JSON-ключи, расчёты и прошлые результаты не меняются. Риск **lookahead** сохраняется: фильтры и ранжирование могут использовать OHLC текущей 15-минутной свечи при входе на её открытии. Проверки на будущих окнах и отложенных данных (walk-forward/OOS) нет; историческая прибыль не обещает будущих результатов. Подробности, прежние названия разделов, профили и графики — в [руководстве](doc/ui_guide.md).

## Development and delivery

Responsibilities and artifact schema: [architecture](doc/architecture.md), [artifacts](doc/artifacts.md), [Pareto objectives](doc/objectives_and_pareto.md). There is one public API in `hypertrade/api.py` and no alternate historical results reader.

T-194 is one AI Pipeline v8.5 task. Use the installed `~/.ai-pipeline-v85/bin/ai-pipeline`; local legacy task queues/scripts were removed. Independent checks, review of the exact Pipeline-owned commit, and Draft PR are required before delivery. Executor never commits, pushes, merges or deploys. Old queue tasks T-003/T-004 are superseded and must not run. See [pipeline](doc/ai/pipeline.md).

T-197 сравнение по календарному времени закрытия с общим начальным нулём и конечным плато сохранено. Шкала по порядку сделок остаётся доступной; значения капитала, исходная стратегия и цели оптимизатора не меняются. См. [руководство](doc/ui_guide.md).
