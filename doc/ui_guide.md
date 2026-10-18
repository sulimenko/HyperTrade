# Research terminal · EXPERIMENTAL

Start `.venv/bin/python run_dashboard.py`. Five views share the experiment selector, a dark Plotly theme and session candidate selection. New T-197 behavior remains EXPERIMENTAL until separately approved Weekly Hardening.

## Research Home

Browse studies and baseline status. `no_valid_candidates` is a valid research outcome, not a technical failure.

## New Experiment

Выберите CSV fixture, существующий путь или загрузите CSV. Даты без часового пояса означают UTC. Исходные файлы и прошлые bundles не изменяются.

Trial presets: **20** — smoke/research test, **100** и **200** — обычные исследования; **Вручную** — 1…10 000. В предпросмотре видно точное `n_trials`, которое получит engine. Benchmark smoke в Compare & Validation по-прежнему запускает **1 trial на fixture**.

**Профиль поиска** включает objectives, directions, constraints, shortlist policy, полный search space, costs, trial preset, n_trials и seed. Это не новая формула доходности. Текущий default «Качество сигналов» остаётся первым вариантом; необязательные шаблоны предлагают одну цель доходности капитала или баланс доходности капитала, портфельной просадки и исследовательского PF. Шаблоны сохраняют исходные ограничения, которые можно полностью редактировать.

При выборе профиля или «Восстановить сохранённые поля» восстанавливаются все его поля. Именованные профили сохраняются в **`data/cache/profiles/*.json`**, уже исключённом из Git правилом `/data/`. Повторное сохранение того же имени заменяет этот профиль; для отдельного варианта задайте другое имя. Старые objective-only `config/defaults/profiles/*.json` доступны как Legacy: недостающие настройки берутся из default. Сохранение переносит полный профиль в runtime-каталог, сохраняя legacy-файл. Имена: 1–64 латинских буквы/цифры/underscore/hyphen, начало — буква/цифра. Смена профиля восстанавливает редакторы, а не меняет существующие эксперименты.

**Источник search_payload** выбирается явно:

- «Визуальные поля»: SL/TP min/max/step в процентах, ATR min/max/step в множителях ATR, фильтры, ranking и costs. В «Расширенных параметрах» JSON содержит остальные параметры; дублировать визуальные поля запрещено.
- «Полный JSON»: только полный JSON передаётся в engine, визуальные поля отключены. Редакторы независимы; при смене режима применяются значения выбранного редактора.

Окончательная конфигурация раскрывается перед запуском и сохранением. Проверяются типы, конечные числа, диапазоны, положительный step и делимость `(max−min)/step`, чтобы Optuna не сокращал верхнюю границу молча. Комиссия — USD/share/side, slippage — доля/side. ATR-кандидат использует ATR-диапазоны, baseline всегда использует процентные минимумы SL/TP.

## Действующие цели и формулы

Default: maximize `sum_trade_returns_pct`, minimize `proxy_drawdown_usd`, maximize `profit_factor`, minimize `avg_hold_minutes`; жёсткие ограничения `trades >= 20`, `profit_factor >= 1.05`. Сумма доходностей сделок — процентные пункты, не доходность капитала. Proxy drawdown сохраняет порядок входов sequential-артефакта.

PF = сумма положительных доходностей сделок / модуль суммы отрицательных. Это исследовательский PF; без убытков он undefined. Undefined-цели и нарушенные ограничения дают PRUNED.

Pareto не использует weighted-score. После отбора недоминируемых кандидатов shortlist по умолчанию сортируется по `utopia_distance = sqrt(sum(normalized_deviation²))`: отклонения от лучших значений нормализованы по диапазону Pareto каждой цели, веса равны в нормализованном пространстве. Это вторичная сортировка, не основная оптимизация. Другие shortlist policies меняют предпочтение внутри Pareto.

Направления `maximize`/`minimize` и ограничения min/max редактируются явно. Default **не оптимизирует portfolio_return_pct**; её и `portfolio_max_drawdown_pct` можно выбрать отдельными целями. Одна цель поддерживается установленным NSGA-II без смены sampler; одномерный Pareto содержит лучшее допустимое значение.

## Optimizer Explorer

3D по умолчанию: X — portfolio drawdown (% peak equity), Y — portfolio return (% initial equity), Z — research profit factor; цвет — average hold (минуты), если доступны. Метрики можно выбирать вручную; при отсутствии метрики используется доступная числовая. 3D поддерживает rotation/zoom; выбор кандидата — selector/table, в 2D также point/box/lasso.

Первый кандидат берётся из `candidate_shortlist.parquet`, с fallback на первый Pareto. Ручной выбор и отдельный список до трёх сравниваемых trials сохраняются между страницами текущей сессии. Исторический shortlist не перезаписывается. Полный session shortlist также остаётся доступным.

После запуска и в Explorer показаны число Pareto-кандидатов на min/max и таблица по числовым sampled-параметрам. Источник границ — фактические сохранённые Optuna distributions, fallback — search_space.json. Условные параметры учитываются только в trials, где они выбирались. Фиксированные диапазоны не вызывают предупреждений. Для SL/TP единицы %, для ATR SL/TP — × ATR. Если ≥50% кандидатов с изменяемым параметром достигли границы, предлагается рассмотреть расширение диапазона. Это сигнал ограниченности исследования, **не доказательство**, что экстремумы или значения за границей лучше.

All trials показывает отдельные COMPLETE/PRUNED/FAIL, таблицу и диаграмму реальных PRUNED-причин. COMPLETE и Pareto различаются: Pareto — недоминируемое подмножество допустимых COMPLETE. Один PRUNED может иметь несколько причин: `<min`, `>max`, `missing:metric` (ограничение без значения), `undefined:metric` (неопределённая цель), `no_trades`. Отсутствующие причины честно отмечаются. FAIL означает техническое исключение со stack trace в Diagnostics; `no_valid_candidates` означает отсутствие допустимых кандидатов.

## Candidate Analysis

Research по умолчанию показывает **календарное время закрытия в UTC**, baseline и до трёх trials. Накопление берётся из сохранённых `normalized_pnl_usd`; legacy fallback — `return_pct × normalized_notional / 100`. PnL признаётся в **exit_event_dt**; если поле/значение отсутствует — в `exit_dt` (timestamp свечи, менее точный legacy fallback). Отсутствие обеих дат или корректного PnL даёт ошибку вместо пропущенных сделок.

Общий период включает все сохранённые исследуемые сигналы и закрытия, включая остальные Pareto-кандидаты. На объединённых timestamps у каждой линии одинаковые начало с нуля и конец, между закрытиями последнее значение переносится вперёд. Одновременные закрытия суммируются однократно; ступенчатая линия не интерполирует выдуманную прибыль. Разные 1659 и 360 сделок теперь имеют одну календарную область. Исходный sequential parquet не меняется.

«По порядку сделок» сохраняет прежние PnL/proxy drawdown в порядке входов; X — номер допущенной сделки. Разная длина здесь отражает разное число сделок, а не короткий backtest. Sequential proxy drawdown в метриках не подменяется просадкой календарной кривой.

Capital / Portfolio equity показывает исходные mark-to-market и drawdown, включая все сохранённые фазы одного timestamp и однократный учёт закрытий. Изменяется только общая область X: пересчёта, reindex и синтетических equity-точек нет.

На одном экране: период сигналов, общий период, количество исходных сигналов, допущенные research/portfolio сделки и доля принятых для baseline и кандидатов. Baseline без фильтров/acceptance rules использует минимальные процентные SL/TP, delay и hold. Кандидаты оптимизируют выходы и ограничения допуска, поэтому улучшение не является чистым эффектом индикаторов. Research использует $10 000 nominal без ограничения бюджета, portfolio — $100 000 капитала с ограничениями. Tabs раскрывают trades, outcomes, admission funnel и Settings.

## Compare & Validation

Compare stored baseline and shortlisted outcomes across runs; inspect identical signals, costs, hashes, assumptions and profiles before interpreting differences. Benchmark smoke remains one trial per fixed fixture. No walk-forward, OOS, ML or live execution is introduced.
