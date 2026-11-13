from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from hypertrade.api import optimize_run
from hypertrade.config.defaults import OBJECTIVE_CATALOG
from hypertrade.config.schemas import CANDIDATE_POLICY_MODES, ObjectiveProfile
from hypertrade.ui.state import ARTIFACT_ROOT, PROFILE_ROOT, LEGACY_PROFILE_ROOT, DEFAULT_BENCHMARKS
from hypertrade.ui.research_profiles import (TRIAL_PRESETS, trial_count, resolve_search, research_profile,
                                              objective_templates, load_profile, save_profile, run_config)
from hypertrade.ui.components.formula import formula_help
from hypertrade.ui.diagnostics import show_boundaries, show_trial_diagnostics
from hypertrade.ui.terms import HELP, FILTERS, label as term, metric_help, value_label, table


def render():
    st.header('Новое исследование')
    st.caption('ЭКСПЕРИМЕНТАЛЬНО · Optuna NSGA-II / Парето · номинал сделки 10 000 USD · начальный капитал 100 000 USD')
    st.caption('Историческая модель: результаты не обещают будущую доходность. Фильтры могут использовать текущую 15-минутную свечу при входе на её открытии — риск заглядывания вперёд сохраняется.')
    options = objective_templates()
    paths = {f'Сохранённый: {p.stem}': p for p in sorted(PROFILE_ROOT.glob('*.json'))}
    paths.update({f'Старый профиль: {p.stem}': p for p in sorted(LEGACY_PROFILE_ROOT.glob('*.json'))})
    choice = st.selectbox('Профиль поиска · цели и все параметры', list(options)+list(paths), help=HELP['profile'],
                          format_func=lambda v: v.replace('текущий default', 'по умолчанию'))
    reload = st.button('Восстановить сохранённые поля', help='Повторно загрузить выбранный профиль целиком. Текущие правки редакторов будут заменены только по этому явному действию или при выборе другого профиля.')
    try:
        if choice != st.session_state.get('research_profile_choice') or reload:
            st.session_state['research_profile_base'] = options[choice] if choice in options else load_profile(paths[choice])
            st.session_state['research_profile_choice'] = choice
            st.session_state['research_editor_version'] = st.session_state.get('research_editor_version', 0)+1
        base = st.session_state['research_profile_base']
    except (ValueError, TypeError, KeyError, OSError) as exc:
        st.error(f'Не удалось прочитать профиль: {exc}')
        return
    prefix = f"research:{st.session_state['research_editor_version']}:"
    objective = base['objective_profile']
    level = st.radio('Режим настроек', ['Основные настройки', 'Расширенные настройки'], horizontal=True, key='research_level',
                     help='Оба режима используют одну конфигурацию. Основные поля видны сразу, остальные доступны в раскрываемой группе. Переключение не сбрасывает значения, ограничения и начальное число генератора.')
    st.caption('Смена профиля восстанавливает все его настройки. Раскрытие расширенных настроек сохраняет текущие правки; старые профили читаются без изменения исходного файла.')
    source = st.selectbox('Источник сигналов CSV', list(DEFAULT_BENCHMARKS) + ['Custom path', 'Upload CSV'],
                          format_func=lambda v: {'Custom path': 'Путь к своему CSV', 'Upload CSV': 'Загрузить CSV'}.get(v, v), help=HELP['csv'])
    upload = None
    if source == 'Upload CSV':
        upload = st.file_uploader('Файл сигналов CSV · разделитель «;»', type=['csv'], help=HELP['csv'])
        path = None
    else:
        path = st.text_input('Путь к CSV', value=DEFAULT_BENCHMARKS.get(source, ''), help=HELP['csv'])
    a,b = st.columns(2)
    run_label = a.text_input('Название исследования', value='research', help='Название помогает найти сохранённое исследование. Не меняет сигналы и расчёты.')
    preset = b.selectbox('Число испытаний', TRIAL_PRESETS, index=TRIAL_PRESETS.index(base['trial_preset']), key=prefix+'preset',
                         format_func=lambda v: '20 · короткая проверка' if v == TRIAL_PRESETS[0] else v, help=HELP['trials'])
    custom = b.number_input('Число испытаний вручную', min_value=1, max_value=10000, value=base['n_trials'],
                            disabled=preset != 'Вручную', key=prefix+'trials', help=HELP['trials'])
    n_trials = trial_count(preset, int(custom))
    payload = dict(base['search_space'])
    # All advanced widgets are rendered inside a collapsed group even in basic
    # mode. Streamlit therefore never cleans up hidden seed/constraint/JSON keys.
    visual = st.session_state.get(prefix+'mode', 'Визуальные поля') == 'Визуальные поля'
    if not visual:
        st.info('Активен полный JSON из расширенных настроек. Поля диапазонов и фильтров ниже отключены; действует итоговая конфигурация из резюме.')
    st.subheader('Выход из сделки и основные фильтры')
    st.caption('SL — ограничение убытка, TP — фиксация прибыли. В процентном режиме единицы — % от цены входа; в режиме ATR — множители волатильности.')
    for stem in ['sl', 'tp']:
        for col, suffix in zip(st.columns(3), ['min', 'max', 'step']):
            field = stem+'_'+suffix
            payload[field] = col.number_input(term(field), min_value=0.0, value=float(payload[field]), step=0.25,
                key=prefix+field, disabled=not visual,
                help='Диапазон расстояния от цены входа в %. Минимум и максимум включены, шаг положительный; разность границ должна делиться на шаг. Исходная стратегия использует процентные минимумы SL/TP.')
    for col, stem in zip(st.columns(3), ['ema', 'rsi', 'atr']):
        payload[stem+'_use'] = col.checkbox(FILTERS[stem][0], value=payload[stem+'_use'], key=prefix+stem+'_use', disabled=not visual,
                                          help=FILTERS[stem][1]+' Включение разрешает поиск с этим индикатором; число свечей подтверждения доступно в расширенном JSON. Историческая модель может использовать текущую свечу.')
    # ATR ranges remain visible for an enabled ATR profile, including basic mode.
    for stem in ['atr_sl', 'atr_tp']:
        for col, suffix in zip(st.columns(3), ['min', 'max', 'step']):
            field = stem+'_'+suffix
            payload[field] = col.number_input(term(field), min_value=0.0, value=float(payload[field]), step=0.25,
                key=prefix+field, disabled=not visual or not payload['atr_use'], help=FILTERS['atr'][1]+' Минимум/максимум положительные, шаг положительный; разность границ должна делиться на шаг.')
    toggles = [n+'_use' for n in FILTERS]
    exposed = toggles + ['commission', 'slippage', 'ranking_mode_choices'] + [stem+'_'+suffix for stem in ['sl','tp','atr_sl','atr_tp'] for suffix in ['min','max','step']]
    with st.expander('Расширенные настройки · цели, ограничения, издержки и JSON', expanded=level == 'Расширенные настройки', key='advanced:'+level):
        seed = st.number_input('Начальное число генератора (seed)', min_value=0, max_value=2**32-1, value=base['seed'], key=prefix+'seed', help=HELP['seed'])
        profile_name = st.text_input('Имя профиля', value=objective['name'], key=prefix+'name', help='1–64 латинских буквы, цифры, дефисы или подчёркивания; начало — буква или цифра. Сохранение того же имени заменяет профиль в data/cache/profiles, но не результаты исследований.')
        selected = st.multiselect('Цели Парето', list(OBJECTIVE_CATALOG), default=[o['metric'] for o in objective['objectives']],
                                  key=prefix+'axes', format_func=term, help=HELP['objectives'])
        saved_directions = {o['metric']: o['direction'] for o in objective['objectives']}
        directions = {}
        for metric in selected:
            directions[metric] = st.selectbox(term(metric), ['maximize','minimize'],
                index=0 if saved_directions.get(metric, OBJECTIVE_CATALOG[metric]['direction']) == 'maximize' else 1,
                key=prefix+'direction:'+metric, format_func=value_label, help=metric_help(metric)+' '+HELP['direction'])
        policy = st.selectbox('Порядок выбора кандидатов', sorted(CANDIDATE_POLICY_MODES),
            index=sorted(CANDIDATE_POLICY_MODES).index(objective['candidate_policy']['mode']), key=prefix+'policy', format_func=value_label, help=HELP['policy'])
        st.caption('Ограничения допуска · минимум и максимум включительно. Подсказки доступны в заголовках столбцов.')
        constraints = st.data_editor(pd.DataFrame(objective['constraints'], columns=['metric','min_value','max_value']),
            num_rows='dynamic', hide_index=True, width='stretch', key=prefix+'constraints',
            column_config={'metric': st.column_config.SelectboxColumn('Метрика', options=list(OBJECTIVE_CATALOG), format_func=term, help=HELP['constraints']),
                           'min_value': st.column_config.NumberColumn('Минимум', help=HELP['constraints']),
                           'max_value': st.column_config.NumberColumn('Максимум', help=HELP['constraints'])})
        st.radio('Редактор параметров поиска', ['Визуальные поля', 'Полный JSON'], horizontal=True, key=prefix+'mode',
                 help='Визуальные поля объединяются только с дополнительными JSON-параметрами. Полный JSON — отдельный явный источник; значения редакторов сохраняются при переключении и не перетирают друг друга. Проверяйте резюме.')
        cols = st.columns(3)
        for i, field in enumerate(f for f in toggles if f not in ['ema_use','rsi_use','atr_use']):
            stem = field[:-4]
            payload[field] = cols[i%3].checkbox(FILTERS[stem][0], value=payload[field], key=prefix+field, disabled=not visual,
                                               help=FILTERS[stem][1]+' Число свечей подтверждения задаётся в дополнительных параметрах; настройки индикатора подбирает Optuna. Учитывайте риск использования текущей свечи.')
        a,b,c = st.columns(3)
        payload['commission'] = a.number_input(term('commission'), min_value=0.0, value=float(payload['commission']), step=.01, key=prefix+'commission', disabled=not visual, help=HELP['commission'])
        payload['slippage'] = b.number_input(term('slippage'), min_value=0.0, max_value=.9999, value=float(payload['slippage']), format='%.4f', key=prefix+'slippage', disabled=not visual, help=HELP['slippage'])
        payload['ranking_mode_choices'] = c.multiselect('Ранжирование сигналов', ['none','strength'], default=payload['ranking_mode_choices'], key=prefix+'ranking', disabled=not visual, format_func=value_label, help=HELP['ranking'])
        if visual:
            st.caption('JSON содержит только остальные поля. Дублирование визуальных полей запрещено: ни один редактор не перетирает другой.')
            raw = st.text_area('Прочие параметры JSON', json.dumps({k:v for k,v in base['search_space'].items() if k not in exposed}, indent=2), height=240, key=prefix+'extra', help=HELP['extra'])
        else:
            st.caption('Единственный источник — полный JSON. Визуальные поля отключены. При смене редактора применяются его сохранённые значения; итог виден в резюме.')
            raw = st.text_area('Полный JSON параметров поиска', json.dumps(base['search_space'], indent=2), height=320, key=prefix+'json', help=HELP['extra'])
    # Retain both JSON editors across switches too (they are not both rendered).
    inactive = prefix+('json' if visual else 'extra')
    if inactive in st.session_state:
        st.session_state[inactive] = st.session_state[inactive]
    config = None
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError('JSON должен быть объектом')
        if visual:
            duplicates = set(parsed) & set(exposed)
            if duplicates:
                raise ValueError(f'Удалите визуальные поля из дополнительных JSON: {sorted(duplicates)}')
            payload = {**{k:v for k,v in payload.items() if k in exposed}, **parsed}
        else:
            payload = parsed
        search = resolve_search(payload)
        rows = [{k: None if pd.isna(v) else v for k,v in row.items()} for row in constraints.to_dict(orient='records')]
        profile = ObjectiveProfile.from_dict(dict(name=profile_name, objectives=[dict(metric=m, direction=directions[m]) for m in selected],
            constraints=[r for r in rows if any(v is not None for v in r.values())], candidate_policy={'mode':policy}))
        config = research_profile(profile.to_dict(), asdict(search), preset, n_trials, int(seed))
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        st.error(f'Конфигурация не готова: {exc}')
    st.subheader('Резюме перед запуском')
    if config:
        st.write(f'CSV: {upload.name if upload else path or "не выбран"} · исследование: {run_label} · испытаний: {n_trials} · seed: {seed}')
        st.write('Цели: ' + '; '.join(f'{term(m)} — {value_label(d)}' for m,d in directions.items()))
        st.write('Эффективные ограничения: ' + ('; '.join(f"{term(r['metric'])}: минимум {r['min_value'] if r['min_value'] is not None else 'не задан'}, максимум {r['max_value'] if r['max_value'] is not None else 'не задан'}" for r in config['objective_profile']['constraints']) or 'нет'))
        st.write('Выбор кандидатов: '+value_label(policy))
        effective = config['search_space']
        st.write('Включены: '+(', '.join(FILTERS[k[:-4]][0] for k in toggles if effective[k]) or 'без индикаторов и дополнительных правил'))
        table(pd.DataFrame([{'Параметр': term(stem), 'Минимум': effective[stem+'_min'], 'Максимум': effective[stem+'_max'], 'Шаг': effective[stem+'_step']} for stem in ['sl','tp','atr_sl','atr_tp']]), hide_index=True, width='stretch')
        st.caption(f"Комиссия: {effective['commission']} USD/акция/сторона · проскальзывание: {effective['slippage']} доли на сторону. Исходная стратегия использует процентные минимумы SL/TP без ATR и фильтров.")
        with st.expander('Техническая конфигурация · именно она будет сохранена и передана движку'):
            st.json(config)
    else:
        st.warning('Исправьте конфигурацию в расширенных настройках; запуск и сохранение недоступны.')
    if st.button('Сохранить профиль поиска', disabled=config is None):
        try:
            target = save_profile(PROFILE_ROOT, config)
            st.success(f'Сохранён {target}. Выберите его выше для восстановления всех полей.')
        except (ValueError, TypeError, KeyError, OSError) as exc:
            st.error(str(exc))
    formula_help()
    with st.expander('Допущения исполнения и ограничения модели'):
        st.write('Фильтры и ранжирование могут использовать OHLC текущей 15-минутной свечи входа на её открытии: риск заглядывания вперёд не исправлен. Сохранена эвристика порядка SL/TP внутри свечи по её направлению. Известный скользящий стоп активен сразу; Parabolic SAR ждёт инициализации. VWAP — по UTC-дню на свечах основной сессии.')
        st.write('Исходная стратегия: без фильтров и правил допуска, минимальные процентные SL/TP, задержка и срок удержания из поиска. Изменение результата может происходить от настройки выходов, а не только фильтров. Номинал и издержки одинаковы. Проверки на будущих и отложенных данных нет.')
    if st.button('Запустить исследование', type='primary', width='stretch', disabled=config is None):
        try:
            if upload is not None:
                data = upload.getvalue()
                cached = Path('data/cache/uploads')
                cached.mkdir(parents=True, exist_ok=True)
                path = str(cached/(hashlib.sha256(data).hexdigest()+'.csv'))
                Path(path).write_bytes(data)
            if not path or not Path(path).is_file():
                raise ValueError('Выберите существующий CSV или загрузите файл')
            with st.spinner('Расчёт исходной стратегии и кандидатов…'):
                run = optimize_run(run_config(config, path, run_label, ARTIFACT_ROOT), profile, search)
            manifest = json.loads((run/'manifest.json').read_text())
            st.session_state['pending_run'] = str(run)
            if manifest['status'] == 'failed':
                st.error(f'Ошибка исследования: {run.name}. Диагностика сохранена в каталоге результатов.')
                with st.expander('Техническая диагностика ошибки'):
                    st.json(json.loads((run/'errors.json').read_text()))
            else:
                st.success(f"{value_label(manifest['status'])}: {run.name}")
            show_trial_diagnostics(run, pd.read_parquet(run/'trials.parquet'))
            show_boundaries(run)
            st.caption('Откройте «Обзор оптимизации» или «Анализ кандидата» в боковой панели.')
        except (ValueError, TypeError, OSError) as exc:
            st.error(str(exc))
