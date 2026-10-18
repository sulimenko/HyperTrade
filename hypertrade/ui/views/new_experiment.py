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


def render():
    st.header('New Experiment')
    st.caption('EXPERIMENTAL · NSGA-II / Pareto · $10,000 normalized nominal · $100,000 initial portfolio')
    formula_help()
    options = objective_templates()
    paths = {f'Сохранённый: {p.stem}': p for p in sorted(PROFILE_ROOT.glob('*.json'))}
    paths.update({f'Legacy: {p.stem}': p for p in sorted(LEGACY_PROFILE_ROOT.glob('*.json'))})
    choice = st.selectbox('Профиль поиска · цели и все параметры', list(options)+list(paths))
    reload = st.button('Восстановить сохранённые поля')
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
    st.caption('Профиль поиска сохраняет цели, направления, ограничения, shortlist policy, диапазоны, costs, число trials и seed. Это не формула доходности. Legacy-профиль при сохранении переносится в data/cache/profiles; исходный файл остаётся.')
    source = st.selectbox('Signal set', list(DEFAULT_BENCHMARKS) + ['Custom path', 'Upload CSV'])
    upload = None
    if source == 'Upload CSV':
        upload = st.file_uploader('External signals (; separated)', type=['csv'])
        path = None
    else:
        path = st.text_input('CSV path', value=DEFAULT_BENCHMARKS.get(source, ''))
    a,b,c = st.columns(3)
    label = a.text_input('Experiment label', value='research')
    preset = b.selectbox('Trial preset', TRIAL_PRESETS, index=TRIAL_PRESETS.index(base['trial_preset']), key=prefix+'preset')
    custom = b.number_input('Trials вручную', min_value=1, max_value=10000, value=base['n_trials'],
                            disabled=preset != 'Вручную', key=prefix+'trials')
    n_trials = trial_count(preset, int(custom))
    b.caption(f'В engine: n_trials = {n_trials}. 20 — smoke/test; 100/200 — исследование.')
    seed = c.number_input('Seed', min_value=0, max_value=2**32-1, value=base['seed'], key=prefix+'seed')
    st.caption('Benchmark smoke остаётся отдельным режимом: 1 trial на fixture. Даты CSV без часового пояса означают UTC; dd.mm.yyyy и ISO поддерживаются.')
    with st.expander('Цели и жёсткие ограничения', expanded=True):
        profile_name = st.text_input('Имя профиля', value=objective['name'], key=prefix+'name')
        selected = st.multiselect('Pareto-цели', list(OBJECTIVE_CATALOG), default=[o['metric'] for o in objective['objectives']], key=prefix+'axes')
        saved_directions = {o['metric']: o['direction'] for o in objective['objectives']}
        directions = {}
        for metric in selected:
            directions[metric] = st.selectbox(OBJECTIVE_CATALOG[metric]['label'], ['maximize','minimize'],
                index=0 if saved_directions.get(metric, OBJECTIVE_CATALOG[metric]['direction']) == 'maximize' else 1,
                key=prefix+'direction:'+metric)
        policy = st.selectbox('Shortlist policy', sorted(CANDIDATE_POLICY_MODES),
            index=sorted(CANDIDATE_POLICY_MODES).index(objective['candidate_policy']['mode']), key=prefix+'policy')
        constraints = st.data_editor(pd.DataFrame(objective['constraints'], columns=['metric','min_value','max_value']),
            num_rows='dynamic', hide_index=True, width='stretch', key=prefix+'constraints',
            column_config={'metric': st.column_config.SelectboxColumn('Метрика', options=list(OBJECTIVE_CATALOG)),
                           'min_value': st.column_config.NumberColumn('Минимум'), 'max_value': st.column_config.NumberColumn('Максимум')})
    payload = dict(base['search_space'])
    mode = st.radio('Источник search_payload', ['Визуальные поля', 'Полный JSON'], horizontal=True, key=prefix+'mode')
    visual = mode == 'Визуальные поля'
    toggles = [n+'_use' for n in ['ema','rsi','adx','macd','bb','vwap','volume','atr','donchian','psar','ts','time_gate','cooldown']]
    exposed = toggles + ['commission', 'slippage', 'ranking_mode_choices']
    with st.expander('Диапазоны SL/TP и costs', expanded=True):
        a,b,c = st.columns(3)
        for i, field in enumerate(toggles):
            payload[field] = [a,b,c][i%3].checkbox(field[:-4].upper(), value=payload[field], key=prefix+field, disabled=not visual)
        st.caption('Процентный режим: SL/TP в %. ATR mode: atr_sl/atr_tp — множители ATR. Baseline всегда использует процентные минимумы SL/TP, без ATR и фильтров.')
        for stem in ['sl','tp','atr_sl','atr_tp']:
            cols = st.columns(3)
            for col, suffix in zip(cols, ['min','max','step']):
                field = stem+'_'+suffix
                payload[field] = col.number_input(field, min_value=0.0, value=float(payload[field]), step=0.25,
                    key=prefix+field, disabled=not visual or (stem.startswith('atr_') and not payload['atr_use']))
        # Even inactive ATR fields remain part of the one saved/resolved payload.
        exposed += [stem+'_'+suffix for stem in ['sl','tp','atr_sl','atr_tp'] for suffix in ['min','max','step']]
        payload['commission'] = a.number_input('Commission · USD/share/side', min_value=0.0, value=float(payload['commission']), step=.01, key=prefix+'commission', disabled=not visual)
        payload['slippage'] = b.number_input('Slippage · fraction/side', min_value=0.0, max_value=.9999, value=float(payload['slippage']), format='%.4f', key=prefix+'slippage', disabled=not visual)
        payload['ranking_mode_choices'] = c.multiselect('Ranking modes', ['none','strength'], default=payload['ranking_mode_choices'], key=prefix+'ranking', disabled=not visual)
    with st.expander('Расширенные параметры'):
        if visual:
            st.caption('JSON содержит только остальные поля. Дублирование визуальных полей запрещено: ни один редактор не перетирает другой.')
            raw = st.text_area('Прочие параметры JSON', json.dumps({k:v for k,v in base['search_space'].items() if k not in exposed}, indent=2), height=240, key=prefix+'extra')
        else:
            st.caption('Единственный источник — этот полный JSON. Визуальные поля отключены и не применяются. При переключении режима используются значения соответствующего редактора; итог всегда виден ниже.')
            raw = st.text_area('Полный search space JSON', json.dumps(base['search_space'], indent=2), height=320, key=prefix+'json')
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
    with st.expander('Окончательная конфигурация · именно она будет сохранена и передана engine', expanded=True):
        if config:
            st.json(config)
    if st.button('Сохранить профиль поиска', disabled=config is None):
        try:
            target = save_profile(PROFILE_ROOT, config)
            st.success(f'Сохранён {target}. Выберите его выше для восстановления всех полей.')
        except (ValueError, TypeError, KeyError, OSError) as exc:
            st.error(str(exc))
    with st.expander('Execution assumptions'):
        st.write('Фильтры и ranking могут использовать OHLC свечи входа на её открытии. Сохранена bullish/bearish same-bar SL/TP эвристика. SL/TP и известный trailing level активны сразу; PSAR ждёт инициализации. VWAP — UTC-day на RTH-свечах.')
        st.write('Baseline: без фильтров и acceptance rules, минимальные процентные SL/TP, delay и holding period из поиска. Изменение результата может происходить от настройки выходов, а не только фильтров. Nominal и costs одинаковы.')
    if st.button('Start research', type='primary', width='stretch', disabled=config is None):
        try:
            if upload is not None:
                data = upload.getvalue()
                cached = Path('data/cache/uploads')
                cached.mkdir(parents=True, exist_ok=True)
                path = str(cached/(hashlib.sha256(data).hexdigest()+'.csv'))
                Path(path).write_bytes(data)
            if not path or not Path(path).is_file():
                raise ValueError('Choose an existing CSV or upload a file')
            with st.spinner('Evaluating baseline and candidates…'):
                run = optimize_run(run_config(config, path, label, ARTIFACT_ROOT), profile, search)
            manifest = json.loads((run/'manifest.json').read_text())
            st.session_state['pending_run'] = str(run)
            if manifest['status'] == 'failed':
                st.error(f'Research failed: {run.name}. Diagnostics are saved in the run bundle.')
                st.json(json.loads((run/'errors.json').read_text()))
            else:
                st.success(f"{manifest['status']}: {run.name}")
            show_trial_diagnostics(run, pd.read_parquet(run/'trials.parquet'))
            show_boundaries(run)
            st.caption('Open Optimizer Explorer or Candidate Analysis from the sidebar.')
        except (ValueError, TypeError, OSError) as exc:
            st.error(str(exc))
