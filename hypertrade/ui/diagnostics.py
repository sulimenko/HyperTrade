"""Factual diagnostics of saved studies, with no study execution or mutation."""
import json
from pathlib import Path
import sqlite3
import math
import pandas as pd
import plotly.express as px
import streamlit as st
from hypertrade.ui.components.charts import show
from hypertrade.ui.state import load_json_file, load_run_table_safe


def prune_summary(trials):
    states = trials.get('state', pd.Series(dtype=str))
    counts = states.value_counts().reindex(['COMPLETE', 'PRUNED', 'FAIL'], fill_value=0).rename_axis('Статус').reset_index(name='Испытаний')
    reasons = []
    for row in trials[states.eq('PRUNED')].to_dict('records'):
        raw = row.get('user_attrs_constraint_violations', row.get('constraint_violations'))
        reasons.extend(set(str(raw).split(',')) if pd.notna(raw) and str(raw) else ['Причина не сохранена'])
    summary = pd.Series(reasons, dtype=str).value_counts().rename_axis('Причина').reset_index(name='Испытаний')
    return counts, summary


def boundary_summary(run, pareto, search):
    """Use actual Optuna numeric distributions when saved; fall back to configured ranges."""
    distributions = {}
    db = Path(run)/'study.sqlite3'
    if db.exists():
        try:
            with sqlite3.connect(db.resolve().as_uri()+'?mode=ro', uri=True) as connection:
                for number, param, raw in connection.execute('SELECT t.number, p.param_name, p.distribution_json FROM trial_params p JOIN trials t ON p.trial_id=t.trial_id'):
                    attrs = json.loads(raw)['attributes']
                    if 'low' in attrs and 'high' in attrs:
                        distributions[(number, param)] = (attrs['low'], attrs['high'])
        except (sqlite3.Error, ValueError, KeyError):
            pass  # Old bundles can still expose their explicitly configured search ranges.
    aliases = {'accept_start_minute': 'accept_start', 'accept_end_minute': 'accept_end', 'cooldown_minutes': 'cooldown'}
    rows, touched = [], set()
    for param in pareto.columns:
        key = aliases.get(param, param)
        bounds = (search.get(key+'_min'), search.get(key+'_max'))
        observations = []
        for row in pareto.to_dict('records'):
            low, high = distributions.get((int(row['number']), param), bounds)
            value = row.get(param)
            if low is None or high is None or pd.isna(value) or isinstance(value, (bool, str)):
                continue
            at_min = math.isclose(float(value), low, rel_tol=1e-9, abs_tol=1e-9)
            at_max = math.isclose(float(value), high, rel_tol=1e-9, abs_tol=1e-9)
            observations.append((low, high, at_min, at_max))
            if (at_min or at_max) and low != high:
                touched.add(int(row['number']))
        if observations:
            n = len(observations)
            varying = [o for o in observations if o[0] != o[1]]
            variable = bool(varying)
            rows.append(dict(Параметр=param, Единица='× ATR' if param in ('atr_sl','atr_tp') else '%' if param in ('sl','tp') else 'значение',
                             min=min(o[0] for o in observations), max=max(o[1] for o in observations),
                             Допущено=n, На_min=sum(o[2] for o in observations), На_max=sum(o[3] for o in observations),
                             Доля_на_границе=sum(a or b for _,_,a,b in varying)/len(varying) if variable else 0,
                             Фиксирован=not variable))
    return pd.DataFrame(rows), len(touched)


def show_boundaries(run):
    pareto = load_run_table_safe(run, 'pareto_trials.parquet')
    search = load_json_file(Path(run)/'search_space.json', {})
    table, count = boundary_summary(run, pareto, search)
    st.caption(f'Границы поиска: {count} из {len(pareto)} Pareto-кандидатов достигли min/max хотя бы одного числового параметра (фиксированные диапазоны исключены).')
    if not table.empty:
        if table['Доля_на_границе'].ge(.5).any():
            st.warning('Не менее 50% кандидатов с заданным параметром достигли его границы. Рассмотрите расширение диапазона; это не доказывает, что за границей результат лучше.')
        with st.expander('Параметры Pareto на границах'):
            st.dataframe(table, hide_index=True, width='stretch')
            st.caption('Условные параметры учитываются только там, где они выбраны. Min/max берутся из сохранённых распределений Optuna; для старых bundles — из search_space.json. Фиксированные диапазоны не вызывают предупреждение.')
    else:
        st.caption('Нет сохранённых числовых диапазонов для диагностики.')


def show_trial_diagnostics(run, trials):
    counts, reasons = prune_summary(trials)
    st.dataframe(counts, hide_index=True, width='stretch')
    if not reasons.empty:
        st.dataframe(reasons, hide_index=True, width='stretch')
        show(px.bar(reasons, x='Испытаний', y='Причина', orientation='h', title='Причины PRUNED'))
    st.caption('COMPLETE: рассчитаны допустимые цели. Pareto — недоминируемое подмножество COMPLETE. PRUNED: ограничения или неопределённые метрики; один trial может иметь несколько причин. undefined: метрика не определена; missing: нет значения для ограничения; <min / >max: нарушен порог. FAIL: техническое исключение. no_valid_candidates — корректный результат без допустимых кандидатов.')
