from hypertrade.ui.terms import HELP, value_label, table
import streamlit as st
from hypertrade.ui.state import load_metrics_summary


def render(runs):
    st.header('Обзор исследований')
    st.caption('Внешние сигналы → исходная стратегия → кандидаты с фильтрами → исследование Парето')
    if runs.empty:
        st.info('Исследований пока нет. Откройте «Новое исследование»; результаты появятся здесь.')
        return
    latest = runs.iloc[0]
    summary = load_metrics_summary(latest['path'])
    a,b,c = st.columns(3)
    a.metric('Последнее исследование', value_label(latest.get('status') or 'unknown'), help='Статус последнего сохранённого запуска. Отсутствие допустимых кандидатов — корректный результат исследования.')
    b.metric('Исходная стратегия', value_label(summary.get('baseline_status', 'unknown')), help='Стратегия без фильтров и правил допуска, с минимальными процентными SL/TP из поиска. Номинал и издержки те же, что у кандидатов.')
    c.metric('Кандидатов Парето', summary.get('pareto_candidate_count', 0), help=HELP['objectives'])
    query = st.text_input('Найти исследование', placeholder='Название исследования или набора сигналов')
    frame = runs.copy()
    if query:
        frame = frame[frame.astype(str).apply(lambda row: row.str.contains(query, case=False, regex=False).any(), axis=1)]
    columns = [n for n in ['run_id','status','benchmark_name','created_at','n_trials','objective_profile_name'] if n in frame]
    table(frame[columns], hide_index=True, width='stretch')
    st.caption('Выберите исследование в боковой панели: оно связывает обзор оптимизации, анализ кандидата и сравнение.')
