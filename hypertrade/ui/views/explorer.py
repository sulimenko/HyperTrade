from hypertrade.ui.terms import HELP, label as term, metric_help, value_label, table, metric_legend
import pandas as pd
import plotly.express as px
import streamlit as st
from hypertrade.ui.components.charts import show
from hypertrade.ui.components.selection import candidate_key, choose_candidate, choose_shortlist, choose_comparison
from hypertrade.ui.state import load_run_table_safe, load_json_file
from pathlib import Path
from hypertrade.ui.diagnostics import show_boundaries, show_trial_diagnostics
from hypertrade.ui.components.formula import formula_help


def render(run):
    st.header('Обзор оптимизации')
    if not run:
        st.info('Запустите или выберите исследование.')
        return
    formula_help()
    metric_legend()
    show_boundaries(run)
    pareto = load_run_table_safe(run,'pareto_trials.parquet')
    trials = load_run_table_safe(run,'trials.parquet')
    status_options = trials['state'].dropna().unique().tolist() if 'state' in trials else []
    statuses = st.multiselect('Статус испытания', status_options, default=status_options, format_func=value_label, help=HELP['rejections'])
    visible_trials = trials[trials['state'].isin(statuses)] if 'state' in trials else trials
    universe = st.radio('Результаты на графике', ['Граница Парето','Все допустимые результаты'], horizontal=True, help=HELP['objectives'])
    frame = pareto.copy() if universe == 'Граница Парето' else visible_trials[visible_trials['state'].eq('COMPLETE')].copy() if 'state' in visible_trials else visible_trials.copy()
    if universe == 'Граница Парето' and 'COMPLETE' not in statuses:
        frame = frame.iloc[:0]
    metrics = [n for n in frame.select_dtypes('number').columns if frame[n].notna().any() and n not in ['number','trade_count'] and not n.startswith(('params_','values_','user_attrs_'))]
    if len(metrics) >= 2 and not frame.empty:
        controls = st.columns(5)
        mode = controls[0].selectbox('Размерность',['3D','2D'], help=HELP['chart'])
        axes = []
        defaults = ['portfolio_max_drawdown_pct', 'portfolio_return_pct', 'profit_factor']
        labels = {m: term(m) for m in frame.columns}
        for i,name in enumerate(['X','Y','Z'] if mode=='3D' else ['X','Y']):
            axes.append(controls[i+1].selectbox('Ось '+name,metrics,index=metrics.index(defaults[i]) if defaults[i] in metrics else min(i,len(metrics)-1), key='axis_'+name, format_func=term, help=HELP['chart']))
        color = controls[4].selectbox('Цвет · дополнительный показатель',['None']+metrics,index=1+metrics.index('avg_hold_minutes') if 'avg_hold_minutes' in metrics else 0, format_func=lambda m: value_label(m) if m == 'None' else term(m), help=HELP['chart'])
        args = dict(data_frame=frame, x=axes[0],y=axes[1],color=None if color=='None' else color,
                    custom_data=['number'],hover_data=['number'],color_continuous_scale=['#37C7F3','#9263F4'], labels=labels)
        if mode == '3D':
            fig = px.scatter_3d(z=axes[2], **args)
            fig.update_traces(marker_size=6)
            fig.update_layout(height=560, scene_camera=dict(eye=dict(x=1.5,y=1.5,z=1.2)))
            show(fig,key='pareto_3d_'+Path(run).name)
            st.caption('Перетащите для вращения, прокрутите для масштаба. В 3D выбирайте кандидата списком или таблицей ниже.')
        else:
            fig = px.scatter(**args)
            fig.update_traces(marker_size=10)
            event = show(fig,on_select='rerun',selection_mode=['points','box','lasso'],key='pareto_2d_'+Path(run).name)
            points = event.selection.points if event else []
            token = repr(points)
            if points and token != st.session_state.get('last_selection'):
                number = int(points[0]['customdata'][0])
                if number in pareto.get('number', pd.Series(dtype=int)).tolist():
                    st.session_state[candidate_key(run)] = number
                st.session_state['last_selection'] = token
    else:
        st.info('Для выбранных статусов нет кандидатов со значениями показателей графика.')
    number = choose_candidate(run)
    if number is not None:
        choose_comparison(run, pareto['number'].astype(int).tolist(), number)
        automatic = load_run_table_safe(run,'candidate_shortlist.parquet')
        choose_shortlist(run,pareto['number'].astype(int).tolist(),automatic.get('number',pd.Series(dtype=int)).astype(int).tolist())
        row = pareto[pareto['number']==number].iloc[0]
        a,b,c=st.columns(3)
        a.metric(term('sum_trade_returns_pct'),f"{row['sum_trade_returns_pct']:.2f}", help=metric_help('sum_trade_returns_pct'))
        b.metric(term('portfolio_return_pct'),f"{row['portfolio_return_pct']:.2f}%", help=metric_help('portfolio_return_pct'))
        c.metric(term('portfolio_max_drawdown_pct'),f"{row['portfolio_max_drawdown_pct']:.2f}%", help=metric_help('portfolio_max_drawdown_pct'))
        st.caption(f'Испытание №{number} связано со страницей «Анализ кандидата». Выбранные кандидаты сохраняются в этой сессии интерфейса.')
    tab1,tab2,tab3=st.tabs(['Кандидаты Парето','Все испытания','Диагностика'])
    with tab1:
        table(pareto,hide_index=True,width='stretch',on_select='rerun',selection_mode='single-row',key='candidate_table_'+Path(run).name)
        event = st.session_state.get('candidate_table_'+Path(run).name,{})
        selected = event.get('selection',{}).get('rows',[])
        if selected:
            token = (Path(run).name, tuple(selected))
            if token != st.session_state.get('last_table_selection'):
                st.session_state['pending_candidate'] = int(pareto.iloc[selected[0]]['number'])
                st.session_state['last_table_selection'] = token
                st.rerun()
    with tab2:
        show_trial_diagnostics(run, trials)
        table(visible_trials,hide_index=True,width='stretch')
    with tab3:
        st.caption('Сводка статусов и причин отклонения — во вкладке «Все испытания».')
        with st.expander('Технические сведения об ошибках'):
            st.json(load_json_file(Path(run)/'errors.json',{}))
        st.caption('Ошибка расчёта означает техническое исключение; отклонение — нарушение ограничений или неопределённую цель. Отсутствие допустимых кандидатов может быть корректным результатом исследования.')
