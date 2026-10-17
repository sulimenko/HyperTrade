import pandas as pd
import plotly.express as px
import streamlit as st
from hypertrade.ui.components.charts import show
from hypertrade.ui.components.selection import candidate_key, choose_candidate, choose_shortlist
from hypertrade.ui.state import load_run_table_safe, load_json_file
from pathlib import Path


def render(run):
    st.header('Optimizer Explorer')
    if not run:
        st.info('Start or select an experiment.')
        return
    pareto = load_run_table_safe(run,'pareto_trials.parquet')
    trials = load_run_table_safe(run,'trials.parquet')
    status_options = trials['state'].dropna().unique().tolist() if 'state' in trials else []
    statuses = st.multiselect('Trial status', status_options, default=status_options)
    visible_trials = trials[trials['state'].isin(statuses)] if 'state' in trials else trials
    universe = st.radio('Plot', ['Pareto frontier','All completed trials'], horizontal=True)
    frame = pareto.copy() if universe == 'Pareto frontier' else visible_trials.copy()
    if universe == 'Pareto frontier' and 'COMPLETE' not in statuses:
        frame = frame.iloc[:0]
    metrics = [n for n in frame.select_dtypes('number').columns if n not in ['number','trade_count'] and not n.startswith(('params_','values_','user_attrs_'))]
    if len(metrics) >= 2 and not frame.empty:
        controls = st.columns(5)
        mode = controls[0].selectbox('Dimensions',['3D','2D'])
        axes = []
        for i,name in enumerate(['X','Y','Z'] if mode=='3D' else ['X','Y']):
            axes.append(controls[i+1].selectbox(name,metrics,index=min(i,len(metrics)-1),key='axis_'+name))
        color = controls[4].selectbox('Color · fourth metric',['None']+metrics,index=min(4,len(metrics)))
        args = dict(data_frame=frame, x=axes[0],y=axes[1],color=None if color=='None' else color,
                    custom_data=['number'],hover_data=['number'],color_continuous_scale=['#37C7F3','#9263F4'])
        if mode == '3D':
            fig = px.scatter_3d(z=axes[2], **args)
            fig.update_traces(marker_size=6)
            fig.update_layout(height=560, scene_camera=dict(eye=dict(x=1.5,y=1.5,z=1.2)))
            show(fig,key='pareto_3d_'+Path(run).name)
            st.caption('Drag to rotate · scroll to zoom. Use the linked selector/table below to select a 3D candidate.')
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
        st.info('No candidates with plotted objective values for this status filter.')
    number = choose_candidate(run)
    if number is not None:
        automatic = load_run_table_safe(run,'candidate_shortlist.parquet')
        choose_shortlist(run,pareto['number'].astype(int).tolist(),automatic.get('number',pd.Series(dtype=int)).astype(int).tolist())
        row = pareto[pareto['number']==number].iloc[0]
        a,b,c=st.columns(3)
        a.metric('Research sum · percentage points',f"{row['sum_trade_returns_pct']:.2f}")
        b.metric('Portfolio return',f"{row['portfolio_return_pct']:.2f}%")
        c.metric('Portfolio peak drawdown',f"{row['portfolio_max_drawdown_pct']:.2f}%")
        st.caption(f'Trial {number} is linked to Candidate Analysis. Shortlist changes are kept for this UI session.')
    tab1,tab2,tab3=st.tabs(['Pareto candidates','All trials','Diagnostics'])
    with tab1:
        st.dataframe(pareto,hide_index=True,width='stretch',on_select='rerun',selection_mode='single-row',key='candidate_table_'+Path(run).name)
        event = st.session_state.get('candidate_table_'+Path(run).name,{})
        selected = event.get('selection',{}).get('rows',[])
        if selected:
            token = (Path(run).name, tuple(selected))
            if token != st.session_state.get('last_table_selection'):
                st.session_state['pending_candidate'] = int(pareto.iloc[selected[0]]['number'])
                st.session_state['last_table_selection'] = token
                st.rerun()
    with tab2:
        st.dataframe(visible_trials,hide_index=True,width='stretch')
    with tab3:
        st.json(load_json_file(Path(run)/'errors.json',{}))
        st.caption('FAIL is an unexpected exception with a stack trace. PRUNED is a valid but ineligible candidate or undefined objective. A correct run may have no_valid_candidates.')
