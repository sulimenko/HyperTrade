import streamlit as st
from hypertrade.ui.state import load_metrics_summary


def render(runs):
    st.header('Research Home')
    st.caption('External signals → baseline → filtered candidates → Pareto research')
    if runs.empty:
        st.info('No experiments yet. Start with New Experiment; results will appear here.')
        return
    latest = runs.iloc[0]
    summary = load_metrics_summary(latest['path'])
    a,b,c = st.columns(3)
    a.metric('Latest run', latest.get('status') or 'unknown')
    b.metric('Baseline', summary.get('baseline_status', 'unknown'))
    c.metric('Pareto candidates', summary.get('pareto_candidate_count', 0))
    query = st.text_input('Find experiment', placeholder='Run name or signal set')
    frame = runs.copy()
    if query:
        frame = frame[frame.astype(str).apply(lambda row: row.str.contains(query, case=False, regex=False).any(), axis=1)]
    columns = [n for n in ['run_id','status','benchmark_name','created_at','n_trials','objective_profile_name'] if n in frame]
    st.dataframe(frame[columns], hide_index=True, width='stretch')
    st.caption('Select a run in the sidebar to connect the Explorer, Candidate Analysis and Compare views.')
