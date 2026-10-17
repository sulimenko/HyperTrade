from pathlib import Path
import pandas as pd
import streamlit as st
from hypertrade.api import run_benchmark_suite
from hypertrade.ui.state import benchmark_suite_options, load_json_file, load_manifest, load_run_table_safe, ARTIFACT_ROOT, BENCHMARK_ROOT


def render(runs):
    st.header('Compare & Validation')
    st.caption('Observed research results · benchmark evidence · objective profiles')
    if not runs.empty:
        selected = st.multiselect('Compare runs',runs['path'].tolist(),default=runs['path'].head(2).tolist(),format_func=lambda p: Path(p).name)
        rows=[]
        for run in selected:
            manifest=load_manifest(run)
            base=load_json_file(Path(run)/'baseline_metrics.json',{})
            shortlist=load_run_table_safe(run,'candidate_shortlist.parquet')
            for label,values in [('baseline',base)]+[(f'trial {int(r.number)}',r.to_dict()) for _,r in shortlist.iterrows()]:
                rows.append(dict(run=Path(run).name,candidate=label,status=manifest.get('status'),
                                 profile=manifest.get('objective_profile_name'),
                                 **{k:values.get(k) for k in ['trades','sum_trade_returns_pct','profit_factor','normalized_pnl_usd','portfolio_return_pct','portfolio_max_drawdown_pct']}))
        st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
        st.caption('Compare identical signal sets and costs before interpreting improvements. The shortlist is Pareto-based; objectives are never collapsed into a weighted score.')
        with st.expander('Objective profiles & comparability'):
            for run in selected:
                st.write(Path(run).name)
                st.json(load_json_file(Path(run)/'objective_profile.json',{}))
                manifest=load_manifest(run)
                st.json({'model':manifest.get('model'),'data_hashes':manifest.get('data_hashes'),'baseline_params':manifest.get('baseline_params')})
    with st.expander('Benchmark smoke runs',expanded=True):
        st.caption('PF20250597 + signals · one trial each · uses real local market data')
        if st.button('Run benchmark smoke',type='primary'):
            with st.spinner('Running both benchmark fixtures…'):
                suite=run_benchmark_suite(n_trials=1,artifacts_root=str(ARTIFACT_ROOT),benchmark_root=str(BENCHMARK_ROOT))
            st.session_state['latest_suite']=str(suite)
        suites=benchmark_suite_options()
        if not suites.empty:
            suite=st.selectbox('Benchmark suite',suites['path'].tolist(),format_func=lambda p: Path(p).name)
            st.dataframe(load_run_table_safe(suite,'benchmark_runs.parquet'),hide_index=True,width='stretch')
    st.info('Walk-forward and out-of-sample validation are deferred. In-sample Pareto results do not establish predictive performance.')
