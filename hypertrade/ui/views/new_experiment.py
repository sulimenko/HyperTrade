from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from hypertrade.api import optimize_run
from hypertrade.config.defaults import DEFAULT_FILTER_SEARCH_SPACE, DEFAULT_OBJECTIVE_PROFILE, OBJECTIVE_CATALOG
from hypertrade.config.schemas import CANDIDATE_POLICY_MODES, FilterSearchSpace, ObjectiveProfile, OptimizationRunConfig
from hypertrade.ui.state import ARTIFACT_ROOT, PROFILE_ROOT, DEFAULT_BENCHMARKS


def render():
    st.header('New Experiment')
    st.caption('NSGA-II / multi-objective Optuna · $10,000 normalized nominal · $100,000 initial portfolio')
    source = st.selectbox('Signal set', list(DEFAULT_BENCHMARKS) + ['Custom path', 'Upload CSV'])
    upload = None
    if source == 'Upload CSV':
        upload = st.file_uploader('External signals (; separated)', type=['csv'])
        path = None
    else:
        path = st.text_input('CSV path', value=DEFAULT_BENCHMARKS.get(source, ''))
    a,b,c=st.columns(3)
    label = a.text_input('Experiment label', value='research')
    n_trials = b.number_input('Trials', min_value=1, max_value=10000, value=20)
    seed = c.number_input('Seed', min_value=0, value=42)
    st.caption('Naive CSV dates are explicitly UTC; dd.mm.yyyy and ISO are accepted. Original files are preserved.')
    profiles = sorted(PROFILE_ROOT.glob('*.json'))
    profile_choice = st.selectbox('Objective profile', ['Default research profile']+[p.name for p in profiles])
    base = DEFAULT_OBJECTIVE_PROFILE.to_dict() if profile_choice == 'Default research profile' else json.loads((PROFILE_ROOT/profile_choice).read_text())
    with st.expander('Objectives & constraints', expanded=True):
        selected = st.multiselect('Pareto axes', list(OBJECTIVE_CATALOG), default=[o['metric'] for o in base['objectives']])
        directions = {}
        for metric in selected:
            directions[metric] = st.selectbox(OBJECTIVE_CATALOG[metric]['label'], ['maximize','minimize'],
                index=0 if OBJECTIVE_CATALOG[metric]['direction']=='maximize' else 1, key='objective_'+metric)
        policy = st.selectbox('Shortlist policy', sorted(CANDIDATE_POLICY_MODES),
                              index=sorted(CANDIDATE_POLICY_MODES).index(base.get('candidate_policy',{}).get('mode','utopia_distance')))
        constraints = st.data_editor(pd.DataFrame(base['constraints'], columns=['metric','min_value','max_value']),
                                    num_rows='dynamic', hide_index=True, width='stretch')
        st.caption('Research sum is percentage points across trades. Proxy drawdown uses a sequential normalized PnL curve; portfolio drawdown uses time equity.')
    with st.expander('Search parameters & transaction costs'):
        payload = asdict(DEFAULT_FILTER_SEARCH_SPACE)
        a,b,c = st.columns(3)
        for i, name in enumerate(['ema','rsi','adx','macd','bb','vwap','volume','atr','donchian','psar','ts','time_gate','cooldown']):
            payload[name+'_use'] = [a,b,c][i%3].checkbox(name.upper(), value=payload[name+'_use'])
        payload['commission'] = a.number_input('Commission · USD / share / side', min_value=0.0, value=0.02, step=0.01)
        payload['slippage'] = b.number_input('Slippage · fraction / side', min_value=0.0, max_value=0.1, value=0.0004, format='%.4f')
        payload['ranking_mode_choices'] = c.multiselect('Ranking modes', ['none','strength'], default=['none','strength'])
        # One complete editable schema; every exposed parameter is passed to the optimizer.
        raw = st.text_area('Full search space JSON (toggles/costs above override these fields)', json.dumps(asdict(DEFAULT_FILTER_SEARCH_SPACE),indent=2), height=240)
    with st.expander('Execution assumptions'):
        st.write('Filters/ranking may use the entry candle OHLC at its open. The existing bullish/bearish same-bar SL/TP heuristic is retained. Entry-bar PSAR waits for initialization; SL/TP and the known entry trailing level are active immediately. VWAP stays UTC-day on RTH-only candles.')
        st.write('Baseline has no filters or acceptance rules and uses the search minimum SL, TP, delay and holding period. Candidate improvements may include exit tuning. Both use identical nominal and costs.')
    if st.button('Start research', type='primary', width='stretch'):
        try:
            if not selected:
                raise ValueError('Select at least one objective')
            search_payload = json.loads(raw)
            for field in [k for k in payload if k.endswith('_use')] + ['commission','slippage','ranking_mode_choices']:
                search_payload[field] = payload[field]
            search = FilterSearchSpace.from_dict(search_payload)
            constraint_rows = json.loads(constraints.to_json(orient='records'))
            profile = ObjectiveProfile.from_dict(dict(name='ui_research', objectives=[dict(metric=m,direction=directions[m]) for m in selected],
                       constraints=[r for r in constraint_rows if r.get('metric')], candidate_policy={'mode':policy}))
            if upload is not None:
                data = upload.getvalue()
                cached = Path('data/cache/uploads')
                cached.mkdir(parents=True,exist_ok=True)
                path = str(cached/(hashlib.sha256(data).hexdigest()+'.csv'))
                Path(path).write_bytes(data)
            if not path or not Path(path).is_file():
                raise ValueError('Choose an existing CSV or upload a file')
            with st.spinner('Evaluating baseline and candidates…'):
                run = optimize_run(OptimizationRunConfig(signals_path=path,n_trials=int(n_trials),run_label=label,
                                    artifacts_root=str(ARTIFACT_ROOT),seed=int(seed)),profile,search)
            manifest = json.loads((run/'manifest.json').read_text())
            st.session_state['pending_run'] = str(run)
            if manifest['status'] == 'failed':
                st.error(f'Research failed: {run.name}. Diagnostics are saved in the run bundle.')
                st.json(json.loads((run/'errors.json').read_text()))
            else:
                st.success(f"{manifest['status']}: {run.name}")
            st.caption('Open Optimizer Explorer or Candidate Analysis from the sidebar.')
        except (ValueError, TypeError, OSError) as exc:
            st.error(str(exc))
