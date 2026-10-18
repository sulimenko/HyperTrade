from pathlib import Path
import streamlit as st
from hypertrade.ui.state import load_run_table_safe


def candidate_key(run):
    return f'selected_candidate_{Path(run).name}'


def persist_widget_value(storage_key, widget_key):
    # A regular session key survives Streamlit's cleanup of off-page widget keys.
    st.session_state[storage_key] = st.session_state[widget_key]


def choose_candidate(run):
    frame = load_run_table_safe(run, 'pareto_trials.parquet')
    if frame.empty:
        st.info('This run has no valid Pareto candidates. Inspect trial statuses and diagnostics in Optimizer Explorer.')
        return None
    numbers = frame['number'].astype(int).tolist()
    storage_key = candidate_key(run)
    if st.session_state.get(storage_key) not in numbers:
        automatic = load_run_table_safe(run, 'candidate_shortlist.parquet')
        preferred = [int(n) for n in automatic.get('number', []) if int(n) in numbers]
        st.session_state[storage_key] = preferred[0] if preferred else numbers[0]
    widget_key = storage_key+'_widget'
    st.session_state[widget_key] = st.session_state[storage_key]
    return st.selectbox('Selected candidate · linked across views', numbers, key=widget_key,
                        format_func=lambda n: f'Trial {n}', on_change=persist_widget_value,
                        args=(storage_key,widget_key))


def choose_shortlist(run, numbers, automatic):
    storage_key = 'shortlist_'+Path(run).name
    widget_key = storage_key+'_widget'
    st.session_state[widget_key] = st.session_state.get(storage_key,automatic)
    st.multiselect('Research shortlist', numbers, key=widget_key, on_change=persist_widget_value,
                   args=(storage_key,widget_key))
    if storage_key not in st.session_state:
        st.session_state[storage_key] = list(automatic)


def selected_table(run, name, number):
    frame = load_run_table_safe(run, name)
    if 'candidate_trial' in frame:
        return frame[frame['candidate_trial'] == number].copy()
    return frame.iloc[:0]


def choose_comparison(run, numbers, current):
    storage_key = 'comparison_'+Path(run).name
    widget_key = storage_key+'_widget'
    saved = st.session_state.get(storage_key, [current])
    previous = st.session_state.get(storage_key+'_primary', current)
    # The initial single curve follows the primary selector. Explicit overlay
    # choices remain independent and survive navigation.
    if saved == [previous] and previous != current:
        saved = [current]
        st.session_state[storage_key] = saved
    st.session_state[storage_key+'_primary'] = current
    st.session_state[widget_key] = [n for n in saved if n in numbers][:3]
    result = st.multiselect('Кандидаты на графике · до 3 + baseline', numbers, key=widget_key,
                            max_selections=3, format_func=lambda n: f'Trial {n}',
                            on_change=persist_widget_value, args=(storage_key, widget_key))
    if storage_key not in st.session_state:
        st.session_state[storage_key] = list(result)
    return result
