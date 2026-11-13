from pathlib import Path
import streamlit as st
from hypertrade.ui.state import run_options
from hypertrade.ui.components.selection import candidate_key
from hypertrade.ui.views import home, new_experiment, explorer, candidate, compare

PAGES = ['Обзор исследований','Новое исследование','Обзор оптимизации','Анализ кандидата','Сравнение и проверка']


def main():
    st.set_page_config(page_title='HyperTrade · Исследования',page_icon='◈',layout='wide')
    st.markdown('''<style>
    .stApp {background:#0B1220;color:#E8EEF8} [data-testid="stSidebar"] {background:#101B2D}
    [data-testid="stMetric"] {background:#18243A;border:1px solid #2B3C58;border-radius:12px;padding:16px}
    h1,h2,h3 {letter-spacing:-.025em} .stButton button[kind="primary"] {background:#37C7F3;color:#0B1220}
    [data-testid="stCaptionContainer"] {color:#B3C2D8} div[data-testid="stExpander"] {border-color:#2B3C58}
    </style>''',unsafe_allow_html=True)
    st.sidebar.markdown('## ◈ HYPERTRADE')
    st.sidebar.caption('ИССЛЕДОВАТЕЛЬСКАЯ ПЛАТФОРМА\n\nЭКСПЕРИМЕНТАЛЬНО · v8.5')
    page=st.sidebar.radio('Раздел',PAGES,label_visibility='collapsed')
    runs=run_options()
    run=None
    if not runs.empty:
        paths=runs['path'].tolist()
        pending_run = st.session_state.pop('pending_run',None)
        if pending_run in paths:
            st.session_state['active_run'] = pending_run
        if st.session_state.get('active_run') not in paths:
            st.session_state['active_run']=paths[0]
        run=st.sidebar.selectbox('Исследование',paths,key='active_run',format_func=lambda p: Path(p).name)
        pending=st.session_state.pop('pending_candidate',None)
        if pending is not None:
            st.session_state[candidate_key(run)]=pending
    st.sidebar.divider()
    st.sidebar.caption('Качество сигналов и портфель учитываются отдельно. Только исторические CSV-сигналы; реальной торговли нет.')
    if page==PAGES[0]: home.render(runs)
    elif page==PAGES[1]: new_experiment.render()
    elif page==PAGES[2]: explorer.render(run)
    elif page==PAGES[3]: candidate.render(run)
    else: compare.render(runs)


if __name__=='__main__':
    main()
