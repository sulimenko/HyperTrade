from hypertrade.ui.terms import HELP, table, metric_legend
from pathlib import Path
import pandas as pd
import streamlit as st
from hypertrade.api import run_benchmark_suite
from hypertrade.ui.state import benchmark_suite_options, load_json_file, load_manifest, load_run_table_safe, ARTIFACT_ROOT, BENCHMARK_ROOT


def render(runs):
    st.header('Сравнение и проверка')
    st.caption('Наблюдаемые исторические результаты · контрольные запуски · профили целей')
    metric_legend()
    if not runs.empty:
        selected = st.multiselect('Сравнить исследования',runs['path'].tolist(),default=runs['path'].head(2).tolist(),format_func=lambda p: Path(p).name, help='Для содержательного сравнения проверьте совпадение сигналов, издержек, истории рынка и профилей целей в технических деталях ниже.')
        rows=[]
        for run in selected:
            manifest=load_manifest(run)
            base=load_json_file(Path(run)/'baseline_metrics.json',{})
            shortlist=load_run_table_safe(run,'candidate_shortlist.parquet')
            for label,values in [('baseline',base)]+[(f'Испытание №{int(r.number)}',r.to_dict()) for _,r in shortlist.iterrows()]:
                rows.append(dict(run=Path(run).name,candidate=label,status=manifest.get('status'),
                                 profile=manifest.get('objective_profile_name'),
                                 **{k:values.get(k) for k in ['trades','sum_trade_returns_pct','profit_factor','normalized_pnl_usd','portfolio_return_pct','portfolio_max_drawdown_pct']}))
        table(pd.DataFrame(rows),hide_index=True,width='stretch')
        st.caption('Перед сравнением проверьте одинаковые сигналы и издержки. Кандидаты отобраны на основе Парето; цели не сводятся во взвешенную оценку.')
        with st.expander('Профили целей и сопоставимость · технические детали'):
            for run in selected:
                st.write(Path(run).name)
                st.json(load_json_file(Path(run)/'objective_profile.json',{}))
                manifest=load_manifest(run)
                st.json({'model':manifest.get('model'),'data_hashes':manifest.get('data_hashes'),'baseline_params':manifest.get('baseline_params')})
    with st.expander('Короткие контрольные запуски',expanded=True):
        st.caption('PF20250597 + signals · по одному испытанию · используются локальные рыночные данные')
        if st.button('Запустить короткую проверку',type='primary', help='Запускает по одному испытанию на каждом из двух контрольных CSV и сохраняет новые результаты. Это проверка работоспособности, не доказательство эффективности стратегии.'):
            with st.spinner('Выполняются два контрольных набора…'):
                suite=run_benchmark_suite(n_trials=1,artifacts_root=str(ARTIFACT_ROOT),benchmark_root=str(BENCHMARK_ROOT))
            st.session_state['latest_suite']=str(suite)
        suites=benchmark_suite_options()
        if not suites.empty:
            suite=st.selectbox('Набор контрольных запусков',suites['path'].tolist(),format_func=lambda p: Path(p).name)
            table(load_run_table_safe(suite,'benchmark_runs.parquet'),hide_index=True,width='stretch')
    st.info('Проверка на последующих временных окнах и отложенных данных не реализована. Результаты Парето на исследуемой истории не доказывают будущую эффективность.')
