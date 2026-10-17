import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from hypertrade.ui.components.charts import show, line
from hypertrade.ui.components.selection import choose_candidate, selected_table
from hypertrade.ui.state import load_run_table_safe, load_json_file


def render(run):
    st.header('Candidate Analysis')
    if not run:
        st.info('Select an experiment.')
        return
    number = choose_candidate(run)
    if number is None:
        return
    pareto = load_run_table_safe(run,'pareto_trials.parquet')
    metrics = pareto[pareto['number']==number].iloc[0].to_dict()
    baseline = load_json_file(Path(run)/'baseline_metrics.json',{})
    model = st.radio('Statistics model',['Research / Normalized','Capital / Portfolio equity'],horizontal=True)
    research = model.startswith('Research')
    if research:
        st.caption('Fixed $10,000 nominal per trade, unlimited budget. Sum of trade returns is not a return on invested capital. Sequential drawdown is a proxy.')
        fields = [('sum_trade_returns_pct','Sum · percentage points'),('profit_factor','Profit factor'),('normalized_pnl_usd','Normalized PnL · USD')]
    else:
        st.caption('Initial $100,000 · maximum $10,000 gross notional · integer shares · no leverage · 100% short collateral, locked proceeds. Portfolio equity includes unrealized PnL on a shared 15-minute candle axis.')
        fields = [('portfolio_return_pct','Portfolio return · %'),('portfolio_max_drawdown_pct','Peak equity drawdown · %'),('portfolio_trade_count','Portfolio trades')]
    cols=st.columns(3)
    for col,(field,label) in zip(cols,fields):
        value = metrics.get(field)
        ref = baseline.get(field)
        col.metric(label, 'undefined' if value is None or pd.isna(value) else f'{value:,.2f}',
                   delta=f'{value-ref:+,.2f} vs baseline' if value is not None and ref is not None else None,
                   delta_color='inverse' if 'drawdown' in field else 'normal')
    st.caption('Baseline has no filters or acceptance rules; its fixed exit settings are in Settings. Differences may include optimized exits as well as filters.')
    tab1,tab2,tab3,tab4=st.tabs(['Curves & comparison','Trade outcomes','Admission funnel','Settings'])
    with tab1:
        if research:
            curve = selected_table(run,'sequential.parquet',number)
            base = load_run_table_safe(run,'baseline_sequential.parquet')
            y1,y2,x = 'cumulative_normalized_pnl','proxy_drawdown_usd','sequence'
        else:
            curve = selected_table(run,'equity.parquet',number)
            base = load_run_table_safe(run,'baseline_equity.parquet')
            y1,y2,x = 'equity_usd','drawdown_pct','datetime'
        frames=[]
        for frame,label in [(curve,'Filtered'),(base,'Baseline')]:
            if not frame.empty:
                frame=frame.copy();frame['source']=label;frames.append(frame)
        if frames:
            comparison = pd.concat(frames,ignore_index=True)
            line(comparison,x,y1,'Sequential normalized PnL · USD' if research else 'Time equity · USD',color='source')
            line(comparison,x,y2,'Sequential proxy drawdown · USD' if research else 'Portfolio drawdown · % of peak',color='source')
        fields = ['trades','sum_trade_returns_pct','mean_trade_return_pct','median_trade_return_pct','win_rate_pct',
                  'profit_factor','normalized_pnl_usd','proxy_drawdown_usd'] if research else ['portfolio_trade_count','portfolio_pnl_usd','portfolio_return_pct','portfolio_max_drawdown_pct']
        comparison = pd.DataFrame([dict(metric=k,baseline=baseline.get(k),filtered=metrics.get(k)) for k in fields])
        st.dataframe(comparison,hide_index=True,width='stretch')
        if not research:
            with st.expander('Cash, securities & collateral ledger'):
                st.dataframe(curve,hide_index=True,width='stretch')
    with tab2:
        trades = selected_table(run,'trades.parquet' if research else 'portfolio_trades.parquet',number)
        if trades.empty:
            st.info('No admitted trades in this model.')
        else:
            show(px.histogram(trades,x='return_pct',color='exit_reason',nbins=40,title='Net trade return distribution · %'))
            show(px.scatter(trades,x='entry_dt',y='return_pct',color='exit_reason',hover_data=['symbol','direction'],title='Trade outcomes'))
            st.dataframe(trades,hide_index=True,width='stretch')
        st.caption('Prices include directional slippage. pnl is net USD per share; commission is USD/share/side, applied at entry and exit. Portfolio cash_pnl_usd multiplies net per-share PnL by integer quantity.')
    with tab3:
        stats = selected_table(run,'signal_stats.parquet' if research else 'portfolio_signal_stats.parquet',number)
        count_cols = [c for c in stats if c.startswith('rejected_')]
        if count_cols:
            total = int(stats['symbols_traded'].sum()+stats['symbols_rejected'].sum())
            reasons = stats[count_cols].fillna(0).sum()
            technical_rejects = sum(v for k,v in reasons.items() if k not in ['rejected_cooldown','rejected_side_cap','rejected_insufficient_capital'])
            technical = total-int(technical_rejects)
            cooldown = technical-int(reasons.get('rejected_cooldown',0))
            capped = cooldown-int(reasons.get('rejected_side_cap',0))
            admitted = capped-int(reasons.get('rejected_insufficient_capital',0))
            funnel = pd.DataFrame({'stage':['Input','Technical eligibility','Cooldown eligibility','Side cap eligibility','Final admitted'],
                                   'candidates':[total,technical,cooldown,capped,admitted]})
            show(px.funnel(funnel,x='candidates',y='stage',title='Final-admission funnel'))
            counts = stats[count_cols].fillna(0).sum().rename_axis('reason').reset_index(name='count')
            show(px.bar(counts,x='count',y='reason',orientation='h',title='Rejection counts by final reason'))
        details = []
        for raw in stats.get('rejection_details',[]):
            details.extend(json.loads(raw))
        st.dataframe(pd.DataFrame(details),hide_index=True,width='stretch')
        st.caption('Eligibility → read-only cooldown → ranking → side cap → affordable admission / next eligible → reserve + cooldown update. Research and portfolio maintain independent final-admission state.')
    with tab4:
        parameters = load_json_file(Path(run)/'candidate_parameters.json',{})
        st.write('Candidate settings')
        st.json(parameters.get(str(number),{}))
        manifest = load_json_file(Path(run)/'manifest.json',{})
        st.write('Baseline settings')
        st.json(manifest.get('baseline_params',{}))
        st.write('Execution assumptions')
        st.json(manifest.get('assumptions',[]))
        st.caption('EXPERIMENTAL. No walk-forward/OOS validation is implemented. Market-data coverage and deferred indicator cache invalidation limit reproducibility across source updates.')
