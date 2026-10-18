import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from hypertrade.ui.components.charts import show, line
from hypertrade.ui.components.selection import choose_candidate, choose_comparison, selected_table
from hypertrade.ui.state import load_run_table_safe, load_json_file
from hypertrade.ui.comparison import calendar_domain, realized_comparison
from hypertrade.ui.components.formula import formula_help


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
    numbers = choose_comparison(run, pareto['number'].astype(int).tolist(), number)
    formula_help()
    base_trades = load_run_table_safe(run, 'baseline_trades.parquet')
    all_trades = load_run_table_safe(run, 'trades.parquet')
    signal_stats = load_run_table_safe(run, 'baseline_signal_stats.parquet')
    period = None
    try:
        period = calendar_domain([base_trades, all_trades, signal_stats])
        st.caption(f'Общий период сигналов и закрытий (UTC): {period[0]} — {period[1]}')
        if not signal_stats.empty and 'datetime' in signal_stats:
            dates = pd.to_datetime(signal_stats['datetime'], utc=True)
            st.caption(f'Исследуемые сигналы (UTC): {dates.min()} — {dates.max()}')
    except ValueError as exc:
        st.warning(str(exc))
    counts = []
    for trial, row in [(None, baseline)] + [(n, pareto[pareto['number']==n].iloc[0].to_dict()) for n in dict.fromkeys([number]+numbers)]:
        stats = signal_stats if trial is None else selected_table(run, 'signal_stats.parquet', trial)
        input_count = int(stats['symbols_total'].sum()) if 'symbols_total' in stats and not stats.empty else None
        accepted = row.get('trades')
        pstats = load_run_table_safe(run, 'baseline_portfolio_signal_stats.parquet') if trial is None else selected_table(run, 'portfolio_signal_stats.parquet', trial)
        portfolio_input = int(pstats['symbols_total'].sum()) if 'symbols_total' in pstats and not pstats.empty else None
        portfolio_accepted = row.get('portfolio_trade_count')
        counts.append({'Стратегия': 'Baseline' if trial is None else f'Trial {trial}', 'Исходные сигналы research': input_count,
                       'Сделки research': accepted, 'Принято research · %': 100*accepted/input_count if input_count and accepted is not None else None,
                       'Исходные сигналы portfolio': portfolio_input, 'Сделки portfolio': portfolio_accepted,
                       'Принято portfolio · %': 100*portfolio_accepted/portfolio_input if portfolio_input and portfolio_accepted is not None else None})
    st.dataframe(pd.DataFrame(counts), hide_index=True, width='stretch')
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
    st.caption('Baseline: без фильтров и acceptance rules, минимальные процентные SL/TP, delay и hold из поиска. Сравнение включает настройку выходов кандидата и не измеряет чистый эффект индикаторов. Research использует фиксированные $10,000 на сделку без ограничения бюджета; portfolio учитывает доступный капитал. Настройки baseline — во вкладке Settings.')
    tab1,tab2,tab3,tab4=st.tabs(['Curves & comparison','Trade outcomes','Admission funnel','Settings'])
    with tab1:
        index_mode = research and st.radio('Шкала сравнения', ['Календарное время закрытия', 'По порядку сделок'], horizontal=True) == 'По порядку сделок'
        frames = []
        if research and not index_mode:
            by_label = {'Baseline': base_trades}
            by_label.update({f'Trial {n}': selected_table(run, 'trades.parquet', n) for n in numbers})
            try:
                manifest = load_json_file(Path(run)/'manifest.json', {})
                nominal = manifest.get('model', {}).get('normalized_notional_usd', baseline.get('normalized_notional_usd', 10000))
                comparison = realized_comparison(by_label, period or (), nominal)
                line(comparison, 'datetime', 'cumulative_normalized_pnl', 'Накопленный realized normalized PnL · USD · UTC', color='source', line_shape='hv', labels={'datetime':'Время закрытия · UTC', 'cumulative_normalized_pnl':'Накопленный PnL · USD', 'source':'Стратегия'})
            except (ValueError, KeyError, TypeError) as exc:
                st.error(f'Невозможно честно построить календарную кривую: {exc}')
            st.caption('PnL учитывается только в exit_event_dt; для старых записей без него используется exit_dt (время свечи, точность ниже). Одновременные закрытия суммируются один раз. Все линии имеют общий старт с нуля и конечное плато; между закрытиями — последнее значение без новых сделок. Sequential proxy drawdown в таблице сохраняет исходный порядок входов и не пересчитывается по этой кривой.')
        else:
            name = 'sequential.parquet' if research else 'equity.parquet'
            base = load_run_table_safe(run, 'baseline_'+name)
            curve = selected_table(run, name, number)
            for frame, label in [(base, 'Baseline')] + [(selected_table(run, name, n), f'Trial {n}') for n in numbers]:
                if not frame.empty:
                    frame = frame.copy()
                    frame['source'] = label
                    frames.append(frame)
            if frames:
                comparison = pd.concat(frames, ignore_index=True)
                if research:
                    line(comparison, 'sequence', 'cumulative_normalized_pnl', 'PnL по порядку входа сделок · USD', color='source', labels={'sequence':'Номер допущенной сделки', 'cumulative_normalized_pnl':'PnL · USD', 'source':'Стратегия'})
                    line(comparison, 'sequence', 'proxy_drawdown_usd', 'Sequential proxy drawdown · USD', color='source', labels={'sequence':'Номер допущенной сделки', 'proxy_drawdown_usd':'Proxy drawdown · USD', 'source':'Стратегия'})
                    st.caption('X — номер допущенной сделки, не календарное время. Линии разной длины при разном количестве сделок; это не усечение backtest. Число сделок каждой стратегии показано выше. Порядок и значения сохранённого sequential.parquet не изменены.')
                else:
                    # Keep every saved mark and event phase, including duplicate times.
                    # Only the shared display domain changes; no reindex or revaluation.
                    domain = calendar_domain([comparison], period or ())
                    for field, title in [('equity_usd','Portfolio equity · USD'), ('drawdown_pct','Portfolio drawdown · % of peak')]:
                        fig = px.line(comparison, x='datetime', y=field, color='source', title=title)
                        fig.update_xaxes(range=list(domain), title='Календарное время · UTC')
                        show(fig)
                    st.caption('Сохранённые mark-to-market, закрытия и порядок событий показаны без пересчёта. Общая область дат — только настройка отображения; отсутствующие equity-точки не заменяются синтетическими прибылями.')
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
