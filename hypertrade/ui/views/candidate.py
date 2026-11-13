from hypertrade.ui.terms import HELP, LABELS, label as term, metric_help, display_frame, table, metric_legend
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
    st.header('Анализ кандидата')
    if not run:
        st.info('Выберите исследование.')
        return
    number = choose_candidate(run)
    if number is None:
        return
    pareto = load_run_table_safe(run,'pareto_trials.parquet')
    metrics = pareto[pareto['number']==number].iloc[0].to_dict()
    baseline = load_json_file(Path(run)/'baseline_metrics.json',{})
    numbers = choose_comparison(run, pareto['number'].astype(int).tolist(), number)
    formula_help()
    metric_legend()
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
        counts.append({'Стратегия': 'Исходная стратегия' if trial is None else f'Испытание №{trial}', 'Исходные сигналы · исследование': input_count,
                       'Сделки · исследование': accepted, 'Принято в исследование · %': 100*accepted/input_count if input_count and accepted is not None else None,
                       'Исходные сигналы · портфель': portfolio_input, 'Сделки · портфель': portfolio_accepted,
                       'Принято в портфель · %': 100*portfolio_accepted/portfolio_input if portfolio_input and portfolio_accepted is not None else None})
    table(pd.DataFrame(counts), hide_index=True, width='stretch')
    model = st.radio('Модель статистики',['Исследовательская · нормированная','Портфельная · капитал'],horizontal=True, help=HELP['model'])
    research = model.startswith('Исследовательская')
    if research:
        st.caption('Фиксированный номинал 10 000 USD на сделку, бюджет не ограничен. Сумма доходностей сделок не равна доходности капитала. Просадка по порядку сделок — исследовательская оценка.')
        fields = ['sum_trade_returns_pct','profit_factor','normalized_pnl_usd']
    else:
        st.caption('Начальный капитал 100 000 USD · до 10 000 USD номинала позиции · целые акции · без плеча · 100% обеспечения коротких позиций, выручка заблокирована. Капитал включает нереализованную прибыль на общей оси 15-минутных свечей.')
        fields = ['portfolio_return_pct','portfolio_max_drawdown_pct','portfolio_trade_count']
    cols=st.columns(3)
    for col,field in zip(cols,fields):
        value = metrics.get(field)
        ref = baseline.get(field)
        col.metric(term(field), 'не определено' if value is None or pd.isna(value) else f'{value:,.2f}',
                   delta=f'{value-ref:+,.2f} к исходной стратегии' if value is not None and ref is not None else None,
                   delta_color='inverse' if 'drawdown' in field else 'normal', help=metric_help(field))
    if research:
        for col,field in zip(st.columns(2), ['cvar_loss_5_pct', 'avg_hold_minutes']):
            value = metrics.get(field)
            col.metric(term(field), 'не определено' if value is None or pd.isna(value) else f'{value:,.2f}',
                       help=metric_help(field))
    st.caption('Исходная стратегия: без фильтров и правил допуска, минимальные процентные SL/TP, задержка и срок удержания из поиска. Сравнение включает настройку выходов кандидата и не измеряет чистый эффект индикаторов. Исследовательская модель использует 10 000 USD на сделку без лимита бюджета; портфельная учитывает доступный капитал. Настройки исходной стратегии — во вкладке «Настройки».')
    tab1,tab2,tab3,tab4=st.tabs(['Кривые и сравнение','Результаты сделок','Допуск и отказы','Настройки'])
    with tab1:
        index_mode = research and st.radio('Шкала сравнения', ['Календарное время закрытия', 'По порядку сделок'], horizontal=True, help=HELP['time']) == 'По порядку сделок'
        frames = []
        if research and not index_mode:
            by_label = {'Исходная стратегия': base_trades}
            by_label.update({f'Испытание №{n}': selected_table(run, 'trades.parquet', n) for n in numbers})
            try:
                manifest = load_json_file(Path(run)/'manifest.json', {})
                nominal = manifest.get('model', {}).get('normalized_notional_usd', baseline.get('normalized_notional_usd', 10000))
                comparison = realized_comparison(by_label, period or (), nominal)
                line(comparison, 'datetime', 'cumulative_normalized_pnl', 'Накопленная реализованная прибыль · USD · UTC', color='source', line_shape='hv', labels={'datetime':'Время закрытия · UTC', 'cumulative_normalized_pnl':'Накопленная прибыль · USD', 'source':'Стратегия'})
            except (ValueError, KeyError, TypeError) as exc:
                st.error(f'Невозможно честно построить календарную кривую: {exc}')
            st.caption('PnL учитывается только в exit_event_dt; для старых записей без него используется exit_dt (время свечи, точность ниже). Одновременные закрытия суммируются один раз. Все линии имеют общий старт с нуля и конечное плато; между закрытиями — последнее значение без новых сделок. Просадка по сделкам в таблице сохраняет исходный порядок входов и не пересчитывается по этой кривой.')
        else:
            name = 'sequential.parquet' if research else 'equity.parquet'
            base = load_run_table_safe(run, 'baseline_'+name)
            curve = selected_table(run, name, number)
            for frame, label in [(base, 'Исходная стратегия')] + [(selected_table(run, name, n), f'Испытание №{n}') for n in numbers]:
                if not frame.empty:
                    frame = frame.copy()
                    frame['source'] = label
                    frames.append(frame)
            if frames:
                comparison = pd.concat(frames, ignore_index=True)
                if research:
                    line(comparison, 'sequence', 'cumulative_normalized_pnl', 'Прибыль по порядку входа сделок · USD', color='source', labels={'sequence':'Номер допущенной сделки', 'cumulative_normalized_pnl':'PnL · USD', 'source':'Стратегия'})
                    line(comparison, 'sequence', 'proxy_drawdown_usd', 'Просадка по порядку сделок · USD', color='source', labels={'sequence':'Номер допущенной сделки', 'proxy_drawdown_usd':'Просадка по сделкам · USD', 'source':'Стратегия'})
                    st.caption('X — номер допущенной сделки, не календарное время. Линии разной длины при разном количестве сделок; это не усечение исторического расчёта. Число сделок каждой стратегии показано выше. Порядок и значения сохранённого sequential.parquet не изменены.')
                else:
                    # Keep every saved mark and event phase, including duplicate times.
                    # Only the shared display domain changes; no reindex or revaluation.
                    domain = calendar_domain([comparison], period or ())
                    for field, title in [('equity_usd','Капитал · USD'), ('drawdown_pct','Просадка капитала · % от пика')]:
                        fig = px.line(comparison, x='datetime', y=field, color='source', title=title, labels=LABELS)
                        fig.update_xaxes(range=list(domain), title='Календарное время · UTC')
                        show(fig)
                    st.caption('Сохранённые рыночные оценки, закрытия и порядок событий показаны без пересчёта. Общая область дат — только настройка отображения; отсутствующие точки капитала не заменяются синтетическими прибылями.')
        fields = ['trades','sum_trade_returns_pct','mean_trade_return_pct','median_trade_return_pct','win_rate_pct',
                  'profit_factor','normalized_pnl_usd','proxy_drawdown_usd','cvar_loss_5_pct','avg_hold_minutes'] if research else ['portfolio_trade_count','portfolio_pnl_usd','portfolio_return_pct','portfolio_max_drawdown_pct']
        comparison = pd.DataFrame([dict(metric=k,baseline=baseline.get(k),filtered=metrics.get(k)) for k in fields])
        table(comparison,hide_index=True,width='stretch')
        if not research:
            with st.expander('Средства, позиции и обеспечение'):
                table(curve,hide_index=True,width='stretch')
    with tab2:
        trades = selected_table(run,'trades.parquet' if research else 'portfolio_trades.parquet',number)
        if trades.empty:
            st.info('В этой модели нет допущенных сделок.')
        else:
            shown_trades = display_frame(trades)
            fig = px.histogram(shown_trades,x='return_pct',color='exit_reason',nbins=40,title='Распределение чистых доходностей сделок · %', labels=LABELS)
            fig.update_yaxes(title='Число сделок')
            fig.update_traces(hovertemplate='Чистая доходность сделки · %: %{x}<br>Число сделок: %{y}<extra>%{fullData.name}</extra>')
            show(fig)
            show(px.scatter(shown_trades,x='entry_dt',y='return_pct',color='exit_reason',hover_data=['symbol','direction'],title='Результаты сделок', labels=LABELS))
            table(trades,hide_index=True,width='stretch')
        st.caption('Цены включают проскальзывание по направлению сделки. Чистая прибыль на акцию указана в USD, комиссия — USD/акция/сторона на входе и выходе. Портфельная прибыль позиции равна чистой прибыли на акцию, умноженной на целое число акций.')
    with tab3:
        st.caption('Причины окончательного отказа', help=HELP['rejections'])
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
            show(px.funnel(display_frame(funnel),x='candidates',y='stage',title='Этапы окончательного допуска', labels=LABELS))
            counts = stats[count_cols].fillna(0).sum().rename_axis('reason').reset_index(name='count')
            show(px.bar(display_frame(counts),x='count',y='reason',orientation='h',title='Число отказов по окончательной причине', labels=LABELS))
        details = []
        for raw in stats.get('rejection_details',[]):
            details.extend(json.loads(raw))
        table(pd.DataFrame(details),hide_index=True,width='stretch')
        st.caption('Проверка фильтров → проверка паузы → ранжирование → лимит направления → проверка доступного капитала / следующий кандидат → резерв и обновление паузы. Исследовательский и портфельный допуск учитываются независимо.')
    with tab4:
        parameters = load_json_file(Path(run)/'candidate_parameters.json',{})
        st.write('Настройки кандидата · технические детали')
        st.json(parameters.get(str(number),{}))
        manifest = load_json_file(Path(run)/'manifest.json',{})
        st.write('Настройки исходной стратегии · технические детали')
        st.json(manifest.get('baseline_params',{}))
        st.write('Допущения исполнения · сохранённые технические сведения')
        st.json(manifest.get('assumptions',[]))
        st.caption('ЭКСПЕРИМЕНТАЛЬНО. Проверки на будущих окнах и отложенных данных нет. Покрытие истории и отложенная очистка кэша индикаторов ограничивают воспроизводимость после обновления источников. Фильтры могут использовать текущую 15-минутную свечу при входе на её открытии; риск заглядывания вперёд сохраняется.')
