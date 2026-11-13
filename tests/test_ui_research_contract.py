"""Focused EXPERIMENTAL UI contracts; no simulation or fixture changes."""
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from hypertrade.config.defaults import DEFAULT_FILTER_SEARCH_SPACE
from hypertrade.ui.comparison import realized_comparison
from hypertrade.ui.terms import label as term
from hypertrade.ui.diagnostics import boundary_summary, prune_summary
from hypertrade.ui.research_profiles import (TRIAL_PRESETS, load_profile, objective_templates,
                                             research_profile, resolve_search, save_profile)


def test_exit_calendar_1659_vs_360_preserves_every_pnl_once():
    start, end = pd.Timestamp('2025-07-01', tz='UTC'), pd.Timestamp('2025-12-31', tz='UTC')
    frames = {}
    for label, count in [('Baseline', 1659), ('Trial 1', 360)]:
        # Out-of-order, simultaneous exits and entry-vs-close ordering differences.
        exits = pd.date_range('2025-08-01', periods=(count+1)//2, freq='3h', tz='UTC').repeat(2)[:count][::-1]
        frames[label] = pd.DataFrame({'exit_event_dt': exits, 'exit_dt': exits-pd.Timedelta(minutes=15),
                                      'normalized_pnl_usd': (np.arange(count) % 7 - 3)*1.25})
    originals = {label: frame.copy(deep=True) for label, frame in frames.items()}
    chart = realized_comparison(frames, [start, end])
    for label, trades in frames.items():
        line = chart[chart.source.eq(label)].set_index('datetime')['cumulative_normalized_pnl']
        assert (line.index.min(), line.index.max()) == (start, end)
        assert line.iloc[0] == 0
        assert line.iloc[-1] == pytest.approx(trades.normalized_pnl_usd.sum())
        expected = trades.groupby('exit_event_dt').normalized_pnl_usd.sum().sort_index().cumsum()
        pd.testing.assert_series_equal(line, expected.reindex(line.index).ffill().fillna(0), check_names=False)
        pd.testing.assert_frame_equal(trades, originals[label])
    assert chart.groupby('source').size().nunique() == 1
    legacy = frames['Trial 1'].drop(columns='exit_event_dt')
    fallback = realized_comparison({'Legacy': legacy}, [start, end])
    assert fallback.iloc[-1].cumulative_normalized_pnl == pytest.approx(legacy.normalized_pnl_usd.sum())
    invalid = legacy.drop(columns='exit_dt')
    with pytest.raises(ValueError):
        realized_comparison({'Invalid': invalid}, [start, end])


def test_presets_visual_json_and_full_profile_reach_launch(tmp_path, monkeypatch):
    from hypertrade.ui.views import new_experiment
    monkeypatch.setattr(new_experiment, 'PROFILE_ROOT', tmp_path/'profiles')
    monkeypatch.setattr(new_experiment, 'LEGACY_PROFILE_ROOT', tmp_path/'legacy')
    result = tmp_path/'run'
    result.mkdir()
    (result/'manifest.json').write_text(json.dumps({'status': 'no_valid_candidates'}))
    pd.DataFrame([{'state': 'PRUNED', 'user_attrs_constraint_violations': 'no_trades'}]).to_parquet(result/'trials.parquet')
    captured = []

    def launch(config, profile, search):
        captured.append((config, profile.to_dict(), asdict(search)))
        return result

    monkeypatch.setattr(new_experiment, 'optimize_run', launch)
    app = AppTest.from_string('from hypertrade.ui.views.new_experiment import render\nrender()', default_timeout=20).run()

    def widget(kind, label):
        return next(w for w in getattr(app, kind) if w.label == label)

    assert not app.exception
    assert widget('radio', 'Режим настроек').value == 'Основные настройки'
    widget('radio', 'Режим настроек').set_value('Расширенные настройки').run()
    widget('selectbox', 'Число испытаний').set_value(TRIAL_PRESETS[1]).run()
    widget('number_input', term('sl_min')).set_value(1.5)
    widget('number_input', term('tp_max')).set_value(16.0)
    widget('checkbox', 'ATR · выходы по волатильности').check().run()
    widget('number_input', term('atr_sl_min')).set_value(.5)
    widget('number_input', term('atr_tp_max')).set_value(4.0)
    widget('number_input', term('commission')).set_value(.03)
    widget('number_input', 'Начальное число генератора (seed)').set_value(123).run()
    widget('radio', 'Режим настроек').set_value('Основные настройки').run()
    widget('button', 'Запустить исследование').click().run()
    assert not app.exception and not app.error
    config, objective, payload = captured[-1]
    widget('radio', 'Режим настроек').set_value('Расширенные настройки').run()
    widget('button', 'Запустить исследование').click().run()
    assert captured[-1] == (config, objective, payload)
    assert config.n_trials == 100 and config.seed == 123
    assert (payload['sl_min'], payload['tp_max'], payload['atr_sl_min'], payload['atr_tp_max'], payload['commission']) == (1.5, 16, .5, 4, .03)
    widget('button', 'Сохранить профиль поиска').click().run()
    saved = load_profile(tmp_path/'profiles'/'research_pareto.json')
    assert saved['objective_profile'] == objective and saved['search_space'] == payload
    assert (saved['trial_preset'], saved['n_trials'], saved['seed']) == (TRIAL_PRESETS[1], 100, 123)
    widget('selectbox', 'Число испытаний').set_value(TRIAL_PRESETS[2]).run()
    widget('button', 'Запустить исследование').click().run()
    assert captured[-1][0].n_trials == 200
    # Switch away and load the named profile: all controls restore together.
    widget('selectbox', 'Профиль поиска · цели и все параметры').set_value('Сохранённый: research_pareto').run()
    assert widget('selectbox', 'Число испытаний').value == TRIAL_PRESETS[1]
    assert widget('number_input', 'Начальное число генератора (seed)').value == 123
    assert widget('number_input', term('atr_tp_max')).value == 4
    widget('radio', 'Редактор параметров поиска').set_value('Полный JSON').run()
    raw = dict(payload, sl_max=8.0, atr_sl_max=4.0)
    widget('text_area', 'Полный JSON параметров поиска').set_value(json.dumps(raw)).run()
    widget('button', 'Запустить исследование').click().run()
    assert not app.exception and not app.error
    assert captured[-1][0].n_trials == 100 and captured[-1][2] == raw
    widget('radio', 'Редактор параметров поиска').set_value('Визуальные поля').run()
    widget('radio', 'Редактор параметров поиска').set_value('Полный JSON').run()
    assert json.loads(widget('text_area', 'Полный JSON параметров поиска').value) == raw
    raw['atr_tp_step'] = 0
    widget('text_area', 'Полный JSON параметров поиска').set_value(json.dumps(raw)).run()
    assert app.error and widget('button', 'Запустить исследование').disabled
    for invalid in [{'sl_min': True}, {'atr_use': 'false'}, {'holding_minutes_step': 1.5}, {'sl_step': .3}, {'atr_sl_min': 0}]:
        with pytest.raises(ValueError):
            resolve_search({**asdict(DEFAULT_FILTER_SEARCH_SPACE), **invalid})
    # Legacy migration and edited directions/constraints/policy are lossless.
    objective['name'] = 'custom'
    objective['objectives'][0]['direction'] = 'minimize'
    objective['constraints'][0]['min_value'] = 30
    objective['candidate_policy']['mode'] = 'max_profit_factor'
    legacy = tmp_path/'legacy.json'
    legacy.write_text(json.dumps(objective))
    loaded = load_profile(legacy)
    assert loaded['objective_profile'] == objective
    full = research_profile(objective, payload, 'Вручную', 321, 123)
    assert load_profile(save_profile(tmp_path/'profiles', full)) == full
    app.run()
    widget('selectbox', 'Профиль поиска · цели и все параметры').set_value('Сохранённый: custom').run()
    widget('radio', 'Режим настроек').set_value('Основные настройки').run()
    widget('button', 'Запустить исследование').click().run()
    assert not app.exception and not app.error
    assert captured[-1][0].seed == 123 and captured[-1][0].n_trials == 321
    assert captured[-1][1] == objective and captured[-1][2] == payload
    widget('radio', 'Режим настроек').set_value('Расширенные настройки').run()
    widget('button', 'Запустить исследование').click().run()
    assert captured[-1] == captured[-2]
    assert len(objective_templates()['Доходность капитала · одна цель']['objective_profile']['objectives']) == 1


def test_shortlist_selection_navigation_and_factual_diagnostics(tmp_path):
    pd.DataFrame([{'number': 1}, {'number': 7}]).to_parquet(tmp_path/'pareto_trials.parquet')
    pd.DataFrame([{'number': 7}, {'number': 1}]).to_parquet(tmp_path/'candidate_shortlist.parquet')
    app = AppTest.from_string(f'''
import streamlit as st
from hypertrade.ui.components.selection import choose_candidate, choose_comparison
page = st.radio('Page', ['Explorer', 'Home', 'Candidate'])
if page != 'Home':
    number = choose_candidate({str(tmp_path)!r})
    choose_comparison({str(tmp_path)!r}, [1,7], number)
''').run()
    assert app.selectbox[0].value == 7
    app.selectbox[0].set_value(1).run()
    assert app.multiselect[0].value == [1]
    app.multiselect[0].set_value([7,1]).run()
    app.radio[0].set_value('Home').run()
    app.radio[0].set_value('Candidate').run()
    assert not app.exception
    assert app.selectbox[0].value == 1 and app.multiselect[0].value == [7,1]
    search = dict(sl_min=2, sl_max=7, atr_sl_min=.75, atr_sl_max=3, atr_tp_min=.5, atr_tp_max=3)
    table, count = boundary_summary(tmp_path, pd.DataFrame([{'number': 1, 'sl': 7}, {'number': 7, 'sl': 2}]), search)
    assert count == 2 and table.iloc[0]['Доля_на_границе'] == 1
    table, count = boundary_summary(tmp_path, pd.DataFrame([{'number': 7, 'atr_sl': 3, 'atr_tp': 1.5}]), search)
    assert count == 1 and table.iloc[0]['Единица'] == '× ATR'
    statuses, reasons = prune_summary(pd.DataFrame([
        {'state': 'COMPLETE'}, {'state': 'FAIL'}, {'state': 'PRUNED', 'user_attrs_constraint_violations': 'trades<min,undefined:profit_factor'},
        {'state': 'PRUNED', 'user_attrs_constraint_violations': 'undefined:profit_factor'}]))
    assert statuses.set_index('Статус')['Испытаний'].to_dict() == {'COMPLETE': 1, 'PRUNED': 2, 'FAIL': 1}
    assert reasons.set_index('Причина')['Испытаний'].to_dict() == {'undefined:profit_factor': 2, 'trades<min': 1}


def test_russian_five_view_render_keeps_plot_ids_and_selection(tmp_path, monkeypatch):
    """Reuse the existing synthetic bundle fixture; never launch a real search."""
    import base64
    from test_optimizer_end_to_end import _synthetic_backtest, _synthetic_signals
    from hypertrade.config.schemas import OptimizationRunConfig
    from hypertrade.optimization import runner
    from hypertrade.ui import app as ui_app
    from hypertrade.ui.components.selection import candidate_key
    from hypertrade.ui.views import compare, new_experiment

    monkeypatch.setattr(runner, 'load_external_signals', lambda _: _synthetic_signals())
    monkeypatch.setattr(runner, 'evaluate', _synthetic_backtest)
    run = runner.run_optimization(OptimizationRunConfig(
        signals_path='benchmarks/fixtures/PF20250597.csv', artifacts_root=str(tmp_path), n_trials=3, seed=7))
    before = {p: p.read_bytes() for p in run.iterdir() if p.is_file()}
    pareto = pd.read_parquet(run/'pareto_trials.parquet')
    numbers = pareto.number.astype(int).tolist()
    monkeypatch.setattr(ui_app, 'run_options', lambda: pd.DataFrame([dict(
        path=str(run), run_id=run.name, status='success', n_trials=3)]))
    monkeypatch.setattr(compare, 'benchmark_suite_options', lambda: pd.DataFrame())
    monkeypatch.setattr(new_experiment, 'PROFILE_ROOT', tmp_path/'profiles')
    monkeypatch.setattr(new_experiment, 'LEGACY_PROFILE_ROOT', tmp_path/'legacy')
    app = AppTest.from_string('from hypertrade.ui.app import main\nmain()', default_timeout=20).run()

    def widget(kind, label):
        return next(w for w in getattr(app, kind) if w.label == label)

    def figure(kind):
        return next(spec for plot in app.get('plotly_chart')
                    if (spec := json.loads(plot.proto.spec))['data'][0]['type'] == kind)

    for page in ui_app.PAGES:
        widget('radio', 'Раздел').set_value(page).run()
        assert not app.exception and not app.error
        assert app.header[0].value == page
        if page == 'Новое исследование':
            for name in ['Путь к CSV', 'Число испытаний', 'Начальное число генератора (seed)']:
                kind = 'text_input' if name == 'Путь к CSV' else 'selectbox' if name == 'Число испытаний' else 'number_input'
                assert widget(kind, name).proto.help
        if page == 'Обзор оптимизации':
            for mode, kind in [('3D', 'scatter3d'), ('2D', 'scatter')]:
                widget('selectbox', 'Размерность').set_value(mode).run()
                assert not app.exception
                fig = figure(kind)
                data = fig['data'][0]['customdata']
                ids = (np.frombuffer(base64.b64decode(data['bdata']), dtype=data['dtype']).tolist()
                       if isinstance(data, dict) else [row[0] for row in data])
                assert ids == numbers
                axes = fig['layout']['scene'] if mode == '3D' else fig['layout']
                assert axes['xaxis']['title']['text'] == term('portfolio_max_drawdown_pct')
                assert 'Номер испытания' in fig['data'][0]['hovertemplate']
            widget('selectbox', 'Выбранный кандидат · общий для страниц').set_value(numbers[-1]).run()
            overlays = numbers[:3]
            widget('multiselect', 'Кандидаты на графике · до 3 и исходная стратегия').set_value(overlays).run()
        if page == 'Анализ кандидата':
            assert widget('selectbox', 'Выбранный кандидат · общий для страниц').value == numbers[-1]
            assert widget('multiselect', 'Кандидаты на графике · до 3 и исходная стратегия').value == overlays
            for model in ['Исследовательская · нормированная', 'Портфельная · капитал']:
                widget('radio', 'Модель статистики').set_value(model).run()
                assert not app.exception and not app.error
                assert all(card.proto.help for card in app.metric)
            widget('radio', 'Модель статистики').set_value('Исследовательская · нормированная').run()
            widget('radio', 'Шкала сравнения').set_value('По порядку сделок').run()
            assert not app.exception and not app.error
    assert app.session_state[candidate_key(run)] == numbers[-1]
    assert before == {p: p.read_bytes() for p in run.iterdir() if p.is_file()}
