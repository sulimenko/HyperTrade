from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import time
import traceback
import optuna
import pandas as pd
from hypertrade.config.defaults import DEFAULT_FILTER_SEARCH_SPACE, DEFAULT_OBJECTIVE_PROFILE
from hypertrade.config.schemas import FilterSearchSpace, ObjectiveProfile, OptimizationRunConfig
from hypertrade.config.strategy import StrategyParams
from hypertrade.data.signals import load_external_signals
from hypertrade.experiments.registry import ExperimentRegistry
from hypertrade.experiments.artifacts import REQUIRED_RUN_ARTIFACTS
from hypertrade.optimization.search_space import build_strategy_params
from hypertrade.optimization.pareto import constraint_violations, pick_preferred_candidates, _objective_values
from hypertrade.simulation.backtest import evaluate

def _trial_frame(study: optuna.study.Study, profile: ObjectiveProfile) -> pd.DataFrame:
    from hypertrade.config.defaults import OBJECTIVE_CATALOG
    frame = study.trials_dataframe(attrs=('number','state','values','params','user_attrs'))
    for index, objective in enumerate(profile.objectives):
        column = f'values_{index}'
        if column in frame:
            frame[objective.metric] = frame[column]
    for metric in OBJECTIVE_CATALOG:
        attr = 'user_attrs_'+metric
        if attr in frame:
            frame[metric] = pd.to_numeric(frame[attr],errors='coerce')
    return frame


def run_optimization(run_config: OptimizationRunConfig, objective_profile: ObjectiveProfile | None = None,
                     search_space: FilterSearchSpace | None = None) -> Path:
    registry = ExperimentRegistry(run_config.artifacts_root)
    label = run_config.run_label or run_config.study_name or run_config.benchmark_name or Path(run_config.signals_path).stem
    run_dir = registry.create_run_dir(label)
    started = datetime.now(timezone.utc)
    monotonic = time.perf_counter()
    errors, logs = [], []
    environment = registry.build_environment_snapshot()
    search = search_space or DEFAULT_FILTER_SEARCH_SPACE
    profile = objective_profile or DEFAULT_OBJECTIVE_PROFILE
    study = None
    baseline = None
    candidate_results = {}
    trials = pd.DataFrame()
    pareto = pd.DataFrame()
    shortlist = pd.DataFrame()
    candidate_params = {}
    baseline_params = StrategyParams(sl=search.sl_min, tp=search.tp_min, delay_open=search.delay_open_min,
                                     holding_minutes=search.holding_minutes_min, commission=search.commission,
                                     slippage=search.slippage)
    signals = []
    status = 'failed'
    try:
        if objective_profile is None and run_config.objective_profile_path:
            profile = ObjectiveProfile.load(run_config.objective_profile_path)
        from hypertrade.config.defaults import OBJECTIVE_CATALOG
        for metric in [o.metric for o in profile.objectives] + [c.metric for c in profile.constraints]:
            if metric not in OBJECTIVE_CATALOG:
                raise ValueError(f'Unknown objective/constraint metric: {metric}')
        if not profile.objectives or run_config.n_trials < 1:
            raise ValueError('At least one objective and trial are required')
        profile.candidate_policy.normalized_mode()
        signals = load_external_signals(run_config.signals_path)
        study = optuna.create_study(study_name=run_config.study_name or label, directions=profile.directions(),
                                    sampler=optuna.samplers.NSGAIISampler(seed=run_config.seed),
                                    storage=f"sqlite:///{(run_dir/'study.sqlite3').resolve()}")
        baseline = evaluate(signals, baseline_params)

        def objective(trial):
            try:
                params = build_strategy_params(trial, search)
                result = evaluate(signals, params)
                metrics = result.metrics
                trial.set_user_attr('trade_count', len(result.trades))
                for key, value in metrics.items():
                    trial.set_user_attr(key, value)
                violations = constraint_violations(metrics, profile)
                undefined = [o.metric for o in profile.objectives if metrics.get(o.metric) is None or not math.isfinite(float(metrics[o.metric]))]
                if not result.trades or violations or undefined:
                    trial.set_user_attr('constraint_violations', ','.join(violations+['undefined:'+m for m in undefined]) or 'no_trades')
                    raise optuna.TrialPruned()
                # The entire final Pareto frontier is replayed; no all-trial trade memory growth.
                return _objective_values(metrics, profile)
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                diagnostic = dict(trial=trial.number, error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
                errors.append(diagnostic)
                trial.set_user_attr('error', diagnostic['error'])
                trial.set_user_attr('traceback', diagnostic['traceback'])
                logs.append(diagnostic['traceback'])
                raise

        study.optimize(objective, n_trials=run_config.n_trials, catch=(Exception,))
        trials = _trial_frame(study, profile)
        rows = []
        for trial in study.best_trials:
            if trial.user_attrs.get('constraint_violations'):
                continue
            rows.append({'number': trial.number, **trial.params, **trial.user_attrs,
                         **{o.metric: trial.values[i] for i, o in enumerate(profile.objectives)}})
        pareto = pd.DataFrame(rows)
        shortlist = pick_preferred_candidates(pareto, profile, limit=5)
        for number in pareto.get('number', []):
            trial = study.trials[int(number)]
            params = build_strategy_params(trial, search)
            candidate_params[str(int(number))] = asdict(params)
            candidate_results[int(number)] = evaluate(signals, params)
        status = 'failed' if errors else ('success' if not shortlist.empty else 'no_valid_candidates')
    except Exception as exc:
        errors.append(dict(trial=None, error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()))
        logs.append(errors[-1]['traceback'])
        if study is not None:
            trials = _trial_frame(study, profile)

    def result_table(result, name):
        if result is None:
            return pd.DataFrame()
        value = getattr(result, name)
        return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)

    table_fields = {'trades': 'trades', 'signal_stats': 'signal_stats', 'portfolio_trades': 'portfolio_trades',
                    'portfolio_signal_stats': 'portfolio_signal_stats', 'equity': 'equity', 'sequential': 'sequential'}
    for file_name, field in table_fields.items():
        frames = []
        for number, result in candidate_results.items():
            frame = result_table(result, field)
            if not frame.empty:
                frame['candidate_trial'] = number
                frames.append(frame)
        registry.save_table(run_dir/f'{file_name}.parquet', pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
    for name, field in {'trades': 'trades', 'portfolio_trades': 'portfolio_trades', 'signal_stats': 'signal_stats', 'portfolio_signal_stats': 'portfolio_signal_stats',
                        'equity': 'equity', 'sequential': 'sequential'}.items():
        registry.save_table(run_dir/f'baseline_{name}.parquet', result_table(baseline, field))
    registry.save_study_tables(run_dir, trials, pareto, shortlist)
    # A malformed input can fail before creating an Optuna study; still retain a valid empty DB.
    if not (run_dir/'study.sqlite3').exists():
        import sqlite3
        sqlite3.connect(run_dir/'study.sqlite3').close()
    summary = dict(status=status, valid_trial_count=int((trials.get('state', pd.Series(dtype=str)) == 'COMPLETE').sum()),
                   failed_trial_count=int((trials.get('state', pd.Series(dtype=str)) == 'FAIL').sum()),
                   completed_trial_count=int((trials.get('state', pd.Series(dtype=str)) == 'COMPLETE').sum()),
                   pruned_trial_count=int((trials.get('state', pd.Series(dtype=str)) == 'PRUNED').sum()),
                   pareto_candidate_count=len(pareto), shortlist_candidate_count=len(shortlist),
                   baseline_status='success' if baseline and baseline.trades else ('no_valid_candidates' if baseline else 'failed'),
                   baseline=baseline.metrics if baseline else {})
    finished = datetime.now(timezone.utc)
    benchmark = dict(benchmark_name=run_config.benchmark_name or Path(run_config.signals_path).stem,
                     signals_path=run_config.signals_path, signals_sha256=registry.file_sha256(run_config.signals_path))
    source_files = []
    symbols = {symbol for signal in signals for side in ('long','short') for symbol in signal.get(side,[])}
    for symbol in sorted(symbols):
        for directory in ('data/ohlc','data/indicators'):
            source = Path(directory)/f'{symbol}.parquet'
            if source.is_file():
                stat = source.stat()
                source_files.append(dict(path=str(source),size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
    registry.save_json(run_dir/'data_snapshot.json', {'sources':source_files,'identity':'size and mtime, not a content hash; cache invalidation deferred'})
    manifest = dict(run_id=run_dir.name, run_dir=str(run_dir), artifact_version=3, status=status, behavior='EXPERIMENTAL',
                    created_at=started.isoformat(), finished_at=finished.isoformat(), duration_seconds=round(time.perf_counter()-monotonic,3),
                    run_label=label, benchmark_name=benchmark['benchmark_name'], signals_path=run_config.signals_path,
                    study_name=study.study_name if study else label, n_trials=run_config.n_trials, seed=run_config.seed,
                    objective_profile_name=profile.name, objective_profile_path=run_config.objective_profile_path,
                    config_hashes={k: registry.payload_sha256(v) for k,v in {'run_config_sha256': asdict(run_config),
                                  'objective_profile_sha256': profile.to_dict(), 'search_space_sha256': asdict(search)}.items()},
                    data_hashes={'signals_sha256': benchmark['signals_sha256']}, environment=environment,
                    model=dict(normalized_notional_usd=10000, initial_capital_usd=100000, maximum_position_notional_usd=10000,
                               short_collateral_fraction=1.0, commission_unit='USD/share/side', slippage_unit='fraction/side'),
                    baseline_params=asdict(baseline_params),
                    assumptions=['Entry OHLC may inform filters/ranking at entry open', 'Existing bullish/bearish same-bar SL/TP priority',
                                 'RTH input, UTC-day VWAP', 'Intrabar exits settle at candle close; gap/time exits at open; zero-duration exits settle after the entry batch',
                                 'Fixed baseline uses search minimum SL/TP/delay/hold; candidates tune exits and rules'])
    for name, payload in [('manifest', manifest), ('objective_profile', profile.to_dict()), ('search_space', asdict(search))]:
        registry.save_json(run_dir/f'{name}.json', payload)
        registry.save_yaml(run_dir/f'{name}.yaml', payload)
    for name, payload in [('metrics_summary', summary), ('baseline_metrics', baseline.metrics if baseline else {}),
                          ('benchmark_profile', benchmark), ('environment', environment), ('errors', {'errors': errors})]:
        registry.save_json(run_dir/f'{name}.json', payload)
    logs.append(json.dumps({'status': status, 'trials': len(trials), 'baseline': summary['baseline_status']}))
    registry.save_text(run_dir/'logs.txt', '\n'.join(logs))
    # Index excludes its own hash (a self-hash cannot be consistent).
    registry.save_json(run_dir/'candidate_parameters.json', candidate_params)
    index = registry.artifact_inventory(run_dir)
    index.append({'name':'artifact_index.json','sha256':None,'note':'self hash excluded'})
    registry.save_json(run_dir/'artifact_index.json', {'run_id': run_dir.name, 'artifacts': index,
                       'index_note': 'Index intentionally excludes its own hash', 'required': list(REQUIRED_RUN_ARTIFACTS)})
    return run_dir
