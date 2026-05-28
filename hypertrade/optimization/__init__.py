from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import contextlib
import io
import math
from pathlib import Path
import time

import optuna
import pandas as pd

from hypertrade.artifacts import ExperimentRegistry
from hypertrade.config import DEFAULT_FILTER_SEARCH_SPACE, DEFAULT_OBJECTIVE_PROFILE, FilterSearchSpace, ObjectiveProfile, OptimizationRunConfig
from hypertrade.signals import load_external_signals
from hypertrade.signals.filter_space import build_strategy_params
from hypertrade.simulation import run_backtest


def pareto_trials_dataframe(study) -> pd.DataFrame:
    rows = []
    for trial in study.best_trials:
        rows.append(
            {
                "number": trial.number,
                "values": list(trial.values or []),
                "params": dict(trial.params),
                "user_attrs": dict(trial.user_attrs),
                "state": str(trial.state),
            }
        )
    return pd.DataFrame(rows)


def pick_preferred_candidates(df: pd.DataFrame, objective_profile: ObjectiveProfile, limit: int = 5) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    frame = df.copy()
    if "constraint_violations" in frame.columns:
        constraint_series = frame["constraint_violations"].fillna("").astype(str)
        frame = frame[constraint_series.eq("")]
    if frame.empty:
        return frame

    for objective in objective_profile.objectives:
        if objective.metric not in frame.columns:
            return frame.iloc[0:0].copy()
        column = f"metric_{objective.metric}"
        frame[column] = pd.to_numeric(frame[objective.metric], errors="coerce")
        series = frame[column]
        lo = series.min()
        hi = series.max()
        denom = (hi - lo) if pd.notna(hi) and pd.notna(lo) and hi != lo else 1.0
        if objective.direction == "maximize":
            frame[f"norm_{objective.metric}"] = (hi - series) / denom
        else:
            frame[f"norm_{objective.metric}"] = (series - lo) / denom

    score = 0.0
    for objective in objective_profile.objectives:
        score = score + frame[f"norm_{objective.metric}"].pow(2)
    frame["utopia_distance"] = score.apply(lambda x: math.sqrt(float(x)))

    mode = objective_profile.candidate_policy.normalized_mode()
    if mode == "max_total_pnl" and "total_pnl" in frame.columns:
        return frame.sort_values(["total_pnl", "utopia_distance"], ascending=[False, True]).head(limit)
    if mode == "max_profit_factor" and "profit_factor" in frame.columns:
        return frame.sort_values(["profit_factor", "utopia_distance"], ascending=[False, True]).head(limit)
    if mode == "min_max_drawdown" and "max_drawdown" in frame.columns:
        return frame.sort_values(["max_drawdown", "utopia_distance"], ascending=[True, True]).head(limit)
    if mode == "min_avg_hold_minutes" and "avg_hold_minutes" in frame.columns:
        return frame.sort_values(["avg_hold_minutes", "utopia_distance"], ascending=[True, True]).head(limit)
    return frame.sort_values("utopia_distance", ascending=True).head(limit)


def constraint_violations(metrics: dict, objective_profile: ObjectiveProfile) -> list[str]:
    violations = []
    for rule in objective_profile.constraints:
        value = metrics.get(rule.metric)
        if value is None:
            violations.append(f"missing:{rule.metric}")
            continue
        if rule.min_value is not None and float(value) < float(rule.min_value):
            violations.append(f"{rule.metric}<min")
        if rule.max_value is not None and float(value) > float(rule.max_value):
            violations.append(f"{rule.metric}>max")
    return violations


def _objective_values(metrics: dict, objective_profile: ObjectiveProfile) -> tuple[float, ...]:
    return tuple(float(metrics.get(objective.metric, 0.0)) for objective in objective_profile.objectives)


def _trial_frame(study: optuna.study.Study, objective_profile: ObjectiveProfile) -> pd.DataFrame:
    frame = study.trials_dataframe(attrs=("number", "state", "values", "params", "user_attrs"))
    if frame.empty:
        return frame
    if "values_0" in frame.columns:
        for index, objective in enumerate(objective_profile.objectives):
            values_column = f"values_{index}"
            if values_column in frame.columns:
                frame[objective.metric] = frame[values_column]
    return frame


def run_optimization(
    run_config: OptimizationRunConfig,
    objective_profile: ObjectiveProfile | None = None,
    search_space: FilterSearchSpace | None = None,
) -> Path:
    if objective_profile is None and run_config.objective_profile_path:
        objective_profile = ObjectiveProfile.load(run_config.objective_profile_path)
    objective_profile = objective_profile or DEFAULT_OBJECTIVE_PROFILE
    search_space = search_space or DEFAULT_FILTER_SEARCH_SPACE

    signals = load_external_signals(run_config.signals_path)
    benchmark_name = run_config.benchmark_name or Path(run_config.signals_path).stem
    run_label = run_config.run_label or run_config.study_name or benchmark_name

    registry = ExperimentRegistry(run_config.artifacts_root)
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.perf_counter()
    run_dir = registry.create_run_dir(prefix=run_label)
    run_id = run_dir.name
    storage_url = f"sqlite:///{(run_dir / 'study.sqlite3').resolve()}"
    environment = registry.build_environment_snapshot()
    log_buffer = io.StringIO()
    search_space_payload = asdict(search_space)
    objective_profile_payload = objective_profile.to_dict()
    benchmark_profile = {
        "benchmark_name": benchmark_name,
        "signals_path": run_config.signals_path,
        "signals_sha256": registry.file_sha256(run_config.signals_path),
    }
    run_config_payload = asdict(run_config)

    sampler = optuna.samplers.NSGAIISampler(seed=run_config.seed)
    study = optuna.create_study(
        study_name=run_config.study_name or run_label,
        directions=objective_profile.directions(),
        sampler=sampler,
        storage=storage_url,
        load_if_exists=False,
    )

    def objective(trial: optuna.trial.Trial) -> tuple[float, ...]:
        params = build_strategy_params(trial, search_space)
        trades, signal_stats, metrics = run_backtest(signals, params)
        trial.set_user_attr("benchmark_name", benchmark_name)
        trial.set_user_attr("signals_path", run_config.signals_path)
        trial.set_user_attr("trade_count", len(trades))
        trial.set_user_attr("signal_count", len(signal_stats))
        for key, value in metrics.items():
            try:
                trial.set_user_attr(key, float(value))
            except Exception:
                pass
        violations = constraint_violations(metrics, objective_profile)
        if not trades or violations:
            trial.set_user_attr("constraint_violations", ",".join(violations) if violations else "no_trades")
            raise optuna.TrialPruned()
        return _objective_values(metrics, objective_profile)

    with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
        study.optimize(objective, n_trials=run_config.n_trials)

    trials_df = _trial_frame(study, objective_profile)
    valid_trials = []
    pareto_rows = []
    try:
        best_trials = study.best_trials
    except Exception:
        best_trials = []
    for trial in best_trials:
        if trial.user_attrs.get("constraint_violations"):
            continue
        row = {"number": trial.number, **trial.params, **trial.user_attrs}
        if trial.values is not None:
            for index, objective in enumerate(objective_profile.objectives):
                row[objective.metric] = trial.values[index]
        pareto_rows.append(row)
        valid_trials.append(trial)
    pareto_df = pd.DataFrame(pareto_rows)
    shortlist_df = pick_preferred_candidates(pareto_df, objective_profile, limit=5)
    registry.save_study_tables(run_dir, trials_df, pareto_df, shortlist_df)

    shortlist_trades = []
    shortlist_signal_stats = []
    for _, row in shortlist_df.iterrows():
        trial_number = int(row["number"])
        trial = study.trials[trial_number]
        params = build_strategy_params(trial, search_space)
        trades, signal_stats, _metrics = run_backtest(signals, params)
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df["candidate_trial"] = trial_number
            shortlist_trades.append(trades_df)
        signal_stats_df = pd.DataFrame(signal_stats)
        if not signal_stats_df.empty:
            signal_stats_df["candidate_trial"] = trial_number
            shortlist_signal_stats.append(signal_stats_df)

    registry.save_table(run_dir / "trades.parquet", pd.concat(shortlist_trades, ignore_index=True) if shortlist_trades else pd.DataFrame())
    registry.save_table(run_dir / "signal_stats.parquet", pd.concat(shortlist_signal_stats, ignore_index=True) if shortlist_signal_stats else pd.DataFrame())

    metrics_summary = {
        "valid_trial_count": int(len(valid_trials)),
        "completed_trial_count": int((trials_df.get("state") == "COMPLETE").sum()) if not trials_df.empty and "state" in trials_df.columns else 0,
        "pruned_trial_count": int((trials_df.get("state") == "PRUNED").sum()) if not trials_df.empty and "state" in trials_df.columns else 0,
        "pareto_candidate_count": int(len(pareto_df)),
        "shortlist_candidate_count": int(len(shortlist_df)),
        "best_shortlist_total_pnl": float(shortlist_df["total_pnl"].max()) if "total_pnl" in shortlist_df.columns and not shortlist_df.empty else None,
        "best_shortlist_profit_factor": float(shortlist_df["profit_factor"].max()) if "profit_factor" in shortlist_df.columns and not shortlist_df.empty else None,
        "best_shortlist_min_drawdown": float(shortlist_df["max_drawdown"].min()) if "max_drawdown" in shortlist_df.columns and not shortlist_df.empty else None,
    }
    finished_at = datetime.now(timezone.utc)
    duration_seconds = round(time.perf_counter() - started_monotonic, 3)
    manifest_payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "created_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": duration_seconds,
        "run_label": run_label,
        "benchmark_name": benchmark_name,
        "signals_path": run_config.signals_path,
        "study_name": study.study_name,
        "n_trials": run_config.n_trials,
        "seed": run_config.seed,
        "objective_profile_path": run_config.objective_profile_path,
        "objective_profile_name": objective_profile.name,
        "artifact_version": 2,
        "storage_url": storage_url,
        "config_hashes": {
            "run_config_sha256": registry.payload_sha256(run_config_payload),
            "objective_profile_sha256": registry.payload_sha256(objective_profile_payload),
            "search_space_sha256": registry.payload_sha256(search_space_payload),
        },
        "data_hashes": {"signals_sha256": benchmark_profile["signals_sha256"]},
        "environment": {
            "python_executable": environment.get("python_executable"),
            "platform": environment.get("platform"),
            "conda_default_env": environment.get("conda_default_env"),
            "git": environment.get("git"),
        },
    }
    registry.save_json(run_dir / "manifest.json", manifest_payload)
    registry.save_yaml(run_dir / "manifest.yaml", manifest_payload)
    registry.save_json(run_dir / "objective_profile.json", objective_profile_payload)
    registry.save_yaml(run_dir / "objective_profile.yaml", objective_profile_payload)
    registry.save_json(run_dir / "search_space.json", search_space_payload)
    registry.save_yaml(run_dir / "search_space.yaml", search_space_payload)
    registry.save_json(run_dir / "benchmark_profile.json", benchmark_profile)
    registry.save_json(run_dir / "metrics_summary.json", metrics_summary)
    registry.save_json(run_dir / "environment.json", environment)
    registry.save_text(run_dir / "logs.txt", log_buffer.getvalue())
    artifact_index_path = run_dir / "artifact_index.json"
    registry.save_json(artifact_index_path, {"run_id": run_id, "artifacts": []})
    registry.save_json(artifact_index_path, {"run_id": run_id, "artifacts": registry.artifact_inventory(run_dir)})
    return run_dir


__all__ = [
    "constraint_violations",
    "pareto_trials_dataframe",
    "pick_preferred_candidates",
    "run_optimization",
]
