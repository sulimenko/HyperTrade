from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys

from hypertrade.artifacts import load_run_table, list_benchmark_suites, list_experiment_runs, run_benchmark_suite as _run_benchmark_suite
from hypertrade.config.schemas import FilterSearchSpace, ObjectiveProfile, OptimizationRunConfig
from hypertrade.optimization import run_optimization as _run_optimization


def optimize_run(
    run_config: OptimizationRunConfig,
    objective_profile: ObjectiveProfile | None = None,
    search_space: FilterSearchSpace | None = None,
) -> Path:
    return _run_optimization(
        run_config=run_config,
        objective_profile=objective_profile,
        search_space=search_space,
    )


def run_benchmark_suite(
    benchmark_names: list[str] | None = None,
    n_trials: int = 1,
    artifacts_root: str = "artifacts/experiments",
    benchmark_root: str = "artifacts/benchmarks",
    objective_profile: ObjectiveProfile | None = None,
    objective_profile_path: str | None = None,
    search_space: FilterSearchSpace | None = None,
    seed: int = 42,
) -> Path:
    return _run_benchmark_suite(
        run_optimization_fn=_run_optimization,
        benchmark_names=benchmark_names,
        n_trials=n_trials,
        artifacts_root=artifacts_root,
        benchmark_root=benchmark_root,
        objective_profile=objective_profile,
        objective_profile_path=objective_profile_path,
        search_space=search_space,
        seed=seed,
    )


def launch_dashboard() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    app_path = repo_root / "streamlit_app.py"
    env = os.environ.copy()
    python_path_entries = [str(repo_root)]
    if env.get("PYTHONPATH"):
        python_path_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path_entries)
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)], env=env)


def load_run(run_dir: str | Path) -> dict:
    run_path = Path(run_dir)
    manifest_path = run_path / "manifest.json"
    metrics_path = run_path / "metrics_summary.json"
    return {
        "manifest": json.loads(manifest_path.read_text()) if manifest_path.exists() else {},
        "metrics_summary": json.loads(metrics_path.read_text()) if metrics_path.exists() else {},
        "candidate_shortlist": load_run_table(run_path, "candidate_shortlist.parquet"),
    }


def load_benchmark_suite(suite_dir: str | Path) -> dict:
    suite_path = Path(suite_dir)
    manifest_path = suite_path / "manifest.json"
    summary_path = suite_path / "summary.json"
    return {
        "manifest": json.loads(manifest_path.read_text()) if manifest_path.exists() else {},
        "summary": json.loads(summary_path.read_text()) if summary_path.exists() else {},
    }
