from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from hypertrade.artifacts import list_benchmark_suites, list_experiment_runs, load_run_table
from hypertrade.config import DEFAULT_FILTER_SEARCH_SPACE, DEFAULT_OBJECTIVE_PROFILE, FilterSearchSpace, ObjectiveProfile

ARTIFACT_ROOT = Path("artifacts/experiments")
BENCHMARK_ROOT = Path("artifacts/benchmarks")
PROFILE_ROOT = Path("artifacts/objective_profiles")
SEARCH_SPACE_ROOT = Path("artifacts/search_spaces")
PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
SEARCH_SPACE_ROOT.mkdir(parents=True, exist_ok=True)
BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_BENCHMARKS = {
    "PF20250597": "benchmarks/fixtures/PF20250597.csv",
    "signals": "benchmarks/fixtures/signals.csv",
}


def _ensure_default_profiles() -> None:
    default_profile_path = PROFILE_ROOT / f"{DEFAULT_OBJECTIVE_PROFILE.name}.json"
    if not default_profile_path.exists():
        DEFAULT_OBJECTIVE_PROFILE.save(default_profile_path)

    default_search_path = SEARCH_SPACE_ROOT / "default_search_space.json"
    if not default_search_path.exists():
        DEFAULT_FILTER_SEARCH_SPACE.save(default_search_path)


_ensure_default_profiles()


def run_options() -> pd.DataFrame:
    return list_experiment_runs(ARTIFACT_ROOT)


def benchmark_suite_options() -> pd.DataFrame:
    return list_benchmark_suites(BENCHMARK_ROOT)


def load_manifest(run_dir: str) -> dict:
    path = Path(run_dir) / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_objective_profile_for_run(run_dir: str) -> dict:
    path = Path(run_dir) / "objective_profile.json"
    if not path.exists():
        return DEFAULT_OBJECTIVE_PROFILE.to_dict()
    return json.loads(path.read_text())


def load_metrics_summary(run_dir: str) -> dict:
    path = Path(run_dir) / "metrics_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_json_file(path: Path, fallback: dict) -> dict:
    if not path.exists():
        return fallback
    return json.loads(path.read_text())


def numeric_columns(df: pd.DataFrame, excluded: set[str] | None = None) -> list[str]:
    excluded = excluded or set()
    return [col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])]


def safe_metric(frame: pd.DataFrame, metric: str) -> float | None:
    if metric not in frame.columns or frame.empty:
        return None
    series = pd.to_numeric(frame[metric], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[0])


def load_run_table_safe(run_dir: str, name: str) -> pd.DataFrame:
    return load_run_table(run_dir, name)
