from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from hypertrade.experiments.artifacts import list_benchmark_suites, list_experiment_runs, load_run_table

ARTIFACT_ROOT = Path("artifacts/experiments")
BENCHMARK_ROOT = Path("artifacts/benchmarks")
PROFILE_ROOT = Path('config/defaults/profiles')

DEFAULT_BENCHMARKS = {
    "PF20250597": "benchmarks/fixtures/PF20250597.csv",
    "signals": "benchmarks/fixtures/signals.csv",
}


def run_options() -> pd.DataFrame:
    return list_experiment_runs(ARTIFACT_ROOT)


def benchmark_suite_options() -> pd.DataFrame:
    return list_benchmark_suites(BENCHMARK_ROOT)


def load_manifest(run_dir: str) -> dict:
    path = Path(run_dir) / "manifest.json"
    if not path.exists():
        return {}
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
