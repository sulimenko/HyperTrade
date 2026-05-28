from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import traceback

import pandas as pd
import yaml

from hypertrade.config import DEFAULT_FILTER_SEARCH_SPACE, DEFAULT_OBJECTIVE_PROFILE, FilterSearchSpace, ObjectiveProfile, OptimizationRunConfig


class ExperimentRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run_dir(self, prefix: str) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.root / f"{prefix}_{timestamp}"
        suffix = 0
        candidate = run_dir
        while candidate.exists():
            suffix += 1
            candidate = self.root / f"{prefix}_{timestamp}_{suffix:02d}"
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    def save_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, indent=2, default=str))

    def save_yaml(self, path: Path, payload: dict) -> None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False))

    def save_table(self, path: Path, frame: pd.DataFrame) -> None:
        if frame.empty:
            frame = pd.DataFrame()
        frame.to_parquet(path, index=False)

    def save_study_tables(self, run_dir: Path, trials_df: pd.DataFrame, pareto_df: pd.DataFrame, shortlist_df: pd.DataFrame) -> None:
        self.save_table(run_dir / "trials.parquet", trials_df)
        self.save_table(run_dir / "pareto_trials.parquet", pareto_df)
        self.save_table(run_dir / "candidate_shortlist.parquet", shortlist_df)

    def save_text(self, path: Path, content: str) -> None:
        path.write_text(content)

    def payload_sha256(self, payload: dict) -> str:
        normalized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def file_sha256(self, path: str | Path) -> str | None:
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            return None
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def git_snapshot(self) -> dict:
        def _run_git(args: list[str]) -> str | None:
            try:
                result = subprocess.run(["git", *args], check=True, capture_output=True, text=True, cwd=self.root)
            except Exception:
                return None
            value = result.stdout.strip()
            return value or None

        return {
            "commit": _run_git(["rev-parse", "HEAD"]),
            "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "status_short": _run_git(["status", "--short"]),
        }

    def build_environment_snapshot(self) -> dict:
        return {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "hostname": socket.gethostname(),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "git": self.git_snapshot(),
        }

    def artifact_inventory(self, run_dir: str | Path) -> list[dict]:
        root = Path(run_dir)
        rows = []
        for path in sorted(root.glob("*")):
            if not path.is_file():
                continue
            rows.append({"name": path.name, "size_bytes": path.stat().st_size, "sha256": self.file_sha256(path)})
        return rows

    def validate_artifacts(self, run_dir: str | Path, required_names: list[str] | tuple[str, ...]) -> tuple[bool, list[str]]:
        root = Path(run_dir)
        missing = [name for name in required_names if not (root / name).exists()]
        return not missing, missing


@dataclass(frozen=True)
class BenchmarkDefinition:
    name: str
    signals_path: str


BENCHMARKS = {
    "PF20250597": BenchmarkDefinition(name="PF20250597", signals_path="benchmarks/fixtures/PF20250597.csv"),
    "signals": BenchmarkDefinition(name="signals", signals_path="benchmarks/fixtures/signals.csv"),
}

REQUIRED_RUN_ARTIFACTS = (
    "manifest.json",
    "manifest.yaml",
    "objective_profile.json",
    "objective_profile.yaml",
    "search_space.json",
    "search_space.yaml",
    "benchmark_profile.json",
    "metrics_summary.json",
    "environment.json",
    "logs.txt",
    "artifact_index.json",
    "study.sqlite3",
    "trials.parquet",
    "pareto_trials.parquet",
    "candidate_shortlist.parquet",
    "trades.parquet",
    "signal_stats.parquet",
)


def smoke_run_config(name: str, n_trials: int = 1, artifacts_root: str = "artifacts/experiments") -> OptimizationRunConfig:
    benchmark = BENCHMARKS[name]
    return OptimizationRunConfig(
        signals_path=benchmark.signals_path,
        benchmark_name=benchmark.name,
        run_label=benchmark.name,
        n_trials=n_trials,
        artifacts_root=artifacts_root,
        study_name=f"{benchmark.name}_smoke",
    )


def resolve_benchmarks(names: list[str] | None = None) -> list[BenchmarkDefinition]:
    selected = names or list(BENCHMARKS.keys())
    unknown = [name for name in selected if name not in BENCHMARKS]
    if unknown:
        raise ValueError(f"Unknown benchmarks: {', '.join(unknown)}")
    return [BENCHMARKS[name] for name in selected]


def validate_run_artifacts(run_dir: str | Path) -> tuple[bool, list[str]]:
    registry = ExperimentRegistry(Path(run_dir).parent)
    return registry.validate_artifacts(run_dir, REQUIRED_RUN_ARTIFACTS)


def run_benchmark_suite(
    run_optimization_fn=None,
    benchmark_names: list[str] | None = None,
    n_trials: int = 1,
    artifacts_root: str = "artifacts/experiments",
    benchmark_root: str = "artifacts/benchmarks",
    objective_profile: ObjectiveProfile | None = None,
    objective_profile_path: str | None = None,
    search_space: FilterSearchSpace | None = None,
    seed: int = 42,
) -> Path:
    if run_optimization_fn is None:
        from hypertrade.optimization import run_optimization as run_optimization_fn

    selected = resolve_benchmarks(benchmark_names)
    objective_profile = objective_profile or (
        ObjectiveProfile.load(objective_profile_path) if objective_profile_path else DEFAULT_OBJECTIVE_PROFILE
    )
    search_space = search_space or DEFAULT_FILTER_SEARCH_SPACE

    registry = ExperimentRegistry(benchmark_root)
    suite_dir = registry.create_run_dir(prefix="benchmark_suite")
    rows: list[dict] = []

    for benchmark in selected:
        started_at = datetime.now(timezone.utc)
        row = {
            "benchmark_name": benchmark.name,
            "signals_path": benchmark.signals_path,
            "started_at": started_at.isoformat(),
            "status": "success",
            "run_path": None,
            "duration_seconds": None,
            "artifact_complete": False,
            "missing_artifacts": "",
            "error": None,
        }
        try:
            run_dir = run_optimization_fn(
                OptimizationRunConfig(
                    signals_path=benchmark.signals_path,
                    n_trials=n_trials,
                    benchmark_name=benchmark.name,
                    run_label=benchmark.name,
                    artifacts_root=artifacts_root,
                    objective_profile_path=objective_profile_path,
                    study_name=f"{benchmark.name}_benchmark",
                    seed=seed,
                ),
                objective_profile=objective_profile,
                search_space=search_space,
            )
            row["run_path"] = str(run_dir)
            artifact_complete, missing = validate_run_artifacts(run_dir)
            row["artifact_complete"] = artifact_complete
            row["missing_artifacts"] = ",".join(missing)
            metrics_path = Path(run_dir) / "metrics_summary.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text())
                for key, value in metrics.items():
                    row[key] = value
                if int(metrics.get("pareto_candidate_count", 0) or 0) <= 0:
                    row["status"] = "no_valid_candidates"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{exc.__class__.__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
        finished_at = datetime.now(timezone.utc)
        row["finished_at"] = finished_at.isoformat()
        row["duration_seconds"] = round((finished_at - started_at).total_seconds(), 3)
        rows.append(row)

    results_df = pd.DataFrame(rows)
    summary_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_dir),
        "benchmark_names": [benchmark.name for benchmark in selected],
        "n_trials": n_trials,
        "artifacts_root": artifacts_root,
        "success_count": int((results_df["status"] == "success").sum()) if not results_df.empty else 0,
        "failure_count": int((results_df["status"] == "failed").sum()) if not results_df.empty else 0,
        "no_valid_candidate_count": int((results_df["status"] == "no_valid_candidates").sum()) if not results_df.empty else 0,
        "artifact_complete_count": int(results_df["artifact_complete"].fillna(False).sum()) if not results_df.empty else 0,
        "objective_profile_name": objective_profile.name,
    }

    manifest_payload = {
        "run_id": suite_dir.name,
        "run_dir": str(suite_dir),
        "created_at": summary_payload["created_at"],
        "suite_type": "benchmark_suite",
        "benchmark_names": summary_payload["benchmark_names"],
        "n_trials": n_trials,
        "artifacts_root": artifacts_root,
        "objective_profile_name": objective_profile.name,
        "artifact_version": 2,
    }
    registry.save_json(suite_dir / "manifest.json", manifest_payload)
    registry.save_yaml(suite_dir / "manifest.yaml", manifest_payload)
    registry.save_json(suite_dir / "summary.json", summary_payload)
    registry.save_json(suite_dir / "objective_profile.json", objective_profile.to_dict())
    registry.save_yaml(suite_dir / "objective_profile.yaml", objective_profile.to_dict())
    registry.save_json(suite_dir / "search_space.json", asdict(search_space))
    registry.save_yaml(suite_dir / "search_space.yaml", asdict(search_space))
    registry.save_table(suite_dir / "benchmark_runs.parquet", results_df)
    artifact_index_path = suite_dir / "artifact_index.json"
    registry.save_json(artifact_index_path, {"run_id": suite_dir.name, "artifacts": []})
    registry.save_json(artifact_index_path, {"run_id": suite_dir.name, "artifacts": registry.artifact_inventory(suite_dir)})
    return suite_dir


def list_experiment_runs(root: str | Path) -> pd.DataFrame:
    root_path = Path(root)
    rows = []
    for run_dir in sorted(root_path.glob("*"), reverse=True):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        rows.append(
            {
                "run_id": run_dir.name,
                "path": str(run_dir),
                "benchmark_name": manifest.get("benchmark_name"),
                "run_label": manifest.get("run_label"),
                "signals_path": manifest.get("signals_path"),
                "study_name": manifest.get("study_name"),
                "created_at": manifest.get("created_at"),
                "finished_at": manifest.get("finished_at"),
                "duration_seconds": manifest.get("duration_seconds"),
                "n_trials": manifest.get("n_trials"),
                "seed": manifest.get("seed"),
                "artifact_version": manifest.get("artifact_version"),
                "objective_profile_name": (
                    manifest.get("objective_profile_name")
                    or manifest.get("objective_profile", {}).get("name")
                    or manifest.get("objective_profile_path")
                ),
            }
        )
    return pd.DataFrame(rows)


def list_benchmark_suites(root: str | Path) -> pd.DataFrame:
    root_path = Path(root)
    rows = []
    for run_dir in sorted(root_path.glob("*"), reverse=True):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        manifest_path = run_dir / "manifest.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        rows.append(
            {
                "suite_id": run_dir.name,
                "path": str(run_dir),
                "created_at": summary.get("created_at") or manifest.get("created_at"),
                "n_trials": manifest.get("n_trials"),
                "success_count": summary.get("success_count"),
                "failure_count": summary.get("failure_count"),
                "artifact_complete_count": summary.get("artifact_complete_count"),
                "objective_profile_name": summary.get("objective_profile_name") or manifest.get("objective_profile_name"),
                "benchmark_names": ",".join(summary.get("benchmark_names", [])),
            }
        )
    return pd.DataFrame(rows)


def load_run_table(run_dir: str | Path, name: str) -> pd.DataFrame:
    path = Path(run_dir) / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
