from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
import pandas as pd
from hypertrade.config.defaults import DEFAULT_FILTER_SEARCH_SPACE, DEFAULT_OBJECTIVE_PROFILE
from hypertrade.config.schemas import FilterSearchSpace, ObjectiveProfile, OptimizationRunConfig
from hypertrade.experiments.registry import ExperimentRegistry
from hypertrade.experiments.artifacts import validate_run_artifacts


@dataclass(frozen=True)
class BenchmarkDefinition:
    name: str
    signals_path: str


BENCHMARKS = {
    "PF20250597": BenchmarkDefinition(name="PF20250597", signals_path="benchmarks/fixtures/PF20250597.csv"),
    "signals": BenchmarkDefinition(name="signals", signals_path="benchmarks/fixtures/signals.csv"),
}

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
        from hypertrade.optimization.runner import run_optimization as run_optimization_fn

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
            manifest = json.loads((Path(run_dir) / 'manifest.json').read_text())
            row['status'] = manifest.get('status', 'success')
            artifact_complete, missing = validate_run_artifacts(run_dir)
            row["artifact_complete"] = artifact_complete
            row["missing_artifacts"] = ",".join(missing)
            if not artifact_complete:
                row['status'] = 'failed'
                row['error'] = 'Incomplete artifact bundle'
            if row['status'] == 'failed' and (Path(run_dir)/'errors.json').exists():
                row['diagnostics'] = (Path(run_dir)/'errors.json').read_text()
            metrics_path = Path(run_dir) / "metrics_summary.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text())
                for key, value in metrics.items():
                    row[key] = value
                if row["status"] != "failed" and int(metrics.get("pareto_candidate_count", 0) or 0) <= 0:
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
        "artifact_version": 3,
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


