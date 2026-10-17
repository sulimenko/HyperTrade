from pathlib import Path
import json
import pandas as pd

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
    "baseline_metrics.json",
    "baseline_trades.parquet",
    "baseline_portfolio_trades.parquet",
    "baseline_signal_stats.parquet",
    "baseline_portfolio_signal_stats.parquet",
    "baseline_equity.parquet",
    "baseline_sequential.parquet",
    "portfolio_trades.parquet",
    "portfolio_signal_stats.parquet",
    "equity.parquet",
    "sequential.parquet",
    "errors.json",
    "candidate_parameters.json",
    "data_snapshot.json",
)


def validate_run_artifacts(run_dir):
    missing = [n for n in REQUIRED_RUN_ARTIFACTS if not (Path(run_dir)/n).is_file()]
    return not missing, missing


def list_experiment_runs(root: str | Path) -> pd.DataFrame:
    root_path = Path(root)
    rows = []
    for run_dir in sorted(root_path.glob("*"), reverse=True):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        rows.append(
            {
                "run_id": run_dir.name,
                "status": manifest.get("status"),
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
                "objective_profile_name": manifest.get("objective_profile_name"),
            }
        )
    return pd.DataFrame(rows).sort_values("created_at", ascending=False, kind="stable") if rows else pd.DataFrame()


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
