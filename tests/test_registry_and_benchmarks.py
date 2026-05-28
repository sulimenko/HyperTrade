import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from hypertrade.artifacts import (
    BENCHMARKS,
    ExperimentRegistry,
    REQUIRED_RUN_ARTIFACTS,
    list_benchmark_suites,
    list_experiment_runs,
    load_run_table,
    run_benchmark_suite,
    smoke_run_config,
    validate_run_artifacts,
)


class RegistryAndBenchmarksTests(unittest.TestCase):
    def test_registry_writes_tables_and_environment_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ExperimentRegistry(tmpdir)
            run_dir = registry.create_run_dir("demo")
            registry.save_json(run_dir / "manifest.json", {"benchmark_name": "demo", "signals_path": "signals.csv"})
            registry.save_yaml(run_dir / "manifest.yaml", {"benchmark_name": "demo", "signals_path": "signals.csv"})
            registry.save_json(run_dir / "environment.json", registry.build_environment_snapshot())
            registry.save_json(run_dir / "metrics_summary.json", {"pareto_candidate_count": 1})
            registry.save_text(run_dir / "logs.txt", "hello")
            registry.save_study_tables(
                run_dir,
                pd.DataFrame([{"number": 1, "total_pnl": 10.0}]),
                pd.DataFrame([{"number": 1, "total_pnl": 10.0}]),
                pd.DataFrame([{"number": 1, "total_pnl": 10.0}]),
            )

            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertTrue((run_dir / "manifest.yaml").exists())
            self.assertTrue((run_dir / "environment.json").exists())
            self.assertTrue((run_dir / "metrics_summary.json").exists())
            self.assertTrue((run_dir / "logs.txt").exists())

            env = json.loads((run_dir / "environment.json").read_text())
            self.assertIn("python_version", env)
            self.assertIn("git", env)
            trials = load_run_table(run_dir, "trials.parquet")
            self.assertEqual(len(trials), 1)

    def test_benchmark_smoke_config_uses_known_inputs(self) -> None:
        config = smoke_run_config("PF20250597", n_trials=2, artifacts_root="tmp-artifacts")
        self.assertEqual(config.benchmark_name, "PF20250597")
        self.assertEqual(config.signals_path, BENCHMARKS["PF20250597"].signals_path)
        self.assertEqual(config.n_trials, 2)

    def test_list_experiment_runs_reads_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "demo_20260101T000000Z"
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "benchmark_name": "demo",
                        "signals_path": "data/signals/demo.csv",
                        "study_name": "demo_study",
                        "created_at": "2026-01-01T00:00:00Z",
                    }
                )
            )

            runs = list_experiment_runs(tmpdir)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs.iloc[0]["benchmark_name"], "demo")

    def test_validate_run_artifacts_detects_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "demo_run"
            run_dir.mkdir()
            for name in REQUIRED_RUN_ARTIFACTS[:-1]:
                (run_dir / name).write_text("{}")
            complete, missing = validate_run_artifacts(run_dir)
            self.assertFalse(complete)
            self.assertEqual(missing, [REQUIRED_RUN_ARTIFACTS[-1]])

    def test_run_benchmark_suite_writes_summary_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            experiments_root = Path(tmpdir) / "experiments"
            benchmarks_root = Path(tmpdir) / "benchmarks"

            def fake_run_optimization(*args, **kwargs):
                registry = ExperimentRegistry(experiments_root)
                run_dir = registry.create_run_dir("PF20250597")
                registry.save_json(run_dir / "manifest.json", {"benchmark_name": "PF20250597", "signals_path": "benchmarks/fixtures/PF20250597.csv"})
                registry.save_yaml(run_dir / "manifest.yaml", {"benchmark_name": "PF20250597", "signals_path": "benchmarks/fixtures/PF20250597.csv"})
                registry.save_json(run_dir / "objective_profile.json", {"name": "demo", "objectives": []})
                registry.save_yaml(run_dir / "objective_profile.yaml", {"name": "demo", "objectives": []})
                registry.save_json(run_dir / "search_space.json", {})
                registry.save_yaml(run_dir / "search_space.yaml", {})
                registry.save_json(run_dir / "benchmark_profile.json", {"benchmark_name": "PF20250597"})
                registry.save_json(run_dir / "metrics_summary.json", {"pareto_candidate_count": 1, "best_shortlist_total_pnl": 5.0})
                registry.save_json(run_dir / "environment.json", registry.build_environment_snapshot())
                registry.save_text(run_dir / "logs.txt", "ok")
                registry.save_text(run_dir / "study.sqlite3", "")
                for table_name in ("trials.parquet", "pareto_trials.parquet", "candidate_shortlist.parquet", "trades.parquet", "signal_stats.parquet"):
                    registry.save_table(run_dir / table_name, pd.DataFrame([{"number": 0}]))
                registry.save_json(run_dir / "artifact_index.json", {"run_id": run_dir.name, "artifacts": registry.artifact_inventory(run_dir)})
                return run_dir

            suite_dir = run_benchmark_suite(
                fake_run_optimization,
                benchmark_names=["PF20250597"],
                n_trials=1,
                artifacts_root=str(experiments_root),
                benchmark_root=str(benchmarks_root),
            )

            self.assertTrue((suite_dir / "summary.json").exists())
            self.assertTrue((suite_dir / "manifest.yaml").exists())
            self.assertTrue((suite_dir / "benchmark_runs.parquet").exists())
            suites = list_benchmark_suites(benchmarks_root)
            self.assertEqual(len(suites), 1)
            rows = pd.read_parquet(suite_dir / "benchmark_runs.parquet")
            self.assertEqual(rows.iloc[0]["status"], "success")
            self.assertTrue(bool(rows.iloc[0]["artifact_complete"]))

    def test_benchmark_suite_marks_no_valid_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            experiments_root = Path(tmpdir) / "experiments"
            benchmarks_root = Path(tmpdir) / "benchmarks"

            def fake_run_optimization(*args, **kwargs):
                registry = ExperimentRegistry(experiments_root)
                run_dir = registry.create_run_dir("signals")
                registry.save_json(run_dir / "manifest.json", {"benchmark_name": "signals", "signals_path": "benchmarks/fixtures/signals.csv"})
                registry.save_yaml(run_dir / "manifest.yaml", {"benchmark_name": "signals", "signals_path": "benchmarks/fixtures/signals.csv"})
                registry.save_json(run_dir / "objective_profile.json", {"name": "demo", "objectives": [{"metric": "total_pnl", "direction": "maximize"}]})
                registry.save_yaml(run_dir / "objective_profile.yaml", {"name": "demo", "objectives": [{"metric": "total_pnl", "direction": "maximize"}]})
                registry.save_json(run_dir / "search_space.json", {})
                registry.save_yaml(run_dir / "search_space.yaml", {})
                registry.save_json(run_dir / "benchmark_profile.json", {"benchmark_name": "signals"})
                registry.save_json(run_dir / "metrics_summary.json", {"pareto_candidate_count": 0, "shortlist_candidate_count": 0})
                registry.save_json(run_dir / "environment.json", registry.build_environment_snapshot())
                registry.save_text(run_dir / "logs.txt", "ok")
                registry.save_text(run_dir / "study.sqlite3", "")
                for table_name in ("trials.parquet", "pareto_trials.parquet", "candidate_shortlist.parquet", "trades.parquet", "signal_stats.parquet"):
                    registry.save_table(run_dir / table_name, pd.DataFrame())
                registry.save_json(run_dir / "artifact_index.json", {"run_id": run_dir.name, "artifacts": []})
                return run_dir

            suite_dir = run_benchmark_suite(
                fake_run_optimization,
                benchmark_names=["signals"],
                n_trials=1,
                artifacts_root=str(experiments_root),
                benchmark_root=str(benchmarks_root),
            )

            rows = pd.read_parquet(suite_dir / "benchmark_runs.parquet")
            self.assertEqual(rows.iloc[0]["status"], "no_valid_candidates")


if __name__ == "__main__":
    unittest.main()
