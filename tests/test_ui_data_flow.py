import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from hypertrade.artifacts import list_benchmark_suites, list_experiment_runs
from hypertrade.ui import state


class UiDataFlowTests(unittest.TestCase):
    def test_run_and_suite_listing_reads_extended_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            experiments_root = tmp_path / "experiments"
            benchmarks_root = tmp_path / "benchmarks"
            run_dir = experiments_root / "demo_20260101T000000Z"
            suite_dir = benchmarks_root / "benchmark_suite_20260101T000500Z"
            run_dir.mkdir(parents=True)
            suite_dir.mkdir(parents=True)

            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": run_dir.name,
                        "benchmark_name": "demo",
                        "run_label": "demo_manual",
                        "signals_path": "data/signals/demo.csv",
                        "study_name": "demo_study",
                        "created_at": "2026-01-01T00:00:00Z",
                        "finished_at": "2026-01-01T00:01:00Z",
                        "duration_seconds": 60.0,
                        "n_trials": 5,
                        "seed": 42,
                        "artifact_version": 2,
                        "objective_profile_name": "alpha_demo",
                    }
                )
            )
            (run_dir / "metrics_summary.json").write_text(
                json.dumps(
                    {
                        "pareto_candidate_count": 3,
                        "best_shortlist_total_pnl": 12.0,
                        "best_shortlist_profit_factor": 1.4,
                        "best_shortlist_min_drawdown": 2.0,
                    }
                )
            )
            pd.DataFrame([{"number": 0, "total_pnl": 12.0}]).to_parquet(run_dir / "candidate_shortlist.parquet", index=False)

            (suite_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": suite_dir.name,
                        "created_at": "2026-01-01T00:05:00Z",
                        "n_trials": 1,
                        "objective_profile_name": "alpha_demo",
                    }
                )
            )
            (suite_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "created_at": "2026-01-01T00:05:00Z",
                        "benchmark_names": ["demo"],
                        "success_count": 1,
                        "failure_count": 0,
                        "artifact_complete_count": 1,
                        "objective_profile_name": "alpha_demo",
                    }
                )
            )
            pd.DataFrame([{"benchmark_name": "demo", "status": "success", "duration_seconds": 1.5}]).to_parquet(
                suite_dir / "benchmark_runs.parquet",
                index=False,
            )

            runs_df = list_experiment_runs(experiments_root)
            suites_df = list_benchmark_suites(benchmarks_root)
            self.assertEqual(runs_df.iloc[0]["objective_profile_name"], "alpha_demo")
            self.assertEqual(runs_df.iloc[0]["run_label"], "demo_manual")
            self.assertEqual(int(runs_df.iloc[0]["n_trials"]), 5)
            self.assertEqual(int(suites_df.iloc[0]["success_count"]), 1)

            with patch.object(state, "ARTIFACT_ROOT", experiments_root), patch.object(state, "BENCHMARK_ROOT", benchmarks_root):
                state_runs = state.run_options()
                state_suites = state.benchmark_suite_options()
                metrics = state.load_metrics_summary(str(run_dir))
                shortlist = state.load_run_table_safe(str(run_dir), "candidate_shortlist.parquet")

            self.assertEqual(state_runs.iloc[0]["run_id"], run_dir.name)
            self.assertEqual(state_suites.iloc[0]["suite_id"], suite_dir.name)
            self.assertEqual(metrics["best_shortlist_total_pnl"], 12.0)
            self.assertEqual(len(shortlist), 1)


if __name__ == "__main__":
    unittest.main()
