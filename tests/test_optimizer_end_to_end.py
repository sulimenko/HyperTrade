import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import run_optimize
from hypertrade.config import DEFAULT_FILTER_SEARCH_SPACE, DEFAULT_OBJECTIVE_PROFILE, ConstraintRule, ObjectiveProfile, OptimizationRunConfig
from hypertrade.artifacts import REQUIRED_RUN_ARTIFACTS, load_run_table, validate_run_artifacts
from hypertrade.optimization import run_optimization


def _synthetic_signals() -> list[dict]:
    return [
        {
            "datetime": pd.Timestamp("2025-07-10T14:00:00Z"),
            "long": ["AAA", "BBB"],
            "short": [],
        }
    ]


def _synthetic_backtest(signals, params):
    pnl = float(params.tp * 10.0 - params.sl * 4.0 - params.delay_open / 300.0)
    drawdown = float(max(params.sl * 2.0, 1.0))
    profit_factor = float(max(1.01, params.tp / max(params.sl, 0.5)))
    hold_minutes = float(params.holding_minutes)

    trades = [
        {
            "symbol": "AAA",
            "direction": 1,
            "entry_dt": pd.Timestamp("2025-07-10T14:00:00Z"),
            "exit_dt": pd.Timestamp("2025-07-10T18:00:00Z"),
            "entry_price": 100.0,
            "exit_price": 101.0,
            "pnl": pnl,
            "return_pct": pnl / 100.0,
            "hold_bars": 16,
            "hold_minutes": hold_minutes,
            "is_win": pnl > 0,
            "exit_reason": "time_exit",
            "rejected": False,
        }
    ]
    signal_stats = [
        {
            "datetime": signals[0]["datetime"],
            "symbols_total": 2,
            "symbols_traded": 1,
            "symbols_rejected": 1,
            "total_pnl": pnl,
            "avg_pnl": pnl,
            "input_long": 2,
            "input_short": 0,
            "accepted_long": 1,
            "accepted_short": 0,
            "rejected_time_gate": 0,
            "rejected_cooldown": 0,
            "rejected_no_data": 0,
        }
    ]
    metrics = {
        "total_pnl": pnl,
        "max_drawdown": drawdown,
        "profit_factor": profit_factor,
        "avg_hold_minutes": hold_minutes,
        "trades": float(len(trades) + 25),
        "calmar": pnl / drawdown,
        "cvar_5": -0.02,
    }
    return trades, signal_stats, metrics


class OptimizerEndToEndTests(unittest.TestCase):
    @patch("hypertrade.optimization.run_backtest", side_effect=_synthetic_backtest)
    @patch("hypertrade.optimization.load_external_signals", side_effect=lambda _: _synthetic_signals())
    def test_run_optimization_persists_complete_bundle(self, _mock_signals, _mock_backtest) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = run_optimization(
                OptimizationRunConfig(
                    signals_path="benchmarks/fixtures/PF20250597.csv",
                    n_trials=3,
                    benchmark_name="synthetic_e2e",
                    artifacts_root=tmpdir,
                    seed=7,
                ),
                objective_profile=DEFAULT_OBJECTIVE_PROFILE,
                search_space=DEFAULT_FILTER_SEARCH_SPACE,
            )

            complete, missing = validate_run_artifacts(run_dir)
            self.assertTrue(complete, msg=f"Missing artifacts: {missing}")

            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["run_id"], run_dir.name)
            self.assertEqual(manifest["benchmark_name"], "synthetic_e2e")
            self.assertEqual(manifest["run_label"], "synthetic_e2e")
            self.assertEqual(manifest["n_trials"], 3)
            self.assertEqual(manifest["objective_profile_name"], DEFAULT_OBJECTIVE_PROFILE.name)
            self.assertIn("config_hashes", manifest)
            self.assertIn("data_hashes", manifest)

            artifact_index = json.loads((run_dir / "artifact_index.json").read_text())
            artifact_names = {row["name"] for row in artifact_index["artifacts"]}
            self.assertTrue(set(REQUIRED_RUN_ARTIFACTS).issubset(artifact_names))

            trials_df = load_run_table(run_dir, "trials.parquet")
            pareto_df = load_run_table(run_dir, "pareto_trials.parquet")
            shortlist_df = load_run_table(run_dir, "candidate_shortlist.parquet")
            trades_df = load_run_table(run_dir, "trades.parquet")
            signal_stats_df = load_run_table(run_dir, "signal_stats.parquet")

            self.assertEqual(len(trials_df), 3)
            self.assertFalse(pareto_df.empty)
            self.assertFalse(shortlist_df.empty)
            self.assertFalse(trades_df.empty)
            self.assertFalse(signal_stats_df.empty)
            self.assertIn("candidate_trial", trades_df.columns)
            self.assertIn("candidate_trial", signal_stats_df.columns)

    @patch("hypertrade.optimization.run_backtest", side_effect=_synthetic_backtest)
    @patch("hypertrade.optimization.load_external_signals", side_effect=lambda _: _synthetic_signals())
    def test_invalid_trials_are_pruned_from_pareto_outputs(self, _mock_signals, _mock_backtest) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            impossible_profile = ObjectiveProfile(
                name="impossible_profile",
                objectives=DEFAULT_OBJECTIVE_PROFILE.objectives,
                constraints=[ConstraintRule(metric="trades", min_value=10_000.0)],
                candidate_policy=DEFAULT_OBJECTIVE_PROFILE.candidate_policy,
            )
            run_dir = run_optimization(
                OptimizationRunConfig(
                    signals_path="benchmarks/fixtures/PF20250597.csv",
                    n_trials=3,
                    benchmark_name="synthetic_invalid",
                    run_label="synthetic_invalid_run",
                    artifacts_root=tmpdir,
                ),
                objective_profile=impossible_profile,
                search_space=DEFAULT_FILTER_SEARCH_SPACE,
            )

            trials_df = load_run_table(run_dir, "trials.parquet")
            pareto_df = load_run_table(run_dir, "pareto_trials.parquet")
            summary = json.loads((run_dir / "metrics_summary.json").read_text())
            self.assertIn("PRUNED", set(trials_df["state"]))
            self.assertTrue((pareto_df.get("constraint_violations", pd.Series(dtype=str)).fillna("") == "").all())
            self.assertEqual(summary["pareto_candidate_count"], len(pareto_df))
            self.assertGreaterEqual(summary["pruned_trial_count"], 1)

    @patch("run_optimize.optimize_run", return_value=Path("/tmp/demo_run"))
    def test_run_optimize_cli_prints_run_dir(self, mock_run_optimization) -> None:
        buffer = io.StringIO()
        argv = [
            "run_optimize.py",
            "--signals",
            "benchmarks/fixtures/PF20250597.csv",
            "--n_trials",
            "2",
            "--benchmark_name",
            "PF20250597",
            "--run_label",
            "cli_check",
        ]
        with patch("sys.argv", argv), redirect_stdout(buffer):
            run_optimize.main()

        self.assertIn("/tmp/demo_run", buffer.getvalue())
        self.assertEqual(mock_run_optimization.call_count, 1)


if __name__ == "__main__":
    unittest.main()
