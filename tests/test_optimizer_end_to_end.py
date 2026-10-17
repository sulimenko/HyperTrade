import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import run_optimize
from hypertrade.config.defaults import DEFAULT_FILTER_SEARCH_SPACE, DEFAULT_OBJECTIVE_PROFILE, ConstraintRule, ObjectiveProfile, OptimizationRunConfig
from hypertrade.experiments.artifacts import REQUIRED_RUN_ARTIFACTS, load_run_table, validate_run_artifacts
from hypertrade.optimization.runner import run_optimization
from hypertrade.simulation.backtest import BacktestResult
from hypertrade.simulation.metrics import compute_metrics, sequential_curve
from hypertrade.simulation.portfolio import portfolio_equity


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
            "sum_trade_returns_pct": pnl,
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
        "sum_trade_returns_pct": pnl,
        "proxy_drawdown_usd": drawdown,
        "profit_factor": profit_factor,
        "avg_hold_minutes": hold_minutes,
        "trades": float(len(trades) + 25),
        "calmar": pnl / drawdown,
        "cvar_5": -0.02,
    }
    # Reuse the existing synthetic outcomes; artifact assertions exercise the full owner result.
    metrics = {**metrics, **compute_metrics(trades,params), 'trades':26.0,
               'portfolio_return_pct':1.0,'portfolio_max_drawdown_pct':0.5,'portfolio_trade_count':1.0}
    metrics['profit_factor'] = profit_factor
    equity,_ = portfolio_equity([],{},params.initial_capital)
    return BacktestResult(trades,signal_stats,metrics,[],[],equity,sequential_curve(trades))


class OptimizerEndToEndTests(unittest.TestCase):
    @patch("hypertrade.optimization.runner.evaluate", side_effect=_synthetic_backtest)
    @patch("hypertrade.optimization.runner.load_external_signals", side_effect=lambda _: _synthetic_signals())
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
            self.assertEqual(manifest['status'],'success')
            self.assertTrue((run_dir/'baseline_trades.parquet').is_file())
            self.assertEqual(json.loads((run_dir/'baseline_metrics.json').read_text())['normalized_notional_usd'],10000)
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

    @patch("hypertrade.optimization.runner.evaluate", side_effect=_synthetic_backtest)
    @patch("hypertrade.optimization.runner.load_external_signals", side_effect=lambda _: _synthetic_signals())
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
            self.assertEqual(summary['status'],'no_valid_candidates')

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


def test_unexpected_trial_error_is_failed_with_complete_diagnostic_bundle(tmp_path):
    from hypertrade.simulation.backtest import evaluate
    calls=0
    def broken_after_baseline(signals,params):
        nonlocal calls
        calls+=1
        if calls==1:
            return evaluate([],params)
        raise RuntimeError('execution failure sentinel')
    with patch('hypertrade.optimization.runner.load_external_signals',return_value=_synthetic_signals()), \
         patch('hypertrade.optimization.runner.evaluate',side_effect=broken_after_baseline):
        run=run_optimization(OptimizationRunConfig(signals_path='benchmarks/fixtures/PF20250597.csv',n_trials=1,artifacts_root=str(tmp_path)))
    assert validate_run_artifacts(run)==(True,[])
    assert json.loads((run/'manifest.json').read_text())['status']=='failed'
    assert load_run_table(run,'trials.parquet').iloc[0]['state']=='FAIL'
    errors=json.loads((run/'errors.json').read_text())['errors']
    assert 'RuntimeError: execution failure sentinel' in errors[0]['traceback']
    assert 'broken_after_baseline' in errors[0]['traceback']


def test_missing_market_history_fails_upfront_and_preparation_stays_separate(tmp_path):
    import pytest
    from hypertrade.data import market
    stamp = pd.Timestamp('2025-07-10T14:00:00Z')
    frame = pd.DataFrame(dict(datetime=pd.date_range(stamp-pd.Timedelta(minutes=15), periods=4, freq='15min'),
                              open=100., high=101., low=99., close=100.))
    csv = tmp_path/'external.csv'
    csv.write_text('datetime;long_symbols;short_symbols\n2025-07-10T14:00:00Z;NEW_US_SYMBOL;\n')
    source = tmp_path/'source'
    staged = tmp_path/'staged'
    config = OptimizationRunConfig(signals_path=str(csv), n_trials=1, artifacts_root=str(tmp_path/'runs'))
    with patch.object(market, 'MARKET_PATH', source), patch.object(market, 'PREPARED_PATH', staged), \
         patch.object(market.requests, 'get', side_effect=AssertionError('Research must not fetch')), \
         patch.object(market.requests, 'post', side_effect=AssertionError('Research must not fetch')):
        failed = run_optimization(config)
        assert json.loads((failed/'manifest.json').read_text())['status']=='failed'
        errors = json.loads((failed/'errors.json').read_text())['errors']
        assert errors[0]['stage']=='market_data_preparation'
        assert 'NEW_US_SYMBOL' in errors[0]['error'] and '--signals <CSV>' in errors[0]['error']
        assert load_run_table(failed, 'trials.parquet').empty
        with pytest.raises(market.MissingMarketDataError, match='DATA_BASE_URL is not configured'):
            market.prepare_market_history('NEW_US_SYMBOL', stamp)
        # Extend an insufficient immutable source into a separate staged history.
        source.mkdir()
        original = source/'NEW_US_SYMBOL.parquet'
        frame.iloc[2:].to_parquet(original, index=False)
        original_bytes = original.read_bytes()
        with patch.object(market, 'fetch_market_data', return_value=frame):
            prepared = market.prepare_market_history('NEW_US_SYMBOL', stamp, base_url='http://configured-source')
        assert prepared.parent==staged and original.read_bytes()==original_bytes
        # Complete prepared data is usable without an API, including another preparation.
        assert market.prepare_market_history('NEW_US_SYMBOL', stamp)==prepared
        assert market.ensure_market_history('NEW_US_SYMBOL', stamp).datetime.min()==frame.datetime.min()
        ready = run_optimization(config)
        assert json.loads((ready/'manifest.json').read_text())['status']=='no_valid_candidates'
        assert not json.loads((ready/'errors.json').read_text())['errors']
        assert validate_run_artifacts(ready)==(True, [])
