import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hypertrade.config.schemas import CandidateSelectionPolicy, ConstraintRule, ObjectiveMetric, ObjectiveProfile
from hypertrade.optimization.pareto import pick_preferred_candidates


class ObjectiveProfileTests(unittest.TestCase):
    def test_round_trip_json(self) -> None:
        profile = ObjectiveProfile(
            name="test_profile",
            objectives=[
                ObjectiveMetric(metric="sum_trade_returns_pct", direction="minimize"),
                ObjectiveMetric(metric="proxy_drawdown_usd", direction="maximize"),
            ],
            constraints=[ConstraintRule(metric="trades", min_value=10)],
            candidate_policy=CandidateSelectionPolicy('max_profit_factor'),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles" / "test_profile.json"
            profile.save(path)
            loaded = ObjectiveProfile.load(path)

        self.assertEqual(loaded.name, "test_profile")
        self.assertEqual(loaded.directions(), ["minimize", "maximize"])
        self.assertEqual(loaded.constraints[0].metric, "trades")
        self.assertEqual(loaded.to_dict(), profile.to_dict())
        # Invalid reusable profiles fail before saving or launching a study.
        for update in [dict(name='../escape'), dict(objectives=[dict(metric='fake',direction='maximize')]),
                       dict(objectives=[dict(metric='trades',direction='sideways')]),
                       dict(constraints=[dict(metric='trades',min_value=20,max_value=10)]),
                       dict(constraints=[dict(metric='trades',min_value='ten')]),
                       dict(constraints=[dict(metric='trades')])]:
            with self.assertRaises(ValueError):
                ObjectiveProfile.from_dict({**profile.to_dict(), **update})

    def test_profile_requires_objectives(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "objective_profile.json"
            path.write_text('{"name":"bad_profile","objectives":[],"constraints":[],"candidate_policy":{"mode":"utopia_distance"}}')
            with self.assertRaises(ValueError):
                ObjectiveProfile.load(path)

    def test_candidate_policy_changes_preferred_selection(self) -> None:
        frame = pd.DataFrame(
            [
                {"number": 1, "sum_trade_returns_pct": 10.0, "proxy_drawdown_usd": 1.0, "profit_factor": 1.2, "avg_hold_minutes": 20.0},
                {"number": 2, "sum_trade_returns_pct": 20.0, "proxy_drawdown_usd": 5.0, "profit_factor": 1.1, "avg_hold_minutes": 25.0},
            ]
        )
        profile = ObjectiveProfile.from_dict(
            {
                "name": "policy_test",
                "objectives": [
                    {"metric": "sum_trade_returns_pct", "direction": "maximize"},
                    {"metric": "proxy_drawdown_usd", "direction": "minimize"},
                ],
                "constraints": [],
                "candidate_policy": {"mode": "max_sum_trade_returns_pct"},
            }
        )
        preferred = pick_preferred_candidates(frame, profile, limit=1)
        self.assertEqual(int(preferred.iloc[0]["number"]), 2)


if __name__ == "__main__":
    unittest.main()
