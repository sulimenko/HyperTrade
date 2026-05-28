import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hypertrade.config.schemas import ConstraintRule, ObjectiveMetric, ObjectiveProfile
from hypertrade.optimization import pick_preferred_candidates


class ObjectiveProfileTests(unittest.TestCase):
    def test_round_trip_json(self) -> None:
        profile = ObjectiveProfile(
            name="test_profile",
            objectives=[
                ObjectiveMetric(metric="total_pnl", direction="maximize"),
                ObjectiveMetric(metric="max_drawdown", direction="minimize"),
            ],
            constraints=[ConstraintRule(metric="trades", min_value=10)],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "objective_profile.json"
            profile.save(path)
            loaded = ObjectiveProfile.load(path)

        self.assertEqual(loaded.name, "test_profile")
        self.assertEqual(loaded.directions(), ["maximize", "minimize"])
        self.assertEqual(loaded.constraints[0].metric, "trades")

    def test_profile_requires_objectives(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "objective_profile.json"
            path.write_text('{"name":"bad_profile","objectives":[],"constraints":[],"candidate_policy":{"mode":"utopia_distance"}}')
            with self.assertRaises(ValueError):
                ObjectiveProfile.load(path)

    def test_candidate_policy_changes_preferred_selection(self) -> None:
        frame = pd.DataFrame(
            [
                {"number": 1, "total_pnl": 10.0, "max_drawdown": 1.0, "profit_factor": 1.2, "avg_hold_minutes": 20.0},
                {"number": 2, "total_pnl": 20.0, "max_drawdown": 5.0, "profit_factor": 1.1, "avg_hold_minutes": 25.0},
            ]
        )
        profile = ObjectiveProfile.from_dict(
            {
                "name": "policy_test",
                "objectives": [
                    {"metric": "total_pnl", "direction": "maximize"},
                    {"metric": "max_drawdown", "direction": "minimize"},
                ],
                "constraints": [],
                "candidate_policy": {"mode": "max_total_pnl"},
            }
        )
        preferred = pick_preferred_candidates(frame, profile, limit=1)
        self.assertEqual(int(preferred.iloc[0]["number"]), 2)


if __name__ == "__main__":
    unittest.main()
