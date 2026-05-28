import unittest
from unittest.mock import patch

import pandas as pd

from hypertrade.config.strategy import StrategyParams, default_indicator_config
from hypertrade.signals import apply_signal_policy
from hypertrade.simulation.market_time import compute_entry_time


def _market_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2025-07-10T14:00:00Z", "2025-07-10T14:15:00Z", "2025-07-10T14:30:00Z"],
                utc=True,
            ),
            "open": [10.0, 10.1, 10.2],
            "high": [10.2, 10.3, 10.4],
            "low": [9.9, 10.0, 10.1],
            "close": [10.1, 10.2, 10.3],
            "ema_10": [10.0, 10.1, 10.2],
            "ema_40": [9.8, 9.9, 10.0],
            "rsi_12": [60.0, 62.0, 63.0],
        }
    )


class SignalAcceptanceTests(unittest.TestCase):
    @patch("hypertrade.signals.ensure_market_data")
    @patch("hypertrade.signals.compute_entry_time")
    def test_time_gate_and_max_signals_per_side(self, mock_entry_time, mock_market_data) -> None:
        mock_entry_time.return_value = pd.Timestamp("2025-07-10T14:00:00Z")
        mock_market_data.return_value = _market_frame()

        params = StrategyParams(
            delay_open=0,
            accept_start_minute=0,
            accept_end_minute=60,
            max_signals_per_side=1,
            ranking_mode="strength",
            indicator_config=default_indicator_config(),
        )
        params.indicator_config["ema"] = {"enabled": True, "sign": "above", "fast": 10, "slow": 40}
        params.indicator_config["rsi"] = {"enabled": True, "sign": "above", "level": 55, "period": 12}

        signals = [
            {
                "datetime": pd.Timestamp("2025-07-10T14:05:00Z"),
                "long": ["AAA", "BBB"],
                "short": [],
            }
        ]

        accepted, stats = apply_signal_policy(signals, params)
        self.assertEqual(len(accepted[0]["long"]), 1)
        self.assertEqual(stats[0]["accepted_long"], 1)

    @patch("hypertrade.signals.ensure_market_data")
    @patch("hypertrade.signals.compute_entry_time")
    def test_cooldown_blocks_repeated_symbol(self, mock_entry_time, mock_market_data) -> None:
        mock_market_data.return_value = _market_frame()
        mock_entry_time.side_effect = [
            pd.Timestamp("2025-07-10T14:00:00Z"),
            pd.Timestamp("2025-07-10T14:10:00Z"),
        ]

        params = StrategyParams(
            delay_open=0,
            cooldown_minutes=30,
            ranking_mode="none",
            indicator_config=default_indicator_config(),
        )

        signals = [
            {"datetime": pd.Timestamp("2025-07-10T14:00:00Z"), "long": ["AAA"], "short": []},
            {"datetime": pd.Timestamp("2025-07-10T14:10:00Z"), "long": ["AAA"], "short": []},
        ]

        accepted, stats = apply_signal_policy(signals, params)
        self.assertEqual(accepted[0]["long"], ["AAA"])
        self.assertEqual(accepted[1]["long"], [])
        self.assertEqual(stats[1]["rejected_cooldown"], 1)

    def test_compute_entry_time_skips_market_holiday(self) -> None:
        entry_time = pd.Timestamp(compute_entry_time(pd.Timestamp("2025-07-04T15:00:00Z"), 0))
        self.assertEqual(entry_time.tz_convert("UTC"), pd.Timestamp("2025-07-07T13:30:00Z"))


if __name__ == "__main__":
    unittest.main()
