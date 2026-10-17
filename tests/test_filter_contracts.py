import unittest

from hypertrade.config.schemas import FilterSearchSpace
import pandas as pd
from hypertrade.filters.rules import filters
from hypertrade.config.strategy import default_indicator_config
from hypertrade.optimization.search_space import build_strategy_params


class FakeTrial:
    def suggest_float(self, name, low, high, step=None):
        return low

    def suggest_int(self, name, low, high, step=1):
        return low

    def suggest_categorical(self, name, choices):
        return choices[-1]


class FilterContractsTests(unittest.TestCase):
    def test_indicator_contract_uses_named_keys(self) -> None:
        config = default_indicator_config()
        self.assertEqual(config["bb"], {"enabled": False, "mode": None, "period": None, "std": None, "lower": None, "upper": None})
        self.assertEqual(config["adx"], {"enabled": False, "mode": None, "minimum": None, "period": None})

    def test_bb_modes_change_eligibility(self) -> None:
        trial = FakeTrial()
        search_space = FilterSearchSpace(bb_use=True)
        params = build_strategy_params(trial, search_space)
        self.assertEqual(params.indicator_config["bb"], {"enabled": True, "mode": "trend_continuation", "period": 10, "std": 1.5, "lower": 0.05, "upper": 0.6})

        frame = pd.DataFrame({'bb_p_10_1.5':[0.02]})
        for mode, expected_long, expected_short in [('avoid_extremes',False,False), ('mean_reversion',True,False), ('trend_continuation',False,True)]:
            params.indicator_config['bb']['mode'] = mode
            self.assertEqual(filters(frame,0,params,1),expected_long)
            self.assertEqual(filters(frame,0,params,-1),expected_short)
        params.indicator_config['bb'].update(mode='mean_reversion',lower=0.01)
        self.assertFalse(filters(frame,0,params,1))

    def test_adx_contract_is_sign_min_period(self) -> None:
        trial = FakeTrial()
        search_space = FilterSearchSpace(adx_use=True)
        params = build_strategy_params(trial, search_space)
        self.assertEqual(params.indicator_config["adx"], {"enabled": True, "mode": "range", "minimum": 10.0, "period": 10})


if __name__ == "__main__":
    unittest.main()
