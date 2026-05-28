import unittest

from hypertrade.config.schemas import FilterSearchSpace
from hypertrade.config.strategy import default_indicator_config
from hypertrade.signals.filter_space import build_strategy_params


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
        self.assertEqual(config["bb"], {"enabled": False, "sign": None, "period": None, "std": None})
        self.assertEqual(config["adx"], {"enabled": False, "mode": None, "minimum": None, "period": None})

    def test_bb_contract_is_sign_period_std(self) -> None:
        trial = FakeTrial()
        search_space = FilterSearchSpace(bb_use=True)
        params = build_strategy_params(trial, search_space)
        self.assertEqual(params.indicator_config["bb"], {"enabled": True, "sign": "below", "period": 10, "std": 1.5})

    def test_adx_contract_is_sign_min_period(self) -> None:
        trial = FakeTrial()
        search_space = FilterSearchSpace(adx_use=True)
        params = build_strategy_params(trial, search_space)
        self.assertEqual(params.indicator_config["adx"], {"enabled": True, "mode": "range", "minimum": 10.0, "period": 10})


if __name__ == "__main__":
    unittest.main()
