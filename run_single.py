import argparse
import pandas as pd

from loader.signals import load_signals
from core.baskets import backtest
from core.metrics import compute_metrics
from config.params import StrategyParams
from utils.save import create_dir, save_csv, save_json

SIGNALS_PATH = "data/signals/signals.csv"

def main(save=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--signals", type=str, default=SIGNALS_PATH)
    parser.add_argument("--sl", type=float, default=2.5)
    parser.add_argument("--tp", type=float, default=4.0)
    parser.add_argument("--delay_open", type=int, default=120)
    parser.add_argument("--holding_minutes", type=int, default=60*24*3)

    args = parser.parse_args()

    params = StrategyParams(**vars(args))

    signals = load_signals(args.signals)
    trades, signal_stats = backtest(signals, params)

    metrics = compute_metrics(trades)

    if save:
        results_dir = create_dir()
        save_csv(pd.DataFrame(trades), results_dir, "trades.csv")
        save_csv(pd.DataFrame(signal_stats), results_dir, "signals.csv")
        save_json(params, results_dir, "params.json")
        save_json(metrics, results_dir, "summary.json")

    print("\n=== STRATEGY SUMMARY ===")
    for k, v in metrics.items():
        print(f"{k:15}: {v}")

    return metrics, trades

if __name__ == "__main__":
    main()
