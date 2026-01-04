import argparse
import pandas as pd

from loader.signals import load_signals
from core.baskets import backtest
from core.metrics import compute_metrics
from config.params import build_single_params
from utils.save import save_csv

SIGNALS_PATH = "data/signals/signals.csv"

def main(save=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--signals", type=str, default=SIGNALS_PATH)
    parser.add_argument("--sl", type=float, default=2.5)
    parser.add_argument("--tp", type=float, default=4.0)
    parser.add_argument("--delay_open", type=int, default=120)
    parser.add_argument("--holding_minutes", type=int, default=60*24*3)

    parser.add_argument("--ema_enabled", type=bool, default=False)
    parser.add_argument("--ema_sign", type=str, default="above")
    parser.add_argument("--ema_fast", type=int, default=20)
    parser.add_argument("--ema_slow", type=int, default=60)

    parser.add_argument("--rsi_enabled", type=bool, default=False)
    parser.add_argument("--rsi_sign", type=str, default="above")
    parser.add_argument("--rsi_level", type=int, default=50)
    parser.add_argument("--rsi_period", type=int, default=18)

    args = parser.parse_args()

    params = build_single_params(args)

    signals = load_signals(args.signals)
    trades, signal_stats = backtest(signals, params)

    metrics = compute_metrics(trades, params)

    if save:
        save_csv(pd.DataFrame(trades), "trades.csv")
        save_csv(pd.DataFrame(signal_stats), "signals.csv")
        save_csv(pd.DataFrame([params]), "params.csv")
        save_csv(pd.DataFrame([metrics]), "summary.csv")

    print("\n=== STRATEGY SUMMARY ===")
    for k, v in metrics.items():
        print(f"{k:15}: {v}")

    return metrics, trades

if __name__ == "__main__":
    main()
