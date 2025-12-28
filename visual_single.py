# import os
# os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# import matplotlib
# matplotlib.use("Agg")

import sys
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

path = Path(sys.argv[1])

trades_path = path / "trades.csv"

if not trades_path.exists():
    raise FileNotFoundError("trades.csv not found in results directory")

trades = pd.read_csv(trades_path, dtype={"pnl": "float32"})

required_cols = {"pnl", "exit_dt"}
missing = required_cols - set(trades.columns)
if missing:
    raise ValueError(f"Missing columns in trades.csv: {missing}")

trades["exit_datetime"] = pd.to_datetime(trades["exit_dt"])
trades = trades.sort_values("exit_datetime")

trades["equity"] = trades["pnl"].cumsum()

if trades.empty:
    print("No trades to plot")
    sys.exit(0)

# plot
plt.figure(figsize=(12, 6))
plt.plot(trades["exit_datetime"], trades["equity"])
plt.title("Equity Curve")
plt.xlabel("Time")
plt.ylabel("Cumulative PnL")
plt.grid(True)
plt.tight_layout()
plt.show()

# out_file = path / "equity_curve.png"
# plt.savefig(out_file, dpi=150)
# plt.close()

# print(f"Saved equity curve to {out_file}")