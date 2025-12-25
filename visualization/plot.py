import pandas as pd
import matplotlib.pyplot as plt

wf = pd.read_csv("data/results/walk_forward.csv")

plt.plot(wf["end"], wf["avg_pnl"], marker="o")
plt.title("Walk-forward performance")
plt.ylabel("Avg PnL")
plt.xlabel("Period end")
plt.grid(True)
plt.show()
