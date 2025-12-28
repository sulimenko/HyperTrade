import os
import sys
# os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/results/optuna"
CSV_PATH = f"{PATH}/trials.csv"

df = pd.read_csv(CSV_PATH)

# ---------- 1. DISTRIBUTION ----------
plt.figure(figsize=(10,5))
sns.histplot(df["value"], bins=50, kde=True)
plt.axvline(df["value"].max(), color="red", linestyle="--")
plt.title("Objective Distribution")
plt.grid()
plt.show()

# ---------- 2. TOP 10% ----------
threshold = df["value"].quantile(0.90)
top = df[df["value"] >= threshold]

# ---------- 3. 3D SCATTER ----------
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    top["params_sl"],
    top["params_tp"],
    top["params_delay_open"],
    c=top["value"],
    cmap="viridis",
    s=40
)

ax.set_xlabel("SL")
ax.set_ylabel("TP")
ax.set_zlabel("Delay Open")
ax.set_title("Top 10% parameter space")

plt.show()

# ---------- 4. HEATMAP ----------
pivot = top.pivot_table(
    values="value",
    index="params_sl",
    columns="params_tp",
    aggfunc="mean"
)

plt.figure(figsize=(10,6))
sns.heatmap(pivot, cmap="viridis")
plt.title("SL vs TP (mean objective)")
plt.show()

# ---------- 5. PARALLEL COORDINATES ----------
from pandas.plotting import parallel_coordinates

plot_df = top[
    ["value", "params_sl", "params_tp", "params_delay_open"]
].copy()

if plot_df["value"].nunique() > 1:
    plot_df["bucket"] = pd.qcut(
        plot_df["value"],
        q=3,
        labels=["low", "mid", "high"],
        duplicates="drop"
    )
else:
    plot_df["bucket"] = "top"

plt.figure(figsize=(12,6))
parallel_coordinates(plot_df, "bucket", alpha=0.3)
plt.title("Parallel Coordinates (Top 10%)")
plt.grid()
plt.show()

# ---------- 6. ROBUST PARAMS ----------
print("\n=== ROBUST PARAMS (MEDIAN OF TOP 10%) ===")
print(top[[
    "params_sl",
    "params_tp",
    "params_delay_open",
    "params_holding_minutes"
]].median())
