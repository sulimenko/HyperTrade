import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# =======================
# LOAD
# =======================
def load_trials(results_dir: Path) -> pd.DataFrame:
    trials_file = results_dir / "trials.csv"
    if not trials_file.exists():
        raise FileNotFoundError(trials_file)

    df = pd.read_csv(trials_file)

    # Только завершённые
    if "state" in df.columns:
        df = df[df["state"] == "COMPLETE"]

    # params_sl → sl
    df.columns = [
        c.replace("params_", "") if c.startswith("params_") else c
        for c in df.columns
    ]

    return df.reset_index(drop=True)


# =======================
# 2D PLOTS
# =======================
def plot_score_by_trial(df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.scatter(df["number"], df["value"], alpha=0.7)
    plt.xlabel("Trial")
    plt.ylabel("Score (PnL)")
    plt.title("Optuna: Score by Trial")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_param_2d(df: pd.DataFrame, param: str):
    if param not in df.columns:
        print(f"⚠️ Param '{param}' not found")
        return

    plt.figure(figsize=(8, 5))
    plt.scatter(df[param], df["value"], alpha=0.7)
    plt.xlabel(param)
    plt.ylabel("Score (PnL)")
    plt.title(f"Score vs {param}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =======================
# 3D PLOTS
# =======================
def plot_3d(df: pd.DataFrame, x: str, y: str, z: str = "value"):
    for c in (x, y, z):
        if c not in df.columns:
            print(f"⚠️ Column '{c}' not found")
            return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        df[x],
        df[y],
        df[z],
        c=df[z],
        cmap="viridis",
        s=40,
        alpha=0.8,
    )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel("Score (PnL)")
    ax.set_title(f"3D: {x} vs {y} vs Score")

    fig.colorbar(sc, ax=ax, shrink=0.6, label="Score")

    plt.tight_layout()
    plt.show()


# =======================
# 3D WITH EMA SPLIT
# =======================
def plot_3d_by_ema(df: pd.DataFrame, x: str, y: str):
    if "ema_enabled" not in df.columns:
        print("⚠️ ema_enabled not found")
        return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for ema_value, color in [(True, "green"), (False, "red")]:
        subset = df[df["ema_enabled"] == ema_value]
        ax.scatter(
            subset[x],
            subset[y],
            subset["value"],
            label=f"ema_enabled={ema_value}",
            alpha=0.7,
            s=40,
        )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel("Score (PnL)")
    ax.set_title(f"3D by EMA: {x} vs {y}")
    ax.legend()

    plt.tight_layout()
    plt.show()


# =======================
# MAIN
# =======================
def main():
    if len(sys.argv) < 2:
        print("Usage: python visual_optuna.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    df = load_trials(results_dir)

    print(f"Loaded {len(df)} completed trials")

    # 2D
    plot_score_by_trial(df)

    for p in ["sl", "tp", "holding_minutes", "delay_open"]:
        plot_param_2d(df, p)

    # 3D
    plot_3d(df, "sl", "tp")
    plot_3d(df, "holding_minutes", "delay_open")

    # EMA split
    plot_3d_by_ema(df, "sl", "tp")


if __name__ == "__main__":
    main()


# from mpl_toolkits.mplot3d import Axes3D  # noqa

# PATH = sys.argv[1] if len(sys.argv) > 1 else "results/optuna"
# CSV_PATH = f"{PATH}/trials.csv"

# df = pd.read_csv(CSV_PATH)

# # ---------- 1. DISTRIBUTION ----------
# plt.figure(figsize=(8,6))
# plt.scatter(
#     top["user_attrs_max_drawdown"],
#     top["user_attrs_total_pnl"],
#     c=top["value"],
#     cmap="viridis",
#     s=40
# )
# # sns.histplot(df["value"], bins=50, kde=True)
# # plt.axvline(df["value"].max(), color="red", linestyle="--")
# plt.xlabel("Max Drawdown")
# plt.ylabel("Total PnL")
# plt.colorbar(label="Score")
# plt.title("PnL vs Drawdown (Top 10%)")
# plt.grid()
# plt.show()


# plt.figure(figsize=(12,6))
# sns.boxplot(data=top[params])
# plt.title("Parameter Stability (Top 10%)")
# plt.grid()
# plt.show()

# # ---------- 2. TOP 10% ----------
# threshold = df["value"].quantile(0.90)
# top = df[df["value"] >= threshold]

# # ---------- 3. 3D SCATTER ----------
# fig = plt.figure(figsize=(10,7))
# ax = fig.add_subplot(111, projection="3d")

# ax.scatter(
#     top["params_sl"],
#     top["params_tp"],
#     top["params_delay_open"],
#     c=top["value"],
#     cmap="viridis",
#     s=40
# )

# ax.set_xlabel("SL")
# ax.set_ylabel("TP")
# ax.set_zlabel("Delay Open")
# ax.set_title("Top 10% parameter space")

# plt.show()

# # ---------- 4. HEATMAP ----------
# pivot = top.pivot_table(
#     values="value",
#     index="params_sl",
#     columns="params_tp",
#     aggfunc="mean"
# )

# plt.figure(figsize=(10,6))
# sns.heatmap(pivot, cmap="viridis")
# plt.title("SL vs TP (mean objective)")
# plt.show()

# # ---------- 5. PARALLEL COORDINATES ----------
# from pandas.plotting import parallel_coordinates

# plot_df = top[
#     ["value", "params_sl", "params_tp", "params_delay_open"]
# ].copy()

# if plot_df["value"].nunique() > 1:
#     plot_df["bucket"] = pd.qcut(
#         plot_df["value"],
#         q=3,
#         labels=["low", "mid", "high"],
#         duplicates="drop"
#     )
# else:
#     plot_df["bucket"] = "top"

# plt.figure(figsize=(12,6))
# parallel_coordinates(plot_df, "bucket", alpha=0.3)
# plt.title("Parallel Coordinates (Top 10%)")
# plt.grid()
# plt.show()

# # ---------- 6. ROBUST PARAMS ----------
# print("\n=== ROBUST PARAMS (MEDIAN OF TOP 10%) ===")
# print(top[[
#     "params_sl",
#     "params_tp",
#     "params_delay_open",
#     "params_holding_minutes"
# ]].median())
