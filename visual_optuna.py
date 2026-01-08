import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =======================
# HELPER
# =======================
def clip_series(s: pd.Series, q_low=0.01, q_high=0.99):
    lo = s.quantile(q_low)
    hi = s.quantile(q_high)
    return s.clip(lo, hi)

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
        df = df[df["state"] == "COMPLETE"].copy()

    df.columns = [c.replace("params_", "") if c.startswith("params_") else c for c in df.columns]
    df.columns = [c.replace("user_attrs_", "") if c.startswith("user_attrs_") else c for c in df.columns]

    for b in ("ema_enabled", "rsi_enabled"):
        if b in df.columns:
            df[b] = df[b].astype(str).str.lower().map({"true": True, "false": False}).fillna(df[b]) # может быть True/False, "True"/"False", 1/0

    return df.reset_index(drop=True)


# =======================
# 2D PLOTS
# =======================
def plot_score_by_trial(df: pd.DataFrame):
    y = clip_series(df["value"])
    plt.figure(figsize=(10, 5))
    plt.scatter(df["number"], y, alpha=0.7)
    plt.xlabel("Trial")
    plt.ylabel("Score")
    plt.title("Optuna: Score by Trial (robust scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_param_2d(df: pd.DataFrame, param: str):
    if param not in df.columns:
        print(f"⚠️ Param '{param}' not found")
        return

    plt.figure(figsize=(8, 5))
    y = clip_series(df["value"])
    plt.scatter(df[param], y, alpha=0.7)
    plt.xlabel(param)
    plt.ylabel("Score")
    plt.title(f"Score vs {param}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_box_by_flag(df: pd.DataFrame, flag: str):
    if flag not in df.columns:
        print(f"⚠️ Flag '{flag}' not found")
        return

    # группируем только по True/False
    sub = df[df[flag].isin([True, False])].copy()
    if sub.empty:
        print(f"⚠️ No boolean data for '{flag}'")
        return

    groups = [sub[sub[flag] == False]["value"].values, sub[sub[flag] == True]["value"].values]
    plt.figure(figsize=(7, 4))
    # plt.boxplot(groups, labels=[f"{flag}=False", f"{flag}=True"], showfliers=False)
    plt.boxplot(groups, tick_labels=[f"{flag}=False", f"{flag}=True"], showfliers=False)
    plt.ylabel("Score")
    plt.title(f"Score distribution by {flag}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# =======================
# PARETO (PnL vs DD)
# =======================
def plot_pareto_pnl_dd(df: pd.DataFrame):
    # Это не multi-objective, но визуально помогает выбрать компромисс.
    # Чем выше total_pnl и ниже max_drawdown — тем лучше.
    need = ["total_pnl", "max_drawdown", "value"]
    for c in need:
        if c not in df.columns:
            print(f"⚠️ Column '{c}' not found (need {need})")
            return

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df["max_drawdown"], df["total_pnl"], c=df["value"], alpha=0.8)
    plt.xlabel("Max Drawdown")
    plt.ylabel("Total PnL")
    plt.title("Pareto view: PnL vs Max Drawdown (color = Score)")
    plt.grid(True)
    plt.colorbar(sc, label="Score")
    plt.tight_layout()
    plt.show()

def plot_hold_sanity(df: pd.DataFrame):
    need = ["avg_hold_minutes", "holding_minutes"]
    for c in need:
        if c not in df.columns:
            print(f"⚠️ Column '{c}' not found (need {need})")
            return

    x = df["holding_minutes"]
    y = df["avg_hold_minutes"]

    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, alpha=0.6)

    # линия y = x
    mn = float(min(x.min(), y.min()))
    mx = float(max(x.max(), y.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")

    plt.xlabel("holding_minutes (param)")
    plt.ylabel("avg_hold_minutes (fact)")
    plt.title("Hold sanity check: avg_hold_minutes vs holding_minutes")
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

    zc = clip_series(df[z])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        df[x],
        df[y],
        zc,
        c=zc,
        cmap="viridis",
        s=40,
        alpha=0.8,
    )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(f"{z} (clipped)")
    ax.set_title(f"3D: {x} vs {y} vs {z}")
    fig.colorbar(sc, ax=ax, shrink=0.6, label=f"{z} (clipped)")

    plt.tight_layout()
    plt.show()


# =======================
# 4D Bubble: sl vs tp (size=avg_hold, color=score)
# =======================
def plot_bubble_sl_tp(df: pd.DataFrame):
    need = ["sl", "tp", "avg_hold_minutes", "value"]
    for c in need:
        if c not in df.columns:
            print(f"⚠️ Column '{c}' not found (need {need})")
            return

    x = df["sl"]
    y = df["tp"]
    score = df["value"]
    hold = df["avg_hold_minutes"]

    # нормируем размер пузырей
    hold_norm = (hold - hold.min()) / (hold.max() - hold.min() + 1e-9)
    sizes = 30 + 170 * hold_norm  # 30..200

    plt.figure(figsize=(9, 6))
    sc = plt.scatter(x, y, s=sizes, c=score, alpha=0.75, cmap="viridis")
    plt.xlabel("sl")
    plt.ylabel("tp")
    plt.title("Bubble 4D: sl vs tp (size=avg_hold_minutes, color=Score)")
    plt.grid(True)
    plt.colorbar(sc, label="Score")
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

    if "value" in df.columns:
        df = df[df["value"] > -1e8].copy()

    print(f"Loaded {len(df)} completed trials")

    # 2D score history
    plot_score_by_trial(df)

    # 2D: score vs ключевые параметры
    for p in ["sl", "tp", "holding_minutes", "delay_open"]:
        plot_param_2d(df, p)

    # 2D: распределение score по флагам
    plot_box_by_flag(df, "ema_enabled")
    plot_box_by_flag(df, "rsi_enabled")

    # Pareto (если в user_attrs сохранены total_pnl и max_drawdown)
    plot_pareto_pnl_dd(df)

    plot_hold_sanity(df)

    # 3D
    plot_3d(df, "sl", "tp", "value")
    plot_3d(df, "holding_minutes", "delay_open", "value")

    # 4D bubble
    plot_bubble_sl_tp(df)


if __name__ == "__main__":
    main()
