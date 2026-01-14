import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import mplcursors

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

    if "value" in df.columns:
        df = df[df["value"] > -1e8].copy()

    df.columns = [c.replace("params_", "") if c.startswith("params_") else c for c in df.columns]
    df.columns = [c.replace("user_attrs_", "ua_") if c.startswith("user_attrs_") else c for c in df.columns]

    for b in ("ema_enabled", "rsi_enabled", "psar_enabled", "ts_enabled"):
        if b in df.columns:
            s = df[b]
            if s.dtype == object:
                mapped = s.astype(str).str.lower().map({"true": True, "false": False})
                df[b] = mapped.fillna(df[b])

    return df.reset_index(drop=True)

def _as_num(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.replace([float("inf"), float("-inf")], pd.NA)

# =======================
# 2D PLOTS
# =======================

def plot_score_by_trial(df: pd.DataFrame):
    if "number" not in df.columns or "value" not in df.columns:
        return
    
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
    if param not in df.columns or "value" not in df.columns:
        print(f"⚠️ Param '{param}' not found")
        return

    y = clip_series(df["value"])
    plt.figure(figsize=(9, 5))
    plt.scatter(df[param], y, alpha=0.7)
    plt.xlabel(param)
    plt.ylabel("Score")
    plt.title(f"Score vs {param}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_box_by_flag(df: pd.DataFrame, flag: str):
    if flag not in df.columns or "value" not in df.columns:
        print(f"⚠️ Flag '{flag}' not found")
        return

    # группируем только по True/False
    sub = df[df[flag].isin([True, False])].copy()
    if sub.empty:
        print(f"⚠️ No boolean data for '{flag}'")
        return

    groups = [sub[sub[flag] == False]["value"].values, sub[sub[flag] == True]["value"].values]
    plt.figure(figsize=(6, 4))
    # plt.boxplot(groups, labels=[f"{flag}=False", f"{flag}=True"], showfliers=False)
    plt.boxplot(groups, tick_labels=[f"{flag}=False", f"{flag}=True"], showfliers=False)
    plt.ylabel("Score")
    plt.title(f"Score distribution by {flag}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_exit_reason_stacked(df: pd.DataFrame):
    need = ["value", "ua_exit_sl_frac", "ua_exit_tp_frac", "ua_exit_time_exit_frac"]
    for c in need:
        if c not in df.columns:
            print(f"⚠️ Column '{c}' not found (need {need}). "
                  f"Убедись, что run_optuna пишет ua_exit_*_frac")
            return

    has_psar = "ua_exit_psar_frac" in df.columns
    has_ts = "ua_exit_ts_frac" in df.columns
    # Сгруппируем trials по квантилям score (чтобы было видно зависимость)
    tmp = df.copy()
    tmp["score_q"] = pd.qcut(tmp["value"], q=5, labels=["Q1","Q2","Q3","Q4","Q5"], duplicates="drop")

    agg_cols = {
        "ua_exit_sl_frac": "mean",
        "ua_exit_tp_frac": "mean",
        "ua_exit_time_exit_frac": "mean",
    }
    if has_psar:
        agg_cols["ua_exit_psar_frac"] = "mean"
    if has_ts:
        agg_cols["ua_exit_ts_frac"] = "mean"
    
    grp = tmp.groupby("score_q", observed=False).agg(agg_cols)

    cols = ["ua_exit_sl_frac", "ua_exit_tp_frac", "ua_exit_time_exit_frac"]
    labels = ["sl", "tp", "time_exit"]
    if has_psar:
        cols.append("ua_exit_psar_frac")
        labels.append("psar")
    if has_ts:
        cols.append("ua_exit_ts_frac")
        labels.append("ts")

    data = grp[cols].fillna(0.0)

    row_sum = data.sum(axis=1)
    other = (1.0 - row_sum).clip(lower=0.0)
    if (other > 1e-6).any():
        data["other"] = other
        cols.append("other")
        labels.append("other")
    
    plt.figure(figsize=(10, 5))
    bottom = None
    x = range(len(data))
    for i, col in enumerate(cols):
        if bottom is None:
            plt.bar(x, data[col].values, label=labels[i])
            bottom = data[col].values
        else:
            plt.bar(x, data[col].values, bottom=bottom, label=labels[i])
            bottom = bottom + data[col].values

    plt.xticks(x, [str(i) for i in data.index.astype(str)], rotation=25, ha="right")
    plt.ylabel("Average exit_reason fraction")
    plt.xlabel("Score quantile (low → high)")
    plt.title("Exit reason fractions vs score quantiles")
    plt.legend()
    plt.tight_layout()
    plt.show()

# =======================
# PARETO (PnL vs DD)
# =======================

def plot_pareto_pnl_dd(df: pd.DataFrame):
    # Это не multi-objective, но визуально помогает выбрать компромисс.
    # Чем выше total_pnl и ниже max_drawdown — тем лучше.
    need = ["ua_total_pnl", "ua_max_drawdown", "value"]
    for c in need:
        if c not in df.columns:
            print(f"⚠️ Column '{c}' not found (need {need})")
            return

    x = _as_num(df["ua_max_drawdown"])
    y = _as_num(df["ua_total_pnl"])
    c = clip_series(df["value"]) if "value" in df.columns else None

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, y, c=c, alpha=0.8, cmap="viridis")
    plt.xlabel("Max Drawdown")
    plt.ylabel("Total PnL")
    plt.title("Pareto view: PnL vs Max Drawdown (color = Score)")
    plt.grid(True)
    plt.colorbar(sc, label="Score")
    if "number" in df.columns:
        cursor = mplcursors.cursor(sc, hover=True)

        @cursor.connect("add")
        def _(sel):
            i = sel.index
            sel.annotation.set_text(
                f"trial: {int(df.iloc[i]['number'])}\n"
                f"score: {df.iloc[i]['value']:.3f}\n"
                f"PnL: {df.iloc[i]['ua_total_pnl']:.1f}\n"
                f"MDD: {df.iloc[i]['ua_max_drawdown']:.1f}"
            )

    plt.tight_layout()
    plt.show()

# =======================
# PARETO (PnL vs AHM)
# =======================

def plot_pareto_pnl_vs_hold(df: pd.DataFrame):
    # Pareto: pnl vs avg_hold_minutes (оба берём из ua_*, потому что это метрики)
    need = ["ua_total_pnl", "ua_avg_hold_minutes", "value"]
    for c in need:
        if c not in df.columns:
            print(f"⚠️ Column '{c}' not found (need {need}). "
                  f"Убедись, что run_optuna пишет user_attrs_total_pnl и user_attrs_avg_hold_minutes.")
            return

    x = _as_num(df["ua_avg_hold_minutes"])
    y = _as_num(df["ua_total_pnl"])
    c = clip_series(df["value"]) if "value" in df.columns else None

    plt.figure(figsize=(9, 6))
    sc = plt.scatter(x, y, c=c, alpha=0.7, cmap="viridis")
    plt.xlabel("avg_hold_minutes (fact)")
    plt.ylabel("total_pnl")
    plt.title("Pareto: total_pnl vs avg_hold_minutes (color = Score)")
    plt.grid(True)
    plt.colorbar(sc, label="Score")
    if "number" in df.columns:
        cursor = mplcursors.cursor(sc, hover=True)

        @cursor.connect("add")
        def _(sel):
            i = sel.index
            sel.annotation.set_text(
                f"trial: {int(df.iloc[i]['number'])}\n"
                f"score: {df.iloc[i]['value']:.3f}\n"
                f"PnL: {df.iloc[i]['ua_total_pnl']:.1f}\n"
                f"AHM: {df.iloc[i]['ua_avg_hold_minutes']:.1f}"
            )

    plt.tight_layout()
    plt.show()

# =======================
# PARETO (HM vs AHM)
# =======================

def plot_hold_sanity(df: pd.DataFrame):
    need = ["ua_avg_hold_minutes", "holding_minutes", "value"]
    for c in need:
        if c not in df.columns:
            print(f"⚠️ Column '{c}' not found (need {need})")
            return

    x = _as_num(df["holding_minutes"])
    y = _as_num(df["ua_avg_hold_minutes"])
    c = clip_series(df["value"]) if "value" in df.columns else None

    # m = x.notna() & y.notna()
    # x = x[m]
    # y = y[m]

    # if len(x) == 0:
    #     print("⚠️ No valid data for hold sanity plot")
    #     return

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(x, y, c=c, alpha=0.6)

    # линия y = x
    mn = float(min(x.min(), y.min()))
    mx = float(max(x.max(), y.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")

    plt.xlabel("holding_minutes (param)")
    plt.ylabel("avg_hold_minutes (fact)")
    plt.title("Sanity: avg_hold_minutes vs holding_minutes (y=x)")
    plt.grid(True)
    plt.colorbar(sc, label="Score")
    if "number" in df.columns:
        cursor = mplcursors.cursor(sc, hover=True)

        @cursor.connect("add")
        def _(sel):
            i = sel.index
            sel.annotation.set_text(
                f"trial: {int(df.iloc[i]['number'])}\n"
                f"score: {df.iloc[i]['value']:.3f}\n"
                f"HM: {df.iloc[i]['holding_minutes']:.1f}\n"
                f"AHM: {df.iloc[i]['ua_avg_hold_minutes']:.1f}"
            )

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
    need = ["sl", "tp", "ua_avg_hold_minutes", "value"]
    for c in need:
        if c not in df.columns:
            print(f"⚠️ Column '{c}' not found (need {need})")
            return

    x = df["sl"]
    y = df["tp"]
    score = df["value"]
    hold = df["ua_avg_hold_minutes"]

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
    for p in ["sl", "tp", "holding_minutes", "delay_open", "psar_step", "psar_max", "ts_dist", "ts_step"]:
        plot_param_2d(df, p)

    # 2D: распределение score по флагам
    for flag in ("ema_use", "rsi_use", "psar_use", "ts_use"):
        if flag in df.columns:
                plot_box_by_flag(df, flag)

    # Pareto (если в user_attrs сохранены total_pnl и max_drawdown)
    plot_pareto_pnl_dd(df)

    # Pareto: pnl vs avg hold
    plot_pareto_pnl_vs_hold(df)

    plot_hold_sanity(df)

    # Exit reasons vs score
    plot_exit_reason_stacked(df)

    # 3D
    plot_3d(df, "sl", "tp", "value")
    # plot_3d(df, "holding_minutes", "delay_open", "value")

    # 4D bubble
    plot_bubble_sl_tp(df)

if __name__ == "__main__":
    main()
