import pandas as pd

def max_drawdown(pnl_series: pd.Series) -> float:
    equity = pnl_series.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    return abs(dd.min())

def compute_metrics(trades: list[dict]) -> dict:
    df = pd.DataFrame(trades)
    if df.empty:
        return {}

    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]

    gross_profit = wins["pnl"].sum()
    gross_loss = losses["pnl"].sum()

    mdd = max_drawdown(df["pnl"])

    expectancy = (
        (len(wins) / len(df)) * wins["pnl"].mean()
        + (len(losses) / len(df)) * losses["pnl"].mean()
    )

    score = expectancy - mdd * 0.5

    return {
        "trades": len(df),
        "win_rate": len(wins) / len(df),
        "profit_factor": (
            gross_profit / abs(gross_loss)
            if gross_loss != 0 else float("inf")
        ),
        # "avg_win": wins["pnl"].mean() if not wins.empty else 0,
        # "avg_loss": losses["pnl"].mean() if not losses.empty else 0,
        "expectancy": expectancy,
        "total_pnl": df["pnl"].sum(),
        "max_drawdown": mdd,
        "score": score,
    }
