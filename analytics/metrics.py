import pandas as pd


def compute_metrics(trades: list[dict]) -> dict:
    df = pd.DataFrame(trades)

    if df.empty:
        return {}

    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]

    gross_profit = wins["pnl"].sum()
    gross_loss = losses["pnl"].sum()

    return {
        "trades": len(df),
        "win_rate": len(wins) / len(df),
        "profit_factor": (
            gross_profit / abs(gross_loss)
            if gross_loss != 0 else float("inf")
        ),
        "avg_win": wins["pnl"].mean() if not wins.empty else 0,
        "avg_loss": losses["pnl"].mean() if not losses.empty else 0,
        "expectancy": (
            (len(wins) / len(df)) * wins["pnl"].mean()
            + (len(losses) / len(df)) * losses["pnl"].mean()
        ),
        "total_pnl": df["pnl"].sum()
    }
