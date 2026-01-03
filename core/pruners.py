import optuna

class DailyPerformancePruner:
    def __init__(self, min_avg_pnl=-0.5, warmup_days=5):
        self.min_avg_pnl = min_avg_pnl
        self.warmup_days = warmup_days

    def __call__(self, study, trial):
        stats = trial.user_attrs.get("daily_stats")
        if not stats or len(stats) < self.warmup_days:
            return

        avg = sum(s["avg_pnl"] for s in stats) / len(stats)
        if avg < self.min_avg_pnl:
            raise optuna.TrialPruned()
