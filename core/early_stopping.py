class EarlyStopper:
    def __init__(self, patience: int = 50, warmup: int = 30):
        self.patience = patience
        self.warmup = warmup
        self.best = None
        self.counter = 0

    def __call__(self, study, trial):
        if len(study.trials) < self.warmup:
            return

        # берем первую цель (return)
        current_best = max(
            t.values[0] for t in study.best_trials
            if t.values is not None
        )

        if self.best is None or current_best > self.best:
            self.best = current_best
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            study.stop()
