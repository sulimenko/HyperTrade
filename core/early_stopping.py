class EarlyStopper:
    def __init__(self, patience=40, min_delta=0.01, warmup=30):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup
        self.best = None
        self.counter = 0

    def __call__(self, study, trial):
        # не останавливаемся слишком рано
        if trial.number < self.warmup:
            return

        current_best = study.best_value

        if self.best is None or current_best > self.best + self.min_delta:
            self.best = current_best
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            study.stop()
