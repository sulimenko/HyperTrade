class EarlyStopper:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def __call__(self, study, trial):
        if self.best is None or trial.value > self.best + self.min_delta:
            self.best = trial.value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            study.stop()