import pandas as pd

def save_optimization_results(study):
    df = study.trials_dataframe()
    df.to_csv("data/results/optimization_results.csv", index=False)
