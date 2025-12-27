import pandas as pd

def save_walk_forward_results(results, path="data/results/walk_forward.csv"):
    """
    results: list[dict]
    """
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)