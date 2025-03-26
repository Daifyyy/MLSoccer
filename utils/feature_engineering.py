import pandas as pd

def compute_stats(subset):
    if subset.empty:
        return pd.Series({"HS": 0, "AS": 0, "HST": 0, "AST": 0, "HC": 0, "AC": 0})
    return subset[["HS", "AS", "HST", "AST", "HC", "AC"]].mean()

def aggregate_features(home_stats, away_stats, h2h_stats, overall_stats):
    return (home_stats + away_stats + h2h_stats + overall_stats) / 4
