
import pandas as pd
import numpy as np

def generate_team_stats_features(df, mode="train"):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)

    # === Cílové proměnné ===
    if mode == "train":
        df["target_goals_home"] = df["FTHG"]
        df["target_goals_away"] = df["FTAG"]
        df["target_shots_home"] = df["HS"]
        df["target_shots_away"] = df["AS"]
        df["target_shots_on_home"] = df["HST"]
        df["target_shots_on_away"] = df["AST"]
        df["target_corners_home"] = df["HC"]
        df["target_corners_away"] = df["AC"]
        df["target_fouls_home"] = df["HF"]
        df["target_fouls_away"] = df["AF"]
        df["target_yellows_home"] = df["HY"]
        df["target_yellows_away"] = df["AY"]
        df["target_reds_home"] = df["HR"]
        df["target_reds_away"] = df["AR"]

    # === Průměry pro posledních 5 zápasů pro střely, fauly a rohy ===
    df["shots_home_last5_mean"] = df.groupby("HomeTeam")["HS"].transform(lambda x: x.rolling(5).mean())
    df["fouls_home_last5_mean"] = df.groupby("HomeTeam")["HF"].transform(lambda x: x.rolling(5).mean())
    df["corners_home_last5_mean"] = df.groupby("HomeTeam")["HC"].transform(lambda x: x.rolling(5).mean())

    df["shots_away_last5_mean"] = df.groupby("AwayTeam")["AS"].transform(lambda x: x.rolling(5).mean())
    df["fouls_away_last5_mean"] = df.groupby("AwayTeam")["AF"].transform(lambda x: x.rolling(5).mean())
    df["corners_away_last5_mean"] = df.groupby("AwayTeam")["AC"].transform(lambda x: x.rolling(5).mean())

    # === Taktické změny pro domácí a venkovní zápasy ===
    df["shots_home_vs_away"] = df["shots_home_last5_mean"] / df["shots_away_last5_mean"]
    df["fouls_home_vs_away"] = df["fouls_home_last5_mean"] / df["fouls_away_last5_mean"]
    df["corners_home_vs_away"] = df["corners_home_last5_mean"] / df["corners_away_last5_mean"]

    # === Dynamická data o soupeřích (historické statistiky soupeřů) ===
    df["avg_shots_against_home"] = df.groupby("HomeTeam")["shots_away_last5_mean"].transform("mean")
    df["avg_fouls_against_home"] = df.groupby("HomeTeam")["fouls_away_last5_mean"].transform("mean")
    df["avg_corners_against_home"] = df.groupby("HomeTeam")["corners_away_last5_mean"].transform("mean")

    df["avg_shots_against_away"] = df.groupby("AwayTeam")["shots_home_last5_mean"].transform("mean")
    df["avg_fouls_against_away"] = df.groupby("AwayTeam")["fouls_home_last5_mean"].transform("mean")
    df["avg_corners_against_away"] = df.groupby("AwayTeam")["corners_home_last5_mean"].transform("mean")

    # === Vytvoření features pro model ===
    features = [col for col in df.columns if (
        col.endswith("_last5") or
        col.endswith("_score") or
        col.endswith("_diff") or
        "avg" in col or  # nově přidané pro soupeře
        "home_vs_away" in col
    )]

    target_cols = [col for col in df.columns if col.startswith("target_")]
    print(df.columns)

    return df[features + target_cols + ["HomeTeam", "AwayTeam", "Date"]]
