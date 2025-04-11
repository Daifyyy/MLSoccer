
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

    stats = {
        "HomeTeam": {
            "goals": "FTHG", "conceded": "FTAG",
            "shots": "HS", "shots_on": "HST",
            "corners": "HC", "fouls": "HF",
            "yellow": "HY", "red": "HR"
        },
        "AwayTeam": {
            "goals": "FTAG", "conceded": "FTHG",
            "shots": "AS", "shots_on": "AST",
            "corners": "AC", "fouls": "AF",
            "yellow": "AY", "red": "AR"
        }
    }

    for team_type, mapping in stats.items():
        prefix = "home" if team_type == "HomeTeam" else "away"
        for stat_key, col in mapping.items():
            rolled = (
                df.groupby(team_type)[col]
                .apply(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )
            df[f"{stat_key}_{prefix}_last5"] = rolled

    # Zjednodušené tempo a chaos metriky
    df["tempo_score"] = df[[
        "shots_home_last5", "shots_away_last5",
        "shots_on_home_last5", "shots_on_away_last5",
        "corners_home_last5", "corners_away_last5"
    ]].sum(axis=1)

    df["chaos_index"] = df[[
        "fouls_home_last5", "fouls_away_last5",
        "yellow_home_last5", "yellow_away_last5"
    ]].sum(axis=1)

    # Rozdíl v očekávaných metrikách
    df["shots_diff"] = df["shots_home_last5"] - df["shots_away_last5"]
    df["corners_diff"] = df["corners_home_last5"] - df["corners_away_last5"]
    df["fouls_diff"] = df["fouls_home_last5"] - df["fouls_away_last5"]

    features = [col for col in df.columns if (
        col.endswith("_last5") or
        col.endswith("_score") or
        col.endswith("_diff")
    )]

    target_cols = [col for col in df.columns if col.startswith("target_")]

    return df[features + target_cols + ["HomeTeam", "AwayTeam", "Date"]]
