
import pandas as pd
import numpy as np
from pathlib import Path
def calculate_elo(df, k=30, base_rating=1500):
    elo_dict = {}
    elo_home_list = []
    elo_away_list = []

    df = df.sort_values("Date").copy()

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        home_goals = row["FTHG"]
        away_goals = row["FTAG"]

        elo_home = elo_dict.get(home, base_rating)
        elo_away = elo_dict.get(away, base_rating)

        if home_goals > away_goals:
            result = 1
        elif home_goals == away_goals:
            result = 0.5
        else:
            result = 0

        expected_home = 1 / (1 + 10 ** ((elo_away - elo_home) / 400))
        change = k * (result - expected_home)

        elo_dict[home] = elo_home + change
        elo_dict[away] = elo_away - change

        elo_home_list.append(elo_home)
        elo_away_list.append(elo_away)

    df["elo_home"] = elo_home_list
    df["elo_away"] = elo_away_list
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    return df

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

    df = calculate_elo(df)

    # Výpočet ELO strength
    df["elo_all"] = df[["elo_home", "elo_away"]].mean(axis=1)
    df["elo_weak_threshold"] = df["elo_all"].expanding().quantile(0.33)
    df["elo_strong_threshold"] = df["elo_all"].expanding().quantile(0.66)

    def categorize_strength(row, side):
        opp_elo = row["elo_away"] if side == "HomeTeam" else row["elo_home"]
        weak_th = row["elo_weak_threshold"]
        strong_th = row["elo_strong_threshold"]
        if pd.isna(opp_elo) or pd.isna(weak_th) or pd.isna(strong_th):
            return np.nan
        if opp_elo < weak_th:
            return "weak"
        elif opp_elo < strong_th:
            return "average"
        else:
            return "strong"

    df["opponent_strength_home"] = df.apply(lambda row: categorize_strength(row, "HomeTeam"), axis=1)
    df["opponent_strength_away"] = df.apply(lambda row: categorize_strength(row, "AwayTeam"), axis=1)

    metrics = {
        "goals": {"HomeTeam": "FTHG", "AwayTeam": "FTAG"},
        "shots": {"HomeTeam": "HS", "AwayTeam": "AS"},
        "shots_on": {"HomeTeam": "HST", "AwayTeam": "AST"},
    }

    for side in ["HomeTeam", "AwayTeam"]:
        team_col = side
        side_label = "home" if side == "HomeTeam" else "away"
        strength_col = f"opponent_strength_{side_label}"
        for strength in ["weak", "average", "strong"]:
            strength_mask_col = f"mask_{side}_{strength}"
            df[strength_mask_col] = df[strength_col] == strength

        for metric_key, column_map in metrics.items():
            value_col = column_map[side]
            for strength in ["weak", "average", "strong"]:
                output_col = f"{metric_key}_vs_{strength}_{side_label}"
                mask_col = f"mask_{side}_{strength}"
                df[output_col] = np.nan
                for team in df[team_col].unique():
                    team_mask = df[team_col] == team
                    team_df = df[team_mask].copy()
                    team_df["val"] = team_df.apply(
                        lambda row: row[value_col] if row[mask_col] else np.nan, axis=1
                    )
                    df.loc[team_mask, output_col] = (
                        team_df["val"].shift(1).rolling(window=10, min_periods=1).mean().values
                    )

    for side in ["home", "away"]:
        for strength in ["weak", "average", "strong"]:
            goals_col = f"goals_vs_{strength}_{side}"
            shots_col = f"shots_vs_{strength}_{side}"
            shots_on_col = f"shots_on_vs_{strength}_{side}"
            conv_col = f"conversion_vs_{strength}_{side}"
            eff_col = f"efficiency_vs_{strength}_{side}"
            df[conv_col] = df[goals_col] / (df[shots_col] + 0.01)
            df[eff_col] = df[goals_col] / (df[shots_on_col] + 0.01)

    features = [col for col in df.columns if (
        col.startswith("conversion_vs_") or
        col.startswith("efficiency_vs_") or
        col.startswith("goals_vs_") or
        col.startswith("shots_vs_") or
        col.startswith("shots_on_vs_")
    )]
    
    
    target_cols = [col for col in df.columns if col.startswith("target_")]
    print(df.columns)
    df.to_csv("debug_stats.csv",index=False)

    return df[features + target_cols + ["HomeTeam", "AwayTeam", "Date"]]
