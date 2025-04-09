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

def generate_features(df, mode="train"):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)

    # Vytvoření cílového sloupce bez data leakage
    df["FTG"] = df["FTHG"] + df["FTAG"]
    df["target_over25"] = (df["FTG"] > 2.5).astype(int)

    df = calculate_elo(df)

    # Target encoding bez leakage
    df["home_team_target_enc"] = df.groupby("HomeTeam")["target_over25"].transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())
    df["away_team_target_enc"] = df.groupby("AwayTeam")["target_over25"].transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())
    df["home_team_avg_goals_enc"] = df.groupby("HomeTeam")["FTHG"].transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())
    df["away_team_avg_goals_enc"] = df.groupby("AwayTeam")["FTAG"].transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())

    stats = {
        "HomeTeam": {
            "goals": "FTHG", "conceded": "FTAG",
            "shots": "HS", "shots_on_target": "HST",
            "corners": "HC", "fouls": "HF",
            "yellow": "HY", "red": "HR"
        },
        "AwayTeam": {
            "goals": "FTAG", "conceded": "FTHG",
            "shots": "AS", "shots_on_target": "AST",
            "corners": "AC", "fouls": "AF",
            "yellow": "AY", "red": "AR"
        }
    }

    missing_features = []

    for team_type, mapping in stats.items():
        prefix = "home" if team_type == "HomeTeam" else "away"
        for stat_key, col in mapping.items():
            try:
                rolled = (
                    df.groupby(team_type)[col]
                    .apply(lambda x: x.shift(1).rolling(5, min_periods=1).agg(['mean', 'median', 'var']))
                    .reset_index(level=0, drop=True)
                )
                df[f"{stat_key}_{prefix}_last5_mean"] = rolled['mean']
                df[f"{stat_key}_{prefix}_last5_median"] = rolled['median']
                df[f"{stat_key}_{prefix}_last5_var"] = rolled['var']
            except Exception as e:
                missing_features.extend([
                    f"{stat_key}_{prefix}_last5_mean",
                    f"{stat_key}_{prefix}_last5_median",
                    f"{stat_key}_{prefix}_last5_var"
                ])
                print(f"Warning: Could not calculate rolling stats for {col} ({prefix}): {e}")

        # Odvozené metriky
        df[f"shot_conversion_rate_{prefix}"] = df.get(f"goals_{prefix}_last5_mean", 0) / (df.get(f"shots_{prefix}_last5_mean", 0) + 0.01)
        df[f"attacking_pressure_{prefix}"] = df.get(f"shots_on_target_{prefix}_last5_mean", 0) / (df.get(f"shots_{prefix}_last5_mean", 0) + 0.01)
        df[f"disciplinary_index_{prefix}"] = (df.get(f"yellow_{prefix}_last5_mean", 0) + 2 * df.get(f"red_{prefix}_last5_mean", 0)) / (df.get(f"fouls_{prefix}_last5_mean", 0) + 0.01)
        df[f"goal_per_shot_on_target_{prefix}"] = df.get(f"goals_{prefix}_last5_mean", 0) / (df.get(f"shots_on_target_{prefix}_last5_mean", 0) + 0.01)

    # Tempo score jako indikátor útočného tempa
    df["tempo_score"] = (
        df.get("shots_home_last5_mean", 0) + df.get("shots_away_last5_mean", 0) +
        df.get("shots_on_target_home_last5_mean", 0) + df.get("shots_on_target_away_last5_mean", 0) +
        df.get("corners_home_last5_mean", 0) + df.get("corners_away_last5_mean", 0)
    )

    # Rozdílové metriky
    df["conversion_rate_diff"] = df["shot_conversion_rate_home"] - df["shot_conversion_rate_away"]
    df["attacking_pressure_diff"] = df["attacking_pressure_home"] - df["attacking_pressure_away"]
    df["goal_per_shot_on_target_diff"] = df["goal_per_shot_on_target_home"] - df["goal_per_shot_on_target_away"]
    df["disciplinary_index_diff"] = df["disciplinary_index_home"] - df["disciplinary_index_away"]

        # === Váhování zápasů podle stáří ===
    if mode == "train":
        df["match_weight"] = np.exp(-(df.index.max() - df.index) / df.shape[0])
    else:
        df["match_weight"] = 1.0
    
        # === Home advantage weight ===
    df["home_advantage_weight"] = df["elo_home"] - df["elo_away"]   
        # === Sample uncertainty weight ===
    df["sample_uncertainty_weight"] = 1 / (1 + df["elo_diff"].abs() / 400)
    # Vysoký rozptyl znamená nespolehlivost týmového výkonu → nižší váha
    df["recent_goal_variance_weight"] = 1 / (
        1 + df["goals_home_last5_var"].fillna(0) + df["goals_away_last5_var"].fillna(0)
    )
    
    df["style_chaos_index"] = (
    df["corners_home_last5_mean"] + df["fouls_home_last5_mean"] + df["shots_home_last5_mean"] +
    df["corners_away_last5_mean"] + df["fouls_away_last5_mean"] + df["shots_away_last5_mean"]
    )
    df["style_chaos_diff"] = (
    (df["corners_home_last5_mean"] + df["fouls_home_last5_mean"] + df["shots_home_last5_mean"]) -
    (df["corners_away_last5_mean"] + df["fouls_away_last5_mean"] + df["shots_away_last5_mean"])
    )


    
        # === Výkon proti různě silným soupeřům ===
    def categorize_opponent_strength(elo, thresholds):
        if elo < thresholds[0]:
            return 'weak'
        elif elo < thresholds[1]:
            return 'average'
        else:
            return 'strong'
    
    elo_thresholds = df[["elo_home", "elo_away"]].quantile([0.33, 0.66]).values.T

    for side in ["HomeTeam", "AwayTeam"]:
        team_col = "HomeTeam" if side == "HomeTeam" else "AwayTeam"
        opponent_elo_col = "elo_away" if side == "HomeTeam" else "elo_home"
        goals_col = "FTHG" if side == "HomeTeam" else "FTAG"

        df[f"opponent_strength_{side.lower()}"] = df[opponent_elo_col].apply(lambda x: categorize_opponent_strength(x, elo_thresholds[0]))

        for strength in ["weak", "average", "strong"]:
            mask = df[f"opponent_strength_{side.lower()}"] == strength
            df[f"goals_vs_{strength}_{side.lower()}"] = (
                df.groupby(team_col)[goals_col]
                .transform(lambda x: x.shift(1).where(mask).rolling(10, min_periods=1).mean())
            )
    
    
    
    final_features = [
        "home_team_target_enc", "away_team_target_enc",
        "home_team_avg_goals_enc", "away_team_avg_goals_enc",
        "elo_diff",#,"elo_home", "elo_away"
    ]

    agg_keys = [
        "goals", "conceded", "shots", "shots_on_target", "corners",
        "fouls", "yellow", "red"
    ]
    aggs = ["mean", "median", "var"]
    sides = ["home", "away"]

    feature_block_aggs = [f"{key}_{side}_last5_{agg}" for key in agg_keys for side in sides for agg in aggs]
    feature_block_derived = [f"{metric}_{side}" for metric in ["shot_conversion_rate", "attacking_pressure", "goal_per_shot_on_target"] for side in sides]
    feature_block_diffs = [
        "tempo_score","conversion_rate_diff", "attacking_pressure_diff", "goal_per_shot_on_target_diff",
        "sample_uncertainty_weight","home_advantage_weight","recent_goal_variance_weight","style_chaos_diff","disciplinary_index_diff"
        
        
        ]

    #final_features += feature_block_aggs + feature_block_derived + ["tempo_score"] + feature_block_diffs
    final_features += feature_block_diffs
    # === Uložení features_list.py ===
    features_list_code = "feature_cols = [\n" + ",\n".join([f'    \"{f}\"' for f in final_features]) + "\n]"
    Path("features_list.py").write_text(features_list_code)

    return df[final_features + ["HomeTeam", "AwayTeam", "Date", "target_over25", "match_weight"]]
