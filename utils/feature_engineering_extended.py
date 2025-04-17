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

def save_debug_outputs(df):
    debug_cols_strength = ["HomeTeam", "AwayTeam", "Date", "goals_vs_weak_home", "shots_vs_strong_away"]
    #debug_cols_h2h = ["HomeTeam", "AwayTeam", "Date", "h2h_avg_goals_total", "h2h_over25_ratio"]

    df[debug_cols_strength].tail(20).to_csv("strength_metrics_tail.csv", index=False)
    #df[debug_cols_h2h].tail(20).to_csv("h2h_metrics_tail.csv", index=False)

def generate_features(df, mode="train"):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)

    df["FTG"] = df["FTHG"] + df["FTAG"]
    df["target_over25"] = (df["FTG"] > 2.5).astype(int)

    df = calculate_elo(df)

    df["home_team_target_enc"] = df.groupby("HomeTeam")["target_over25"].transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())
    df["away_team_target_enc"] = df.groupby("AwayTeam")["target_over25"].transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())
    df["home_team_avg_goals_enc"] = df.groupby("HomeTeam")["FTHG"].transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())
    df["away_team_avg_goals_enc"] = df.groupby("AwayTeam")["FTAG"].transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())

    df["league_avg_goals"] = (df["FTHG"] + df["FTAG"]).expanding().mean()
    df["league_over25_ratio"] = ((df["FTHG"] + df["FTAG"]) > 2.5).expanding().mean()

    for side in ["home", "away"]:
        for strength in ["weak", "average", "strong"]:
            for metric in ["goals", "conceded", "shots", "shots_on", "conversion", "efficiency"]:
                col = f"{metric}_vs_{strength}_{side}"
                if col not in df.columns:
                    df[col] = np.nan

    if mode == "predict":
        rolling_cols = [col for col in df.columns if "_last5_" in col or col.endswith("_last5_mean")]
        for col in rolling_cols:
            col_mean = df[col].mean()
            df[col] = df[col].fillna(col_mean if not np.isnan(col_mean) else 0)

    
    
    stats = {
        "HomeTeam": {"goals": "FTHG", "conceded": "FTAG", "shots": "HS", "shots_on_target": "HST", "corners": "HC", "fouls": "HF", "yellow": "HY", "red": "HR"},
        "AwayTeam": {"goals": "FTAG", "conceded": "FTHG", "shots": "AS", "shots_on_target": "AST", "corners": "AC", "fouls": "AF", "yellow": "AY", "red": "AR"}
    }

    missing_features = []
    for team_type, mapping in stats.items():
        prefix = "home" if team_type == "HomeTeam" else "away"
        for stat_key, col in mapping.items():
            try:
                rolled = df.groupby(team_type)[col].apply(lambda x: x.shift(1).rolling(5, min_periods=1).agg(['mean', 'median', 'var'])).reset_index(level=0, drop=True)
                df[f"{stat_key}_{prefix}_last5_mean"] = rolled['mean']
                df[f"{stat_key}_{prefix}_last5_median"] = rolled['median']
                df[f"{stat_key}_{prefix}_last5_var"] = rolled['var']
            except Exception as e:
                missing_features.extend([f"{stat_key}_{prefix}_last5_mean", f"{stat_key}_{prefix}_last5_median", f"{stat_key}_{prefix}_last5_var"])
                print(f"Warning: Could not calculate rolling stats for {col} ({prefix}): {e}")

        df[f"shot_conversion_rate_{prefix}"] = df.get(f"goals_{prefix}_last5_mean", 0) / (df.get(f"shots_{prefix}_last5_mean", 0) + 0.01)
        df[f"attacking_pressure_{prefix}"] = df.get(f"shots_on_target_{prefix}_last5_mean", 0) / (df.get(f"shots_{prefix}_last5_mean", 0) + 0.01)
        df[f"disciplinary_index_{prefix}"] = (df.get(f"yellow_{prefix}_last5_mean", 0) + 2 * df.get(f"red_{prefix}_last5_mean", 0)) / (df.get(f"fouls_{prefix}_last5_mean", 0) + 0.01)
        df[f"goal_per_shot_on_target_{prefix}"] = df.get(f"goals_{prefix}_last5_mean", 0) / (df.get(f"shots_on_target_{prefix}_last5_mean", 0) + 0.01)

    
    
    df["tempo_score"] = df.get("shots_home_last5_mean", 0) + df.get("shots_away_last5_mean", 0) + df.get("shots_on_target_home_last5_mean", 0) + df.get("shots_on_target_away_last5_mean", 0) + df.get("corners_home_last5_mean", 0) + df.get("corners_away_last5_mean", 0)
    df["tempo_score_norm"] = (df["tempo_score"] - df["tempo_score"].mean()) / (df["tempo_score"].std() + 1e-5)
    
    df["conversion_rate_diff"] = df["shot_conversion_rate_home"] - df["shot_conversion_rate_away"]
    df["attacking_pressure_diff"] = df["attacking_pressure_home"] - df["attacking_pressure_away"]
    df["goal_per_shot_on_target_diff"] = df["goal_per_shot_on_target_home"] - df["goal_per_shot_on_target_away"]
    df["disciplinary_index_diff"] = df["disciplinary_index_home"] - df["disciplinary_index_away"]

    if mode == "train":
        df["match_weight"] = np.exp(-(df.index.max() - df.index) / df.shape[0])
    else:
        df["match_weight"] = 1.0

    
    df["elo_home_norm"] = (df["elo_home"] - df["elo_home"].mean()) / (df["elo_home"].std() + 1e-5)
    df["elo_away_norm"] = (df["elo_away"] - df["elo_away"].mean()) / (df["elo_away"].std() + 1e-5)
    df["home_advantage_weight"] = df["elo_home_norm"] - df["elo_away_norm"]
    df["home_advantage_weight_norm"] = (df["home_advantage_weight"] - df["home_advantage_weight"].mean()) / (df["home_advantage_weight"].std() + 1e-5)
    df["sample_uncertainty_weight"] = 1 / (1 + df["elo_diff"].abs() / 400)
    df["recent_goal_variance_raw"] = df["goals_home_last5_var"].fillna(0) + df["goals_away_last5_var"].fillna(0)
    df["recent_goal_variance_weight"] = np.log1p(df["recent_goal_variance_raw"])

    df["style_chaos_index"] = df["corners_home_last5_mean"] + df["fouls_home_last5_mean"] + df["shots_home_last5_mean"] + df["corners_away_last5_mean"] + df["fouls_away_last5_mean"] + df["shots_away_last5_mean"]
    df["style_chaos_index_norm"] = (df["style_chaos_index"] - df["style_chaos_index"].mean()) / (df["style_chaos_index"].std() + 1e-5)
    df["style_chaos_diff"] = (df["corners_home_last5_mean"] + df["fouls_home_last5_mean"] + df["shots_home_last5_mean"]) - (df["corners_away_last5_mean"] + df["fouls_away_last5_mean"] + df["shots_away_last5_mean"])

    df["elo_all"] = df[["elo_home", "elo_away"]].mean(axis=1)
    df["elo_all_norm"] = (df["elo_all"] - df["elo_all"].mean()) / (df["elo_all"].std() + 1e-5)
    df["elo_weak_threshold"] = df["elo_all_norm"].expanding().quantile(0.33)
    df["elo_strong_threshold"] = df["elo_all_norm"].expanding().quantile(0.66)


    # === xG-proxy metriky ===
    df["home_xg_proxy"] = (
        df["shots_on_target_home_last5_mean"] * 0.3 +
        (df["shots_home_last5_mean"] - df["shots_on_target_home_last5_mean"]) * 0.1
    )
    df["away_xg_proxy"] = (
        df["shots_on_target_away_last5_mean"] * 0.3 +
        (df["shots_away_last5_mean"] - df["shots_on_target_away_last5_mean"]) * 0.1
    )
    df["xg_proxy_diff"] = df["home_xg_proxy"] - df["away_xg_proxy"]

    # === Low tempo index ===
    df["low_tempo_index"] = (
        (df["shots_home_last5_mean"] + df["shots_away_last5_mean"] +
        df["fouls_home_last5_mean"] + df["fouls_away_last5_mean"]) < 20
    ).astype(int)

    # === Defense suppression score ===
    df["defense_suppression_score"] = (
        (1 - df["shot_conversion_rate_away"]) * df["disciplinary_index_home"] +
        (1 - df["shot_conversion_rate_home"]) * df["disciplinary_index_away"]
    )

    

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
        "conceded": {"HomeTeam": "FTAG", "AwayTeam": "FTHG"},
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

    performance_vs_strength_features = []
    for side in ["home", "away"]:
        for strength in ["weak", "average", "strong"]:
            for metric in ["goals", "conceded", "shots", "shots_on"]:
                performance_vs_strength_features.append(f"{metric}_vs_{strength}_{side}")
            for derived in ["conversion", "efficiency"]:
                performance_vs_strength_features.append(f"{derived}_vs_{strength}_{side}")

    # df["h2h_avg_goals_total"] = np.nan
    # df["h2h_over25_ratio"] = np.nan
    for idx, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        date = row["Date"]
        past_h2h = df[((df["HomeTeam"].str.strip() == home) & (df["AwayTeam"].str.strip() == away)) | ((df["HomeTeam"].str.strip() == away) & (df["AwayTeam"].str.strip() == home)) & (df["Date"] < date)].sort_values("Date").tail(5)
        if len(past_h2h) >= 2:
            goals = past_h2h["FTHG"] + past_h2h["FTAG"]
            h2h_goals_mean = goals.mean()
            df.at[idx, "h2h_avg_goals_total"] = h2h_goals_mean
            df.at[idx, "h2h_avg_goals_total_adj"] = h2h_goals_mean - df.at[idx, "league_avg_goals"]
            df.at[idx, "h2h_over25_ratio"] = (goals > 2.5).mean()
            

    #df["h2h_avg_goals_total_adj_norm"] = (df["h2h_avg_goals_total_adj"] - df["h2h_avg_goals_total_adj"].mean()) / (df["h2h_avg_goals_total_adj"].std() + 1e-5)


    

    #df[["HomeTeam", "AwayTeam", "Date", "goals_vs_weak_home", "shots_vs_strong_away"]].tail(20)
    #df[["HomeTeam", "AwayTeam", "Date", "h2h_avg_goals_total", "h2h_over25_ratio"]].tail(20)


    final_features = [
        "home_team_target_enc", "away_team_target_enc",
        "home_team_avg_goals_enc", "away_team_avg_goals_enc",
        "elo_diff","xg_proxy_diff", "low_tempo_index", "defense_suppression_score"
    ]

    agg_keys = ["goals", "conceded", "shots", "shots_on_target", "corners", "fouls", "yellow", "red"]
    aggs = ["mean", "median", "var"]
    sides = ["home", "away"]

    feature_block_aggs = [f"{key}_{side}_last5_{agg}" for key in agg_keys for side in sides for agg in aggs]
    feature_block_derived = [f"{metric}_{side}" for metric in ["shot_conversion_rate", "attacking_pressure", "goal_per_shot_on_target"] for side in sides]
    feature_block_diffs = [
        "tempo_score_norm", "conversion_rate_diff", "attacking_pressure_diff", "goal_per_shot_on_target_diff",
        "sample_uncertainty_weight", "home_advantage_weight_norm", "recent_goal_variance_weight",
        "style_chaos_diff", "disciplinary_index_diff" #h2h_avg_goals_total_adj_norm
    ]

    final_features += feature_block_aggs + feature_block_derived + feature_block_diffs + performance_vs_strength_features

    features_list_code = "feature_cols = [\n" + ",\n".join([f'    \"{f}\"' for f in final_features]) + "\n]"
    Path("features_list.py").write_text(features_list_code)

    
    if mode == "predict":
        rolling_cols = [col for col in df.columns if "_last5_" in col or col.endswith("_last5_mean")]
        for col in rolling_cols:
            col_mean = df[col].mean()
            df[col] = df[col].fillna(col_mean if not np.isnan(col_mean) else 0)
            
    if mode == "train":
            save_debug_outputs(df)    
            df.to_csv("debug_all_features.csv", index=False)
            

    return df[final_features + ["HomeTeam", "AwayTeam", "Date", "target_over25", "match_weight"]]
