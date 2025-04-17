import pandas as pd
import numpy as np

def generate_match_result_features(df, mode="train"):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)

    df["target_result"] = df["FTR"].map({"H": 0, "D": 1, "A": 2})
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])

    def calculate_elo(df, k=30, base_rating=1500):
        elo_dict = {}
        elo_home_list = []
        elo_away_list = []
        for _, row in df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            home_goals = row.get("FTHG", 0)
            away_goals = row.get("FTAG", 0)
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

    df = calculate_elo(df)

    stats = {
        "HomeTeam": {"goals": "FTHG", "conceded": "FTAG", "shots": "HS", "shots_on_target": "HST", "corners": "HC", "fouls": "HF", "yellow": "HY", "red": "HR"},
        "AwayTeam": {"goals": "FTAG", "conceded": "FTHG", "shots": "AS", "shots_on_target": "AST", "corners": "AC", "fouls": "AF", "yellow": "AY", "red": "AR"}
    }

    for team_type, mapping in stats.items():
        prefix = "home" if team_type == "HomeTeam" else "away"
        for stat_key, col in mapping.items():
            rolled = df.groupby(team_type)[col].apply(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).reset_index(level=0, drop=True)
            df[f"{stat_key}_{prefix}_last5"] = rolled

    df["goal_diff_last5"] = df["goals_home_last5"] - df["goals_away_last5"]

    df["chaos_index_prenorm"] = (
        df["shots_home_last5"] + df["shots_away_last5"] +
        df["corners_home_last5"] + df["corners_away_last5"] +
        df["fouls_home_last5"] + df["fouls_away_last5"]
    )
    # 🔧 Chaos index normalizace
    df["chaos_index"] = (df["chaos_index_prenorm"] - df["chaos_index_prenorm"].mean()) / (df["chaos_index_prenorm"].std() + 1e-5)

    df["disciplinary_index_home_prenorm"] = (df["yellow_home_last5"] + 2 * df["red_home_last5"]) / (df["fouls_home_last5"] + 0.1)
    df["disciplinary_index_away_prenorm"] = (df["yellow_away_last5"] + 2 * df["red_away_last5"]) / (df["fouls_away_last5"] + 0.1)
    # 🔧 Disciplinární index normalizace
    df["disciplinary_index_home"] = (df["disciplinary_index_home_prenorm"] - df["disciplinary_index_home_prenorm"].mean()) / (df["disciplinary_index_home_prenorm"].std() + 1e-5)

    df["disciplinary_index_away"] = (df["disciplinary_index_away_prenorm"] - df["disciplinary_index_away_prenorm"].mean()) / (df["disciplinary_index_away_prenorm"].std() + 1e-5)

    df["form_home_last5_avg"] = df.groupby("HomeTeam")["target_result"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).apply(lambda r: ((r==0)*3 + (r==1)*1).mean()))
    df["form_away_last5_avg"] = df.groupby("AwayTeam")["target_result"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).apply(lambda r: ((r==2)*3 + (r==1)*1).mean()))

    # H2H metriky
    #df["h2h_avg_goals"] = np.nan
    df["h2h_draw_ratio"] = np.nan
    df["h2h_home_win_ratio"] = np.nan
    df["h2h_away_win_ratio"] = np.nan
    for idx, row in df.iterrows():
        home, away, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
        past_h2h = df[
            (((df["HomeTeam"] == home) & (df["AwayTeam"] == away)) | ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))) &
            (df["Date"] < date)
        ].sort_values("Date").tail(5)
        if len(past_h2h) >= 2:
            goals = past_h2h["FTHG"] + past_h2h["FTAG"]
            results = past_h2h["FTR"]
            #df.at[idx, "h2h_avg_goals"] = goals.mean()
            df.at[idx, "h2h_draw_ratio"] = (results == "D").mean()
            df.at[idx, "h2h_home_win_ratio"] = (results == "H").mean()
            df.at[idx, "h2h_away_win_ratio"] = (results == "A").mean()

    # Remízové metriky
    df["draw_tendency_index"] = df.groupby("HomeTeam")["target_result"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).apply(lambda r: (r==1).mean())
    )
    df["draw_rate_home"] = df.groupby("HomeTeam")["target_result"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).apply(lambda r: (r==1).sum() / len(r) if len(r) > 0 else 0)
    )
    df["draw_rate_away"] = df.groupby("AwayTeam")["target_result"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).apply(lambda r: (r==1).sum() / len(r) if len(r) > 0 else 0)
    )
    df["elo_diff_close"] = df["elo_diff"].abs().apply(lambda x: 1 if x < 20 else 0)

    features = [
        "elo_home", "elo_away", "elo_diff",
        "goals_home_last5", "goals_away_last5",
        "conceded_home_last5", "conceded_away_last5",
        "shots_home_last5", "shots_away_last5",
        "shots_on_target_home_last5", "shots_on_target_away_last5",
        "corners_home_last5", "corners_away_last5",
        "fouls_home_last5", "fouls_away_last5",
        "goal_diff_last5", "chaos_index",
        "disciplinary_index_home", "disciplinary_index_away",
        "form_home_last5_avg", "form_away_last5_avg",
        "h2h_draw_ratio", "h2h_home_win_ratio", "h2h_away_win_ratio",#h2h_avg_goals
        "draw_tendency_index", "draw_rate_home", "draw_rate_away", "elo_diff_close"
    ]
    
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
    
    for idx, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        date = row["Date"]

        past_matches = df[
            (((df["HomeTeam"] == home) & (df["AwayTeam"] == away)) |
             ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))) &
            (df["Date"] < date)
        ].sort_values("Date").tail(5)

        if len(past_matches) >= 2:
            goals = past_matches["FTHG"] + past_matches["FTAG"]
            #df.at[idx, "h2h_avg_goals"] = goals.mean()
            df.at[idx, "h2h_draw_ratio"] = (past_matches["FTHG"] == past_matches["FTAG"]).mean()
            df.at[idx, "h2h_home_win_ratio"] = ((past_matches["HomeTeam"] == home) & (past_matches["FTHG"] > past_matches["FTAG"])).mean()
            df.at[idx, "h2h_away_win_ratio"] = ((past_matches["AwayTeam"] == away) & (past_matches["FTAG"] > past_matches["FTHG"])).mean()

    df.to_csv("debug_match_result.csv",index=False)
    return df[features + ["HomeTeam", "AwayTeam", "Date", "target_result"]]