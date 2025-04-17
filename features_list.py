feature_cols = [
    "home_team_target_enc",
    "away_team_target_enc",
    "home_team_avg_goals_enc",
    "away_team_avg_goals_enc",
    "elo_diff",
    "xg_proxy_diff",
    "low_tempo_index",
    "defense_suppression_score",
    "goals_home_last5_mean",
    "goals_home_last5_median",
    "goals_home_last5_var",
    "goals_away_last5_mean",
    "goals_away_last5_median",
    "goals_away_last5_var",
    "conceded_home_last5_mean",
    "conceded_home_last5_median",
    "conceded_home_last5_var",
    "conceded_away_last5_mean",
    "conceded_away_last5_median",
    "conceded_away_last5_var",
    "shots_home_last5_mean",
    "shots_home_last5_median",
    "shots_home_last5_var",
    "shots_away_last5_mean",
    "shots_away_last5_median",
    "shots_away_last5_var",
    "shots_on_target_home_last5_mean",
    "shots_on_target_home_last5_median",
    "shots_on_target_home_last5_var",
    "shots_on_target_away_last5_mean",
    "shots_on_target_away_last5_median",
    "shots_on_target_away_last5_var",
    "corners_home_last5_mean",
    "corners_home_last5_median",
    "corners_home_last5_var",
    "corners_away_last5_mean",
    "corners_away_last5_median",
    "corners_away_last5_var",
    "fouls_home_last5_mean",
    "fouls_home_last5_median",
    "fouls_home_last5_var",
    "fouls_away_last5_mean",
    "fouls_away_last5_median",
    "fouls_away_last5_var",
    "yellow_home_last5_mean",
    "yellow_home_last5_median",
    "yellow_home_last5_var",
    "yellow_away_last5_mean",
    "yellow_away_last5_median",
    "yellow_away_last5_var",
    "red_home_last5_mean",
    "red_home_last5_median",
    "red_home_last5_var",
    "red_away_last5_mean",
    "red_away_last5_median",
    "red_away_last5_var",
    "shot_conversion_rate_home",
    "shot_conversion_rate_away",
    "attacking_pressure_home",
    "attacking_pressure_away",
    "goal_per_shot_on_target_home",
    "goal_per_shot_on_target_away",
    "tempo_score_norm",
    "conversion_rate_diff",
    "attacking_pressure_diff",
    "goal_per_shot_on_target_diff",
    "sample_uncertainty_weight",
    "home_advantage_weight_norm",
    "recent_goal_variance_weight",
    "style_chaos_diff",
    "disciplinary_index_diff",
    "h2h_avg_goals_total_adj_norm",
    "goals_vs_weak_home",
    "conceded_vs_weak_home",
    "shots_vs_weak_home",
    "shots_on_vs_weak_home",
    "conversion_vs_weak_home",
    "efficiency_vs_weak_home",
    "goals_vs_average_home",
    "conceded_vs_average_home",
    "shots_vs_average_home",
    "shots_on_vs_average_home",
    "conversion_vs_average_home",
    "efficiency_vs_average_home",
    "goals_vs_strong_home",
    "conceded_vs_strong_home",
    "shots_vs_strong_home",
    "shots_on_vs_strong_home",
    "conversion_vs_strong_home",
    "efficiency_vs_strong_home",
    "goals_vs_weak_away",
    "conceded_vs_weak_away",
    "shots_vs_weak_away",
    "shots_on_vs_weak_away",
    "conversion_vs_weak_away",
    "efficiency_vs_weak_away",
    "goals_vs_average_away",
    "conceded_vs_average_away",
    "shots_vs_average_away",
    "shots_on_vs_average_away",
    "conversion_vs_average_away",
    "efficiency_vs_average_away",
    "goals_vs_strong_away",
    "conceded_vs_strong_away",
    "shots_vs_strong_away",
    "shots_on_vs_strong_away",
    "conversion_vs_strong_away",
    "efficiency_vs_strong_away"
]