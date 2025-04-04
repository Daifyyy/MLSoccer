# core/predictor.py
import pandas as pd
from datetime import datetime
import numpy as np
import joblib
from utils.feature_engineering_extended import generate_extended_features
from utils.data_loader import load_data_by_league, filter_team_matches, filter_h2h_matches

FEATURES = [
    "shooting_efficiency",
    "boring_match_score",
    "away_xg",
    "home_xg",
    "passivity_score",
    "home_form_xg",
    "match_weight",
    "away_form_xg",
    "home_form_shots",
    "elo_rating_away",
    "prob_under25",
    "over25_expectation_gap",
    "away_form_shots",
    "momentum_score",
    "behavior_balance",
    "corner_diff_last5",
    "shot_diff_last5m",
    "elo_rating_home",
    "tempo_score",
    "log_odds_under25",
    "prob_over25",
    "fouls_diff",
    "aggressiveness_score",
    "card_diff",
    "shot_on_target_diff_last5",
    "xg_away_last5",
    "xg_home_last5",
    "missing_xg_home_last5",
    "missing_xg_away_last5",
    "missing_home_form_xg",
    "missing_home_form_shots",
    "missing_away_form_xg",
    "missing_away_form_shots",
    "missing_log_odds_under25",
    "xg_conceded_home_last5",
    "xg_conceded_away_last5",
    "avg_xg_conceded",
    "xg_ratio",
    "defensive_pressure",
    "missing_xg_conceded_home_last5",
    "missing_xg_conceded_away_last5",
    "missing_avg_xg_conceded",
    "missing_xg_ratio",
    "missing_defensive_pressure",  
]
  # sem vložíš ten tvůj rozsáhlý seznam

def calculate_optimal_threshold(model, X_val, y_val):
    y_probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 80)
    best_thresh, best_j_score = 0.5, -1
    for t in thresholds:
        y_pred = y_probs > t
        tp = ((y_val == 1) & (y_pred == 1)).sum()
        fn = ((y_val == 1) & (y_pred == 0)).sum()
        tn = ((y_val == 0) & (y_pred == 0)).sum()
        fp = ((y_val == 0) & (y_pred == 1)).sum()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        j = sensitivity + specificity - 1
        if j > best_j_score:
            best_j_score = j
            best_thresh = t
    return best_thresh

def get_confidence(prob):
    if prob >= 0.8:
        return "🔥 Vysoká"
    elif prob >= 0.65:
        return "✅ Střední"
    else:
        return "⚠️ Nízká"

def predict_match(league_code, home_team, away_team):
    df_raw = load_data_by_league(league_code)
    df_filtered = pd.concat([
        filter_team_matches(df_raw, home_team),
        filter_team_matches(df_raw, away_team),
        filter_h2h_matches(df_raw, home_team, away_team)
    ]).drop_duplicates().reset_index(drop=True)

    future_match = pd.DataFrame([{
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'Date': pd.to_datetime(datetime.today().date()),
    }])

    df_pred_input = pd.concat([df_filtered, future_match], ignore_index=True)
    df_ext = generate_extended_features(df_pred_input, mode="predict")

    match_row = df_ext[
        (df_ext["HomeTeam"] == home_team) &
        (df_ext["AwayTeam"] == away_team) &
        (df_ext["Date"].dt.date == datetime.today().date())
    ]

    for col in FEATURES:
        if col not in match_row.columns:
            match_row[col] = df_ext[col].mean() if col in df_ext.columns else 0

    if match_row.empty:
        return {"error": "⚠️ Nepodařilo se najít vstupní data pro predikci."}

    X_input = match_row[FEATURES].fillna(0)
    rf_model = joblib.load(f"models/{league_code}_rf_model.joblib")
    xgb_model = joblib.load(f"models/{league_code}_xgb_model.joblib")

    df_val_ext = generate_extended_features(df_filtered, mode="train")
    X_val = df_val_ext[FEATURES].fillna(0)
    y_val = df_val_ext["Over_2.5"]

    rf_thresh = calculate_optimal_threshold(rf_model, X_val, y_val)
    xgb_thresh = calculate_optimal_threshold(xgb_model, X_val, y_val)

    rf_prob = rf_model.predict_proba(X_input)[0][1]
    xgb_prob = xgb_model.predict_proba(X_input)[0][1]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "rf_prob": rf_prob,
        "xgb_prob": xgb_prob,
        "rf_pred": rf_prob > rf_thresh,
        "xgb_pred": xgb_prob > xgb_thresh,
        "rf_conf": get_confidence(rf_prob),
        "xgb_conf": get_confidence(xgb_prob)
    }
