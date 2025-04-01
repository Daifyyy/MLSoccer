import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime 
from utils.feature_engineering_extended import generate_extended_features
from utils.data_loader import load_data_by_league, filter_team_matches, filter_h2h_matches

st.set_page_config(layout="wide")
st.title("⚽ Predikce počtu gólů – Over 2.5")

league_code = st.selectbox(
    "Zadej zkratku ligy (např. E0 nebo SP1):",
    ["E0", "SP1", "D1", "N1", "I1", "T1", "F1"]
)

df_raw = load_data_by_league(league_code)

tems = sorted(set(df_raw["HomeTeam"]).union(set(df_raw["AwayTeam"])))
home_team = st.selectbox("Domácí tým:", tems)
away_team = st.selectbox("Hostující tým:", tems)

rf_threshold = 0.32
xgb_threshold = 0.29

features = [
    "shooting_efficiency",
    "elo_rating_home",
    "elo_rating_away",
    "momentum_score",
    "home_xg",
    "away_xg",
    "xg_home_last5",
    "xg_away_last5",
    "corner_diff_last5",
    "shot_on_target_diff_last5",
    "shot_diff_last5m",
    "fouls_diff",
    "card_diff",
    "boring_match_score",
    "match_weight",
    "tempo_score",
    "passivity_score",
    "missing_corner_diff_last5",
    "missing_shot_on_target_diff_last5",
    "missing_shot_diff_last5m",
    "missing_fouls_diff",
    "missing_card_diff",
    "missing_xg_away_last5",
    "missing_xg_home_last5",
]

if st.button("🔍 Spustit predikci"):
    try:
        df_filtered = pd.concat([
            filter_team_matches(df_raw, home_team),
            filter_team_matches(df_raw, away_team),
            filter_h2h_matches(df_raw, home_team, away_team)
        ]).drop_duplicates().reset_index(drop=True)

        future_match = pd.DataFrame([{
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'Date': pd.to_datetime(datetime.today().date()),
            'FTHG': np.nan,
            'FTAG': np.nan,
            'HS': np.nan,
            'AS': np.nan,
            'HST': np.nan,
            'AST': np.nan,
            'HF': np.nan,
            'AF': np.nan,
            'HC': np.nan,
            'AC': np.nan,
            'HY': np.nan,
            'AY': np.nan,
            'HR': np.nan,
            'AR': np.nan,
        }])

        df_pred_input = pd.concat([df_filtered, future_match], ignore_index=True)
        df_ext = generate_extended_features(df_pred_input, mode="predict")

        rf_model_path = f"models/{league_code}_rf_model.joblib"
        xgb_model_path = f"models/{league_code}_xgb_model.joblib"
        rf_model = joblib.load(rf_model_path)
        xgb_model = joblib.load(xgb_model_path)

        match_row = df_ext[
            (df_ext["HomeTeam"] == home_team) &
            (df_ext["AwayTeam"] == away_team) &
            (df_ext["Date"].dt.date == datetime.today().date())
        ]

        if match_row.empty:
            st.warning("⚠️ Nepodařilo se najít vstupní data pro predikci.")
        else:
            X_input = match_row[features].fillna(0)
            rf_prob = rf_model.predict_proba(X_input)[0][1]
            xgb_prob = xgb_model.predict_proba(X_input)[0][1]
            rf_pred = rf_prob > rf_threshold
            xgb_pred = xgb_prob > xgb_threshold

            st.subheader("📊 Výsledky predikce:")
            st.markdown(f"🎲 Random Forest – pravděpodobnost Over 2.5: **{rf_prob*100:.2f}%** → {'✅ ANO' if rf_pred else '❌ NE'}")
            st.markdown(f"🚀 XGBoost – pravděpodobnost Over 2.5: **{xgb_prob*100:.2f}%** → {'✅ ANO' if xgb_pred else '❌ NE'}")

    except Exception as e:
        st.error(f"❌ Nastala chyba během predikce: {e}")
