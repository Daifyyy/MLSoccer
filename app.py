import streamlit as st
import pandas as pd
import joblib
from utils.feature_engineering_extended import generate_extended_features
from utils.model_utils import prepare_features_for_prediction
import os

st.set_page_config(page_title="Predikce zápasu", layout="centered")
st.title("⚽ Predikce výsledku fotbalového zápasu")

# === Výběr ligy ===
league_files = [f for f in os.listdir("data") if f.endswith("combined_full.csv")]
selected_league_file = st.selectbox("Vyber ligu:", league_files)

# === Načtení dat ===
if selected_league_file:
    league_code = selected_league_file.split("_")[0]
    df = pd.read_csv(f"data/{selected_league_file}")
    df_ext = generate_extended_features(df)

    available_teams = sorted(set(df_ext["HomeTeam"]).union(df_ext["AwayTeam"]))
    home_team = st.selectbox("Domácí tým:", available_teams)
    away_team = st.selectbox("Hostující tým:", [team for team in available_teams if team != home_team])

    if st.button("🔮 Provést predikci"):
        try:
            model_path_rf = f"models/{league_code}_rf_model.joblib"
            model_path_xgb = f"models/{league_code}_xgb_model.joblib"

            if not os.path.exists(model_path_rf) or not os.path.exists(model_path_xgb):
                st.error("Model pro tuto ligu nebyl nalezen. Nejdřív jej natrénuj.")
            else:
                model_rf = joblib.load(model_path_rf)
                model_xgb = joblib.load(model_path_xgb)

                X_pred = prepare_features_for_prediction(df_ext, home_team, away_team)

                proba_rf = model_rf.predict_proba(X_pred)[0][1]
                proba_xgb = model_xgb.predict_proba(X_pred)[0][1]
                avg_proba = (proba_rf + proba_xgb) / 2

                st.subheader("📊 Výsledky predikce")
                st.write(f"**Random Forest pravděpodobnost Over 2.5 gólů:** {proba_rf:.2f}")
                st.write(f"**XGBoost pravděpodobnost Over 2.5 gólů:** {proba_xgb:.2f}")
                st.success(f"**Průměrná pravděpodobnost (Over 2.5): {avg_proba:.2f}**")

        except Exception as e:
            st.error(f"Nastala chyba během predikce: {e}")
