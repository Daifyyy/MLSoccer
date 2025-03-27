import streamlit as st
import pandas as pd
import joblib
from utils.feature_engineering_extended import generate_extended_features
from utils.data_loader import load_combined_data, get_teams
from utils.model_utils import prepare_features_for_prediction

st.title("⚽ Predikce více než 2,5 gólů")

# Výběr ligy
LEAGUE = st.selectbox("Vyber ligu", ["E0", "SP1", "D1", "I1", "F1", "N1"])
data = load_combined_data(LEAGUE)
teams = get_teams(data)

home_team = st.selectbox("Vyber domácí tým", teams)
away_team = st.selectbox("Vyber hostující tým", teams)

if st.button("🔍 Predikovat"):
    try:
        df_ext = generate_extended_features(data)
        features = prepare_features_for_prediction(df_ext, home_team, away_team)

        # Načti oba modely
        rf_model = joblib.load(f"models/{LEAGUE}_rf_model.joblib")
        xgb_model = joblib.load(f"models/{LEAGUE}_xgb_model.joblib")

        rf_prob = rf_model.predict_proba([features])[0][1]
        xgb_prob = xgb_model.predict_proba([features])[0][1]

        st.markdown("## 📊 Výsledky predikce")
        st.write(f"🎯 **Random Forest** pravděpodobnost Over 2.5: `{rf_prob:.2%}`")
        st.write(f"🎯 **XGBoost** pravděpodobnost Over 2.5: `{xgb_prob:.2%}`")

        rf_pred = "✅ Ano" if rf_prob > 0.5 else "❌ Ne"
        xgb_pred = "✅ Ano" if xgb_prob > 0.5 else "❌ Ne"

        st.write(f"📌 RF říká: **{rf_pred}**  |  📌 XGB říká: **{xgb_pred}**")

    except Exception as e:
        st.error(f"Nastala chyba: {e}")
