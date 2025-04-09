import streamlit as st
from features_list import feature_cols
import pandas as pd
import numpy as np
import joblib
import json
import os
import importlib
import sys
from datetime import datetime
from utils.feature_engineering_extended import generate_features
from utils.data_loader import load_data_by_league, filter_team_matches, filter_h2h_matches

@st.cache_data(show_spinner=False)
def load_model(path):
    if path in sys.modules:
        importlib.reload(sys.modules[path])

def get_model_importance(model, feature_cols):
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "estimator_") and hasattr(model.estimator_, "feature_importances_"):
            importances = model.estimator_.feature_importances_
        else:
            return None, None

        if importances is None:
            return None, None

        if len(importances) != len(feature_cols):
            importances = importances[:len(feature_cols)]
            feature_cols = feature_cols[:len(importances)]

        return importances, feature_cols
    except Exception as e:
        return None, None

st.set_page_config(layout="wide")
st.title("⚽ Predikce Over 2.5 gólů se sílou sázkové příležitosti")

league_code = st.selectbox("Zvol ligu:", ["E0","E1", "SP1", "D1","D2", "I1", "F1","B1","P1","T1"])

df_raw = load_data_by_league(league_code)
teams = sorted(set(df_raw["HomeTeam"]).union(set(df_raw["AwayTeam"])))
home_team = st.selectbox("Domácí tým:", teams)
away_team = st.selectbox("Hostující tým:", teams)

thresholds_path = f"models/{league_code}_thresholds.json"
if os.path.exists(thresholds_path):
    with open(thresholds_path, "r") as f:
        thresholds = json.load(f)
        rf_thresh = thresholds.get("rf_best_threshold", 0.5)
        xgb_thresh = thresholds.get("xgb_best_threshold", 0.5)
        lgb_thresh = thresholds.get("lgb_best_threshold", 0.5)
else:
    rf_thresh = xgb_thresh = lgb_thresh = 0.5

selected_model = st.radio("Vyber model k analýze:", ["LightGBM", "XGBoost", "Random Forest"])

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
        }])

        df_pred_input = pd.concat([df_filtered, future_match], ignore_index=True)
        df_ext = generate_features(df_pred_input, mode="predict")

        match_row = df_ext[
            (df_ext["HomeTeam"] == home_team) &
            (df_ext["AwayTeam"] == away_team) &
            (df_ext["Date"].dt.date == datetime.today().date())
        ]

        model_path_rf = f"models/{league_code}_rf_model.joblib"
        model_path_xgb = f"models/{league_code}_xgb_model.joblib"
        model_path_lgb = f"models/{league_code}_lgb_model.joblib"

        rf_model = joblib.load(model_path_rf)
        xgb_model = joblib.load(model_path_xgb)
        lgb_model = joblib.load(model_path_lgb)

        for col in feature_cols:
            if col not in match_row.columns:
                match_row[col] = df_ext[col].mean() if col in df_ext.columns else 0

        if match_row.empty:
            st.warning("⚠️ Nepodařilo se najít vstupní data pro predikci.")
        else:
            X_input = match_row[feature_cols].fillna(0)
            if 'match_weight' in X_input.columns:
                X_input = X_input.drop(columns=['match_weight'])

            rf_prob = rf_model.predict_proba(X_input)[0][1]
            xgb_prob = xgb_model.predict_proba(X_input)[0][1]
            lgb_prob = lgb_model.predict_proba(X_input)[0][1]

            rf_pred = rf_prob >= rf_thresh
            xgb_pred = xgb_prob >= xgb_thresh
            lgb_pred = lgb_prob >= lgb_thresh

            shap_pred_model = selected_model

            def get_confidence(prob):
                if prob >= 0.8:
                    return "🔥 Vysoká"
                elif prob >= 0.65:
                    return "✅ Střední"
                else:
                    return "⚠️ Nízká"

            st.subheader("📊 Predikce:")
            st.markdown(f"**Random Forest:** {rf_prob:.2%} ({1 / rf_prob:.2f}) pravděpodobnost Over 2.5 → {'✅ ANO' if rf_pred else '❌ NE'}")
            st.markdown(f"Confidence: {get_confidence(rf_prob)} (threshold: {rf_thresh:.2f})")
            st.markdown("---")

            st.markdown(f"**XGBoost:** {xgb_prob:.2%} ({1 / xgb_prob:.2f}) pravděpodobnost Over 2.5 → {'✅ ANO' if xgb_pred else '❌ NE'}")
            st.markdown(f"Confidence: {get_confidence(xgb_prob)} (threshold: {xgb_thresh:.2f})")
            st.markdown("---")

            st.markdown(f"**LightGBM:** {lgb_prob:.2%} ({1 / lgb_prob:.2f}) pravděpodobnost Over 2.5 → {'✅ ANO' if lgb_pred else '❌ NE'}")
            st.markdown(f"Confidence: {get_confidence(lgb_prob)} (threshold: {lgb_thresh:.2f})")
            st.markdown("---")

            # st.subheader(f"🧠 SHAP vysvětlení predikce ({shap_pred_model})")
            # try:
            #     import shap
            #     import matplotlib.pyplot as plt
            #     model_for_shap = {
            #         "LightGBM": lgb_model,
            #         "XGBoost": xgb_model,
            #         "Random Forest": rf_model
            #     }.get(shap_pred_model, None)

            #     if model_for_shap is not None:
            #         explainer = shap.Explainer(model_for_shap)
            #         shap_values = explainer(X_input)

            #         fig = plt.figure()
            #         shap.plots.waterfall(shap_values[0], max_display=15, show=False)
            #         st.pyplot(fig)
            #     else:
            #         st.warning("Model není dostupný pro SHAP analýzu.")
            # except Exception as e:
            #     st.warning(f"⚠️ Nelze zobrazit SHAP vysvětlení: {e}")

    except Exception as e:
        st.error(f"Chyba při predikci: {e}")

# === ANALÝZA MODELU ===
# st.markdown("---")
# st.header("🔎 Analýza modelu")

model_paths = {
    "LightGBM": f"models/{league_code}_lgb_model.joblib",
    "XGBoost": f"models/{league_code}_xgb_model.joblib",
    "Random Forest": f"models/{league_code}_rf_model.joblib"
}

thresholds_display = {
    "LightGBM": lgb_thresh,
    "XGBoost": xgb_thresh,
    "Random Forest": rf_thresh
}

# try:
#     selected_model_path = model_paths[selected_model]
#     model = joblib.load(selected_model_path)
#     importances, features_used = get_model_importance(model, feature_cols)

#     st.subheader(f"🎯 Feature importance pro {selected_model}")
#     if importances is None:
#         st.warning("Feature importance není dostupná pro vybraný model.")
#     else:
#         fi_df = pd.DataFrame({
#             "Feature": features_used,
#             "Importance": importances
#         }).sort_values(by="Importance", ascending=False)

#         st.dataframe(fi_df.head(20))

#         st.subheader("🔢 Top 20 vlivných proměnných")
#         import matplotlib.pyplot as plt
#         fig, ax = plt.subplots(figsize=(10, 6))
#         top_fi = fi_df.head(20)
#         ax.barh(top_fi["Feature"], top_fi["Importance"])
#         ax.invert_yaxis()
#         ax.set_xlabel("Důležitost")
#         ax.set_title(f"{selected_model} – Feature Importance")
#         st.pyplot(fig)

#     st.info(f"Použitý threshold pro {selected_model}: **{thresholds_display[selected_model]:.2f}**")

# except Exception as e:
#     st.error(f"❌ Chyba při načítání modelu nebo feature importance: {e}")