import streamlit as st
from features_list import feature_cols
import pandas as pd
import numpy as np
from scipy.special import softmax
import joblib
import json
import os
import importlib
import sys
from datetime import datetime
from utils.feature_engineering_extended import generate_features
from utils.data_loader import load_data_by_league, filter_team_matches, filter_h2h_matches
from utils.feature_engineering_match_result import generate_match_result_features

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

league_code = st.selectbox("Zvol ligu:", ["E0","E1", "SP1", "D1","D2", "I1", "F1","B1","P1","T1","N1"])

df_raw = load_data_by_league(league_code)
df_raw["HomeTeam"] = df_raw["HomeTeam"].astype(str).str.strip()
df_raw["AwayTeam"] = df_raw["AwayTeam"].astype(str).str.strip()
df_raw = df_raw[(df_raw["HomeTeam"] != "nan") & (df_raw["AwayTeam"] != "nan")]
df_raw = df_raw[(df_raw["HomeTeam"] != "") & (df_raw["AwayTeam"] != "")]

invalid_rows = df_raw[df_raw["HomeTeam"].isna() | df_raw["AwayTeam"].isna()]
if not invalid_rows.empty:
    st.warning("⚠️ Chybné nebo prázdné názvy týmů byly detekovány a vynechány.")
    st.dataframe(invalid_rows)

teams = sorted(set(df_raw["HomeTeam"]).union(set(df_raw["AwayTeam"])))
home_team = st.selectbox("Domácí tým:", teams)
away_team = st.selectbox("Hostující tým:", teams)

thresholds_path = f"models/{league_code}_thresholds.json"
if os.path.exists(thresholds_path):
    with open(thresholds_path, "r") as f:
        thresholds = json.load(f)
        rf_thresh = thresholds.get("rf_best_threshold", 0.5)
        catboost_thresh = thresholds.get("catboost_best_threshold", 0.5)
else:
    rf_thresh = catboost_thresh = 0.5

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
            (df_ext["AwayTeam"] == away_team)
        ].sort_values("Date").tail(1)

        if match_row.empty:
            st.warning("⚠️ Nepodařilo se najít vstupní data pro predikci.")
            st.stop()

        model_path_rf = f"models/{league_code}_rf_model.joblib"
        model_path_catboost = f"models/{league_code}_catboost_model.joblib"
        rf_model = joblib.load(model_path_rf)
        catboost_model = joblib.load(model_path_catboost)

        for col in feature_cols:
            if col not in match_row.columns:
                match_row[col] = df_ext[col].mean() if col in df_ext.columns else 0

        X_input = match_row[feature_cols].fillna(0)
        if 'match_weight' in X_input.columns:
            X_input = X_input.drop(columns=['match_weight'])

        rf_prob = rf_model.predict_proba(X_input)[0][1]
        catboost_prob = catboost_model.predict_proba(X_input)[0][1]

        rf_pred = rf_prob >= rf_thresh
        catboost_pred = catboost_prob >= catboost_thresh

        data = {
            "Model": ["Random Forest", "CatBoost"],
            "Under 2.5": [
                f"{(1 - rf_prob)*100:.2f}% - {1 / (1 - rf_prob):.2f}",
                f"{(1 - catboost_prob)*100:.2f}% - {1 / (1 - catboost_prob):.2f}"
            ],
            "Over 2.5": [
                f"{rf_prob*100:.2f}% - {1 / rf_prob:.2f}",
                f"{catboost_prob*100:.2f}% - {1 / catboost_prob:.2f}"
            ]
        }

        df_predictions = pd.DataFrame(data)
        st.subheader(f"📊 Predikce: **{home_team} vs {away_team}**")
        st.write(df_predictions.style.hide(axis="index"))

        # === PŘEDPOVĚĎ TÝMOVÝCH STATISTIK ===
        st.subheader("📈 Očekávané týmové statistiky")

        from utils.feature_engineering_team_stats import generate_team_stats_features

        df_stats_input = pd.concat([df_filtered, future_match], ignore_index=True)
        df_team_stats = generate_team_stats_features(df_stats_input, mode="predict")

        match_stats_row = df_team_stats[
            (df_team_stats["HomeTeam"] == home_team) &
            (df_team_stats["AwayTeam"] == away_team)
        ].sort_values("Date").tail(1)
        #st.write("📋 match_stats_row:", match_stats_row)

        if not match_stats_row.empty:
            model_dir = f"models/{league_code}_team_stats"
            stat_targets = [
                "target_goals_home", "target_goals_away",
                "target_shots_home", "target_shots_away",
                "target_shots_on_home", "target_shots_on_away",
                "target_corners_home", "target_corners_away",
                "target_fouls_home", "target_fouls_away",
                "target_yellows_home", "target_yellows_away",
                "target_reds_home", "target_reds_away"
            ]

            stats_results = []
            for target in stat_targets:
                model_path = os.path.join(model_dir, f"{league_code}_{target}_model.joblib")
                if os.path.exists(model_path):
                    #st.write("✅ Model found:", model_path)
                    model = joblib.load(model_path)
                    drop_cols = [col for col in stat_targets + ["HomeTeam", "AwayTeam", "Date"] if col in match_stats_row.columns]
                    X_input = match_stats_row.drop(columns=drop_cols)

                    pred = model.predict(X_input.fillna(0))[0]
                    stats_results.append((target, pred))
                    #st.write(f"🔢 Predikce pro {target}: {pred:.2f}")

            # Seskupíme a zobrazíme jako porovnání
            label_map = {
                "goals": "⚽ Góly", 
                "shots": "🎯 Střely", 
                "shots_on": "🎯🎯 Střely na bránu",
                "corners": "🚩 Rohy", 
                "fouls": "💥 Fauly", 
                "yellows": "🟨 Žluté karty", 
                "reds": "🟥 Červené karty"
            }

            # Zobrazíme hlavičku zápasu
            st.markdown(f"<h3 style='text-align:center'>{home_team} vs {away_team}</h3>", unsafe_allow_html=True)


            for key in ["goals", "shots", "shots_on", "corners", "fouls", "yellows", "reds"]:
                h_key = f"target_{key}_home"
                a_key = f"target_{key}_away"
                h_val = next((v for k, v in stats_results if k == h_key), None)
                a_val = next((v for k, v in stats_results if k == a_key), None)

                if h_val is not None and a_val is not None:
                    max_val = max(h_val, a_val, 1)
                    h_bar = int((h_val / max_val) * 25)
                    a_bar = int((a_val / max_val) * 25)

                    row = st.columns([5, 6, 6, 6, 5])
                    # Ztučníme vyšší hodnotu
                    
                    row[0].markdown(f"<div style='text-align:right'>{h_val:.1f}</div>", unsafe_allow_html=True)
                    row[1].markdown(
                        f"<div style='background-color:#1f77b4;width:{h_bar*4}px;height:16px;border-radius:4px;margin-left:auto'></div>",
                        unsafe_allow_html=True
                    )
                    row[2].markdown(f"<center><strong>{label_map[key]}</strong></center>", unsafe_allow_html=True)
                    row[3].markdown(
                        f"<div style='background-color:#d62728;width:{a_bar*4}px;height:16px;border-radius:4px'></div>",
                        unsafe_allow_html=True
                    )
                    row[4].markdown(f"<div style='text-align:left'>{a_val:.1f}</div>", unsafe_allow_html=True)
                    
            # === VÝSLEDEK ZÁPASU ===
        df_result = generate_match_result_features(df_pred_input, mode="predict")
        result_row = df_result[
            (df_result["HomeTeam"] == home_team) &
            (df_result["AwayTeam"] == away_team) &
            (df_result["Date"].dt.date == datetime.today().date())
        ]

        if result_row.empty:
            st.warning("⚠️ Výsledek zápasu: Nenalezeny vstupní featury pro predikci.")
        else:
            result_input = result_row.drop(columns=["HomeTeam", "AwayTeam", "Date", "target_result"], errors="ignore").fillna(0)
            result_model_path = f"models/{league_code}_result_model.joblib"
            result_model = joblib.load(result_model_path)
            # Načtení teploty pro danou ligu
            temperature_path = f"models/{league_code}_temperature.json"
            if os.path.exists(temperature_path):
                with open(temperature_path, "r") as f:
                    T = json.load(f)["temperature"]
            else:
                T = 1.0  # fallback hodnota

            # Kalibrace pomocí temperature scaling
            raw_probs = result_model.predict_proba(result_input)[0]
            logits = np.log(raw_probs + 1e-15)
            result_probs = softmax(logits / T)
            result_labels = ["🏠 Výhra domácích", "🤝 Remíza", "🛫 Výhra hostů"]

            st.subheader("📈 Predikce výsledku zápasu (1X2):")
            for i, label in enumerate(result_labels):
                st.markdown(f"**{label}:** {result_probs[i]:.2%} ({1 / result_probs[i]:.2f} odds)")



    except Exception as e:
        st.error(f"Chyba při predikci: {e}")



        
