import os
import pandas as pd
import streamlit as st
import joblib

from utils.data_loader import load_data_by_league, get_teams_by_league, filter_team_matches, filter_h2h_matches, get_teams
from utils.feature_engineering_extended import generate_extended_features

st.title("⚽ Predikce zápasu – Over 2.5 goals")

# 📂 Výběr ligy
leagues = ["E0", "SP1", "D1", "I1", "F1", "N1"]
league_code = st.selectbox("Vyber ligu", leagues)

# 📊 Načtení dat a týmů
df_raw = load_data_by_league(league_code)
teams = get_teams(df_raw)

# ⚽ Výběr týmů
home_team = st.selectbox("Domácí tým", teams)
away_team = st.selectbox("Hostující tým", teams)

# 🔮 Spuštění predikce
if st.button("🔍 Spustit predikci"):
    try:
        # ➕ Filtrování dat pro zvolené týmy + H2H zápasy
        df_filtered = pd.concat([
            filter_team_matches(df_raw, home_team),
            filter_team_matches(df_raw, away_team),
            filter_h2h_matches(df_raw, home_team, away_team)
        ]).drop_duplicates().reset_index(drop=True)

        # ➕ Vygenerování rozšířených featur
        df_ext = generate_extended_features(df_filtered)

        # 🧠 Načtení modelů
        rf_model_path = f"models/{league_code}_rf_model.joblib"
        xgb_model_path = f"models/{league_code}_xgb_model.joblib"
        rf_model = joblib.load(rf_model_path)
        xgb_model = joblib.load(xgb_model_path)

        # 🧾 Výběr posledního známého zápasu
        latest_home = df_ext[df_ext["HomeTeam"] == home_team].iloc[-1:]
        latest_away = df_ext[df_ext["AwayTeam"] == away_team].iloc[-1:]
        if latest_home.empty or latest_away.empty:
            st.warning("Není dostatek dat pro tento zápas.")
        else:
            match_row = latest_home if latest_home["Date"].values[0] > latest_away["Date"].values[0] else latest_away

            # 🎯 Feature výběr – musí odpovídat tréninku!
            features = [
                col for col in df_ext.columns
                if col.endswith("_form") or col.endswith("_diff")
                or col.startswith("over25") or col.startswith("elo_rating")
                or col.endswith("_last5") or col.endswith("_weight")
                or col.endswith("_cards") or col.endswith("_fouls")
                or col.startswith("xg") or col.startswith("boring")
            ]

            # 🧪 Vstupní data pro model
            X_input = match_row[features].fillna(0)

            rf_pred = rf_model.predict_proba(X_input)[0][1]
            xgb_pred = xgb_model.predict_proba(X_input)[0][1]

            # 📈 Výstup
            st.subheader("📊 Výsledky predikce:")
            st.write(f"🎲 Random Forest – pravděpodobnost Over 2.5: **{rf_pred:.2%}**")
            st.write(f"🚀 XGBoost – pravděpodobnost Over 2.5: **{xgb_pred:.2%}**")

    except Exception as e:
        st.error(f"❌ Nastala chyba během predikce: {e}")
