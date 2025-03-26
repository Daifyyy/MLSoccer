import streamlit as st
import pandas as pd
import joblib
import os
from utils.data_loader import load_data, get_teams, filter_team_matches, filter_h2h_matches
from utils.feature_engineering_extended import generate_extended_features
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Fotbalová predikce: Over 2.5 gólu", page_icon="\U0001F3C0")
st.title("\U0001F3C0 Fotbalová predikce: Over 2.5 gólu")

status = st.empty()
status.success("Aplikace byla úspěšně načtena.")

# Výběr ligy
league = st.selectbox("Vyber ligu", ["E0", "SP1"])

# Cesty k souborům
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", f"{league}_combined_full.csv")
model_path = os.path.join(BASE_DIR, "models", f"{league}_rf_model.joblib")

# Načtení dat a modelu
status.info("Načítám data a model...")
try:
    data = pd.read_csv(data_path)
    model = joblib.load(model_path)
    status.success("Data načtena")
    status.success("Model načten")
except Exception as e:
    status.error(f"Chyba při načítání: {e}")
    st.stop()

teams = get_teams(data)
home_team = st.selectbox("Vyber domácí tým", teams)
away_team = st.selectbox("Vyber hostující tým", [t for t in teams if t != home_team])

if st.button("\U0001F4AB Predikovat"):
    # Připrav vstupní data
    team_data = filter_h2h_matches(data, home_team, away_team)
    if team_data.empty or len(team_data) < 1:
        st.error("Nedostatek historických dat pro vybrané týmy.")
    else:
        team_data = generate_extended_features(team_data)
        latest = team_data.tail(1)
        features = model.feature_names_in_
        X_input = latest[features]

        prediction = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0][1] * 100

        st.subheader("\U0001F4CA Výsledek predikce")
        st.write(f"**Pravděpodobnost OVER 2.5:** {proba:.2f}%")
        if prediction == 1:
            st.success("\U00002705 Model predikuje: OVER 2.5 gólu")
        else:
            st.error("\U0000274C Model predikuje: UNDER 2.5 gólu")