import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "over25_model.joblib")

def load_model(model_path):
    return joblib.load(model_path)


def predict_over25(model, features):
    X_input = pd.DataFrame([features.values], columns=features.index)
    probability = model.predict_proba(X_input)[0][1]
    prediction = model.predict(X_input)[0]
    return probability, prediction

def prepare_features_for_prediction(df_ext, home_team, away_team):
    row = df_ext[(df_ext["HomeTeam"] == home_team) & (df_ext["AwayTeam"] == away_team)]

    if row.empty:
        raise ValueError("Zápas mezi zadanými týmy nebyl nalezen v datech.")

    # Seznam relevantních sloupců, které používáme při trénování
    features = [
        col for col in df_ext.columns
        if col.endswith("_form") or col.endswith("_diff")
        or col.endswith("_last5") or col.startswith("elo_rating")
        or col.endswith("_fouls") or col.endswith("_cards")
        or col.startswith("over25")
    ]

    return row[features].iloc[0].fillna(0)
