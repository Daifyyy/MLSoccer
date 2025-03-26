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
