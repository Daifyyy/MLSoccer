import pandas as pd
import joblib
import os
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from utils.data_loader import load_data_by_league
from utils.feature_engineering_match_result import generate_match_result_features
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train_match_result_model(league_code):
    print(f"\U0001F3C6 Trénink modelu pro predikci výsledku zápasu ({league_code})")
    df = load_data_by_league(league_code)

    df_train = df.iloc[:-int(len(df) * 0.2)]
    df_test = df.iloc[-int(len(df) * 0.2):]

    df_train_fe = generate_match_result_features(df_train, mode="train")
    df_test_fe = generate_match_result_features(df_test, mode="train")

    X_train = df_train_fe.drop(columns=["HomeTeam", "AwayTeam", "Date", "target_result"])
    y_train = df_train_fe["target_result"]
    X_test = df_test_fe.drop(columns=["HomeTeam", "AwayTeam", "Date", "target_result"])
    y_test = df_test_fe["target_result"]

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    sample_weights = y_train.map(class_weights_dict)

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.08,
        loss_function="MultiClass",
        random_seed=42,
        verbose=0
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)

    # === Platt scaling (multinomial logistic regression on validation set) ===
    probs_val = model.predict_proba(X_test)
    platt_model = LogisticRegression(max_iter=1000, multi_class='multinomial')
    platt_model.fit(probs_val, y_test)

    y_pred = model.predict(X_test)
    print("\n📊 Výsledky na testovací sadě:")
    print(classification_report(y_test, y_pred, target_names=["Výhra domácích", "Remíza", "Výhra hostů"]))
    print("\nMaticová chyba (confusion matrix):")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{league_code}_result_model.joblib")
    joblib.dump(platt_model, f"models/{league_code}_result_model_platt.joblib")
    print(f"\n✅ Model a Platt kalibrace uloženy pro ligu {league_code}")

if __name__ == "__main__":
    league_list = ["E0", "E1", "SP1", "D1", "D2", "I1", "F1", "B1", "P1", "T1", "N1"]
    for liga in league_list:
        train_match_result_model(liga)
    print("\n✅ Všechny modely výsledků zápasu dokončeny.")
