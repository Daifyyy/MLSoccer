import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from utils.data_loader import load_data_by_league
from utils.feature_engineering_match_result import generate_match_result_features

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train_match_result_model(league_code):
    print(f"\U0001F3C6 Trénink modelu pro predikci výsledku zápasu ({league_code})")
    df = load_data_by_league(league_code)
    df_train = df.iloc[:-int(len(df)*0.2)]
    df_test = df.iloc[-int(len(df)*0.2):]

    df_train_fe = generate_match_result_features(df_train, mode="train")
    df_test_fe = generate_match_result_features(df_test, mode="train")

    X_train = df_train_fe.drop(columns=["HomeTeam", "AwayTeam", "Date", "target_result"])
    y_train = df_train_fe["target_result"]
    X_test = df_test_fe.drop(columns=["HomeTeam", "AwayTeam", "Date", "target_result"])
    y_test = df_test_fe["target_result"]

    # Výpočet váhování podle třídy (pro nevyvážené výsledky)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=6,
        n_estimators=300,
        learning_rate=0.08,
        eval_metric="mlogloss",
        random_state=42,
        scale_pos_weight=None,
        enable_categorical=False
    )

    model.fit(X_train, y_train, sample_weight=y_train.map(class_weights_dict))

    y_pred = model.predict(X_test)
    print("\n📊 Výsledky na testovací sadě:")
    print(classification_report(y_test, y_pred, target_names=["Výhra domácích", "Remíza", "Výhra hostů"]))
    print("\nMaticová chyba (confusion matrix):")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    model_path = f"models/{league_code}_result_model.joblib"
    joblib.dump(model, model_path)
    print(f"\n✅ Model uložen do {model_path}")

if __name__ == "__main__":
    liga = input("Zadej kód ligy (např. E0, SP1): ")
    train_match_result_model(liga)
