import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_loader import load_data_by_league
from utils.feature_engineering_extended import generate_extended_features

# === Vstup od uživatele ===
league_code = input("Zadej zkratku ligy (např. E0 nebo SP1): ")

# === Načtení surových dat ===
df = load_data_by_league(league_code)

# === Rozdělení dat na train/test před feature engineeringem ===
df = df.sort_values("Date")
split_index = int(len(df) * 0.8)
df_train = df.iloc[:split_index]
df_test = df.iloc[split_index:]

# === Vygenerování featur ===
df_train_ext = generate_extended_features(df_train, mode="train")
df_test_ext = generate_extended_features(df_test, mode="train")

features = [
    "shooting_efficiency",
    "elo_rating_home",
    "elo_rating_away",
    "momentum_score",
    "home_xg",
    "away_xg",
    "xg_home_last5",
    "xg_away_last5",
    "corner_diff_last5",
    "shot_on_target_diff_last5",
    "shot_diff_last5m",
    "fouls_diff",
    "card_diff",
    "boring_match_score",
    "match_weight",
    "h2h_goal_avg",
    "defensive_stability",
    "tempo_score",
    "passivity_score",
    "home_under25_last5",
    "away_under25_last5",
    "home_avg_goals_last5_home",
    "away_avg_goals_last5_away"
]

X_train = df_train_ext[features].fillna(0)
y_train = df_train_ext["Over_2.5"]
w_train = df_train_ext["match_weight"].fillna(1.0)

X_test = df_test_ext[features].fillna(0)
y_test = df_test_ext["Over_2.5"]
w_test = df_test_ext["match_weight"].fillna(1.0)

# === Ruční ladění Random Forest ===
best_rf_model = None
best_rf_score = 0
best_rf_params = {}

print("\n🔍 Ladění Random Forest")
for depth in [5, 10]:
    for estimators in [100, 200]:
        rf = RandomForestClassifier(max_depth=depth, n_estimators=estimators, random_state=42)
        rf.fit(X_train, y_train, sample_weight=w_train)
        preds = rf.predict(X_test)
        score = f1_score(y_test, preds)
        print(f"RF: depth={depth}, estimators={estimators} → F1: {score:.4f}")

        if score > best_rf_score:
            best_rf_model = rf
            best_rf_score = score
            best_rf_params = {"max_depth": depth, "n_estimators": estimators}

print("\n✅ Nejlepší RF parametry:", best_rf_params)
print("Random Forest – výstup na testovací sadě:")
print(classification_report(y_test, best_rf_model.predict(X_test)))
print("Confusion matrix (RF):")
print(confusion_matrix(y_test, best_rf_model.predict(X_test)))

# === Ruční ladění XGBoost ===
best_xgb_model = None
best_xgb_score = 0
best_xgb_params = {}

print("\n🔍 Ladění XGBoost")
for depth in [3, 6]:
    for lr in [0.05, 0.1]:
        for estimators in [100, 200]:
            xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=depth,
                                learning_rate=lr, n_estimators=estimators, random_state=42)
            xgb.fit(X_train, y_train, sample_weight=w_train)
            preds = xgb.predict(X_test)
            score = f1_score(y_test, preds)
            print(f"XGB: depth={depth}, lr={lr}, estimators={estimators} → F1: {score:.4f}")

            if score > best_xgb_score:
                best_xgb_model = xgb
                best_xgb_score = score
                best_xgb_params = {"max_depth": depth, "learning_rate": lr, "n_estimators": estimators}

print("\n✅ Nejlepší XGBoost parametry:", best_xgb_params)
print("XGBoost – výstup na testovací sadě:")
print(classification_report(y_test, best_xgb_model.predict(X_test)))
print("Confusion matrix (XGB):")
print(confusion_matrix(y_test, best_xgb_model.predict(X_test)))

# === Uložení modelů ===
os.makedirs("models", exist_ok=True)
joblib.dump(best_rf_model, f"models/{league_code}_rf_model.joblib")
joblib.dump(best_xgb_model, f"models/{league_code}_xgb_model.joblib")
print("\n✅ Nejlepší modely byly uloženy.")
