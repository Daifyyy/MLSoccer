import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

from utils.data_loader import load_data_by_league
from utils.feature_engineering_extended import generate_extended_features

# === Vstup od uživatele ===
league_code = input("Zadej zkratku ligy (např. E0 nebo SP1): ")

# === Načtení a rozdělení dat ===
df = load_data_by_league(league_code)
df = df.sort_values("Date")
split_index = int(len(df) * 0.8)
df_train = df.iloc[:split_index]
df_test = df.iloc[split_index:]

# === Generování featur ===
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
]

X_train = df_train_ext[features].fillna(0)
y_train = df_train_ext["Over_2.5"]

# === GridSearchCV pro Random Forest ===
rf = RandomForestClassifier(random_state=42)
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}

print("\n🔍 Ladění Random Forest...")
rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='f1', verbose=1, n_jobs=-1)
rf_grid.fit(X_train, y_train)
print("Nejlepší parametry RF:", rf_grid.best_params_)
print("Výsledek:")
print(classification_report(y_train, rf_grid.predict(X_train)))

# === GridSearchCV pro XGBoost ===
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 6, 10],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
}

print("\n🔍 Ladění XGBoost...")
xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, scoring='f1', verbose=1, n_jobs=-1)
xgb_grid.fit(X_train, y_train)
print("Nejlepší parametry XGBoost:", xgb_grid.best_params_)
print("Výsledek:")
print(classification_report(y_train, xgb_grid.predict(X_train)))

# === Uložení modelů ===
joblib.dump(rf_grid.best_estimator_, f"models/{league_code}_rf_model.joblib")
joblib.dump(xgb_grid.best_estimator_, f"models/{league_code}_xgb_model.joblib")
print("\n✅ Modely uloženy.")
