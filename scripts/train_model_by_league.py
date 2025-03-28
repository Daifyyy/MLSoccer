import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
from utils.feature_engineering_extended import generate_extended_features
import os

# === Vstup od uzivatele ===
LEAGUE = input("Zadej zkratku ligy (např. E0 nebo SP1): ")

data_path = f"data/{LEAGUE}_combined_full.csv"
df = pd.read_csv(data_path)
df_ext = generate_extended_features(df)

# === Vyber feature ===
features = [
    col for col in df_ext.columns
    if col.endswith("_form") or col.endswith("_diff")
    or col.endswith("_last5") or col.startswith("elo_rating")
    or col.endswith("_fouls") or col.endswith("_cards")
    or col.startswith("over25")
]

X = df_ext[features].fillna(0)
y = df_ext["Over_2.5"]
sample_weights = df_ext["match_weight"].fillna(1.0)

# === Rozdeleni dat ===
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=42
)

# === Random Forest ===
print("\n🔧 Trénuji Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)
rf_model.fit(X_train, y_train, sample_weight=w_train)

model_path = f"models/{LEAGUE}_rf_model.joblib"
joblib.dump(rf_model, model_path)
print(f"\n📂 Random Forest model pro ligu {LEAGUE} uložen do {model_path}")

# === XGBoost ===
print("\n🔧 Trénuji XGBoost model...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = XGBClassifier(
    max_depth=6,
    n_estimators=300,
    learning_rate=0.1,
    subsample=0.6,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train, sample_weight=w_train)

model_path_xgb = f"models/{LEAGUE}_xgb_model.joblib"
joblib.dump(xgb_model, model_path_xgb)
print(f"\n📂 XGBoost model pro ligu {LEAGUE} uložen do {model_path_xgb}")