from features_list import feature_cols

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import optuna

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_squared_error, r2_score

from utils.data_loader import load_data_by_league
from utils.feature_engineering_extended import generate_features

def prepare_data(league_code):
    df = load_data_by_league(league_code)
    df = df.sort_values("Date")

    split_index = int(len(df) * 0.8)
    df_train = df.iloc[:split_index]
    df_test = df.iloc[split_index:]

    df_train_ext = generate_features(df_train, mode="train")
    df_test_ext = generate_features(df_test, mode="train")

    X_train = df_train_ext[feature_cols].fillna(0)
    X_test = df_test_ext[feature_cols].fillna(0)

    y_train = df_train["FTHG"] + df_train["FTAG"]
    y_test = df_test["FTHG"] + df_test["FTAG"]


    w_train = df_train_ext["match_weight"]
    w_test = df_test_ext["match_weight"]

    return X_train, X_test, y_train, y_test, w_train, w_test

def optimize_model(X_train, y_train, X_val, y_val):
    def objective(trial):
        model = LGBMRegressor(
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            num_leaves=trial.suggest_int("num_leaves", 16, 128),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 10, 30),
            feature_fraction=trial.suggest_float("feature_fraction", 0.3, 1.0),
            random_state=42
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[early_stopping(20)],
        )
        preds = model.predict(X_val)
        return -mean_squared_error(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    return study.best_params

def evaluate(model, X_test, y_test, label):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n📊 Hodnocení regrese pro {label}:")
    print(f"MSE: {mse:.3f}")
    print(f"R2 score: {r2:.3f}")

    plt.figure(figsize=(8, 4))
    sns.histplot(preds, bins=20, color='lightgreen', kde=True, label="Predikované")
    sns.histplot(y_test, bins=20, color='gray', kde=True, label="Skutečné", alpha=0.5)
    plt.legend()
    plt.title(f"Distribuce predikovaných vs. skutečných gólů ({label})")
    plt.xlabel("Počet gólů v zápase")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def train_and_save_models(league_code):
    print(f"🔁 Trénuji regresní modely pro ligu {league_code}...")
    X_train, X_test, y_train, y_test, w_train, w_test = prepare_data(league_code)

    best_params = optimize_model(X_train, y_train, X_test, y_test)
    print(f"✅ Nejlepší LightGBM parametry: {best_params}")

    lgb_model = LGBMRegressor(**best_params, random_state=42)
    lgb_model.fit(X_train, y_train, sample_weight=w_train)
    joblib.dump(lgb_model, f"models/{league_code}_lgb_reg.joblib")

    rf = RandomForestRegressor(max_depth=6, n_estimators=300, random_state=42)
    rf.fit(X_train, y_train, sample_weight=w_train)
    joblib.dump(rf, f"models/{league_code}_rf_reg.joblib")

    xgb = XGBRegressor(max_depth=4, n_estimators=200, learning_rate=0.08, eval_metric='rmse')
    xgb.fit(X_train, y_train, sample_weight=w_train)
    joblib.dump(xgb, f"models/{league_code}_xgb_reg.joblib")

    print("✅ Regresní modely uloženy.")
    return lgb_model, rf, xgb, X_test, y_test

if __name__ == "__main__":
    league_code = input("Zadej zkratku ligy (např. E0 nebo SP1): ")
    lgb_model, rf_model, xgb_model, X_test, y_test = train_and_save_models(league_code)

    evaluate(rf_model, X_test, y_test, "Random Forest")
    evaluate(xgb_model, X_test, y_test, "XGBoost")
    evaluate(lgb_model, X_test, y_test, "LightGBM")

    print("✅ Vše dokončeno.")
