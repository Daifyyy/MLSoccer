from features_list import feature_cols

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import optuna

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import VarianceThreshold

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

    y_train = ((df_train["FTHG"] + df_train["FTAG"]) > 2.5).astype(int)
    y_test = ((df_test["FTHG"] + df_test["FTAG"]) > 2.5).astype(int)

    w_train = pd.Series([1.0] * len(df_train_ext))
    w_test = pd.Series([1.0] * len(df_test_ext))

    return X_train, X_test, y_train, y_test, w_train, w_test

def optimize_model(X_train, y_train, X_val, y_val):
    def objective(trial):
        model = LGBMClassifier(
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            n_estimators=trial.suggest_int("n_estimators", 100, 300),
            num_leaves=trial.suggest_int("num_leaves", 16, 64),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 30),
            feature_fraction=trial.suggest_float("feature_fraction", 0.7, 1.0),
            class_weight='balanced',
            random_state=42
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            early_stopping_rounds=20,
            verbose=False
        )
        preds = model.predict(X_val)
        return f1_score(y_val, preds, average="macro")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    return study.best_params

def plot_probability_distribution(model, X_test):
    probs = model.predict_proba(X_test)[:, 1]
    plt.figure(figsize=(8, 4))
    sns.histplot(probs, bins=20, kde=True, color='skyblue')
    plt.title("Distribuce predikovaných pravděpodobností (pro třídu OVER)")
    plt.xlabel("Pravděpodobnost OVER 2.5 gólů")
    plt.ylabel("Počet zápasů")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def train_and_save_models(league_code):
    print(f"🔁 Trénuji modely pro ligu {league_code}...")
    X_train, X_test, y_train, y_test, w_train, w_test = prepare_data(league_code)

    best_params = optimize_model(X_train, y_train, X_test, y_test)
    print(f"✅ Nejlepší LightGBM parametry: {best_params}")

    lgb_model = LGBMClassifier(**best_params, class_weight='balanced', random_state=42)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        early_stopping_rounds=20,
        verbose=False
    )
    joblib.dump(lgb_model, f"models/{league_code}_lgb_model.joblib")

    rf = RandomForestClassifier(max_depth=6, n_estimators=200, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train, sample_weight=w_train)

    xgb = XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.15, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train, sample_weight=w_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, f"models/{league_code}_rf_model.joblib")
    joblib.dump(xgb, f"models/{league_code}_xgb_model.joblib")

    print("✅ Modely uloženy.")
    return lgb_model, rf, xgb, X_test, y_test

def optimize_threshold(model, X_test, y_test):
    thresholds = np.arange(0.3, 0.7, 0.05)
    best_thresh, best_f1 = 0.5, 0
    probs = model.predict_proba(X_test)[:, 1]
    for t in thresholds:
        preds = (probs >= t).astype(int)
        macro_f1 = f1_score(y_test, preds, average='macro')
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_thresh = t
    return best_thresh

def evaluate(model, X_test, y_test, label):
    print(f"\n📊 Výsledky pro {label}:")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))
    probs = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, probs)
    print(f"ROC AUC: {auc_score:.3f}")
    print("-" * 40)
    plot_probability_distribution(model, X_test)

if __name__ == "__main__":
    league_code = input("Zadej zkratku ligy (např. E0 nebo SP1): ")
    lgb_model, rf_model, xgb_model, X_test, y_test = train_and_save_models(league_code)

    thresholds = {
        "rf_best_threshold": optimize_threshold(rf_model, X_test, y_test),
        "xgb_best_threshold": optimize_threshold(xgb_model, X_test, y_test),
        "lgb_best_threshold": optimize_threshold(lgb_model, X_test, y_test)
    }

    with open(f"models/{league_code}_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=4)

    evaluate(rf_model, X_test, y_test, "Random Forest")
    evaluate(xgb_model, X_test, y_test, "XGBoost")
    evaluate(lgb_model, X_test, y_test, "LightGBM")
    print("✅ Vše dokončeno.")
