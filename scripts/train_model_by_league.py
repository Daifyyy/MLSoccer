from features_list import feature_cols
import pandas as pd
import numpy as np
import os
import joblib
import json
import optuna
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_auc_score, brier_score_loss
from utils.data_loader import load_data_by_league
from utils.feature_engineering_extended import generate_features

def prepare_data(league_code):
    df = load_data_by_league(league_code).sort_values("Date")
    split_index = int(len(df) * 0.8)
    df_train, df_test = df.iloc[:split_index], df.iloc[split_index:]
    df_train_ext = generate_features(df_train, mode="train")
    df_test_ext = generate_features(df_test, mode="train")
    X_train, y_train, w_train = df_train_ext[feature_cols].fillna(0), df_train_ext["target_over25"], df_train_ext["match_weight"]
    X_test, y_test, w_test = df_test_ext[feature_cols].fillna(0), df_test_ext["target_over25"], df_test_ext["match_weight"]
    return X_train, X_test, y_train, y_test, w_train, w_test

def optimize_threshold(model, X_test, y_test):
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_thresh, best_f1 = 0.55, 0
    probs = model.predict_proba(X_test)[:, 1]
    for t in thresholds:
        preds = (probs >= t).astype(int)
        macro_f1 = f1_score(y_test, preds, average='macro')
        if macro_f1 > best_f1:
            best_f1, best_thresh = macro_f1, t
    return best_thresh

def train_and_save_models(league_code):
    print(f"\n🔁 Trénuji modely pro ligu {league_code}...")
    try:
        X_train, X_test, y_train, y_test, w_train, w_test = prepare_data(league_code)

        catboost_model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.08,
            loss_function="Logloss",
            random_seed=42,
            verbose=0
        )
        catboost_model.fit(X_train, y_train, sample_weight=w_train)

        rf = RandomForestClassifier(max_depth=4, n_estimators=300, class_weight="balanced", random_state=42)
        rf.fit(X_train, y_train, sample_weight=w_train)

        # Platt scaling - CatBoost
        probs_cat = catboost_model.predict_proba(X_test)[:, 1]
        platt_cat = LogisticRegression(max_iter=1000)
        platt_cat.fit(probs_cat.reshape(-1, 1), y_test)
        calibrated_cat = platt_cat.predict_proba(probs_cat.reshape(-1, 1))[:, 1]
        brier_cat = brier_score_loss(y_test, calibrated_cat)
        print(f"🧪 CatBoost Brier score (Platt): {brier_cat:.4f}")

        # Platt scaling - Random Forest
        probs_rf = rf.predict_proba(X_test)[:, 1]
        platt_rf = LogisticRegression(max_iter=1000)
        platt_rf.fit(probs_rf.reshape(-1, 1), y_test)
        calibrated_rf = platt_rf.predict_proba(probs_rf.reshape(-1, 1))[:, 1]
        brier_rf = brier_score_loss(y_test, calibrated_rf)
        print(f"🧪 Random Forest Brier score (Platt): {brier_rf:.4f}")

        os.makedirs("models", exist_ok=True)
        joblib.dump(catboost_model, f"models/{league_code}_catboost_model.joblib")
        joblib.dump(rf, f"models/{league_code}_rf_model.joblib")
        joblib.dump(platt_cat, f"models/{league_code}_catboost_platt.joblib")
        joblib.dump(platt_rf, f"models/{league_code}_rf_platt.joblib")
        print(f"✅ Modely pro ligu {league_code} byly úspěšně uloženy.")
        return catboost_model, rf, X_test, y_test
    except Exception as e:
        print(f"❌ Chyba při tréninku modelů pro ligu {league_code}: {e}")
        return None, None, None, None

def evaluate(model, X_test, y_test, label, threshold=0.5):
    if model is None: return
    print(f"\n📊 Výsledky pro {label} (cutoff = {threshold}):")
    preds = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    print(classification_report(y_test, preds))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))
    print(f"ROC AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.3f}")

if __name__ == "__main__":
    league_list = ["E0", "E1", "SP1", "D1", "D2", "I1", "F1", "B1", "P1", "T1", "N1"]
    #league_list = ["E0"]
    for league_code in league_list:
        catboost_model, rf_model, X_test, y_test = train_and_save_models(league_code)
        if catboost_model is None:
            continue
        thresholds = {
            "rf_best_threshold": optimize_threshold(rf_model, X_test, y_test),
            "catboost_best_threshold": optimize_threshold(catboost_model, X_test, y_test)
        }
        with open(f"models/{league_code}_thresholds.json", "w") as f:
            json.dump(thresholds, f, indent=4)
        evaluate(rf_model, X_test, y_test, f"{league_code} – Random Forest", thresholds["rf_best_threshold"])
        evaluate(catboost_model, X_test, y_test, f"{league_code} – CatBoost", thresholds["catboost_best_threshold"])
    print("\n✅ Všechny ligy dokončeny.")
