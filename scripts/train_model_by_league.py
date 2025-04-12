from features_list import feature_cols
import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import joblib
import json
import optuna
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
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

def optimize_model(X_train, y_train, X_val, y_val):
    def objective(trial):
        model = LGBMClassifier(
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            num_leaves=trial.suggest_int("num_leaves", 16, 128),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 10, 40),
            feature_fraction=trial.suggest_float("feature_fraction", 0.1, 1),
            class_weight='balanced', random_state=42
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc", callbacks=[early_stopping(20)])
        return f1_score(y_val, model.predict(X_val), average="macro")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    return study.best_params

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
        best_params = optimize_model(X_train, y_train, X_test, y_test)

        lgb_model = LGBMClassifier(**best_params, class_weight='balanced', random_state=42)
        lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="auc", callbacks=[early_stopping(20)])

        rf = RandomForestClassifier(max_depth=4, n_estimators=300, class_weight="balanced", random_state=42)
        rf.fit(X_train, y_train, sample_weight=w_train)

        xgb = XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.08, eval_metric='logloss')
        xgb.fit(X_train, y_train, sample_weight=w_train)

        os.makedirs("models", exist_ok=True)
        joblib.dump(lgb_model, f"models/{league_code}_lgb_model.joblib")
        joblib.dump(rf, f"models/{league_code}_rf_model.joblib")
        joblib.dump(xgb, f"models/{league_code}_xgb_model.joblib")
        print(f"✅ Modely pro ligu {league_code} byly úspěšně uloženy.")
        return lgb_model, rf, xgb, X_test, y_test
    except Exception as e:
        print(f"❌ Chyba při tréninku modelů pro ligu {league_code}: {e}")
        return None, None, None, None, None

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
    for league_code in league_list:
        lgb_model, rf_model, xgb_model, X_test, y_test = train_and_save_models(league_code)
        if lgb_model is None:
            continue
        thresholds = {
            "rf_best_threshold": optimize_threshold(rf_model, X_test, y_test),
            "xgb_best_threshold": optimize_threshold(xgb_model, X_test, y_test),
            "lgb_best_threshold": optimize_threshold(lgb_model, X_test, y_test)
        }
        with open(f"models/{league_code}_thresholds.json", "w") as f:
            json.dump(thresholds, f, indent=4)
        evaluate(rf_model, X_test, y_test, f"{league_code} – Random Forest", thresholds["rf_best_threshold"])
        evaluate(xgb_model, X_test, y_test, f"{league_code} – XGBoost", thresholds["xgb_best_threshold"])
        evaluate(lgb_model, X_test, y_test, f"{league_code} – LightGBM", thresholds["lgb_best_threshold"])
    print("\n✅ Všechny ligy dokončeny.")
