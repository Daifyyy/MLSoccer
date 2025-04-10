from features_list import feature_cols
import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import optuna
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
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
    #print(df_train_ext[["HomeTeam", "AwayTeam", "Date", "h2h_avg_goals_total", "h2h_over25_ratio"]].tail(10))

    X_train = df_train_ext[feature_cols].fillna(0)
    X_test = df_test_ext[feature_cols].fillna(0)

    y_train = df_train_ext["target_over25"]
    y_test = df_test_ext["target_over25"]

    w_train = df_train_ext["match_weight"]
    w_test = df_train_ext["match_weight"]

    # === Feature selection pomocí LightGBM ===
    selector_model = LGBMClassifier(n_estimators=100, random_state=42)
    selector_model.fit(X_train, y_train)
    selector = SelectFromModel(selector_model, threshold="median", prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    selected_features = X_train.columns[selector.get_support()] if hasattr(X_train, 'columns') else [f"f{i}" for i in range(X_train_selected.shape[1])]
    print(f"📉 Po výběru feature shape trénovacích dat: {X_train_selected.shape}")
    print(f"📋 Vybrané feature: {selected_features}")



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
            class_weight='balanced',
            random_state=42
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[early_stopping(20)],
        )
        preds = model.predict(X_val)
        return f1_score(y_val, preds, average="macro")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    return study.best_params

def plot_probability_distribution(model, X_test, label):
    probs = model.predict_proba(X_test)[:, 1]
    # plt.figure(figsize=(8, 4))
    # sns.histplot(probs, bins=20, kde=True, color='skyblue')
    # plt.title(f"Distribuce predikovaných pravděpodobností – {label}")
    # plt.xlabel("Pravděpodobnost OVER 2.5 gólů")
    # plt.ylabel("Počet zápasů")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

def plot_roc_and_pr_curves(model, X_test, y_test, label):
    probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, probs)
    fpr, tpr, _ = roc_curve(y_test, probs)

    # plt.figure(figsize=(12, 5))

    # plt.subplot(1, 2, 1)
    # plt.plot(fpr, tpr, marker='.')
    # plt.title(f'ROC křivka – {label}')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.grid(True)

    # plt.subplot(1, 2, 2)
    # plt.plot(recall, precision, marker='.')
    # plt.title(f'Precision-Recall křivka – {label}')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()

def optimize_threshold(model, X_test, y_test):
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_thresh, best_f1 = 0.55, 0
    f1_scores = []
    probs = model.predict_proba(X_test)[:, 1]
    for t in thresholds:
        preds = (probs >= t).astype(int)
        macro_f1 = f1_score(y_test, preds, average='macro')
        f1_scores.append(macro_f1)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_thresh = t

    # plt.plot(thresholds, f1_scores)
    # plt.title("Optimalizace thresholdu podle F1")
    # plt.xlabel("Threshold")
    # plt.ylabel("F1 score")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    return best_thresh

def train_and_save_models(league_code):
    print(f"\U0001F501 Trénuji modely pro ligu {league_code}...")
    X_train, X_test, y_train, y_test, w_train, w_test = prepare_data(league_code)

    best_params = optimize_model(X_train, y_train, X_test, y_test)
    print(f"✅ Nejlepší LightGBM parametry: {best_params}")

    lgb_model = LGBMClassifier(**best_params, class_weight='balanced', random_state=42)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[early_stopping(20), lgb.log_evaluation(0)],
    )
    os.makedirs("models", exist_ok=True)
    joblib.dump(lgb_model, f"models/{league_code}_lgb_model.joblib")
    print(f"✅ Model uložen se {len(X_train.columns)} featurami.")
    #print(list(X_train.columns))
    rf = RandomForestClassifier(max_depth=4, n_estimators=300, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train, sample_weight=w_train)

    xgb = XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.08, eval_metric='logloss')
    xgb.fit(X_train, y_train, sample_weight=w_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, f"models/{league_code}_rf_model.joblib")
    joblib.dump(xgb, f"models/{league_code}_xgb_model.joblib")

    print("✅ Modely uloženy.")
    return lgb_model, rf, xgb, X_test, y_test

def evaluate(model, X_test, y_test, label, threshold=0.5):
    print(f"\n📊 Výsledky pro {label} (cutoff = {threshold}):")
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    print(classification_report(y_test, preds))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))
    auc_score = roc_auc_score(y_test, probs)
    print(f"ROC AUC: {auc_score:.3f}")
    print("-" * 40)
    plot_probability_distribution(model, X_test, label)
    plot_roc_and_pr_curves(model, X_test, y_test, label)

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

    evaluate(rf_model, X_test, y_test, "Random Forest", threshold=thresholds["rf_best_threshold"])
    evaluate(xgb_model, X_test, y_test, "XGBoost", threshold=thresholds["xgb_best_threshold"])
    evaluate(lgb_model, X_test, y_test, "LightGBM", threshold=thresholds["lgb_best_threshold"])

    print("✅ Vše dokončeno.")
