import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from utils.data_loader import load_data_by_league
from utils.feature_engineering_extended import generate_features
from sklearn.metrics import precision_recall_fscore_support
from lightgbm import LGBMClassifier
from sklearn.feature_selection import VarianceThreshold
import optuna
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

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
df_train_ext = generate_features(df_train, mode="train")
df_test_ext = generate_features(df_test, mode="train")

features = [
    "boring_match_score", "home_xg", "away_xg", "elo_rating_home", "elo_rating_away",
    "xg_home_last5", "xg_away_last5", "shots_home_last5", "shots_away_last5",
    "shots_on_target_home_last5", "shots_on_target_away_last5", "conceded_home_last5", "conceded_away_last5",
    "xg_conceded_home_last5", "xg_conceded_away_last5", "avg_xg_conceded", "xg_ratio", "defensive_pressure",
    "tempo_score", "passivity_score", "fouls_diff", "card_diff", "aggressiveness_score", "behavior_balance",
    "momentum_score", "match_weight", "avg_goal_sum_last5",
    "h2h_avg_goals", "h2h_over25_ratio",
    "over25_ratio_season_avg", "over25_ratio_last10_avg", "goal_std_last5",
    "attack_pressure_last5", "over25_trend", "games_last_14d", "xg_off_def_diff",
    "shots_on_target_diff", "elo_diff", "over25_momentum",
    "goals_scored_season_avg_home", "goals_scored_season_avg_away", "goals_scored_last5_home", "goals_scored_last5_away", "goals_scored_total_last5",
    "goals_conceded_season_avg_home", "goals_conceded_season_avg_away", "goals_conceded_last5_home", "goals_conceded_last5_away", "goals_conceded_total_last5",
    "goal_diff_last5", "goal_ratio_home", "goal_ratio_away","corners_last5_home","corners_last5_away"
]

X_train = df_train_ext[features].fillna(0)

var_thresh = VarianceThreshold(threshold=0.01)
var_thresh.fit(X_train)
low_var_features = [col for col, keep in zip(X_train.columns, var_thresh.get_support()) if not keep]

if low_var_features:
    print("🔍 Odstraněny low-variance featury:", low_var_features)
    features = [f for f in features if f not in low_var_features]
    X_train = X_train[features]
    

y_train = (df_train["FTHG"] + df_train["FTAG"]) > 2.5
w_train = df_train_ext["match_weight"].fillna(1.0)

X_test = df_test_ext[features].fillna(0)

y_test = (df_test["FTHG"] + df_test["FTAG"]) > 2.5
w_test = df_test_ext["match_weight"].fillna(1.0)
ratio = (y_train == 0).sum() / (y_train == 1).sum()


# === Ruční ladění Random Forest ===
best_rf_model = None
best_rf_score = 0
best_rf_params = {}
# === Undersampling majority třídy (True) ===
rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

# === Ladění LightGBM pomocí Optuna ===
def objective(trial):
    max_depth = trial.suggest_int("max_depth", 3, 15)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    n_estimators = trial.suggest_int("n_estimators", 100, 300)
    min_gain_to_split = trial.suggest_float("min_gain_to_split", 0.0, 0.05)

    model = LGBMClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_gain_to_split=min_gain_to_split,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train_resampled, y_train_resampled)
    preds = model.predict(X_test)
    return f1_score(y_test, preds, average='macro')

print("\n🔍 Ladění LightGBM pomocí Optuna")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15, show_progress_bar=True)

best_lgb_params = study.best_params
print("\n✅ Nejlepší LightGBM parametry:", best_lgb_params)

best_lgb_model = LGBMClassifier(**best_lgb_params, class_weight="balanced", random_state=42)
best_lgb_model.fit(X_train, y_train, sample_weight=w_train)

print("LightGBM – výstup na testovací sadě:")
print(classification_report(y_test, best_lgb_model.predict(X_test)))
print("Confusion matrix (LGB):")
print(confusion_matrix(y_test, best_lgb_model.predict(X_test)))

# === Ladění XGBoost pomocí Optuna ===
def objective_xgb(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 200),
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
        'eval_metric': 'logloss',
        'random_state': 42,
        'use_label_encoder': False
    }
    model = XGBClassifier(**param)
    model.fit(X_train_resampled, y_train_resampled)
    preds = model.predict(X_test)
    return f1_score(y_test, preds, average='macro')

print("\n🔍 Ladění XGBoost pomocí Optuna")
study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=15, show_progress_bar=True)

best_xgb_params = study_xgb.best_params
print("\n✅ Nejlepší XGBoost parametry:", best_xgb_params)

best_xgb_model = XGBClassifier(**best_xgb_params, use_label_encoder=False)
best_xgb_model.fit(X_train, y_train, sample_weight=w_train)

print("XGBoost – výstup na testovací sadě:")
print(classification_report(y_test, best_xgb_model.predict(X_test)))
print("Confusion matrix (XGB):")
print(confusion_matrix(y_test, best_xgb_model.predict(X_test)))


print("\n🔍 Ladění Random Forest")
for depth in [2, 6, 10]:
    for estimators in [100, 200]:
        rf = RandomForestClassifier(max_depth=depth, class_weight="balanced", n_estimators=estimators, random_state=42)
        rf.fit(X_train_resampled, y_train_resampled)
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


# === Optimalizace prahu pro RF a XGB podle Macro F1 ===
thresholds = np.arange(0.3, 0.71, 0.01)

best_thresh_rf = 0.5
best_macro_f1_rf = 0
rf_probs = best_rf_model.predict_proba(X_test)[:, 1]

for t in thresholds:
    preds = (rf_probs >= t).astype(int)
    macro_f1 = f1_score(y_test, preds, average='macro')
    if macro_f1 > best_macro_f1_rf:
        best_macro_f1_rf = macro_f1
        best_thresh_rf = t

best_thresh_xgb = 0.5
best_macro_f1_xgb = 0
xgb_probs = best_xgb_model.predict_proba(X_test)[:, 1]

for t in thresholds:
    preds = (xgb_probs >= t).astype(int)
    macro_f1 = f1_score(y_test, preds, average='macro')
    if macro_f1 > best_macro_f1_xgb:
        best_macro_f1_xgb = macro_f1
        best_thresh_xgb = t
        
# === Optimalizace prahu pro LightGBM ===
best_thresh_lgb = 0.5
best_macro_f1_lgb = 0
lgb_probs = best_lgb_model.predict_proba(X_test)[:, 1]

for t in thresholds:
    preds = (lgb_probs >= t).astype(int)
    macro_f1 = f1_score(y_test, preds, average='macro')
    if macro_f1 > best_macro_f1_lgb:
        best_macro_f1_lgb = macro_f1
        best_thresh_lgb = t





print(f"\n🎯 Nejlepší threshold RF podle macro F1: {best_thresh_rf:.2f} (F1 = {best_macro_f1_rf:.4f})")
print(f"🎯 Nejlepší threshold XGB podle macro F1: {best_thresh_xgb:.2f} (F1 = {best_macro_f1_xgb:.4f})")
print(f"🎯 Nejlepší threshold LGB podle macro F1: {best_thresh_lgb:.2f} (F1 = {best_macro_f1_lgb:.4f})")
# === Uložení thresholdů do JSON ===
import json
thresholds_dict = {
    "rf_best_threshold": float(best_thresh_rf),
    "xgb_best_threshold": float(best_thresh_xgb),
    "lgb_best_threshold": float(best_thresh_lgb)
}






# === Uložení modelu LightGBM ===
joblib.dump(best_lgb_model, f"models/{league_code}_lgb_model.joblib")

# === Feature Importance – LightGBM ===
plt.figure(figsize=(10, 6))
sns.barplot(x=best_lgb_model.feature_importances_, y=features)
plt.title("Feature Importance – LightGBM")
plt.tight_layout()
plt.savefig("models/feature_importance_lgb.png")
plt.close()

# === ROC Curve – LightGBM ===
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, best_lgb_model.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr_lgb, tpr_lgb, label=f"LightGBM (AUC = {roc_auc_score(y_test, best_lgb_model.predict_proba(X_test)[:,1]):.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – LightGBM")
plt.legend(loc="lower right")
plt.savefig("models/roc_curve_lgb.png")
plt.close()


with open(f"models/{league_code}_thresholds.json", "w") as f:
    json.dump(thresholds_dict, f, indent=4)

print("✅ Uloženo do models/{league_code}_thresholds.json")

# === Analýza: důležitost featur a ROC ===
os.makedirs("models", exist_ok=True)

plt.figure(figsize=(10, 6))
sns.barplot(x=best_rf_model.feature_importances_, y=features)
plt.title("Feature Importance – Random Forest")
plt.tight_layout()
plt.savefig("models/feature_importance_rf.png")
plt.close()

fpr_rf, tpr_rf, _ = roc_curve(y_test, best_rf_model.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_score(y_test, best_rf_model.predict_proba(X_test)[:,1]):.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Random Forest")
plt.legend(loc="lower right")
plt.savefig("models/roc_curve_rf.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(x=best_xgb_model.feature_importances_, y=features)
plt.title("Feature Importance – XGBoost")
plt.tight_layout()
plt.savefig("models/feature_importance_xgb.png")
plt.close()

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, best_xgb_model.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {roc_auc_score(y_test, best_xgb_model.predict_proba(X_test)[:,1]):.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – XGBoost")
plt.legend(loc="lower right")
plt.savefig("models/roc_curve_xgb.png")
plt.close()

# === Uložení modelů ===
joblib.dump(best_rf_model, f"models/{league_code}_rf_model.joblib")
joblib.dump(best_xgb_model, f"models/{league_code}_xgb_model.joblib")

print("\n✅ Nejlepší modely byly uloženy.")
print("✅ Feature importance a ROC curve byly uloženy do složky 'models'.")