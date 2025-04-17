
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, brier_score_loss
from utils.feature_engineering_extended import generate_features

# === Nastavení ===
LEAGUE_CODE = "E0"
VALIDATION_CSV = "data_validation/E0_validate_set.csv"
MODEL_PATH = f"models/{LEAGUE_CODE}_catboost_model.joblib"
THRESHOLDS_PATH = f"models/{LEAGUE_CODE}_thresholds.json"

# === Načtení dat ===
df = pd.read_csv(VALIDATION_CSV)
df["FTG"] = df["FTHG"] + df["FTAG"]
df["target_over25"] = (df["FTG"] > 2.5).astype(int)

# === Feature engineering ===
df_fe = generate_features(df, mode="predict")
feature_cols = [col for col in df_fe.columns if col not in ["HomeTeam", "AwayTeam", "Date", "target_over25", "match_weight"]]
X_val = df_fe[feature_cols].fillna(0)
y_val = df_fe["target_over25"]

# === Načtení modelu a thresholdu ===
model = joblib.load(MODEL_PATH)
with open(THRESHOLDS_PATH) as f:
    thresholds = json.load(f)
threshold = thresholds["catboost_best_threshold"]

# === Predikce a klasifikace ===
probs = model.predict_proba(X_val)[:, 1]
preds = (probs >= threshold).astype(int)

# === Základní metriky ===
print("\n📊 Výsledky:")
print(classification_report(y_val, preds, target_names=["Under 2.5", "Over 2.5"]))
print("Confusion Matrix:")
print(confusion_matrix(y_val, preds))
print(f"ROC AUC: {roc_auc_score(y_val, probs):.4f}")
print(f"Brier score: {brier_score_loss(y_val, probs):.4f}")

# === Feature Importance ===
importances = model.feature_importances_
top_idx = np.argsort(importances)[::-1][:20]
top_features = [feature_cols[i] for i in top_idx]
top_importances = importances[top_idx]

print("\n🔝 Top 20 featur podle importance:")
for f, imp in zip(top_features, top_importances):
    print(f"{f:35s} {imp:.4f}")

# === SHAP analýza ===
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# === SHAP summary plot ===
plt.figure()
shap.summary_plot(shap_values, X_val, feature_names=feature_cols, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot.png")
print("\n📈 SHAP summary plot uložen jako shap_summary_plot.png")

# === SHAP barplot pro nové metriky ===
custom_features = ["xg_proxy_diff", "low_tempo_index", "defense_suppression_score"]
missing = [f for f in custom_features if f not in feature_cols]
if missing:
    print(f"⚠️ Následující metriky chybí ve feature_cols: {missing}")
else:
    idxs = [feature_cols.index(f) for f in custom_features]
    shap_vals = np.abs(shap_values)[:, idxs].mean(axis=0)
    plt.figure()
    plt.barh(custom_features, shap_vals)
    plt.xlabel("Mean |SHAP value|")
    plt.title("🔍 SHAP význam nových metrik")
    plt.tight_layout()
    plt.savefig("shap_custom_metrics.png")
    print("💥 SHAP barplot pro nové metriky uložen jako shap_custom_metrics.png")
