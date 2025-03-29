import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

from utils.feature_engineering_extended import generate_extended_features

# === Načtení a příprava dat ===
df = pd.read_csv("data/E0_combined_full.csv")
df_ext = generate_extended_features(df)

features = [
    col for col in df_ext.columns
    if col.endswith("_form") or col.endswith("_diff")
    or col.startswith("over25") or col.startswith("elo_rating")
    or col.endswith("_last5") or col.endswith("_weight")
    or col.endswith("_cards") or col.endswith("_fouls")
    or col.startswith("xg") or col.startswith("boring")
]

X = df_ext[features].fillna(0)
y = df_ext["Over_2.5"]
sample_weights = df_ext["match_weight"].fillna(1.0)

# === Kontrola vstupních dat ===
print("\n🧪 Kontrola vstupních dat:")
print("Počet NaN ve featurech:", X.isna().sum().sum())
print("Počet nulových sloupců:", (X == 0).all().sum())
print("Popisová statistika:")
print(X.describe().T.tail(10))

# === Rozdělení dat ===
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, sample_weights, test_size=0.2, random_state=42)

# === Modely ===
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
}

roc_results = {}

# === Trénování modelů ===
for name, model in models.items():
    print(f"\n▶ {name}")
    model.fit(X_train, y_train, sample_weight=w_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='f1')
        print(f"Cross-val F1 průměr: {scores.mean():.4f} | Rozptyl: {scores.std():.4f}")
    except Exception as e:
        print(f"⚠️  Cross-val selhal: {e}")

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        roc_results[name] = (fpr, tpr, roc_auc)

        plt.figure()
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title(f"ROC Curve - {name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
# === Kalibrace: Random Forest ===
print("\n▶ Kalibrace: Random Forest")

rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train, sample_weight=w_train)

calibrated_rf = CalibratedClassifierCV(estimator=rf_model, method='sigmoid', cv=5)
calibrated_rf.fit(X_train, y_train, sample_weight=w_train)

# 📏 Vyhodnocení kalibrovaného Random Forest
y_pred_rf_cal = calibrated_rf.predict(X_test)
print("Classification Report – Random Forest (Calibrovaný):")
print(classification_report(y_test, y_pred_rf_cal))

# 📉 ROC křivka pro kalibrovaný Random Forest
y_prob_rf_cal = calibrated_rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf_cal)
roc_auc_rf = auc(fpr_rf, tpr_rf)
roc_results["Random Forest (calibrated)"] = (fpr_rf, tpr_rf, roc_auc_rf)
# === F1 vs Threshold – Kalibrovaný RF
if hasattr(calibrated_rf, "predict_proba"):
    probs = calibrated_rf.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 50)
    f1s = [f1_score(y_test, probs > t) for t in thresholds]
    best_thresh_rf = thresholds[np.argmax(f1s)]
    print(f"\n✅ Optimální threshold pro Kalibrovaný Random Forest: {best_thresh_rf:.2f}")

    plt.figure()
    plt.plot(thresholds, f1s, marker='o')
    plt.axvline(best_thresh_rf, color='red', linestyle='--', label=f"Optimální threshold: {best_thresh_rf:.2f}")
    plt.title("F1-score vs Threshold – Random Forest (Kalibrovaný)")
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plt.figure()
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (Kalibrovaný, AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve – Random Forest (Kalibrovaný)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()



print("\n▶ Kalibrace: XGBoost (ručně)")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Trénuj XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train, sample_weight=w_train)

# 2. Získej pravděpodobnosti
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# === ROC křivka před kalibrací (XGBoost)
if hasattr(xgb_model, "predict_proba"):
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
    auc_xgb = auc(fpr_xgb, tpr_xgb)
    plt.figure()
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("ROC Curve – XGBoost")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 3. Natrénuj kalibrační model (logistická regrese na pravděpodobnostech)
calibration_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
calibration_model.fit(y_prob_xgb.reshape(-1, 1), y_test)

# 4. Kalibrované pravděpodobnosti
y_prob_cal = calibration_model.predict_proba(y_prob_xgb.reshape(-1, 1))[:, 1]
y_pred_cal = (y_prob_cal > 0.5).astype(int)

# 5. Vyhodnocení
print(classification_report(y_test, y_pred_cal))


# ROC křivka
fpr_cal, tpr_cal, _ = roc_curve(y_test, y_prob_cal)
roc_auc_cal = auc(fpr_cal, tpr_cal)
roc_results["XGBoost (calibrated)"] = (fpr_cal, tpr_cal, roc_auc_cal)

plt.figure()
plt.plot(fpr_cal, tpr_cal, label=f'XGBoost (AUC = {roc_auc_cal:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve - XGBoost (Kalibrovaný)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Threshold optimalizace pro XGBoost ===
thresholds = np.linspace(0.1, 0.9, 81)
f1_scores = []

for thresh in thresholds:
    y_pred_thresh = (y_prob_cal >= thresh).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    f1_scores.append(f1)

best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]
best_f1 = f1_scores[best_idx]



plt.figure()
plt.plot(thresholds, f1_scores, marker='o')
plt.axvline(x=best_thresh, linestyle='--', color='red', label=f'Optimální threshold: {best_thresh:.2f}')
plt.title(f'F1-score vs Threshold - XGBoost (Kalibrovaný)')
plt.xlabel('Threshold')
plt.ylabel('F1-score')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(f"📏 Kalibrovaný model XGBoost – nejlepší threshold: {best_thresh:.2f} | F1: {best_f1:.4f}")

# === Shrnutí ROC všech modelů ===
plt.figure(figsize=(8, 6))
for name, (fpr, tpr, roc_auc) in roc_results.items():
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
