import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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
    or col.startswith("xg_pseudo") or col.startswith("boring")
]

X = df_ext[features].fillna(0)
y = df_ext["Over_2.5"]
sample_weights = df_ext["match_weight"].fillna(1.0)

# === Ověření vstupních dat ===
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
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# === ROC shromažďování ===
roc_results = {}

for name, model in models.items():
    print(f"\n▶ {name}")
    try:
        model.fit(X_train, y_train, sample_weight=w_train)
    except TypeError:
        model.fit(X_train, y_train)

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
        plt.axvline(x=0.5, color='red', linestyle='--', label='Práh rozhodnutí: 0.5')

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title(f"ROC Curve - {name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    # === Analýza F1-score při různých threshold hodnotách ===
    if hasattr(model, "predict_proba"):
        thresholds = np.linspace(0.1, 0.9, 81)
        f1_scores = []

        for thresh in thresholds:
            y_pred_thresh = (y_prob >= thresh).astype(int)
            f1 = f1_score(y_test, y_pred_thresh)
            f1_scores.append(f1)

        plt.figure()
        plt.plot(thresholds, f1_scores, marker='o')
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        plt.axvline(x=best_thresh, linestyle='--', color='red', label=f'Optimální threshold: {best_thresh:.2f}')
        plt.title(f'F1-score vs Threshold - {name}')
        plt.xlabel('Threshold')
        plt.ylabel('F1-score')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f"📈 Nejlepší threshold pro {name}: {best_thresh:.2f} s F1-score: {best_f1:.4f}")

# === Sloučená ROC křivka ===
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
