import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import numpy as np
from utils.feature_engineering_extended import generate_extended_features
import sys
sys.stdout.flush()

# Načtení dat
df = pd.read_csv("data/E0_combined_full.csv")
df_ext = generate_extended_features(df)

# Výběr featur
features = [
    col for col in df_ext.columns
    if col.endswith("_form") or col.endswith("_diff")
    or col.startswith("over25") or col.startswith("elo_rating")
    or col.endswith("_last5") or col.endswith("_weight")
    or col.endswith("_cards") or col.endswith("_fouls")
]

X = df_ext[features].fillna(0)
y = df_ext["Over_2.5"]
sample_weights = df_ext["match_weight"].fillna(1.0)

print("=== ZAČÍNÁ DIAGNOSTIKA DAT ===")
print(X.head())
print(y.value_counts())
print("=== KONEC DIAGNOSTIKY ===")

# 💡 Diagnostika před tréninkem
print("📊 Náhled na vstupní data (X.describe()):")
print(X.describe().T,flush=True)

print("\n🔍 Počet nulových hodnot ve featurách:")
print(X.isnull().sum())

print("\n🔍 Počet unikátních hodnot ve featurách:")
print(X.nunique())

# Kontrola: odstranění konstantních sloupců
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
if constant_cols:
    print(f"❗ Odstraňuji konstantní sloupce: {constant_cols}")
    X.drop(columns=constant_cols, inplace=True)
    features = [f for f in features if f not in constant_cols]

# Výpis rozložení cílové proměnné
print("\n📊 Rozložení tříd v cílové proměnné:")
print(y.value_counts())

# Rozdělení dat
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, sample_weights, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(
        max_depth=6,#4
        n_estimators=300,#300
        learning_rate=0.1,#0.1
        subsample=0.6,#0.6
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss",
    )
}

for name, model in models.items():
    print(f"\n▶ {name}")
    try:
        model.fit(X_train, y_train, sample_weight=w_train)
    except TypeError:
        model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.45).astype(int)
    else:
        y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

    # Cross-validation
    print(f"📘 Cross-validating {name}...")
    try:
        if hasattr(model, "predict_proba"):
            scores = []
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                w_tr = sample_weights.iloc[train_idx]
                model.fit(X_tr, y_tr, sample_weight=w_tr)
                y_proba_val = model.predict_proba(X_val)[:, 1]
                y_val_pred = (y_proba_val >= 0.45).astype(int)
                scores.append(f1_score(y_val, y_val_pred))
            print(f"Cross-val F1 průměr: {np.mean(scores):.4f} | Rozptyl: {np.std(scores):.4f}")
        else:
            scores = cross_val_score(model, X, y, cv=5, scoring='f1')
            print(f"Cross-val F1 průměr: {scores.mean():.4f} | Rozptyl: {scores.std():.4f}")
    except Exception as e:
        print(f"⚠️  Cross-val selhal: {e}")

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = importances.argsort()[::-1]
        plt.figure()
        plt.title(f"{name} - Feature Importances")
        plt.bar([features[i] for i in sorted_idx], importances[sorted_idx])
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    elif hasattr(model, "coef_"):
        importances = model.coef_[0]
        sorted_idx = abs(importances).argsort()[::-1]
        plt.figure()
        plt.title(f"{name} - Coefficient Importance")
        plt.bar([features[i] for i in sorted_idx], importances[sorted_idx])
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
