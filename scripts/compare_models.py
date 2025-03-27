import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid, cross_val_score
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import numpy as np
from utils.feature_engineering_extended import generate_extended_features
from xgboost import XGBClassifier

# Načtení dat
df = pd.read_csv("data/E0_combined_full.csv")
df_ext = generate_extended_features(df)

features = [
    col for col in df_ext.columns
    if col.endswith("_form") or col.endswith("_diff")
    or col.startswith("over25") or col.startswith("elo_rating")
    or col.endswith("_last5")
]

X = df_ext[features].fillna(0)
y = df_ext["Over_2.5"]
sample_weights = df_ext["match_weight"].fillna(1.0)

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42)
}

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

# === XGBoost ruční ladění ===
print("\n▶ XGBoost")

num_pos = sum(y_train == 1)
num_neg = sum(y_train == 0)
scale_pos_weight = num_neg / num_pos

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'scale_pos_weight': [scale_pos_weight]
}

grids = list(ParameterGrid(param_grid))
best_model = None
best_score = 0
best_params = None

for params in grids:
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **params)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[test_idx]
        w_tr = w_train.iloc[train_idx]
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        y_pred = model.predict(X_val)
        scores.append(f1_score(y_val, y_pred))
    avg_score = np.mean(scores)
    if avg_score > best_score:
        best_score = avg_score
        best_model = model
        best_params = params

print("\n✅ Nejlepší parametry pro XGBoost:")
print(best_params)

best_model.fit(X_train, y_train, sample_weight=w_train)
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, test_idx in skf.split(X, y):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    best_model.fit(X_tr, y_tr)
    y_pred = best_model.predict(X_te)
    scores.append(f1_score(y_te, y_pred))
print(f"Cross-val F1 průměr: {np.mean(scores):.4f} | Rozptyl: {np.std(scores):.4f}")

importances = best_model.feature_importances_
sorted_idx = importances.argsort()[::-1]
plt.figure()
plt.title("XGBoost - Feature Importances")
plt.bar([features[i] for i in sorted_idx], importances[sorted_idx])
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
