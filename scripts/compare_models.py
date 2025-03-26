import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from utils.feature_engineering_extended import generate_extended_features

# Načtení dat
df = pd.read_csv("data/E0_combined_full.csv")
df_ext = generate_extended_features(df)

features = [col for col in df_ext.columns if col.endswith("_form") or col.endswith("_diff") or col.startswith("over25")]
X = df_ext[features].fillna(0)
y = df_ext["Over_2.5"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

for name, model in models.items():
    print(f"\n▶ {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='f1')
        print(f"Cross-val F1 průměr: {scores.mean():.4f} | Rozptyl: {scores.std():.4f}")
    except Exception as e:
        print(f"⚠️  Cross-val selhal: {e}")

    # Feature importances
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