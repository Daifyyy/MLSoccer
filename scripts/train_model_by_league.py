import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
from xgboost import XGBClassifier
from utils.feature_engineering_extended import generate_extended_features

# Výběr ligy
league = input("Zadej zkratku ligy (např. E0 nebo SP1): ")

# Cesty
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
combined_path = os.path.join(BASE_DIR, "data", f"{league}_combined_full.csv")
model_path = os.path.join(BASE_DIR, "models", f"{league}_rf_model.joblib")
xgb_model_path = os.path.join(BASE_DIR, "models", f"{league}_xgb_model.joblib")

# Načti a připrav data
df = pd.read_csv(combined_path)
df_ext = generate_extended_features(df)
features = [col for col in df_ext.columns if col.endswith("_form") or col.endswith("_diff") or col.startswith("over25") or col.startswith("elo_rating")]
X = df_ext[features].fillna(0)
y = df_ext["Over_2.5"]
sample_weights = df_ext["match_weight"].fillna(1.0)

# Train/test split
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, sample_weights, test_size=0.2, random_state=42)

# Parametry pro GridSearch
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

print("\nLadím hyperparametry pro Random Forest pomocí GridSearchCV...")
grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),
                           param_grid,
                           cv=3,
                           scoring='f1',
                           n_jobs=-1,
                           verbose=1)
grid_search.fit(X_train, y_train, sample_weight=w_train)

best_rf_model = grid_search.best_estimator_
print("\nNejlepší parametry pro RF:")
print(grid_search.best_params_)

# Ulož RF model
joblib.dump(best_rf_model, model_path)
print(f"\nRandom Forest model pro ligu {league} uložen do {model_path}")

# Trénuj XGBoost s vážením
print("\nTrénuji XGBoost model...")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train, sample_weight=w_train)
joblib.dump(xgb_model, xgb_model_path)
print(f"\nXGBoost model pro ligu {league} uložen do {xgb_model_path}")
