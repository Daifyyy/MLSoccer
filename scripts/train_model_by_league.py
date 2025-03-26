import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
from utils.feature_engineering_extended import generate_extended_features

# Výběr ligy
league = input("Zadej zkratku ligy (např. E0 nebo SP1): ")

# Cesty
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
combined_path = os.path.join(BASE_DIR, "data", f"{league}_combined_full.csv")
model_path = os.path.join(BASE_DIR, "models", f"{league}_rf_model.joblib")

# Načti a připrav data
df = pd.read_csv(combined_path)
df_ext = generate_extended_features(df)
features = [col for col in df_ext.columns if col.endswith("_form") or col.endswith("_diff") or col.startswith("over25")]
X = df_ext[features]
y = df_ext["Over_2.5"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parametry pro GridSearch
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

print("\nLadím hyperparametry pomocí GridSearchCV...")
grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),
                           param_grid,
                           cv=3,
                           scoring='f1',
                           n_jobs=-1,
                           verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("\nNejlepší parametry:")
print(grid_search.best_params_)

# Ulož model
joblib.dump(best_model, model_path)
print(f"\nModel Random Forest pro ligu {league} byl uložen do {model_path}")
