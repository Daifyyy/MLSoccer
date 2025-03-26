import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Dynamická cesta k datům relativní k tomuto skriptu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DATA = os.path.join(BASE_DIR, "data", "E0_combined_full.csv")
MODEL_OUTPUT = os.path.join(BASE_DIR, "models", "over25_model.joblib")

# Načtení dat
train_df = pd.read_csv(TRAINING_DATA)

# Trénink
features = ["HS", "AS", "HST", "AST", "HC", "AC"]
X_train = train_df[features]
y_train = train_df["Over_2.5"]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Uložení modelu
joblib.dump(model, MODEL_OUTPUT)
print(f"Model uložen do: {MODEL_OUTPUT}")
