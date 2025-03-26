import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Dynamická cesta k datům relativní k tomuto skriptu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DATA = os.path.join(BASE_DIR, "data", "E0_combined_full.csv")
TEST_DATA = os.path.join(BASE_DIR, "data", "E0 test set.csv")


# Načtení dat
train_df = pd.read_csv(TRAINING_DATA)
test_df = pd.read_csv(TEST_DATA)

# Přidání cílové proměnné do testovací sady
test_df["TotalGoals"] = test_df["FTHG"] + test_df["FTAG"]
test_df["Over_2.5"] = (test_df["TotalGoals"] > 2.5).astype(int)

# Vstupní proměnné
features = ["HS", "AS", "HST", "AST", "HC", "AC"]
X_train = train_df[features]
y_train = train_df["Over_2.5"]
X_test = test_df[features]
y_test = test_df["Over_2.5"]

# Trénování modelu
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predikce
predictions = model.predict(X_test)

# Uložíme model do souboru
joblib.dump(model, "../models/over25_model.joblib")
print("Model uložen jako over25_model.joblib")

# Vyhodnocení
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
