
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from catboost import CatBoostClassifier
from utils.feature_engineering_extended import generate_features

# === Parametry ===
league_code = "E0"
input_csv = f"data/{league_code}_combined_full.csv"
model_output_path = f"models/{league_code}_catboost_model.joblib"
threshold_output_path = f"models/{league_code}_thresholds.json"

# === Načtení a příprava dat ===
df = pd.read_csv(input_csv)
df = generate_features(df, mode="train")
feature_cols = [col for col in df.columns if col not in ["HomeTeam", "AwayTeam", "Date", "target_over25", "match_weight"]]
X = df[feature_cols].fillna(0)
y = df["target_over25"]

# === Trénink modelu ===
model = CatBoostClassifier(eval_metric="Logloss", verbose=0, random_seed=42)

model.fit(X, y)
joblib.dump(model, model_output_path)

# === Optimalizace thresholdu ===
probs = model.predict_proba(X)[:, 1]
thresholds = np.linspace(0.1, 0.9, 81)
f1s = [f1_score(y, probs >= t) for t in thresholds]
accs = [accuracy_score(y, probs >= t) for t in thresholds]

best_f1_threshold = thresholds[np.argmax(f1s)]
best_acc_threshold = thresholds[np.argmax(accs)]

# === Uložení výsledků ===
import json
with open(threshold_output_path, "w") as f:
    json.dump({
        "catboost_best_threshold": float(best_f1_threshold),
        "catboost_best_accuracy_threshold": float(best_acc_threshold)
    }, f)

# === Vykreslení grafu ===
plt.figure(figsize=(8, 5))
plt.plot(thresholds, f1s, label="F1 Score")
plt.plot(thresholds, accs, label="Accuracy")
plt.axvline(best_f1_threshold, color="blue", linestyle="--", label=f"Best F1: {best_f1_threshold:.2f}")
plt.axvline(best_acc_threshold, color="orange", linestyle="--", label=f"Best Acc: {best_acc_threshold:.2f}")
plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Optimal Threshold for Over 2.5 Prediction")
plt.legend()
plt.tight_layout()
plt.savefig(f"threshold_optimization_{league_code}.png")
print(f"Optimal F1 threshold: {best_f1_threshold:.2f}")
print(f"Optimal Accuracy threshold: {best_acc_threshold:.2f}")
print(f"Graf uložen jako threshold_optimization_{league_code}.png")
