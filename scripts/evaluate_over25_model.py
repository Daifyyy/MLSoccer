
import pandas as pd
import numpy as np
import joblib
import json
import sys

# Připojíme vlastní feature engineering skript
sys.path.append(".")

from utils.feature_engineering_extended import generate_features

# === Načtení validačních dat ===
val_df = pd.read_csv("data_validation/E0_validate_set.csv")
val_df["FTG"] = val_df["FTHG"] + val_df["FTAG"]
val_df["target_over25"] = (val_df["FTG"] > 2.5).astype(int)

# === Feature engineering ===
val_fe = generate_features(val_df, mode="predict")
feature_cols = [col for col in val_fe.columns if col not in ["HomeTeam", "AwayTeam", "Date", "target_over25", "match_weight"]]
X_val = val_fe[feature_cols].fillna(0)
y_val = val_fe["target_over25"]

# === Načtení modelu a thresholdu ===
model = joblib.load("models/E0_catboost_model.joblib")
with open("models/E0_thresholds.json") as f:
    thresholds = json.load(f)
threshold = thresholds["catboost_best_threshold"]

# === Výpočet predikcí ===
y_proba = model.predict_proba(X_val)[:, 1]
y_pred = (y_proba >= threshold).astype(int)

# === Přehled predikcí ===
result_df = pd.DataFrame({
    "Date": val_fe["Date"],
    "HomeTeam": val_fe["HomeTeam"],
    "AwayTeam": val_fe["AwayTeam"],
    "Predicted_Over25_Probability": y_proba,
    "Predicted_Over25_Label": y_pred,
    "Actual_Over25_Label": y_val,
})
result_df["Prediction_Result"] = result_df.apply(
    lambda row: "OK" if row["Predicted_Over25_Label"] == row["Actual_Over25_Label"] else "NOK", axis=1
)

# === Sázkové rozhodnutí ===
def classify_confidence(prob):
    if prob >= 0.65:
        return "Over"
    elif prob <= 0.40:
        return "Under"
    else:
        return "No Bet"

result_df["Confidence_Bet"] = result_df["Predicted_Over25_Probability"].apply(classify_confidence)

# === Vyhodnocení sázky ===
result_df["Confidence_Result"] = result_df.apply(
    lambda row: "OK" if (
        (row["Confidence_Bet"] == "Over" and row["Actual_Over25_Label"] == 1) or
        (row["Confidence_Bet"] == "Under" and row["Actual_Over25_Label"] == 0)
    ) else ("NOK" if row["Confidence_Bet"] != "No Bet" else ""), axis=1
)

# === Shrnutí ===
total = len(result_df)
total_bets = result_df[result_df["Confidence_Bet"] != "No Bet"].shape[0]
correct_bets = result_df[result_df["Confidence_Result"] == "OK"].shape[0]
accuracy = correct_bets / total_bets if total_bets > 0 else 0

print(f"Celkem zápasů: {total}")
print(f"Zápasů s tipem (Over ≥ 65% nebo Under ≤ 40%): {total_bets}")
print(f"Z toho správných: {correct_bets}")
print(f"Přesnost predikcí v tipovaných zápasech: {accuracy:.2%}")

# === Uložení výsledků ===
result_df.to_csv("over25_predictions_betting_analysis.csv", index=False)
print("Detailní výsledky uloženy do 'over25_predictions_betting_analysis.csv'")
