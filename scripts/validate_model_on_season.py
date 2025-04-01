import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from utils.feature_engineering_extended import generate_extended_features
from sklearn.metrics import f1_score


# === Uživatelské vstupy ===
league_code = "E0"
validation_csv_path = f"data_validation/{league_code}_validate_set.csv"
rf_model_path = f"models/{league_code}_rf_model.joblib"
xgb_model_path = f"models/{league_code}_xgb_model.joblib"

# === 1. Načti validační data ===
df_val_raw = pd.read_csv(validation_csv_path)

# === 2. Odstraň sázkové kurzy ===
betting_related = [col for col in df_val_raw.columns if any(x in col for x in ["B365", "WHH", "PCA", "Max", "Avg", "BF"])]
df_val_raw = df_val_raw.drop(columns=betting_related, errors='ignore')

# === 3. Vygeneruj featury ===
df_val_ext = generate_extended_features(df_val_raw, mode="train")

# === 4. Definuj featury ===
features = [
    "shooting_efficiency",
    "elo_rating_home",
    "elo_rating_away",
    "momentum_score",
    "home_xg",
    "away_xg",
    "xg_home_last5",
    "xg_away_last5",
    "corner_diff_last5",
    "shot_on_target_diff_last5",
    "shot_diff_last5m",
    "fouls_diff",
    "card_diff",
    "boring_match_score",
    "match_weight",
    "tempo_score",
    "passivity_score",
    "missing_corner_diff_last5",
    "missing_shot_on_target_diff_last5",
    "missing_shot_diff_last5m",
    "missing_fouls_diff",
    "missing_card_diff",
    "missing_xg_away_last5",
    "missing_xg_home_last5",
]

X_val = df_val_ext[features].fillna(0)
y_val = df_val_ext["Over_2.5"]

# === 5. Načti modely ===
models = {
    "Random Forest": joblib.load(rf_model_path),
    "XGBoost": joblib.load(xgb_model_path)
}

# === 6. Vyhodnocení s optimalizací thresholdu ===
thresholds = np.arange(0.1, 0.9, 0.01)

for name, model in models.items():
    y_probs = model.predict_proba(X_val)[:, 1]

    best_thresh = 0.5
    best_f1 = 0

    for t in thresholds:
        y_pred = y_probs > t
        score = f1_score(y_val, y_pred)
        if score > best_f1:
            best_f1 = score
            best_thresh = t

    print(f"\n\n📊 Výsledky modelu: {name}")
    print(f"🎯 Nejlepší threshold: {best_thresh:.2f} s F1-skóre: {best_f1:.4f}")
    y_pred_best = y_probs > best_thresh
    print(classification_report(y_val, y_pred_best))
    print("🧩 Matrice záměn:")
    print(confusion_matrix(y_val, y_pred_best))
