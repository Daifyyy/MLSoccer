import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score,balanced_accuracy_score
from utils.feature_engineering_extended import generate_extended_features

# === Uživatelské vstupy ===
league_code = "E0"
validation_csv_path = f"data_validation/{league_code}_validate_set.csv"
rf_model_path = f"models/{league_code}_rf_model.joblib"
xgb_model_path = f"models/{league_code}_xgb_model.joblib"

# === 1. Načti validační data ===
df_val_raw = pd.read_csv(validation_csv_path)

# === 2. Odstraň sázkové kurzy ===
betting_related = [col for col in df_val_raw.columns if any(x in col for x in ["B365", "WHH", "PCA", "Max", "BF"])]
df_val_raw = df_val_raw.drop(columns=betting_related, errors='ignore')

# === 3. Vygeneruj featury ===
df_val_ext = generate_extended_features(df_val_raw, mode="train")

# === 4. Definuj featury ===
features = [
    "shooting_efficiency",
    "boring_match_score",
    "away_xg",
    "home_xg",
    "passivity_score",
    "home_form_xg",
    "match_weight",
    "away_form_xg",
    "home_form_shots",
    "elo_rating_away",
    "prob_under25",
    "over25_expectation_gap",
    "away_form_shots",
    "momentum_score",
    "behavior_balance",
    "corner_diff_last5",
    "shot_diff_last5m",
    "elo_rating_home",
    "tempo_score",
    "log_odds_under25",
    "prob_over25",
    "fouls_diff",
    "aggressiveness_score",
    "card_diff",
    "shot_on_target_diff_last5",
    "xg_away_last5",
    "xg_home_last5",
    "missing_xg_home_last5",
    "missing_xg_away_last5",
    "missing_home_form_xg",
    "missing_home_form_shots",
    "missing_away_form_xg",
    "missing_away_form_shots",
    "missing_log_odds_under25",
    "xg_conceded_home_last5",
    "xg_conceded_away_last5",
    "avg_xg_conceded",
    "xg_ratio",
    "defensive_pressure",
    "missing_xg_conceded_home_last5",
    "missing_xg_conceded_away_last5",
    "missing_avg_xg_conceded",
    "missing_xg_ratio",
    "missing_defensive_pressure",  

    
]


X_val = df_val_ext[features].fillna(0)
y_val = df_val_ext["Over_2.5"]

# === 5. Načti modely ===
models = {
    "Random Forest": joblib.load(rf_model_path),
    "XGBoost": joblib.load(xgb_model_path)
}

# === 6. Vyhodnocení ===
thresh_rf = 0.5
thresh_xgb = 0.5
thresholds = {
    "Random Forest": thresh_rf,
    "XGBoost": thresh_xgb
}

# === 6. Vyhodnocení + Optimalizace thresholdu přes Youden's J statistic ===
for name, model in models.items():
    y_probs = model.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 80)
    best_thresh = 0.5
    best_j_score = -1
    j_scores = []

    for t in thresholds:
        y_pred = y_probs > t
        tp = ((y_val == 1) & (y_pred == 1)).sum()
        fn = ((y_val == 1) & (y_pred == 0)).sum()
        tn = ((y_val == 0) & (y_pred == 0)).sum()
        fp = ((y_val == 0) & (y_pred == 1)).sum()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        j = sensitivity + specificity - 1
        j_scores.append(j)

        if j > best_j_score:
            best_j_score = j
            best_thresh = t

    y_pred = y_probs > best_thresh

    print(f"\n\n📊 Výsledky modelu: {name}")
    print(f"🎯 Dynamicky optimalizovaný threshold (Youden's J): {best_thresh:.2f}")
    print(classification_report(y_val, y_pred))
    print("🧩 Matrice záměn:")
    print(confusion_matrix(y_val, y_pred))
