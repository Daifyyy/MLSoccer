import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from utils.feature_engineering_extended import generate_extended_features

# === Uživatelské vstupy ===
league_code = "E0"  # nebo SP1 atd.
model_type = "rf"  # "rf" nebo "xgb"
validation_csv_path = "data_validation/E0_validate_set.csv"
model_path = f"models/{league_code}_{model_type}_model.joblib"

# === 1. Načti validační data (aktuální sezóna) ===
df_val_raw = pd.read_csv(validation_csv_path)



# === 2. Odstraň pouze sázkové kurzy (zatím NE výsledky zápasu)
betting_related = [col for col in df_val_raw.columns if any(x in col for x in ["B365", "WHH", "PCA", "Max", "Avg", "BF"])]
df_val_raw = df_val_raw.drop(columns=betting_related)


# === 3. Vygeneruj featury ===
df_val_ext = generate_extended_features(df_val_raw, mode="train")

columns_to_remove = [
    "FTHG", "FTAG", "FTR",  # výsledky zápasu
    "HTHG", "HTAG", "HTR",  # poločasové výsledky
]

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
    "h2h_goal_avg",
    "defensive_stability",
    "tempo_score",
    "passivity_score",
    "home_under25_last5",
    "away_under25_last5",
    "home_avg_goals_last5_home",
    "away_avg_goals_last5_away"
]

X_val = df_val_ext[features].fillna(0)
y_val = df_val_ext["Over_2.5"]

# === 5. Načti model ===
model = joblib.load(model_path)
y_probs = model.predict_proba(X_val)[:, 1]
y_pred = y_probs > 0.5  # threshold můžeš vyladit podle potřeby

# === 6. Výstupy ===
print("\n\n📊 Klasifikace na validační sadě")
print(classification_report(y_val, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_val, y_pred))