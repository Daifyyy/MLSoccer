
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from utils.feature_engineering_extended import generate_features

# === Nastavení ===
LEAGUE_CODE = "E0"
VALIDATION_CSV = "data_validation/E0_validate_set.csv"
MODEL_PATH = f"models/{LEAGUE_CODE}_catboost_model.joblib"

# === Načtení dat ===
df = pd.read_csv(VALIDATION_CSV)
df["FTG"] = df["FTHG"] + df["FTAG"]
df["target_over25"] = (df["FTG"] > 2.5).astype(int)

# === Feature engineering ===
df_fe = generate_features(df, mode="predict")
feature_cols = [col for col in df_fe.columns if col not in ["HomeTeam", "AwayTeam", "Date", "target_over25", "match_weight"]]
X_val = df_fe[feature_cols].fillna(0)

# === Načtení modelu ===
model = joblib.load(MODEL_PATH)

# === SHAP setup ===
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# === Vybrané řádky k analýze ===
rows_to_plot = [316, 317, 318, 319, 320]  # indexy jsou 0-based
rows_to_plot = [i for i in rows_to_plot if i < len(X_val)]

for idx in rows_to_plot:
    sample = X_val.iloc[[idx]]
    shap_val = shap_values[idx:idx+1]

    # SHAP decision plot
    try:
        plt.figure()
        shap.decision_plot(
            explainer.expected_value,
            shap_val,
            sample,
            feature_names=feature_cols,
            show=False
        )
        plt.tight_layout()
        plt.savefig(f"shap_decision_plot_row_{idx+1}.png")
        print(f"✅ Decision plot saved: shap_decision_plot_row_{idx+1}.png")
    except Exception as e:
        print(f"❌ Decision plot error (row {idx+1}): {e}")

    # SHAP force plot
    try:
        plt.figure()
        shap.force_plot(
            explainer.expected_value,
            shap_val[0],
            sample.iloc[0],
            matplotlib=True,
            show=False
        )
        plt.savefig(f"shap_force_plot_row_{idx+1}.png", bbox_inches='tight')
        print(f"✅ Force plot saved: shap_force_plot_row_{idx+1}.png")
    except Exception as e:
        print(f"❌ Force plot error (row {idx+1}): {e}")

