import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from utils.feature_engineering_extended import generate_extended_features

# === Načtení a příprava dat ===

df = pd.read_csv("data/E0_combined_full.csv")
df_ext = generate_extended_features(df, mode="train")


# Výběr relevantních vlastností včetně nových metrik
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
]




# Odstranění redundantních vlastností použitých při výpočtu pokročilých metrik
redundant_features = ['HS', 'AS', 'HST', 'AST', 'FTHG', 'FTAG']
features = [col for col in features if col not in redundant_features]

X = df_ext[features].fillna(0)
y = df_ext["Over_2.5"]
sample_weights = df_ext["match_weight"].fillna(1.0)

# === Normalizace všech vlastností ===

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Rozdělení dat na trénovací a testovací sadu
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_scaled, y, sample_weights, test_size=0.2, random_state=42)

# Vytvoření DataFrame pro trénovací data
X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_train_df['Over_2.5'] = y_train.values  # Přidání cílové proměnné

# Zobrazení posledních 50 řádků trénovacích dat
print("\nPosledních 50 řádků trénovacích dat:")
print(X_train_df.tail(50)[['boring_match_score', 'Over_2.5']])

# === Modely ===

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        class_weight='balanced', max_depth=3,
        min_samples_split=20,
        random_state=42),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        reg_lambda=20,
        alpha=10,
        random_state=42)
}

roc_results = {}

# === Trénování modelů a vyhodnocení ===

for name, model in models.items():
    print(f"\n▶ Trénování modelu: {name}")
    try:
        model.fit(X_train, y_train)
        print(f"✅ Model {name} úspěšně natrénován.")
    except Exception as e:
        print(f"❌ Chyba při trénování modelu {name}: {e}")
        continue

    # Predikce a vyhodnocení
    try:
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
    except Exception as e:
        print(f"❌ Chyba při predikci modelem {name}: {e}")

# === SHAP analýza pro XGBoost ===

xgb_model = models["XGBoost"]
explainer = shap.TreeExplainer(xgb_model)
X_test_df = pd.DataFrame(X_test, columns=X.columns)
shap_values = explainer.shap_values(X_test_df)

# Grafické zobrazení SHAP hodnot
shap.summary_plot(shap_values, X_test_df)
shap.dependence_plot("boring_match_score", shap_values, X_test_df)

# Scatter plot boring_match_score vs Over_2.5
plt.figure(figsize=(8, 6))
plt.scatter(X_train_df['boring_match_score'], X_train_df['Over_2.5'], alpha=0.6)
plt.xlabel("Boring Match Score (Normalized)")
plt.ylabel("Over 2.5 Goals")
plt.title("Vztah mezi Boring Match Score a Over_2.5")
plt.grid(True)
plt.show()

# Identifikace méně významných vlastností na základě SHAP hodnot
mean_shap_values = np.abs(shap_values).mean(axis=0)  # Průměrné absolutní SHAP hodnoty pro každou vlastnost
shap_importance_df = pd.DataFrame({'Feature': X_test_df.columns, 'Mean_SHAP_Value': mean_shap_values})
shap_importance_df.sort_values(by='Mean_SHAP_Value', ascending=True, inplace=True)

# Zobrazení méně významných vlastností (např. s nízkými SHAP hodnotami)
print("\nMéně významné vlastnosti na základě SHAP analýzy:")
print(shap_importance_df.head(10))  # Zobrazení top 10 nejméně významných vlastností

# Odstranění méně významných vlastností (např. s nízkými SHAP hodnotami)
threshold = 0.01  # Nastavení prahové hodnoty pro SHAP významnost
low_importance_features = shap_importance_df[shap_importance_df['Mean_SHAP_Value'] < threshold]['Feature'].tolist()
print(f"\nOdstraňované vlastnosti: {low_importance_features}")

X_reduced = X.drop(columns=low_importance_features)

# === Normalizace po odstranění méně významných vlastností ===

X_scaled_reduced = scaler.fit_transform(X_reduced)

# Rozdělení dat na trénovací a testovací sadu po odstranění méně významných vlastností
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced, w_train_reduced, w_test_reduced = train_test_split(
    X_scaled_reduced, y, sample_weights, test_size=0.2, random_state=42)

# === Trénování modelů na redukovaném datasetu ===

for name, model in models.items():
    print(f"\n▶ Trénování modelu na redukovaném datasetu: {name}")
    try:
        model.fit(X_train_reduced, y_train_reduced)
        print(f"✅ Model {name} úspěšně natrénován.")
    except Exception as e:
        print(f"❌ Chyba při trénování modelu {name}: {e}")
