import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from utils.feature_engineering_extended import generate_extended_features
from utils.data_loader import load_data_by_league, filter_team_matches, filter_h2h_matches
from sklearn.metrics import f1_score
from pyro.infer import Predictive
import torch
import pyro
import pyro.distributions as dist
import torch.nn as nn
import pyro.poutine as poutine
import torch.serialization
from torch import serialization
import dill
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, roc_curve, auc
from sklearn.calibration import calibration_curve
import json
from pyro.infer.autoguide import AutoDiagonalNormal
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.serialization import safe_globals
from torch.distributions import constraints as torch_constraints
from pyro.distributions import constraints as pyro_constraints

def plot_confidence_vs_accuracy(y_pred_samples, y_true, threshold=0.5):
    """
    Vykreslí scatter plot:
    - osa X: průměrná pravděpodobnost
    - osa Y: šířka CI
    - barva: správná vs. špatná predikce
    """
    mean_probs = y_pred_samples.mean(axis=0)
    lower = np.percentile(y_pred_samples, 10, axis=0)
    upper = np.percentile(y_pred_samples, 90, axis=0)
    ci_width = upper - lower

    preds = (mean_probs > threshold).astype(int)
    correct = preds == y_true

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=mean_probs, y=ci_width, hue=correct, palette={True: "green", False: "red"})
    plt.axhline(0.5, color="gray", linestyle="--", label="CI = 0.5")
    plt.axvline(threshold, color="gray", linestyle="--", label=f"Threshold = {threshold}")
    plt.title("Vztah mezi jistotou (šířkou CI), predikcí a přesností")
    plt.xlabel("Predikovaná pravděpodobnost Over 2.5")
    plt.ylabel("Šířka intervalu spolehlivosti (10–90 %)")
    plt.legend(title="Správná predikce")
    plt.tight_layout()
    fig = plt.gcf()  # aktuální figure
    st.pyplot(fig)



st.set_page_config(layout="wide")
st.title("⚽ Predikce Over 2.5 gólů se sílou sázkové příležitosti")

league_code = st.selectbox("Zvol ligu:", ["E0", "SP1", "D1", "I1", "F1","E1", "T1", "D2", "N1", "B1","P1"])

df_raw = load_data_by_league(league_code)
teams = sorted(set(df_raw["HomeTeam"]).union(set(df_raw["AwayTeam"])))
home_team = st.selectbox("Domácí tým:", teams)
away_team = st.selectbox("Hostující tým:", teams)

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

if st.button("🔍 Spustit predikci"):
    try:
        df_filtered = pd.concat([
            filter_team_matches(df_raw, home_team),
            filter_team_matches(df_raw, away_team),
            filter_h2h_matches(df_raw, home_team, away_team)
        ]).drop_duplicates().reset_index(drop=True)

        future_match = pd.DataFrame([{
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'Date': pd.to_datetime(datetime.today().date()),
        }])

        df_pred_input = pd.concat([df_filtered, future_match], ignore_index=True)
        df_ext = generate_extended_features(df_pred_input, mode="predict")
        
        


        match_row = df_ext[
            (df_ext["HomeTeam"] == home_team) &
            (df_ext["AwayTeam"] == away_team) &
            (df_ext["Date"].dt.date == datetime.today().date())
        ]
        
        # === Načtení featur z trénování ===
        with open(f"models/{league_code}_bayes_feature_groups.json", "r") as f:
            feature_groups = json.load(f)
            feature_cols = feature_groups["feature_cols"]
            binary_cols = feature_groups["binary_cols"]
            features = feature_cols + binary_cols  # 🧠 celkový seznam přesně 39 sloupců

        
        print("Featury v match_row:", match_row.columns.tolist())
        missing = [col for col in features if col not in match_row.columns]
        print("❌ Chybějící featury:", missing)
        if match_row.empty:
            st.warning("⚠️ Nepodařilo se najít vstupní data pro predikci.")
        else:
           
            # Přidání chybějících sloupců s fallbackem
            # Přidání chybějících sloupců před výběrem X_input
            for col in features:
                if col not in match_row.columns:
                    match_row[col] = 0  # nebo df_val_ext[col].mean() pokud chceš lepší fallback

            X_input = match_row[features].fillna(0)
            rf_model = joblib.load(f"models/{league_code}_rf_model.joblib")
            xgb_model = joblib.load(f"models/{league_code}_xgb_model.joblib")

            df_val_ext = generate_extended_features(df_filtered, mode="train")
            X_val = df_val_ext[features].fillna(0)
            y_val = df_val_ext["Over_2.5"]
     
            
            
            def calculate_optimal_threshold(model):
                y_probs = model.predict_proba(X_val)[:, 1]
                thresholds = np.linspace(0.1, 0.9, 80)
                best_thresh = 0.5
                best_j_score = -1
                for t in thresholds:
                    y_pred = y_probs > t
                    tp = ((y_val == 1) & (y_pred == 1)).sum()
                    fn = ((y_val == 1) & (y_pred == 0)).sum()
                    tn = ((y_val == 0) & (y_pred == 0)).sum()
                    fp = ((y_val == 0) & (y_pred == 1)).sum()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    j = sensitivity + specificity - 1
                    if j > best_j_score:
                        best_j_score = j
                        best_thresh = t
                return best_thresh

            rf_thresh = calculate_optimal_threshold(rf_model)
            xgb_thresh = calculate_optimal_threshold(xgb_model)

            rf_prob = rf_model.predict_proba(X_input)[0][1]
            xgb_prob = xgb_model.predict_proba(X_input)[0][1]

            rf_pred = rf_prob > rf_thresh
            xgb_pred = xgb_prob > xgb_thresh

            def get_confidence(prob):
                if prob >= 0.8:
                    return "🔥 Vysoká"
                elif prob >= 0.65:
                    return "✅ Střední"
                else:
                    return "⚠️ Nízká"

            st.subheader("📊 Predikce:")
            st.markdown(f"**Random Forest:** {rf_prob * 100:.2f}% ({1 / rf_prob:.2f}) pravděpodobnost Over 2.5 → {'✅ ANO' if rf_pred else '❌ NE'}")
            st.markdown(f"Confidence: {get_confidence(rf_prob)} (threshold: {rf_thresh:.2f})")
            st.markdown("---")

            st.markdown(f"**XGBoost:** {xgb_prob * 100:.2f}% ({1 / xgb_prob:.2f}) pravděpodobnost Over 2.5 → {'✅ ANO' if xgb_pred else '❌ NE'}")
            st.markdown(f"Confidence: {get_confidence(xgb_prob)} (threshold: {xgb_thresh:.2f})")
            st.markdown("---")
            
            # === Definice Bayesovského modelu ===
            class BayesianMLP(PyroModule):
                def __init__(self, in_features, hidden_size=64, dropout_rate=0.2):
                    super().__init__()
                    self.fc1 = PyroModule[nn.Linear](in_features, hidden_size)
                    self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_size, in_features]).to_event(2))
                    self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_size]).to_event(1))
                    self.dropout = nn.Dropout(p=dropout_rate)
                    self.out = PyroModule[nn.Linear](hidden_size, 1)
                    self.out.weight = PyroSample(dist.Normal(0., 1.).expand([1, hidden_size]).to_event(2))
                    self.out.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))
                    self.sigmoid = nn.Sigmoid()

                def forward(self, x, y=None):
                    x = torch.relu(self.fc1(x))
                    x = self.dropout(x)
                    logits = self.out(x).squeeze(-1)
                    probs = self.sigmoid(logits)
                    with pyro.plate("data", x.shape[0]):
                        pyro.sample("obs", dist.Bernoulli(probs), obs=y)
                    return probs
            # === Načtení feature definice ===
            with open(f"models/{league_code}_bayes_feature_groups.json", "r") as f:
                feature_groups = json.load(f)
                feature_cols = feature_groups["feature_cols"]
                binary_cols = feature_groups["binary_cols"]
                expected_features = feature_cols + binary_cols

            # === Doplnění chybějících sloupců ===
            for col in expected_features:
                if col not in match_row.columns:
                    match_row[col] = 0

            # === Sjednocení a řazení sloupců ===
            X_inputB = match_row[expected_features].copy().fillna(0)
            
            # === Urovnání pořadí ===
            X_inputB = X_inputB[expected_features]

            # === Načtení scaleru a škálování ===
            scaler = joblib.load(f"models/{league_code}_bayes_scaler.joblib")
            X_scaled_part = scaler.transform(X_inputB[feature_cols].values)
            X_scaled = np.concatenate([X_scaled_part, X_inputB[binary_cols].values], axis=1)

            # === Převod na tensor ===
            x_tensor = torch.tensor(X_scaled, dtype=torch.float32).reshape(1, -1)

            # === Inicializace modelu ===
            model = BayesianMLP(in_features=x_tensor.shape[1], hidden_size=64, dropout_rate=0.2)

            # === Načtení guide ===
            with open(f"models/{league_code}_bayes_guide.pkl", "rb") as f:
                guide = dill.load(f)

            # === Predikce ===
            predictive = Predictive(model, guide=guide, num_samples=2000, return_sites=["obs"])
            samples = predictive(x_tensor)
            y_pred_samples = samples["obs"].float().numpy()

            mean_prob = y_pred_samples.mean()
            lower_ci = np.percentile(y_pred_samples, 10)
            upper_ci = np.percentile(y_pred_samples, 90)
            

            # === Funkce pro interpretaci výsledku ===
            def classify_with_uncertainty(mean, lower, upper, threshold=0.5, tolerance=0.15):
                if upper < threshold - tolerance:
                    return "Under (s vysokou důvěrou)"
                elif lower > threshold + tolerance:
                    return "Over (s vysokou důvěrou)"
                else:
                    return "Neurčité / Vyrovnané"
            # Převod na skalár:
            if isinstance(mean_prob, np.ndarray):
                mean_prob = float(mean_prob) if mean_prob.size == 1 else mean_prob  # nebo .item()
                
            lower_ci = float(lower_ci)
            upper_ci = float(upper_ci)
            # === Výstup pro jeden zápas ===
            st.subheader("📊 Výsledek Bayesovského modelu")
            st.markdown(f"**Pravděpodobnost Over 2.5**: {mean_prob * 100:.2f}%")
            st.markdown(f"**Interval spolehlivosti (10–90 %)**: {lower_ci * 100:.1f}% – {upper_ci * 100:.1f}%")
            st.markdown(f"🧠 **Klasifikace**: {classify_with_uncertainty(mean_prob, lower_ci, upper_ci)}")
            st.write(X_inputB)

            plt.hist(y_pred_samples.flatten(), bins=50)
            plt.title("Distribuce predikovaných pravděpodobností")
            plt.show()
            # === Analýza na validační sadě ===
            X_val_scaled = scaler.transform(X_inputB[feature_cols].values)
            x_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float)
            samples_val = predictive(x_val_tensor)
            y_val_pred_samples = samples_val["obs"].float().numpy()
            bayes_probs = y_val_pred_samples.mean(axis=0)
            ci_widths = np.percentile(y_val_pred_samples, 90, axis=0) - np.percentile(y_val_pred_samples, 10, axis=0)
            bayes_preds = (bayes_probs > 0.5).astype(int)
            # === 📊 Nový scatter plot – vztah mezi jistotou a přesností ===
            plot_confidence_vs_accuracy(y_val_pred_samples, y_val.values)


            # === Vyhodnocení ===
            st.subheader("🧪 Evaluace Bayesovského modelu")
            st.text("Classification report:")
            st.text(classification_report(y_val, bayes_preds))
            st.text("Confusion matrix:")
            st.text(confusion_matrix(y_val, bayes_preds))
            st.text(f"Balanced accuracy: {balanced_accuracy_score(y_val, bayes_preds):.3f}")
            st.text(f"📏 Průměrná šířka CI (80%): {ci_widths.mean():.3f}")

            # === Kalibrační křivka ===
            prob_true, prob_pred = calibration_curve(y_val, bayes_probs, n_bins=10)
            fig1, ax1 = plt.subplots()
            ax1.plot(prob_pred, prob_true, marker="o", label="Bayes")
            ax1.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax1.set_title("Kalibrační křivka")
            ax1.set_xlabel("Predikovaná pravděpodobnost")
            ax1.set_ylabel("Skutečný podíl pozitivních")
            ax1.legend()
            st.pyplot(fig1)

            # === ROC křivka ===
            fpr, tpr, _ = roc_curve(y_val, bayes_probs)
            roc_auc = auc(fpr, tpr)
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"Bayes (AUC = {roc_auc:.2f})")
            ax2.plot([0, 1], [0, 1], "k--")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("ROC křivka Bayes modelu")
            ax2.legend()
            st.pyplot(fig2)

            # === Největší přehmaty modelu s malým intervalem ===
            df_val = pd.DataFrame({
                "true": y_val,
                "pred": bayes_preds,
                "prob": bayes_probs,
                "ci_width": ci_widths
            })
            bad_cases = df_val[(df_val["true"] != df_val["pred"]) & (df_val["ci_width"] < 0.2)]
            if not bad_cases.empty:
                st.subheader("⚠️ Falešné predikce s vysokou jistotou")
                st.dataframe(bad_cases.head(10))


            
            

    except Exception as e:
        st.error(f"Chyba při predikci: {e}")
