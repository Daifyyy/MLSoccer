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
from torch.distributions import constraints
import torch.serialization
from torch import serialization
import dill

st.set_page_config(layout="wide")
st.title("⚽ Predikce Over 2.5 gólů se sílou sázkové příležitosti")

league_code = st.selectbox("Zvol ligu:", ["E0", "SP1", "D1", "I1", "F1"])

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
            st.markdown(f"**Random Forest:** {rf_prob:.2%} pravděpodobnost Over 2.5 → {'✅ ANO' if rf_pred else '❌ NE'}")
            st.markdown(f"Confidence: {get_confidence(rf_prob)} (threshold: {rf_thresh:.2f})")
            st.markdown("---")

            st.markdown(f"**XGBoost:** {xgb_prob:.2%} pravděpodobnost Over 2.5 → {'✅ ANO' if xgb_pred else '❌ NE'}")
            st.markdown(f"Confidence: {get_confidence(xgb_prob)} (threshold: {xgb_thresh:.2f})")
            st.markdown("---")
            
            # #   === Třetí model – Bayesovský přístup ===
            # class BayesianMLP(pyro.nn.PyroModule):
            #     def __init__(self, in_features, hidden_size=32):
            #         super().__init__()
            #         self.fc1 = pyro.nn.PyroModule[nn.Linear](in_features, hidden_size)
            #         self.fc1.weight = pyro.nn.PyroSample(dist.Normal(0., 1.).expand([hidden_size, in_features]).to_event(2))
            #         self.fc1.bias = pyro.nn.PyroSample(dist.Normal(0., 1.).expand([hidden_size]).to_event(1))
            #         self.out = pyro.nn.PyroModule[nn.Linear](hidden_size, 1)
            #         self.out.weight = pyro.nn.PyroSample(dist.Normal(0., 1.).expand([1, hidden_size]).to_event(2))
            #         self.out.bias = pyro.nn.PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))
            #         self.sigmoid = nn.Sigmoid()

            #     def forward(self, x, y=None):
            #         x = torch.relu(self.fc1(x))
            #         logits = self.out(x).squeeze(-1)
            #         probs = self.sigmoid(logits)
            #         with pyro.plate("data", x.shape[0]):
            #             obs = pyro.sample("obs", dist.Bernoulli(probs), obs=y)
            #         return probs

            # # === Načtení scaleru a vstupu ===
            # scaler = joblib.load(f"models/{league_code}_bayes_scaler.joblib")
            # X_scaled = scaler.transform(X_input)
            # x_tensor = torch.tensor(X_scaled, dtype=torch.float)

            # # === Inicializuj model a guide, nahraj parametry z param_store ===
            # model = BayesianMLP(x_tensor.shape[1])
            # # ✅ Povolení načítání constraintů – whitelisting _Real
            # serialization.add_safe_globals({
            #     "torch.distributions.constraints._Real": constraints.real
            # })

            
            # with open(f"models/{league_code}_bayes_guide.pkl", "rb") as f:
            #     guide = dill.load(f)
            # # === Predikce ===
            # predictive = Predictive(model, guide=guide, num_samples=1000)
            # samples = predictive(x_tensor)
            # mean_prob = samples["obs"].float().mean().item()

            # st.markdown(f"**Bayesovský model:** {mean_prob:.2%} pravděpodobnost Over 2.5")
            # st.markdown(f"Confidence: {get_confidence(mean_prob)}")
            

    except Exception as e:
        st.error(f"Chyba při predikci: {e}")
