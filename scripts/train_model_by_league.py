import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from utils.data_loader import load_data_by_league
from utils.feature_engineering_extended import generate_extended_features
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from pyro.nn import PyroModule, PyroSample
from sklearn.preprocessing import StandardScaler
import dill

# === Vstup od uživatele ===
league_code = input("Zadej zkratku ligy (např. E0 nebo SP1): ")

# === Načtení surových dat ===
df = load_data_by_league(league_code)

# === Rozdělení dat na train/test před feature engineeringem ===
df = df.sort_values("Date")
split_index = int(len(df) * 0.8)
df_train = df.iloc[:split_index]
df_test = df.iloc[split_index:]

# === Vygenerování featur ===
df_train_ext = generate_extended_features(df_train, mode="train")
df_test_ext = generate_extended_features(df_test, mode="train")

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


X_train = df_train_ext[features].fillna(0)
y_train = df_train_ext["Over_2.5"]
w_train = df_train_ext["match_weight"].fillna(1.0)



X_test = df_test_ext[features].fillna(0)
y_test = df_test_ext["Over_2.5"]
w_test = df_test_ext["match_weight"].fillna(1.0)
ratio = (y_train == 0).sum() / (y_train == 1).sum()

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# === Ruční ladění Random Forest ===
best_rf_model = None
best_rf_score = 0
best_rf_params = {}

print("\n🔍 Ladění Random Forest")
for depth in [5, 10,15]:
    for estimators in [100, 200,300]:
        rf = RandomForestClassifier(max_depth=depth,class_weight="balanced"  , n_estimators=estimators, random_state=42)
        rf.fit(X_train_resampled, y_train_resampled)
        preds = rf.predict(X_test)
        score = f1_score(y_test, preds)
        #print(f"RF: depth={depth}, estimators={estimators} → F1: {score:.4f}")

        if score > best_rf_score:
            best_rf_model = rf
            best_rf_score = score
            best_rf_params = {"max_depth": depth, "n_estimators": estimators}

print("\n✅ Nejlepší RF parametry:", best_rf_params)
print("Random Forest – výstup na testovací sadě:")
print(classification_report(y_test, best_rf_model.predict(X_test)))
print("Confusion matrix (RF):")
print(confusion_matrix(y_test, best_rf_model.predict(X_test)))

# === Ruční ladění XGBoost ===
best_xgb_model = None
best_xgb_score = 0
best_xgb_params = {}

print("\n🔍 Ladění XGBoost")
for depth in [3, 6,10]:
    for lr in [0.05, 0.1,0.2]:
        for estimators in [100, 200]:
            xgb = XGBClassifier(scale_pos_weight=ratio, eval_metric='logloss', max_depth=depth,
                                learning_rate=lr, n_estimators=estimators, random_state=42)
            xgb.fit(X_train, y_train, sample_weight=w_train)
            preds = xgb.predict(X_test)
            score = f1_score(y_test, preds)
            #print(f"XGB: depth={depth}, lr={lr}, estimators={estimators} → F1: {score:.4f}")

            if score > best_xgb_score:
                best_xgb_model = xgb
                best_xgb_score = score
                best_xgb_params = {"max_depth": depth, "learning_rate": lr, "n_estimators": estimators}



# === Vytvoření tensorů ===
# === Vytvoření tensorů ===
assert not X_train.isnull().any().any()
assert not y_train.isnull().any()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_tensor = torch.tensor(X_train_scaled, dtype=torch.float)
y_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float)

print(X_train_scaled[:5])




# === Bayesovský model ===
class BayesianMLP(PyroModule):
    def __init__(self, in_features, hidden_size=32):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](in_features, hidden_size)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_size, in_features]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_size]).to_event(1))

        self.out = PyroModule[nn.Linear](hidden_size, 1)
        self.out.weight = PyroSample(dist.Normal(0., 1.).expand([1, hidden_size]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y=None):
        x = x.clone().detach() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float)

        x = torch.relu(self.fc1(x))
        logits = self.out(x).squeeze(-1)
        probs = self.sigmoid(logits)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Bernoulli(probs), obs=y)
        return probs

model = BayesianMLP(X_train_scaled.shape[1])

guide = pyro.infer.autoguide.AutoDiagonalNormal(model)
optimizer = ClippedAdam({"lr": 0.01})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

X_tensor = torch.tensor(X_train_scaled, dtype=torch.float)
y_tensor = torch.tensor(y_train.values, dtype=torch.float)

print("\n🔍 Trénink Bayesovského modelu")
pyro.clear_param_store()
for step in range(1000):
    loss = svi.step(X_tensor, y_tensor)
    if step % 100 == 0:
        print(f"Iterace {step} - Ztráta: {loss:.2f}")

# === Uložení ===
pyro.get_param_store().save(f"models/{league_code}_bayes_params.pt")
with open(f"models/{league_code}_bayes_guide.pkl", "wb") as f:
    dill.dump(guide, f)
joblib.dump(scaler, f"models/{league_code}_bayes_scaler.joblib")

print("✅ Bayesovský model byl natrénován a uložen.")



print("\n✅ Nejlepší XGBoost parametry:", best_xgb_params)
print("XGBoost – výstup na testovací sadě:")
print(classification_report(y_test, best_xgb_model.predict(X_test)))
print("Confusion matrix (XGB):")
print(confusion_matrix(y_test, best_xgb_model.predict(X_test)))
print(y_train.value_counts())

# === Uložení modelů ===
os.makedirs("models", exist_ok=True)
joblib.dump(best_rf_model, f"models/{league_code}_rf_model.joblib")
joblib.dump(best_xgb_model, f"models/{league_code}_xgb_model.joblib")
print("\n✅ Nejlepší modely byly uloženy.")
