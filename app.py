import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay, precision_recall_curve, f1_score
from sklearn.model_selection import train_test_split
from utils.feature_engineering_extended import generate_extended_features
from utils.data_loader import load_data_by_league, filter_team_matches, filter_h2h_matches

st.set_page_config(layout="wide")
st.title("⚽ Predikce počtu gólů – Over 2.5")

league_code = st.selectbox(
    "Zadej zkratku ligy (např. E0 nebo SP1):",
    ["E0", "SP1", "D1", "N1", "I1", "T1", "F1"]
)


df_raw = load_data_by_league(league_code)

tems = sorted(set(df_raw["HomeTeam"]).union(set(df_raw["AwayTeam"])))
home_team = st.selectbox("Domácí tým:", tems)
away_team = st.selectbox("Hostující tým:", tems)

rf_threshold = 0.44
xgb_threshold = 0.45

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
    "away_avg_goals_last5_away",
    "home_goals",
    "away_goals"
]

if st.button("🔍 Spustit predikci"):
    try:
        # Načtení historických dat pro oba týmy
        df_filtered = pd.concat([
            filter_team_matches(df_raw, home_team),
            filter_team_matches(df_raw, away_team),
            filter_h2h_matches(df_raw, home_team, away_team)
        ]).drop_duplicates().reset_index(drop=True)

        # Přidání budoucího (fiktivního) zápasu bez známého výsledku
        future_match = pd.DataFrame([{
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'Date': pd.to_datetime(datetime.today().date()),
            # Zbytek sloupců vyplníme NaN, protože zatím neznáme výsledek ani statistiky
            'FTHG': np.nan,
            'FTAG': np.nan,
            'HS': np.nan,
            'AS': np.nan,
            'HST': np.nan,
            'AST': np.nan,
            'HF': np.nan,
            'AF': np.nan,
            'HC': np.nan,
            'AC': np.nan,
            'HY': np.nan,
            'AY': np.nan,
            'HR': np.nan,
            'AR': np.nan,
        }])

        df_pred_input = pd.concat([df_filtered, future_match], ignore_index=True)

        # Vytvoření rozšířených features pro predikci
        df_ext = generate_extended_features(df_pred_input, mode="predict")

        rf_model_path = f"models/{league_code}_rf_model.joblib"
        xgb_model_path = f"models/{league_code}_xgb_model.joblib"
        rf_model = joblib.load(rf_model_path)
        xgb_model = joblib.load(xgb_model_path)

        # Najdeme právě ten řádek s budoucím zápasem
        match_row = df_ext[
            (df_ext["HomeTeam"] == home_team) &
            (df_ext["AwayTeam"] == away_team) &
            (df_ext["Date"].dt.date == datetime.today().date())
        ]

        if match_row.empty:
            st.warning("⚠️ Nepodařilo se najít vstupní data pro predikci.")
        else:
            X_input = match_row[features].fillna(0)
            st.markdown("### 🔍 Použité featury pro predikci:")
            st.dataframe(X_input.T)  # transponované pro lepší čitelnost
            st.markdown(f"🧾 Počet zápasů pro generování features: {len(df_filtered)}")

            
            rf_prob = rf_model.predict_proba(X_input)[0][1]
            xgb_prob = xgb_model.predict_proba(X_input)[0][1]
            rf_pred = rf_prob > rf_threshold
            xgb_pred = xgb_prob > xgb_threshold

            st.subheader("📊 Výsledky predikce:")
            st.markdown(f"🎲 Random Forest – pravděpodobnost Over 2.5: **{rf_prob*100:.2f}%** → {'✅ ANO' if rf_pred else '❌ NE'}")
            st.markdown(f"🚀 XGBoost – pravděpodobnost Over 2.5: **{xgb_prob*100:.2f}%** → {'✅ ANO' if xgb_pred else '❌ NE'}")

    except Exception as e:
        st.error(f"❌ Nastala chyba během predikce: {e}")


# st.subheader("🤞 Analýza modelů")
# if st.checkbox("🔍 Zobrazit analýzu modelů na validačních datech"):
#     try:
#         df_train = load_data_by_league(league_code)
#         df_train_ext = generate_extended_features(df_train, mode="train")

#         X = df_train_ext[features].fillna(0)
#         y = df_train_ext["Over_2.5"]
#         w = df_train_ext["match_weight"].fillna(1.0)

#         X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
#             X, y, w, test_size=0.2, random_state=42
#         )

#         rf_model = joblib.load(f"models/{league_code}_rf_model.joblib")
#         xgb_model = joblib.load(f"models/{league_code}_xgb_model.joblib")

#         rf_probs = rf_model.predict_proba(X_test)[:, 1]
#         xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
#         rf_preds = rf_probs > rf_threshold
#         xgb_preds = xgb_probs > xgb_threshold

#         st.markdown("### 🎲 Random Forest – klasifikační metriky")
#         st.text(classification_report(y_test, rf_preds))

#         st.markdown("### 🚀 XGBoost – klasifikační metriky")
#         st.text(classification_report(y_test, xgb_preds))

#         st.markdown("### 🔁 Matice záměn – Random Forest")
#         cm_rf = confusion_matrix(y_test, rf_preds)
#         st.dataframe(pd.DataFrame(cm_rf, columns=["Pred: NE", "Pred: ANO"], index=["Skut.: NE", "Skut.: ANO"]))

#         st.markdown("### 🔁 Matice záměn – XGBoost")
#         cm_xgb = confusion_matrix(y_test, xgb_preds)
#         st.dataframe(pd.DataFrame(cm_xgb, columns=["Pred: NE", "Pred: ANO"], index=["Skut.: NE", "Skut.: ANO"]))

#         st.markdown("### 📈 ROC křivka")
#         fig, ax = plt.subplots(figsize=(3, 1))
#         RocCurveDisplay.from_predictions(y_test, rf_probs, name="Random Forest", ax=ax)
#         RocCurveDisplay.from_predictions(y_test, xgb_probs, name="XGBoost", ax=ax)
#         st.pyplot(fig)

#         st.markdown("### 📊 F1-skóre podle prahové hodnoty (Random Forest a XGBoost)")

#         thresholds = np.linspace(0.1, 0.9, 50)
#         f1_scores_rf = [f1_score(y_test, rf_probs > t) for t in thresholds]
#         f1_scores_xgb = [f1_score(y_test, xgb_probs > t) for t in thresholds]

#         fig, ax = plt.subplots(figsize=(3, 1))
#         ax.plot(thresholds, f1_scores_rf, marker='o', label="Random Forest", color="blue")
#         ax.plot(thresholds, f1_scores_xgb, marker='o', label="XGBoost", color="orange")
#         ax.set_xlabel("Práh")
#         ax.set_ylabel("F1-skóre")
#         ax.set_title("Optimalizace prahové hodnoty – F1-skóre")
#         ax.legend()
#         st.pyplot(fig)

#         precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_probs)
#         precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, xgb_probs)
#         fig, ax = plt.subplots(figsize=(3, 1))
#         ax.plot(recall_rf, precision_rf, label="Random Forest", color="blue")
#         ax.plot(recall_xgb, precision_xgb, label="XGBoost", color="orange")
#         ax.set_xlabel("Recall")
#         ax.set_ylabel("Precision")
#         ax.set_title("Precision-Recall Curve")
#         ax.legend()
#         st.pyplot(fig)

#         st.subheader("🔎 SHAP vysvětlení (Random Forest)")
#         explainer = shap.Explainer(rf_model, X_train)
#         shap_values = explainer(X_test[:1])
#         fig_shap = shap.plots.waterfall(shap_values[0], show=False)
#         st.pyplot(fig_shap)

#     except Exception as e:
#         st.error(f"Chyba v analytické sekci: {e}")
