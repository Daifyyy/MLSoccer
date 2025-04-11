
import pandas as pd
import joblib
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from utils.data_loader import load_data_by_league
from utils.feature_engineering_team_stats import generate_team_stats_features


def train_team_stat_models(league_code):
    df = load_data_by_league(league_code)
    df_features = generate_team_stats_features(df, mode="train")

    output_dir = f"models/{league_code}_team_stats"
    os.makedirs(output_dir, exist_ok=True)

    target_cols = [col for col in df_features.columns if col.startswith("target_")]
    features = [col for col in df_features.columns if col not in target_cols + ["HomeTeam", "AwayTeam", "Date"]]

    results = []

    for target in target_cols:
        X = df_features[features].fillna(0)
        y = df_features[target].fillna(0)

        split_index = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"✅ {target} – MSE: {mse:.2f}")

        # Nový kód s přidáním league_code do názvu souboru
        joblib.dump(model, os.path.join(output_dir, f"{league_code}_{target}_model.joblib"))

        # Ukládání MSE nebo jiných výsledků
        results.append((f"{league_code}_{target}", mse))

    return results

if __name__ == "__main__":
    league_code = input("Zadej ligu (např. SP1 nebo E0): ")
    train_team_stat_models(league_code)
