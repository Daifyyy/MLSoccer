import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBRegressor
from utils.data_loader import load_data_by_league
from utils.feature_engineering_team_stats import generate_team_stats_features

def train_team_stats_models(league_code, nan_threshold=0.3):
    print(f"\U0001F3C6 Trénink modelů týmových statistik pro ligu {league_code}")
    df = load_data_by_league(league_code)

    df_train = df.iloc[:-int(len(df)*0.2)]
    df_test = df.iloc[-int(len(df)*0.2):]

    df_train_fe = generate_team_stats_features(df_train, mode="train")
    df_test_fe = generate_team_stats_features(df_test, mode="train")

    target_cols = [col for col in df_train_fe.columns if col.startswith("target_")]
    drop_cols = target_cols + ["HomeTeam", "AwayTeam", "Date"]

    X_train = df_train_fe.drop(columns=drop_cols).fillna(0)
    X_test = df_test_fe.drop(columns=drop_cols).fillna(0)

    os.makedirs(f"models/{league_code}_team_stats", exist_ok=True)

    total_rows = len(df_train_fe)

    for target in target_cols:
        y_train = df_train_fe[target]
        y_test = df_test_fe[target]

        n_nan = y_train.isna().sum()
        n_inf = (~np.isfinite(y_train)).sum()
        pct_nan = n_nan / total_rows
        pct_inf = n_inf / total_rows

        if pct_nan > nan_threshold or pct_inf > nan_threshold:
            print(f"⚠️ Přeskočeno {target}: {pct_nan*100:.1f}% NaN, {pct_inf*100:.1f}% inf")
            continue

        print(f"\n🎯 Trénuji model pro: {target} ({pct_nan*100:.1f}% NaN, {pct_inf*100:.1f}% inf)")

        valid_mask = y_train.notna() & np.isfinite(y_train)
        y_train_clean = y_train[valid_mask]
        X_train_clean = X_train.loc[valid_mask]

        if len(y_train_clean) == 0:
            print(f"❌ Žádná validní data pro {target}, model nebude trénován.")
            continue

        model = XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            objective="reg:squarederror",
            random_state=42
        )

        model.fit(X_train_clean, y_train_clean)

        preds = model.predict(X_test)
        mse = np.mean((preds - y_test) ** 2)
        print(f"MSE na testovací sadě pro {target}: {mse:.4f}")

        model_path = f"models/{league_code}_team_stats/{league_code}_{target}_model.joblib"
        joblib.dump(model, model_path)
        print(f"✅ Model uložen: {model_path}")

if __name__ == "__main__":
    league_list = ["E0", "E1", "SP1", "D1", "D2", "I1", "F1", "B1", "P1", "T1", "N1"]
    for league_code in league_list:
        try:
            train_team_stats_models(league_code)
        except Exception as e:
            print(f"❌ Chyba při tréninku pro ligu {league_code}: {e}")
    print("\n✅ Všechny ligy dokončeny.")