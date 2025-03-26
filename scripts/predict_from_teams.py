import os
from utils.data_loader import load_data, get_teams, filter_team_matches, filter_h2h_matches
from utils.feature_engineering import compute_stats, aggregate_features
from utils.model_utils import load_model, predict_over25

# Načtení dat a modelu
df = load_data()
model = load_model()

# Výběr týmů
teams = get_teams(df)
print("\nDostupné týmy:")
print(", ".join(teams))
home = input("\nZadej domácí tým: ")
away = input("Zadej hostující tým: ")

# Výpočty statistik
home_matches = filter_team_matches(df, home)
away_matches = filter_team_matches(df, away)
h2h_matches = filter_h2h_matches(df, home, away)

home_form = compute_stats(home_matches.tail(5))
away_form = compute_stats(away_matches.tail(5))
h2h_form = compute_stats(h2h_matches)
overall_stats = compute_stats(df[
    (df["HomeTeam"] == home) | (df["AwayTeam"] == home) |
    (df["HomeTeam"] == away) | (df["AwayTeam"] == away)
])

# Spojení statistik
features = aggregate_features(home_form, away_form, h2h_form, overall_stats)

# Predikce
probability, prediction = predict_over25(model, features)

print("\nPravděpodobnost, že padne více než 2.5 gólu:")
print(f"{probability * 100:.2f}%")

if prediction == 1:
    print("✅ Model predikuje: OVER 2.5 goals")
else:
    print("❌ Model predikuje: UNDER 2.5 goals")
