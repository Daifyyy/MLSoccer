import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_LEAGUE = "E0"
DATA_PATH = os.path.join(DATA_DIR, f"{DEFAULT_LEAGUE}_combined_full.csv")

# ✅ Původní funkce
def load_data():
    return pd.read_csv(DATA_PATH)

def get_teams(df):
    home_teams = df["HomeTeam"].unique()
    away_teams = df["AwayTeam"].unique()
    return sorted(set(home_teams) | set(away_teams))

def load_data_by_league(league_code):
    path = os.path.join(DATA_FOLDER, f"{league_code}_combined_full.csv")
    return pd.read_csv(path)

def filter_team_matches(df, team):
    return df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]

def filter_h2h_matches(df, home, away):
    return df[((df["HomeTeam"] == home) & (df["AwayTeam"] == away)) |
              ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))]

# ✅ Nové funkce pro app.py
def get_available_leagues():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('_combined_full.csv')]
    return [f.split('_')[0] for f in files]

def get_teams_by_league(league_code):
    file_path = os.path.join(DATA_DIR, f"{league_code}_combined_full.csv")
    df = pd.read_csv(file_path)
    teams = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))
    return teams

def load_combined_data(league_code):
    file_path = os.path.join(DATA_DIR, f"{league_code}_combined_full.csv")
    return pd.read_csv(file_path)

# 📂 Dynamicky načti CSV pro danou ligu (např. E0 → E0_combined_full.csv)
def load_data_by_league(league_code):
    path = os.path.join(DATA_DIR, f"{league_code}_combined_full.csv")
    return pd.read_csv(path)
# 📊 Zápasy vybraného týmu
def filter_team_matches(df, team):
    return df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]

# 📊 H2H zápasy mezi dvěma týmy
def filter_h2h_matches(df, home, away):
    return df[((df["HomeTeam"] == home) & (df["AwayTeam"] == away)) |
              ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))]