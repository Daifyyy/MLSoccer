import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "E0_combined_full.csv")

def load_data():
    return pd.read_csv(DATA_PATH)

def get_teams(df):
    home_teams = df["HomeTeam"].unique()
    away_teams = df["AwayTeam"].unique()
    return sorted(set(home_teams) | set(away_teams))


def filter_team_matches(df, team):
    return df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]

def filter_h2h_matches(df, home, away):
    return df[((df["HomeTeam"] == home) & (df["AwayTeam"] == away)) |
              ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))]
    
def load_combined_data(league_code):
    file_path = f"data/{league_code}_combined_full.csv"
    df = pd.read_csv(file_path)
    return df