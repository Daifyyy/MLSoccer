import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "E0_combined_full.csv")

def load_data():
    return pd.read_csv(DATA_PATH)

def get_teams(df):
    teams = sorted(set(df["HomeTeam"]).union(df["AwayTeam"]))
    return teams

def filter_team_matches(df, team):
    return df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]

def filter_h2h_matches(df, home, away):
    return df[((df["HomeTeam"] == home) & (df["AwayTeam"] == away)) |
              ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))]