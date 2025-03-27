import pandas as pd
import numpy as np
from datetime import datetime

def calculate_elo(team_elo, home_team, away_team, home_goals, away_goals, k=20):
    expected_home = 1 / (1 + 10 ** ((team_elo.get(away_team, 1500) - team_elo.get(home_team, 1500)) / 400))
    result = 1 if home_goals > away_goals else 0 if home_goals < away_goals else 0.5
    change = k * (result - expected_home)
    team_elo[home_team] = team_elo.get(home_team, 1500) + change
    team_elo[away_team] = team_elo.get(away_team, 1500) - change
    return team_elo

def generate_extended_features(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    team_elo = {}
    rows = []

    for idx, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        match_date = row['Date']

        past_matches = df[df['Date'] < match_date]
        home_matches = past_matches[(past_matches['HomeTeam'] == home) | (past_matches['AwayTeam'] == home)].tail(6)
        away_matches = past_matches[(past_matches['HomeTeam'] == away) | (past_matches['AwayTeam'] == away)].tail(6)

        home_home_matches = past_matches[past_matches['HomeTeam'] == home].tail(5)
        away_away_matches = past_matches[past_matches['AwayTeam'] == away].tail(5)

        def avg_stat(matches, team, col):
            home_vals = matches[matches['HomeTeam'] == team][col]
            away_vals = matches[matches['AwayTeam'] == team][col]
            return pd.concat([home_vals, away_vals]).mean()

        features = {
            'HS_form_home_form': avg_stat(home_matches, home, 'HS'),
            'HS_form_away_form': avg_stat(away_matches, away, 'HS'),
            'AS_form_home_form': avg_stat(home_matches, home, 'AS'),
            'AS_form_away_form': avg_stat(away_matches, away, 'AS'),
            'HST_form_home_form': avg_stat(home_matches, home, 'HST'),
            'HST_form_away_form': avg_stat(away_matches, away, 'HST'),
            'AST_form_home_form': avg_stat(home_matches, home, 'AST'),
            'AST_form_away_form': avg_stat(away_matches, away, 'AST'),
            'HC_form_home_form': avg_stat(home_matches, home, 'HC'),
            'HC_form_away_form': avg_stat(away_matches, away, 'HC'),
            'AC_form_home_form': avg_stat(home_matches, home, 'AC'),
            'AC_form_away_form': avg_stat(away_matches, away, 'AC'),
            'goals_scored_home_form': avg_stat(home_matches, home, 'FTHG'),
            'goals_scored_away_form': avg_stat(away_matches, away, 'FTAG'),
            'goals_conceded_home_form': avg_stat(home_matches, home, 'FTAG'),
            'goals_conceded_away_form': avg_stat(away_matches, away, 'FTHG'),

            # Nové featury – domácí a venkovní forma
            'avg_goals_scored_home_last5': home_home_matches['FTHG'].mean(),
            'avg_goals_conceded_home_last5': home_home_matches['FTAG'].mean(),
            'avg_goals_scored_away_last5': away_away_matches['FTAG'].mean(),
            'avg_goals_conceded_away_last5': away_away_matches['FTHG'].mean(),
            'avg_shots_home_last5': home_home_matches['HS'].mean(),
            'avg_shots_on_target_home_last5': home_home_matches['HST'].mean(),
            'avg_shots_away_last5': away_away_matches['AS'].mean(),
            'avg_shots_on_target_away_last5': away_away_matches['AST'].mean(),
        }

        for key in list(features.keys()):
            if key.endswith('_home_form'):
                base = key.replace('_home_form', '')
                away_key = f'{base}_away_form'
                diff_key = f'{base}_diff'
                features[diff_key] = features[key] - features.get(away_key, 0)

        def calc_over25_ratio(matches):
            over25 = matches[(matches['FTHG'] + matches['FTAG']) > 2.5]
            return len(over25) / 6 if len(matches) >= 1 else 0.0

        features['over25_form_ratio_home'] = calc_over25_ratio(home_matches)
        features['over25_form_ratio_away'] = calc_over25_ratio(away_matches)

        features['elo_rating_home'] = team_elo.get(home, 1500)
        features['elo_rating_away'] = team_elo.get(away, 1500)

        if pd.notnull(match_date):
            days_since = (datetime.now() - match_date).days
            features['match_weight'] = 1 / (days_since + 1)
        else:
            features['match_weight'] = 1.0

        row_features = row.to_dict()
        row_features.update(features)
        rows.append(row_features)

        if not pd.isnull(row['FTHG']) and not pd.isnull(row['FTAG']):
            team_elo = calculate_elo(team_elo, home, away, row['FTHG'], row['FTAG'])

    return pd.DataFrame(rows)
