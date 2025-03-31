import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

# === Pomocné funkce ===
def calculate_elo_rating(df):
    elo_ratings = defaultdict(lambda: 1500)
    k = 20
    ratings = []

    for _, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        home_elo = elo_ratings[home]
        away_elo = elo_ratings[away]

        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        expected_away = 1 - expected_home

        if row['FTHG'] > row['FTAG']:
            score_home, score_away = 1, 0
        elif row['FTHG'] < row['FTAG']:
            score_home, score_away = 0, 1
        else:
            score_home, score_away = 0.5, 0.5

        elo_ratings[home] += k * (score_home - expected_home)
        elo_ratings[away] += k * (score_away - expected_away)

        ratings.append((home_elo, away_elo))

    return ratings

# === Hlavní funkce ===
def generate_extended_features(df):
    df = df.copy()
    df = df[df['FTHG'].notnull() & df['FTAG'].notnull()]

    # Výpočet Elo
    elo = calculate_elo_rating(df)
    df['elo_rating_home'], df['elo_rating_away'] = zip(*elo)

    # Over 2.5
    df['Over_2.5'] = (df['FTHG'] + df['FTAG']) > 2.5

    # Gólové průměry (celkem, za posledních 5 zápasů)
    for team_type in ['HomeTeam', 'AwayTeam']:
        side = 'home' if team_type == 'HomeTeam' else 'away'

        df[f'{side}_goals'] = df.apply(lambda row: row['FTHG'] if side == 'home' else row['FTAG'], axis=1)
        df[f'{side}_shots'] = df.apply(lambda row: row['HS'] if side == 'home' else row['AS'], axis=1)
        df[f'{side}_shots_on_target'] = df.apply(lambda row: row['HST'] if side == 'home' else row['AST'], axis=1)
        df[f'{side}_corners'] = df.apply(lambda row: row['HC'] if side == 'home' else row['AC'], axis=1)
        df[f'{side}_fouls'] = df.apply(lambda row: row['HF'] if side == 'home' else row['AF'], axis=1)
        df[f'{side}_cards'] = df.apply(lambda row: row['HY'] + row['HR'] if side == 'home' else row['AY'] + row['AR'], axis=1)

        for metric in ['goals', 'shots', 'shots_on_target', 'corners', 'fouls', 'cards']:
            df[f'{metric}_{side}_last5'] = (
                df.groupby(team_type)[f'{side}_{metric}']
                .transform(lambda x: x.shift().rolling(window=6, min_periods=1).mean())
            )

    # Diference statistik
    df['goal_diff_last5'] = df['goals_home_last5'] - df['goals_away_last5']
    df['shot_diff_last5'] = df['shots_home_last5'] - df['shots_away_last5']
    df['shot_on_target_diff_last5'] = df['shots_on_target_home_last5'] - df['shots_on_target_away_last5']
    df['corner_diff_last5'] = df['corners_home_last5'] - df['corners_away_last5']
    df['fouls_diff'] = df['fouls_home_last5'] - df['fouls_away_last5']
    df['card_diff'] = df['cards_home_last5'] - df['cards_away_last5']

    # xG na základě střel a střel na branku
    df['home_xg'] = df['HS'] * 0.09 + df['HST'] * 0.2
    df['away_xg'] = df['AS'] * 0.09 + df['AST'] * 0.2

    df['xg_home_last5'] = df.groupby('HomeTeam')['home_xg'].transform(lambda x: x.shift().rolling(window=6, min_periods=1).mean())
    df['xg_away_last5'] = df.groupby('AwayTeam')['away_xg'].transform(lambda x: x.shift().rolling(window=6, min_periods=1).mean())

    # Nudný zápas skóre (méně střel a gólů)
    df['boring_match_score'] = (
        (df['HS'] + df['AS']) * 0.1 + (df['HST'] + df['AST']) * 0.2) * 0.005
    

    # Váha zápasu dle stáří s logaritmickým poklesem
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['match_weight'] = df['Date'].apply(
            lambda date: 1 / (np.log((datetime.now() - date).days + 2)) if pd.notnull(date) else 1.0
        )
    else:
        df['match_weight'] = 1.0

    return df
