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

def generate_extended_features(df, mode="train"):
    df = df.copy()

    if mode == "train":
        df = df[df['FTHG'].notnull() & df['FTAG'].notnull()]
    else:
        df['FTHG'] = np.nan
        df['FTAG'] = np.nan

    # Výpočet Elo
    elo = calculate_elo_rating(df)
    df['elo_rating_home'], df['elo_rating_away'] = zip(*elo)

    if mode == "train":
        df['Over_2.5'] = (df['FTHG'] + df['FTAG']) > 2.5

    for team_type in ['HomeTeam', 'AwayTeam']:
        side = 'home' if team_type == 'HomeTeam' else 'away'
    
        df[f'{side}_goals'] = df.apply(lambda row: row['FTHG'] if side == 'home' else row['FTAG'], axis=1)
        df[f'{side}_shots'] = df.apply(lambda row: row['HS'] if side == 'home' else row['AS'], axis=1)
        df[f'{side}_shots_on_target'] = df.apply(lambda row: row['HST'] if side == 'home' else row['AST'], axis=1)
        df[f'{side}_corners'] = df.apply(lambda row: row['HC'] if side == 'home' else row['AC'], axis=1)
        df[f'{side}_fouls'] = df.apply(lambda row: row['HF'] if side == 'home' else row['AF'], axis=1)
        df[f'{side}_cards'] = df.apply(lambda row: row['HY'] + row['HR'] if side == 'home' else row['AY'] + row['AR'], axis=1)
        df[f'{side}_conceded'] = df.apply(lambda row: row['FTAG'] if side == 'home' else row['FTHG'], axis=1)
    
        for metric in ['goals', 'conceded', 'shots', 'shots_on_target', 'corners', 'fouls', 'cards']:
            col_name = f'{metric}_{side}_last5'
            df[col_name] = (
                df.groupby(team_type)[f'{side}_{metric}']
                  .transform(lambda x: x.shift().rolling(window=6, min_periods=1).mean())
            )

        df["average_scored_goals"] = (df["goals_home_last5"] + df["goals_away_last5"]) / 2
        df["average_conceded_goals"] = (df["conceded_home_last5"] + df["conceded_away_last5"]) / 2
    
        df['goal_diff_last5'] = df['goals_home_last5'] - df['goals_away_last5']
        df['shot_diff_last5m'] = df['shots_home_last5'] - df['shots_away_last5']
        df['shot_on_target_diff_last5'] = df['shots_on_target_home_last5'] - df['shots_on_target_away_last5']
        df['corner_diff_last5'] = df['corners_home_last5'] - df['corners_away_last5']
        df['fouls_diff'] = df['fouls_home_last5'] - df['fouls_away_last5']
        df['card_diff'] = df['cards_home_last5'] - df['cards_away_last5']
    
    # 🛠️ Fallback pro chybějící sloupce a NaN hodnoty
    needed_last5_cols = [
        "goals_home_last5", "goals_away_last5",
        'shots_home_last5', 'shots_away_last5',
        'shots_on_target_home_last5', 'shots_on_target_away_last5',
        'corners_home_last5', 'corners_away_last5',
        'fouls_home_last5', 'fouls_away_last5',
        'cards_home_last5', 'cards_away_last5',
        'conceded_home_last5', 'conceded_away_last5',
    ]
    
    for col in needed_last5_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].fillna(0)

        
    
        
        


        # Přidání počtu zápasů s under 2.5 za posledních 5 zápasů
        if mode == "train":
            for team_type in ['HomeTeam', 'AwayTeam']:
                side = 'home' if team_type == 'HomeTeam' else 'away'
                df[f'{side}_under25_last5'] = (
                    df.groupby(team_type)
                      .apply(lambda group: ((group['FTHG'] + group['FTAG']) <= 2).shift().rolling(6, min_periods=1).sum())
                      .reset_index(level=0, drop=True)
                )
        else:
            # fallback – podíl předchozích zápasů s málo střelami jako náhradní odhad
            df["home_under25_last5"] = (
                ((df["shots_home_last5"].fillna(0) + df["shots_away_last5"].fillna(0)) < 15)
                .astype(int)
            )
            df["away_under25_last5"] = (
                ((df["shots_home_last5"].fillna(0) + df["shots_away_last5"].fillna(0)) < 15)
                .astype(int)
            )
        
        # Nové: forma doma/venku za poslední 3 zápasy (střely, xG)
        if side == 'home':
            df[f'{side}_form_shots'] = df.groupby(team_type)['HS'].transform(lambda x: x.shift().rolling(window=3, min_periods=1).mean())
            df[f'{side}_form_xg'] = df.groupby(team_type).apply(
                lambda g: (g['HS'] * 0.09 + g['HST'] * 0.2).shift().rolling(window=3, min_periods=1).mean()
            ).reset_index(level=0, drop=True)
        else:
            df[f'{side}_form_shots'] = df.groupby(team_type)['AS'].transform(lambda x: x.shift().rolling(window=3, min_periods=1).mean())
            df[f'{side}_form_xg'] = df.groupby(team_type).apply(
                lambda g: (g['AS'] * 0.09 + g['AST'] * 0.2).shift().rolling(window=3, min_periods=1).mean()
            ).reset_index(level=0, drop=True)
            
        # Samostatné metriky pro domácí zápasy domácího týmu a venkovní zápasy hostujícího týmu
        if mode == "train":
            # Domácí zápasy pro HomeTeam
            df['home_avg_goals_last5_home'] = (
                df[df['HomeTeam'] == df['HomeTeam']]
                .groupby('HomeTeam')['FTHG']
                .transform(lambda x: x.shift().rolling(window=6, min_periods=1).mean())
            )
        
            # Venkovní zápasy pro AwayTeam
            df['away_avg_goals_last5_away'] = (
                df[df['AwayTeam'] == df['AwayTeam']]
                .groupby('AwayTeam')['FTAG']
                .transform(lambda x: x.shift().rolling(window=6, min_periods=1).mean())
            )
        else:
            # fallback pro prediction
            df["home_avg_goals_last5_home"] = df["goals_home_last5"].fillna(0)
            df["away_avg_goals_last5_away"] = df["goals_away_last5"].fillna(0)


    

    df['home_xg'] = df['HS'] * 0.09 + df['HST'] * 0.2
    df['away_xg'] = df['AS'] * 0.09 + df['AST'] * 0.2
    df['xg_home_last5'] = df.groupby('HomeTeam')['home_xg'].transform(lambda x: x.shift().rolling(window=6, min_periods=1).mean())
    df['xg_away_last5'] = df.groupby('AwayTeam')['away_xg'].transform(lambda x: x.shift().rolling(window=6, min_periods=1).mean())
    if mode == "predict":
        df.loc[(df['home_xg'].isna()) | (df['home_xg'] == 0), 'home_xg'] = df['xg_home_last5']
        df.loc[(df['away_xg'].isna()) | (df['away_xg'] == 0), 'away_xg'] = df['xg_away_last5']


    if mode == "train":
        df['boring_match_score'] = (
            (df['HS'].fillna(0) + df['AS'].fillna(0)) * 0.05 +
            (df['HST'].fillna(0) + df['AST'].fillna(0)) * 0.1
        )
    else:
        df['boring_match_score'] = (
            (df['shots_home_last5'].fillna(0) + df['shots_away_last5'].fillna(0)) * 0.05 +
            (df['shots_on_target_home_last5'].fillna(0) + df['shots_on_target_away_last5'].fillna(0)) * 0.1
        )


    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        df['match_weight'] = df['Date'].apply(
            lambda date: 1 / (np.log((datetime.now() - date).days + 2)) if pd.notnull(date) else 1.0
        )
    else:
        df['match_weight'] = 1.0

    if mode == "train":
        df['h2h_goal_avg'] = df.apply(lambda row: (
            df[
                (((df['HomeTeam'] == row['HomeTeam']) & (df['AwayTeam'] == row['AwayTeam'])) |
                 ((df['HomeTeam'] == row['AwayTeam']) & (df['AwayTeam'] == row['HomeTeam'])))
                & (df.index < row.name)  # použij index přímo, místo get_loc()
            ][['FTHG', 'FTAG']].sum(axis=1)
            .rolling(window=5, min_periods=1).mean().iloc[-1]
        ) if row.name >= 1 else np.nan, axis=1)
    
    else:
        # fallback pro predikci, použij průměr vstřelených gólů týmů
        df['h2h_goal_avg'] = (
            (df['goals_home_last5'] + df['goals_away_last5']) / 2
        )


    # Momentum score (vždy se vytvoří, fallback přes fillna)
    df["momentum_score"] = (
        (df["elo_rating_home"].fillna(0) - df["elo_rating_away"].fillna(0)) +
        (df["xg_home_last5"].fillna(0) - df["xg_away_last5"].fillna(0))
    )

    # Defensive stability (rovněž vytvoř vždy)
    df["defensive_stability"] = (
        df["conceded_home_last5"].fillna(0) + df["conceded_away_last5"].fillna(0)
    ) / 2


    
    # Pokročilé metriky – záleží na režimu
    if mode == "train":
        df["shooting_efficiency"] = (df["HST"] + df["AST"]) / (df["HS"] + df["AS"] + 1)
        df["tempo_score"] = df["HS"] + df["AS"] + df["HC"] + df["AC"]
        df["passivity_score"] = (
            (df["HS"] + df["AS"]) * 0.05 +
            (df["HC"] + df["AC"]) * 0.1 +
            (df["home_xg"] + df["away_xg"]) * 0.15
        )
    else:
        df["shooting_efficiency"] = (df["shots_on_target_home_last5"] + df["shots_on_target_away_last5"]) / (df["shots_home_last5"] + df["shots_away_last5"] + 1)
        df["tempo_score"] = df["shots_home_last5"] + df["shots_away_last5"] + df["corners_home_last5"] + df["corners_away_last5"]
        df["passivity_score"] = (
            (df["shots_home_last5"] + df["shots_away_last5"]) * 0.05 +
            (df["corners_home_last5"] + df["corners_away_last5"]) * 0.1 +
            (df["xg_home_last5"] + df["xg_away_last5"]) * 0.15
        )
    df.to_csv("debug_features.csv", index=False)
    return df
