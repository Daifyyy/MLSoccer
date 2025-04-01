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
    
    df["HomeTeam"] = df["HomeTeam"].str.strip()
    df["AwayTeam"] = df["AwayTeam"].str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    
    df = df.sort_values(by='Date', ascending=True, na_position='last')

    
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
        df[f'{side}_shots'] = df.apply(lambda row: row['HS'] if side == 'home' else row['AS'], axis=1)
        df[f'{side}_shots_on_target'] = df.apply(lambda row: row['HST'] if side == 'home' else row['AST'], axis=1)
        df[f'{side}_corners'] = df.apply(lambda row: row['HC'] if side == 'home' else row['AC'], axis=1)
        df[f'{side}_fouls'] = df.apply(lambda row: row['HF'] if side == 'home' else row['AF'], axis=1)
        df[f'{side}_cards'] = df.apply(lambda row: row['HY'] + row['HR'] if side == 'home' else row['AY'] + row['AR'], axis=1)

    
    for metric in ['shots', 'shots_on_target', 'corners', 'fouls', 'cards']:
        for side, team_type in [('home', 'HomeTeam'), ('away', 'AwayTeam')]:
            col_name = f'{metric}_{side}_last5'
            base_col = f'{side}_{metric}'

            df[col_name] = (
                df.groupby(team_type)[base_col]
                .transform(lambda x: x.shift().rolling(window=6, min_periods=3).mean())
            )

            # Missing flag
            df[f'{col_name}_missing'] = df[col_name].isna().astype(int)
            df[col_name] = df[col_name].fillna(0)

    
    # Missing flags
    df['missing_shot_diff_last5m'] = df[['shots_home_last5', 'shots_away_last5']].isna().any(axis=1).astype(int)
    df['missing_shot_on_target_diff_last5'] = df[['shots_on_target_home_last5', 'shots_on_target_away_last5']].isna().any(axis=1).astype(int)
    df['missing_corner_diff_last5'] = df[['corners_home_last5', 'corners_away_last5']].isna().any(axis=1).astype(int)
    df['missing_fouls_diff'] = df[['fouls_home_last5', 'fouls_away_last5']].isna().any(axis=1).astype(int)
    df['missing_card_diff'] = df[['cards_home_last5', 'cards_away_last5']].isna().any(axis=1).astype(int)

    # Odvozené rozdílové metriky
    df['shot_diff_last5m'] = df['shots_home_last5'] - df['shots_away_last5']
    df['shot_on_target_diff_last5'] = df['shots_on_target_home_last5'] - df['shots_on_target_away_last5']
    df['corner_diff_last5'] = df['corners_home_last5'] - df['corners_away_last5']
    df['fouls_diff'] = df['fouls_home_last5'] - df['fouls_away_last5']
    df['card_diff'] = df['cards_home_last5'] - df['cards_away_last5']

        
    # Nové: forma doma/venku za poslední 3 zápasy (střely, xG)
    if side == 'home':
        df[f'{side}_form_shots'] = df.groupby(team_type)['HS'].transform(lambda x: x.shift().rolling(window=3, min_periods=3).mean())
        df[f'{side}_form_xg'] = df.groupby(team_type).apply(
            lambda g: (g['HS'] * 0.09 + g['HST'] * 0.2).shift().rolling(window=3, min_periods=3).mean()
        ).reset_index(level=0, drop=True)

            
    

        
    
    # 🛠️ Fallback pro chybějící sloupce a NaN hodnoty
    needed_last5_cols = [
        'shots_home_last5', 'shots_away_last5',
        'shots_on_target_home_last5', 'shots_on_target_away_last5',
        'corners_home_last5', 'corners_away_last5',
        'fouls_home_last5', 'fouls_away_last5',
        'cards_home_last5', 'cards_away_last5',
    ]
    
    for col in needed_last5_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].fillna(0)

        
    
        
        


        


    

    df['home_xg'] = df['HS'] * 0.09 + df['HST'] * 0.2
    df['away_xg'] = df['AS'] * 0.09 + df['AST'] * 0.2
    df = df.sort_values(by=['HomeTeam', 'Date'], ascending=[True, True], na_position='last')

    df['xg_home_last5'] = (
        df.groupby('HomeTeam')
        .apply(lambda g: g.sort_values('Date')
                            .assign(xg_home_last5 = g['home_xg'].shift().rolling(window=6, min_periods=1).mean()))
        .reset_index(drop=True)['xg_home_last5']
    )
    df = df.sort_values(by=['AwayTeam', 'Date'], ascending=[True, True], na_position='last')

    df['xg_away_last5'] = (
        df.groupby('AwayTeam')
        .apply(lambda g: g.sort_values('Date')
                            .assign(xg_away_last5 = g['away_xg'].shift().rolling(window=6, min_periods=1).mean()))
        .reset_index(drop=True)['xg_away_last5']
    )


    df['missing_xg_home_last5'] = df['xg_home_last5'].isna().astype(int)
    df['missing_xg_away_last5'] = df['xg_away_last5'].isna().astype(int)

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

    



    # Momentum score (vždy se vytvoří, fallback přes fillna)
    df["momentum_score"] = (
        (df["elo_rating_home"].fillna(0) - df["elo_rating_away"].fillna(0)) +
        (df["xg_home_last5"].fillna(0) - df["xg_away_last5"].fillna(0))
    )



    
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
