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
    invalid_dates = df[~df['Date'].astype(str).str.match(r'\d{4}-\d{2}-\d{2}')]


    # A pak vyhoď jen řádky s NaT až úplně na konci
    df = df.dropna(subset=["Date"])    
    df = df.sort_values(by='Date', ascending=True, na_position='last')
   
    
    
    #if mode == "train":
    #    df = df[df['FTHG'].notnull() & df['FTAG'].notnull()]
    #else:
    #    df['FTHG'] = np.nan
     #   df['FTAG'] = np.nan

    # Výpočet Elo
    elo = calculate_elo_rating(df)
    df['elo_rating_home'], df['elo_rating_away'] = zip(*elo)

    if mode == "train":
        df['Over_2.5'] = (df['FTHG'] + df['FTAG']) > 2.5
        
     # Výpočet odds features
    if 'Avg>2.5' in df.columns and 'Avg<2.5' in df.columns:
        df["prob_over25"] = 1 / df["Avg>2.5"]
        df["prob_under25"] = 1 / df["Avg<2.5"]
        df["over25_expectation_gap"] = df["prob_over25"] - df["prob_under25"]
        df["missing_odds_info"] = df[["prob_over25", "prob_under25"]].isna().any(axis=1).astype(int)
    else:
        df["prob_over25"] = np.nan
        df["prob_under25"] = np.nan
        df["over25_expectation_gap"] = np.nan
        df["missing_odds_info"] = 1

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
                .transform(lambda x: x.shift().rolling(window=6, min_periods=1).mean())
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

        
    # Forma týmů doma/venku (střely a xG za poslední 3 zápasy)
    df["home_form_shots"] = (
        df.groupby("HomeTeam")["HS"]
        .transform(lambda x: x.shift().rolling(window=5, min_periods=1).mean())
    )
    df["home_form_xg"] = (
        df.groupby("HomeTeam")
        .apply(lambda g: (g["HS"] * 0.09 + g["HST"] * 0.2).shift().rolling(window=5, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    df["missing_home_form_shots"] = df["home_form_shots"].isna().astype(int)
    df["missing_home_form_xg"] = df["home_form_xg"].isna().astype(int)
    df["home_form_shots"] = df["home_form_shots"].fillna(0)
    df["home_form_xg"] = df["home_form_xg"].fillna(0)

    df["away_form_shots"] = (
        df.groupby("AwayTeam")["AS"]
        .transform(lambda x: x.shift().rolling(window=5, min_periods=1).mean())
    )
    df["away_form_xg"] = (
        df.groupby("AwayTeam")
        .apply(lambda g: (g["AS"] * 0.09 + g["AST"] * 0.2).shift().rolling(window=5, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    df["missing_away_form_shots"] = df["away_form_shots"].isna().astype(int)
    df["missing_away_form_xg"] = df["away_form_xg"].isna().astype(int)
    df["away_form_shots"] = df["away_form_shots"].fillna(0)
    df["away_form_xg"] = df["away_form_xg"].fillna(0)

    if 'Avg<2.5' in df.columns:
        df["log_odds_under25"] = np.log(df["Avg<2.5"] + 1)
    else:
        df["log_odds_under25"] = np.nan
    df["missing_log_odds_under25"] = df["log_odds_under25"].isna().astype(int)
    df["log_odds_under25"] = df["log_odds_under25"].fillna(0)
        
    
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
        
    # Kombinace agresivity a pasivity obou týmů
    df["aggressiveness_score"] = df["fouls_diff"].abs() + df["card_diff"].abs()
    df["behavior_balance"] = df["passivity_score"].fillna(0) + df["aggressiveness_score"].fillna(0)
    
    

    for team_type in ['HomeTeam', 'AwayTeam']:
        side = 'home' if team_type == 'HomeTeam' else 'away'
        conceded_col = f'{side}_conceded'

        # Gól inkasovaný v zápase (jen pro train a predikci, kde je historie)
        df[conceded_col] = df.apply(
            lambda row: row['FTAG'] if side == 'home' else row['FTHG'], axis=1
        )

        conceded_last5 = f'conceded_{side}_last5'
        df[conceded_last5] = (
            df.groupby(team_type)[conceded_col]
              .transform(lambda x: x.shift().rolling(window=5, min_periods=1).mean())
        )

        # Missing flag
        df[f'missing_{conceded_last5}'] = df[conceded_last5].isna().astype(int)
        df[conceded_last5] = df[conceded_last5].fillna(0)
     
    df["defensive_stability"] = (df["conceded_home_last5"] + df["conceded_away_last5"]) / 2   
        
    df["xg_conceded_home_last5"] = df["shots_away_last5"] * 0.09 + df["shots_on_target_away_last5"] * 0.2
    df["xg_conceded_away_last5"] = df["shots_home_last5"] * 0.09 + df["shots_on_target_home_last5"] * 0.2
    df["avg_xg_conceded"] = (df["xg_conceded_home_last5"] + df["xg_conceded_away_last5"]) / 2
    df["xg_ratio"] = (df["xg_home_last5"] + df["xg_away_last5"]) / (df["avg_xg_conceded"] + 0.1)
    df["defensive_pressure"] = df["fouls_diff"] + df["card_diff"]
    
    df["missing_xg_conceded_home_last5"] = df["xg_conceded_home_last5"].isna().astype(int)
    df["missing_xg_conceded_away_last5"] = df["xg_conceded_away_last5"].isna().astype(int)
    df["missing_avg_xg_conceded"] = df["avg_xg_conceded"].isna().astype(int)
    df["missing_xg_ratio"] = df["xg_ratio"].isna().astype(int)
    df["missing_defensive_pressure"] = df["defensive_pressure"].isna().astype(int)

    
        

    
    df.to_csv("debug_features.csv", index=False)
    return df
