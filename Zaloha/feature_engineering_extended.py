import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

# === Pomocné funkce ===
def validate_columns(df, required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def calculate_elo_rating(df):
    elo_ratings = defaultdict(lambda: 1500)
    k = 20
    ratings = []
    for _, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        home_elo, away_elo = elo_ratings[home], elo_ratings[away]
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        expected_away = 1 - expected_home
        score_home, score_away = (1, 0) if row['FTHG'] > row['FTAG'] else (0, 1) if row['FTHG'] < row['FTAG'] else (0.5, 0.5)
        elo_ratings[home] += k * (score_home - expected_home)
        elo_ratings[away] += k * (score_away - expected_away)
        ratings.append((home_elo, away_elo))
    return ratings

def compute_last5_rolling(df, group_col, base_col):
    return (
        df.groupby(group_col)[base_col]
          .transform(lambda x: x.shift().rolling(window=5, min_periods=1).mean())
          .fillna(0)
    )

def calculate_expected_goals(df):
    df['home_xg'] = df['HS'] * 0.09 + df['HST'] * 0.2
    df['away_xg'] = df['AS'] * 0.09 + df['AST'] * 0.2
    return df

def compute_h2h_stats(df, n_matches=5):
    df['h2h_avg_goals'] = 0.0
    df['h2h_over25_ratio'] = 0.0
    for idx, row in df.iterrows():
        h_team = row['HomeTeam']
        a_team = row['AwayTeam']
        date = row['Date']
        past_matches = df[
            (((df['HomeTeam'] == h_team) & (df['AwayTeam'] == a_team)) |
             ((df['HomeTeam'] == a_team) & (df['AwayTeam'] == h_team))) &
            (df['Date'] < date)
        ].sort_values(by='Date', ascending=False).head(n_matches)
        if not past_matches.empty:
            goals = past_matches['FTHG'] + past_matches['FTAG']
            df.at[idx, 'h2h_avg_goals'] = goals.mean()
            df.at[idx, 'h2h_over25_ratio'] = (goals > 2.5).sum() / len(goals)
    df['h2h_avg_goals'] = df['h2h_avg_goals'].fillna(0)
    df['h2h_over25_ratio'] = df['h2h_over25_ratio'].fillna(0)
    return df

def compute_team_level_features(df):
    df['is_over_2_5'] = (df['FTHG'] + df['FTAG']) > 2.5

    over25_ratio_season_home = df.groupby("HomeTeam")['is_over_2_5'].transform(lambda x: x.expanding().mean().shift())
    over25_ratio_season_away = df.groupby("AwayTeam")['is_over_2_5'].transform(lambda x: x.expanding().mean().shift())
    df['over25_ratio_season_avg'] = (over25_ratio_season_home + over25_ratio_season_away) / 2

    over25_ratio_last10_home = df.groupby("HomeTeam")['is_over_2_5'].transform(lambda x: x.shift().rolling(10, min_periods=1).mean())
    over25_ratio_last10_away = df.groupby("AwayTeam")['is_over_2_5'].transform(lambda x: x.shift().rolling(10, min_periods=1).mean())
    df['over25_ratio_last10_avg'] = (over25_ratio_last10_home + over25_ratio_last10_away) / 2

    over25_ratio_last5_home = df.groupby("HomeTeam")['is_over_2_5'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    over25_ratio_last5_away = df.groupby("AwayTeam")['is_over_2_5'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    df['over25_ratio_last5_avg'] = (over25_ratio_last5_home + over25_ratio_last5_away) / 2
    df['over25_trend'] = df['over25_ratio_last5_avg'] - df['over25_ratio_last10_avg']

    df['goal_sum'] = df['FTHG'] + df['FTAG']
    df['goal_std_last5'] = df.groupby("HomeTeam")['goal_sum'].transform(lambda x: x.shift().rolling(5, min_periods=1).std()).fillna(0)

    df['attack_pressure'] = df['HS'] + df['HST'] + df['HC'] + df['AS'] + df['AST'] + df['AC']
    df['attack_pressure_last5'] = df.groupby("HomeTeam")['attack_pressure'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(0)

    df['games_last_14d'] = 0
    for team in df['HomeTeam'].unique():
        team_dates = df[df['HomeTeam'] == team][['Date']].copy()
        team_dates = team_dates.sort_values('Date')
        game_counts = []
        for current_index, current_date in enumerate(team_dates['Date']):
            window_start = current_date - pd.Timedelta(days=14)
            count = team_dates[(team_dates['Date'] < current_date) & (team_dates['Date'] >= window_start)].shape[0]
            game_counts.append(count)
        df.loc[df['HomeTeam'] == team, 'games_last_14d'] = game_counts
        
    # Interakční featury – rozdíly mezi útokem a obranou
    df['xg_off_def_diff'] = df['xg_home_last5'] - df['xg_conceded_away_last5']
    df['shots_on_target_diff'] = df['shots_on_target_home_last5'] - df['shots_on_target_away_last5']
    df['elo_diff'] = df['elo_rating_home'] - df['elo_rating_away']
    df['over25_momentum'] = df['over25_ratio_last5_avg'] - df['over25_ratio_last10_avg']


    return df

# === Hlavní funkce ===
def generate_features(df, mode="train", debug=False):
    df = df.copy()
    df["HomeTeam"] = df["HomeTeam"].str.strip()
    df["AwayTeam"] = df["AwayTeam"].str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"])
    df = df.sort_values(by='Date')

    required_cols = ['HomeTeam', 'AwayTeam', 'HS', 'AS', 'HST', 'AST', 'FTHG', 'FTAG']
    validate_columns(df, required_cols)

    df['elo_rating_home'], df['elo_rating_away'] = zip(*calculate_elo_rating(df))
    df = calculate_expected_goals(df)

    df['TotalGoals'] = df['FTHG'] + df['FTAG']

    # Průměr vstřelených gólů doma i venku za sezónu
    df['goals_scored_season_avg_home'] = df.groupby('HomeTeam')['FTHG'].transform(lambda x: x.expanding().mean())
    df['goals_scored_season_avg_away'] = df.groupby('AwayTeam')['FTAG'].transform(lambda x: x.expanding().mean())

    # Průměr obdržených gólů doma i venku za sezónu
    df['goals_conceded_season_avg_home'] = df.groupby('HomeTeam')['FTAG'].transform(lambda x: x.expanding().mean())
    df['goals_conceded_season_avg_away'] = df.groupby('AwayTeam')['FTHG'].transform(lambda x: x.expanding().mean())

    # Posledních 5 zápasů – průměr vstřelených gólů
    df['goals_scored_last5_home'] = df.groupby('HomeTeam')['FTHG'].transform(lambda x: x.shift().rolling(5).mean())
    df['goals_scored_last5_away'] = df.groupby('AwayTeam')['FTAG'].transform(lambda x: x.shift().rolling(5).mean())
    df['goals_scored_total_last5'] = df.groupby('HomeTeam')['FTHG'].transform(lambda x: x.shift().rolling(5).mean()) + df.groupby('AwayTeam')['FTAG'].transform(lambda x: x.shift().rolling(5).mean())

    # Posledních 5 zápasů – průměr obdržených gólů
    df['goals_conceded_last5_home'] = df.groupby('HomeTeam')['FTAG'].transform(lambda x: x.shift().rolling(5).mean())
    df['goals_conceded_last5_away'] = df.groupby('AwayTeam')['FTHG'].transform(lambda x: x.shift().rolling(5).mean())
    df['goals_conceded_total_last5'] = df.groupby('HomeTeam')['FTAG'].transform(lambda x: x.shift().rolling(5).mean()) + df.groupby('AwayTeam')['FTHG'].transform(lambda x: x.shift().rolling(5).mean())

    # Průměr rohů za posledních 5 zápasů
    df['corners_last5_home'] = df.groupby('HomeTeam')['HC'].transform(lambda x: x.shift().rolling(5).mean())
    df['corners_last5_away'] = df.groupby('AwayTeam')['AC'].transform(lambda x: x.shift().rolling(5).mean())
    
    # Rozdíl a poměry
    df['goal_diff_last5'] = df['goals_scored_total_last5'] - df['goals_conceded_total_last5']
    df['goal_ratio_home'] = df['goals_scored_last5_home'] / (df['goals_conceded_last5_home'] + 1e-5)
    df['goal_ratio_away'] = df['goals_scored_last5_away'] / (df['goals_conceded_last5_away'] + 1e-5)
     
    df['xg_home_last5'] = compute_last5_rolling(df, 'HomeTeam', 'home_xg')
    df['xg_away_last5'] = compute_last5_rolling(df, 'AwayTeam', 'away_xg')
    df['shots_home_last5'] = compute_last5_rolling(df, 'HomeTeam', 'HS')
    df['shots_away_last5'] = compute_last5_rolling(df, 'AwayTeam', 'AS')
    df['shots_on_target_home_last5'] = compute_last5_rolling(df, 'HomeTeam', 'HST')
    df['shots_on_target_away_last5'] = compute_last5_rolling(df, 'AwayTeam', 'AST')

    
    df['xg_conceded_home_last5'] = compute_last5_rolling(df, 'HomeTeam', 'away_xg')
    df['xg_conceded_away_last5'] = compute_last5_rolling(df, 'AwayTeam', 'home_xg')
    df['avg_xg_conceded'] = (df['xg_conceded_home_last5'] + df['xg_conceded_away_last5']) / 2
    df['xg_ratio'] = (df['xg_home_last5'] + df['xg_away_last5']) / (df['avg_xg_conceded'] + 0.1)
    df['defensive_pressure'] = (df['HS'] - df['AS']).abs() + (df['HST'] - df['AST']).abs()

    df['tempo_score'] = df['HS'] + df['AS'] + df['HC'] + df['AC']
    df['passivity_score'] = (df['HS'] + df['AS']) * 0.05 + (df['HC'] + df['AC']) * 0.1 + (df['home_xg'] + df['away_xg']) * 0.15
    df['card_diff'] = (df['HY'] + df['HR']) - (df['AY'] + df['AR'])
    df['fouls_diff'] = df['HF'] - df['AF']
    df['aggressiveness_score'] = df['fouls_diff'].abs() + df['card_diff'].abs()
    df['behavior_balance'] = df['passivity_score'] + df['aggressiveness_score']

    df['conceded_home_last5'] = compute_last5_rolling(df, 'HomeTeam', 'FTAG')
    df['conceded_away_last5'] = compute_last5_rolling(df, 'AwayTeam', 'FTHG')
    df['goal_sum'] = df['FTHG'] + df['FTAG']
    df['avg_goal_sum_last5'] = compute_last5_rolling(df, 'HomeTeam', 'goal_sum')

    df['momentum_score'] = (df['elo_rating_home'] - df['elo_rating_away']) + (df['xg_home_last5'] - df['xg_away_last5'])
    df['match_weight'] = df['Date'].apply(
        lambda date: 1 / (np.log((datetime.now() - date).days + 2)) if pd.notnull(date) else 1.0
    )

    if mode == "train":
        df['boring_match_score'] = (df['HS'] + df['AS']) * 0.05 + (df['HST'] + df['AST']) * 0.1
    else:
        df['boring_match_score'] = (df['xg_home_last5'] + df['xg_away_last5']) * 0.15

    df = compute_h2h_stats(df)
    df = compute_team_level_features(df)
    
    if debug:
        df.to_csv(f"debug_features_{mode}.csv", index=False)

    return df
