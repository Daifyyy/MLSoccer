import pandas as pd

def generate_extended_features(df):
    df = df.copy()

    # Seznam týmů
    teams = set(df['HomeTeam']).union(set(df['AwayTeam']))

    # Výstupní datová struktura
    rows = []

    for idx, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        match_date = row['Date'] if 'Date' in row else idx

        # Historie domácích a hostů do data zápasu
        home_matches = df[((df['HomeTeam'] == home) | (df['AwayTeam'] == home)) & (df.index < idx)].tail(6)
        away_matches = df[((df['HomeTeam'] == away) | (df['AwayTeam'] == away)) & (df.index < idx)].tail(6)

        # Výpočty průměrů
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
        }

        # Rozdíly mezi domácími a hosty
        for key in list(features.keys()):
            if key.endswith('_home_form'):
                base = key.replace('_home_form', '')
                away_key = f'{base}_away_form'
                diff_key = f'{base}_diff'
                features[diff_key] = features[key] - features.get(away_key, 0)

        # Nová featura: procento zápasů Over 2.5 za posledních 6 utkání
        def calc_over25_ratio(matches):
            over25 = matches[(matches['FTHG'] + matches['FTAG']) > 2.5]
            return len(over25) / 6 if len(matches) >= 1 else 0.0

        features['over25_form_ratio_home'] = calc_over25_ratio(home_matches)
        features['over25_form_ratio_away'] = calc_over25_ratio(away_matches)

        # Výsledek
        row_features = row.to_dict()
        row_features.update(features)
        rows.append(row_features)

    return pd.DataFrame(rows)
