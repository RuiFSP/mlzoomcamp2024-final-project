import pandas as pd
import os
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def save_data(df, file_path):
    df.to_csv(file_path, index=False)

def rename_columns(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    columns = {
        'hometeam': 'home_team',
        'awayteam': 'away_team',
        'fthg': 'home_total_goals',
        'ftag': 'away_total_goals',
        'ftr': 'full_time_result',
        'hthg': 'home_half_goals',
        'htag': 'away_half_goals',
        'htr': 'half_time_result',
        'hs': 'home_total_shots',
        'as': 'away_total_shots',
        'hst': 'home_shots_on_target',
        'ast': 'away_shots_on_target',
        'hf': 'home_fouls',
        'af': 'away_fouls',
        'hc': 'home_corners',
        'ac': 'away_corners',
        'hy': 'home_yellow_cards',
        'ay': 'away_yellow_cards',
        'hr': 'home_red_cards',
        'ar': 'away_red_cards',
        'b365h': 'market_home_odds',
        'b365d': 'market_draw_odds',
        'b365a': 'market_away_odds'
    }
    df.rename(columns=columns, inplace=True)
    return df

def clean_team_names(df):
    for col in ['home_team', 'away_team']:
        if col in df.columns:
            df[col] = df[col].str.lower().str.replace("'", "")
    return df

def clean_referee_names(df):
    if 'referee' in df.columns:
        df['referee'] = df['referee'].str.lower().replace(' ', '_')
    return df

def fixing_columns_teams_referees(df):
    df = rename_columns(df)
    df = clean_team_names(df)
    df = clean_referee_names(df)
    return df

def add_goal_difference(df):
    df['goal_difference'] = df['home_total_goals'] - df['away_total_goals']
    return df

def add_aggregated_match_statistics(df):
    df['total_shots'] = df['home_total_shots'] + df['away_total_shots']
    df['total_shots_on_target'] = df['home_shots_on_target'] + df['away_shots_on_target']
    df['total_fouls'] = df['home_fouls'] + df['away_fouls']
    df['total_corners'] = df['home_corners'] + df['away_corners']
    df['home_shot_accuracy'] = df['home_shots_on_target'] / df['home_total_shots'].replace(0, 1)
    df['away_shot_accuracy'] = df['away_shots_on_target'] / df['away_total_goals'].replace(0, 1)
    return df

def add_time_based_features(df):
    df['original_date'] = df['date']
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y', errors='coerce')
    df['date'] = df['date'].combine_first(pd.to_datetime(df['original_date'], format='%d/%m/%Y', errors='coerce'))
    df['date'] = df['date'].dt.strftime('%d/%m/%y')
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y')
    df.drop(columns=['original_date'], inplace=True)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 6.0)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 6.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    return df

def add_team_based_features(df):
    df['ratio_h_a_shots'] = df['home_total_shots'] / df['away_total_shots'].replace(0, 1)
    df['ratio_h_a_fouls'] = df['home_fouls'] / df['away_fouls'].replace(0, 1)
    df['ratio_a_h_shots'] = df['away_total_shots'] / df['home_total_shots'].replace(0, 1)
    df['ratio_a_h_fouls'] = df['away_fouls'] / df['home_fouls'].replace(0, 1)
    return df

def add_betting_odds_features(df):
    df['implied_home_win_prob'] = 1 / df['market_home_odds']
    df['implied_draw_prob'] = 1 / df['market_draw_odds']
    df['implied_away_win_prob'] = 1 / df['market_away_odds']
    total_prob = df['implied_home_win_prob'] + df['implied_draw_prob'] + df['implied_away_win_prob']
    df['implied_home_win_prob'] /= total_prob
    df['implied_draw_prob'] /= total_prob
    df['implied_away_win_prob'] /= total_prob
    return df

def add_rolling_averages(df):
    features = ['home_total_goals', 'away_total_goals', 'home_total_shots', 'away_total_shots', 
                'home_shots_on_target', 'away_shots_on_target', 'home_fouls', 'away_fouls',
                'home_corners', 'away_corners', 'home_yellow_cards', 'away_yellow_cards',
                'home_red_cards', 'away_red_cards', 'home_shot_accuracy', 'away_shot_accuracy',
                'ratio_h_a_shots', 'ratio_h_a_fouls', 'ratio_a_h_shots', 
                'ratio_a_h_fouls', 'goal_difference']
    new_columns = []
    for i in [3, 5]:
        for feature in features:
            home_rolling = (
                df.sort_values(['season', 'home_team', 'date'])
                  .groupby(['season', 'home_team'])[feature]
                  .apply(lambda x: x.shift(1).rolling(window=i).mean())
                  .reset_index(level=[0,1], drop=True)
                  .fillna(0)
            )
            away_rolling = (
                df.sort_values(['season', 'away_team', 'date'])
                  .groupby(['season', 'away_team'])[feature]
                  .apply(lambda x: x.shift(1).rolling(window=i).mean())
                  .reset_index(level=[0,1], drop=True)
                  .fillna(0)
            )
            new_columns.append(home_rolling.rename(f'home_roll_{i}_avg_{feature}'))
            new_columns.append(away_rolling.rename(f'away_roll_{i}_avg_{feature}'))
    df = pd.concat([df] + new_columns, axis=1)
    return df

def add_cumulative_points(df):
    home_points = df['full_time_result'].apply(lambda x: 3 if x == 'H' else (1 if x == 'D' else 0))
    away_points = df['full_time_result'].apply(lambda x: 3 if x == 'A' else (1 if x == 'D' else 0))
    df = pd.concat([df, home_points.rename('home_points'), away_points.rename('away_points')], axis=1)
    df['home_cumulative_points'] = df.groupby(['season', 'home_team'])['home_points'].transform('cumsum')
    df['away_cumulative_points'] = df.groupby(['season', 'away_team'])['away_points'].transform('cumsum')
    df.drop(columns=['home_points', 'away_points'], inplace=True)
    return df

def drop_unnecessary_columns(df):
    columns_to_drop = ['date', 'home_total_goals', 'away_total_goals', 'home_half_goals',
                       'away_half_goals', 'half_time_result', 'home_total_shots', 'away_total_shots',
                       'home_shots_on_target', 'away_shots_on_target', 'home_fouls', 'away_fouls',
                       'home_corners', 'away_corners', 'home_yellow_cards', 'away_yellow_cards',
                       'home_red_cards', 'away_red_cards', 'goal_difference', 'total_shots',
                       'total_shots_on_target', 'total_fouls', 'total_corners', 'home_shot_accuracy',
                       'away_shot_accuracy', 'ratio_h_a_shots', 'ratio_h_a_fouls', 'ratio_a_h_shots',
                       'ratio_a_h_fouls', 'referee', 'market_home_odds', 'market_draw_odds', 'market_away_odds']
    df = df.drop(columns=columns_to_drop)
    return df

def feature_engineering(df):
    df = add_goal_difference(df)
    df = add_aggregated_match_statistics(df)
    df = add_time_based_features(df)
    df = add_team_based_features(df)
    df = add_betting_odds_features(df)
    df = add_rolling_averages(df)
    df = add_cumulative_points(df)
    df = drop_unnecessary_columns(df)
    return df

def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'all_concat_football_data.csv')
    df = load_data(data_path)
    df = fixing_columns_teams_referees(df)
    df = df.dropna().reset_index(drop=True)
    df = feature_engineering(df)
    
    teams_stats_2024 = df[df['season'] == 2024]
    save_data(teams_stats_2024, os.path.join(os.path.dirname(__file__),'..','data', 'processed', 'teams_stats_2024.csv'))
    
    # Drop season column
    df = df.drop(columns=['season'])
    
    save_data(df, os.path.join(os.path.dirname(__file__),'..','data', 'processed', 'prepared_football_data.csv'))
    

if __name__ == "__main__":
    main()
