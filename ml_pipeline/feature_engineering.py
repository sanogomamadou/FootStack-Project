import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FootballFeatureEngineer:
    def __init__(self, db_url: str, lookback_matches: int = 5):
        """
        Initialize the feature engineer
        
        Args:
            db_url: Database connection URL
            lookback_matches: Number of previous matches to consider for features
        """
        self.engine = create_engine(db_url)
        self.lookback_matches = lookback_matches
        self.team_encoder = LabelEncoder()
        self.competition_encoder = LabelEncoder()
        
    def load_data(self) -> pd.DataFrame:
        """Load cleaned matches data from database"""
        logger.info("Loading matches data from database...")
        query = """
        SELECT 
            match_id, date, competition_name, home_team, away_team,
            home_score, away_score, result, matchday, country, stage
        FROM matches_cleaned 
        ORDER BY date
        """
        df = pd.read_sql(query, self.engine)
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"Loaded {len(df)} matches")
        return df
    
    def calculate_team_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate recent form for each team (last N matches)
        Returns form features like wins/draws/losses in last 5 games
        """
        logger.info("Calculating team form features...")
        
        # Vérifier que les colonnes nécessaires existent dans le DataFrame principal
        required_columns = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'result']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing columns in dataframe: {missing_columns}. Available columns: {df.columns.tolist()}")
        
        # Create a melted dataframe with team perspective - CORRECTION ICI
        home_games = df[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'result']].copy()
        home_games['team'] = home_games['home_team']
        home_games['opponent'] = home_games['away_team']  # Maintenant away_team est disponible
        home_games['goals_for'] = home_games['home_score']
        home_games['goals_against'] = home_games['away_score']
        home_games['is_home'] = True
        home_games['points'] = home_games['result'].map({'Home': 3, 'Draw': 1, 'Away': 0})
        
        away_games = df[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'result']].copy()
        away_games['team'] = away_games['away_team']
        away_games['opponent'] = away_games['home_team']  # home_team est disponible
        away_games['goals_for'] = away_games['away_score']
        away_games['goals_against'] = away_games['home_score']
        away_games['is_home'] = False
        away_games['points'] = away_games['result'].map({'Home': 0, 'Draw': 1, 'Away': 3})
        
        # Combine home and away games
        team_games = pd.concat([home_games, away_games], ignore_index=True)
        team_games = team_games.sort_values(['team', 'date']).reset_index(drop=True)
        
        # Calculate rolling statistics for each team
        form_features = []
        
        for team in team_games['team'].unique():
            team_data = team_games[team_games['team'] == team].copy()
            
            # Rolling features for last N matches
            for window in [5, 10]:
                team_data[f'points_avg_{window}'] = (
                    team_data['points']
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .shift(1)  # Use data only from previous matches
                )
                
                team_data[f'goals_for_avg_{window}'] = (
                    team_data['goals_for']
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .shift(1)
                )
                
                team_data[f'goals_against_avg_{window}'] = (
                    team_data['goals_against']
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .shift(1)
                )
                
                team_data[f'goal_diff_avg_{window}'] = (
                    team_data[f'goals_for_avg_{window}'] - team_data[f'goals_against_avg_{window}']
                )
            
            # Recent form (last 5 matches points)
            team_data['recent_form'] = (
                team_data['points']
                .rolling(window=5, min_periods=1)
                .sum()
                .shift(1)
            )
            
            form_features.append(team_data)
        
        form_df = pd.concat(form_features, ignore_index=True)
        logger.info(f"Form calculation completed. Processed {len(form_df)} team-game records")
        return form_df
    
    def calculate_head_to_head(self, df: pd.DataFrame, form_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate head-to-head statistics between teams
        """
        logger.info("Calculating head-to-head features...")
        
        h2h_features = []
        
        for idx, match in df.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            match_date = match['date']
            
            # Get previous matches between these two teams
            previous_matches = df[
                (df['date'] < match_date) & 
                (((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
                 ((df['home_team'] == away_team) & (df['away_team'] == home_team)))
            ].tail(10)  # Last 10 encounters
            
            if len(previous_matches) > 0:
                # Calculate H2H statistics
                home_wins = len(previous_matches[
                    ((previous_matches['home_team'] == home_team) & (previous_matches['result'] == 'Home')) |
                    ((previous_matches['away_team'] == home_team) & (previous_matches['result'] == 'Away'))
                ])
                
                away_wins = len(previous_matches[
                    ((previous_matches['home_team'] == away_team) & (previous_matches['result'] == 'Home')) |
                    ((previous_matches['away_team'] == away_team) & (previous_matches['result'] == 'Away'))
                ])
                
                draws = len(previous_matches[previous_matches['result'] == 'Draw'])
                
                total_matches = len(previous_matches)
                h2h_win_rate_home = home_wins / total_matches if total_matches > 0 else 0.5
                h2h_win_rate_away = away_wins / total_matches if total_matches > 0 else 0.5
                
            else:
                h2h_win_rate_home = 0.5  # Neutral prior if no history
                h2h_win_rate_away = 0.5
            
            h2h_features.append({
                'match_id': match['match_id'],
                'h2h_win_rate_home': h2h_win_rate_home,
                'h2h_win_rate_away': h2h_win_rate_away,
                'h2h_matches_played': len(previous_matches)
            })
        
        return pd.DataFrame(h2h_features)
    
    def create_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create contextual features like days since last game, matchday importance, etc.
        """
        logger.info("Creating contextual features...")
        
        contextual_features = []
        
        for idx, match in df.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            match_date = match['date']
            
            # Days since last game for each team
            home_previous = df[
                (df['date'] < match_date) & 
                ((df['home_team'] == home_team) | (df['away_team'] == home_team))
            ].tail(1)
            
            away_previous = df[
                (df['date'] < match_date) & 
                ((df['home_team'] == away_team) | (df['away_team'] == away_team))
            ].tail(1)
            
            home_days_rest = (match_date - home_previous['date'].iloc[0]).days if len(home_previous) > 0 else 7
            away_days_rest = (match_date - away_previous['date'].iloc[0]).days if len(away_previous) > 0 else 7
            
            # Matchday importance (late season matches might be more important)
            season_matches = df[
                (df['competition_name'] == match['competition_name']) & 
                (df['date'].dt.year == match['date'].year)
            ]
            total_matchdays = season_matches['matchday'].max() if not season_matches.empty else 38
            matchday_importance = match['matchday'] / total_matchdays if total_matchdays > 0 else 0.5
            
            contextual_features.append({
                'match_id': match['match_id'],
                'home_days_rest': home_days_rest,
                'away_days_rest': away_days_rest,
                'rest_advantage': home_days_rest - away_days_rest,
                'matchday_importance': matchday_importance,
                'is_weekend': 1 if match_date.weekday() >= 5 else 0  # Saturday or Sunday
            })
        
        return pd.DataFrame(contextual_features)
    
    def build_features(self) -> pd.DataFrame:
        """
        Main method to build all features
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Load base data
        df = self.load_data()
        
        # Calculate team form
        form_df = self.calculate_team_form(df)
        
        # Merge form features back to main dataframe
        features_list = []
        
        for idx, match in df.iterrows():
            # Home team features (from their last game before this match)
            home_team_data = form_df[
                (form_df['team'] == match['home_team']) & 
                (form_df['date'] < match['date'])
            ].tail(1)
            
            # Away team features
            away_team_data = form_df[
                (form_df['team'] == match['away_team']) & 
                (form_df['date'] < match['date'])
            ].tail(1)
            
            if len(home_team_data) > 0 and len(away_team_data) > 0:
                feature_row = {
                    'match_id': match['match_id'],
                    'date': match['date'],
                    'competition': match['competition_name'],
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'target': match['result'],  # This is what we want to predict
                    
                    # Home team recent form
                    'home_points_avg_5': home_team_data['points_avg_5'].iloc[0],
                    'home_goals_for_avg_5': home_team_data['goals_for_avg_5'].iloc[0],
                    'home_goals_against_avg_5': home_team_data['goals_against_avg_5'].iloc[0],
                    'home_goal_diff_avg_5': home_team_data['goal_diff_avg_5'].iloc[0],
                    'home_recent_form': home_team_data['recent_form'].iloc[0],
                    
                    # Away team recent form
                    'away_points_avg_5': away_team_data['points_avg_5'].iloc[0],
                    'away_goals_for_avg_5': away_team_data['goals_for_avg_5'].iloc[0],
                    'away_goals_against_avg_5': away_team_data['goals_against_avg_5'].iloc[0],
                    'away_goal_diff_avg_5': away_team_data['goal_diff_avg_5'].iloc[0],
                    'away_recent_form': away_team_data['recent_form'].iloc[0],
                    
                    # Difference features
                    'form_difference': home_team_data['points_avg_5'].iloc[0] - away_team_data['points_avg_5'].iloc[0],
                    'goal_difference': home_team_data['goal_diff_avg_5'].iloc[0] - away_team_data['goal_diff_avg_5'].iloc[0]
                }
                features_list.append(feature_row)
        
        features_df = pd.DataFrame(features_list)
        
        # Add head-to-head features
        h2h_df = self.calculate_head_to_head(df, form_df)
        features_df = features_df.merge(h2h_df, on='match_id', how='left')
        
        # Add contextual features
        contextual_df = self.create_contextual_features(df)
        features_df = features_df.merge(contextual_df, on='match_id', how='left')
        
        # Fill NaN values (for teams with no previous matches)
        fill_values = {
            'home_points_avg_5': 1.0,
            'away_points_avg_5': 1.0,
            'home_goals_for_avg_5': 1.0,
            'away_goals_for_avg_5': 1.0,
            'home_goals_against_avg_5': 1.0,
            'away_goals_against_avg_5': 1.0,
            'home_goal_diff_avg_5': 0.0,  # Ajouté
            'away_goal_diff_avg_5': 0.0,  # Ajouté
            'home_recent_form': 0.0,      # Ajouté
            'away_recent_form': 0.0,      # Ajouté
            'form_difference': 0.0,       # Ajouté
            'goal_difference': 0.0,       # Ajouté
            'h2h_win_rate_home': 0.5,
            'h2h_win_rate_away': 0.5,
            'h2h_matches_played': 0,
            'home_days_rest': 7,
            'away_days_rest': 7,
            'rest_advantage': 0,
            'matchday_importance': 0.5,
            'is_weekend': 0
        }
        features_df = features_df.fillna(fill_values)
        
        logger.info(f"Feature engineering completed. Final dataset: {len(features_df)} matches")
        logger.info(f"Features created: {list(features_df.columns)}")
        
        return features_df

# Usage example
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    
    engineer = FootballFeatureEngineer(db_url)
    features_df = engineer.build_features()
    
    # Save features to CSV for inspection
    features_df.to_csv("data/features_dataset.csv", index=False)
    print("Features saved to data/features_dataset.csv")
    print(f"Dataset shape: {features_df.shape}")
    print(f"Target distribution:\n{features_df['target'].value_counts()}")
    
    # Afficher les premières lignes pour inspection
    print("\nFirst 5 rows:")
    print(features_df.head())
    
    # Statistiques descriptives des features
    print("\nFeature statistics:")
    print(features_df.describe())