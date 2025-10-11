import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from .feature_engineering import FootballFeatureEngineer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FootballPredictor:
    def __init__(self, model_path: str, encoder_path: str):
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.feature_columns = [
            'home_points_avg_5', 'home_goals_for_avg_5', 'home_goals_against_avg_5',
            'home_goal_diff_avg_5', 'home_recent_form', 'away_points_avg_5',
            'away_goals_for_avg_5', 'away_goals_against_avg_5', 'away_goal_diff_avg_5',
            'away_recent_form', 'form_difference', 'goal_difference',
            'h2h_win_rate_home', 'h2h_win_rate_away', 'h2h_matches_played',
            'home_days_rest', 'away_days_rest', 'rest_advantage', 
            'matchday_importance', 'is_weekend'
        ]
    
    def predict_match(self, features: pd.DataFrame):
        """Prédire un match unique"""
        # S'assurer que toutes les features sont présentes
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0.0
        
        features = features[self.feature_columns]
        
        # Prédiction
        probabilities = self.model.predict_proba(features)[0]
        prediction = self.model.predict(features)[0]
        
        # Décoder la prédiction
        result = self.label_encoder.inverse_transform([prediction])[0]
        
        # Probabilités pour chaque classe
        class_probabilities = {
            self.label_encoder.inverse_transform([i])[0]: prob 
            for i, prob in enumerate(probabilities)
        }
        
        return {
            'prediction': result,
            'probabilities': class_probabilities,
            'confidence': max(probabilities)
        }
    
    def predict_upcoming_matches(self, db_url: str, days_ahead: int = 7):
        """Prédire les matchs à venir"""
        # Charger les données récentes pour calculer les features
        engineer = FootballFeatureEngineer(db_url)
        df = engineer.load_data()
        
        # Filtrer les matchs à venir (simulation)
        latest_date = df['date'].max()
        future_date = latest_date + timedelta(days=days_ahead)
        
        # Ici, normalement tu récupérerais les matchs à venir via l'API
        # Pour l'instant, on va simuler avec les derniers matchs
        upcoming_matches = df[df['date'] > latest_date - timedelta(days=3)].tail(5)
        
        predictions = []
        for _, match in upcoming_matches.iterrows():
            # Dans la réalité, tu calculerais les features pour les matchs à venir
            # Pour la démo, on utilise les features existantes
            pred = self.predict_match(pd.DataFrame([match]))
            predictions.append({
                'match_id': match['match_id'],
                'home_team': match['home_team'],
                'away_team': match['away_team'],
                'date': match['date'],
                'prediction': pred['prediction'],
                'probabilities': pred['probabilities'],
                'confidence': pred['confidence']
            })
        
        return predictions

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    # Charger le dernier modèle entraîné
    model_files = [f for f in os.listdir("models") if f.startswith("xgboost") and f.endswith(".joblib")]
    latest_model = sorted(model_files)[-1] if model_files else "xgboost_optimized.joblib"
    encoder_files = [f for f in os.listdir("models") if f.startswith("label_encoder") and f.endswith(".joblib")]
    latest_encoder = sorted(encoder_files)[-1] if encoder_files else None
    
    if latest_encoder:
        predictor = FootballPredictor(
            model_path=f"models/{latest_model}",
            encoder_path=f"models/{latest_encoder}"
        )
        
        # Exemple de prédiction
        print("=== PRÉDICTIONS DE DÉMONSTRATION ===")
        
        # Créer des features d'exemple
        sample_features = pd.DataFrame([{
            'home_points_avg_5': 2.0,
            'home_goals_for_avg_5': 1.8,
            'home_goals_against_avg_5': 0.8,
            'home_goal_diff_avg_5': 1.0,
            'home_recent_form': 8.0,
            'away_points_avg_5': 1.2,
            'away_goals_for_avg_5': 1.0,
            'away_goals_against_avg_5': 1.5,
            'away_goal_diff_avg_5': -0.5,
            'away_recent_form': 4.0,
            'form_difference': 0.8,
            'goal_difference': 1.5,
            'h2h_win_rate_home': 0.6,
            'h2h_win_rate_away': 0.2,
            'h2h_matches_played': 5,
            'home_days_rest': 6,
            'away_days_rest': 4,
            'rest_advantage': 2,
            'matchday_importance': 0.7,
            'is_weekend': 1
        }])
        
        prediction = predictor.predict_match(sample_features)
        print(f"Prédiction: {prediction}")