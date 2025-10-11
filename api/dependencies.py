from functools import lru_cache
import joblib
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

@lru_cache()
def get_db_engine():
    """Connection à la base de données"""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL non définie dans .env")
    return create_engine(db_url)

def get_db():
    """Générateur de session DB (pour Dependency Injection)"""
    engine = get_db_engine()
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@lru_cache()
def get_ml_model():
    """Charger le modèle ML optimisé"""
    try:
        model_path = "models/xgboost_optimized.joblib"
        encoder_path = "models/label_encoder_optimized.joblib"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        return model, encoder
    except Exception as e:
        raise RuntimeError(f"Erreur chargement modèle: {e}")

def get_feature_columns():
    """Liste des features attendues par le modèle"""
    return [
        'home_points_avg_5', 'home_goals_for_avg_5', 'home_goals_against_avg_5',
        'home_goal_diff_avg_5', 'home_recent_form', 'away_points_avg_5',
        'away_goals_for_avg_5', 'away_goals_against_avg_5', 'away_goal_diff_avg_5',
        'away_recent_form', 'form_difference', 'goal_difference',
        'h2h_win_rate_home', 'h2h_win_rate_away', 'h2h_matches_played',
        'home_days_rest', 'away_days_rest', 'rest_advantage', 
        'matchday_importance', 'is_weekend'
    ]