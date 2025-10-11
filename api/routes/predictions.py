from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import pandas as pd
from datetime import datetime
import logging

from ..dependencies import get_ml_model, get_feature_columns, get_db
from ..services.feature_calculator import FeatureCalculator
from ..schemas import SimplePredictionRequest, PredictionResponse, PredictionRequest

router = APIRouter(prefix="/predictions", tags=["predictions"])
logger = logging.getLogger(__name__)

@router.post("/predict-auto", response_model=PredictionResponse)
async def predict_match_auto(request: SimplePredictionRequest, db: Session = Depends(get_db)):
    """
    Prédire automatiquement un match avec calcul des features
    
    - **home_team**: Nom de l'équipe à domicile
    - **away_team**: Nom de l'équipe à l'extérieur  
    - **match_date**: Date du match (optionnel, défaut = maintenant)
    """
    try:
        logger.info(f"Prédiction automatique demandée: {request.home_team} vs {request.away_team}")
        
        # Date du match (par défaut maintenant)
        match_date = request.match_date or datetime.now()
        
        # Calculer automatiquement les features
        calculator = FeatureCalculator(db)
        features = calculator.calculate_all_features(
            request.home_team, 
            request.away_team, 
            match_date
        )
        
        # Charger modèle
        model, encoder = get_ml_model()
        feature_columns = get_feature_columns()
        
        # Préparer DataFrame
        features_df = pd.DataFrame([features])
        features_df = features_df[feature_columns]
        
        # Prédiction
        probabilities = model.predict_proba(features_df)[0]
        prediction = model.predict(features_df)[0]
        result = encoder.inverse_transform([prediction])[0]
        
        # Formater réponse
        class_probabilities = {
            encoder.inverse_transform([i])[0]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        logger.info(f"Prédiction réussie: {result} (confiance: {max(probabilities):.2%})")
        
        return PredictionResponse(
            home_team=request.home_team,
            away_team=request.away_team,
            prediction=result,
            probabilities=class_probabilities,
            confidence=float(max(probabilities)),
            features_used=features  # Montrer les features calculées
        )
        
    except Exception as e:
        logger.error(f"Erreur prédiction automatique: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur prédiction automatique: {str(e)}")

@router.post("/predict", response_model=PredictionResponse)
async def predict_match_manual(request: PredictionRequest):
    """
    Prédire un match avec features fournies manuellement
    
    - Fournir toutes les 20 features statistiques
    - Pour usage avancé uniquement
    """
    try:
        logger.info(f"Prédiction manuelle demandée: {request.home_team} vs {request.away_team}")
        
        # Charger modèle et encodeur
        model, encoder = get_ml_model()
        feature_columns = get_feature_columns()
        
        # Préparer les features
        features_dict = request.dict()
        features_df = pd.DataFrame([features_dict])
        
        # S'assurer d'avoir toutes les colonnes nécessaires
        for col in feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0.0
        
        # Sélectionner uniquement les features nécessaires
        features = features_df[feature_columns]
        
        # Prédiction
        probabilities = model.predict_proba(features)[0]
        prediction = model.predict(features)[0]
        
        # Décoder la prédiction
        result = encoder.inverse_transform([prediction])[0]
        
        # Formater les probabilités
        class_probabilities = {
            encoder.inverse_transform([i])[0]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        logger.info(f"Prédiction manuelle réussie: {result} (confiance: {max(probabilities):.2%})")
        
        return PredictionResponse(
            home_team=request.home_team,
            away_team=request.away_team,
            prediction=result,
            probabilities=class_probabilities,
            confidence=float(max(probabilities)),
            features_used={col: float(features[col].iloc[0]) for col in feature_columns}
        )
        
    except Exception as e:
        logger.error(f"Erreur prédiction manuelle: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur prédiction: {str(e)}")

@router.get("/upcoming")
async def get_upcoming_predictions(limit: int = 10, db: Session = Depends(get_db)):
    """
    Récupérer les prédictions pour les prochains matchs basés sur les données récentes
    """
    try:
        logger.info(f"Récupération des {limit} prochaines prédictions")
        
        calculator = FeatureCalculator(db)
        model, encoder = get_ml_model()
        feature_columns = get_feature_columns()
        
        # Charger les matchs récents comme "prochains" pour la démo
        from sqlalchemy import text
        query = text("""
        SELECT DISTINCT home_team, away_team, date, competition_name as competition
        FROM matches_cleaned 
        ORDER BY date DESC 
        LIMIT :limit
        """)
        
        upcoming_matches = pd.read_sql_query(query, db.bind, params={"limit": limit})
        
        predictions = []
        for _, match in upcoming_matches.iterrows():
            try:
                # Calculer les features pour ce match
                features = calculator.calculate_all_features(
                    match['home_team'],
                    match['away_team'], 
                    match['date']
                )
                
                # Préparer DataFrame
                features_df = pd.DataFrame([features])
                features_df = features_df[feature_columns]
                
                # Prédiction
                probabilities = model.predict_proba(features_df)[0]
                prediction = model.predict(features_df)[0]
                result = encoder.inverse_transform([prediction])[0]
                
                # Formater les probabilités
                class_probabilities = {
                    encoder.inverse_transform([i])[0]: float(prob) 
                    for i, prob in enumerate(probabilities)
                }
                
                predictions.append({
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'date': match['date'].strftime('%Y-%m-%d %H:%M:%S'),
                    'competition': match['competition'],
                    'prediction': result,
                    'probabilities': class_probabilities,
                    'confidence': float(max(probabilities))
                })
                
            except Exception as e:
                logger.warning(f"Impossible de prédire {match['home_team']} vs {match['away_team']}: {e}")
                continue
        
        logger.info(f"Généré {len(predictions)} prédictions pour matchs à venir")
        
        return {
            "count": len(predictions),
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Erreur récupération prédictions à venir: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur récupération prédictions: {str(e)}")

@router.get("/teams")
async def get_available_teams(limit: int = 50, db: Session = Depends(get_db)):
    """
    Récupérer la liste des équipes disponibles dans la base
    """
    try:
        from sqlalchemy import text
        
        query = text("""
        SELECT DISTINCT home_team as team_name 
        FROM matches_cleaned 
        ORDER BY home_team
        LIMIT :limit
        """)
        
        teams = pd.read_sql_query(query, db.bind, params={"limit": limit})
        
        return {
            "count": len(teams),
            "teams": teams['team_name'].tolist()
        }
        
    except Exception as e:
        logger.error(f"Erreur récupération équipes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur récupération équipes: {str(e)}")