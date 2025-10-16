import pandas as pd
import joblib
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

def predict_upcoming_matches():
    """Prédire les prochains matchs basés sur les features calculées"""
    load_dotenv()
    
    # Charger le modèle optimisé
    predictor = FootballPredictor(
        model_path="models/xgboost_optimized.joblib",
        encoder_path="models/label_encoder_optimized.joblib"
    )
    
    # Charger le dataset AVEC LES FEATURES CALCULÉES
    features_df = pd.read_csv("data/features_dataset.csv")
    features_df['date'] = pd.to_datetime(features_df['date'])
    
    # Prendre les 10 derniers matchs comme "prochains" pour la démo
    latest_matches = features_df.sort_values('date', ascending=False).head(10)
    
    print("=== PRÉDICTIONS DES PROCHAINS MATCHS ===")
    print("=" * 50)
    
    predictions = []
    for _, match in latest_matches.iterrows():
        try:
            # Convertir la date en string pour l'affichage
            match_date = match['date']
            if hasattr(match_date, 'strftime'):
                date_str = match_date.strftime('%Y-%m-%d')
            else:
                date_str = str(match_date)[:10]
            
            # Préparer les features pour ce match (elles sont déjà dans le DataFrame)
            features = pd.DataFrame([match])
            
            prediction = predictor.predict_match(features)
            
            print(f"  {match['home_team']} vs {match['away_team']}")
            print(f"     {date_str} |   {match['competition']}")
            print(f"     Prédiction: {prediction['prediction']}")
            print(f"     Probabilités: Home {prediction['probabilities']['Home']:.1%} | Draw {prediction['probabilities']['Draw']:.1%} | Away {prediction['probabilities']['Away']:.1%}")
            print(f"     Confiance: {prediction['confidence']:.1%}")
            print("-" * 50)
            
            predictions.append({
                'home_team': match['home_team'],
                'away_team': match['away_team'],
                'date': date_str,
                'competition': match['competition'],
                'prediction': prediction['prediction'],
                'probabilities': prediction['probabilities'],
                'confidence': prediction['confidence']
            })
            
        except Exception as e:
            print(f"Erreur pour {match['home_team']} vs {match['away_team']}: {e}")
            # Continuer avec le match suivant
    
    # Résumé des prédictions
    if predictions:
        print("\n=== RÉSUMÉ DES PRÉDICTIONS ===")
        home_wins = sum(1 for p in predictions if p['prediction'] == 'Home')
        draws = sum(1 for p in predictions if p['prediction'] == 'Draw') 
        away_wins = sum(1 for p in predictions if p['prediction'] == 'Away')
        
        print(f"Home: {home_wins} | Draw: {draws} | Away: {away_wins}")
        print(f"Confiance moyenne: {sum(p['confidence'] for p in predictions)/len(predictions):.1%}")
    else:
        print("\n  Aucune prédiction n'a pu être calculée")
    
    return predictions

if __name__ == "__main__":
    # Import ici pour éviter les problèmes circulaires
    from ml_pipeline.prediction import FootballPredictor
    
    predictions = predict_upcoming_matches()