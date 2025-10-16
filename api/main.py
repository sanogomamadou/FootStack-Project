# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text  # AJOUT IMPORTANT
import uvicorn
import logging
from .routes import talent_scout 

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import des routes
from .routes import predictions, health

# Création de l'application FastAPI
app = FastAPI(
    title="FootStack API",
    description="API de prédiction de résultats de matchs de football et Detection de talents basée sur l'IA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "FootStack Team",
        "url": "https://github.com/your-username/footstack",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routes
app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(talent_scout.router)


@app.on_event("startup")
async def startup_event():
    """Événement au démarrage de l'application"""
    logger.info("  Démarrage de FootStack API...")
    logger.info("  Chargement des modèles ML...")
    
    # Test du chargement des modèles au démarrage
    try:
        from .dependencies import get_ml_model, get_db_engine
        model, encoder = get_ml_model()
        engine = get_db_engine()
        
        # Test connection DB - CORRECTION ICI
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))  # AJOUT text()
        
        logger.info("  Modèles ML et base de données chargés avec succès")
        logger.info(f"  Modèle: {type(model).__name__}")
        logger.info(f"  Classes: {list(encoder.classes_)}")
        
    except Exception as e:
        logger.error(f"  Erreur au démarrage: {e}")
        # Ne pas lever l'exception pour permettre le démarrage même si DB échoue
        logger.warning("   Démarrage en mode dégradé (sans base de données)")

@app.on_event("shutdown")
async def shutdown_event():
    """Événement à l'arrêt de l'application"""
    logger.info("  Arrêt de FootStack API...")

@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "Bienvenue sur FootStack API  ",
        "version": "1.0.0",
        "description": "API de prédiction de résultats de matchs de football basée sur l'IA",
        "model_accuracy": "49.7%",
        "endpoints": {
            "documentation": "/docs",
            "health_check": "/health",
            "automatic_prediction": "/predictions/predict-auto",
            "manual_prediction": "/predictions/predict",
            "upcoming_predictions": "/predictions/upcoming",
            "available_teams": "/predictions/teams"
        },
        "usage_example": {
            "automatic_prediction": {
                "method": "POST",
                "endpoint": "/predictions/predict-auto",
                "body": {
                    "home_team": "Paris Saint-Germain FC",
                    "away_team": "Olympique de Marseille"
                }
            }
        }
    }

@app.get("/status")
async def status():
    """Endpoint de status détaillé"""
    from .dependencies import get_ml_model, get_db_engine
    import os
    from datetime import datetime
    
    try:
        model, encoder = get_ml_model()
        engine = get_db_engine()
        
        # Test DB - CORRECTION ICI
        db_connected = False
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))  # AJOUT text()
            db_connected = True
        except:
            db_connected = False
        
        # Info modèles
        model_files = []
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
        
        return {
            "status": "healthy" if db_connected else "degraded",
            "timestamp": datetime.now().isoformat(),
            "model": {
                "loaded": True,
                "type": type(model).__name__,
                "classes": list(encoder.classes_),
                "accuracy": 0.497
            },
            "database": {
                "connected": db_connected
            },
            "files": {
                "models_available": model_files
            }
        }
        
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Pour lancer en développement
if __name__ == "__main__":
    logger.info("  Lancement de FootStack API en mode développement...")
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )