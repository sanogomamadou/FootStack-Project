from fastapi import APIRouter, Depends
from sqlalchemy import text  # AJOUT IMPORTANT
from datetime import datetime
import os
import joblib

from ..dependencies import get_db_engine, get_ml_model
from ..schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Health check de l'API"""
    
    # Vérifier modèle
    model_loaded = False
    try:
        model, encoder = get_ml_model()
        model_loaded = True
    except:
        model_loaded = False
    
    # Vérifier base de données
    database_connected = False
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))  # CORRECTION ICI
        database_connected = True
    except:
        database_connected = False
    
    # Déterminer le status global
    status = "healthy" if model_loaded and database_connected else "degraded"
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        database_connected=database_connected,
        timestamp=datetime.now()
    )

@router.get("/detailed")
async def detailed_health_check():
    """Health check détaillé avec plus d'informations"""
    
    health_info = {
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Vérification modèle ML
    try:
        model, encoder = get_ml_model()
        health_info["components"]["ml_model"] = {
            "status": "healthy",
            "model_type": type(model).__name__,
            "classes": list(encoder.classes_),
            "n_classes": len(encoder.classes_)
        }
    except Exception as e:
        health_info["components"]["ml_model"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Vérification base de données
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            # Test simple
            conn.execute(text("SELECT 1"))  # CORRECTION ICI
            
            # Compter les matchs
            result = conn.execute(text("SELECT COUNT(*) FROM matches_cleaned"))  # CORRECTION ICI
            match_count = result.scalar()
            
        health_info["components"]["database"] = {
            "status": "healthy",
            "match_count": match_count
        }
    except Exception as e:
        health_info["components"]["database"] = {
            "status": "unhealthy", 
            "error": str(e)
        }
    
    # Vérification fichiers modèles
    models_dir = "models"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
        health_info["components"]["model_files"] = {
            "status": "healthy",
            "files": model_files,
            "count": len(model_files)
        }
    else:
        health_info["components"]["model_files"] = {
            "status": "unhealthy",
            "error": "Dossier models non trouvé"
        }
    
    # Status global
    all_healthy = all(comp["status"] == "healthy" for comp in health_info["components"].values())
    health_info["overall_status"] = "healthy" if all_healthy else "degraded"
    
    return health_info