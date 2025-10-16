# api/routes/talent_scout.py (version simplifiée)
from fastapi import APIRouter, HTTPException, Query
import logging
from typing import List, Optional

from ..services.talent_scout_service import talent_scout_service
from ..schemas import (
    UndervaluedPlayerResponse,
    ClusterAnalysisResponse,
    PlayerSearchResponse
)

router = APIRouter(prefix="/talent-scout", tags=["talent-scout"])
logger = logging.getLogger(__name__)

@router.get("/undervalued", response_model=List[UndervaluedPlayerResponse])
async def get_undervalued_players(
    limit: int = Query(20, ge=1, le=50),
    min_score: float = Query(0.3, ge=0.0, le=1.0)  # Score plus bas par défaut
):
    """Obtenir les joueurs sous-évalués"""
    logger.info(f"  API: Récupération des {limit} joueurs sous-évalués (score >= {min_score})")
    
    players = talent_scout_service.get_undervalued_players(limit, min_score)
    return players

@router.get("/clusters", response_model=List[ClusterAnalysisResponse])
async def get_cluster_analysis():
    """Obtenir l'analyse de tous les clusters"""
    logger.info("  API: Analyse des clusters")
    
    clusters = talent_scout_service.get_cluster_analysis()
    return clusters

@router.get("/players/search", response_model=List[PlayerSearchResponse])
async def search_players(
    name: Optional[str] = Query(None),
    team: Optional[str] = Query(None),
    position: Optional[str] = Query(None),
    cluster: Optional[int] = Query(None, ge=0),
    limit: int = Query(20, ge=1, le=50)
):
    """Rechercher des joueurs"""
    logger.info(f"  API: Recherche joueurs - name={name}, team={team}, position={position}, cluster={cluster}")
    
    players = talent_scout_service.search_players(name, team, position, cluster, limit)
    return players

@router.get("/players/{player_name}", response_model=PlayerSearchResponse)
async def get_player_details(player_name: str):
    """Obtenir les détails d'un joueur spécifique"""
    logger.info(f"  API: Détails du joueur {player_name}")
    
    player = talent_scout_service.get_player_details(player_name)
    if player is None:
        raise HTTPException(status_code=404, detail="Joueur non trouvé")
    
    return player